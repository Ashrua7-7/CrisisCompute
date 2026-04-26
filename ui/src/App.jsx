import { useState, useEffect, useRef, useCallback, Component } from 'react'
import {
  healthCheck, resetEpisode, stepEnvironment,
  buildHybridActions, setBackend, BACKENDS,
} from './api.js'
import { normalizeObservation, poolResources } from './normalize.js'
import ResultsPanel from './ResultsPanel.jsx'

// ─── Error Boundary — prevents blank screen on render crash ──────────────────
class ErrorBoundary extends Component {
  constructor(props) { super(props); this.state = { error: null } }
  static getDerivedStateFromError(e) { return { error: e } }
  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 32, color: '#ef4444', fontFamily: 'JetBrains Mono, monospace', fontSize: 13 }}>
          <div style={{ marginBottom: 8, fontSize: 16 }}>⚠ Render Error</div>
          <div style={{ color: '#94a3b8', marginBottom: 12 }}>{this.state.error.message}</div>
          <button onClick={() => this.setState({ error: null })} style={{ background: '#1e293b', border: '1px solid #334155', color: '#e2e8f0', padding: '6px 14px', borderRadius: 6, cursor: 'pointer', fontFamily: 'inherit' }}>
            Dismiss & Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

// ─── Constants ────────────────────────────────────────────────────────────────

const AGENTS = {
  data_loader: {
    label: 'Data Loader',
    emoji: '📂',
    color: '#3b82f6',
    bg: 'rgba(59,130,246,0.08)',
    role: 'Loads CSV files • LOW priority • No deadline',
  },
  data_cleaner: {
    label: 'Data Cleaner',
    emoji: '🧹',
    color: '#10b981',
    bg: 'rgba(16,185,129,0.08)',
    role: 'Cleans data • MEDIUM priority • 3h deadline',
  },
  ml_trainer: {
    label: 'ML Trainer',
    emoji: '🧠',
    color: '#a855f7',
    bg: 'rgba(168,85,247,0.08)',
    role: 'Trains models • HIGH priority • 4h deadline',
  },
}

// Clickable quick-test commands shown above the input box
const QUICK_CMDS = [
  { label: '↺ Reset episode',       cmd: 'reset',   color: '#3b82f6' },
  { label: '▶ One step',            cmd: 'step',    color: '#10b981' },
  { label: '⚡ Run 5 steps',        cmd: 'run 5',   color: '#a855f7' },
  { label: '🚀 Run 20 steps',       cmd: 'run 20',  color: '#f59e0b' },
  { label: '♾ Auto to end',        cmd: 'auto',    color: '#06b6d4' },
  { label: '📊 Status',             cmd: 'status',  color: '#64748b' },
]

const HELP_TEXT = `HYBRID MODE — Agents combine RL Q-values + LLM reasoning

Commands:
  reset          Start a fresh episode
  step           Take one negotiation step (all 3 agents act)
  run [N]        Run N steps automatically (default: 10)
  auto           Run until the episode ends
  stop           Pause auto-run mid-flight
  status         Show current state, hour, rewards
  help           Show this message
  clear          Clear chat history`

// ─── Helpers ──────────────────────────────────────────────────────────────────

function ts() {
  return new Date().toLocaleTimeString('en-US', { hour12: false })
}

function mkMsg(type, content, extra = {}) {
  return { id: Date.now() + Math.random(), type, content, time: ts(), ...extra }
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function ResourceBar({ label, used, total, color }) {
  const pct = total > 0 ? Math.min(100, Math.round((used / total) * 100)) : 0
  const barColor = pct > 80 ? '#ef4444' : pct > 50 ? '#f59e0b' : color
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>
        <span>{label}</span>
        <span style={{ fontFamily: 'JetBrains Mono, monospace' }}>{used}/{total} ({pct}%)</span>
      </div>
      <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: barColor, borderRadius: 3, transition: 'width 0.5s ease, background 0.3s ease' }} />
      </div>
    </div>
  )
}

function AgentCard({ agentId, obs, totalReward }) {
  const cfg = AGENTS[agentId]
  const tasks   = obs?.my_tasks ?? []
  const pending = tasks.filter((t) => t.status === 'pending').length
  const running = tasks.filter((t) => t.status === 'running').length
  const done    = tasks.filter((t) => t.status === 'done').length
  const statusColor = running > 0 ? '#10b981' : pending > 0 ? '#f59e0b' : '#475569'
  const statusLabel = running > 0 ? 'WORKING' : pending > 0 ? 'QUEUED' : 'IDLE'

  return (
    <div style={{ background: cfg.bg, border: `1px solid ${cfg.color}33`, borderLeft: `3px solid ${cfg.color}`, borderRadius: 10, padding: '12px 14px', marginBottom: 10, transition: 'all 0.3s ease' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
        <span style={{ fontSize: 18 }}>{cfg.emoji}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: 13, fontWeight: 600, color: cfg.color }}>{cfg.label}</div>
          <div style={{ fontSize: 10, color: '#64748b', marginTop: 1 }}>{cfg.role}</div>
        </div>
        <div style={{ fontSize: 9, fontWeight: 700, color: statusColor, background: `${statusColor}18`, padding: '2px 6px', borderRadius: 4, letterSpacing: 0.5 }}>
          {statusLabel}
        </div>
      </div>
      <div style={{ display: 'flex', gap: 6, fontSize: 10, color: '#94a3b8' }}>
        <span style={{ color: '#f59e0b' }}>⏳ {pending}</span>
        <span>•</span>
        <span style={{ color: '#10b981' }}>▶ {running}</span>
        <span>•</span>
        <span style={{ color: '#475569' }}>✓ {done}</span>
      </div>
      {totalReward !== undefined && (
        <div style={{ marginTop: 6, fontSize: 11, color: '#94a3b8' }}>
          Reward: <span style={{ color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace', fontWeight: 600 }}>{totalReward.toFixed(1)}</span>
        </div>
      )}
    </div>
  )
}

function ChatMessage({ msg }) {
  if (msg.type === 'system') {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', margin: '10px 0' }}>
        <div style={{ background: '#1e293b', border: '1px solid #334155', borderRadius: 20, padding: '4px 14px', fontSize: 11, color: '#64748b' }}>
          {msg.content}
        </div>
      </div>
    )
  }

  if (msg.type === 'user') {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
        <div style={{ maxWidth: '65%' }}>
          <div style={{ background: 'linear-gradient(135deg,#3b82f6,#6366f1)', borderRadius: '18px 18px 4px 18px', padding: '10px 14px', fontSize: 13, color: '#fff', lineHeight: 1.5 }}>
            {msg.content}
          </div>
          <div style={{ fontSize: 10, color: '#475569', marginTop: 3, textAlign: 'right' }}>{msg.time}</div>
        </div>
      </div>
    )
  }

  if (msg.type === 'agent') {
    const cfg = AGENTS[msg.agentId] ?? { color: '#94a3b8', bg: '#1e293b', emoji: '🤖', label: msg.agentId }
    const strategyColor = { aggressive: '#ef4444', standard: '#f59e0b', conservative: '#3b82f6' }[msg.strategy] ?? '#64748b'
    return (
      <div style={{ display: 'flex', gap: 10, marginBottom: 12, alignItems: 'flex-start' }}>
        <div style={{ width: 34, height: 34, borderRadius: '50%', background: cfg.bg, border: `2px solid ${cfg.color}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 16, flexShrink: 0 }}>
          {cfg.emoji}
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 12, fontWeight: 600, color: cfg.color }}>{cfg.label}</span>
            <span style={{ fontSize: 10, color: '#475569' }}>{msg.time}</span>
            {/* Mode badge — always HYBRID */}
            <span style={{ fontSize: 9, padding: '1px 6px', borderRadius: 4, background: '#1e1033', color: '#c084fc', border: '1px solid #7c3aed44', fontWeight: 700, letterSpacing: 0.5 }}>
              HYBRID
            </span>
            {msg.action && (
              <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: msg.action === 'wait' ? '#1e293b' : `${cfg.color}22`, color: msg.action === 'wait' ? '#475569' : cfg.color, border: `1px solid ${msg.action === 'wait' ? '#334155' : cfg.color + '44'}`, fontFamily: 'JetBrains Mono, monospace' }}>
                {msg.action}
              </span>
            )}
            {msg.strategy && (
              <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: `${strategyColor}15`, color: strategyColor, border: `1px solid ${strategyColor}33` }}>
                {msg.strategy}
              </span>
            )}
            {msg.reward !== undefined && (
              <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: '#fbbf2412', color: '#fbbf24', border: '1px solid #fbbf2430', fontFamily: 'JetBrains Mono, monospace' }}>
                +{typeof msg.reward === 'number' ? msg.reward.toFixed(1) : msg.reward} pts
              </span>
            )}
          </div>
          <div style={{ background: '#0f172a', border: `1px solid ${cfg.color}22`, borderRadius: '4px 18px 18px 18px', padding: '10px 14px', fontSize: 13, color: '#cbd5e1', lineHeight: 1.6 }}>
            {msg.content}
          </div>
          {msg.taskId && (
            <div style={{ marginTop: 4, fontSize: 10, color: '#475569' }}>
              Task: <code style={{ fontFamily: 'JetBrains Mono, monospace', color: cfg.color }}>{msg.taskId}</code>
            </div>
          )}
        </div>
      </div>
    )
  }

  if (msg.type === 'step_result') {
    return (
      <div style={{ background: '#0f172a', border: '1px solid #1e293b', borderRadius: 10, padding: '10px 14px', marginBottom: 12, fontSize: 12 }}>
        <div style={{ color: '#334155', marginBottom: 6, fontSize: 10, fontFamily: 'JetBrains Mono, monospace' }}>
          ─── Step {msg.step} ───────────────────────────────────────
        </div>
        <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap' }}>
          <span style={{ color: '#94a3b8' }}>Team reward: <span style={{ color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace' }}>{typeof msg.teamReward === 'number' ? msg.teamReward.toFixed(1) : '—'}</span></span>
          <span style={{ color: '#94a3b8' }}>Hour: <span style={{ color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>{msg.hour ?? '—'}/{msg.maxHours ?? 8}</span></span>
          {msg.done && <span style={{ color: '#ef4444', fontWeight: 600 }}>● Episode ended</span>}
        </div>
        {msg.events && msg.events.length > 0 && (
          <div style={{ marginTop: 6, display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {msg.events.map((ev, i) => {
              const color = ev.startsWith('task_done') ? '#10b981'
                : ev.startsWith('allocated') ? '#3b82f6'
                : ev.startsWith('invalid') ? '#ef4444'
                : ev.includes('crisis') || ev.includes('outage') ? '#f59e0b'
                : '#64748b'
              return (
                <span key={i} style={{ fontSize: 10, padding: '2px 7px', borderRadius: 10, background: `${color}18`, color, border: `1px solid ${color}33`, fontFamily: 'JetBrains Mono, monospace' }}>
                  {ev}
                </span>
              )
            })}
          </div>
        )}
      </div>
    )
  }

  if (msg.type === 'info') {
    return (
      <div style={{ background: '#0f172a', border: '1px solid #334155', borderRadius: 10, padding: '12px 14px', marginBottom: 12, fontSize: 12, color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace', whiteSpace: 'pre-wrap', lineHeight: 1.8 }}>
        {msg.content}
      </div>
    )
  }

  if (msg.type === 'error') {
    return (
      <div style={{ background: '#1e0a0a', border: '1px solid #ef444433', borderRadius: 10, padding: '10px 14px', marginBottom: 12, fontSize: 12, color: '#ef4444' }}>
        ⚠ {msg.content}
      </div>
    )
  }

  return null
}

// ─── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [tab, setTab]               = useState('chat')
  const [messages, setMessages]     = useState([])
  const [input, setInput]           = useState('')
  const [isRunning, setIsRunning]   = useState(false)
  const [backendOnline, setBackendOnline] = useState(null)
  const [backendMode, setBackendMode]     = useState('hf')
  const [observation, setObservation]     = useState(null)
  const [stepCount, setStepCount]         = useState(0)
  const [agentRewards, setAgentRewards]   = useState({})
  const [episodeStarted, setEpisodeStarted] = useState(false)

  const chatEndRef  = useRef(null)
  const autoRunRef  = useRef(false)
  const inputRef    = useRef(null)

  const push = useCallback((...msgs) => setMessages((prev) => [...prev, ...msgs]), [])

  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  // ── Backend ping ─────────────────────────────────────────────────────────

  const switchBackend = useCallback((mode) => {
    setBackendMode(mode)
    setBackend(mode === 'hf' ? BACKENDS.hf : BACKENDS.local)
    setBackendOnline(null)
    setEpisodeStarted(false)
    setObservation(null)
    setStepCount(0)
    setAgentRewards({})
    setMessages([mkMsg('system', `Switching to ${mode === 'hf' ? '☁ HuggingFace Space' : '💻 Local'} backend…`)])
  }, [])

  useEffect(() => {
    setBackendOnline(null)
    const label = backendMode === 'hf' ? 'HuggingFace Space' : 'Local (localhost:7860)'
    push(mkMsg('system', `Connecting to ${label}…`))
    healthCheck()
      .then(() => {
        setBackendOnline(true)
        push(
          mkMsg('system', '✓ Connected — SatyaEnv ready  |  Mode: HYBRID'),
          mkMsg('info', HELP_TEXT),
        )
      })
      .catch(() => {
        setBackendOnline(false)
        if (backendMode === 'local') {
          push(mkMsg('error', 'Local backend offline.\n  .venv\\Scripts\\Activate.ps1\n  uvicorn server.app:app --reload --port 7860'))
        } else {
          push(mkMsg('error', 'HuggingFace Space unreachable — check https://huggingface.co/spaces/Gautam0898/crisiscompute'))
        }
      })
  }, [backendMode])

  // ── Core actions ──────────────────────────────────────────────────────────

  const doReset = useCallback(async () => {
    push(mkMsg('user', 'reset'))
    try {
      const data = await resetEpisode()
      const obs  = normalizeObservation(data.observation)
      setObservation(obs)
      setStepCount(0)
      setAgentRewards({})
      setEpisodeStarted(true)
      push(
        mkMsg('system', `Episode ${obs.episode_index ?? '?'} started — Hour 0/${obs.max_hours ?? 8}`),
        ...Object.keys(AGENTS).map((agentId) =>
          mkMsg('agent', 'New episode! Standing by for resource allocation. RL Q-table loaded.', { agentId, action: 'idle' })
        ),
      )
    } catch (e) {
      push(mkMsg('error', `Reset failed: ${e.message}`))
    }
  }, [push])

  const doStep = useCallback(async (silent = false) => {
    if (!episodeStarted) {
      push(mkMsg('error', 'No active episode. Type "reset" to start one.'))
      return false
    }
    try {
      const actions = buildHybridActions(observation)
      const data    = await stepEnvironment(actions)
      const obs     = normalizeObservation(data.observation)
      setObservation(obs)
      setStepCount((s) => s + 1)

      setAgentRewards((prev) => {
        const next = { ...prev }
        for (const [k, v] of Object.entries(data.step_rewards ?? {})) {
          next[k] = (prev[k] ?? 0) + v
        }
        return next
      })

      if (!silent) {
        // Parse events for richer agent messages
        const events      = obs.recent_events ?? []
        const doneEvents  = events.filter((e) => e.startsWith('task_done:'))
        const allocEvents = events.filter((e) => e.startsWith('allocated '))
        const cleanEvents = events.filter((e) =>
          !['episode_reset', 'contracts_kept:0', 'contracts_broken:0'].includes(e) &&
          !e.startsWith('contracts_')
        )

        const agentMsgs = Object.entries(actions).map(([agentId, act]) => {
          const reward    = data.step_rewards?.[agentId]
          const doneTasks = doneEvents
            .filter((e) => e.includes(act.task_id ?? ''))
            .map((e) => e.replace('task_done:', ''))

          let msg = act.reasoning ?? ''
          if (doneTasks.length > 0) {
            msg = `✅ Task completed: ${doneTasks.join(', ')}. ${act.reasoning ?? ''}`
          } else if (act.action === 'wait' || !act.task_id) {
            const agentWaitEvent = events.find((e) => e.includes(agentId.replace('_', ' ') + ' waits'))
            msg = agentWaitEvent
              ? `Waiting — dependencies not met or resources unavailable.`
              : (act.reasoning ?? 'Standing by.')
          }

          return mkMsg('agent', msg, {
            agentId,
            action:   act.action,
            taskId:   act.task_id,
            strategy: act._strategy,
            reward,
          })
        })

        push(
          ...agentMsgs,
          mkMsg('step_result', '', {
            step:       stepCount + 1,
            teamReward: data.team_reward,
            hour:       obs.current_hour,
            maxHours:   obs.max_hours,
            done:       data.done,
            events:     cleanEvents,
          }),
        )
      }

      if (data.done) {
        push(mkMsg('system', '🏁 Episode complete! Type "reset" to start a new one.'))
        return false
      }
      return true
    } catch (e) {
      push(mkMsg('error', `Step failed: ${e.message}`))
      return false
    }
  }, [episodeStarted, observation, push, stepCount])

  const doAutoRun = useCallback(async (maxSteps = 999) => {
    if (isRunning) return
    autoRunRef.current = true
    setIsRunning(true)
    push(mkMsg('system', `Auto-running (HYBRID mode)${maxSteps < 999 ? ` — ${maxSteps} steps` : ' until episode ends'}…`))

    let cont = true, steps = 0
    while (cont && autoRunRef.current && steps < maxSteps) {
      cont = await doStep(false)
      steps++
      await new Promise((r) => setTimeout(r, 550))
    }

    setIsRunning(false)
    autoRunRef.current = false
    push(mkMsg('system', `Done — ${steps} steps completed.`))
  }, [isRunning, doStep, push])

  const doStatus = useCallback(() => {
    if (!observation) { push(mkMsg('info', 'No active episode. Type "reset" to start.')); return }
    const obs = observation
    const lines = [
      `Episode: ${obs.episode_index ?? '?'}    Hour: ${obs.current_hour}/${obs.max_hours}    Steps: ${stepCount}`,
      `Mode:    HYBRID (RL Q-learning + LLM reasoning)`,
      '',
      'Resources available:',
      `  CPU: ${obs.available_cpu ?? '?'} cores   GPU: ${obs.available_gpu ?? '?'}   Memory: ${obs.available_memory ?? '?'} GB`,
      '',
      'Rewards this episode:',
      ...Object.entries(agentRewards).map(([k, v]) => `  ${k.padEnd(14)} ${v.toFixed(1)} pts`),
    ]
    push(mkMsg('info', lines.join('\n')))
  }, [observation, stepCount, agentRewards, push])

  // ── Command parser ────────────────────────────────────────────────────────

  const handleCommand = useCallback(async (raw) => {
    const cmd = raw.trim().toLowerCase()
    if (!cmd) return

    if (cmd === 'reset')                      { await doReset() }
    else if (cmd === 'step')                  { push(mkMsg('user', 'step')); await doStep(false) }
    else if (cmd.startsWith('run'))           { const n = parseInt(cmd.split(/\s+/)[1]) || 10; push(mkMsg('user', cmd)); await doAutoRun(n) }
    else if (cmd === 'auto')                  { push(mkMsg('user', 'auto')); await doAutoRun() }
    else if (cmd === 'stop' || cmd === 'pause') { autoRunRef.current = false; setIsRunning(false); push(mkMsg('system', 'Auto-run stopped.')) }
    else if (cmd === 'status')                { push(mkMsg('user', 'status')); doStatus() }
    else if (cmd === 'help')                  { push(mkMsg('user', 'help'), mkMsg('info', HELP_TEXT)) }
    else if (cmd === 'clear')                 { setMessages([mkMsg('system', 'Chat cleared.')]) }
    else { push(mkMsg('user', raw), mkMsg('error', `Unknown command: "${cmd}". Type "help" to see commands.`)) }
  }, [doReset, doStep, doAutoRun, doStatus, push])

  const handleSubmit = (e) => {
    e.preventDefault()
    const val = input.trim()
    if (!val) return
    if (val.toLowerCase() === 'stop') { autoRunRef.current = false; setIsRunning(false); push(mkMsg('system', 'Auto-run stopped.')); setInput(''); return }
    if (isRunning) return
    handleCommand(val)
    setInput('')
  }

  // ── Derived display ───────────────────────────────────────────────────────

  const agentObs  = observation?.agent_observations ?? {}
  const { cpuTotal, gpuTotal, memTotal, cpuUsed, gpuUsed, memUsed } = poolResources(observation)
  const hour      = observation?.current_hour ?? 0
  const maxHours  = observation?.max_hours    ?? 8
  const hourPct   = maxHours > 0 ? (hour / maxHours) * 100 : 0

  // ── Render ────────────────────────────────────────────────────────────────

  return (
  <ErrorBoundary>
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', background: '#030712', color: '#e2e8f0', fontFamily: 'Inter, system-ui, sans-serif', overflow: 'hidden' }}>

      {/* ── Top bar ── */}
      <div style={{ height: 52, background: '#0f172a', borderBottom: '1px solid #1e293b', display: 'flex', alignItems: 'center', padding: '0 18px', gap: 10, flexShrink: 0 }}>
        <span style={{ fontSize: 20 }}>🚨</span>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color: '#f1f5f9', letterSpacing: 0.3 }}>CrisisCompute</div>
          <div style={{ fontSize: 10, color: '#475569' }}>Multi-Agent Negotiation</div>
        </div>

        {/* HYBRID badge */}
        <div style={{ marginLeft: 4, fontSize: 10, fontWeight: 700, padding: '2px 8px', borderRadius: 5, background: '#1e1033', color: '#c084fc', border: '1px solid #7c3aed55', letterSpacing: 0.8 }}>
          HYBRID MODE
        </div>

        {/* Tabs */}
        <div style={{ display: 'flex', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, overflow: 'hidden', marginLeft: 10, fontSize: 12, fontWeight: 600 }}>
          {[{ key: 'chat', label: '💬 Chat' }, { key: 'results', label: '📊 Results' }].map(({ key, label }) => (
            <button key={key} onClick={() => setTab(key)} style={{ padding: '5px 14px', border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif', fontWeight: 600, fontSize: 12, background: tab === key ? '#1d4ed8' : 'transparent', color: tab === key ? '#bfdbfe' : '#475569', transition: 'all 0.15s' }}>
              {label}
            </button>
          ))}
        </div>

        <div style={{ flex: 1 }} />

        {/* Backend selector */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <div style={{ display: 'flex', background: '#0f172a', border: '1px solid #1e293b', borderRadius: 8, overflow: 'hidden', fontSize: 11 }}>
            {[{ key: 'hf', label: '☁ HF Space' }, { key: 'local', label: '💻 Local' }].map(({ key, label }) => (
              <button key={key} onClick={() => switchBackend(key)} style={{ padding: '4px 10px', border: 'none', cursor: 'pointer', fontFamily: 'Inter, sans-serif', fontWeight: 600, fontSize: 11, background: backendMode === key ? '#1e40af' : 'transparent', color: backendMode === key ? '#93c5fd' : '#475569', transition: 'all 0.2s' }}>
                {label}
              </button>
            ))}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11 }}>
            <div style={{ width: 7, height: 7, borderRadius: '50%', background: backendOnline === null ? '#f59e0b' : backendOnline ? '#10b981' : '#ef4444', boxShadow: backendOnline ? '0 0 5px #10b981' : backendOnline === false ? '0 0 5px #ef4444' : '0 0 5px #f59e0b' }} />
            <span style={{ color: '#64748b' }}>{backendOnline === null ? 'Connecting…' : backendOnline ? 'Online' : 'Offline'}</span>
          </div>
        </div>

        {/* Quick action buttons */}
        <button onClick={doReset} disabled={isRunning} style={btn('#3b82f6')}>↺ Reset</button>
        <button onClick={() => { push(mkMsg('user', 'step')); doStep(false) }} disabled={isRunning || !episodeStarted} style={btn('#10b981')}>▶ Step</button>
        <button
          onClick={() => { if (isRunning) { autoRunRef.current = false; setIsRunning(false) } else { push(mkMsg('user', 'auto')); doAutoRun() } }}
          disabled={!episodeStarted}
          style={btn(isRunning ? '#ef4444' : '#a855f7')}
        >
          {isRunning ? '⏹ Stop' : '⚡ Auto'}
        </button>
      </div>

      {/* ── Body ── */}
      {tab === 'results' ? (
        <ResultsPanel />
      ) : (
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>

          {/* Left — agent status */}
          <div style={{ width: 230, background: '#0a0f1a', borderRight: '1px solid #1e293b', padding: 14, overflowY: 'auto', flexShrink: 0 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1.5, marginBottom: 12, textTransform: 'uppercase' }}>Agents</div>
            {Object.keys(AGENTS).map((id) => (
              <AgentCard key={id} agentId={id} obs={agentObs[id]} totalReward={agentRewards[id]} />
            ))}

            {episodeStarted && (
              <div style={{ marginTop: 16 }}>
                <div style={{ fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1.5, marginBottom: 8, textTransform: 'uppercase' }}>Timeline</div>
                <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>Hour {hour} / {maxHours}</div>
                <div style={{ height: 6, background: '#1e293b', borderRadius: 3, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${hourPct}%`, background: hourPct > 80 ? '#ef4444' : hourPct > 60 ? '#f59e0b' : '#3b82f6', borderRadius: 3, transition: 'width 0.5s ease' }} />
                </div>
                <div style={{ fontSize: 10, color: '#475569', marginTop: 4 }}>Steps: {stepCount}</div>
              </div>
            )}
          </div>

          {/* Center — chat */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
            <div style={{ flex: 1, overflowY: 'auto', padding: '18px 22px', display: 'flex', flexDirection: 'column' }}>
              {messages.map((msg) => <ChatMessage key={msg.id} msg={msg} />)}
              <div ref={chatEndRef} />
            </div>

            {/* Quick-command chips */}
            <div style={{ padding: '6px 20px 0', display: 'flex', gap: 6, flexWrap: 'wrap', borderTop: '1px solid #0f172a' }}>
              {QUICK_CMDS.map(({ label, cmd, color }) => (
                <button
                  key={cmd}
                  onClick={() => { if (!isRunning || cmd === 'stop') handleCommand(isRunning ? 'stop' : cmd) }}
                  style={{ fontSize: 11, padding: '4px 10px', borderRadius: 16, border: `1px solid ${color}33`, background: `${color}10`, color, cursor: 'pointer', fontFamily: 'Inter, sans-serif', transition: 'all 0.15s', whiteSpace: 'nowrap' }}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Input */}
            <form onSubmit={handleSubmit} style={{ borderTop: '1px solid #1e293b', background: '#0a0f1a', padding: '10px 18px', display: 'flex', gap: 10, alignItems: 'center' }}>
              <div style={{ flex: 1, background: '#0f172a', border: '1px solid #334155', borderRadius: 12, display: 'flex', alignItems: 'center', padding: '0 14px' }}>
                <span style={{ color: '#475569', fontSize: 13, marginRight: 8, fontFamily: 'JetBrains Mono, monospace' }}>&gt;</span>
                <input
                  ref={inputRef}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={isRunning ? 'Running… type "stop" to pause' : 'reset  •  step  •  run 5  •  auto  •  status  •  help'}
                  style={{ flex: 1, background: 'transparent', border: 'none', outline: 'none', color: '#e2e8f0', fontSize: 13, fontFamily: 'Inter, sans-serif', padding: '10px 0' }}
                />
              </div>
              <button type="submit" style={{ ...btn('#3b82f6'), padding: '10px 18px', fontSize: 13 }}>Send</button>
            </form>
          </div>

          {/* Right — resources + tasks */}
          <div style={{ width: 250, background: '#0a0f1a', borderLeft: '1px solid #1e293b', padding: 14, overflowY: 'auto', flexShrink: 0 }}>
            <div style={{ fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1.5, marginBottom: 12, textTransform: 'uppercase' }}>Resource Pool</div>
            <ResourceBar label="CPU Cores" used={cpuUsed} total={cpuTotal} color="#3b82f6" />
            <ResourceBar label="GPU Units"  used={gpuUsed} total={gpuTotal} color="#a855f7" />
            <ResourceBar label="Memory GB"  used={memUsed} total={memTotal} color="#10b981" />

            <div style={{ fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1.5, margin: '18px 0 10px', textTransform: 'uppercase' }}>Tasks</div>
            {Object.keys(AGENTS).map((agentId) => {
              const cfg   = AGENTS[agentId]
              const tasks = agentObs[agentId]?.my_tasks ?? []
              if (!tasks.length) return <div key={agentId} style={{ fontSize: 11, color: '#334155', marginBottom: 6 }}>{cfg.emoji} {cfg.label}: —</div>
              return (
                <div key={agentId} style={{ marginBottom: 12 }}>
                  <div style={{ fontSize: 11, color: cfg.color, fontWeight: 600, marginBottom: 4 }}>{cfg.emoji} {cfg.label}</div>
                  {tasks.slice(0, 6).map((task) => (
                    <div key={task.task_id ?? task.id} style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 3, fontSize: 10 }}>
                      <span style={{ width: 6, height: 6, borderRadius: '50%', background: task.status === 'done' ? '#10b981' : task.status === 'running' ? '#f59e0b' : '#334155', flexShrink: 0 }} />
                      <span style={{ fontFamily: 'JetBrains Mono, monospace', color: task.status === 'done' ? '#475569' : task.status === 'running' ? '#f59e0b' : '#94a3b8', textDecoration: task.status === 'done' ? 'line-through' : 'none' }}>
                        {task.task_id ?? task.id}
                      </span>
                    </div>
                  ))}
                  {tasks.length > 6 && <div style={{ fontSize: 10, color: '#334155' }}>+{tasks.length - 6} more</div>}
                </div>
              )
            })}

            {observation?.metrics && Object.keys(observation.metrics).length > 0 && (
              <>
                <div style={{ fontSize: 10, fontWeight: 700, color: '#475569', letterSpacing: 1.5, margin: '18px 0 10px', textTransform: 'uppercase' }}>Metrics</div>
                {Object.entries(observation.metrics).map(([k, v]) => (
                  <div key={k} style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, marginBottom: 4 }}>
                    <span style={{ color: '#64748b' }}>{k.replace(/_/g, ' ')}</span>
                    <span style={{ color: '#94a3b8', fontFamily: 'JetBrains Mono, monospace' }}>{typeof v === 'number' ? v.toFixed(2) : String(v)}</span>
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      )}
    </div>
  </ErrorBoundary>
  )
}

function btn(color) {
  return { background: `${color}18`, border: `1px solid ${color}55`, color, borderRadius: 8, padding: '6px 12px', fontSize: 12, fontWeight: 600, cursor: 'pointer', fontFamily: 'Inter, sans-serif', transition: 'background 0.2s', whiteSpace: 'nowrap' }
}
