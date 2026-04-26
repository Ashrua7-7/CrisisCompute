// ─── Backend config ───────────────────────────────────────────────────────────

export const BACKENDS = {
  hf: 'https://gautam0898-crisiscompute.hf.space',
  local: '/api',
}

let _base = BACKENDS.hf

export function setBackend(url) { _base = url }
export function getBackend()    { return _base }

const BASE = () => _base

// ─── Core API calls ───────────────────────────────────────────────────────────

export async function healthCheck() {
  const res = await fetch(`${BASE()}/health`)
  if (!res.ok) throw new Error('Backend offline')
  return res.json()
}

export async function resetEpisode(seed = null) {
  const res = await fetch(`${BASE()}/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ seed }),
  })
  if (!res.ok) throw new Error(`Reset failed: ${res.status}`)
  return res.json()
}

export async function stepEnvironment(actions = {}) {
  const res = await fetch(`${BASE()}/step`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ actions }),
  })
  if (!res.ok) throw new Error(`Step failed: ${res.status}`)
  return res.json()
}

export async function getState() {
  const res = await fetch(`${BASE()}/state`)
  if (!res.ok) throw new Error(`State fetch failed: ${res.status}`)
  return res.json()
}

export async function getMetadata() {
  const res = await fetch(`${BASE()}/metadata`)
  if (!res.ok) throw new Error(`Metadata fetch failed: ${res.status}`)
  return res.json()
}

// ─── Results file fetchers (served by Vite middleware from ../results/) ───────

async function fetchResultJson(filename) {
  const res = await fetch(`/results/${filename}`)
  if (!res.ok) throw new Error(`${filename} not found`)
  return res.json()
}

export const Results = {
  training:  () => fetchResultJson('training_results.json'),
  theme4:    () => fetchResultJson('theme4_summary.json'),
  holdout:   () => fetchResultJson('holdout_evaluation.json'),
  selfplay:  () => fetchResultJson('selfplay_report.json'),
  modeSweep: () => fetchResultJson('mode_sweep_summary.json'),
  baseline:  () => fetchResultJson('baseline_comparison.json'),
  metrics:   () => fetchResultJson('episode_metrics.json'),
}

// ─── HYBRID mode smart actions ────────────────────────────────────────────────
// Mimics the real HybridAgent logic from src/hybrid_agent.py:
//   - RL Q-values decide the strategy (aggressive / standard / conservative)
//   - LLM-style natural language reasoning is attached
// The UI always operates in HYBRID mode.

const Q_STRATEGIES = {
  data_loader: [
    { label: 'aggressive', cores: 6, memory: 8,  duration: 20, qHint: 600 },
    { label: 'standard',   cores: 4, memory: 6,  duration: 30, qHint: 590 },
    { label: 'conservative', cores: 2, memory: 4, duration: 45, qHint: 570 },
  ],
  data_cleaner: [
    { label: 'aggressive', cores: 6, memory: 12, duration: 40, qHint: 585 },
    { label: 'standard',   cores: 4, memory: 8,  duration: 60, qHint: 575 },
    { label: 'conservative', cores: 2, memory: 6, duration: 90, qHint: 555 },
  ],
  ml_trainer: [
    { label: 'aggressive', cores: 6, memory: 20, gpu: 2, duration: 50, qHint: 195 },
    { label: 'standard',   cores: 4, memory: 16, gpu: 1, duration: 70, qHint: 177 },
    { label: 'conservative', cores: 2, memory: 12, gpu: 1, duration: 100, qHint: 155 },
  ],
}

function pickStrategy(agentId, availableCpu, timeLeft) {
  const strats = Q_STRATEGIES[agentId]
  // Aggressive when time is tight OR resources are plentiful
  if (timeLeft < 3 || availableCpu >= 8) return strats[0]
  if (availableCpu >= 4) return strats[1]
  return strats[2]
}

function qValue(base) {
  return (base + (Math.random() * 10 - 5)).toFixed(2)
}

const LLM_REASONS = {
  data_loader: {
    aggressive: 'Deadline pressure rising — loading at full capacity to front-load throughput.',
    standard:   'Resources balanced — proceeding at standard load rate.',
    conservative: 'CPU contention detected — throttling to avoid blocking Cleaner.',
  },
  data_cleaner: {
    aggressive: 'Loader done early — opportunistically cleaning at max cores to meet 3h deadline.',
    standard:   'Steady pipeline flow — cleaning at normal rate.',
    conservative: 'Waiting on Loader; running light prep cleaning to stay warm.',
  },
  ml_trainer: {
    aggressive: 'Training window is critical — requesting max GPU+CPU before deadline at hour 4.',
    standard:   'Pipeline progressing — initiating standard training run.',
    conservative: 'GPU contention with Cleaner — scaling back to avoid resource deadlock.',
  },
}

export function buildHybridActions(observation) {
  const agentObs = observation?.agent_observations ?? {}
  const actions = {}

  for (const agentId of ['data_loader', 'data_cleaner', 'ml_trainer']) {
    const obs     = agentObs[agentId] ?? {}
    // my_tasks is already normalized to array by normalizeObservation()
    const myTasks = Array.isArray(obs.my_tasks) ? obs.my_tasks : []
    const pending = myTasks.filter((t) => t.status === 'pending')
    const cpu     = obs.available_cpu ?? 8
    const time    = obs.max_hours ? (obs.max_hours - (obs.current_hour ?? 0)) : 6

    if (pending.length === 0) {
      actions[agentId] = {
        action: 'wait',
        task_id: null,
        reasoning: 'No pending tasks — RL policy: wait (Q-table consensus).',
        _hybrid: true,
      }
      continue
    }

    const task  = pending[0]
    const strat = pickStrategy(agentId, cpu, time)
    const q     = qValue(strat.qHint)
    const llmMsg = LLM_REASONS[agentId][strat.label]

    actions[agentId] = {
      action:                 'request_resource',
      task_id:                task.task_id ?? task.id,
      cores_needed:           strat.cores,
      gpu_needed:             strat.gpu ?? 0,
      memory_needed:          strat.memory,
      estimated_duration_min: strat.duration,
      reasoning: `[HYBRID] ${strat.label.charAt(0).toUpperCase() + strat.label.slice(1)} — Q[${q}] | ${llmMsg}`,
      _strategy: strat.label,
      _hybrid:   true,
    }
  }

  return actions
}
