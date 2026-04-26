import { useEffect, useState } from 'react'
import { Results } from './api.js'

// ─── SVG helpers ──────────────────────────────────────────────────────────────

function LineChart({ data, xKey, yKey, color = '#3b82f6', label, height = 180 }) {
  if (!data || data.length < 2) return <Empty>Not enough data</Empty>

  const W = 520, H = height, PAD = { t: 16, r: 16, b: 36, l: 52 }
  const xs = data.map((d) => d[xKey])
  const ys = data.map((d) => d[yKey])
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys) * 0.97, yMax = Math.max(...ys) * 1.03

  const px = (x) => PAD.l + ((x - xMin) / (xMax - xMin || 1)) * (W - PAD.l - PAD.r)
  const py = (y) => H - PAD.b - ((y - yMin) / (yMax - yMin || 1)) * (H - PAD.t - PAD.b)

  const pts = data.map((d) => `${px(d[xKey])},${py(d[yKey])}`).join(' ')
  const fillPts = `${px(xs[0])},${py(yMin)} ${pts} ${px(xs[xs.length - 1])},${py(yMin)}`

  const yTicks = 4
  const xTicks = Math.min(data.length, 8)

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ display: 'block' }}>
      {/* Grid */}
      {Array.from({ length: yTicks + 1 }, (_, i) => {
        const y = py(yMin + (i / yTicks) * (yMax - yMin))
        const val = (yMin + (i / yTicks) * (yMax - yMin)).toFixed(0)
        return (
          <g key={i}>
            <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} stroke="#1e293b" strokeWidth={1} />
            <text x={PAD.l - 6} y={y + 4} textAnchor="end" fill="#475569" fontSize={10}>{val}</text>
          </g>
        )
      })}

      {/* X labels */}
      {data
        .filter((_, i) => i % Math.ceil(data.length / xTicks) === 0)
        .map((d) => (
          <text key={d[xKey]} x={px(d[xKey])} y={H - PAD.b + 16} textAnchor="middle" fill="#475569" fontSize={10}>
            {d[xKey]}
          </text>
        ))}

      {/* Area fill */}
      <polygon points={fillPts} fill={color} fillOpacity={0.08} />

      {/* Line */}
      <polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinejoin="round" />

      {/* Dots */}
      {data.map((d) => (
        <circle key={d[xKey]} cx={px(d[xKey])} cy={py(d[yKey])} r={3} fill={color} />
      ))}

      {/* Axis labels */}
      <text x={PAD.l} y={H - 2} textAnchor="start" fill="#334155" fontSize={10}>Episode →</text>
      <text
        x={10} y={H / 2} textAnchor="middle" fill="#334155" fontSize={10}
        transform={`rotate(-90, 10, ${H / 2})`}
      >
        {label}
      </text>
    </svg>
  )
}

function BarPair({ label, a, b, aLabel, bLabel, aColor, bColor, max }) {
  const pct = (v) => `${Math.min(100, (v / max) * 100).toFixed(1)}%`
  return (
    <div style={{ marginBottom: 16 }}>
      <div style={{ fontSize: 12, color: '#94a3b8', marginBottom: 6 }}>{label}</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
        <div style={{ width: 80, fontSize: 10, color: aColor, textAlign: 'right' }}>{aLabel}</div>
        <div style={{ flex: 1, height: 18, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: pct(a), height: '100%', background: aColor, borderRadius: 4, transition: 'width 0.8s ease', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6 }}>
            <span style={{ fontSize: 9, color: '#000', fontWeight: 700 }}>{typeof a === 'number' ? a.toFixed(1) : a}</span>
          </div>
        </div>
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <div style={{ width: 80, fontSize: 10, color: bColor, textAlign: 'right' }}>{bLabel}</div>
        <div style={{ flex: 1, height: 18, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
          <div style={{ width: pct(b), height: '100%', background: bColor, borderRadius: 4, transition: 'width 0.8s ease', display: 'flex', alignItems: 'center', justifyContent: 'flex-end', paddingRight: 6 }}>
            <span style={{ fontSize: 9, color: '#000', fontWeight: 700 }}>{typeof b === 'number' ? b.toFixed(1) : b}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function Card({ title, children, accent = '#3b82f6' }) {
  return (
    <div
      style={{
        background: '#0f172a',
        border: `1px solid #1e293b`,
        borderTop: `3px solid ${accent}`,
        borderRadius: 10,
        padding: '16px 18px',
        marginBottom: 18,
      }}
    >
      <div style={{ fontSize: 12, fontWeight: 700, color: accent, letterSpacing: 0.5, marginBottom: 12, textTransform: 'uppercase' }}>
        {title}
      </div>
      {children}
    </div>
  )
}

function Empty({ children }) {
  return <div style={{ color: '#334155', fontSize: 12, padding: '12px 0', textAlign: 'center' }}>{children}</div>
}

function Stat({ label, value, sub, color = '#e2e8f0' }) {
  return (
    <div style={{ background: '#0a0f1a', border: '1px solid #1e293b', borderRadius: 8, padding: '10px 14px', minWidth: 110 }}>
      <div style={{ fontSize: 11, color: '#475569', marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 20, fontWeight: 700, color, fontFamily: 'JetBrains Mono, monospace' }}>{value}</div>
      {sub && <div style={{ fontSize: 10, color: '#334155', marginTop: 2 }}>{sub}</div>}
    </div>
  )
}

// ─── Main ResultsPanel ────────────────────────────────────────────────────────

export default function ResultsPanel() {
  const [training, setTraining]   = useState(null)
  const [theme4,   setTheme4]     = useState(null)
  const [holdout,  setHoldout]    = useState(null)
  const [selfplay, setSelfplay]   = useState(null)
  const [loading,  setLoading]    = useState(true)
  const [error,    setError]      = useState(null)

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)

    Promise.allSettled([
      Results.training(),
      Results.theme4(),
      Results.holdout(),
      Results.selfplay(),
    ]).then(([t, th, h, sp]) => {
      if (cancelled) return
      if (t.status  === 'fulfilled') setTraining(t.value)
      if (th.status === 'fulfilled') setTheme4(th.value)
      if (h.status  === 'fulfilled') setHoldout(h.value)
      if (sp.status === 'fulfilled') setSelfplay(sp.value)

      const allFailed = [t, th, h, sp].every((r) => r.status === 'rejected')
      if (allFailed) setError('Results files not found. Run training first:\n  python train.py')
      setLoading(false)
    })

    return () => { cancelled = true }
  }, [])

  // ── Derived data ─────────────────────────────────────────────────────────

  const rewardCurve = training
    ? training
        .filter((ep) => ep.episode && ep.total_reward)
        .map((ep) => ({ episode: ep.episode, reward: Math.round(ep.total_reward) }))
    : []

  const completionCurve = training
    ? training
        .filter((ep) => ep.episode && ep.completed_tasks !== undefined)
        .map((ep) => ({
          episode: ep.episode,
          pct: ep.completed_tasks ? Math.round((ep.completed_tasks / (ep.total_tasks || 1)) * 100) : 0,
        }))
    : []

  const firstReward = rewardCurve[0]?.reward ?? 0
  const lastReward  = rewardCurve[rewardCurve.length - 1]?.reward ?? 0
  const delta       = lastReward - firstReward
  const deltaPct    = firstReward > 0 ? ((delta / firstReward) * 100).toFixed(1) : '—'

  const curriculum  = theme4?.curriculum_history ?? []
  const duels       = theme4?.league_duels        ?? []

  const holdoutTrained = holdout?.trained_summary ?? holdout?.holdout_trained_summary
  const holdoutFresh   = holdout?.fresh_summary   ?? holdout?.holdout_fresh_summary
  const holdoutMax     = holdoutTrained
    ? Math.max(holdoutTrained.avg_total_reward, holdoutFresh?.avg_total_reward ?? 0) * 1.05
    : 1000

  const selfplaySnaps = selfplay?.league_recent_snapshots ?? []

  // ── Render ───────────────────────────────────────────────────────────────

  if (loading) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#475569', fontSize: 14 }}>
        Loading results…
      </div>
    )
  }

  if (error && !training && !theme4 && !holdout) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
        <div style={{ background: '#1e0a0a', border: '1px solid #ef444433', borderRadius: 10, padding: 24, maxWidth: 400, textAlign: 'center' }}>
          <div style={{ fontSize: 24, marginBottom: 8 }}>📂</div>
          <div style={{ color: '#ef4444', fontSize: 13, marginBottom: 8 }}>Results Not Found</div>
          <div style={{ color: '#64748b', fontSize: 12, fontFamily: 'JetBrains Mono, monospace', whiteSpace: 'pre' }}>{error}</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ padding: '20px 28px', overflowY: 'auto', height: '100%' }}>

      {/* ── Key stats row ── */}
      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
        <Stat
          label="Episodes Trained"
          value={rewardCurve.length || '—'}
          color="#38bdf8"
        />
        <Stat
          label="Reward Ep.1"
          value={firstReward || '—'}
          color="#94a3b8"
          sub="starting baseline"
        />
        <Stat
          label={`Reward Ep.${rewardCurve.length}`}
          value={lastReward || '—'}
          color="#10b981"
          sub="after training"
        />
        <Stat
          label="Improvement"
          value={delta > 0 ? `+${delta}` : delta || '—'}
          color={delta > 0 ? '#10b981' : '#ef4444'}
          sub={`${deltaPct}% over baseline`}
        />
        {theme4 && (
          <Stat
            label="Curriculum Phases"
            value={curriculum.length}
            color="#a855f7"
            sub={`${duels.length} league duels`}
          />
        )}
      </div>

      {/* ── Reward curve ── */}
      {rewardCurve.length > 1 && (
        <Card title="📈 Reward Curve — Episode 1 → Last" accent="#3b82f6">
          <LineChart data={rewardCurve} xKey="episode" yKey="reward" color="#3b82f6" label="Total Reward" />
          <div style={{ display: 'flex', gap: 16, marginTop: 8, fontSize: 11, color: '#475569' }}>
            <span>Start: <strong style={{ color: '#94a3b8' }}>{firstReward}</strong></span>
            <span>End: <strong style={{ color: '#10b981' }}>{lastReward}</strong></span>
            <span>Gain: <strong style={{ color: '#fbbf24' }}>+{delta} ({deltaPct}%)</strong></span>
          </div>
        </Card>
      )}

      {/* ── Task completion curve ── */}
      {completionCurve.length > 1 && (
        <Card title="✅ Task Completion % per Episode" accent="#10b981">
          <LineChart data={completionCurve} xKey="episode" yKey="pct" color="#10b981" label="Completion %" height={150} />
        </Card>
      )}

      {/* ── Holdout: Trained vs Fresh ── */}
      {holdoutTrained && holdoutFresh && (
        <Card title="🧪 Holdout Test — Trained Agent vs Fresh Agent" accent="#f59e0b">
          <div style={{ fontSize: 11, color: '#475569', marginBottom: 14 }}>
            Unseen compound crisis scenarios (GPU outage + urgent task injection)
          </div>
          <BarPair
            label="Total Reward"
            a={holdoutTrained.avg_total_reward}
            b={holdoutFresh.avg_total_reward}
            aLabel="Trained"
            bLabel="Fresh"
            aColor="#10b981"
            bColor="#64748b"
            max={holdoutMax}
          />
          <BarPair
            label="Belief Accuracy"
            a={holdoutTrained.avg_belief_accuracy * 100}
            b={holdoutFresh.avg_belief_accuracy * 100}
            aLabel="Trained"
            bLabel="Fresh"
            aColor="#a855f7"
            bColor="#64748b"
            max={100}
          />
          {holdout?.delta && (
            <div style={{ marginTop: 12, padding: '8px 12px', background: '#052e16', border: '1px solid #16a34a33', borderRadius: 8, fontSize: 12, color: '#4ade80' }}>
              ✓ Trained agent outperforms fresh by{' '}
              <strong>+{holdout.delta.avg_total_reward?.toFixed(1) ?? '—'} reward</strong>
              {' '}— confirms real learning
            </div>
          )}
        </Card>
      )}

      {/* ── Curriculum phases ── */}
      {curriculum.length > 0 && (
        <Card title="🎓 Curriculum Phases (Adaptive Difficulty)" accent="#a855f7">
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
              <thead>
                <tr>
                  {['Phase', 'Level', 'Completion', 'Fairness', 'Avg Reward', 'Promoted'].map((h) => (
                    <th key={h} style={{ textAlign: 'left', padding: '6px 10px', color: '#475569', borderBottom: '1px solid #1e293b', fontWeight: 600, whiteSpace: 'nowrap' }}>
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {curriculum.map((row, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #0f172a' }}>
                    <td style={{ padding: '6px 10px', color: '#94a3b8' }}>{row.phase}</td>
                    <td style={{ padding: '6px 10px', color: '#38bdf8', fontFamily: 'JetBrains Mono, monospace' }}>L{row.level}</td>
                    <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{(row.completion * 100).toFixed(0)}%</td>
                    <td style={{ padding: '6px 10px', color: '#e2e8f0' }}>{row.fairness?.toFixed(2) ?? '—'}</td>
                    <td style={{ padding: '6px 10px', color: '#fbbf24', fontFamily: 'JetBrains Mono, monospace' }}>
                      {row.avg_total_reward?.toFixed(0) ?? '—'}
                    </td>
                    <td style={{ padding: '6px 10px' }}>
                      <span style={{
                        fontSize: 10, fontWeight: 700, padding: '2px 7px', borderRadius: 4,
                        background: row.promoted ? '#052e16' : '#1e0a0a',
                        color: row.promoted ? '#4ade80' : '#ef4444',
                      }}>
                        {row.promoted ? '↑ YES' : '— NO'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {/* ── Self-play league duels ── */}
      {duels.length > 0 && (
        <Card title="⚔️ Self-Play League Duels" accent="#f43f5e">
          <div style={{ fontSize: 11, color: '#475569', marginBottom: 10 }}>
            Current policy vs snapshot from earlier phase — on unseen crisis scenarios
          </div>
          {duels.map((d, i) => (
            <div key={i} style={{ display: 'flex', gap: 10, alignItems: 'center', marginBottom: 8, fontSize: 12 }}>
              <span style={{ color: '#64748b', minWidth: 60 }}>Phase {d.phase}</span>
              <span style={{ color: '#a855f7', fontFamily: 'JetBrains Mono, monospace' }}>{d.learner_agent}</span>
              <span style={{ color: '#334155' }}>vs Phase {d.opponent_phase} snapshot</span>
              <span style={{ color: '#fbbf24', marginLeft: 'auto', fontFamily: 'JetBrains Mono, monospace' }}>
                {d.reward?.toFixed(0) ?? '—'} pts
              </span>
              <span style={{ color: '#64748b', fontSize: 10, minWidth: 100 }}>({d.scenario})</span>
            </div>
          ))}
        </Card>
      )}

      {/* ── Self-play Q-table growth ── */}
      {selfplaySnaps.length > 0 && (
        <Card title="🧠 Q-Table State Space Growth" accent="#06b6d4">
          <div style={{ fontSize: 11, color: '#475569', marginBottom: 10 }}>
            More Q-states = more nuanced decision making — proof of learning
          </div>
          {Object.entries(selfplaySnaps[selfplaySnaps.length - 1]?.q_state_counts ?? {}).map(([agent, count]) => {
            const firstCount = selfplaySnaps[0]?.q_state_counts?.[agent] ?? 1
            const growth = count - firstCount
            const cfg = { rl_data_loader: '#3b82f6', rl_data_cleaner: '#10b981', rl_ml_trainer: '#a855f7' }
            const color = cfg[agent] ?? '#64748b'
            return (
              <div key={agent} style={{ marginBottom: 10 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 11, color: '#64748b', marginBottom: 3 }}>
                  <span>{agent.replace('rl_', '')}</span>
                  <span style={{ color, fontFamily: 'JetBrains Mono, monospace' }}>
                    {firstCount} → {count} <span style={{ color: '#10b981' }}>(+{growth})</span>
                  </span>
                </div>
                <div style={{ height: 8, background: '#1e293b', borderRadius: 4, overflow: 'hidden' }}>
                  <div style={{ height: '100%', width: `${Math.min(100, (count / 500) * 100)}%`, background: color, borderRadius: 4, transition: 'width 1s ease' }} />
                </div>
              </div>
            )
          })}
        </Card>
      )}

      {/* ── PNG charts if available ── */}
      <Card title="📊 Training Plots" accent="#64748b">
        <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
          {['reward_curve.png', 'metrics_dashboard.png'].map((img) => (
            <div key={img} style={{ flex: 1, minWidth: 220 }}>
              <div style={{ fontSize: 10, color: '#475569', marginBottom: 6, fontFamily: 'JetBrains Mono, monospace' }}>{img}</div>
              <img
                src={`/results/${img}`}
                alt={img}
                style={{ width: '100%', borderRadius: 8, border: '1px solid #1e293b', background: '#0f172a' }}
                onError={(e) => { e.target.style.display = 'none' }}
              />
            </div>
          ))}
        </div>
      </Card>

    </div>
  )
}
