/**
 * normalize.js — Adapts the actual HF Space API response format to a clean
 * internal shape that the UI can render without crashing.
 *
 * Real API quirks discovered:
 *  - my_tasks is {pending:"t1 t2", running:"", done:""} not an array
 *  - available_resources.cpu is "@{total=16; available=16; allocated=0}" string
 */

// Parse PowerShell-style resource string "@{total=16; available=16; allocated=0}"
export function parseResourceStr(val) {
  if (val === null || val === undefined) return { total: 0, available: 0, allocated: 0 }
  if (typeof val === 'object') return val  // already parsed
  const m = String(val).match(/@\{([^}]+)\}/)
  if (!m) return { total: 0, available: 0, allocated: 0 }
  const out = {}
  m[1].split(';').forEach((part) => {
    const [k, v] = part.trim().split('=')
    if (k) out[k.trim()] = isNaN(v) ? (v || '').trim() : Number(v)
  })
  return out
}

// Convert my_tasks object or array → uniform array of {task_id, status}
export function normalizeTasks(myTasks) {
  if (!myTasks) return []
  if (Array.isArray(myTasks)) return myTasks  // already array (local format)

  // Object format: {pending: "t1 t2", running: "t3", done: "t4"}
  const out = []
  for (const status of ['pending', 'running', 'done']) {
    const ids = String(myTasks[status] || '').trim().split(/\s+/).filter(Boolean)
    for (const id of ids) out.push({ task_id: id, id, status })
  }
  return out
}

// Normalise one agent's observation slice into a consistent shape
export function normalizeAgentObs(raw) {
  if (!raw) return { my_tasks: [], available_cpu: 0, available_gpu: 0, available_memory: 0 }

  const tasks = normalizeTasks(raw.my_tasks)

  // Parse resource strings (or use direct fields if already numbers)
  const res = raw.available_resources ?? {}
  const cpu    = parseResourceStr(res.cpu    ?? res.cpu_cores)
  const gpu    = parseResourceStr(res.gpu)
  const memory = parseResourceStr(res.memory ?? res.memory_gb)

  return {
    ...raw,
    my_tasks:         tasks,
    available_cpu:    cpu.available    ?? raw.available_cpu    ?? 0,
    available_gpu:    gpu.available    ?? raw.available_gpu    ?? 0,
    available_memory: memory.available ?? raw.available_memory ?? 0,
    total_cpu:        cpu.total    ?? 16,
    total_gpu:        gpu.total    ?? 1,
    total_memory:     memory.total ?? 32,
  }
}

// Normalise the full joint observation
export function normalizeObservation(obs) {
  if (!obs) return null
  const rawAgentObs = obs.agent_observations ?? {}
  const agent_observations = {}
  for (const [k, v] of Object.entries(rawAgentObs)) {
    agent_observations[k] = normalizeAgentObs(v)
  }
  return { ...obs, agent_observations }
}

// Get pool-level resource totals from normalised observation
export function poolResources(obs) {
  if (!obs) return { cpuTotal: 16, gpuTotal: 1, memTotal: 32, cpuUsed: 0, gpuUsed: 0, memUsed: 0 }
  const agents  = Object.values(obs.agent_observations ?? {})
  const first   = agents[0] ?? {}
  const cpuTotal  = first.total_cpu    ?? 16
  const gpuTotal  = first.total_gpu    ?? 1
  const memTotal  = first.total_memory ?? 32
  const cpuAvail  = first.available_cpu    ?? cpuTotal
  const gpuAvail  = first.available_gpu    ?? gpuTotal
  const memAvail  = first.available_memory ?? memTotal
  return {
    cpuTotal, gpuTotal, memTotal,
    cpuUsed: cpuTotal - cpuAvail,
    gpuUsed: gpuTotal - gpuAvail,
    memUsed: memTotal - memAvail,
  }
}
