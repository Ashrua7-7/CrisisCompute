from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from .models import Task
from .tasks import dependency_ready


@dataclass
class AgentIntent:
    agent_id: str
    action: str
    task_id: str | None
    cpu: int
    gpu: int
    memory: int
    urgency_claim: float
    predicted_competitor_gpu: float
    predicted_competitor_cpu: float
    predicted_competitor_memory: float
    yield_probability: float


@dataclass
class NegotiationSnapshot:
    emergency_charter: bool
    crisis_agents: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    concessions: List[str] = field(default_factory=list)
    yields: List[str] = field(default_factory=list)
    coalitions: List[str] = field(default_factory=list)
    contracts_kept: int = 0
    contracts_broken: int = 0
    belief_accuracy: Dict[str, float] = field(default_factory=dict)
    fairness_score: float = 1.0
    deadlock: bool = False


def _task_urgency(task: Task, current_hour: int) -> float:
    time_left = max(task.deadline_hour - current_hour, 0.0)
    pressure = 1.0 / (time_left + 1.0)
    duration = task.estimated_duration_min / 60.0
    return pressure * 4.0 + duration


def _critical_window(task: Task, current_hour: int) -> bool:
    return task.status != "done" and (task.deadline_hour - current_hour) <= 1.0


def build_intents(
    agent_order: List[str],
    actions: Dict[str, Dict],
    tasks: Dict[str, Task],
    beliefs: Dict[str, Dict[str, Dict[str, float]]],
    current_hour: int,
) -> Dict[str, AgentIntent]:
    intents: Dict[str, AgentIntent] = {}
    for agent_id in agent_order:
        action = actions[agent_id]
        task = tasks.get(action.get("task_id"))
        cpu = int(action.get("cores_needed", 0))
        gpu = int(action.get("gpu_needed", 0))
        memory = int(action.get("memory_needed", 0))
        urgency_claim = float(action.get("urgency_claim", 0.0))
        if task is not None:
            cpu = max(cpu, task.cores_needed)
            gpu = max(gpu, task.gpu_needed)
            memory = max(memory, task.memory_needed)
            urgency_claim = max(urgency_claim, _task_urgency(task, current_hour))

        others = beliefs.get(agent_id, {})
        peer_count = max(len(others), 1)
        pred_gpu = sum(peer.get("predicted_gpu_demand", 0.0) for peer in others.values()) / peer_count
        pred_cpu = sum(peer.get("predicted_cpu_demand", 0.0) for peer in others.values()) / peer_count
        pred_mem = sum(peer.get("predicted_memory_demand", 0.0) for peer in others.values()) / peer_count
        pred_yield = sum(peer.get("predicted_yield_probability", 0.4) for peer in others.values()) / peer_count

        intents[agent_id] = AgentIntent(
            agent_id=agent_id,
            action=action.get("action", "wait"),
            task_id=action.get("task_id"),
            cpu=cpu,
            gpu=gpu,
            memory=memory,
            urgency_claim=urgency_claim,
            predicted_competitor_gpu=pred_gpu,
            predicted_competitor_cpu=pred_cpu,
            predicted_competitor_memory=pred_mem,
            yield_probability=pred_yield,
        )
    return intents


def run_negotiation(
    intents: Dict[str, AgentIntent],
    tasks: Dict[str, Task],
    done_tasks: set[str],
    current_hour: int,
    total_cpu: int,
    total_gpu: int,
    total_memory: int,
    reputation: Dict[str, float],
    open_contracts: List[Dict],
) -> Tuple[Dict[str, Dict], NegotiationSnapshot]:
    negotiated = {
        agent_id: {
            "action": intent.action,
            "task_id": intent.task_id,
            "cores_needed": intent.cpu,
            "gpu_needed": intent.gpu,
            "memory_needed": intent.memory,
            "estimated_duration_min": 60,
        }
        for agent_id, intent in intents.items()
    }
    snapshot = NegotiationSnapshot(emergency_charter=False)

    runnable: Dict[str, Task] = {}
    for agent_id, intent in intents.items():
        if intent.action != "run_task" or not intent.task_id:
            continue
        task = tasks.get(intent.task_id)
        if task is None or not dependency_ready(task, done_tasks):
            continue
        runnable[agent_id] = task

    crisis_agents = [agent_id for agent_id, task in runnable.items() if _critical_window(task, current_hour)]
    snapshot.crisis_agents = crisis_agents
    snapshot.emergency_charter = len(crisis_agents) > 0

    total_req_cpu = sum(intents[a].cpu for a in runnable)
    total_req_gpu = sum(intents[a].gpu for a in runnable)
    total_req_mem = sum(intents[a].memory for a in runnable)
    if total_req_cpu > total_cpu:
        snapshot.conflicts.append("cpu")
    if total_req_gpu > total_gpu:
        snapshot.conflicts.append("gpu")
    if total_req_mem > total_memory:
        snapshot.conflicts.append("memory")

    kept = 0
    broken = 0
    for contract in open_contracts:
        debtor = contract["debtor"]
        if debtor not in runnable:
            kept += 1
            contract["status"] = "kept"
            continue
        should_yield = contract["resource"] in snapshot.conflicts and contract["expires_at_hour"] >= current_hour
        if should_yield:
            negotiated[debtor]["action"] = "wait"
            contract["status"] = "kept"
            kept += 1
            snapshot.concessions.append(f"{debtor}:contract_yield:{contract['resource']}")
            snapshot.yields.append(f"{debtor}:{contract['resource']}")
        else:
            contract["status"] = "broken"
            broken += 1
    snapshot.contracts_kept = kept
    snapshot.contracts_broken = broken

    if not snapshot.conflicts:
        return negotiated, snapshot

    contenders = [agent_id for agent_id in runnable if negotiated[agent_id]["action"] == "run_task"]
    scored: List[Tuple[float, str]] = []
    for agent_id in contenders:
        intent = intents[agent_id]
        task = runnable[agent_id]
        urgency = _task_urgency(task, current_hour)
        emergency_bonus = 5.0 if snapshot.emergency_charter and agent_id in crisis_agents else 0.0
        score = urgency + emergency_bonus + (reputation.get(agent_id, 1.0) * 0.8) - (intent.yield_probability * 0.5)
        scored.append((score, agent_id))
    scored.sort(reverse=True)

    cpu_left, gpu_left, mem_left = total_cpu, total_gpu, total_memory
    winners: List[str] = []
    losers: List[str] = []

    for _, agent_id in scored:
        cpu = negotiated[agent_id]["cores_needed"]
        gpu = negotiated[agent_id]["gpu_needed"]
        mem = negotiated[agent_id]["memory_needed"]
        if cpu <= cpu_left and gpu <= gpu_left and mem <= mem_left:
            winners.append(agent_id)
            cpu_left -= cpu
            gpu_left -= gpu
            mem_left -= mem
        else:
            negotiated[agent_id]["action"] = "wait"
            losers.append(agent_id)
            blocked_resources: List[str] = []
            if cpu > cpu_left:
                blocked_resources.append("cpu")
            if gpu > gpu_left:
                blocked_resources.append("gpu")
            if mem > mem_left:
                blocked_resources.append("memory")
            if blocked_resources:
                snapshot.concessions.append(f"{agent_id}:capacity_yield:{'+'.join(blocked_resources)}")
                snapshot.yields.extend(f"{agent_id}:{res}" for res in blocked_resources)

    if winners and losers:
        top_winner = winners[0]
        for loser in losers:
            if intents[top_winner].gpu > 0 and intents[loser].gpu > 0:
                snapshot.coalitions.append(f"{loser}->{top_winner}:gpu_yield")
                open_contracts.append(
                    {
                        "creditor": loser,
                        "debtor": top_winner,
                        "resource": "gpu",
                        "created_at_hour": current_hour,
                        "expires_at_hour": current_hour + 2,
                        "status": "open",
                    }
                )

    starvation_penalty = 0.0
    if contenders:
        wait_ratio = len(losers) / len(contenders)
        starvation_penalty = max(0.0, wait_ratio - 0.5)
    snapshot.fairness_score = max(0.0, 1.0 - starvation_penalty)
    snapshot.deadlock = len(winners) == 0 and len(contenders) > 0
    return negotiated, snapshot


def belief_accuracy_from_demands(
    beliefs: Dict[str, Dict[str, Dict[str, float]]],
    intents: Dict[str, AgentIntent],
) -> Dict[str, float]:
    accuracy: Dict[str, float] = {}
    for observer, peer_beliefs in beliefs.items():
        if not peer_beliefs:
            accuracy[observer] = 1.0
            continue
        scores: List[float] = []
        for peer, belief in peer_beliefs.items():
            peer_intent = intents.get(peer)
            if peer_intent is None:
                continue
            cpu_err = abs(belief.get("predicted_cpu_demand", 0.0) - peer_intent.cpu) / max(peer_intent.cpu, 1)
            gpu_err = abs(belief.get("predicted_gpu_demand", 0.0) - peer_intent.gpu) / max(peer_intent.gpu, 1)
            mem_err = abs(belief.get("predicted_memory_demand", 0.0) - peer_intent.memory) / max(peer_intent.memory, 1)
            mean_err = (cpu_err + gpu_err + mem_err) / 3.0
            scores.append(max(0.0, 1.0 - mean_err))
        accuracy[observer] = sum(scores) / len(scores) if scores else 1.0
    return accuracy

