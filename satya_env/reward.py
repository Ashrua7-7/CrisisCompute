from __future__ import annotations

from typing import Dict, Iterable

from .models import AgentRuntimeState, Task


def _task_reward(task: Task) -> float:
    """
    Calculate reward for a single task with deadline sensitivity.
    Rewards on-time completion. Penalizes lateness but not excessively.
    """
    if task.status != "done":
        return 0.0
    
    if task.completed_at_hour is None:
        return 8.0  # reward for completion
    
    # Calculate how much time the task had before deadline
    time_remaining = task.deadline_hour - task.completed_at_hour
    
    if task.completed_at_hour <= task.deadline_hour:
        # Completed on time: reward increases with how much early
        base_reward = 12.0
        early_bonus = max(0, time_remaining * 1.5)  # +1.5 points per hour early
        return base_reward + early_bonus
    else:
        # MISSED DEADLINE: moderate penalty
        time_overdue = task.completed_at_hour - task.deadline_hour
        penalty = time_overdue * 1.5  # -1.5 per hour late
        return max(0.0, 10.0 - penalty)  # floor at 0


def calculate_individual_rewards(
    agent_states: Dict[str, AgentRuntimeState],
    tasks: Dict[str, Task],
) -> Dict[str, float]:
    """
    Calculate individual rewards with deadline incentives.
    Rewards on-time completion, small penalty for lateness.
    """
    rewards: Dict[str, float] = {agent_id: 0.0 for agent_id in agent_states.keys()}

    for task in tasks.values():
        task_reward = _task_reward(task)
        rewards[task.owner_agent] += task_reward
        
        # On-time bonus
        if task.status == "done" and task.completed_at_hour is not None:
            if task.completed_at_hour <= task.deadline_hour:
                rewards[task.owner_agent] += 2.0  # Small bonus for meeting deadline
            else:
                time_overdue = task.completed_at_hour - task.deadline_hour
                rewards[task.owner_agent] -= min(4.0, time_overdue)  # Small penalty

    # Agent efficiency penalties
    for agent_id, state in agent_states.items():
        # Small penalty for waiting (encourages negotiation)
        wait_penalty = (state.waiting_minutes / 10.0) * 0.2
        rewards[agent_id] -= wait_penalty
        
        # Penalty for missed deadlines
        deadline_penalty = state.missed_deadlines * 3.0
        rewards[agent_id] -= deadline_penalty

    return rewards


def calculate_team_reward(
    tasks: Dict[str, Task],
    max_hours: int,
    current_hour: int,
    recent_events: Iterable[str],
    negotiation_snapshot: Dict | None = None,
) -> float:
    """
    Calculate team-level reward.
    Rewards completion and on-time performance.
    Small penalties for incomplete work.
    """
    total = len(tasks)
    if total == 0:
        return 0.0

    done = sum(1 for task in tasks.values() if task.status == "done")
    on_time = sum(
        1
        for task in tasks.values()
        if task.status == "done"
        and task.completed_at_hour is not None
        and task.completed_at_hour <= task.deadline_hour
    )
    
    utilization_hint = sum(1 for event in recent_events if "allocated" in event)
    repeated_blocking = sum(
        1
        for event in recent_events
        if ("conflict prevented" in event) or ("blocked by dependency" in event)
    )

    # Completion bonus
    completion_bonus = 20.0 * (done / total)
    
    # On-time bonus
    on_time_bonus = 20.0 * (on_time / total)
    
    # Negotiation bonus
    negotiation_bonus = min(utilization_hint, 3) * 1.5
    
    # Time efficiency
    time_efficiency = max(0, (max_hours - current_hour) / max_hours) * 5.0
    
    # Penalty for incomplete work (mild)
    incomplete_penalty = 10.0 * ((total - done) / total)
    
    # Finish bonus if everything done on time
    finish_bonus = 20.0 if done == total and on_time == total else 0.0

    diplomacy_bonus = 0.0
    if negotiation_snapshot:
        fairness = float(negotiation_snapshot.get("fairness_score", 1.0))
        emergency_bonus = 4.0 if negotiation_snapshot.get("emergency_charter") else 0.0
        contract_penalty = float(negotiation_snapshot.get("contracts_broken", 0)) * 1.5
        contracts_kept = float(negotiation_snapshot.get("contracts_kept", 0))
        yields_count = len(negotiation_snapshot.get("yields", []))
        deadlock_penalty = 2.0 if negotiation_snapshot.get("deadlock") else 0.0
        fairness_bonus = max(0.0, fairness - 0.8) * 5.0
        high_fairness_bonus = 1.0 if fairness >= 0.92 else 0.0
        cooperative_yield_bonus = yields_count * 0.4
        repeated_blocking_penalty = max(0.0, repeated_blocking - 1) * 0.5
        diplomacy_bonus += (
            (fairness * 4.0)
            + emergency_bonus
            + (contracts_kept * 0.6)
            + cooperative_yield_bonus
            + fairness_bonus
            + high_fairness_bonus
            - contract_penalty
            - deadlock_penalty
            - repeated_blocking_penalty
        )

    return completion_bonus + on_time_bonus + negotiation_bonus + time_efficiency + finish_bonus + diplomacy_bonus - incomplete_penalty


def calculate_final_rewards(
    individual: Dict[str, float],
    team_reward: float,
    weight_individual: float = 0.6,
    weight_team: float = 0.4,
) -> Dict[str, float]:
    return {
        agent_id: weight_individual * ind_reward + weight_team * team_reward
        for agent_id, ind_reward in individual.items()
    }
