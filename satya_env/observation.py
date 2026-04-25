from __future__ import annotations

from typing import Dict, List

from .models import AgentRuntimeState, EpisodeState, ResourcePool, Task
from .tasks import tasks_for_agent


def build_observation(
    requesting_agent: str,
    episode_index: int,
    episode: EpisodeState,
    pool: ResourcePool,
    tasks: Dict[str, Task],
    agent_states: Dict[str, AgentRuntimeState],
    reputation: Dict[str, float],
    beliefs: Dict[str, Dict[str, float]],
    negotiation_snapshot: Dict | None,
) -> Dict:
    my_tasks = tasks_for_agent(tasks, requesting_agent)
    pending = [task.task_id for task in my_tasks if task.status == "pending"]
    running = [task.task_id for task in my_tasks if task.status == "running"]
    done = [task.task_id for task in my_tasks if task.status == "done"]

    other_agents_status: List[Dict] = []
    for agent_id, state in agent_states.items():
        if agent_id == requesting_agent:
            continue
        other_agents_status.append(
            {
                "agent_id": agent_id,
                "running_task_id": state.running_task_id,
                "completed_tasks": state.completed_tasks,
                "missed_deadlines": state.missed_deadlines,
            }
        )

    return {
        "episode": episode_index,
        "hour": episode.hour,
        "time_left_hours": max(episode.max_hours - episode.hour, 0),
        "available_resources": {
            "cpu": {
                "total": pool.total_cpu,
                "available": pool.available_cpu,
                "allocated": pool.total_cpu - pool.available_cpu,
            },
            "gpu": {
                "total": pool.total_gpu,
                "available": pool.available_gpu,
                "allocated": pool.total_gpu - pool.available_gpu,
            },
            "memory": {
                "total": pool.total_memory,
                "available": pool.available_memory,
                "allocated": pool.total_memory - pool.available_memory,
            },
        },
        "my_tasks": {
            "pending": pending,
            "running": running,
            "done": done,
        },
        "other_agents_status": other_agents_status,
        "diplomacy": {
            "my_reputation": reputation.get(requesting_agent, 1.0),
            "peer_beliefs": beliefs,
            "latest_negotiation": negotiation_snapshot or {},
        },
        "recent_events": episode.recent_events[-8:],
        "system_messages": [
            "Action schema: run_task or wait with urgency_claim",
            "Dependencies must be satisfied before running",
            "Negotiation charter can preempt greedy allocation",
        ],
    }
