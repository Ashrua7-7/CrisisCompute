from __future__ import annotations

from typing import Dict, List, Tuple

from .models import Allocation, ResourcePool, Task
from .tasks import dependency_ready


def _urgency_score(task: Task, current_hour: int) -> float:
    time_left = max(task.deadline_hour - current_hour, 0)
    deadline_pressure = 1.0 / (time_left + 1)
    duration_factor = task.estimated_duration_min / 60.0
    return deadline_pressure * 3.0 + duration_factor


def resolve_and_allocate(
    proposed_actions: Dict[str, Dict],
    tasks: Dict[str, Task],
    pool: ResourcePool,
    done_tasks: set[str],
    current_hour: int,
) -> Tuple[List[Allocation], List[str]]:
    accepted: List[Allocation] = []
    events: List[str] = []

    run_candidates: List[Tuple[str, Dict, Task]] = []
    for agent_id, action in proposed_actions.items():
        if action.get("action") != "run_task":
            events.append(f"{agent_id} waits")
            continue

        task_id = action.get("task_id")
        task = tasks.get(task_id)
        if task is None:
            events.append(f"{agent_id} proposed unknown task")
            continue
        if task.status == "done":
            events.append(f"{agent_id} proposed already completed task {task.task_id}")
            continue
        if not dependency_ready(task, done_tasks):
            events.append(f"{agent_id} blocked by dependency for {task.task_id}")
            continue
        run_candidates.append((agent_id, action, task))

    run_candidates.sort(key=lambda item: _urgency_score(item[2], current_hour), reverse=True)

    cpu_left = pool.total_cpu
    gpu_left = pool.total_gpu
    mem_left = pool.total_memory

    for agent_id, action, task in run_candidates:
        requested_cpu = int(action.get("cores_needed", task.cores_needed))
        requested_gpu = int(action.get("gpu_needed", task.gpu_needed))
        requested_mem = int(action.get("memory_needed", task.memory_needed))

        cpu = max(task.cores_needed, requested_cpu)
        gpu = max(task.gpu_needed, requested_gpu)
        mem = max(task.memory_needed, requested_mem)

        if cpu <= cpu_left and gpu <= gpu_left and mem <= mem_left:
            accepted.append(
                Allocation(
                    agent_id=agent_id,
                    task_id=task.task_id,
                    cpu=cpu,
                    gpu=gpu,
                    memory=mem,
                )
            )
            cpu_left -= cpu
            gpu_left -= gpu
            mem_left -= mem
            events.append(f"allocated {task.task_id} to {agent_id}")
        else:
            events.append(f"conflict prevented {agent_id} from running {task.task_id}")

    pool.available_cpu = cpu_left
    pool.available_gpu = gpu_left
    pool.available_memory = mem_left
    return accepted, events
