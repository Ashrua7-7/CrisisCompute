from __future__ import annotations

from typing import Dict, Tuple

from .models import ResourcePool, Task
from .tasks import dependency_ready


REQUIRED_ACTION_KEYS = {
    "action",
    "task_id",
    "cores_needed",
    "gpu_needed",
    "memory_needed",
    "estimated_duration_min",
    "urgency_claim",
}


def normalize_action(action: Dict) -> Dict:
    normalized = {
        "action": action.get("action", "wait"),
        "task_id": action.get("task_id"),
        "cores_needed": int(action.get("cores_needed", 0) or 0),
        "gpu_needed": int(action.get("gpu_needed", 0) or 0),
        "memory_needed": int(action.get("memory_needed", 0) or 0),
        "estimated_duration_min": int(action.get("estimated_duration_min", 0) or 0),
        "urgency_claim": float(action.get("urgency_claim", 0.0) or 0.0),
    }
    return normalized


def validate_action(agent_id: str, action: Dict, tasks: Dict[str, Task]) -> Tuple[bool, str]:
    if not REQUIRED_ACTION_KEYS.issubset(set(action.keys())):
        return False, "missing_action_keys"

    if action["action"] not in {"run_task", "wait"}:
        return False, "unknown_action"

    if action["action"] == "wait":
        return True, "ok"

    task_id = action["task_id"]
    if not task_id or task_id not in tasks:
        return False, "task_not_found"

    task = tasks[task_id]
    if task.owner_agent != agent_id:
        return False, "wrong_task_owner"

    if task.status == "done":
        return False, "task_already_done"

    if action["cores_needed"] <= 0 or action["memory_needed"] <= 0:
        return False, "invalid_resource_request"

    return True, "ok"


def validate_task_runtime_constraints(task: Task, done_tasks: set[str], pool: ResourcePool) -> Tuple[bool, str]:
    if not dependency_ready(task, done_tasks):
        return False, "dependency_not_ready"

    if task.cores_needed > pool.total_cpu or task.gpu_needed > pool.total_gpu:
        return False, "request_exceeds_system_capacity"

    if task.memory_needed > pool.total_memory:
        return False, "request_exceeds_memory_capacity"

    return True, "ok"
