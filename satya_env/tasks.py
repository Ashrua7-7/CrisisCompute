from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .models import Task


def load_tasks_from_json(path: str | Path) -> Dict[str, Task]:
    task_path = Path(path)
    with task_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    tasks: Dict[str, Task] = {}
    for item in payload.get("tasks", []):
        task = Task(
            task_id=item["task_id"],
            owner_agent=item["owner_agent"],
            title=item.get("title", item["task_id"]),
            cores_needed=int(item["cores_needed"]),
            gpu_needed=int(item.get("gpu_needed", 0)),
            memory_needed=int(item["memory_needed"]),
            estimated_duration_min=int(item["estimated_duration_min"]),
            deadline_hour=int(item["deadline_hour"]),
            dependencies=list(item.get("dependencies", [])),
        )
        tasks[task.task_id] = task
    return tasks


def tasks_for_agent(tasks: Dict[str, Task], agent_id: str) -> List[Task]:
    return [task for task in tasks.values() if task.owner_agent == agent_id]


def dependency_ready(task: Task, done_tasks: Iterable[str]) -> bool:
    done = set(done_tasks)
    return all(dep in done for dep in task.dependencies)


def pending_tasks(tasks: Dict[str, Task]) -> List[Task]:
    return [task for task in tasks.values() if task.status in {"pending", "running"}]


def completion_stats(tasks: Dict[str, Task]) -> Tuple[int, int, int]:
    total = len(tasks)
    done = sum(1 for task in tasks.values() if task.status == "done")
    on_time = sum(
        1
        for task in tasks.values()
        if task.status == "done"
        and task.completed_at_hour is not None
        and task.completed_at_hour <= task.deadline_hour
    )
    return total, done, on_time
