from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class Task:
    task_id: str
    owner_agent: str
    title: str
    cores_needed: int
    gpu_needed: int
    memory_needed: int
    estimated_duration_min: int
    deadline_hour: int
    dependencies: List[str] = field(default_factory=list)
    status: str = "pending"
    remaining_duration_min: int = 0
    started_at_hour: Optional[int] = None
    completed_at_hour: Optional[int] = None

    def __post_init__(self) -> None:
        if self.remaining_duration_min <= 0:
            self.remaining_duration_min = self.estimated_duration_min


@dataclass
class ResourcePool:
    total_cpu: int
    total_gpu: int
    total_memory: int
    available_cpu: int
    available_gpu: int
    available_memory: int

    def reset(self) -> None:
        self.available_cpu = self.total_cpu
        self.available_gpu = self.total_gpu
        self.available_memory = self.total_memory


@dataclass
class Allocation:
    agent_id: str
    task_id: str
    cpu: int
    gpu: int
    memory: int


@dataclass
class AgentRuntimeState:
    agent_id: str
    running_task_id: Optional[str] = None
    waiting_minutes: int = 0
    completed_tasks: int = 0
    missed_deadlines: int = 0
    recent_events: List[str] = field(default_factory=list)


@dataclass
class EpisodeState:
    hour: int = 0
    max_hours: int = 8
    done_tasks: Set[str] = field(default_factory=set)
    recent_events: List[str] = field(default_factory=list)
