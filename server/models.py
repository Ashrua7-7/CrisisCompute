"""
Pydantic Action and Observation models for SatyaEnv (OpenEnv compliant).
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

class AgentAction(Action):
    """
    A single agent's proposed action for one step.
    Each agent picks one of: 'request_resource', 'release_resource', 'wait'.
    """
    agent_id: str = Field(..., description="Which agent is acting (data_loader / data_cleaner / ml_trainer)")
    action: str = Field(default="wait", description="One of: request_resource | release_resource | wait")
    task_id: Optional[str] = Field(default=None, description="Task ID to operate on (required for request/release)")
    reasoning: Optional[str] = Field(default=None, description="Optional reasoning text from the LLM agent")


class MultiAgentAction(Action):
    """
    Joint action from all agents in a single step.
    Maps agent_id -> per-agent action dict.
    """
    actions: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Map of agent_id to action dict {action, task_id, reasoning}"
    )


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------

class TaskInfo(Observation):
    """Info about a single task (embedded in the main observation)."""
    task_id: str
    owner_agent: str
    status: str           # pending | running | done
    priority: int
    deadline_hour: int
    remaining_duration_min: int
    cpu_required: int
    gpu_required: int
    memory_required: int


class AgentObservation(Observation):
    """Per-agent observation slice."""
    agent_id: str
    episode_index: int
    current_hour: int
    max_hours: int
    available_cpu: int
    available_gpu: int
    available_memory: int
    my_tasks: List[Dict[str, Any]] = Field(default_factory=list)
    other_agents_status: Dict[str, Any] = Field(default_factory=dict)
    recent_events: List[str] = Field(default_factory=list)


class MultiAgentObservation(Observation):
    """
    Joint observation returned after each step.
    Contains per-agent views + team-level metrics.
    """
    episode_index: int
    current_hour: int
    max_hours: int
    agent_observations: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-agent observation dict keyed by agent_id"
    )
    step_rewards: Dict[str, float] = Field(
        default_factory=dict,
        description="Per-agent reward earned this step"
    )
    team_reward: float = Field(default=0.0)
    metrics: Dict[str, Any] = Field(default_factory=dict)
    recent_events: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class SatyaState(State):
    """Internal environment state snapshot."""
    episode_index: int = Field(default=0)
    current_hour: int = Field(default=0)
    total_tasks: int = Field(default=0)
    completed_tasks: int = Field(default=0)
    on_time_tasks: int = Field(default=0)
    agent_rewards: Dict[str, float] = Field(default_factory=dict)
