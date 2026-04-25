"""
SatyaEnvironment — OpenEnv compliant server.

Wraps the existing RealEnvironment (multi-agent compute allocation) from
satya_env and exposes it via the OpenEnv Environment base class so it can
be served over HTTP / WebSocket and used in RL training loops.

Architecture
------------
  satya_env.RealEnvironment   ← unchanged domain logic
        ↓
  SatyaEnvironment            ← OpenEnv Environment subclass (this file)
        ↓
  HTTP server (app.py)        ← FastAPI app served on HF Spaces
"""

from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

# Make satya_env importable whether running from repo root or server/
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State, EnvironmentMetadata

from satya_env.env import RealEnvironment
from satya_env.tasks import completion_stats
from satya_env.models import Task

from server.models import (
    MultiAgentAction,
    MultiAgentObservation,
    SatyaState,
)


class SatyaEnvironment(Environment[MultiAgentAction, MultiAgentObservation, SatyaState]):
    """
    Multi-agent compute allocation environment following OpenEnv's Env API.

    Three LLM agents (data_loader, data_cleaner, ml_trainer) negotiate over
    shared CPU / GPU / memory resources to complete ML pipeline tasks within
    an 8-hour episode window.

    Action space  : MultiAgentAction — joint action dict, one entry per agent
    Observation   : MultiAgentObservation — joint obs + per-agent rewards
    State         : SatyaState — episode progress snapshot
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, config_dir: Optional[str] = None):
        super().__init__()
        self._config_dir = config_dir
        self._env = RealEnvironment(config_dir=config_dir)
        self._state = SatyaState(episode_id=str(uuid4()), step_count=0)
        self._last_rewards: Dict[str, float] = {}
        self._last_team_reward: float = 0.0
        self._last_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # OpenEnv required methods
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> MultiAgentObservation:
        """Reset the environment and return the initial joint observation."""
        self._reset_rubric()

        raw_obs = self._env.reset()

        self._state = SatyaState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            episode_index=self._env.episode_index,
            current_hour=self._env.episode.hour,
            total_tasks=len(self._env.tasks),
            completed_tasks=0,
            on_time_tasks=0,
            agent_rewards={a: 0.0 for a in self._env.agent_order},
        )
        self._last_rewards = {a: 0.0 for a in self._env.agent_order}
        self._last_team_reward = 0.0
        self._last_metrics = {}

        return self._build_observation(raw_obs, done=False)

    def _translate_action(self, agent_id: str, agent_action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translate high-level agent action to satya_env's internal format.

        High-level actions accepted:
          - "run_task" / "request_resource"  → maps to "run_task"
          - "wait" / "release_resource"      → maps to "wait"

        The validator requires: action, task_id, cores_needed, gpu_needed,
        memory_needed, estimated_duration_min — auto-filled from the task
        definition when task_id is provided.
        """
        raw_action = agent_action.get("action", "wait")
        action_name = "run_task" if raw_action in ("request_resource", "run_task") else "wait"
        task_id = agent_action.get("task_id")

        # Auto-fill resource requirements from task definition
        cores, gpu, memory, duration = 0, 0, 0, 0
        if task_id and task_id in self._env.tasks:
            task: Task = self._env.tasks[task_id]
            cores    = task.cores_needed
            gpu      = task.gpu_needed
            memory   = task.memory_needed
            duration = task.estimated_duration_min

        return {
            "action":                  action_name,
            "task_id":                 task_id,
            "cores_needed":            int(agent_action.get("cores_needed", cores) or cores),
            "gpu_needed":              int(agent_action.get("gpu_needed", gpu) or gpu),
            "memory_needed":           int(agent_action.get("memory_needed", memory) or memory),
            "estimated_duration_min":  int(agent_action.get("estimated_duration_min", duration) or duration),
        }

    def step(
        self,
        action: MultiAgentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> MultiAgentObservation:
        """
        Execute one time-step in the environment.

        Args:
            action: MultiAgentAction containing per-agent action dicts.
                    If action.actions is empty, all agents default to 'wait'.
        """
        # Translate high-level LLM-friendly actions → satya_env internal format
        raw_actions = {
            agent_id: self._translate_action(agent_id, agent_action)
            for agent_id, agent_action in action.actions.items()
        }

        raw_obs, reward_list, done, info = self._env.step(raw_actions)

        # Map list rewards back to agent names
        agent_rewards = {
            agent_id: reward_list[i]
            for i, agent_id in enumerate(self._env.agent_order)
        }

        self._last_rewards = agent_rewards
        self._last_team_reward = info.get("team_reward", 0.0)
        self._last_metrics = info.get("metrics", {})

        # Update state
        total, completed, on_time = completion_stats(self._env.tasks)
        self._state.step_count += 1
        self._state.current_hour = self._env.episode.hour
        self._state.completed_tasks = completed
        self._state.on_time_tasks = on_time
        for a, r in agent_rewards.items():
            self._state.agent_rewards[a] = self._state.agent_rewards.get(a, 0.0) + r

        obs = self._build_observation(raw_obs, done=done, events=info.get("events", []))
        obs.reward = sum(agent_rewards.values())  # scalar for training loop
        return obs

    @property
    def state(self) -> SatyaState:
        """Return current internal state snapshot."""
        return self._state

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="SatyaEnv",
            description=(
                "Multi-agent compute allocation environment. "
                "Three LLM agents negotiate over shared CPU/GPU/memory to "
                "complete ML pipeline tasks under deadline pressure."
            ),
            version="1.0.0",
            author="Gautam",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        raw_obs: Dict[str, Dict],
        done: bool,
        events: Optional[List[str]] = None,
    ) -> MultiAgentObservation:
        """Convert raw RealEnvironment observation to OpenEnv MultiAgentObservation."""
        pool = self._env.pool

        return MultiAgentObservation(
            done=done,
            reward=None,  # filled by step() after rewards are computed
            episode_index=self._env.episode_index,
            current_hour=self._env.episode.hour,
            max_hours=self._env.max_hours,
            agent_observations=raw_obs,
            step_rewards=self._last_rewards,
            team_reward=self._last_team_reward,
            metrics=self._last_metrics,
            recent_events=events or [],
        )
