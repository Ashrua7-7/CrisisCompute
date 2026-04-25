from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

from .models import AgentRuntimeState, EpisodeState, ResourcePool, Task
from .negotiation import belief_accuracy_from_demands, build_intents, run_negotiation
from .observation import build_observation
from .reward import calculate_final_rewards, calculate_individual_rewards, calculate_team_reward
from .scheduler import resolve_and_allocate
from .tasks import completion_stats, load_tasks_from_json
from .validators import normalize_action, validate_action


class RealEnvironment:
    """Environment contract for multi-agent resource negotiation."""

    def __init__(self, config_dir: str | Path | None = None, seed: int | None = None) -> None:
        del seed
        root_dir = Path(__file__).resolve().parent
        self.config_dir = Path(config_dir) if config_dir else root_dir / "config"

        env_cfg = self._load_env_config(self.config_dir / "env.json")

        self.agent_order: List[str] = list(env_cfg.get("agent_order", ["data_loader", "data_cleaner", "ml_trainer"]))
        self.pool = ResourcePool(
            total_cpu=int(env_cfg.get("resources", {}).get("cpu", 16)),
            total_gpu=int(env_cfg.get("resources", {}).get("gpu", 1)),
            total_memory=int(env_cfg.get("resources", {}).get("memory", 32)),
            available_cpu=0,
            available_gpu=0,
            available_memory=0,
        )
        self.max_hours = int(env_cfg.get("episode_length_hours", 8))

        self.episode_index = 0
        self.episode = EpisodeState(hour=0, max_hours=self.max_hours)
        self.tasks: Dict[str, Task] = {}
        self.agent_states: Dict[str, AgentRuntimeState] = {}
        self.reputation: Dict[str, float] = {agent_id: 1.0 for agent_id in self.agent_order}
        self.beliefs: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.open_contracts: List[Dict] = []
        self.latest_negotiation = None

    def _load_env_config(self, path: Path) -> Dict:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _default_actions(self, actions: Dict[str, Dict] | None) -> Dict[str, Dict]:
        incoming = actions or {}
        prepared: Dict[str, Dict] = {}
        for agent_id in self.agent_order:
            action = normalize_action(incoming.get(agent_id, {"action": "wait", "task_id": None}))
            prepared[agent_id] = action
        return prepared

    def _refresh_running_markers(self) -> None:
        for state in self.agent_states.values():
            state.running_task_id = None
        for task in self.tasks.values():
            if task.status == "running":
                self.agent_states[task.owner_agent].running_task_id = task.task_id

    def _build_joint_observation(self) -> Dict[str, Dict]:
        return {
            agent_id: build_observation(
                requesting_agent=agent_id,
                episode_index=self.episode_index,
                episode=self.episode,
                pool=self.pool,
                tasks=self.tasks,
                agent_states=self.agent_states,
                reputation=self.reputation,
                beliefs=self.beliefs.get(agent_id, {}),
                negotiation_snapshot=self.latest_negotiation,
            )
            for agent_id in self.agent_order
        }

    def _initialize_beliefs(self) -> None:
        self.beliefs = {}
        for observer in self.agent_order:
            self.beliefs[observer] = {}
            for peer in self.agent_order:
                if observer == peer:
                    continue
                self.beliefs[observer][peer] = {
                    "predicted_cpu_demand": 2.0,
                    "predicted_gpu_demand": 0.3,
                    "predicted_memory_demand": 6.0,
                    "predicted_yield_probability": 0.4,
                    "reliability_estimate": 0.8,
                }

    def _update_social_state(self, prepared_actions: Dict[str, Dict], negotiation_snapshot) -> None:
        for agent_id in self.agent_order:
            action = prepared_actions[agent_id]
            for observer in self.agent_order:
                if observer == agent_id:
                    continue
                entry = self.beliefs[observer][agent_id]
                entry["predicted_cpu_demand"] = (entry["predicted_cpu_demand"] * 0.65) + (
                    action.get("cores_needed", 0) * 0.35
                )
                entry["predicted_gpu_demand"] = (entry["predicted_gpu_demand"] * 0.65) + (
                    action.get("gpu_needed", 0) * 0.35
                )
                entry["predicted_memory_demand"] = (entry["predicted_memory_demand"] * 0.65) + (
                    action.get("memory_needed", 0) * 0.35
                )
                entry["predicted_yield_probability"] = min(
                    1.0,
                    max(0.0, entry["predicted_yield_probability"] * 0.8 + (0.2 if action.get("action") == "wait" else 0.0)),
                )
                entry["reliability_estimate"] = min(
                    1.5,
                    max(0.0, entry["reliability_estimate"] + (0.03 if negotiation_snapshot.contracts_broken == 0 else -0.08)),
                )

        for agent_id in self.agent_order:
            if negotiation_snapshot.contracts_broken > 0:
                self.reputation[agent_id] = max(0.2, self.reputation[agent_id] - 0.05)
            if negotiation_snapshot.contracts_kept > 0:
                self.reputation[agent_id] = min(2.0, self.reputation[agent_id] + 0.03)

    def reset(self) -> Dict[str, Dict]:
        self.episode_index += 1
        self.episode = EpisodeState(hour=0, max_hours=self.max_hours)
        self.pool.reset()

        self.tasks = load_tasks_from_json(self.config_dir / "tasks.json")
        self.agent_states = {
            agent_id: AgentRuntimeState(agent_id=agent_id) for agent_id in self.agent_order
        }
        self.reputation = {agent_id: 1.0 for agent_id in self.agent_order}
        self.open_contracts = []
        self._initialize_beliefs()
        self.latest_negotiation = None
        self._refresh_running_markers()

        self.episode.recent_events.append("episode_reset")
        return self._build_joint_observation()

    def _apply_allocations(self, allocations) -> None:
        for allocation in allocations:
            task = self.tasks[allocation.task_id]
            if task.status == "pending":
                task.status = "running"
                task.started_at_hour = self.episode.hour
            task.remaining_duration_min -= 60
            self.agent_states[allocation.agent_id].running_task_id = task.task_id

            if task.remaining_duration_min <= 0:
                task.status = "done"
                task.completed_at_hour = self.episode.hour + 1
                self.episode.done_tasks.add(task.task_id)
                owner_state = self.agent_states[task.owner_agent]
                owner_state.completed_tasks += 1
                owner_state.running_task_id = None
                self.episode.recent_events.append(f"task_done:{task.task_id}")

    def _tick_waiting(self, allocations) -> None:
        allocated_agents = {allocation.agent_id for allocation in allocations}
        for agent_id, state in self.agent_states.items():
            if agent_id not in allocated_agents:
                state.waiting_minutes += 60

    def _mark_deadline_misses(self) -> None:
        for task in self.tasks.values():
            if task.status != "done" and self.episode.hour >= task.deadline_hour:
                self.agent_states[task.owner_agent].missed_deadlines += 1

    def _done(self) -> bool:
        total, done, _ = completion_stats(self.tasks)
        return done == total or self.episode.hour >= self.max_hours

    def step(self, actions: Dict[str, Dict] | None) -> Tuple[Dict[str, Dict], List[float], bool, Dict]:
        self.pool.reset()
        prepared_actions = self._default_actions(actions)

        validation_events: List[str] = []
        for agent_id, action in prepared_actions.items():
            is_valid, reason = validate_action(agent_id, action, self.tasks)
            if not is_valid:
                prepared_actions[agent_id] = normalize_action({"action": "wait", "task_id": None})
                validation_events.append(f"invalid_action:{agent_id}:{reason}")

        intents = build_intents(
            agent_order=self.agent_order,
            actions=prepared_actions,
            tasks=self.tasks,
            beliefs=self.beliefs,
            current_hour=self.episode.hour,
        )
        negotiated_actions, negotiation_snapshot = run_negotiation(
            intents=intents,
            tasks=self.tasks,
            done_tasks=self.episode.done_tasks,
            current_hour=self.episode.hour,
            total_cpu=self.pool.total_cpu,
            total_gpu=self.pool.total_gpu,
            total_memory=self.pool.total_memory,
            reputation=self.reputation,
            open_contracts=self.open_contracts,
        )
        negotiation_snapshot.belief_accuracy = belief_accuracy_from_demands(self.beliefs, intents)
        self.latest_negotiation = {
            "emergency_charter": negotiation_snapshot.emergency_charter,
            "crisis_agents": negotiation_snapshot.crisis_agents,
            "conflicts": negotiation_snapshot.conflicts,
            "coalitions": negotiation_snapshot.coalitions,
            "belief_accuracy": negotiation_snapshot.belief_accuracy,
            "fairness_score": negotiation_snapshot.fairness_score,
            "contracts_kept": negotiation_snapshot.contracts_kept,
            "contracts_broken": negotiation_snapshot.contracts_broken,
        }

        allocations, schedule_events = resolve_and_allocate(
            proposed_actions=negotiated_actions,
            tasks=self.tasks,
            pool=self.pool,
            done_tasks=self.episode.done_tasks,
            current_hour=self.episode.hour,
        )

        self._apply_allocations(allocations)
        self._tick_waiting(allocations)

        self.episode.hour += 1
        self._mark_deadline_misses()
        self._refresh_running_markers()
        self._update_social_state(negotiated_actions, negotiation_snapshot)

        self.episode.recent_events.extend(validation_events)
        if negotiation_snapshot.emergency_charter:
            self.episode.recent_events.append("emergency_charter_triggered")
        if negotiation_snapshot.deadlock:
            self.episode.recent_events.append("negotiation_deadlock")
        for coalition in negotiation_snapshot.coalitions:
            self.episode.recent_events.append(f"coalition:{coalition}")
        self.episode.recent_events.append(f"contracts_kept:{negotiation_snapshot.contracts_kept}")
        self.episode.recent_events.append(f"contracts_broken:{negotiation_snapshot.contracts_broken}")
        self.episode.recent_events.extend(schedule_events)

        individual = calculate_individual_rewards(self.agent_states, self.tasks)
        team = calculate_team_reward(
            tasks=self.tasks,
            max_hours=self.max_hours,
            current_hour=self.episode.hour,
            recent_events=schedule_events,
            negotiation_snapshot=self.latest_negotiation,
        )
        final_rewards = calculate_final_rewards(individual, team)

        observations = self._build_joint_observation()
        reward_list = [round(final_rewards[agent_id], 3) for agent_id in self.agent_order]
        done = self._done()

        total, completed, on_time = completion_stats(self.tasks)
        info = {
            "individual_rewards": individual,
            "team_reward": team,
            "negotiation": self.latest_negotiation,
            "events": self.episode.recent_events[-12:],
            "metrics": {
                "total_tasks": total,
                "completed_tasks": completed,
                "on_time_tasks": on_time,
                "completion_rate": (completed / total) if total else 0.0,
                "on_time_rate": (on_time / total) if total else 0.0,
                "avg_reputation": sum(self.reputation.values()) / len(self.reputation),
                "fairness_score": negotiation_snapshot.fairness_score,
            },
        }
        return observations, reward_list, done, info
