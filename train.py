# train.py
import requests as _req
_orig = _req.Session.request
def _debug(self, method, url, **kw):
    if "11434" in str(url):
        import json as _j
        body = kw.get("json") or kw.get("data")
        print(f"\n[DEBUG] {method} {url}")
        print(f"[DEBUG] payload: {_j.dumps(body, indent=2) if isinstance(body, dict) else body[:200]}")
    return _orig(self, method, url, **kw)
_req.Session.request = _debug

import json
import os
import sys
from copy import deepcopy
from datetime import datetime
from statistics import mean, pstdev
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.inference import LLMAgent
from src.rl_agent import (
    RLDataLoaderAgent,
    RLDataCleanerAgent,
    RLMLTrainerAgent
)
from src.visualize import ResultsVisualizer
from src.evaluate import MetricsCalculator, LearningAnalyzer

# ============================================================
# AUTO-DETECT ENVIRONMENT
# ============================================================

try:
    from satya_env.rl_environment import RLFriendlyEnvironment
    MultiAgentPipelineEnv = RLFriendlyEnvironment
    USE_REAL_ENV = True
    print("✅ RL-FRIENDLY ENVIRONMENT LOADED (With Learning Signals)")
except ImportError:
    try:
        from satya_env.env import RealEnvironment
        MultiAgentPipelineEnv = RealEnvironment
        USE_REAL_ENV = True
        print("✅ REAL ENVIRONMENT LOADED (Satya's Backend)")
    except ImportError as e:
        USE_REAL_ENV = False
        print(f"⚠️  REAL ENVIRONMENT NOT FOUND - Using MOCK\n   Error: {e}")

# Set default agent mode
TRAINING_AGENT_MODE = os.getenv("TRAINING_AGENT_MODE", "llm").strip().lower()
if TRAINING_AGENT_MODE not in {"llm", "rl", "hybrid"}:
    TRAINING_AGENT_MODE = "llm" if USE_REAL_ENV else "rl"


class AdaptiveCurriculum:
    """Simple performance-driven curriculum for self-improvement training."""

    def __init__(self):
        self.level = 0
        self.max_level = 4
        self.history: List[Dict] = []

    def config_for_level(self) -> Dict:
        configs = [
            {"negotiation_enabled": True, "crisis_mode_enabled": False},
            {"negotiation_enabled": True, "crisis_mode_enabled": False},
            {"negotiation_enabled": True, "crisis_mode_enabled": True},
            {"negotiation_enabled": True, "crisis_mode_enabled": True},
            {"negotiation_enabled": True, "crisis_mode_enabled": True},
        ]
        return configs[min(self.level, len(configs) - 1)]

    def update(self, phase: int, summary: Dict) -> bool:
        """Increase difficulty when phase performance crosses thresholds."""
        completion = float(summary.get("avg_completion_rate", 0.0))
        on_time = float(summary.get("avg_on_time_rate", 0.0))
        fairness = float(summary.get("avg_fairness_score", 0.0))
        # Some metric pipelines return percentages (0-100), normalize to [0,1].
        if completion > 1.0:
            completion = completion / 100.0
        if on_time > 1.0:
            on_time = on_time / 100.0
        promoted = False
        completion_gate = 0.35 + 0.08 * self.level
        on_time_gate = 0.30 + 0.07 * self.level
        if (
            self.level < self.max_level
            and completion >= completion_gate
            and on_time >= on_time_gate
            and fairness >= 0.40
        ):
            self.level += 1
            promoted = True
        self.history.append(
            {
                "phase": phase,
                "level": self.level,
                "promoted": promoted,
                "completion": completion,
                "on_time": on_time,
                "fairness": fairness,
            }
        )
        return promoted


class SelfPlayLeague:
    """Keeps rolling policy snapshots to avoid overfitting to one setting."""

    def __init__(self):
        self.snapshots: List[Dict] = []

    def _serialize_q_table(self, q_table: Dict) -> Dict[str, Dict[str, float]]:
        return {str(state): actions for state, actions in q_table.items()}

    def _serialize_agent_snapshot(self, agent) -> Dict:
        rl = agent.rl if hasattr(agent, "rl") else agent
        payload = {
            "name": getattr(rl, "name", str(agent)),
            "q_table": {},
            "epsilon": float(getattr(rl, "epsilon", 0.0)),
        }
        if hasattr(rl, "q_table"):
            payload["q_table"] = self._serialize_q_table(rl.q_table)
        return payload

    def record(self, phase: int, agents: List, summary: Dict):
        q_state_counts = {}
        agent_snapshots = {}
        for agent in agents:
            rl = agent.rl if hasattr(agent, "rl") else agent
            if hasattr(rl, "q_table"):
                q_state_counts[getattr(rl, "name", str(agent))] = len(rl.q_table)
            agent_snapshots[getattr(rl, "name", str(agent))] = self._serialize_agent_snapshot(agent)
        self.snapshots.append(
            {
                "phase": phase,
                "summary": summary,
                "q_state_counts": q_state_counts,
                "agent_snapshots": agent_snapshots,
            }
        )

    def recent(self, n: int = 3) -> List[Dict]:
        return self.snapshots[-n:]

    def sample_previous_snapshot(self, current_phase: int) -> Dict | None:
        historical = [s for s in self.snapshots if int(s.get("phase", 0)) < current_phase]
        if not historical:
            return None
        return deepcopy(historical[-1])


class ChallengeGenerator:
    """Generates progressively harder negotiation scenarios for Theme #4."""

    def __init__(self):
        self.templates = {
            0: [{"name": "stable_market", "crisis_mode_enabled": False}],
            1: [
                {"name": "light_gpu_outage", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 4},
                {"name": "stable_market", "crisis_mode_enabled": False},
            ],
            2: [
                {"name": "urgent_injection", "crisis_mode_enabled": True, "crisis_urgent_task_hour": 3},
                {"name": "gpu_outage_plus_urgent", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 3, "crisis_urgent_task_hour": 5},
            ],
            3: [
                {"name": "early_outage", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 2, "crisis_urgent_task_hour": 4},
                {"name": "late_urgent_spike", "crisis_mode_enabled": True, "crisis_urgent_task_hour": 2},
            ],
            4: [
                {"name": "compound_crisis_a", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 2, "crisis_urgent_task_hour": 3},
                {"name": "compound_crisis_b", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 3, "crisis_urgent_task_hour": 2},
            ],
        }

    def get_plan_for_level(self, level: int, episodes: int) -> List[Dict]:
        base = self.templates.get(level, self.templates[max(self.templates.keys())])
        return [deepcopy(base[i % len(base)]) for i in range(max(1, episodes))]


# ============================================================
# BUILD AGENTS
# ============================================================

def build_agents(agent_mode):
    """Build the three-agent lineup for the selected backend."""

    # ---------- HYBRID (LLM + RL) ----------
    if agent_mode == "hybrid":
        from src.hybrid_agent import HybridAgent

        llm_provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
        if llm_provider == "openrouter":
            model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
            base_url   = None
        elif llm_provider == "groq":
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            base_url   = None
        else:  # ollama
            model_name = os.getenv("OLLAMA_MODEL", "mistral")
            base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        llm_loader  = LLMAgent("data_loader",  {"cpu": 2, "memory": 4,  "gpu": 0}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)
        llm_cleaner = LLMAgent("data_cleaner", {"cpu": 4, "memory": 8,  "gpu": 0}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)
        llm_trainer = LLMAgent("ml_trainer",   {"cpu": 2, "memory": 16, "gpu": 1}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)

        rl_loader  = RLDataLoaderAgent()
        rl_cleaner = RLDataCleanerAgent()
        rl_trainer = RLMLTrainerAgent()

        loader  = HybridAgent(llm_loader,  rl_loader)
        cleaner = HybridAgent(llm_cleaner, rl_cleaner)
        trainer = HybridAgent(llm_trainer, rl_trainer)

        return {
            "data_loader":  loader,
            "data_cleaner": cleaner,
            "ml_trainer":   trainer,
        }, [loader, cleaner, trainer]

    # ---------- LLM only ----------
    if agent_mode == "llm":
        llm_provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

        if llm_provider == "openrouter":
            model_name = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")
            base_url   = None
        elif llm_provider == "groq":
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            base_url   = None
        else:  # ollama
            model_name = os.getenv("OLLAMA_MODEL", "mistral")
            base_url   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        loader  = LLMAgent("data_loader",  {"cpu": 2, "memory": 4,  "gpu": 0}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)
        cleaner = LLMAgent("data_cleaner", {"cpu": 4, "memory": 8,  "gpu": 0}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)
        trainer = LLMAgent("ml_trainer",   {"cpu": 2, "memory": 16, "gpu": 1}, llm_provider=llm_provider, model_name=model_name, base_url=base_url)

        return {
            "data_loader":  loader,
            "data_cleaner": cleaner,
            "ml_trainer":   trainer,
        }, [loader, cleaner, trainer]

    # ---------- RL only ----------
    loader  = RLDataLoaderAgent()
    cleaner = RLDataCleanerAgent()
    trainer = RLMLTrainerAgent()
    return {
        "data_loader":  loader,
        "data_cleaner": cleaner,
        "ml_trainer":   trainer,
    }, [loader, cleaner, trainer]


# ============================================================
# MOCK ENVIRONMENT (Fallback)
# ============================================================

class SimpleEnvironment:
    """Fallback mock environment"""

    def __init__(self, difficulty="easy"):
        self.difficulty      = difficulty
        self.current_episode = 0
        self.current_hour    = 0
        self.episode_rewards = 0

    def reset(self):
        self.current_episode += 1
        self.current_hour    = 0
        self.episode_rewards = 0

        obs = {
            "episode": self.current_episode,
            "hour": self.current_hour,
            "available_resources": {
                "gpu":       {"available": 1,  "total": 1},
                "cpu_cores": {"available": 16, "total": 16},
                "memory_gb": {"available": 32, "total": 32},
            },
            "other_agents_status": {
                "data_loader":  {"status": "idle"},
                "data_cleaner": {"status": "idle"},
                "ml_trainer":   {"status": "idle"},
            },
            "time_left_hours": 8,
            "my_tasks": {"pending": [], "running": [], "done": []},
        }
        return {"data_loader": obs, "data_cleaner": obs, "ml_trainer": obs}

    def step(self, actions):
        self.current_hour += 1

        base_reward    = 8.0
        episode_bonus  = min(self.current_episode / 30.0 * 5.0, 5.0)
        hour_efficiency = 1.0 + (self.current_hour / 8.0) * 0.5

        rewards = [
            base_reward * hour_efficiency + episode_bonus,
            base_reward * hour_efficiency + episode_bonus * 0.9,
            base_reward * hour_efficiency + episode_bonus * 0.8,
        ]
        self.episode_rewards += sum(rewards)
        done = self.current_hour >= 8

        obs = {
            "episode": self.current_episode,
            "hour": self.current_hour,
            "available_resources": {
                "gpu":       {"available": 1,  "total": 1},
                "cpu_cores": {"available": 16, "total": 16},
                "memory_gb": {"available": 32, "total": 32},
            },
            "other_agents_status": {
                "data_loader":  {"status": "done"    if self.current_hour >= 2 else "running"},
                "data_cleaner": {"status": "done"    if self.current_hour >= 4 else "idle"},
                "ml_trainer":   {"status": "done"    if self.current_hour >= 8 else "waiting"},
            },
            "time_left_hours": 8 - self.current_hour,
            "my_tasks": {"pending": [], "running": [], "done": []},
        }
        obs_per_agent = {"data_loader": obs, "data_cleaner": obs, "ml_trainer": obs}

        info = {
            "individual_rewards": {
                "data_loader":  rewards[0],
                "data_cleaner": rewards[1],
                "ml_trainer":   rewards[2],
            },
            "team_reward": sum(rewards),
            "events": [],
            "metrics": {
                "total_tasks":      15,
                "completed_tasks":  min(int(self.episode_rewards / 30.0), 15),
                "on_time_tasks":    min(int(self.episode_rewards / 35.0), 15),
                "completion_rate":  min(self.episode_rewards / 150.0, 1.0),
                "on_time_rate":     min(self.episode_rewards / 175.0, 1.0),
            },
        }
        return obs_per_agent, rewards, done, info


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_agents(num_episodes=30):
    def _resolve_flag(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    negotiation_enabled = _resolve_flag("NEGOTIATION_ENABLED", True)
    crisis_mode_enabled = _resolve_flag("CRISIS_MODE_ENABLED", False)
    multi_seed_raw = os.getenv("MULTI_SEED", "")
    seed_values = [int(x.strip()) for x in multi_seed_raw.split(",") if x.strip()]
    if not seed_values:
        seed_values = [42]

    def _prepare_env_for_mode(env_obj, negotiation_flag: bool, crisis_flag: bool, scenario: Dict | None = None):
        if hasattr(env_obj, "negotiation_enabled"):
            env_obj.negotiation_enabled = negotiation_flag
        if hasattr(env_obj, "crisis_mode_enabled"):
            env_obj.crisis_mode_enabled = crisis_flag
        if scenario:
            if hasattr(env_obj, "crisis_gpu_outage_hour"):
                env_obj.crisis_gpu_outage_hour = scenario.get("crisis_gpu_outage_hour")
            if hasattr(env_obj, "crisis_urgent_task_hour"):
                env_obj.crisis_urgent_task_hour = scenario.get("crisis_urgent_task_hour")

    def _safe_metrics_for_episode(episode_data):
        metrics = episode_data.get("metrics", {})
        if metrics:
            return metrics
        return MetricsCalculator.calculate_metrics(episode_data)

    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _run_single_session(
        session_episodes: int,
        seed: int,
        negotiation_flag: bool,
        crisis_flag: bool,
        scenario_plan: List[Dict] | None = None,
        load_q_tables: bool = True,
        league_opponent_snapshot: Dict | None = None,
        learner_agent_ids: List[str] | None = None,
        track_label: str = "train",
    ):
        env = MultiAgentPipelineEnv(config_dir=None, seed=seed) if USE_REAL_ENV else SimpleEnvironment()
        agents_dict, agents = build_agents(TRAINING_AGENT_MODE)
        learner_agent_ids = learner_agent_ids or ["data_loader", "data_cleaner", "ml_trainer"]

        def _get_rl(agent):
            return agent.rl if hasattr(agent, "rl") else agent

        def _apply_plateau_mitigation(recent_rewards: List[float], min_history: int = 10) -> Dict[str, Dict[str, float]]:
            """Bump exploration slightly when reward curve plateaus."""
            if len(recent_rewards) < min_history:
                return {}
            window = max(3, min(5, len(recent_rewards) // 2))
            if not LearningAnalyzer.detect_plateauing(recent_rewards, window=window):
                return {}

            adjustments: Dict[str, Dict[str, float]] = {}
            for agent_id, agent in agents_dict.items():
                if agent_id not in learner_agent_ids:
                    continue
                rl = _get_rl(agent)
                if not hasattr(rl, "epsilon"):
                    continue
                old_eps = float(getattr(rl, "epsilon", 0.0))
                new_eps = min(0.75, max(0.10, old_eps) + 0.08)
                rl.epsilon = new_eps
                adjustments[agent_id] = {
                    "old_epsilon": old_eps,
                    "new_epsilon": new_eps,
                }
            return adjustments

        if load_q_tables and any(hasattr(_get_rl(a), "load_q_table") for a in agents):
            os.makedirs("q_tables", exist_ok=True)
            for agent in agents:
                rl = _get_rl(agent)
                if hasattr(rl, "load_q_table"):
                    name = rl.name.replace("rl_", "")
                    rl.load_q_table(f"q_tables/{name}_q_table.json")

        def _policy_action_from_snapshot(agent_obj, observation, snapshot_state: Dict):
            rl_obj = _get_rl(agent_obj)
            q_table_json = snapshot_state.get("q_table", {})
            if not q_table_json:
                return None
            state = rl_obj.discretize_state(observation)
            state_key = str(state)
            state_actions = q_table_json.get(state_key, {})
            if not state_actions:
                return None
            best_action_type = max(state_actions.items(), key=lambda kv: kv[1])[0]
            return rl_obj.propose_action(observation, strategy=best_action_type)

        snapshot_map = {}
        if league_opponent_snapshot:
            for snapshot_agent in league_opponent_snapshot.get("agent_snapshots", {}).values():
                snapshot_name = str(snapshot_agent.get("name", ""))
                plain_name = snapshot_name.replace("rl_", "")
                snapshot_map[plain_name] = snapshot_agent

        all_results = []
        episode_metrics = []
        negotiation_trace = []
        recent_reward_window: List[float] = []
        last_plateau_episode = 0

        for episode in range(1, session_episodes + 1):
            scenario = scenario_plan[(episode - 1) % len(scenario_plan)] if scenario_plan else {}
            scenario_crisis_flag = bool(scenario.get("crisis_mode_enabled", crisis_flag))
            _prepare_env_for_mode(env, negotiation_flag, scenario_crisis_flag, scenario=scenario)
            obs_per_agent = env.reset()
            if not isinstance(obs_per_agent, dict) or "data_loader" not in obs_per_agent:
                obs_per_agent = {"data_loader": obs_per_agent, "data_cleaner": obs_per_agent, "ml_trainer": obs_per_agent}

            for agent in agents:
                if hasattr(agent, "reset_for_episode"):
                    agent.reset_for_episode()

            episode_data = {
                "episode": episode,
                "scenario": scenario.get("name", "default"),
                "track": track_label,
                "total_reward": 0,
                "completed_tasks": 0,
                "on_time_tasks": 0,
                "conflicts": 0,
                "agreements": 3,
                "agent_actions": [],
                "steps": [],
                "conflict_count": 0,
                "coalitions_formed": 0,
                "contracts_kept": 0,
                "contracts_broken": 0,
                "avg_fairness_score": 0.0,
                "avg_belief_accuracy": 0.0,
                "deadline_misses": 0,
                "plateau_mitigation_applied": False,
                "plateau_mitigation": {},
                "crisis_renegotiations": 0,
            }

            total_episode_reward = 0.0
            info = {}
            fairness_values = []
            belief_values = []
            conflict_count = 0
            coalitions_formed = 0
            contracts_kept = 0
            contracts_broken = 0
            deadline_misses = 0
            step_count = 0

            for hour in range(1, 9):
                actions_dict = {}
                for agent_id in ["data_loader", "data_cleaner", "ml_trainer"]:
                    agent = agents_dict[agent_id]
                    agent_obs = obs_per_agent.get(agent_id, obs_per_agent)
                    if isinstance(agent_obs, dict):
                        agent_obs["hour"] = hour
                    action = None
                    if agent_id not in learner_agent_ids and snapshot_map:
                        action = _policy_action_from_snapshot(agent, agent_obs, snapshot_map.get(agent_id, {}))
                    if action is None:
                        action = agent.act(agent_obs) if TRAINING_AGENT_MODE == "hybrid" else agent.propose_action(agent_obs)
                    actions_dict[agent_id] = action

                episode_data["agent_actions"].append({"hour": hour, "actions": actions_dict})
                result = env.step(actions_dict)
                if len(result) == 4:
                    next_obs_per_agent, rewards, done, info = result
                else:
                    next_obs_per_agent, rewards, done = result
                    info = {}

                if not isinstance(next_obs_per_agent, dict) or "data_loader" not in next_obs_per_agent:
                    next_obs_per_agent = {"data_loader": next_obs_per_agent, "data_cleaner": next_obs_per_agent, "ml_trainer": next_obs_per_agent}

                for i, agent_id in enumerate(["data_loader", "data_cleaner", "ml_trainer"]):
                    agent = agents_dict[agent_id]
                    next_agent_obs = next_obs_per_agent.get(agent_id, next_obs_per_agent)
                    if TRAINING_AGENT_MODE == "hybrid":
                        if agent_id in learner_agent_ids:
                            agent.learn(obs_per_agent.get(agent_id, obs_per_agent), actions_dict[agent_id], rewards[i], next_agent_obs)
                    else:
                        if agent_id in learner_agent_ids:
                            agent.receive_reward(rewards[i], next_agent_obs)
                    total_episode_reward += rewards[i]

                obs_per_agent = next_obs_per_agent
                step_count += 1

                step_metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
                step_trace = info.get("negotiation_trace_step") if isinstance(info, dict) else None
                if step_trace:
                    negotiation_trace.append({"episode": episode, **deepcopy(step_trace)})
                    if step_trace.get("triggered_renegotiation"):
                        episode_data["crisis_renegotiations"] += 1
                if step_metrics:
                    conflict_count += int(step_metrics.get("conflict_count", 0))
                    coalitions_formed += int(step_metrics.get("coalitions_formed", 0))
                    contracts_kept += int(step_metrics.get("contracts_kept", 0))
                    contracts_broken += int(step_metrics.get("contracts_broken", 0))
                    deadline_misses = max(deadline_misses, int(step_metrics.get("deadline_misses", 0)))
                    fairness_values.append(float(step_metrics.get("fairness_score", 0.0)))
                    belief_values.append(float(step_metrics.get("belief_accuracy", 0.0)))

                episode_data["steps"].append(
                    {
                        "hour": hour,
                        "actions": actions_dict,
                        "step_rewards": rewards,
                        "conflict_count": int(step_metrics.get("conflict_count", 0)),
                        "coalitions_formed": int(step_metrics.get("coalitions_formed", 0)),
                        "contracts_kept": int(step_metrics.get("contracts_kept", 0)),
                        "contracts_broken": int(step_metrics.get("contracts_broken", 0)),
                        "fairness_score": float(step_metrics.get("fairness_score", 0.0)),
                        "belief_accuracy": float(step_metrics.get("belief_accuracy", 0.0)),
                        "deadline_misses": int(step_metrics.get("deadline_misses", 0)),
                    }
                )

                if done:
                    break

            episode_data["total_reward"] = float(total_episode_reward)
            if isinstance(info, dict):
                m = info.get("metrics", {})
                episode_data["completed_tasks"] = int(m.get("completed_tasks", min(int(total_episode_reward / 10.0), 15)))
                episode_data["on_time_tasks"] = int(m.get("on_time_tasks", min(int(total_episode_reward / 12.0), episode_data["completed_tasks"])))
                episode_data["conflicts"] = sum(1 for e in info.get("events", []) if "conflict" in e)
                episode_data["agreements"] = sum(1 for e in info.get("events", []) if "allocated" in e)

            episode_data["conflict_count"] = int(conflict_count)
            episode_data["coalitions_formed"] = int(coalitions_formed)
            episode_data["contracts_kept"] = int(contracts_kept)
            episode_data["contracts_broken"] = int(contracts_broken)
            episode_data["avg_fairness_score"] = float(mean(fairness_values) if fairness_values else 0.0)
            episode_data["avg_belief_accuracy"] = float(mean(belief_values) if belief_values else 0.0)
            episode_data["deadline_misses"] = int(deadline_misses)

            recent_reward_window.append(float(total_episode_reward))
            if len(recent_reward_window) > 12:
                recent_reward_window = recent_reward_window[-12:]
            if (
                TRAINING_AGENT_MODE in {"rl", "hybrid"}
                and episode >= 10
                and (episode - last_plateau_episode) >= 4
                and track_label.startswith("curriculum_train")
            ):
                mitigation = _apply_plateau_mitigation(recent_reward_window)
                if mitigation:
                    episode_data["plateau_mitigation_applied"] = True
                    episode_data["plateau_mitigation"] = mitigation
                    last_plateau_episode = episode

            episode_data["metrics"] = MetricsCalculator.calculate_metrics(episode_data)
            episode_data["metrics"]["total_reward"] = episode_data["total_reward"]
            episode_data["metrics"]["fairness"] = episode_data["avg_fairness_score"]
            episode_data["metrics"]["belief_accuracy"] = episode_data["avg_belief_accuracy"]
            episode_data["metrics"]["negotiation_health"] = max(
                0.0,
                1.0 - (episode_data["contracts_broken"] + episode_data["conflict_count"]) / max(step_count * 3, 1),
            )
            all_results.append(episode_data)

            episode_metrics.append(
                {
                    "episode": episode,
                    "completion_rate": episode_data["metrics"]["completion_rate"],
                    "on_time_rate": episode_data["metrics"]["on_time_rate"],
                    "fairness": episode_data["avg_fairness_score"],
                    "belief_accuracy": episode_data["avg_belief_accuracy"],
                    "conflict_count": episode_data["conflict_count"],
                    "coalitions_formed": episode_data["coalitions_formed"],
                    "contracts_kept": episode_data["contracts_kept"],
                    "contracts_broken": episode_data["contracts_broken"],
                    "deadline_misses": episode_data["deadline_misses"],
                    "negotiation_health": episode_data["metrics"]["negotiation_health"],
                    "crisis_renegotiations": episode_data["crisis_renegotiations"],
                    "plateau_mitigation_applied": bool(episode_data["plateau_mitigation_applied"]),
                }
            )

            for agent in agents:
                rl = _get_rl(agent)
                plain_name = getattr(rl, "name", "").replace("rl_", "")
                if hasattr(rl, "learn_from_episode") and plain_name in learner_agent_ids:
                    rl.learn_from_episode()

        return all_results, episode_metrics, negotiation_trace, agents

    def _summarize_results(results):
        if not results:
            return {
                "avg_total_reward": 0.0,
                "avg_completion_rate": 0.0,
                "avg_on_time_rate": 0.0,
                "avg_fairness_score": 0.0,
                "avg_belief_accuracy": 0.0,
            }
        return {
            "avg_total_reward": mean([r["total_reward"] for r in results]),
            "avg_completion_rate": mean([_safe_metrics_for_episode(r)["completion_rate"] for r in results]),
            "avg_on_time_rate": mean([_safe_metrics_for_episode(r)["on_time_rate"] for r in results]),
            "avg_fairness_score": mean([r.get("avg_fairness_score", 0.0) for r in results]),
            "avg_belief_accuracy": mean([r.get("avg_belief_accuracy", 0.0) for r in results]),
        }

    def _rolling_window_improvement(results, window=5):
        if not results:
            return {"first_window_mean": 0.0, "last_window_mean": 0.0, "improvement_percent": 0.0}
        rewards = [r.get("total_reward", 0.0) for r in results]
        w = max(1, min(window, len(rewards)))
        first_mean = mean(rewards[:w])
        last_mean = mean(rewards[-w:])
        improvement = ((last_mean - first_mean) / abs(first_mean) * 100.0) if first_mean != 0 else 0.0
        return {"first_window_mean": first_mean, "last_window_mean": last_mean, "improvement_percent": improvement}

    def _evaluate_holdout(seed, episodes=8):
        holdout_plan = [
            {"name": "holdout_compound_1", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 2, "crisis_urgent_task_hour": 3},
            {"name": "holdout_compound_2", "crisis_mode_enabled": True, "crisis_gpu_outage_hour": 3, "crisis_urgent_task_hour": 2},
        ]
        trained_results, _, _, _ = _run_single_session(
            session_episodes=episodes,
            seed=seed,
            negotiation_flag=True,
            crisis_flag=True,
            scenario_plan=holdout_plan,
            load_q_tables=True,
        )
        fresh_results, _, _, _ = _run_single_session(
            session_episodes=episodes,
            seed=seed,
            negotiation_flag=True,
            crisis_flag=True,
            scenario_plan=holdout_plan,
            load_q_tables=False,
        )
        trained_summary = _summarize_results(trained_results)
        fresh_summary = _summarize_results(fresh_results)
        return {
            "episodes": episodes,
            "seed": seed,
            "holdout_plan": holdout_plan,
            "trained_summary": trained_summary,
            "fresh_summary": fresh_summary,
            "delta": {k: trained_summary.get(k, 0.0) - fresh_summary.get(k, 0.0) for k in trained_summary.keys()},
        }

    def _evaluate_fixed_scenario(seed, episodes=6):
        fixed_plan = [
            {"name": "fixed_eval_stable", "crisis_mode_enabled": False},
        ]
        trained_results, _, _, _ = _run_single_session(
            session_episodes=episodes,
            seed=seed,
            negotiation_flag=True,
            crisis_flag=False,
            scenario_plan=fixed_plan,
            load_q_tables=True,
        )
        fresh_results, _, _, _ = _run_single_session(
            session_episodes=episodes,
            seed=seed,
            negotiation_flag=True,
            crisis_flag=False,
            scenario_plan=fixed_plan,
            load_q_tables=False,
        )
        trained_summary = _summarize_results(trained_results)
        fresh_summary = _summarize_results(fresh_results)
        return {
            "episodes": episodes,
            "seed": seed,
            "fixed_plan": fixed_plan,
            "trained_summary": trained_summary,
            "fresh_summary": fresh_summary,
            "delta": {k: trained_summary.get(k, 0.0) - fresh_summary.get(k, 0.0) for k in trained_summary.keys()},
        }

    print("\n" + "="*70)
    print("🚀 MULTI-AGENT TRAINING SYSTEM")
    print("="*70)
    print(f"Episodes:    {num_episodes}")
    print(f"Environment: {'REAL (Satya)' if USE_REAL_ENV else 'MOCK (Testing)'}")
    print(f"Agent Mode:  {TRAINING_AGENT_MODE.upper()}")
    print(f"Start Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    print(f"Negotiation Enabled: {negotiation_enabled}")
    print(f"Crisis Mode:         {crisis_mode_enabled}")
    print(f"Seeds:               {seed_values}")
    print("="*70 + "\n")

    all_results = []
    episode_metrics = []
    negotiation_trace = []
    agents = []
    curriculum = AdaptiveCurriculum()
    league = SelfPlayLeague()
    challenge_generator = ChallengeGenerator()
    curriculum_phase_episodes = max(1, int(os.getenv("CURRICULUM_PHASE_EPISODES", "5")))
    total_phases = max(1, int((num_episodes + curriculum_phase_episodes - 1) / curriculum_phase_episodes))
    self_improvement_enabled = _resolve_flag("SELF_IMPROVEMENT_ENABLED", True)

    if self_improvement_enabled and TRAINING_AGENT_MODE in {"rl", "hybrid"}:
        print("🧠 Theme #4 loop: adaptive curriculum + self-play snapshots enabled")
        running_episode_idx = 0
        league_duel_log = []
        for phase in range(1, total_phases + 1):
            phase_cfg = curriculum.config_for_level()
            phase_episodes = min(curriculum_phase_episodes, num_episodes - running_episode_idx)
            if phase_episodes <= 0:
                break
            phase_results, phase_metrics, phase_trace, phase_agents = _run_single_session(
                session_episodes=phase_episodes,
                seed=seed_values[0],
                negotiation_flag=phase_cfg["negotiation_enabled"],
                crisis_flag=phase_cfg["crisis_mode_enabled"],
                scenario_plan=challenge_generator.get_plan_for_level(curriculum.level, phase_episodes),
                track_label="curriculum_train",
            )
            for episode_data in phase_results:
                running_episode_idx += 1
                episode_data["episode"] = running_episode_idx
                episode_data["curriculum_level"] = curriculum.level
                episode_data["curriculum_phase"] = phase
            for episode_row in phase_metrics:
                episode_row["episode"] = (episode_row["episode"] - 1) + (running_episode_idx - len(phase_results) + 1)
                episode_row["curriculum_level"] = curriculum.level
                episode_row["curriculum_phase"] = phase
            all_results.extend(phase_results)
            episode_metrics.extend(phase_metrics)
            negotiation_trace.extend(phase_trace)
            agents = phase_agents

            summary = _summarize_results(phase_results)
            league.record(phase=phase, agents=phase_agents, summary=summary)
            promoted = curriculum.update(phase=phase, summary=summary)
            print(
                f"Phase {phase}/{total_phases} │ lvl={curriculum.level} │ "
                f"episodes={phase_episodes} │ completion={summary['avg_completion_rate']:.2f} │ "
                f"fairness={summary['avg_fairness_score']:.2f} │ promoted={promoted}"
            )

            opponent_snapshot = league.sample_previous_snapshot(phase)
            if opponent_snapshot:
                duel_plan = challenge_generator.get_plan_for_level(min(curriculum.level + 1, curriculum.max_level), 1)
                for learner_id in ["data_loader", "data_cleaner", "ml_trainer"]:
                    duel_results, duel_metrics, _, _ = _run_single_session(
                        session_episodes=1,
                        seed=seed_values[0] + phase,
                        negotiation_flag=True,
                        crisis_flag=True,
                        scenario_plan=duel_plan,
                        league_opponent_snapshot=opponent_snapshot,
                        learner_agent_ids=[learner_id],
                        track_label=f"league_duel_{learner_id}",
                    )
                    if duel_results:
                        duel = duel_results[0]
                        running_episode_idx += 1
                        duel["episode"] = running_episode_idx
                        duel["curriculum_level"] = curriculum.level
                        duel["curriculum_phase"] = phase
                        duel["league_duel"] = True
                        duel["learner_agent"] = learner_id
                        all_results.append(duel)
                        if duel_metrics:
                            duel_row = duel_metrics[0]
                            duel_row["episode"] = running_episode_idx
                            duel_row["curriculum_level"] = curriculum.level
                            duel_row["curriculum_phase"] = phase
                            duel_row["league_duel"] = True
                            duel_row["learner_agent"] = learner_id
                            episode_metrics.append(duel_row)
                        league_duel_log.append(
                            {
                                "phase": phase,
                                "learner_agent": learner_id,
                                "opponent_phase": opponent_snapshot.get("phase"),
                                "reward": duel.get("total_reward", 0.0),
                                "completion_rate": duel.get("metrics", {}).get("completion_rate", 0.0),
                                "scenario": duel.get("scenario", "unknown"),
                            }
                        )
    else:
        all_results, episode_metrics, negotiation_trace, agents = _run_single_session(
            session_episodes=num_episodes,
            seed=seed_values[0],
            negotiation_flag=negotiation_enabled,
            crisis_flag=crisis_mode_enabled,
            scenario_plan=[{"name": "fixed_default", "crisis_mode_enabled": crisis_mode_enabled}],
        )
        league_duel_log = []

    for episode_data in all_results:
        epsilon = 0.0
        if agents:
            first_rl = agents[0].rl if hasattr(agents[0], "rl") else agents[0]
            epsilon = getattr(first_rl, "epsilon", 0.0)
        if episode_data["episode"] % 5 == 0 or episode_data["episode"] == 1:
            completion_rate = MetricsCalculator.calculate_completion_rate(episode_data)
            print(
                f"Episode {episode_data['episode']:2d} │ Reward: {episode_data['total_reward']:7.1f} │ "
                f"Completion: {completion_rate:5.1f}% │ ε: {epsilon:.3f}"
            )

    # ========== POST-TRAINING ==========
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)

    os.makedirs("results", exist_ok=True)
    _write_json("results/training_results.json", all_results)
    _write_json("results/episode_metrics.json", episode_metrics)
    _write_json("results/negotiation_trace.json", negotiation_trace)
    print("✅ Saved: results/training_results.json")
    print("✅ Saved: results/episode_metrics.json")
    print("✅ Saved: results/negotiation_trace.json")

    baseline_results, _, _, _ = _run_single_session(
        session_episodes=num_episodes,
        seed=seed_values[0],
        negotiation_flag=False,
        crisis_flag=False,
        scenario_plan=[{"name": "baseline_no_negotiation", "crisis_mode_enabled": False}],
        load_q_tables=False,
    )
    negotiated_summary = _summarize_results(all_results)
    baseline_summary = _summarize_results(baseline_results)
    baseline_comparison = {
        "config": {"seed": seed_values[0], "negotiation_enabled": negotiation_enabled},
        "negotiation_enabled_run": negotiated_summary,
        "baseline_run_negotiation_disabled": baseline_summary,
        "delta": {k: negotiated_summary.get(k, 0.0) - baseline_summary.get(k, 0.0) for k in negotiated_summary.keys()},
    }
    _write_json("results/baseline_comparison.json", baseline_comparison)
    print("✅ Saved: results/baseline_comparison.json")

    if self_improvement_enabled and TRAINING_AGENT_MODE in {"rl", "hybrid"}:
        level_progression = {}
        for row in all_results:
            level = int(row.get("curriculum_level", 0))
            level_progression.setdefault(level, []).append(float(row.get("total_reward", 0.0)))
        level_progression_summary = {
            f"level_{level}": {
                "episodes": len(scores),
                "mean_reward": mean(scores) if scores else 0.0,
            }
            for level, scores in level_progression.items()
        }
        selfplay_report = {
            "enabled": True,
            "curriculum_history": curriculum.history,
            "league_recent_snapshots": league.recent(5),
            "league_duels": league_duel_log,
            "phases": total_phases,
            "phase_episodes": curriculum_phase_episodes,
            "challenge_templates": challenge_generator.templates,
            "level_progression_summary": level_progression_summary,
        }
        _write_json("results/selfplay_report.json", selfplay_report)
        print("✅ Saved: results/selfplay_report.json")

        holdout_report = _evaluate_holdout(seed=seed_values[0], episodes=max(6, min(12, num_episodes)))
        _write_json("results/holdout_evaluation.json", holdout_report)
        print("✅ Saved: results/holdout_evaluation.json")

        fixed_eval_report = _evaluate_fixed_scenario(seed=seed_values[0], episodes=max(6, min(12, num_episodes)))
        _write_json("results/fixed_evaluation.json", fixed_eval_report)
        print("✅ Saved: results/fixed_evaluation.json")

        theme4_summary = {
            "enabled": True,
            "curriculum_history": curriculum.history,
            "league_duels": league_duel_log,
            "phase_episodes": curriculum_phase_episodes,
            "total_phases": total_phases,
            "challenge_templates": challenge_generator.templates,
            "level_progression_summary": level_progression_summary,
            "baseline_delta": baseline_comparison.get("delta", {}),
            "fixed_eval_delta": fixed_eval_report.get("delta", {}),
            "holdout_delta": holdout_report.get("delta", {}),
            "holdout_trained_summary": holdout_report.get("trained_summary", {}),
            "holdout_fresh_summary": holdout_report.get("fresh_summary", {}),
        }
        _write_json("results/theme4_summary.json", theme4_summary)
        print("✅ Saved: results/theme4_summary.json")

    seed_summaries = []
    for seed in seed_values:
        run_results, _, _, _ = _run_single_session(
            session_episodes=num_episodes,
            seed=seed,
            negotiation_flag=negotiation_enabled,
            crisis_flag=crisis_mode_enabled,
        )
        rewards = [r["total_reward"] for r in run_results]
        completions = [_safe_metrics_for_episode(r)["completion_rate"] for r in run_results]
        on_time = [_safe_metrics_for_episode(r)["on_time_rate"] for r in run_results]
        seed_summaries.append(
            {
                "seed": seed,
                "mean_reward": mean(rewards) if rewards else 0.0,
                "mean_completion_rate": mean(completions) if completions else 0.0,
                "mean_on_time_rate": mean(on_time) if on_time else 0.0,
            }
        )
    judge_summary = {
        "config": {
            "seeds": seed_values,
            "negotiation_enabled": negotiation_enabled,
            "crisis_mode_enabled": crisis_mode_enabled,
            "episodes_per_seed": num_episodes,
        },
        "per_seed": seed_summaries,
        "aggregate": {
            "reward_mean": mean([s["mean_reward"] for s in seed_summaries]) if seed_summaries else 0.0,
            "reward_std": pstdev([s["mean_reward"] for s in seed_summaries]) if len(seed_summaries) > 1 else 0.0,
            "completion_mean": mean([s["mean_completion_rate"] for s in seed_summaries]) if seed_summaries else 0.0,
            "completion_std": pstdev([s["mean_completion_rate"] for s in seed_summaries]) if len(seed_summaries) > 1 else 0.0,
            "on_time_mean": mean([s["mean_on_time_rate"] for s in seed_summaries]) if seed_summaries else 0.0,
            "on_time_std": pstdev([s["mean_on_time_rate"] for s in seed_summaries]) if len(seed_summaries) > 1 else 0.0,
        },
    }
    _write_json("results/judge_summary.json", judge_summary)
    print("✅ Saved: results/judge_summary.json")

    print("\n📊 Generating visualizations...")
    visualizer = ResultsVisualizer(all_results)
    visualizer.plot_reward_curve("results/reward_curve.png")
    visualizer.plot_metrics_dashboard("results/metrics_dashboard.png")
    visualizer.extract_dialogues("results/dialogue_samples.txt")
    visualizer.generate_summary_stats("results/summary_statistics.txt")
    print("✅ Saved plots and stats to results/")

    # RL learning analysis
    print("\n" + "="*70)
    print("🧠 AGENT LEARNING ANALYSIS")
    print("="*70)
    def _get_rl(agent):
        return agent.rl if hasattr(agent, "rl") else agent
    rl_agents = [_get_rl(a) for a in agents if hasattr(_get_rl(a), "q_table")]
    if rl_agents:
        for rl in rl_agents:
            r = rl.episode_rewards
            window = max(1, min(5, len(r)))
            avg_first = sum(r[:window]) / window if r else 0
            avg_last = sum(r[-window:]) / window if r else 0
            gain      = avg_last - avg_first
            print(f"\n👤 {rl.name.upper()}")
            print(f"   Q-table states       : {len(rl.q_table)}")
            print(f"   Avg reward (first {window}) : {avg_first:.1f}")
            print(f"   Avg reward (last {window})  : {avg_last:.1f}")
            print(f"   Learning gain        : {gain:+.1f}")
            print(f"   {'✅ IMPROVING' if gain > 0 else '⚠️  Needs more episodes'}")

        print("\n💾 Saving Q-tables...")
        os.makedirs("q_tables", exist_ok=True)
        for rl in rl_agents:
            name = rl.name.replace("rl_", "")
            rl.save_q_table(f"q_tables/{name}_q_table.json")
            print(f"   ✅ Saved: q_tables/{name}_q_table.json")
    else:
        print("LLM-only mode: learning captured in conversation history.")

    # Final summary
    print("\n" + "="*70)
    print("📈 FINAL SUMMARY")
    print("="*70)
    if all_results:
        roll = _rolling_window_improvement(all_results, window=5)
        print(f"Reward mean (first window) : {roll['first_window_mean']:.1f}")
        print(f"Reward mean (last window)  : {roll['last_window_mean']:.1f}")
        print(f"Window improvement         : {roll['improvement_percent']:+.1f}%")
        level_scores = {}
        for row in all_results:
            level = int(row.get("curriculum_level", 0))
            level_scores.setdefault(level, []).append(float(row.get("total_reward", 0.0)))
        if level_scores:
            print("Curriculum level rewards:")
            for level in sorted(level_scores):
                print(f"  L{level}: {mean(level_scores[level]):.1f}")
        if self_improvement_enabled and TRAINING_AGENT_MODE in {"rl", "hybrid"}:
            fixed_eval_report = locals().get("fixed_eval_report")
            if fixed_eval_report:
                fixed_delta = fixed_eval_report.get("delta", {})
                holdout_delta = holdout_report.get("delta", {}) if 'holdout_report' in locals() else {}
                print("Fixed evaluation delta (trained - fresh):")
                print(f"  Reward         : {fixed_delta.get('avg_total_reward', 0.0):+.1f}")
                print(f"  Completion rate: {fixed_delta.get('avg_completion_rate', 0.0):+.3f}")
                print(f"  On-time rate   : {fixed_delta.get('avg_on_time_rate', 0.0):+.3f}")
                print(f"  Fairness score : {fixed_delta.get('avg_fairness_score', 0.0):+.3f}")
                print(f"  Belief accuracy : {fixed_delta.get('avg_belief_accuracy', 0.0):+.3f}")
                print("Holdout evaluation delta (trained - fresh):")
                print(f"  Reward         : {holdout_delta.get('avg_total_reward', 0.0):+.1f}")
                print(f"  Completion rate: {holdout_delta.get('avg_completion_rate', 0.0):+.3f}")
                print(f"  On-time rate   : {holdout_delta.get('avg_on_time_rate', 0.0):+.3f}")
                print(f"  Fairness score : {holdout_delta.get('avg_fairness_score', 0.0):+.3f}")
                print(f"  Belief accuracy : {holdout_delta.get('avg_belief_accuracy', 0.0):+.3f}")
        print(f"Mode              : {TRAINING_AGENT_MODE.upper()}")
        print(f"Environment       : {'REAL' if USE_REAL_ENV else 'MOCK'}")
    print("="*70)


# Load existing results if present
if os.path.exists("results/training_results.json"):
    with open("results/training_results.json", "r") as f:
        try:
            existing_results = json.load(f)
            print("✅ Loaded real training results!")
        except Exception:
            existing_results = []
else:
    existing_results = []


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    default_episodes = int(os.getenv("NUM_EPISODES", "30"))
    train_agents(num_episodes=default_episodes)