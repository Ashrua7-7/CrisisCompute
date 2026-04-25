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

    def _prepare_env_for_mode(env_obj, negotiation_flag: bool, crisis_flag: bool):
        if hasattr(env_obj, "negotiation_enabled"):
            env_obj.negotiation_enabled = negotiation_flag
        if hasattr(env_obj, "crisis_mode_enabled"):
            env_obj.crisis_mode_enabled = crisis_flag

    def _safe_metrics_for_episode(episode_data):
        metrics = episode_data.get("metrics", {})
        if metrics:
            return metrics
        return MetricsCalculator.calculate_metrics(episode_data)

    def _write_json(path, payload):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def _run_single_session(session_episodes: int, seed: int, negotiation_flag: bool, crisis_flag: bool):
        env = MultiAgentPipelineEnv(config_dir=None, seed=seed) if USE_REAL_ENV else SimpleEnvironment()
        _prepare_env_for_mode(env, negotiation_flag, crisis_flag)
        agents_dict, agents = build_agents(TRAINING_AGENT_MODE)

        def _get_rl(agent):
            return agent.rl if hasattr(agent, "rl") else agent

        if any(hasattr(_get_rl(a), "load_q_table") for a in agents):
            os.makedirs("q_tables", exist_ok=True)
            for agent in agents:
                rl = _get_rl(agent)
                if hasattr(rl, "load_q_table"):
                    name = rl.name.replace("rl_", "")
                    rl.load_q_table(f"q_tables/{name}_q_table.json")

        all_results = []
        episode_metrics = []
        negotiation_trace = []

        for episode in range(1, session_episodes + 1):
            obs_per_agent = env.reset()
            if not isinstance(obs_per_agent, dict) or "data_loader" not in obs_per_agent:
                obs_per_agent = {"data_loader": obs_per_agent, "data_cleaner": obs_per_agent, "ml_trainer": obs_per_agent}

            for agent in agents:
                if hasattr(agent, "reset_for_episode"):
                    agent.reset_for_episode()

            episode_data = {
                "episode": episode,
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
                        agent.learn(obs_per_agent.get(agent_id, obs_per_agent), actions_dict[agent_id], rewards[i], next_agent_obs)
                    else:
                        agent.receive_reward(rewards[i], next_agent_obs)
                    total_episode_reward += rewards[i]

                obs_per_agent = next_obs_per_agent
                step_count += 1

                step_metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
                step_trace = info.get("negotiation_trace_step") if isinstance(info, dict) else None
                if step_trace:
                    negotiation_trace.append({"episode": episode, **deepcopy(step_trace)})
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
                }
            )

            for agent in agents:
                rl = _get_rl(agent)
                if hasattr(rl, "learn_from_episode"):
                    rl.learn_from_episode()

        return all_results, episode_metrics, negotiation_trace, agents

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

    all_results, episode_metrics, negotiation_trace, agents = _run_single_session(
        session_episodes=num_episodes,
        seed=seed_values[0],
        negotiation_flag=negotiation_enabled,
        crisis_flag=crisis_mode_enabled,
    )

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
    )
    negotiated_summary = {
        "avg_total_reward": mean([r["total_reward"] for r in all_results]) if all_results else 0.0,
        "avg_completion_rate": mean([_safe_metrics_for_episode(r)["completion_rate"] for r in all_results]) if all_results else 0.0,
        "avg_on_time_rate": mean([_safe_metrics_for_episode(r)["on_time_rate"] for r in all_results]) if all_results else 0.0,
        "avg_fairness_score": mean([r.get("avg_fairness_score", 0.0) for r in all_results]) if all_results else 0.0,
        "avg_belief_accuracy": mean([r.get("avg_belief_accuracy", 0.0) for r in all_results]) if all_results else 0.0,
    }
    baseline_summary = {
        "avg_total_reward": mean([r["total_reward"] for r in baseline_results]) if baseline_results else 0.0,
        "avg_completion_rate": mean([_safe_metrics_for_episode(r)["completion_rate"] for r in baseline_results]) if baseline_results else 0.0,
        "avg_on_time_rate": mean([_safe_metrics_for_episode(r)["on_time_rate"] for r in baseline_results]) if baseline_results else 0.0,
        "avg_fairness_score": mean([r.get("avg_fairness_score", 0.0) for r in baseline_results]) if baseline_results else 0.0,
        "avg_belief_accuracy": mean([r.get("avg_belief_accuracy", 0.0) for r in baseline_results]) if baseline_results else 0.0,
    }
    baseline_comparison = {
        "config": {"seed": seed_values[0], "negotiation_enabled": negotiation_enabled},
        "negotiation_enabled_run": negotiated_summary,
        "baseline_run_negotiation_disabled": baseline_summary,
        "delta": {k: negotiated_summary.get(k, 0.0) - baseline_summary.get(k, 0.0) for k in negotiated_summary.keys()},
    }
    _write_json("results/baseline_comparison.json", baseline_comparison)
    print("✅ Saved: results/baseline_comparison.json")

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
            avg_first = sum(r[:5])  / 5 if len(r) >= 5 else 0
            avg_last  = sum(r[-5:]) / 5 if len(r) >= 5 else 0
            gain      = avg_last - avg_first
            print(f"\n👤 {rl.name.upper()}")
            print(f"   Q-table states       : {len(rl.q_table)}")
            print(f"   Avg reward (ep 1-5)  : {avg_first:.1f}")
            print(f"   Avg reward (ep 26-30): {avg_last:.1f}")
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
        first = all_results[0]["total_reward"]
        last  = all_results[-1]["total_reward"]
        imp   = ((last - first) / abs(first) * 100) if first != 0 else 0
        print(f"Episode 1 Reward  : {first:.1f}")
        print(f"Episode {len(all_results)} Reward : {last:.1f}")
        print(f"Improvement       : {imp:+.1f}%")
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
    train_agents(num_episodes=5)