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
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

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

    print("\n" + "="*70)
    print("🚀 MULTI-AGENT TRAINING SYSTEM")
    print("="*70)
    print(f"Episodes:    {num_episodes}")
    print(f"Environment: {'REAL (Satya)' if USE_REAL_ENV else 'MOCK (Testing)'}")
    print(f"Agent Mode:  {TRAINING_AGENT_MODE.upper()}")
    print(f"Start Time:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70 + "\n")

    env = MultiAgentPipelineEnv(config_dir=None, seed=42) if USE_REAL_ENV else SimpleEnvironment()
    agents_dict, agents = build_agents(TRAINING_AGENT_MODE)

    # RL agent reference for epsilon display
    def _get_rl(agent):
        return agent.rl if hasattr(agent, "rl") else agent

    # Load Q-tables if available
    if any(hasattr(_get_rl(a), "load_q_table") for a in agents):
        print("📚 Loading previous Q-tables...")
        os.makedirs("q_tables", exist_ok=True)
        for agent in agents:
            rl = _get_rl(agent)
            if hasattr(rl, "load_q_table"):
                name = rl.name.replace("rl_", "")
                if not rl.load_q_table(f"q_tables/{name}_q_table.json"):
                    print(f"   ℹ️  Starting fresh for {rl.name}")
        print()

    all_results = []

    # ========== TRAINING LOOP ==========
    for episode in range(1, num_episodes + 1):

        obs_per_agent = env.reset()

        # Ensure obs is always per-agent dict
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
        }

        total_episode_reward = 0
        info = {}

        for hour in range(1, 9):
            actions_dict = {}

            for agent_id in ["data_loader", "data_cleaner", "ml_trainer"]:
                agent      = agents_dict[agent_id]
                agent_obs  = obs_per_agent.get(agent_id, obs_per_agent)

                if isinstance(agent_obs, dict):
                    agent_obs["hour"] = hour

                # Hybrid uses .act(), others use .propose_action()
                if TRAINING_AGENT_MODE == "hybrid":
                    action = agent.act(agent_obs)
                else:
                    action = agent.propose_action(agent_obs)

                actions_dict[agent_id] = action

            episode_data["agent_actions"].append({"hour": hour, "actions": actions_dict})

            result = env.step(actions_dict)
            if len(result) == 4:
                next_obs_per_agent, rewards, done, info = result
            else:
                next_obs_per_agent, rewards, done = result
                info = {}

            # Ensure next obs is per-agent dict
            if not isinstance(next_obs_per_agent, dict) or "data_loader" not in next_obs_per_agent:
                next_obs_per_agent = {"data_loader": next_obs_per_agent, "data_cleaner": next_obs_per_agent, "ml_trainer": next_obs_per_agent}

            # Agents learn
            for i, agent_id in enumerate(["data_loader", "data_cleaner", "ml_trainer"]):
                agent        = agents_dict[agent_id]
                next_agent_obs = next_obs_per_agent.get(agent_id, next_obs_per_agent)

                if TRAINING_AGENT_MODE == "hybrid":
                    agent.learn(obs_per_agent.get(agent_id, obs_per_agent), actions_dict[agent_id], rewards[i], next_agent_obs)
                else:
                    agent.receive_reward(rewards[i], next_agent_obs)

                total_episode_reward += rewards[i]

            obs_per_agent = next_obs_per_agent

            if done:
                break

        episode_data["total_reward"] = total_episode_reward
        if isinstance(info, dict):
            m = info.get("metrics", {})
            episode_data["completed_tasks"] = int(m.get("completed_tasks", min(int(total_episode_reward / 10.0), 15)))
            episode_data["on_time_tasks"]   = int(m.get("on_time_tasks",   min(int(total_episode_reward / 12.0), episode_data["completed_tasks"])))
            episode_data["conflicts"]       = sum(1 for e in info.get("events", []) if "conflict"  in e)
            episode_data["agreements"]      = sum(1 for e in info.get("events", []) if "allocated" in e)

        all_results.append(episode_data)

        # End-of-episode learning for RL/Hybrid
        for agent in agents:
            rl = _get_rl(agent)
            if hasattr(rl, "learn_from_episode"):
                rl.learn_from_episode()

        if episode % 5 == 0 or episode == 1:
            completion_rate = MetricsCalculator.calculate_completion_rate(episode_data)
            first_rl        = _get_rl(agents[0])
            epsilon         = first_rl.epsilon if hasattr(first_rl, "epsilon") else 0
            print(f"Episode {episode:2d} │ Reward: {total_episode_reward:7.1f} │ Completion: {completion_rate:5.1f}% │ ε: {epsilon:.3f}")

        if len(all_results) > 5:
            recent = [r["total_reward"] for r in all_results[-5:]]
            if len(set(recent)) == 1:
                print(f"⚠️  Plateau detected at episode {episode}")

        if episode % 10 == 0:
            print(f"   💾 Checkpoint episode {episode}")

    # ========== POST-TRAINING ==========
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)

    os.makedirs("results", exist_ok=True)
    with open("results/training_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("✅ Saved: results/training_results.json")

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