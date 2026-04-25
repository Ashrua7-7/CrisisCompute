"""
test_all_modes.py

Tests your SatyaEnv server in 3 modes:
  1. Random Agent     — picks random actions (baseline, dumb)
  2. Greedy RL Agent  — always picks next available task (smart rule-based)
  3. LLM Agent        — uses Ollama locally (if available) or falls back to greedy

Run:
  python test_all_modes.py

Make sure your server is running:
  uvicorn server.app:app --port 7860
"""

import requests
import random
import json
import time

BASE_URL = "http://localhost:7860"

AGENTS = ["data_loader", "data_cleaner", "ml_trainer"]

# Task ownership map (matches your tasks.json)
TASK_OWNER = {
    "load_raw_batch_1": "data_loader",
    "load_raw_batch_2": "data_loader",
    "load_raw_batch_3": "data_loader",
    "clean_batch_1":    "data_cleaner",
    "clean_batch_2":    "data_cleaner",
    "clean_batch_3":    "data_cleaner",
    "train_baseline_model": "ml_trainer",
    "train_advanced_model": "ml_trainer",
    "validate_models":      "ml_trainer",
}

# Task dependency map
TASK_DEPS = {
    "load_raw_batch_1": [],
    "load_raw_batch_2": [],
    "load_raw_batch_3": [],
    "clean_batch_1":    ["load_raw_batch_1"],
    "clean_batch_2":    ["load_raw_batch_2"],
    "clean_batch_3":    ["load_raw_batch_3"],
    "train_baseline_model": ["clean_batch_1", "clean_batch_2"],
    "train_advanced_model": ["clean_batch_3"],
    "validate_models":      ["train_baseline_model", "train_advanced_model"],
}


# ─────────────────────────────────────────────
# Server helpers
# ─────────────────────────────────────────────

def reset():
    r = requests.post(f"{BASE_URL}/reset")
    r.raise_for_status()
    return r.json()

def step(actions: dict):
    r = requests.post(f"{BASE_URL}/step", json={"actions": actions})
    r.raise_for_status()
    return r.json()

def get_state():
    r = requests.get(f"{BASE_URL}/state")
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
# Agent policies
# ─────────────────────────────────────────────

def random_policy(obs):
    """Completely random — picks random task or waits."""
    actions = {}
    all_tasks = list(TASK_OWNER.keys())
    for agent in AGENTS:
        my_tasks = [t for t, o in TASK_OWNER.items() if o == agent]
        if random.random() > 0.4:
            task = random.choice(my_tasks)
            actions[agent] = {"action": "run_task", "task_id": task}
        else:
            actions[agent] = {"action": "wait", "task_id": None}
    return actions


def greedy_policy(obs, completed_tasks: set):
    """
    RL-style greedy — each agent picks their next task
    whose dependencies are all done.
    """
    actions = {}
    for agent in AGENTS:
        my_tasks = [t for t, o in TASK_OWNER.items() if o == agent]
        picked = None
        for task_id in my_tasks:
            if task_id in completed_tasks:
                continue
            deps = TASK_DEPS[task_id]
            if all(d in completed_tasks for d in deps):
                picked = task_id
                break
        if picked:
            actions[agent] = {"action": "run_task", "task_id": picked}
        else:
            actions[agent] = {"action": "wait", "task_id": None}
    return actions


def llm_policy(obs):
    """
    LLM agent — sends observation to Ollama (mistral) and parses action.
    Falls back to greedy if Ollama not available.
    """
    prompt = f"""
You are a resource scheduling agent in an ML pipeline.
Current state:
- Hour: {obs['observation']['current_hour']} / {obs['observation']['max_hours']}
- Metrics: {obs['observation']['metrics']}
- Recent events: {obs['observation']['recent_events'][-3:]}

You control 3 agents: data_loader, data_cleaner, ml_trainer.
Each agent must pick one action:
  - run_task with a task_id (only tasks they own)
  - wait

Task ownership:
  data_loader  -> load_raw_batch_1, load_raw_batch_2, load_raw_batch_3
  data_cleaner -> clean_batch_1, clean_batch_2, clean_batch_3 (need load done first)
  ml_trainer   -> train_baseline_model, train_advanced_model, validate_models (need clean done first)

Respond ONLY with valid JSON like:
{{
  "data_loader":  {{"action": "run_task", "task_id": "load_raw_batch_1"}},
  "data_cleaner": {{"action": "wait", "task_id": null}},
  "ml_trainer":   {{"action": "wait", "task_id": null}}
}}
"""
    try:
        r = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "mistral", "prompt": prompt, "stream": False},
            timeout=15,
        )
        text = r.json()["response"].strip()
        # Extract JSON from response
        start = text.find("{")
        end   = text.rfind("}") + 1
        actions = json.loads(text[start:end])
        # Validate keys
        assert all(a in actions for a in AGENTS)
        return actions
    except Exception as e:
        print(f"  [LLM fallback to greedy: {e}]")
        # Fallback to greedy
        completed = set()
        metrics = obs.get("observation", {}).get("metrics", {})
        # We don't have full completed list from obs alone, use greedy with empty set
        return greedy_policy(obs, completed)


# ─────────────────────────────────────────────
# Episode runner
# ─────────────────────────────────────────────

def run_episode(mode: str):
    print(f"\n{'='*55}")
    print(f"  MODE: {mode.upper()}")
    print(f"{'='*55}")

    obs = reset()
    completed_tasks = set()
    total_reward = 0.0
    hours = obs["observation"]["max_hours"]

    for hour in range(hours):
        # Pick actions based on mode
        if mode == "random":
            actions = random_policy(obs)
        elif mode == "greedy_rl":
            actions = greedy_policy(obs, completed_tasks)
        elif mode == "llm":
            actions = llm_policy(obs)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Step
        result = step(actions)
        obs = result
        reward = result.get("reward") or sum(result.get("step_rewards", {}).values())
        total_reward += reward

        metrics = result["observation"]["metrics"]
        events  = result["observation"]["recent_events"]

        # Track completed tasks
        for event in events:
            for task_id in TASK_OWNER:
                if task_id in event and "allocated" in event:
                    completed_tasks.add(task_id)
                if task_id in event and "completed" in event:
                    completed_tasks.add(task_id)

        print(f"  Hour {hour+1:2d} | reward: {round(reward,2):7.2f} | "
              f"done: {metrics.get('completed_tasks',0)}/9 | "
              f"actions: { {a: v['action']+('→'+v['task_id'] if v.get('task_id') else '') for a,v in actions.items()} }")

        if result["done"]:
            break

    state = get_state()
    print(f"\n  ── FINAL RESULTS ──")
    print(f"  Total reward    : {round(total_reward, 2)}")
    print(f"  Tasks completed : {metrics.get('completed_tasks',0)}/9")
    print(f"  On-time tasks   : {metrics.get('on_time_tasks',0)}/9")
    print(f"  Completion rate : {round(metrics.get('completion_rate',0)*100, 1)}%")
    print(f"  On-time rate    : {round(metrics.get('on_time_rate',0)*100, 1)}%")

    return total_reward, metrics


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("\n🔍 Checking server health...")
    try:
        r = requests.get(f"{BASE_URL}/health")
        print(f"  ✅ Server is up: {r.json()}")
    except Exception:
        print("  ❌ Server not running! Start it with:")
        print("     uvicorn server.app:app --port 7860")
        exit(1)

    results = {}

    # 1. Random agent (baseline — should perform worst)
    reward, metrics = run_episode("random")
    results["random"] = {"reward": round(reward,2), "completion": f"{metrics.get('completed_tasks',0)}/9"}

    time.sleep(0.5)

    # 2. Greedy RL agent (rule-based — should perform better)
    reward, metrics = run_episode("greedy_rl")
    results["greedy_rl"] = {"reward": round(reward,2), "completion": f"{metrics.get('completed_tasks',0)}/9"}

    time.sleep(0.5)

    # 3. LLM agent (uses Ollama mistral — falls back to greedy if not available)
    print("\n⚠️  LLM mode needs Ollama running locally (ollama run mistral)")
    print("   If not available, it will fallback to greedy automatically.")
    reward, metrics = run_episode("llm")
    results["llm"] = {"reward": round(reward,2), "completion": f"{metrics.get('completed_tasks',0)}/9"}

    # Summary
    print(f"\n{'='*55}")
    print("  COMPARISON SUMMARY")
    print(f"{'='*55}")
    print(f"  {'Mode':<15} {'Total Reward':>14} {'Tasks Done':>12}")
    print(f"  {'-'*43}")
    for mode, r in results.items():
        print(f"  {mode:<15} {str(r['reward']):>14} {r['completion']:>12}")
    print(f"{'='*55}")
    print("\n  Higher reward + more tasks = better agent 🎯")
