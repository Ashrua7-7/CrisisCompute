#!/usr/bin/env python3
"""Quick sanity check for Theme #1 and RL"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

print("\n" + "="*70)
print("[VALIDATION] QUICK PROJECT VALIDATION")
print("="*70)

# Test 1: Theme #1 Environment
print("\n[TEST 1] Theme #1 Environment - RealEnvironment")
try:
    from satya_env.env import RealEnvironment
    env = RealEnvironment()
    obs = env.reset()
    print("✅ Environment loads and resets")
    print(f"   Observation keys: {list(obs.keys())}")
    
    # Take one step with all agents waiting
    actions = {
        "data_loader": {"action": "wait", "task_id": None, "cores_needed": 0, "gpu_needed": 0, "memory_needed": 0, "estimated_duration_min": 0},
        "data_cleaner": {"action": "wait", "task_id": None, "cores_needed": 0, "gpu_needed": 0, "memory_needed": 0, "estimated_duration_min": 0},
        "ml_trainer": {"action": "wait", "task_id": None, "cores_needed": 0, "gpu_needed": 0, "memory_needed": 0, "estimated_duration_min": 0},
    }
    result = env.step(actions)
    assert len(result) == 4, f"step() should return 4 items, got {len(result)}"
    obs2, rewards, done, info = result
    print(f"✅ Environment.step() returns (obs, rewards, done, info)")
    print(f"   Rewards: {rewards}")
    print(f"   Done: {done}")
    print(f"   Info keys: {info.keys() if isinstance(info, dict) else 'N/A'}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 2: RL Agent
print("\n[TEST 2] Theme #4 - RL Agent")
try:
    from src.rl_agent import RLDataLoaderAgent
    agent = RLDataLoaderAgent()
    print(f"✅ RL agent loads: {agent.name}")
    print(f"   Q-table size: {len(agent.q_table)}")
    print(f"   Epsilon: {agent.epsilon}")
    
    # Propose an action
    test_obs = {
        "my_tasks": {"pending": [{"id": "t1"}], "running": [], "done": []},
        "available_resources": {
            "cpu_cores": {"available": 10},
            "gpu": {"available": 1},
            "memory_gb": {"available": 24}
        },
        "time_left_hours": 5
    }
    action = agent.propose_action(test_obs)
    print(f"✅ Agent proposes action: {action.get('action')}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Training Script Import
print("\n[TEST 3] Train.py - Main Training Script")
try:
    # Just check it imports without error
    import train
    print(f"✅ train.py imports successfully")
    print(f"   Has build_agents: {hasattr(train, 'build_agents')}")
    print(f"   Has AdaptiveCurriculum: {hasattr(train, 'AdaptiveCurriculum')}")
    print(f"   Has SelfPlayLeague: {hasattr(train, 'SelfPlayLeague')}")
    print(f"   Has ChallengeGenerator: {hasattr(train, 'ChallengeGenerator')}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 4: OpenEnv Environment Wrapper
print("\n[TEST 4] OpenEnv Compliance - SatyaEnvironment")
try:
    from server.environment import SatyaEnvironment
    env = SatyaEnvironment()
    print(f"✅ SatyaEnvironment (OpenEnv) loads")
    obs = env.reset()
    print(f"✅ Reset works: done={obs.done}")
    
    from server.models import MultiAgentAction
    action = MultiAgentAction(actions={
        "data_loader": {"action": "wait"},
        "data_cleaner": {"action": "wait"},
        "ml_trainer": {"action": "wait"},
    })
    obs2 = env.step(action)
    print(f"✅ Step works: reward={obs2.reward}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Visualization & Evaluation
print("\n[TEST 5] Visualization & Metrics")
try:
    from src.visualize import ResultsVisualizer
    from src.evaluate import MetricsCalculator, LearningAnalyzer
    print(f"✅ Visualizer imports")
    print(f"✅ MetricsCalculator imports")
    print(f"✅ LearningAnalyzer imports")
except Exception as e:
    print(f"❌ FAILED: {e}")

print("\n" + "="*70)
print("✅ VALIDATION COMPLETE")
print("="*70 + "\n")
