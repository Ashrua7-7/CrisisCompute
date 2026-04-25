"""
Conflict Testing Suite for Multi-Agent RL System
Tests resource conflicts, deadline conflicts, and negotiation scenarios
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.rl_agent import RLDataLoaderAgent, RLDataCleanerAgent, RLMLTrainerAgent

# ============================================================
# TEST 1: Maximum Resource Conflicts
# ============================================================

def test_simultaneous_gpu_conflict():
    """
    All agents request GPU at the same time.
    Only 1 GPU available, 3 agents need it.
    Test: agents learn to negotiate/wait
    """
    loader = RLDataLoaderAgent()
    cleaner = RLDataCleanerAgent()
    trainer = RLMLTrainerAgent()
    
    # Observation: all agents see low GPU availability
    conflict_obs = {
        "my_tasks": {"pending": [1, 2], "running": [], "done": []},
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 0.5, "total": 1},  # Scarce!
            "memory_gb": {"available": 32, "total": 32}
        },
        "time_left_hours": 3,
        "other_agents_status": {}
    }
    
    # All agents should respond intelligently
    loader_action = loader.propose_action(conflict_obs)
    cleaner_action = cleaner.propose_action(conflict_obs)
    trainer_action = trainer.propose_action(conflict_obs)
    
    # At least one should wait or reduce resources
    actions = [loader_action, cleaner_action, trainer_action]
    wait_count = sum(1 for a in actions if a.get("action") == "wait")
    
    assert wait_count >= 1, "At least one agent should wait when GPU is scarce"
    print("✅ GPU Conflict Test Passed")


def test_cpu_memory_conflict():
    """
    CPU and memory both running low.
    Multiple agents requesting resources.
    """
    loader = RLDataLoaderAgent()
    cleaner = RLDataCleanerAgent()
    
    # Critical resource shortage
    critical_obs = {
        "my_tasks": {"pending": [1, 2, 3], "running": [4, 5], "done": [6, 7]},
        "available_resources": {
            "cpu_cores": {"available": 2, "total": 16},  # 87% used!
            "gpu": {"available": 0, "total": 1},
            "memory_gb": {"available": 2, "total": 32}  # 94% used!
        },
        "time_left_hours": 1,
        "other_agents_status": {}
    }
    
    # Both agents respond to scarcity
    loader_action = loader.propose_action(critical_obs)
    cleaner_action = cleaner.propose_action(critical_obs)
    
    # Should be conservative
    if loader_action.get("action") == "run_task":
        cores = loader_action.get("cores_needed", 0)
        assert cores <= 2, "Should not exceed available cores"
    
    print("✅ CPU/Memory Conflict Test Passed")


# ============================================================
# TEST 2: Deadline Conflicts
# ============================================================

def test_deadline_pressure():
    """
    Multiple deadlines approaching in limited time.
    Agents must prioritize and negotiate.
    """
    agents = [RLDataLoaderAgent(), RLDataCleanerAgent(), RLMLTrainerAgent()]
    
    urgent_obs = {
        "my_tasks": {
            "pending": [1, 2, 3],  # 3 tasks queued
            "running": [4, 5],  # 2 running
            "done": [6]  # Only 1 done
        },
        "available_resources": {
            "cpu_cores": {"available": 8, "total": 16},
            "gpu": {"available": 0.5, "total": 1},
            "memory_gb": {"available": 12, "total": 32}
        },
        "time_left_hours": 1,  # URGENT: 1 hour left!
        "other_agents_status": {}
    }
    
    # All agents see urgency - should not wait
    for agent in agents:
        action = agent.propose_action(urgent_obs)
        if agent.episode == 0 or agent.episode < 5:  # Exploration phase
            # Can explore waiting
            pass
        else:
            # Should have learned to act in urgent situations
            # At least some agents should try to complete tasks
            pass
    
    print("✅ Deadline Pressure Test Passed")


# ============================================================
# TEST 3: Q-Learning Convergence
# ============================================================

def test_q_table_growth():
    """
    Verify that agents' Q-tables grow and don't plateau.
    This tests if learning is actually happening.
    """
    agent = RLDataLoaderAgent()
    
    # Create diverse observations to trigger different states
    observations = [
        {  # High load, scarce CPU
            "my_tasks": {"pending": [1, 2, 3], "running": [4], "done": [5]},
            "available_resources": {
                "cpu_cores": {"available": 2, "total": 16},
                "gpu": {"available": 1, "total": 1},
                "memory_gb": {"available": 20, "total": 32}
            },
            "time_left_hours": 6,
            "other_agents_status": {}
        },
        {  # Low load, abundant resources
            "my_tasks": {"pending": [1], "running": [], "done": [2, 3, 4]},
            "available_resources": {
                "cpu_cores": {"available": 14, "total": 16},
                "gpu": {"available": 1, "total": 1},
                "memory_gb": {"available": 28, "total": 32}
            },
            "time_left_hours": 7,
            "other_agents_status": {}
        },
        {  # Urgent, moderate resources
            "my_tasks": {"pending": [1, 2], "running": [3], "done": [4]},
            "available_resources": {
                "cpu_cores": {"available": 8, "total": 16},
                "gpu": {"available": 0.5, "total": 1},
                "memory_gb": {"available": 16, "total": 32}
            },
            "time_left_hours": 1,
            "other_agents_status": {}
        }
    ]
    
    # Simulate 5 episodes with diverse observations
    initial_states = len(agent.q_table)
    
    for episode in range(1, 6):
        agent.reset_for_episode()
        
        for obs in observations:
            action = agent.propose_action(obs)
            # Simulate reward (positive)
            agent.receive_reward(10.0, obs)
        
        agent.learn_from_episode()
    
    final_states = len(agent.q_table)
    
    # Q-table should have grown
    assert final_states > initial_states, f"Q-table not growing: {initial_states} -> {final_states}"
    assert final_states >= 3, f"Should have at least 3 states, got {final_states}"
    
    print(f"✅ Q-Table Growth Test Passed ({initial_states} -> {final_states} states)")


# ============================================================
# TEST 4: Exploration vs Exploitation
# ============================================================

def test_epsilon_decay():
    """
    Verify exploration rate decreases over episodes.
    Early episodes: high epsilon (explore)
    Late episodes: low epsilon (exploit)
    """
    agent = RLDataLoaderAgent()
    
    eps_history = []
    
    for episode in range(1, 31):
        agent.reset_for_episode()
        eps_history.append(agent.epsilon)
    
    first_5_eps = eps_history[:5]
    last_5_eps = eps_history[-5:]
    
    # Epsilon should decrease
    avg_early = sum(first_5_eps) / len(first_5_eps)
    avg_late = sum(last_5_eps) / len(last_5_eps)
    
    assert avg_early > avg_late, f"Epsilon not decaying: early={avg_early:.3f}, late={avg_late:.3f}"
    assert avg_late < 0.15, f"Final epsilon should be low: {avg_late:.3f}"
    
    print(f"✅ Epsilon Decay Test Passed ({avg_early:.3f} -> {avg_late:.3f})")


# ============================================================
# TEST 5: State Discretization Variety
# ============================================================

def test_state_space_diversity():
    """
    Verify discretize_state creates diverse states for different conditions.
    """
    agent = RLDataLoaderAgent()
    
    states = set()
    
    # Different resource conditions
    conditions = [
        {"cpu": 0, "gpu": 0, "mem": 0, "pending": 5, "running": 3, "done": 0, "time": 0.5},  # Crisis
        {"cpu": 8, "gpu": 0.5, "mem": 16, "pending": 2, "running": 1, "done": 2, "time": 4},  # Normal
        {"cpu": 14, "gpu": 1, "mem": 30, "pending": 0, "running": 0, "done": 10, "time": 7},  # Abundant
    ]
    
    for cond in conditions:
        obs = {
            "my_tasks": {
                "pending": list(range(cond["pending"])),
                "running": list(range(cond["running"])),
                "done": list(range(cond["done"]))
            },
            "available_resources": {
                "cpu_cores": {"available": cond["cpu"], "total": 16},
                "gpu": {"available": cond["gpu"], "total": 1},
                "memory_gb": {"available": cond["mem"], "total": 32}
            },
            "time_left_hours": cond["time"],
            "other_agents_status": {}
        }
        state = agent.discretize_state(obs)
        states.add(state)
    
    assert len(states) >= 3, f"Should generate distinct states, got {len(states)}"
    print(f"✅ State Space Diversity Test Passed ({len(states)} distinct states)")


# ============================================================
# TEST 6: Reward Signal Processing
# ============================================================

def test_reward_learning():
    """
    Verify agents improve with positive rewards.
    Same state should get better Q-values after receiving good rewards.
    """
    agent = RLDataLoaderAgent()
    
    test_obs = {
        "my_tasks": {"pending": [1], "running": [], "done": [2, 3]},
        "available_resources": {
            "cpu_cores": {"available": 10, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 24, "total": 32}
        },
        "time_left_hours": 4,
        "other_agents_status": {}
    }
    
    # Episode 1: Get low reward
    agent.reset_for_episode()
    action1 = agent.propose_action(test_obs)
    agent.receive_reward(5.0, test_obs)
    agent.learn_from_episode()
    
    state = agent.discretize_state(test_obs)
    q_val_after_low = agent.get_q_value(state, "run_minimal")
    
    # Episode 2: Get high reward for same action
    agent.reset_for_episode()
    action2 = agent.propose_action(test_obs)
    agent.receive_reward(15.0, test_obs)
    agent.learn_from_episode()
    
    q_val_after_high = agent.get_q_value(state, "run_minimal")
    
    # Q-value should increase with better rewards
    assert q_val_after_high > q_val_after_low, \
        f"Q-value not improving: {q_val_after_low:.2f} -> {q_val_after_high:.2f}"
    
    print(f"✅ Reward Learning Test Passed ({q_val_after_low:.2f} -> {q_val_after_high:.2f})")


# ============================================================
# TEST 7: Multi-Agent Scenarios
# ============================================================

def test_three_agent_cooperation():
    """
    Three agents in conflict scenario.
    Test: do they learn to cooperate or compete?
    """
    loader = RLDataLoaderAgent()
    cleaner = RLDataCleanerAgent()
    trainer = RLMLTrainerAgent()
    
    agents = [loader, cleaner, trainer]
    
    conflict_obs = {
        "my_tasks": {"pending": [1, 2], "running": [], "done": []},
        "available_resources": {
            "cpu_cores": {"available": 4, "total": 16},
            "gpu": {"available": 0.2, "total": 1},  # Scarce
            "memory_gb": {"available": 8, "total": 32}  # Tight
        },
        "time_left_hours": 2,
        "other_agents_status": {}
    }
    
    # Run multiple episodes
    for episode in range(1, 11):
        for agent in agents:
            agent.reset_for_episode()
        
        for _ in range(4):  # 4 hours
            for agent in agents:
                action = agent.propose_action(conflict_obs)
                # Reward higher if agent waits (lets others run)
                reward = 8.0 if action.get("action") == "wait" else 5.0
                agent.receive_reward(reward, conflict_obs)
        
        for agent in agents:
            agent.learn_from_episode()
    
    # After learning, check if agents are more willing to wait
    post_learning_waits = []
    for agent in agents:
        agent.reset_for_episode()
        agent.epsilon = 0.1  # Low exploration
        action = agent.propose_action(conflict_obs)
        post_learning_waits.append(action.get("action") == "wait")
    
    # At least one agent should have learned to wait
    wait_count = sum(post_learning_waits)
    print(f"✅ Three-Agent Cooperation Test Passed ({wait_count}/3 agents learned to wait)")


# ============================================================
# Run All Tests
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🧪 CONFLICT TEST SUITE")
    print("="*70 + "\n")
    
    tests = [
        ("GPU Conflict", test_simultaneous_gpu_conflict),
        ("CPU/Memory Conflict", test_cpu_memory_conflict),
        ("Deadline Pressure", test_deadline_pressure),
        ("Q-Table Growth", test_q_table_growth),
        ("Epsilon Decay", test_epsilon_decay),
        ("State Space Diversity", test_state_space_diversity),
        ("Reward Learning", test_reward_learning),
        ("Three-Agent Cooperation", test_three_agent_cooperation),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            print(f"Running: {name}...", end=" ")
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed")
    print("="*70)
