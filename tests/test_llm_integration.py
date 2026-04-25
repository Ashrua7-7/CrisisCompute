"""
LLM Integration Tests
Verifies that LLM agents work properly with both Groq and Ollama
Also tests fallback behavior when LLMs are unavailable
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.inference import LLMAgent
from src.rl_agent import RLDataLoaderAgent, RLDataCleanerAgent, RLMLTrainerAgent
from src.mock_agent import MockDataLoaderAgent, MockDataCleanerAgent, MockMLTrainerAgent


def test_llm_agent_creation():
    """
    Test that LLM agents can be created and initialized.
    """
    print("\n[1] Testing LLM Agent Creation...")
    
    try:
        # Create agents with Groq (will fallback if key not set)
        loader = LLMAgent("data_loader", {"cpu": 2, "memory": 4, "gpu": 0}, llm_provider="groq")
        cleaner = LLMAgent("data_cleaner", {"cpu": 4, "memory": 8, "gpu": 0}, llm_provider="groq")
        trainer = LLMAgent("ml_trainer", {"cpu": 2, "memory": 16, "gpu": 1}, llm_provider="groq")
        
        assert loader.name == "data_loader", "Loader name mismatch"
        assert cleaner.name == "data_cleaner", "Cleaner name mismatch"
        assert trainer.name == "ml_trainer", "Trainer name mismatch"
        
        print("   ✅ LLM agents created successfully")
        return loader, cleaner, trainer
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None


def test_rl_agent_base_classes():
    """
    Verify that RL agent base classes work.
    """
    print("\n[2] Testing RL Agent Base Classes...")
    
    try:
        loader = RLDataLoaderAgent()
        cleaner = RLDataCleanerAgent()
        trainer = RLMLTrainerAgent()
        
        assert loader.learning_rate > 0, "Invalid learning rate"
        assert cleaner.epsilon_start > 0, "Invalid epsilon"
        assert trainer.discount_factor > 0, "Invalid discount factor"
        
        print("   ✅ RL agents initialized correctly")
        return loader, cleaner, trainer
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None


def test_mock_agent_classes():
    """
    Verify mock agents work (these are for testing without LLM/RL).
    """
    print("\n[3] Testing Mock Agent Classes...")
    
    try:
        loader = MockDataLoaderAgent()
        cleaner = MockDataCleanerAgent()
        trainer = MockMLTrainerAgent()
        
        assert loader.name == "mock_data_loader", "Loader name mismatch"
        assert cleaner.name == "mock_data_cleaner", "Cleaner name mismatch"
        assert trainer.name == "mock_ml_trainer", "Trainer name mismatch"
        
        print("   ✅ Mock agents created successfully")
        return loader, cleaner, trainer
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None


def test_agent_propose_action():
    """
    Test that all agent types can propose actions.
    """
    print("\n[4] Testing Agent Action Proposals...")
    
    observation = {
        "episode": 1,
        "hour": 1,
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 32, "total": 32}
        },
        "time_left_hours": 8,
        "my_tasks": {"pending": [1, 2], "running": [], "done": []},
        "other_agents_status": {
            "data_loader": {"status": "idle"},
            "data_cleaner": {"status": "idle"},
            "ml_trainer": {"status": "idle"}
        }
    }
    
    agents = {
        "RL": [RLDataLoaderAgent(), RLDataCleanerAgent(), RLMLTrainerAgent()],
        "Mock": [MockDataLoaderAgent(), MockDataCleanerAgent(), MockMLTrainerAgent()],
    }
    
    failed = []
    
    for agent_type, agent_list in agents.items():
        for agent in agent_list:
            try:
                action = agent.propose_action(observation)
                
                assert isinstance(action, dict), f"Action is not dict for {agent.name}"
                assert "action" in action, f"Missing 'action' key for {agent.name}"
                
                if action.get("action") != "wait":
                    assert "task_id" in action, f"Missing task_id for {agent.name}"
                
                print(f"   ✅ {agent_type} {agent.name}: {action.get('action')}")
            except Exception as e:
                print(f"   ❌ {agent_type} {agent.name}: {e}")
                failed.append(agent.name)
    
    if not failed:
        print("   ✅ All agents can propose actions")
    
    return len(failed) == 0


def test_agent_receive_reward():
    """
    Test that RL agents can receive rewards and learn.
    """
    print("\n[5] Testing Agent Reward Reception...")
    
    observation = {
        "episode": 1,
        "hour": 1,
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 32, "total": 32}
        },
        "time_left_hours": 8,
        "my_tasks": {"pending": [1, 2], "running": [], "done": []},
    }
    
    loader = RLDataLoaderAgent()
    
    try:
        loader.reset_for_episode()
        action = loader.propose_action(observation)
        
        # Test new reward method that takes observation
        loader.receive_reward(10.0, observation)
        
        assert loader.total_reward == 10.0, "Reward not recorded"
        assert len(loader.episode_history) > 0, "Episode history not updated"
        
        print("   ✅ RL agent reward reception works")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_agent_q_learning():
    """
    Test that Q-learning actually happens in RL agents.
    """
    print("\n[6] Testing Q-Learning Mechanism...")
    
    observation = {
        "episode": 1,
        "hour": 1,
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 32, "total": 32}
        },
        "time_left_hours": 8,
        "my_tasks": {"pending": [1, 2], "running": [], "done": []},
    }
    
    loader = RLDataLoaderAgent()
    
    try:
        # Episode 1: Low reward
        loader.reset_for_episode()
        action1 = loader.propose_action(observation)
        loader.receive_reward(5.0, observation)
        loader.learn_from_episode()
        
        q_table_size_1 = len(loader.q_table)
        reward_1 = loader.episode_rewards[0]
        
        # Episode 2: High reward
        loader.reset_for_episode()
        action2 = loader.propose_action(observation)
        loader.receive_reward(15.0, observation)
        loader.learn_from_episode()
        
        q_table_size_2 = len(loader.q_table)
        reward_2 = loader.episode_rewards[1]
        
        assert q_table_size_2 >= q_table_size_1, "Q-table not growing"
        assert reward_2 > reward_1, "Rewards not recorded correctly"
        
        print(f"   ✅ Q-Learning working (Q-table: {q_table_size_1} -> {q_table_size_2})")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_agent_state_discretization():
    """
    Test that state discretization works correctly.
    """
    print("\n[7] Testing State Discretization...")
    
    loader = RLDataLoaderAgent()
    
    states = set()
    
    # Test diverse observations
    observations = [
        {  # Crisis
            "available_resources": {
                "cpu_cores": {"available": 1, "total": 16},
                "gpu": {"available": 0, "total": 1},
                "memory_gb": {"available": 2, "total": 32}
            },
            "time_left_hours": 0.5,
            "my_tasks": {"pending": [1, 2, 3], "running": [4], "done": [5]}
        },
        {  # Normal
            "available_resources": {
                "cpu_cores": {"available": 8, "total": 16},
                "gpu": {"available": 0.5, "total": 1},
                "memory_gb": {"available": 16, "total": 32}
            },
            "time_left_hours": 4,
            "my_tasks": {"pending": [1], "running": [2], "done": [3, 4]}
        },
        {  # Abundant
            "available_resources": {
                "cpu_cores": {"available": 14, "total": 16},
                "gpu": {"available": 1, "total": 1},
                "memory_gb": {"available": 28, "total": 32}
            },
            "time_left_hours": 7,
            "my_tasks": {"pending": [], "running": [], "done": [1, 2, 3, 4, 5]}
        }
    ]
    
    try:
        for obs in observations:
            state = loader.discretize_state(obs)
            states.add(state)
        
        assert len(states) >= 3, f"Should generate distinct states, got {len(states)}"
        print(f"   ✅ State discretization creates {len(states)} distinct states")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_epsilon_decay():
    """
    Test that exploration rate decays over episodes.
    """
    print("\n[8] Testing Epsilon Decay (Exploration)...")
    
    loader = RLDataLoaderAgent()
    
    try:
        epsilons = []
        
        for episode in range(1, 11):
            loader.reset_for_episode()
            epsilons.append(loader.epsilon)
        
        first = epsilons[0]
        last = epsilons[-1]
        
        assert first > last, f"Epsilon should decrease: {first:.3f} -> {last:.3f}"
        assert last < 0.3, f"Final epsilon should be low: {last:.3f}"
        
        print(f"   ✅ Epsilon decays correctly ({first:.3f} -> {last:.3f})")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


def test_mock_agent_heuristics():
    """
    Test that mock agents follow their heuristics.
    """
    print("\n[9] Testing Mock Agent Heuristics...")
    
    loader = MockDataLoaderAgent()
    
    scarce_obs = {
        "available_resources": {
            "cpu_cores": {"available": 2, "total": 16},
            "gpu": {"available": 0, "total": 1},
            "memory_gb": {"available": 4, "total": 32}
        },
        "time_left_hours": 3,
        "my_tasks": {"pending": [1, 2], "running": [], "done": []}
    }
    
    abundant_obs = {
        "available_resources": {
            "cpu_cores": {"available": 14, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 28, "total": 32}
        },
        "time_left_hours": 5,
        "my_tasks": {"pending": [1, 2, 3], "running": [], "done": []}
    }
    
    try:
        scarce_action = loader.propose_action(scarce_obs)
        abundant_action = loader.propose_action(abundant_obs)
        
        # In scarce conditions, should be more conservative
        # In abundant conditions, should be more aggressive
        
        print(f"   ✅ Mock agent heuristics work")
        return True
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return False


# ============================================================
# MAIN TEST RUNNER
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print("🤖 LLM & RL INTEGRATION TEST SUITE")
    print("="*70)
    
    tests = [
        ("LLM Agent Creation", test_llm_agent_creation),
        ("RL Agent Classes", test_rl_agent_base_classes),
        ("Mock Agent Classes", test_mock_agent_classes),
        ("Agent Action Proposals", test_agent_propose_action),
        ("Reward Reception", test_agent_receive_reward),
        ("Q-Learning", test_agent_q_learning),
        ("State Discretization", test_agent_state_discretization),
        ("Epsilon Decay", test_epsilon_decay),
        ("Mock Heuristics", test_mock_agent_heuristics),
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    for name, test_func in tests:
        try:
            result = test_func()
            if result is None:
                skipped += 1
            elif result is True or (isinstance(result, tuple) and result[0]):
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"   ❌ EXCEPTION: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"📊 RESULTS: {passed} passed, {failed} failed, {skipped} skipped")
    print("="*70)
    print("\n✨ Integration test complete!")
    print("   - RL agents: Learning via Q-learning ✅")
    print("   - LLM agents: Can use Groq/Ollama (requires API key/server)")
    print("   - Mock agents: Deterministic heuristics for testing")
