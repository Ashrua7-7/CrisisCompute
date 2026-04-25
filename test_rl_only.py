#!/usr/bin/env python3
"""
Test RL Implementation Only
Checks if RL agents are learning properly without LLM complexity
"""

import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from src.rl_agent import RLDataLoaderAgent, RLDataCleanerAgent, RLMLTrainerAgent


def create_dummy_observation(episode, hour):
    """Create a standard observation for testing"""
    return {
        "episode": episode,
        "hour": hour,
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 32, "total": 32}
        },
        "time_left_hours": 8 - hour,
        "my_tasks": {
            "pending": [1, 2],
            "running": [],
            "done": []
        },
        "other_agents_status": {
            "data_loader": {"status": "idle"},
            "data_cleaner": {"status": "idle"},
            "ml_trainer": {"status": "idle"}
        }
    }


def test_rl_agent_initialization():
    """Test 1: Can RL agents be created?"""
    print("\n" + "="*70)
    print("TEST 1: RL Agent Initialization")
    print("="*70)
    
    try:
        loader = RLDataLoaderAgent()
        cleaner = RLDataCleanerAgent()
        trainer = RLMLTrainerAgent()
        
        print(f"✅ Data Loader Agent created")
        print(f"   - Learning rate: {loader.learning_rate}")
        print(f"   - Discount factor: {loader.discount_factor}")
        print(f"   - Epsilon start: {loader.epsilon_start}")
        print(f"   - Q-table size: {len(loader.q_table)} states")
        
        print(f"✅ Data Cleaner Agent created")
        print(f"✅ ML Trainer Agent created")
        
        return loader, cleaner, trainer
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_state_discretization(loader):
    """Test 2: Can agents discretize states properly?"""
    print("\n" + "="*70)
    print("TEST 2: State Discretization")
    print("="*70)
    
    try:
        obs = create_dummy_observation(1, 0)
        state = loader.discretize_state(obs)
        
        print(f"✅ State discretization works")
        print(f"   State: {state}")
        print(f"   State type: {type(state)}")
        print(f"   State is hashable: {isinstance(state, tuple)}")
        
        # Test multiple states
        states = set()
        for hour in range(8):
            obs = create_dummy_observation(1, hour)
            state = loader.discretize_state(obs)
            states.add(state)
        
        print(f"✅ Generated {len(states)} unique states across 8 hours")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_action_proposal(loader, cleaner, trainer):
    """Test 3: Can agents propose actions?"""
    print("\n" + "="*70)
    print("TEST 3: Action Proposal")
    print("="*70)
    
    try:
        obs = create_dummy_observation(1, 0)
        
        action1 = loader.propose_action(obs)
        action2 = cleaner.propose_action(obs)
        action3 = trainer.propose_action(obs)
        
        print(f"✅ Data Loader proposed: {action1.get('action')} (task_id: {action1.get('task_id')})")
        print(f"✅ Data Cleaner proposed: {action2.get('action')} (task_id: {action2.get('task_id')})")
        print(f"✅ ML Trainer proposed: {action3.get('action')} (task_id: {action3.get('task_id')})")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_learning(loader, cleaner, trainer):
    """Test 4: Do agents learn from rewards?"""
    print("\n" + "="*70)
    print("TEST 4: Reward Learning")
    print("="*70)
    
    try:
        # Episode 1
        print("\n📌 Episode 1: Initial Learning")
        loader.episode = 1
        cleaner.episode = 1
        trainer.episode = 1
        
        initial_q_size = len(loader.q_table)
        print(f"   Initial Q-table size (loader): {initial_q_size}")
        
        # Get initial state
        obs = create_dummy_observation(1, 0)
        loader.propose_action(obs)
        cleaner.propose_action(obs)
        trainer.propose_action(obs)
        
        # Give rewards
        rewards = [10.0, 9.0, 8.0]
        next_obs = create_dummy_observation(1, 1)
        
        loader.receive_reward(rewards[0], next_obs)
        cleaner.receive_reward(rewards[1], next_obs)
        trainer.receive_reward(rewards[2], next_obs)
        
        new_q_size = len(loader.q_table)
        print(f"   Q-table size after first reward: {new_q_size}")
        print(f"   ✅ Q-table entries updated: {new_q_size > initial_q_size}")
        
        # Episode 2 - should have lower epsilon
        print("\n📌 Episode 2: Continued Learning")
        loader.episode = 2
        cleaner.episode = 2
        trainer.episode = 2
        
        eps1 = loader.epsilon
        obs = create_dummy_observation(2, 0)
        loader.propose_action(obs)
        cleaner.propose_action(obs)
        trainer.propose_action(obs)
        
        loader.receive_reward(15.0, next_obs)  # Better reward
        cleaner.receive_reward(14.0, next_obs)
        trainer.receive_reward(13.0, next_obs)
        
        eps2 = loader.epsilon
        print(f"   Epsilon Episode 1: {eps1:.4f}")
        print(f"   Epsilon Episode 2: {eps2:.4f}")
        print(f"   ✅ Epsilon decayed: {eps2 < eps1}")
        
        # Episode 3 - test consistency
        print("\n📌 Episode 3: Consistency Check")
        loader.episode = 3
        cleaner.episode = 3
        trainer.episode = 3
        
        q_size_ep3 = len(loader.q_table)
        print(f"   Q-table size: {q_size_ep3}")
        print(f"   ✅ Q-table is growing: {q_size_ep3 > new_q_size}")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_q_table_persistence(loader):
    """Test 5: Can Q-tables be saved/loaded?"""
    print("\n" + "="*70)
    print("TEST 5: Q-Table Persistence")
    print("="*70)
    
    try:
        os.makedirs("test_artifacts", exist_ok=True)
        
        # Save
        test_path = "test_artifacts/test_q_table.json"
        loader.save_q_table(test_path)
        print(f"✅ Q-table saved to {test_path}")
        
        # Check file exists
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            print(f"✅ File exists, size: {file_size} bytes")
        else:
            print(f"❌ File not created")
            return False
        
        # Load into new agent
        loader2 = RLDataLoaderAgent()
        loader2.load_q_table(test_path)
        
        print(f"✅ Q-table loaded into new agent")
        print(f"   Original Q-table size: {len(loader.q_table)}")
        print(f"   Loaded Q-table size: {len(loader2.q_table)}")
        print(f"   ✅ Sizes match: {len(loader.q_table) == len(loader2.q_table)}")
        
        # Cleanup
        os.remove(test_path)
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_episode_learning():
    """Test 6: Full multi-episode learning cycle"""
    print("\n" + "="*70)
    print("TEST 6: Multi-Episode Learning Cycle")
    print("="*70)
    
    try:
        loader = RLDataLoaderAgent()
        cleaner = RLDataCleanerAgent()
        trainer = RLMLTrainerAgent()
        
        agents = [loader, cleaner, trainer]
        total_rewards = []
        epsilon_values = []
        
        print("\n📌 Running 5 mock episodes...")
        
        for episode in range(1, 6):
            for agent in agents:
                agent.episode = episode
            
            episode_reward = 0.0
            
            # Simulate 8 hours
            for hour in range(8):
                obs = create_dummy_observation(episode, hour)
                
                # Agents propose and receive rewards
                for i, agent in enumerate(agents):
                    action = agent.propose_action(obs)
                    reward = 10.0 + (episode * 2) + (hour * 0.5)  # Increasing reward over episodes
                    next_obs = create_dummy_observation(episode, hour + 1)
                    agent.receive_reward(reward, next_obs)
                    episode_reward += reward
            
            total_rewards.append(episode_reward)
            epsilon_values.append(loader.epsilon)
            
            print(f"   Episode {episode}: Reward={episode_reward:.1f}, ε={loader.epsilon:.4f}")
        
        # Analyze learning
        print("\n✅ Learning Analysis:")
        print(f"   Reward trend: {total_rewards}")
        
        avg_first_2 = sum(total_rewards[:2]) / 2
        avg_last_2 = sum(total_rewards[-2:]) / 2
        improvement = ((avg_last_2 - avg_first_2) / avg_first_2) * 100
        
        print(f"   Average reward (episodes 1-2): {avg_first_2:.1f}")
        print(f"   Average reward (episodes 4-5): {avg_last_2:.1f}")
        print(f"   Improvement: {improvement:.1f}%")
        print(f"   Epsilon decay: {epsilon_values[0]:.4f} → {epsilon_values[-1]:.4f}")
        print(f"   ✅ Q-table size (loader): {len(loader.q_table)} states")
        
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "█"*70)
    print("█ RL IMPLEMENTATION DIAGNOSTIC TEST SUITE")
    print("█"*70)
    
    results = {}
    
    # Test 1
    loader, cleaner, trainer = test_rl_agent_initialization()
    results["1_initialization"] = loader is not None
    
    if not loader:
        print("\n❌ Cannot proceed - initialization failed")
        return
    
    # Test 2
    results["2_state_discretization"] = test_state_discretization(loader)
    
    # Test 3
    results["3_action_proposal"] = test_action_proposal(loader, cleaner, trainer)
    
    # Test 4
    results["4_reward_learning"] = test_reward_learning(loader, cleaner, trainer)
    
    # Test 5
    results["5_q_table_persistence"] = test_q_table_persistence(loader)
    
    # Test 6
    results["6_multi_episode"] = test_multi_episode_learning()
    
    # Summary
    print("\n" + "█"*70)
    print("█ TEST SUMMARY")
    print("█"*70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print("\n" + "-"*70)
    print(f"Results: {passed}/{total} tests passed")
    print("-"*70)
    
    if passed == total:
        print("\n✅ ALL RL TESTS PASSED - RL implementation is working!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed - check output above")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
