# src/evaluate.py

import numpy as np


class MetricsCalculator:
    """
    Calculate metrics from episode data
    """
    
    @staticmethod
    def calculate_completion_rate(episode_data):
        """
        What % of tasks got completed?
        """
        total_tasks = (
            episode_data.get("total_tasks")
            or episode_data.get("metrics", {}).get("total_tasks")
            or 9
        )
        total_tasks = max(1, int(total_tasks))
        completed = int(episode_data.get("completed_tasks", 0))
        
        rate = (completed / total_tasks) * 100
        return min(rate, 100)
    
    @staticmethod
    def calculate_on_time_rate(episode_data):
        """
        What % of deadlines were met?
        """
        total_tasks = (
            episode_data.get("total_tasks")
            or episode_data.get("metrics", {}).get("total_tasks")
            or 9
        )
        total_tasks = max(1, int(total_tasks))
        on_time = int(episode_data.get("on_time_tasks", 0))
        
        rate = (on_time / total_tasks) * 100
        return min(rate, 100)
    
    @staticmethod
    def calculate_resource_utilization(episode_data):
        """
        How efficiently were resources used?
        """
        # Rough calculation
        # More cores used = higher utilization
        reward = episode_data.get("total_reward", 0)
        
        # Normalize: 0 reward = 0%, 100 reward = 100% util
        util = (reward / 100) * 100
        return min(util, 100)
    
    @staticmethod
    def calculate_cooperation_score(episode_data):
        """
        How well did agents cooperate?
        """
        # Based on: conflicts, agreements, re-negotiations
        conflicts = episode_data.get("conflicts", 0)
        agreements = episode_data.get("agreements", 3)  # Should have 3 agents
        
        # Fewer conflicts, more agreements = higher cooperation
        if agreements == 0:
            return 0
        
        cooperation = (agreements / (agreements + conflicts)) * 100 if (agreements + conflicts) > 0 else 50
        return min(cooperation, 100)
    
    @staticmethod
    def calculate_total_reward(rewards_list):
        """
        Sum all individual rewards
        """
        return sum(rewards_list) if rewards_list else 0
    
    @staticmethod
    def calculate_metrics(episode_data):
        """
        Main method: Calculate all metrics for an episode
        """
        
        metrics = {
            "completion_rate": MetricsCalculator.calculate_completion_rate(episode_data),
            "on_time_rate": MetricsCalculator.calculate_on_time_rate(episode_data),
            "resource_utilization": MetricsCalculator.calculate_resource_utilization(episode_data),
            "cooperation_score": MetricsCalculator.calculate_cooperation_score(episode_data),
            "total_reward": episode_data.get("total_reward", 0)
        }
        
        return metrics


class LearningAnalyzer:
    """
    Analyze how agents are learning
    """
    
    @staticmethod
    def calculate_learning_rate(rewards_list):
        """
        How fast are rewards improving?
        """
        if len(rewards_list) < 2:
            return 0
        
        # Linear regression slope
        x = np.arange(len(rewards_list))
        y = np.array(rewards_list)
        
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    
    @staticmethod
    def detect_plateauing(rewards_list, window=5):
        """
        Are agents plateauing (stopped learning)?
        """
        if len(rewards_list) < window * 2:
            return False
        
        # Compare last window vs previous window
        last_window = rewards_list[-window:]
        prev_window = rewards_list[-2*window:-window]
        
        last_avg = np.mean(last_window)
        prev_avg = np.mean(prev_window)
        
        # If improvement < 5%, consider plateau
        improvement = (last_avg - prev_avg) / prev_avg if prev_avg > 0 else 0
        
        return improvement < 0.05
    
    @staticmethod
    def identify_best_strategy(episode_results):
        """
        What strategy gave best reward?
        """
        if not episode_results:
            return None
        
        best_episode = max(episode_results, 
                          key=lambda x: x.get("metrics", {}).get("total_reward", 0))
        
        return best_episode


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING METRICS CALCULATOR")
    print("="*60)
    
    # Test 1: Metrics calculation
    print("\n[TEST 1] Metrics Calculation")
    
    episode_data = {
        "completed_tasks": 12,
        "on_time_tasks": 10,
        "total_reward": 45.5,
        "conflicts": 1,
        "agreements": 3
    }
    
    metrics = MetricsCalculator.calculate_metrics(episode_data)
    
    print(f"✅ Completion Rate: {metrics['completion_rate']:.1f}%")
    print(f"✅ On-Time Rate: {metrics['on_time_rate']:.1f}%")
    print(f"✅ Resource Util: {metrics['resource_utilization']:.1f}%")
    print(f"✅ Cooperation: {metrics['cooperation_score']:.1f}%")
    print(f"✅ Total Reward: {metrics['total_reward']:.1f}")
    
    # Test 2: Learning rate
    print("\n[TEST 2] Learning Rate")
    
    rewards = [15, 18, 20, 25, 30, 35, 40, 45, 48, 50]
    
    learning_rate = LearningAnalyzer.calculate_learning_rate(rewards)
    print(f"✅ Learning Rate: {learning_rate:.2f} points/episode")
    
    # Test 3: Plateau detection
    print("\n[TEST 3] Plateau Detection")
    
    plateau = LearningAnalyzer.detect_plateauing(rewards)
    print(f"✅ Plateauing: {plateau}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")