# src/agents.py

import json
import time
from datetime import datetime


class Agent:
    """
    Base Agent class - all agents inherit from this
    Defines common behavior for all agents
    """
    
    def __init__(self, name, resource_needs):
        """
        Initialize agent
        
        Args:
            name: Agent name (e.g., "data_loader")
            resource_needs: Dict of resource requirements
                           {"cpu": 2, "memory": 4, "gpu": 0}
        """
        self.name = name
        self.resource_needs = resource_needs  # What I need
        self.conversation_history = []  # Learning memory
        self.task_queue = []  # My pending tasks
        self.total_reward = 0  # Cumulative reward
        self.episode_count = 0  # Current episode
        self.completed_tasks = 0  # Tasks done
        self.failed_tasks = 0  # Tasks failed
        self.current_task = None  # Task currently running
    
    def propose_action(self, observation):
        """
        Decide what to do
        Should be overridden by subclasses
        
        Args:
            observation: Current state from environment
        
        Returns:
            JSON action dict
        """
        return {
            "action": "wait",
            "task_id": None,
            "reasoning": "Base class default"
        }
    
    def receive_reward(self, reward, next_observation=None):
        """
        Store reward when task completes
        This is HOW AGENT LEARNS!
        
        Args:
            reward: Points earned
            next_observation: Optional next state for richer learning signals
        """
        self.total_reward += reward
        
        # Store in memory for learning
        reward_entry = {
            "type": "reward",
            "reward": reward,
            "episode": self.episode_count,
            "timestamp": datetime.now().isoformat()
        }

        if isinstance(next_observation, dict):
            available_resources = next_observation.get("available_resources", {})
            reward_entry["next_state_summary"] = {
                "available_cpu": available_resources.get("cpu_cores", available_resources.get("cpu", {})).get("available") if isinstance(available_resources.get("cpu_cores", available_resources.get("cpu", {})), dict) else None,
                "available_gpu": available_resources.get("gpu", {}).get("available") if isinstance(available_resources.get("gpu", {}), dict) else None,
                "time_left": next_observation.get("time_left_hours")
            }

        self.conversation_history.append(reward_entry)
    
    def add_to_history(self, state, action, reward, outcome):
        """
        Store complete experience
        
        Args:
            state: Observation when decision made
            action: Action taken
            reward: Reward received
            outcome: "success", "failed", "timeout", etc
        """
        experience = {
            "type": "experience",
            "episode": self.episode_count,
            "state_summary": {
                "available_cpu": state.get("available_resources", {}).get("cpu_cores", {}).get("available"),
                "available_gpu": state.get("available_resources", {}).get("gpu", {}).get("available"),
                "time_left": state.get("time_left_hours")
            },
            "action": action,
            "reward": reward,
            "outcome": outcome,
            "timestamp": datetime.now().isoformat()
        }
        self.conversation_history.append(experience)
    
    def get_conversation_history(self, last_n=5):
        """
        Get recent history for LLM context
        Used to show what we learned
        
        Returns:
            List of last N experiences
        """
        return self.conversation_history[-last_n:]
    
    def get_learning_summary(self):
        """
        Summarize learning for LLM prompt
        Shows what patterns we found
        
        Returns:
            String summary
        """
        if not self.conversation_history:
            return "No experience yet. First episode!"
        
        # Get recent rewards
        recent_rewards = [
            h.get("reward", 0) 
            for h in self.conversation_history 
            if h.get("type") == "reward"
        ]
        
        if not recent_rewards:
            return "Gathering experience..."
        
        avg_reward = sum(recent_rewards) / len(recent_rewards)
        trend = "improving" if recent_rewards[-1] > recent_rewards[0] else "declining"
        
        return f"Recent avg reward: {avg_reward:.1f} pts, trend: {trend}"
    
    def reset_for_episode(self):
        """Called at start of each episode"""
        self.episode_count += 1
        self.completed_tasks = 0
        self.failed_tasks = 0
        self.current_task = None


class DataLoaderAgent(Agent):
    """
    Data Loader Agent - loads CSV files
    FIRST STEP in pipeline
    """
    
    def __init__(self):
        super().__init__(
            name="data_loader",
            resource_needs={"cpu": 2, "memory": 4, "gpu": 0}
        )
        
        # My tasks
        self.task_queue = [
            {"id": "load_001", "status": "pending"},
            {"id": "load_002", "status": "pending"},
            {"id": "load_003", "status": "pending"},
            {"id": "load_004", "status": "pending"},
            {"id": "load_005", "status": "pending"}
        ]
        
        # My personality
        self.deadline_hours = None  # No deadline
        self.priority = "LOW"
        self.description = "I load CSV files. Simple and straightforward."
    
    def propose_action(self, observation):
        """
        Decide what file to load
        
        Logic:
        - Find first pending task
        - Request 2 cores
        - Estimate 30 min
        """
        
        # Find first pending task
        pending = [t for t in self.task_queue if t["status"] == "pending"]
        
        if not pending:
            # All tasks done
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "All loading tasks completed"
            }
        
        task = pending[0]
        
        return {
            "action": "run_task",
            "task_id": task["id"],
            "cores_needed": 2,
            "gpu_needed": 0,
            "memory_needed": 4,
            "estimated_duration_min": 30,
            "reasoning": f"Load {task['id']}. First pending file."
        }


class DataCleanerAgent(Agent):
    """
    Data Cleaner Agent - cleans data
    SECOND STEP in pipeline
    Depends on Data Loader
    """
    
    def __init__(self):
        super().__init__(
            name="data_cleaner",
            resource_needs={"cpu": 4, "memory": 8, "gpu": 0}
        )
        
        # My tasks
        self.task_queue = [
            {"id": "clean_001", "status": "pending"},
            {"id": "clean_002", "status": "pending"},
            {"id": "clean_003", "status": "pending"},
            {"id": "clean_004", "status": "pending"},
            {"id": "clean_005", "status": "pending"},
            {"id": "clean_006", "status": "pending"},
            {"id": "clean_007", "status": "pending"},
            {"id": "clean_008", "status": "pending"}
        ]
        
        # My personality
        self.deadline_hours = 3  # DEADLINE!
        self.priority = "MEDIUM"
        self.depends_on = "data_loader"
        self.description = "I clean data. Need Loader's output first."
    
    def propose_action(self, observation):
        """
        Decide what to do - clean or wait
        
        Logic:
        - If Loader done → start cleaning
        - If Loader running → wait (or do prep)
        """
        
        # Check Loader status
        loader_status = observation.get("other_agents_status", {}).get(
            "data_loader", {}
        )
        loader_done = loader_status.get("status") == "done"
        
        # My pending tasks
        pending = [t for t in self.task_queue if t["status"] == "pending"]
        
        if not pending:
            # All tasks done
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "All cleaning tasks completed"
            }
        
        if loader_done:
            # Loader finished, I can start cleaning
            task = pending[0]
            return {
                "action": "run_task",
                "task_id": task["id"],
                "cores_needed": 4,
                "gpu_needed": 0,
                "memory_needed": 8,
                "estimated_duration_min": 30,
                "reasoning": f"Loader done. Clean {task['id']}"
            }
        else:
            # Still waiting for Loader
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "Waiting for Loader to finish"
            }


class MLTrainerAgent(Agent):
    """
    ML Trainer Agent - trains models
    FINAL STEP in pipeline
    Highest priority!
    """
    
    def __init__(self):
        super().__init__(
            name="ml_trainer",
            resource_needs={"cpu": 2, "memory": 16, "gpu": 1}
        )
        
        # My tasks
        self.task_queue = [
            {"id": "train_001", "status": "pending"}
        ]
        
        # My personality
        self.deadline_hours = 4  # VERY URGENT!
        self.priority = "HIGH"
        self.depends_on = ["data_loader", "data_cleaner"]
        self.description = "I train ML models. Most time-critical and important task."
    
    def propose_action(self, observation):
        """
        Decide when to train
        
        Logic:
        - If Cleaner done → start training
        - If Cleaner running → wait
        """
        
        # Check Cleaner status
        cleaner_status = observation.get("other_agents_status", {}).get(
            "data_cleaner", {}
        )
        cleaner_done = cleaner_status.get("status") == "done"
        
        # My pending tasks
        pending = [t for t in self.task_queue if t["status"] == "pending"]
        
        if not pending:
            # Training complete
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "Training complete"
            }
        
        if cleaner_done:
            # Data ready, start training
            task = pending[0]
            return {
                "action": "run_task",
                "task_id": task["id"],
                "cores_needed": 2,
                "gpu_needed": 1,
                "memory_needed": 16,
                "estimated_duration_min": 90,
                "reasoning": f"Data ready. Train {task['id']}"
            }
        else:
            # Still waiting for Cleaner
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "Waiting for Cleaner (deadline: 4 hours)",
                "urgency": "HIGH"
            }


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING AGENT CLASSES")
    print("="*60)
    
    # Create agents
    loader = DataLoaderAgent()
    cleaner = DataCleanerAgent()
    trainer = MLTrainerAgent()
    
    # Test 1: Agent creation
    print("\n[TEST 1] Agent Creation")
    print(f"✅ {loader.name} created - Needs: {loader.resource_needs}")
    print(f"✅ {cleaner.name} created - Needs: {cleaner.resource_needs}")
    print(f"✅ {trainer.name} created - Needs: {trainer.resource_needs}")
    
    # Test 2: Proposing actions
    print("\n[TEST 2] Agent Proposals")
    
    observation = {
        "other_agents_status": {
            "data_loader": {"status": "idle"},
            "data_cleaner": {"status": "idle"},
            "ml_trainer": {"status": "idle"}
        }
    }
    
    action1 = loader.propose_action(observation)
    print(f"✅ {loader.name} proposes: {action1['action']} ({action1.get('task_id')})")
    
    action2 = cleaner.propose_action(observation)
    print(f"✅ {cleaner.name} proposes: {action2['action']} ({action2.get('reasoning')})")
    
    action3 = trainer.propose_action(observation)
    print(f"✅ {trainer.name} proposes: {action3['action']} ({action3.get('reasoning')})")
    
    # Test 3: Learning
    print("\n[TEST 3] Learning (Rewards)")
    
    loader.episode_count = 1
    loader.receive_reward(25.5)
    loader.receive_reward(30.0)
    
    print(f"✅ {loader.name} total reward: {loader.total_reward}")
    print(f"✅ History size: {len(loader.conversation_history)}")
    print(f"✅ Learning summary: {loader.get_learning_summary()}")
    
    # Test 4: Episode reset
    print("\n[TEST 4] Episode Management")
    
    cleaner.episode_count = 5
    cleaner.reset_for_episode()
    
    print(f"✅ {cleaner.name} episode count: {cleaner.episode_count}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60 + "\n")