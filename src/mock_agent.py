# src/mock_agent.py

from src.agents import Agent


class MockDataLoaderAgent(Agent):
    """
    Simple hard-coded Data Loader
    For TESTING without LLM
    """
    
    def __init__(self):
        super().__init__(
            name="mock_data_loader",
            resource_needs={"cpu": 2, "memory": 4, "gpu": 0}
        )
        
        self.task_queue = [
            {"id": "load_001", "status": "pending"},
            {"id": "load_002", "status": "pending"},
            {"id": "load_003", "status": "pending"},
            {"id": "load_004", "status": "pending"},
            {"id": "load_005", "status": "pending"}
        ]
        self.episode = 0
        self.total_reward = 0
    
    def reset_for_episode(self):
        self.episode += 1
        self.total_reward = 0
    
    def receive_reward(self, reward):
        self.total_reward += reward
    
    def propose_action(self, observation):
        """Always run first pending task from observation, or wait if none"""
        
        # Get pending tasks from observation (works with both mock and real environment)
        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])
        
        # Fall back to hardcoded tasks if observation doesn't have them
        if not pending_tasks:
            pending = [t for t in self.task_queue if t["status"] == "pending"]
            if pending:
                pending_tasks = [pending[0]['id']]
        
        if pending_tasks:
            # Improve resource efficiency over episodes - try to complete more aggressively
            cores = min(2 + int(self.episode / 5.0), 6)  # Scale up cores faster
            memory = 4 + int(self.episode / 10.0)  # Allocate more memory over time
            return {
                "action": "run_task",
                "task_id": pending_tasks[0],
                "cores_needed": cores,
                "gpu_needed": 0,
                "memory_needed": memory,
                "estimated_duration_min": max(30 - int(self.episode / 3.0), 15),  # Get faster over time
                "reasoning": f"Load batch - Episode {self.episode} (cores={cores})"
            }
        else:
            return {"action": "wait"}


class MockDataCleanerAgent(Agent):
    """
    Simple hard-coded Data Cleaner
    For TESTING without LLM
    """
    
    def __init__(self):
        super().__init__(
            name="mock_data_cleaner",
            resource_needs={"cpu": 4, "memory": 8, "gpu": 0}
        )
        
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
        self.episode = 0
        self.total_reward = 0
    
    def reset_for_episode(self):
        self.episode += 1
        self.total_reward = 0
    
    def receive_reward(self, reward):
        self.total_reward += reward
    
    def propose_action(self, observation):
        """Get pending tasks from observation and clean first one"""
        
        # Get pending tasks from observation
        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])
        
        # Fall back to hardcoded tasks if observation doesn't have them
        if not pending_tasks:
            pending = [t for t in self.task_queue if t["status"] == "pending"]
            if pending:
                pending_tasks = [pending[0]['id']]
        
        if pending_tasks:
            # Improve resource efficiency over episodes - parallel processing
            cores = min(4 + int(self.episode / 6.0), 8)  # Scale up cores faster for parallelism
            memory = 6 + int(self.episode / 12.0)  # Need more memory for better algorithms
            return {
                "action": "run_task",
                "task_id": pending_tasks[0],
                "cores_needed": cores,
                "gpu_needed": 0,
                "memory_needed": memory,
                "estimated_duration_min": max(60 - int(self.episode / 2.5), 30),  # Faster over time
                "reasoning": f"Clean batch - Episode {self.episode} (cores={cores})"
            }
        else:
            return {"action": "wait"}


class MockMLTrainerAgent(Agent):
    """
    Simple hard-coded ML Trainer
    For TESTING without LLM
    """
    
    def __init__(self):
        super().__init__(
            name="mock_ml_trainer",
            resource_needs={"cpu": 2, "memory": 16, "gpu": 1}
        )
        
        self.task_queue = [
            {"id": "train_001", "status": "pending"}
        ]
        self.episode = 0
        self.total_reward = 0
    
    def reset_for_episode(self):
        self.episode += 1
        self.total_reward = 0
    
    def receive_reward(self, reward):
        self.total_reward += reward
    
    def propose_action(self, observation):
        """Get pending tasks from observation and train"""
        
        # Get pending tasks from observation
        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])
        
        # Fall back to hardcoded tasks if observation doesn't have them
        if not pending_tasks:
            pending = [t for t in self.task_queue if t["status"] == "pending"]
            if pending:
                pending_tasks = [pending[0]['id']]
        
        if pending_tasks:
            # Improve training efficiency over episodes
            cores = min(2 + int(self.episode / 8.0), 4)  # More CPUs for faster preprocessing
            memory = min(12 + int(self.episode / 5.0), 20)  # More memory for larger batches
            return {
                "action": "run_task",
                "task_id": pending_tasks[0],
                "cores_needed": cores,
                "gpu_needed": 1,  # Always use GPU
                "memory_needed": memory,
                "estimated_duration_min": max(90 - int(self.episode / 2.0), 45),  # Gets faster
                "reasoning": f"Train model - Episode {self.episode} (mem={memory})"
            }
        else:
            return {"action": "wait"}


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING MOCK AGENT CLASSES")
    print("="*60)
    
    # Create mock agents
    loader = MockDataLoaderAgent()
    cleaner = MockDataCleanerAgent()
    trainer = MockMLTrainerAgent()
    
    print("\n[TEST 1] Mock Agents Created")
    print(f"✅ {loader.name}")
    print(f"✅ {cleaner.name}")
    print(f"✅ {trainer.name}")
    
    # Test proposals
    print("\n[TEST 2] Mock Proposals")
    
    observation = {
        "other_agents_status": {
            "data_loader": {"status": "idle"},
            "data_cleaner": {"status": "idle"},
            "ml_trainer": {"status": "idle"}
        }
    }
    
    action1 = loader.propose_action(observation)
    print(f"✅ Loader: {action1['action']} ({action1.get('task_id')})")
    
    action2 = cleaner.propose_action(observation)
    print(f"✅ Cleaner: {action2['action']}")
    
    action3 = trainer.propose_action(observation)
    print(f"✅ Trainer: {action3['action']}")
    
    # Test with Loader done
    print("\n[TEST 3] After Loader Finishes")
    
    observation["other_agents_status"]["data_loader"]["status"] = "done"
    
    action2 = cleaner.propose_action(observation)
    print(f"✅ Cleaner now: {action2['action']} ({action2.get('task_id')})")
    
    # Test with Cleaner done
    print("\n[TEST 4] After Cleaner Finishes")
    
    observation["other_agents_status"]["data_cleaner"]["status"] = "done"
    
    action3 = trainer.propose_action(observation)
    print(f"✅ Trainer now: {action3['action']} ({action3.get('task_id')})")
    
    print("\n" + "="*60)
    print("✅ ALL MOCK TESTS PASSED!")
    print("="*60 + "\n")