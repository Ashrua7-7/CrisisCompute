# src/rl_agent.py
# RL-based agents that learn from rewards using Q-learning

import json
from src.agents import Agent
import random


class RLAgent(Agent):
    """
    Base RL Agent - uses Q-learning to improve policy over episodes
    """
    
    def __init__(self, name, resource_needs, learning_rate=0.2, discount_factor=0.95, epsilon_start=0.7):
        super().__init__(name, resource_needs)
        
        # Initialize episode counter
        self.episode = 0
        
        # RL hyperparameters - AGGRESSIVE LEARNING for faster convergence
        self.learning_rate = learning_rate  # Higher: 0.2 for faster updates
        self.discount_factor = discount_factor  # Higher: 0.95 for long-term value
        self.epsilon_start = epsilon_start  # Higher: 0.7 for more exploration
        self.epsilon = epsilon_start
        self.epsilon_decay = 0.93  # Slower decay: 0.93 to maintain exploration longer
        
        # Q-table: state -> {action -> Q-value}
        self.q_table = {}
        
        # Experience tracking
        self.episode_history = []  # Experiences from current episode
        self.past_episodes = []  # History of all past episodes
        self.episode_rewards = []  # Reward per episode
        
        # State tracking
        self.current_state = None
        self.previous_state = None
        self.previous_action = None

        # Action-type histogram (parity with LLMAgent for cross-mode comparison)
        self.action_histogram: dict = {}
        
    def discretize_state(self, observation):
        """
        Convert continuous observation into discrete state for Q-learning
        IMPROVED: Much larger state space (15-20 combinations) for better learning
        Returns a hashable state representation
        """
        my_tasks = observation.get("my_tasks", {})
        pending = len(my_tasks.get("pending", []))
        running = len(my_tasks.get("running", []))
        done = len(my_tasks.get("done", []))
        
        available_resources = observation.get("available_resources", {})
        cpu_available = available_resources.get("cpu_cores", {}).get("available", 0)
        gpu_available = available_resources.get("gpu", {}).get("available", 0)
        memory_available = available_resources.get("memory_gb", {}).get("available", 0)
        
        time_left = observation.get("time_left_hours", 8)
        
        # GRANULAR STATE DISCRETIZATION for better learning
        # Task load: 0-1, 2-3, 4+
        pending_bucket = "empty" if pending == 0 else "few" if pending <= 2 else "many"
        running_bucket = "none" if running == 0 else "some" if running <= 1 else "busy"
        
        # CPU availability: sparse (0-4), limited (4-8), abundant (8+)
        cpu_bucket = "sparse" if cpu_available < 4 else "limited" if cpu_available < 8 else "abundant"
        
        # Memory: low (<8), medium (8-16), high (16+)
        memory_bucket = "low" if memory_available < 8 else "medium" if memory_available < 16 else "high"
        
        # Time pressure: urgent (<2h), moderate (2-4h), relaxed (4h+)
        time_bucket = "urgent" if time_left < 2 else "moderate" if time_left < 4 else "relaxed"
        
        # Progress: stuck (few done), progressing (some done), advanced (many done)
        total_tasks = pending + running + done
        progress = 0 if total_tasks == 0 else done / total_tasks
        progress_bucket = "stuck" if progress < 0.3 else "progressing" if progress < 0.7 else "advanced"
        
        state = (pending_bucket, running_bucket, cpu_bucket, memory_bucket, time_bucket, progress_bucket)
        return state
    
    def get_q_value(self, state, action):
        """Get Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {}
        if action not in self.q_table[state]:
            self.q_table[state][action] = 0.0
        return self.q_table[state][action]
    
    def set_q_value(self, state, action, value):
        """Set Q-value for state-action pair"""
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = value
    
    def update_q_value(self, state, action, reward, next_state, done):
        """Update Q-value using Q-learning formula"""
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state
        if next_state not in self.q_table or not self.q_table[next_state]:
            max_next_q = 0
        else:
            max_next_q = max(self.q_table[next_state].values())
        
        # Q-learning formula: Q(s,a) = Q(s,a) + lr * (r + gamma * max Q(s',a') - Q(s,a))
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        self.set_q_value(state, action, new_q)
    
    def select_action(self, state, available_actions, strategy=None):
        """
        Select action using epsilon-greedy strategy.
        If strategy hint is provided (from Hybrid/LLM), prefer that action type
        when exploring so LLM guidance actually shapes behaviour.

        available_actions: list of (action_type, task_id) tuples
        strategy: optional string hint from LLM ("run_aggressive", "run_standard",
                  "run_minimal", "request_gpu", "wait")
        """
        action_types = list(set(a[0] for a in available_actions))
        strategy = self._normalize_strategy_hint(strategy, action_types)

        if random.random() < self.epsilon:
            # Explore — but if LLM gave a valid hint, follow it ~60% of the time
            if strategy and strategy in action_types and random.random() < 0.6:
                action_type = strategy
            else:
                action_type = random.choice(action_types)
        else:
            # Exploit: pick the action_type with the highest Q-value
            if state in self.q_table and self.q_table[state]:
                best_q = max(self.get_q_value(state, a) for a in action_types)
                action_type = random.choice([
                    a for a in action_types
                    if self.get_q_value(state, a) == best_q
                ])
            else:
                action_type = random.choice(action_types)

        matching = [a for a in available_actions if a[0] == action_type]
        return random.choice(matching) if matching else available_actions[0]

    def _normalize_strategy_hint(self, strategy, action_types):
        """
        Map unsupported hint types to nearest valid action.
        Keeps hybrid mode robust when hint vocab drifts.
        """
        if not strategy:
            return None
        if strategy in action_types:
            return strategy
        aliases = {
            "request_resource": "run_standard",
            "run_task": "run_standard",
            "run_fast": "run_aggressive",
            "run_safe": "run_minimal",
            "defer": "wait",
            "idle": "wait",
            "request_gpu": "run_aggressive",
        }
        mapped = aliases.get(str(strategy), None)
        if mapped in action_types:
            return mapped
        if "run_standard" in action_types:
            return "run_standard"
        if "run_minimal" in action_types:
            return "run_minimal"
        if "wait" in action_types:
            return "wait"
        return action_types[0] if action_types else None
    def reset_for_episode(self):
        """Reset for new episode"""
        self.episode += 1
        self.episode_history = []
        self.total_reward = 0
        
        # Decay exploration rate
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)
    
    def take_action(self, state, action, observation):
        """
        Record state and action taken. Return action details for environment.
        """
        self.previous_state = state
        self.previous_action = action
        self.current_state = state
        # Store observation for reward calculation
        self.last_observation = observation
        return action
    
    def receive_reward(self, reward, next_observation=None):
        """Receive reward and prepare for Q-learning update"""
        self.total_reward += reward
        
        if next_observation is None:
            next_observation = getattr(self, "last_observation", None)

        if next_observation is None:
            return

        # Discretize the next state from observation
        next_state = self.discretize_state(next_observation)
        
        # Store in episode history for batch learning
        if self.previous_state is not None and self.previous_action is not None:
            self.episode_history.append({
                "state": self.previous_state,
                "action": self.previous_action,
                "reward": reward,
                "next_state": next_state,
                "done": False
            })
    
    def learn_from_episode(self):
        """Learn from collected experiences at end of episode"""
        # Update Q-values from collected experiences
        for exp in self.episode_history:
            self.update_q_value(
                exp["state"],
                exp["action"],
                exp["reward"],
                exp["next_state"],
                done=False
            )
        
        self.past_episodes.append({
            "episode": self.episode,
            "total_reward": self.total_reward,
            "epsilon": self.epsilon,
            "history": self.episode_history
        })
        self.episode_rewards.append(self.total_reward)
    
    def save_q_table(self, filepath):
        """Save Q-table to JSON file for persistence across runs"""
        # Convert q_table to JSON-serializable format
        # (states are tuples, need to convert to strings)
        q_table_json = {}
        for state, actions in self.q_table.items():
            state_key = str(state)  # Convert tuple to string
            q_table_json[state_key] = actions
        
        with open(filepath, 'w') as f:
            json.dump({
                'q_table': q_table_json,
                'episode': self.episode,
                'epsilon': self.epsilon,
                'episode_rewards': self.episode_rewards
            }, f, indent=2)
    
    def load_q_table(self, filepath):
        """Load Q-table from JSON file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Convert JSON back to q_table format
            # (string keys back to tuples)
            self.q_table = {}
            for state_str, actions in data['q_table'].items():
                # Parse the string representation back to tuple
                state_tuple = eval(state_str)  # Safe here since we created it
                self.q_table[state_tuple] = actions
            
            self.episode = data.get('episode', 0)
            self.epsilon = data.get('epsilon', self.epsilon_start)
            self.episode_rewards = data.get('episode_rewards', [])
            
            print(f"   ✅ Loaded Q-table for {self.name}: {len(self.q_table)} states")
            return True
        except FileNotFoundError:
            return False


class RLDataLoaderAgent(RLAgent):
    """RL-based Data Loader that learns from rewards"""
    
    def __init__(self):
        super().__init__(
            name="rl_data_loader",
            resource_needs={"cpu": 2, "memory": 4, "gpu": 0},
            learning_rate=0.25,  # Aggressive learning
            discount_factor=0.95,  # Long-term value
            epsilon_start=0.7  # Lots of exploration
        )
    
    def propose_action(self, observation, strategy=None):
        """Propose action using RL policy. strategy: optional hint from LLM."""
        state = self.discretize_state(observation)
        self.previous_state = self.current_state
        self.current_state  = state

        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])

        available_actions = []
        if pending_tasks:
            for task_id in pending_tasks[:1]:
                available_actions.extend([
                    ("run_minimal",    task_id),
                    ("run_standard",   task_id),
                    ("run_aggressive", task_id),
                ])

        if not available_actions:
            available_actions = [("wait", None)]

        action_type, task_id = self.select_action(state, available_actions, strategy=strategy)
        self.previous_action = action_type
        self.action_histogram[action_type] = self.action_histogram.get(action_type, 0) + 1
        
        # Convert action to environment format
        if action_type == "wait":
            return {"action": "wait"}
        
        elif action_type == "run_minimal":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 2,
                "gpu_needed": 0,
                "memory_needed": 4,
                "estimated_duration_min": 40,
                "reasoning": f"Load (conservative) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_standard":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 4,
                "gpu_needed": 0,
                "memory_needed": 6,
                "estimated_duration_min": 30,
                "reasoning": f"Load (standard) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_aggressive":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 6,
                "gpu_needed": 0,
                "memory_needed": 8,
                "estimated_duration_min": 25,
                "reasoning": f"Load (aggressive) - Q[{self.get_q_value(state, action_type):.2f}]"
            }


class RLDataCleanerAgent(RLAgent):
    """RL-based Data Cleaner that learns from rewards"""
    
    def __init__(self):
        super().__init__(
            name="rl_data_cleaner",
            resource_needs={"cpu": 4, "memory": 8, "gpu": 0},
            learning_rate=0.25,  # Aggressive learning
            discount_factor=0.95,  # Long-term value
            epsilon_start=0.7  # Lots of exploration
        )
    
    def propose_action(self, observation, strategy=None):
        """Propose action using RL policy. strategy: optional hint from LLM."""
        state = self.discretize_state(observation)
        self.previous_state = self.current_state
        self.current_state  = state

        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])

        available_actions = []
        if pending_tasks:
            for task_id in pending_tasks[:1]:
                available_actions.extend([
                    ("run_minimal",    task_id),
                    ("run_standard",   task_id),
                    ("run_aggressive", task_id),
                ])

        if not available_actions:
            available_actions = [("wait", None)]

        action_type, task_id = self.select_action(state, available_actions, strategy=strategy)
        self.previous_action = action_type
        self.action_histogram[action_type] = self.action_histogram.get(action_type, 0) + 1
        
        if action_type == "wait":
            return {"action": "wait"}
        
        elif action_type == "run_minimal":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 4,
                "gpu_needed": 0,
                "memory_needed": 6,
                "estimated_duration_min": 60,
                "reasoning": f"Clean (conservative) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_standard":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 6,
                "gpu_needed": 0,
                "memory_needed": 8,
                "estimated_duration_min": 45,
                "reasoning": f"Clean (standard) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_aggressive":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 8,
                "gpu_needed": 0,
                "memory_needed": 10,
                "estimated_duration_min": 35,
                "reasoning": f"Clean (aggressive) - Q[{self.get_q_value(state, action_type):.2f}]"
            }


class RLMLTrainerAgent(RLAgent):
    """RL-based ML Trainer that learns from rewards"""
    
    def __init__(self):
        super().__init__(
            name="rl_ml_trainer",
            resource_needs={"cpu": 2, "memory": 16, "gpu": 1},
            learning_rate=0.25,  # Aggressive learning
            discount_factor=0.95,  # Long-term value
            epsilon_start=0.7  # Lots of exploration
        )
    
    def propose_action(self, observation, strategy=None):
        """Propose action using RL policy. strategy: optional hint from LLM."""
        state = self.discretize_state(observation)
        self.previous_state = self.current_state
        self.current_state  = state

        my_tasks = observation.get("my_tasks", {})
        pending_tasks = my_tasks.get("pending", [])

        available_actions = []
        if pending_tasks:
            for task_id in pending_tasks[:1]:
                available_actions.extend([
                    ("run_minimal",    task_id),
                    ("run_standard",   task_id),
                    ("run_aggressive", task_id),
                ])

        if not available_actions:
            available_actions = [("wait", None)]

        action_type, task_id = self.select_action(state, available_actions, strategy=strategy)
        self.previous_action = action_type
        self.action_histogram[action_type] = self.action_histogram.get(action_type, 0) + 1
        
        if action_type == "wait":
            return {"action": "wait"}
        
        elif action_type == "run_minimal":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 2,
                "gpu_needed": 1,
                "memory_needed": 12,
                "estimated_duration_min": 90,
                "reasoning": f"Train (conservative) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_standard":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 4,
                "gpu_needed": 1,
                "memory_needed": 16,
                "estimated_duration_min": 70,
                "reasoning": f"Train (standard) - Q[{self.get_q_value(state, action_type):.2f}]"
            }
        
        elif action_type == "run_aggressive":
            return {
                "action": "run_task",
                "task_id": task_id,
                "cores_needed": 6,
                "gpu_needed": 1,
                "memory_needed": 20,
                "estimated_duration_min": 55,
                "reasoning": f"Train (aggressive) - Q[{self.get_q_value(state, action_type):.2f}]"
            }