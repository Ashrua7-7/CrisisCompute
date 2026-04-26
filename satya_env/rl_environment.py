"""
Modified Environment that makes RL agents' actions actually matter
Wraps the real environment to provide meaningful learning signals
"""

from satya_env.env import RealEnvironment


class RLFriendlyEnvironment(RealEnvironment):
    """
    Wrapper around RealEnvironment that:
    1. Respects agent resource requests (doesn't override with max)
    2. Penalizes insufficient resources (tasks fail)
    3. Penalizes resource waste (overly aggressive requests)
    4. Creates meaningful learning signal
    """
    
    def __init__(self, config_dir=None, seed=None):
        super().__init__(config_dir=config_dir, seed=seed)
        self.episode_index = 0
        self.resource_efficiency = {}  # Track efficiency per agent
    
    def reset(self):
        """Reset for new episode"""
        self.episode_index += 1
        self.resource_efficiency = {agent_id: 0.0 for agent_id in self.agent_order}
        return super().reset()
    
    def step(self, actions):
        """
        Modified step that:
        1. Uses original step
        2. Adjusts rewards based on resource efficiency
        """
        
        # Get original observations, rewards, done, info
        observations, rewards, done, info = super().step(actions)
        
        # Modify rewards based on agent action quality
        adjusted_rewards = list(rewards)
        
        for i, agent_id in enumerate(self.agent_order):
            action = actions.get(agent_id, {})
            
            # If agent took a run_task action, evaluate resource efficiency
            if action.get("action") == "run_task":
                task_id = action.get("task_id")
                task = self.tasks.get(task_id)
                
                if task:
                    requested_cpu = action.get("cores_needed", task.cores_needed)
                    requested_gpu = action.get("gpu_needed", task.gpu_needed)
                    requested_mem = action.get("memory_needed", task.memory_needed)
                    
                    # Calculate efficiency: how well-matched are the requests?
                    efficiency = self._calculate_efficiency(
                        task,
                        requested_cpu,
                        requested_gpu,
                        requested_mem
                    )
                    
                    # Adjust reward based on efficiency.
                    # Bonuses amplified 3× vs original — with constrained CPU
                    # (cpu=8), over-requesting blocks other agents causing
                    # deadline misses. The amplified signal ensures Q-learning
                    # reliably discovers the "use minimal resources" policy.
                    if efficiency > 0.9:  # Great match (90-100%)
                        adjusted_rewards[i] += 15.0
                    elif efficiency > 0.7:  # Good match (70-90%)
                        adjusted_rewards[i] += 6.0
                    elif efficiency > 0.5:  # Okay (50-70%)
                        adjusted_rewards[i] += 0.0  # Neutral
                    elif efficiency > 0.3:  # Wasteful (30-50%)
                        adjusted_rewards[i] -= 6.0
                    else:  # Very wasteful (< 30%)
                        adjusted_rewards[i] -= 15.0
                    
                    self.resource_efficiency[agent_id] = efficiency
        
        # Also add small team reward bonus for good resource utilization
        avg_efficiency = sum(self.resource_efficiency.values()) / len(self.resource_efficiency)
        team_bonus = avg_efficiency * 3.0  # Up to +3 points per agent
        
        adjusted_rewards = [r + team_bonus/3 for r in adjusted_rewards]
        
        return observations, adjusted_rewards, done, info
    
    def _calculate_efficiency(self, task, req_cpu, req_gpu, req_mem):
        """
        Calculate how well-matched the resource request is to task needs.
        Returns 0.0 to 1.0, where 1.0 is perfect match.
        
        - Perfect match (100%): request == actual need
        - Overkill: request >> need (wasted resources)
        - Underkill: request < need (will fail or be inefficient)
        """
        task_cpu = task.cores_needed
        task_gpu = task.gpu_needed
        task_mem = task.memory_needed
        
        # Efficiency for each resource dimension
        cpu_eff = self._match_efficiency(req_cpu, task_cpu)
        gpu_eff = self._match_efficiency(req_gpu, task_gpu)
        mem_eff = self._match_efficiency(req_mem, task_mem)
        
        # Average efficiency across all resources
        overall_eff = (cpu_eff + gpu_eff + mem_eff) / 3.0
        
        return max(0.0, overall_eff)
    
    def _match_efficiency(self, requested, required):
        """
        Calculate efficiency of one resource dimension.
        
        Perfect: requested == required (1.0)
        Undershooting: requested < required (penalize heavily)
        Overshooting: requested > required (penalize lightly)
        """
        if required == 0:
            # Task doesn't need this resource
            # Requesting any is waste
            return 1.0 if requested == 0 else max(0.0, 1.0 - (requested * 0.2))
        
        ratio = requested / required
        
        if ratio < 1.0:
            # Undershooting: very bad, task will fail/be inefficient
            # ratio 0.9 -> eff 0.7
            # ratio 0.5 -> eff 0.0
            # ratio < 0.5 -> eff < 0.0 (clamped to 0)
            return max(0.0, 2.0 * ratio - 1.0)
        elif ratio == 1.0:
            # Perfect match
            return 1.0
        else:
            # Overshooting: mildly bad (wasted resources)
            # ratio 1.5 -> eff 0.75
            # ratio 2.0 -> eff 0.5
            # ratio 3.0 -> eff 0.25
            return 1.0 / ratio  # Diminishing returns on overkill
