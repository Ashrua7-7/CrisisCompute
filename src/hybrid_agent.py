class HybridAgent:
    """
    Hybrid Agent: LLM provides a strategic hint, RL makes the final decision.

    FIX: Previously the LLM output was silently thrown away (_).
    Now LLM's action is converted to a strategy string ("request_gpu",
    "run_aggressive", "run_standard", "wait") that biases the RL policy
    via the epsilon-greedy action selection.
    """

    def __init__(self, llm_agent, rl_agent):
        self.llm = llm_agent
        self.rl  = rl_agent
        self.name = rl_agent.name

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _llm_to_strategy(self, llm_action: dict) -> str | None:
        """Convert an LLM action dict into a strategy hint for the RL agent."""
        if not llm_action or llm_action.get("action") == "wait":
            return "wait"

        cores   = llm_action.get("cores_needed", 0) or 0
        gpu     = llm_action.get("gpu_needed",   0) or 0

        if gpu > 0:
            return "request_gpu"
        if cores >= 6:
            return "run_aggressive"
        if cores >= 4:
            return "run_standard"
        return "run_minimal"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, state: dict) -> dict:
        """
        1. Ask LLM for its action.
        2. Convert to strategy hint.
        3. Pass hint to RL so it can bias exploration.
        """
        llm_action = self.llm.propose_action(state)
        strategy   = self._llm_to_strategy(llm_action)

        # RL agent uses strategy to bias its epsilon-greedy selection
        return self.rl.propose_action(state, strategy=strategy)

    def learn(self, state, action, reward, next_state):
        """Only the RL side learns from rewards."""
        self.rl.receive_reward(reward, next_state)

    def reset_for_episode(self):
        if hasattr(self.rl, "reset_for_episode"):
            self.rl.reset_for_episode()