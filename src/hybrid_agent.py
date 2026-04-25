"""
HybridAgent — proper LLM ↔ RL integration.

Design:
    1. LLM proposes a high-level *strategy* hint per episode-state.
    2. RL agent owns the final action via epsilon-greedy that is *biased*
       toward the LLM hint when exploring.
    3. RL is the only learner — Q-table updates from environment rewards.
    4. LLM is called sparingly: results are cached per (discretized state)
       so we don't re-prompt the LLM 24 times per episode and trip rate
       limits. Cache is reset every episode so hints stay fresh.
    5. If the LLM call fails (rate limit, network, parse error, etc.) we
       pass strategy=None to the RL agent. Earlier code converted the
       fallback `{"action": "wait"}` payload into a `"wait"` strategy
       which silently froze every agent — that was the real integration
       bug, not the rate limit itself.
"""


class HybridAgent:
    """LLM provides strategy bias; RL provides the policy + learning signal."""

    # Marker reasoning string that LLMAgent uses when it falls back to wait
    # because the LLM call returned nothing (rate limit, timeout, parse fail).
    _LLM_FALLBACK_MARKERS = ("LLM unavailable", "LLM failed", "fallback")

    def __init__(self, llm_agent, rl_agent):
        self.llm = llm_agent
        self.rl = rl_agent
        self.name = rl_agent.name
        # Per-episode LLM-hint cache keyed by RL-discretized state. Avoids
        # repeated prompts for identical situations within an episode.
        self._strategy_cache: dict = {}
        # Counters for end-of-run telemetry so we can prove integration is live.
        self.llm_hints_used = 0
        self.llm_cache_hits = 0
        self.llm_fallbacks = 0

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_llm_fallback(self, llm_action: dict | None) -> bool:
        """Detect the fallback wait-action emitted when the LLM call failed."""
        if not isinstance(llm_action, dict):
            return True
        reasoning = str(llm_action.get("reasoning", "")).lower()
        if any(marker.lower() in reasoning for marker in self._LLM_FALLBACK_MARKERS):
            return True
        return False

    def _llm_to_strategy(self, llm_action: dict) -> str | None:
        """Convert an LLM action dict into a strategy hint for the RL agent.

        Returns None when the LLM provided no useful guidance — that lets the
        RL agent fall back to its pure epsilon-greedy policy instead of being
        biased toward `wait`.
        """
        if self._is_llm_fallback(llm_action):
            return None

        action_type = str(llm_action.get("action", "")).lower()
        if action_type == "wait":
            # Real LLM-chosen wait (not a fallback) — honour it but only as a
            # weak hint; RL still gets to override via Q-values.
            return "wait"

        cores = int(llm_action.get("cores_needed", 0) or 0)
        gpu = int(llm_action.get("gpu_needed", 0) or 0)

        if gpu > 0:
            return "request_gpu"
        if cores >= 6:
            return "run_aggressive"
        if cores >= 4:
            return "run_standard"
        return "run_minimal"

    def _state_key(self, state):
        """Use the RL agent's own discretization for cache locality."""
        try:
            return str(self.rl.discretize_state(state))
        except Exception:
            # Fall back to raw object id; cache still works for repeated dicts.
            return id(state)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def act(self, state: dict) -> dict:
        """Ask LLM for a strategic hint, then let RL pick the actual action."""
        cache_key = self._state_key(state)
        if cache_key in self._strategy_cache:
            strategy = self._strategy_cache[cache_key]
            self.llm_cache_hits += 1
        else:
            llm_action = self.llm.propose_action(state)
            strategy = self._llm_to_strategy(llm_action)
            self._strategy_cache[cache_key] = strategy
            if strategy is None:
                self.llm_fallbacks += 1
            else:
                self.llm_hints_used += 1

        return self.rl.propose_action(state, strategy=strategy)

    def learn(self, state, action, reward, next_state):
        """Only the RL side learns — LLM is treated as a fixed advisor."""
        self.rl.receive_reward(reward, next_state)

    def reset_for_episode(self):
        # Flush per-episode strategy cache so a fresh exploration cycle starts.
        self._strategy_cache.clear()
        if hasattr(self.rl, "reset_for_episode"):
            self.rl.reset_for_episode()

    # ------------------------------------------------------------------
    # Introspection (used by training summary)
    # ------------------------------------------------------------------

    def integration_stats(self) -> dict:
        return {
            "name": self.name,
            "llm_hints_used": int(self.llm_hints_used),
            "llm_cache_hits": int(self.llm_cache_hits),
            "llm_fallbacks": int(self.llm_fallbacks),
            "epsilon": float(getattr(self.rl, "epsilon", 0.0)),
            "q_table_states": len(getattr(self.rl, "q_table", {})),
        }
