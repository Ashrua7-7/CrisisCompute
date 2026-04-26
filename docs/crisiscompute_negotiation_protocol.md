# CrisisCompute Negotiation Protocol

This document describes the diplomatic negotiation layer added to the environment for full Theme #1 alignment (cooperation, competition, negotiation, coalition formation, and Theory-of-Mind).

## Why this protocol exists

The previous scheduler resolved conflicts purely by urgency sorting. This protocol adds explicit social interaction before allocation:

- Urgency-first emergency governance
- Market bargaining with future obligations
- Reputation economy with promise tracking
- Coalition politics during scarcity
- Belief modeling (Theory-of-Mind) that can be measured over training

## Hourly state machine

Each environment hour executes in this order:

1. **Intent Broadcast**
   - Every agent submits intended task and demanded `cpu`, `gpu`, `memory`.
   - Agents can include `urgency_claim` in actions.

2. **Emergency Charter Check**
   - If any runnable task is in critical window (`deadline - current_hour <= 1`), charter activates.
   - Crisis agents get urgency-safe priority in negotiation scoring.

3. **Market Bargaining**
   - Agents with lower negotiation score can yield in current hour.
   - Yield behavior is influenced by predicted competitor behavior (belief-driven).
   - Winners and losers are chosen under resource constraints.

4. **Coalition Contracting**
   - If conflict persists, loser->winner yield contracts may be created.
   - Contract schema:
     - `creditor`: yielding agent now
     - `debtor`: favored agent now
     - `resource`: constrained resource (e.g., GPU)
     - `expires_at_hour`: deadline for reciprocation

5. **Promise Settlement**
   - Open contracts are checked each hour.
   - Contract kept: trust/reputation improves.
   - Contract broken: trust/reputation decreases.

6. **Scheduler Allocation**
   - Negotiated actions are passed to existing allocator.
   - Final run/wait decisions execute task progress.

7. **Belief + Reputation Updates**
   - Beliefs are updated from observed demand/action.
   - Reputation is adjusted by contract behavior.

## Theory-of-Mind model

Each agent maintains beliefs about every other agent:

- `predicted_cpu_demand`
- `predicted_gpu_demand`
- `predicted_memory_demand`
- `predicted_yield_probability`
- `reliability_estimate`

Belief accuracy is scored each hour from prediction error vs actual demands:

- Reported in `info["negotiation"]["belief_accuracy"]`
- Higher value means stronger opponent modeling

## Observation contract additions

Each agent observation now includes:

- `diplomacy.my_reputation`
- `diplomacy.peer_beliefs`
- `diplomacy.latest_negotiation`

This makes negotiation state directly learnable by RL and consumable by LLM policies.

## Reward alignment

Team reward now includes diplomatic components:

- Fairness bonus from negotiation outcome
- Emergency-charter compliance bonus
- Broken-contract penalty

This keeps objective aligned with:

1. Urgency survival
2. Coordination stability
3. Anti-monopoly fairness
4. Throughput reward

## Theme #4 forward compatibility

The protocol is structured so self-improvement can be added without rewriting the core environment:

- Belief update coefficients can be curriculum-controlled.
- Contract penalties can be adaptively tuned by a self-play teacher.
- Crisis frequency can be automatically escalated by a scenario generator.
- Coalition complexity can be increased over training phases.

This enables recursive curriculum and self-play growth on top of the same environment API.
