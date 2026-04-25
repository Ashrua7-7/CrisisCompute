# CrisisCompute: Multi-Agent Negotiation with Self-Improvement

OpenEnv environment where three agents (`data_loader`, `data_cleaner`, `ml_trainer`) negotiate for shared compute under partial observability, then improve via RL, league self-play, and adaptive curriculum.

This project targets:
- Theme #1: Multi-Agent Interactions
- Theme #4: Self-Improvement

## What We Built

- **Negotiation world**: agents compete/cooperate for CPU, GPU, and memory under deadlines.
- **Diplomacy mechanics**: intent broadcast, bargaining, coalition contracts, and reputation updates.
- **Theory-of-mind signals**: each agent tracks peer demand/yield/reliability beliefs.
- **RL-friendly rewards**: fairness, contract compliance, deadline penalties, and throughput are all encoded.
- **Theme #4 loop**: adaptive curriculum + scenario auto-generation + policy snapshot league + cross-phase duel evaluation + holdout evaluation.

## Theme #4 Implementation

### 1) Adaptive Curriculum
Training is broken into phases. Each phase promotes to harder difficulty only when phase metrics cross thresholds.

Examples of escalation:
- No crisis -> GPU outage
- Single disruption -> compound disruption
- Later urgent injections + tighter conflict windows

### 2) Self-Play League + League Duels
After each phase, we snapshot policy state (Q-table state counts + policy snapshots + metrics summary), then run duels against prior-phase opponents:

- one learner agent is trained at a time
- other agents are frozen to historical league snapshot policies
- duels run on harder scenarios than current phase

Output file:
- `results/selfplay_report.json`

### 3) Auto Challenge Generation
For each curriculum level, a challenge plan is generated and cycled per episode:
- `stable_market`
- `light_gpu_outage`
- `urgent_injection`
- `gpu_outage_plus_urgent`
- `compound_crisis_a`, `compound_crisis_b`

### 4) Holdout Generalization
After training, we evaluate:
- trained policy on unseen compound-crisis holdout
- fresh policy (no loaded Q-table) on same holdout

Output file:
- `results/holdout_evaluation.json`

## Run Training

PowerShell example:

```powershell
$env:TRAINING_AGENT_MODE='rl'
$env:SELF_IMPROVEMENT_ENABLED='true'
$env:CURRICULUM_PHASE_EPISODES='10'
$env:MULTI_SEED='42,123'
$env:NUM_EPISODES='40'
python ".\train.py"
```

### Optional reset for clean run

```powershell
Remove-Item ".\q_tables\*.json" -ErrorAction SilentlyContinue
Remove-Item ".\results\*.json" -ErrorAction SilentlyContinue
```

## Key Artifacts for Judges

Training outputs:
- `results/training_results.json`
- `results/episode_metrics.json`
- `results/negotiation_trace.json`
- `results/baseline_comparison.json`
- `results/judge_summary.json`

Theme #4 outputs:
- `results/selfplay_report.json`
- `results/holdout_evaluation.json`

Plots:
- `results/reward_curve.png`
- `results/metrics_dashboard.png`

## How To Explain Improvement Correctly

Because curriculum increases difficulty over time, naive `episode_1 vs episode_N` reward can dip even when policy improves.

Use these instead:
- rolling-window reward trend
- per-curriculum-level reward means
- holdout trained-vs-fresh delta
- completion/on-time/fairness/belief metrics

## OpenEnv Compliance

- Uses OpenEnv environment contract (`reset`, `step`, `state`-style flow).
- Includes `openenv.yaml`.
- Supports multi-agent joint action/observation.
- Can be packaged for Hugging Face Space hosting.

## Submission Checklist

- [ ] Hugging Face Space URL for environment
- [x] Colab script using TRL (`notebooks/openenv_trl_colab.ipynb`)
- [ ] Reward/metrics plots committed in repo
- [ ] Short demo video (<2 min) or HF mini-blog
- [ ] README links to all artifacts and demo resources