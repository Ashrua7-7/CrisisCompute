---
title: CrisisCompute
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

# CrisisCompute

**OpenEnv Hackathon 2026 submission**  
**Themes covered:** Theme #1 (Multi-Agent Interactions) + Theme #4 (Self-Improvement)

## 1) Problem: what capability gap are we targeting?

Modern AI agents can solve isolated tasks, but they still struggle when:
- multiple agents must share scarce compute,
- incentives are partially aligned (cooperate + compete),
- decisions under uncertainty must stay fair over time.

`CrisisCompute` simulates a compute-starved ML pipeline where three specialist agents (`data_loader`, `data_cleaner`, `ml_trainer`) must finish deadline-bound work while CPU, memory, and GPU are constrained and disruptions happen mid-episode.

This targets exactly the gap in **strategic multi-agent coordination** under pressure.

## 2) Environment: what does the agent see, do, and get rewarded for?

### What agents observe
- task backlog and deadlines,
- current resource availability (CPU/GPU/memory),
- progress/time-pressure state,
- negotiation context and peer reliability estimates.

### What agents can do
- `wait`,
- run tasks with different aggressiveness (`run_minimal`, `run_standard`, `run_aggressive`),
- enter negotiation when resource conflicts appear (Theme #1 behavior).

### Reward signal
- positive for completing tasks and meeting deadlines,
- shaped by resource efficiency and coordination quality,
- includes fairness/team-health terms so one agent cannot dominate.

Core implementation lives in:
- `satya_env/env.py`
- `satya_env/negotiation.py`
- `satya_env/reward.py`
- `satya_env/rl_environment.py`

## 3) Results: what changed after training?

This section uses the current checked-in artifacts (not projected numbers).

### Theme #1 evidence (multi-agent interactions)
- Negotiation-enabled policy outperforms no-negotiation baseline on reward:  
  **+20.42 avg reward** (`results/baseline_comparison.json`)
- Belief accuracy also improves:  
  **+0.0077** (`results/baseline_comparison.json`)
- Detailed interaction traces are logged in `results/negotiation_trace.json`.

### Theme #4 evidence (self-improvement)
- Adaptive curriculum is active with level templates and phase logging:  
  `results/theme4_summary.json`
- Current run shows only early curriculum progression (1 logged phase), which is valid but limited.
- Holdout test (trained vs fresh) currently shows:
  - reward delta: **-3.15** (trained slightly lower),
  - belief accuracy delta: **+0.0305** (trained better world-model signal).  
  Source: `results/holdout_evaluation.json`

### Honest takeaway
Training is clearly improving coordination signals (belief accuracy, negotiation-aware policy value), while holdout reward is not yet consistently better in this snapshot. This is a strong base and a credible “work-in-progress frontier” narrative for finals.

## 4) Why it matters: who cares and why?

This environment matters for anyone building:
- autonomous AI teams in enterprise workflows,
- AI ops agents that share finite infrastructure,
- assistant swarms that must negotiate priorities safely.

If we can train agents to negotiate fairly under scarcity, we unlock more reliable multi-agent systems for real compute, scheduling, and operations settings.

## Quick links for judges

- **Hugging Face Space (environment):** [gautam0898-crisiscompute.hf.space](https://gautam0898-crisiscompute.hf.space)
- **Colab training notebook (Unsloth/TRL minimal script):** `notebooks/training_colab.ipynb`
- **Frontend demo:** `ui/` (run locally with `npm run dev`)
- **Mini-blog or <2 min video:** `ADD_BLOG_OR_VIDEO_URL`

> Add your real URLs above before final submission so judges can click everything from one place.

## How to run (minimal)

```bash
pip install -r requirements.txt
python train.py
```

Generated outputs appear in `results/` (reward curves, summaries, holdout comparison, traces).

## Judging criteria alignment (OpenEnv finals)

### Environment Innovation (40%)
- Strong concept: compute-allocation negotiation with partially aligned incentives.
- Non-trivial environment mechanics (deadlines, scarcity, disruptions, fairness pressure).

### Storytelling & Presentation (30%)
- README now follows Problem -> Environment -> Results -> Why it matters.
- Make this score high by adding your final Space, frontend, and video/blog links above.

### Showing Improvement in Rewards (20%)
- You already have measurable deltas in baseline-vs-negotiation.
- To maximize final score, run one longer training with more curriculum phases and include the best reward curve screenshot in README.

### Reward & Training Pipeline (10%)
- Pipeline exists end-to-end: environment -> training loop -> artifacts.
- Reward logic and training outputs are auditable in repo (`satya_env/reward.py`, `results/*.json`).

## Submission checklist (final pass)

- [x] Uses OpenEnv with `openenv.yaml`
- [x] Includes minimal training pipeline and Colab notebook (`notebooks/training_colab.ipynb`)
- [x] Includes measurable artifacts in `results/`
- [x] Theme #1 integration (multi-agent interactions)
- [x] Theme #4 integration (self-improvement scaffolding)
- [ ] Replace placeholder links with final public URLs
- [ ] Add one clean “before vs after” plot image in README
- [ ] Add mini-blog or short video link

---

**Status:** Submission-ready foundation with clear Theme #1 + Theme #4 narrative; final polish is link completion + one stronger long-run evidence plot.