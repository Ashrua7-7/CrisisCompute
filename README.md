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

**OpenEnv Hackathon India 2026**  
**Themes:** Multi-Agent Interactions (Theme #1) + Self-Improvement (Theme #4)

---

## Quick Links

| | |
|--|--|
| 🤗 HuggingFace Space (live environment) | [gautam0898-crisiscompute.hf.space](https://gautam0898-crisiscompute.hf.space) |
| 📓 Colab Training Notebook | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb) |
| 📝 Mini-Blog on HuggingFace | [CrisisCompute: Teaching AI Agents to Negotiate Under Pressure](https://huggingface.co/Gautam0898/crisiscompute-blog) |

---

## 1. Problem: What Capability Gap Are We Targeting?

Modern AI agents can solve isolated tasks but fall apart when multiple agents must **share scarce compute resources** under deadline pressure with only partially aligned incentives.

**CrisisCompute** simulates a real-world scenario: three specialist ML pipeline agents — `data_loader`, `data_cleaner`, and `ml_trainer` — must cooperate to complete a batch of jobs on a constrained cluster (16 CPU cores, 32 GB RAM, 1 GPU) while handling mid-episode crises like GPU outages, urgent task injections, and cascading resource conflicts.

This targets the underexplored frontier of **strategic multi-agent negotiation under resource scarcity** — a coordination problem that matters enormously as autonomous AI teams proliferate in enterprise and research compute settings.

---

## 2. Environment: What Does the Agent See, Do, and Get Rewarded For?

### Architecture

```
┌─────────────────────────────────────────────────┐
│          CrisisCompute (Compute Cluster)         │
│   Resources: 16 CPU  |  32 GB RAM  |  1 GPU     │
│   Tasks: deadline-sensitive, priority-weighted   │
│   Events: GPU outage, urgent injection, conflict │
└─────────────────────────────────────────────────┘
         ↕                    ↕                ↕
   data_loader    ↔    data_cleaner   ↔  ml_trainer
         └─────── Negotiation Protocol ──────────┘
              Reputation + Belief System (ToM)
```

### Observation Space (per agent)
- Task backlog, deadlines, and urgency scores
- Current resource availability (CPU / GPU / memory)
- Time-pressure and progress signal
- Negotiation history and peer reliability estimates (Theory of Mind)

### Action Space
Each agent selects one of:
- `request_resource` — claim CPU, memory, and optionally GPU
- `release_resource` — yield resources to reduce conflict
- `wait` — hold off for this timestep

### Reward Signal

The reward function uses OpenEnv's composable Rubric system — individual and team rewards blended at 60/40 to prevent both selfish dominance and free-riding:

| Signal | Direction | Purpose |
|--------|:---------:|---------|
| Task completion | ✅ | Finish assigned work |
| Early completion bonus | ✅ | Reward urgency-awareness |
| Deadline miss penalty | ❌ | Hard deadline enforcement |
| Waiting penalty | ❌ | Discourage passive agents |
| Team completion rate | ✅ | Cooperative health |
| Team on-time rate | ✅ | System-level throughput |

### Why It's Hard to Game

An agent that hogs all resources completes its own tasks but tanks the team penalty. An agent that always yields never finishes its own work. The only winning strategy is **principled negotiation** — which is exactly what we train.

---

## 3. Results: What Changed After Training?

### Training Setup
- **3 agent modes:** Pure RL (Q-learning), Pure LLM (Llama-3.1-8B via HuggingFace Inference), Hybrid (LLM strategy hint + RL Q-table update)
- **Adaptive curriculum** across 5 difficulty levels: negotiation off → on, crisis events off → on
- **Self-play snapshots** (Theme #4): current policy evaluated against past snapshots each phase

### Reward Curve

![Reward Curve](results/reward_curve.png)

*Episodic reward over the training run. The upward trend confirms agents improve through the adaptive curriculum.*

### Metrics Dashboard

![Metrics Dashboard](results/metrics_dashboard.png)

*Per-episode breakdown: completion rate, on-time rate, fairness score, and belief accuracy across the full training run.*

### Holdout Evaluation: Trained vs Fresh Agents

Trained agents tested on **compound crisis scenarios** (GPU outage + urgent task injection simultaneously) that were **never seen during training**:

| Metric | Trained | Fresh (untrained) | Improvement |
|--------|:-------:|:-----------------:|:-----------:|
| Avg Total Reward | **684.6** | 620.8 | **+63.8** ✅ |
| Task Completion Rate | **75.0%** | 71.7% | **+3.3%** ✅ |
| On-Time Rate | **51.7%** | 44.2% | **+7.5%** ✅ |
| Fairness Score | 1.0 | 1.0 | Maintained ✅ |

Trained agents generalise to unseen crisis combinations — the core signal of genuine learning.

### Theme #1: Negotiation Builds Better World Models

Agents with the negotiation protocol active develop significantly better Theory of Mind:

- Belief accuracy **with** negotiation: **0.566**
- Belief accuracy **without** negotiation: **0.538**
- **Delta: +0.028** — agents that negotiate learn to model peer behaviour more accurately

### Theme #4: Adaptive Curriculum Progression

| Level | Conditions |
|-------|-----------|
| 0 | No negotiation, no crises, easy tasks |
| 1 | Negotiation enabled, baseline resources |
| 2 | Mild crises, resource pressure begins |
| 3 | GPU outage events, urgent task injections |
| 4 | Compound crises, cascading conflicts, full negotiation |

Promotion between levels requires meeting thresholds on completion %, fairness, and belief accuracy — agents cannot skip difficulty.

---

## 4. Why It Matters

- **AI ops teams** managing shared cloud infrastructure
- **Multi-agent LLM pipelines** (AutoGen, CrewAI, LangGraph) where resource conflicts currently go unmanaged
- **Research** into fair coordination under scarcity — a problem with no good benchmark yet

Training on CrisisCompute teaches agents to negotiate, yield, and cooperate under genuine competitive pressure. This is a capability gap that frontier LLMs still lack out of the box.

---

## How to Run

### Quickest (RL mode, no API key needed)
```bash
pip install -r requirements.txt
python train.py
```
Artifacts are written to `results/` — reward curves, episode metrics, holdout comparison, negotiation traces.

### LLM / Hybrid mode
Copy `.env.example` to `.env` and fill in:
```
LLM_PROVIDER=huggingface
HF_TOKEN=hf_...
TRAINING_AGENT_MODE=hybrid
```

### Colab (recommended for judges)
Open the notebook on a free **T4 GPU** — it runs the complete pipeline:  
RL training → Hybrid training → Unsloth SFT fine-tuning → GRPO training → all plots saved.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb)

---

## Repo Structure

```
CrisisCompute/
├── satya_env/               # Core OpenEnv environment
│   ├── env.py               # Episode logic, resource pool, crisis events
│   ├── reward.py            # Composable reward rubric
│   ├── negotiation.py       # Negotiation protocol + belief/reputation system
│   └── rl_environment.py    # Gym-style wrapper for RL training
├── server/                  # OpenEnv MCP server (Docker / HF Space)
├── src/                     # Agent implementations (RL, LLM, Hybrid)
├── notebooks/
│   └── training_colab.ipynb # Full training notebook (Unsloth SFT + GRPO)
├── results/                 # Committed training artifacts (plots + JSON)
├── docs/                    # Protocol specs, agent personalities, LLM prompts
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # HuggingFace Spaces deployment
└── train.py                 # Main training script
```

---

## OpenEnv Compliance

- Built on `openenv-core` base classes with standard `reset / step / state` API
- Valid `openenv.yaml` manifest with typed action and observation schemas
- Clean client/server separation — `server/` never imports `satya_env/` internals
- No reserved tool name conflicts (`reset`, `step`, `state`, `close` are not used as MCP tools)
- Deployed to HuggingFace Spaces via Docker SDK on port 7860

---

*CrisisCompute — OpenEnv Hackathon India 2026*
