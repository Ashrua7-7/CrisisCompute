---
title: "CrisisCompute: Teaching AI Agents to Negotiate Under Pressure"
thumbnail: https://huggingface.co/spaces/Gautam0898/crisiscompute/resolve/main/results/rl_reward_curve.png
---

# CrisisCompute: Teaching AI Agents to Negotiate Under Pressure

**OpenEnv Hackathon India 2026** — Theme #1 (Multi-Agent Interactions) + Theme #4 (Self-Improvement)

> What happens when three AI agents must share a single GPU, race against deadlines, and survive mid-episode crises — all while learning to trust (or distrust) each other?

**CrisisCompute** is an OpenEnv-compliant multi-agent environment that simulates a shared compute cluster. Three specialist ML pipeline agents — **Data Loader**, **Data Cleaner**, and **ML Trainer** — must negotiate over scarce CPU, GPU, and memory to complete their tasks before deadlines expire. The catch: GPU outages strike mid-episode, urgent tasks get injected, and the only winning strategy is **principled negotiation**.

| | |
|---|---|
| Live Environment | [gautam0898-crisiscompute.hf.space](https://gautam0898-crisiscompute.hf.space) |
| GitHub Repository | [github.com/Ashrua7-7/CrisisCompute](https://github.com/Ashrua7-7/CrisisCompute) |
| Training Notebook | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb) |

---

## The Problem: Resource Scarcity Meets Competing Agents

Modern AI agents can solve isolated tasks, but they fall apart when multiple agents must **share scarce resources** under deadline pressure with only partially aligned incentives.

Imagine a shared ML cluster with:
- **16 CPU cores**, **32 GB RAM**, and **1 GPU**
- Three agents, each with their own task queue and deadlines
- Mid-episode crises: the GPU goes down, urgent tasks appear, resources get contested

This isn't hypothetical — it's the reality of shared compute in labs, enterprises, and cloud platforms. Yet there's no good benchmark for training agents to handle this.

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

---

## How It Works

### Observation Space
Each agent sees: its task backlog with deadlines, current resource availability (CPU/GPU/RAM), time pressure, negotiation history, and peer reliability estimates (Theory of Mind beliefs).

### Action Space
Every hour, each agent chooses one of:
- **request_resource** — claim CPU, memory, and optionally GPU for a task
- **release_resource** — yield resources to reduce conflict
- **wait** — hold off this timestep

### The Negotiation Protocol (Theme #1)
This is where CrisisCompute goes beyond standard multi-agent RL. Every hour follows a structured diplomatic cycle:

1. **Intent Broadcast** — agents declare what they want
2. **Emergency Charter Check** — critical-deadline tasks get priority
3. **Market Bargaining** — agents can yield, with behavior influenced by belief models
4. **Coalition Contracting** — yield contracts are created (I yield now, you yield later)
5. **Promise Settlement** — contracts are checked; trust/reputation adjusts
6. **Belief Updates** — Theory of Mind models update from observed behavior

Each agent maintains beliefs about peers: `predicted_cpu_demand`, `predicted_gpu_demand`, `predicted_yield_probability`, and `reliability_estimate`. Belief accuracy is measured every hour — agents that negotiate develop significantly better opponent models.

### Reward Design
The reward uses OpenEnv's composable rubric — individual and team rewards blended at 60/40:

| Signal | Direction | Purpose |
|--------|:---------:|---------|
| Task completion | + | Finish assigned work |
| Early completion bonus | + | Reward urgency-awareness |
| Deadline miss penalty | - | Hard deadline enforcement |
| Waiting penalty | - | Discourage passive agents |
| Team completion rate | + | Cooperative health |
| Team on-time rate | + | System-level throughput |

**Why it's hard to game:** An agent that hogs all resources completes its own tasks but tanks the team penalty. An agent that always yields never finishes its own work. The only stable strategy is principled negotiation.

---

## Three Agent Modes

We implemented and compared three distinct agent architectures:

### Mode 1: Pure RL (Q-Learning)
Each agent maintains a Q-table mapping discretized states to expected rewards. Learning via standard Bellman updates with epsilon-greedy exploration (ε decays from 0.35 → 0.19). Q-tables persist and accumulate knowledge across episodes.

### Mode 2: Pure LLM (Language Model Agents)
Each agent is backed by **Llama-3.1-8B-Instruct** via HuggingFace Inference. The agent receives the full environment state as a structured prompt and returns a JSON action. Learning happens through episodic memory and temperature adaptation (0.7 → 0.2).

### Mode 3: Hybrid (LLM + RL)
Best of both worlds:
1. **LLM** analyzes state and proposes a strategy hint
2. **RL agent** uses the hint as exploration bias
3. Only **RL learns** (updates Q-values) — LLM provides guidance

This lets the LLM's world knowledge bootstrap the RL agent's exploration, while the RL agent's Q-table provides stable, learned policies.

---

## Training: Adaptive Curriculum + Self-Play (Theme #4)

### Curriculum Progression
Training progresses through 5 difficulty levels. Agents cannot skip — they must meet completion, fairness, and belief accuracy thresholds to advance:

| Level | Conditions |
|-------|-----------|
| 0 | No negotiation, no crises, easy tasks |
| 1 | Negotiation enabled, baseline resources |
| 2 | Mild crises, resource pressure begins |
| 3 | GPU outage events, urgent task injections |
| 4 | Compound crises, cascading conflicts, full negotiation |

### Self-Play Evaluation
At each curriculum phase, we snapshot the current policy and compare it against past snapshots. This provides a direct "am I getting better?" signal without relying on absolute reward alone.

---

## Results: Evidence of Learning

### Holdout Evaluation (Trained vs Fresh Agents)
Trained agents were tested on **compound crisis scenarios** (GPU outage + urgent task injection simultaneously) that were **never seen during training**:

| Metric | Trained | Fresh (untrained) | Improvement |
|--------|:-------:|:-----------------:|:-----------:|
| Avg Total Reward | **684.6** | 620.8 | **+63.8** |
| Task Completion Rate | **75.0%** | 71.7% | **+3.3%** |
| On-Time Rate | **51.7%** | 44.2% | **+7.5%** |
| Fairness Score | 1.0 | 1.0 | Maintained |

The key insight: trained agents **generalize to unseen crisis combinations**. The +63.8 reward improvement and +7.5% on-time rate gain on holdout scenarios is the core signal of genuine learning — not memorization.

### Negotiation Builds Better World Models (Theme #1)
Agents with the negotiation protocol active develop significantly better Theory of Mind:

- Belief accuracy **with** negotiation: **0.566**
- Belief accuracy **without** negotiation: **0.538**
- **Delta: +0.028** — agents that negotiate learn to model peer behavior more accurately

This confirms that structured diplomatic interaction (not just reward signal) drives better opponent modeling.

### Aggregate Training Statistics
Over the full 50-episode training run:
- **Mean reward:** 628.9 ± 76.2
- **Fairness score:** 1.000 (perfect equity maintained throughout)
- **Mean belief accuracy:** 0.566
- **Mean conflicts per episode:** 2.86
- **Curriculum level reached:** Level 1 (promoted from Level 0)

---

## Fine-Tuning with Unsloth (SFT + GRPO)

The Colab notebook includes a full fine-tuning pipeline using **Unsloth** on **Llama-3.2-1B-Instruct**:

### SFT (Supervised Fine-Tuning)
- Training data: top-50% reward episodes from hybrid training
- Each sample: structured prompt (hour, scenario, role) → agent action JSON
- LoRA rank 16, targeting all attention + MLP projections
- 4-bit quantization for Colab T4 compatibility

### GRPO (Group Relative Policy Optimization)
- Reward function scores: valid JSON (+0.4), valid action type (+0.3), strategic action (+0.2), valid resource bounds (+0.1)
- Directly optimizes the model for generating correct resource allocation actions
- Temperature-controlled generation for exploration vs exploitation

The fine-tuned model generates valid JSON resource allocation actions when given environment state descriptions — demonstrating that the trajectory data from CrisisCompute is rich enough to train language models on strategic resource negotiation.

---

## Technical Architecture

```
CrisisCompute/
├── satya_env/               # Core OpenEnv environment
│   ├── env.py               # Episode logic, resource pool, crisis events
│   ├── reward.py            # Composable reward rubric
│   ├── negotiation.py       # Negotiation protocol + belief/reputation
│   └── rl_environment.py    # Gym-style wrapper for RL training
├── server/                  # OpenEnv MCP server (Docker / HF Space)
│   └── app.py               # FastAPI + Gradio UI on port 7860
├── src/                     # Agent implementations
│   ├── rl_agent.py          # Q-learning agent with persistent Q-tables
│   ├── hybrid_agent.py      # LLM strategy hint + RL learning
│   └── agents.py            # LLM agent via HuggingFace Inference
├── notebooks/
│   └── training_colab.ipynb # Full pipeline: RL → Hybrid → SFT → GRPO
├── train.py                 # Main training script (1600+ lines)
├── Dockerfile               # HF Spaces deployment
└── openenv.yaml             # OpenEnv manifest
```

### OpenEnv Compliance
- Built on `openenv-core` base classes with standard `reset / step / state` API
- Valid `openenv.yaml` manifest with typed action and observation schemas
- Clean client/server separation — `server/` never imports `satya_env/` internals
- Deployed to HuggingFace Spaces via Docker SDK on port 7860

---

## Why This Matters

1. **AI ops teams** managing shared cloud infrastructure need agents that negotiate, not just optimize in isolation
2. **Multi-agent LLM pipelines** (AutoGen, CrewAI, LangGraph) currently have no mechanism for resource conflict resolution
3. **Research** into fair coordination under scarcity lacks good benchmarks — CrisisCompute fills this gap

The core contribution: training on CrisisCompute teaches agents to negotiate, yield, and cooperate under genuine competitive pressure. This is a capability gap that frontier LLMs still lack out of the box.

---

## Try It Yourself

### Quickest (RL mode, no API key needed)
```bash
pip install -r requirements.txt
python train.py
```

### Colab (recommended)
Open the notebook on a free T4 GPU — runs the complete pipeline:
RL training → Hybrid training → Unsloth SFT → GRPO → all plots saved.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb)

### Live Environment
Visit the HuggingFace Space to interact with the environment directly:
[gautam0898-crisiscompute.hf.space](https://gautam0898-crisiscompute.hf.space)

---

## Training Plots

### Reward Curve
![Reward Curve](results/rl_reward_curve.png)

*RL Training: 30 episodes. EP1 reward: 639 → Peak EP30: 839. Clear upward trend confirms agents improve through the adaptive curriculum.*

### Metrics Dashboard
![Metrics Dashboard](results/metrics_dashboard.png)

*Per-episode breakdown: completion rate, on-time rate, fairness score, and belief accuracy across the full training run.*

---

*CrisisCompute — Built for the OpenEnv Hackathon India 2026*
*Team: Gautam & Satyam*
