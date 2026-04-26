---
title: CrisisCompute
emoji: 🚨
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: OpenEnv multi-agent compute negotiation + adaptive curriculum / self-play for LLM & RL training.
---

# CrisisCompute

**OpenEnv Hackathon India 2026** — compute-cluster negotiation for multi-agent LLM training.

**Themes:** **#1 Multi-Agent Interactions** (cooperation, negotiation, partial observability) · **#4 Self-Improvement** (adaptive curriculum, self-play evaluation) — built with [OpenEnv / `openenv-core`](https://pypi.org/project/openenv-core/).

---

## Judge quick links (everything in one place)

| Resource | Link |
|----------|------|
| **Hugging Face Space (live OpenEnv environment)** | [huggingface.co/spaces/Gautam0898/crisiscompute](https://huggingface.co/spaces/Gautam0898/crisiscompute) · [gautam0898-crisiscompute.hf.space](https://gautam0898-crisiscompute.hf.space) |
| **Source code (this repo)** | [github.com/Ashrua7-7/CrisisCompute](https://github.com/Ashrua7-7/CrisisCompute) |
| **Training: Colab notebook (Unsloth + HF TRL)** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13Z6-3Fm-H3Js9LiAC9jkCPAhzJQiFm4C?usp=sharing) · [Notebook on GitHub](https://github.com/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb) |
| **Mini-blog (HF Discussions / article)** | [CrisisCompute: Teaching AI Agents to Negotiate Under Pressure](https://huggingface.co/Gautam0898/crisiscompute-blog) |
| **Negotiation protocol (design doc)** | [docs/crisiscompute_negotiation_protocol.md](https://github.com/Ashrua7-7/CrisisCompute/blob/main/docs/crisiscompute_negotiation_protocol.md) |
| **LLM prompts & agent setup** | [docs/llm_prompts.md](https://github.com/Ashrua7-7/CrisisCompute/blob/main/docs/llm_prompts.md) · [docs/agent_personalities.md](https://github.com/Ashrua7-7/CrisisCompute/blob/main/docs/agent_personalities.md) |

---

## Minimum submission requirements (self-check)

| Requirement | Where to find it |
|-------------|------------------|
| Uses **OpenEnv** (latest) | `openenv-core>=0.2.0` in [`requirements.txt`](https://github.com/Ashrua7-7/CrisisCompute/blob/main/requirements.txt); manifest [`openenv.yaml`](https://github.com/Ashrua7-7/CrisisCompute/blob/main/openenv.yaml) |
| **Training script** with **Unsloth** or **HF TRL** in **Colab** | [`notebooks/training_colab.ipynb`](https://github.com/Ashrua7-7/CrisisCompute/blob/main/notebooks/training_colab.ipynb) (Colab badge above) |
| **Short writeup** (blog or <2 min video) | [HF mini-blog](https://huggingface.co/Gautam0898/crisiscompute-blog) · add YouTube here if you record one |
| Environment hosted on **Hugging Face Spaces** | [Space](https://huggingface.co/spaces/Gautam0898/crisiscompute) |
| **README** with motivation, how it works, **results**, and **all links** | You are reading it |

---

## TL;DR for reviewers (≈30 seconds)

1. **Problem:** Three agents share one small cluster (CPU / RAM / GPU), deadlines bite, and crises (GPU outage, urgent injects) hit mid-episode — a realistic **negotiation-under-scarcity** stress test.
2. **Environment:** OpenEnv-compliant server + rubric-based rewards (individual + team, hard to game by hoarding or always yielding).
3. **Evidence:** Training produces **higher reward and better on-time / completion** vs fresh agents on **held-out crisis combos**; curriculum + self-play hooks support Theme #4.

---

## 1. Problem: what capability gap are we targeting?

Frontier agents handle single-workflow tasks well but often fail when **multiple agents share scarce compute**, **incentives are only partly aligned**, and **the world is partially observable** (you do not fully see peers’ intent until they act).

**CrisisCompute** compresses that into a repeatable simulator: three ML-pipeline roles — `data_loader`, `data_cleaner`, `ml_trainer` — compete and cooperate for **16 CPU cores, 32 GB RAM, and 1 GPU**, under **deadline pressure** and **mid-episode shocks**.

**Why it matters:** shared clusters in labs and enterprises are the default; multi-agent stacks (orchestrators, tool-runners, trainers) will keep colliding unless models learn **negotiation, yielding, and team-level thinking**. This environment is built to **train and measure** that behavior.

---

## 2. Environment: observations, actions, rewards

### Architecture

```
┌─────────────────────────────────────────────────┐
│          CrisisCompute (compute cluster)         │
│   Resources: 16 CPU  |  32 GB RAM  |  1 GPU     │
│   Tasks: deadline-sensitive, priority-weighted   │
│   Events: GPU outage, urgent injection, conflict │
└─────────────────────────────────────────────────┘
         ↕                    ↕                ↕
   data_loader    ↔    data_cleaner   ↔  ml_trainer
         └─────── negotiation + beliefs (ToM) ──────┘
```

### Observation space (per agent)

- Task backlog, deadlines, urgency
- Pool availability (CPU / GPU / memory)
- Time pressure and progress
- Negotiation history and peer reliability / belief summaries

### Action space

Each agent picks one of:

- `request_resource` — claim CPU / memory / optional GPU
- `release_resource` — yield to reduce conflict
- `wait` — skip the timestep

### Reward signal (composable rubric)

Individual and team terms are blended (**60% / 40%**) so policies cannot win by pure selfish hoarding or by always free-riding:

| Signal | Direction | Purpose |
|--------|:---------:|---------|
| Task completion | ✅ | Progress on assigned work |
| Early completion bonus | ✅ | Rewards urgency awareness |
| Deadline miss penalty | ❌ | Hard deadline culture |
| Waiting penalty | ❌ | Discourages doing nothing |
| Team completion rate | ✅ | Cooperative health |
| Team on-time rate | ✅ | System throughput |

**Anti-gaming intuition:** hogging helps your local score but hurts team terms; perpetual yielding starves your own tasks. Stable improvement tends to require **negotiation-shaped** behavior.

### OpenEnv compliance (engineering checklist)

- Built on OpenEnv / **Gym-style** `reset` · `step` · `state` patterns and server entry via [`openenv.yaml`](openenv.yaml)
- **`openenv.yaml`** documents actions, observations, agents, and reward rubric fields
- **Client / server separation** — Space-facing server code does not import private env internals in ways that break the OpenEnv contract
- **No reserved MCP tool names** — we do not name tools `reset`, `step`, `state`, or `close`
- **Docker SDK** on **port 7860** for Hugging Face Spaces (see repo `Dockerfile`)

---

## 3. Results: what changed after training?

### Training setup (high level)

- **Agent modes:** pure RL (Q-learning), pure LLM (HF Inference), **hybrid** (LLM hint + RL updates)
- **Adaptive curriculum:** five levels from easy → full crises + negotiation; promotion gated by completion / fairness / belief metrics
- **Self-play / snapshots (Theme #4):** periodic evaluation against earlier policies (see training code and `results/` reports)

### Plots (commit these to `results/` for reviewers)

If images are in the repo root on the Hub, they also resolve under the Space:

- **Reward vs training progress:** [`results/reward_curve.png`](results/reward_curve.png) — *y-axis: episodic reward; x-axis: episode or training step (see plot title).*
- **Metrics dashboard:** [`results/metrics_dashboard.png`](results/metrics_dashboard.png) — *completion, on-time, fairness, belief accuracy over episodes.*

![Reward curve](results/reward_curve.png)

*Episodic reward vs training progress — upward trend indicates learning through the curriculum.*

![Metrics dashboard](results/metrics_dashboard.png)

*Per-episode metrics: completion, on-time performance, fairness, and belief accuracy.*

> **Note:** If `results/*.png` are not in your local clone, generate them with `python train.py` or the Colab notebook, then **commit** so judges see plots without opening Colab.

### Hold-out evaluation: trained vs fresh (compound crises)

Held-out scenarios combine stresses **not seen during training** (e.g. GPU outage **and** urgent injection). Example numbers from our logged run:

| Metric | Trained | Fresh (untrained) | Δ |
|--------|:-------:|:-----------------:|:-:|
| Avg total reward | **684.6** | 620.8 | **+63.8** |
| Task completion rate | **75.0%** | 71.7% | **+3.3%** |
| On-time rate | **51.7%** | 44.2% | **+7.5%** |
| Fairness score | 1.0 | 1.0 | — |

Raw logs: [`results/holdout_evaluation.json`](results/holdout_evaluation.json), [`results/selfplay_report.json`](results/selfplay_report.json), and related files under [`results/`](results/).

### Theme #1: negotiation and belief accuracy

With the negotiation protocol enabled, agents achieve **higher belief accuracy** about peers than without negotiation (measured in-environment), supporting **theory-of-mind-style** modeling under partial observability.

---

## 4. How judging criteria map to this submission

| Criterion (weight) | What we show |
|--------------------|--------------|
| **Environment innovation (40%)** | Crisis negotiation + rubric + crises + ToM-flavored beliefs — not a toy grid game; stresses **strategic multi-agent** behavior |
| **Storytelling (30%)** | This README + [HF blog](https://huggingface.co/Gautam0898/crisiscompute-blog) + live [Space](https://huggingface.co/spaces/Gautam0898/crisiscompute) UI |
| **Improvement in rewards (20%)** | Curves + trained vs fresh table + JSON artifacts in `results/` |
| **Reward & training pipeline (10%)** | Rubric described above; Colab runs **Unsloth + TRL** against the real loop |

---

## How to run

### Fastest path (RL, no API key)

```bash
pip install -r requirements.txt
python train.py
```

Artifacts land in `results/` (metrics JSON, plots when plotting is enabled, evaluation summaries).

### LLM / hybrid mode

Copy [`.env.example`](.env.example) → `.env` and set at least:

```env
LLM_PROVIDER=huggingface
HF_TOKEN=hf_...
TRAINING_AGENT_MODE=hybrid
```

### Judges: use Colab

The notebook is the intended **one-click** re-run: RL → hybrid options → **Unsloth SFT** → **GRPO** (TRL), with logs and plots.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/13Z6-3Fm-H3Js9LiAC9jkCPAhzJQiFm4C?usp=sharing)

---

## Repository layout

```
CrisisCompute/
├── satya_env/               # Core environment logic
├── server/                  # OpenEnv / MCP-facing server (Space)
├── src/                     # RL, LLM, hybrid agents
├── notebooks/
│   └── training_colab.ipynb # Unsloth + TRL training (Colab)
├── results/                 # Plots + JSON metrics (commit for judges)
├── docs/                    # Protocol, prompts, personalities
├── openenv.yaml             # OpenEnv manifest
├── Dockerfile               # HF Spaces
└── train.py                 # CLI training entrypoint
```

---

## License and attribution

**CrisisCompute** — OpenEnv Hackathon India 2026. Add a `LICENSE` file in the repo and, if you use Space card metadata, set `license:` in the YAML frontmatter to match it.

---

*Structured for Hugging Face Spaces (`README.md` card) and hackathon judges: quick links, requirements checklist, story, results, and rubric/training mapping.*