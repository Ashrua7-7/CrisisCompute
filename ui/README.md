# CrisisCompute UI

ChatGPT-style interface to interact with the 3 multi-agents in real time.

## Layout

```
┌─────────────────────────────────────────────────────────────────┐
│  🚨 CrisisCompute — Multi-Agent Negotiation System       [btns] │
├──────────┬──────────────────────────────────────┬───────────────┤
│          │                                      │ Resource Pool │
│  Agents  │        Chat Window                   │ CPU ████░░░░ │
│          │                                      │ GPU ██░░░░░░ │
│ 📂 Loader │  [agent bubbles + step results]     │ MEM ███░░░░░ │
│ 🧹 Cleaner│                                      │               │
│ 🧠 Trainer│                                      │ Tasks         │
│          │  > ___________________________       │ ● load_001 ✓ │
│          │                          [Send]      │ ▶ clean_001   │
└──────────┴──────────────────────────────────────┴───────────────┘
```

## Setup

**Option A — Use the deployed HuggingFace Space (recommended, no local server needed):**

Just start the UI:
```powershell
npm install   # first time only
npm run dev
```
Open http://localhost:3000 — click **☁ HF Space** in the top bar (default).
It connects directly to https://gautam0898-crisiscompute.hf.space

**Option B — Run locally:**
```powershell
# Terminal 1 — backend
cd "d:\Meta Hack\multi-agent"
.\.venv\Scripts\Activate.ps1
uvicorn server.app:app --reload --port 7860

# Terminal 2 — UI
cd ui
npm run dev
```
Open http://localhost:3000 — click **💻 Local** in the top bar.

## Commands

| Command | What it does |
|---------|-------------|
| `reset` | Start a fresh episode |
| `step` | Take one negotiation step |
| `run 5` | Run 5 steps automatically |
| `auto` | Run until episode ends |
| `stop` | Stop auto-run |
| `status` | Show current state |
| `help` | Show all commands |
| `clear` | Clear chat history |

## Notes

- No changes were made to the existing project structure
- The UI proxies all API calls to `localhost:7860` via Vite's dev proxy
- All 3 agents' actions and reasoning appear as chat bubbles
- Resource bars update live after each step
