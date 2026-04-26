"""
app.py — FastAPI server for SatyaEnv (OpenEnv compliant HTTP interface).

Exposes standard OpenEnv endpoints:
  POST /reset   — start a new episode
  POST /step    — take one step
  GET  /state   — get current internal state
  GET  /health  — health check
  GET  /schema  — action + observation JSON schemas

Run locally:
  uvicorn server.app:app --reload --port 7860

On HuggingFace Spaces:
  Set the SDK to "gradio" or use Dockerfile pointing to this with:
  CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import json
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from server.environment import SatyaEnvironment
from server.models import MultiAgentAction

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="SatyaEnv — Multi-Agent Compute Allocation",
    description=(
        "OpenEnv-compliant RL environment where three LLM agents "
        "(data_loader, data_cleaner, ml_trainer) negotiate over shared "
        "CPU/GPU/memory to complete ML pipeline tasks under deadline pressure."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared env instance (set SUPPORTS_CONCURRENT_SESSIONS=True in env for multi-session)
_env = SatyaEnvironment()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: int | None = None
    episode_id: str | None = None


class StepRequest(BaseModel):
    """
    Actions dict for all agents. Example:
    {
      "actions": {
        "data_loader":  {"action": "request_resource", "task_id": "t1", "reasoning": "..."},
        "data_cleaner": {"action": "wait", "task_id": null},
        "ml_trainer":   {"action": "request_resource", "task_id": "t3", "reasoning": "..."}
      }
    }
    """
    actions: Dict[str, Dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "healthy", "env": "SatyaEnv"}


@app.get("/metadata")
def metadata():
    return _env.get_metadata().model_dump()


@app.get("/schema")
def schema():
    return {
        "action": MultiAgentAction.model_json_schema(),
        "observation": _env.reset().model_dump(),  # sample observation as schema hint
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    obs = _env.reset(seed=req.seed, episode_id=req.episode_id)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
    }


@app.post("/step")
def step(req: StepRequest):
    action = MultiAgentAction(actions=req.actions)
    obs = _env.step(action)
    return {
        "observation": obs.model_dump(),
        "reward": obs.reward,
        "done": obs.done,
        "step_rewards": obs.step_rewards,
        "team_reward": obs.team_reward,
        "metrics": obs.metrics,
    }


@app.get("/state")
def state():
    return _env.state.model_dump()


# ---------------------------------------------------------------------------
# Simple Gradio UI (optional — makes HF Spaces happy)
# ---------------------------------------------------------------------------

try:
    import gradio as gr

    def gradio_reset():
        obs = _env.reset()
        return json.dumps(obs.model_dump(), indent=2)

    def gradio_step(actions_json: str):
        try:
            actions = json.loads(actions_json)
        except json.JSONDecodeError:
            return "Invalid JSON — use format: {\"data_loader\": {\"action\": \"wait\"}}"
        action = MultiAgentAction(actions=actions)
        obs = _env.step(action)
        return json.dumps(obs.model_dump(), indent=2)

    with gr.Blocks(title="SatyaEnv") as demo:
        gr.Markdown(
            "# 🤖 SatyaEnv — Multi-Agent Compute Allocation\n"
            "Three agents negotiate over shared resources to complete ML tasks.\n"
            "**OpenEnv compliant** | Theme #1: Multi-Agent Interactions"
        )
        with gr.Row():
            reset_btn = gr.Button("Reset Episode", variant="primary")
        obs_box = gr.Textbox(label="Observation", lines=20, interactive=False)
        reset_btn.click(fn=gradio_reset, outputs=obs_box)

        gr.Markdown("### Take a Step")
        actions_input = gr.Textbox(
            label="Actions JSON",
            value=json.dumps({
                "data_loader":  {"action": "request_resource", "task_id": "t1"},
                "data_cleaner": {"action": "wait", "task_id": None},
                "ml_trainer":   {"action": "request_resource", "task_id": "t3"},
            }, indent=2),
            lines=8,
        )
        step_btn = gr.Button("Step")
        step_box = gr.Textbox(label="Step Result", lines=20, interactive=False)
        step_btn.click(fn=gradio_step, inputs=actions_input, outputs=step_box)

    app = gr.mount_gradio_app(app, demo, path="/")

except ImportError:
    pass  # Gradio optional — pure FastAPI still works
