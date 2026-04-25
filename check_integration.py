#!/usr/bin/env python3
"""Run LLM and RL training sequentially to verify integration."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run_mode(mode: str, episodes: int = 1) -> int:
    env = os.environ.copy()
    env["TRAINING_AGENT_MODE"] = mode

    command = [
        sys.executable,
        "-c",
        f"from train import train_agents; train_agents({episodes})",
    ]

    print("\n" + "=" * 72)
    print(f"Running {mode.upper()} mode sequentially")
    print("=" * 72)

    process = subprocess.run(
        command,
        cwd=ROOT,
        env=env,
        text=True,
    )
    return process.returncode


def main() -> int:
    print("\n" + "#" * 72)
    print("Sequential LLM + RL integration check")
    print("#" * 72)

    llm_code = run_mode("llm", episodes=1)
    rl_code = run_mode("rl", episodes=1)

    print("\n" + "#" * 72)
    print("Integration summary")
    print("#" * 72)
    print(f"LLM mode exit code: {llm_code}")
    print(f"RL mode exit code: {rl_code}")

    if llm_code == 0 and rl_code == 0:
        print("✅ Both sequential runs completed")
        return 0

    print("❌ One or both runs failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())