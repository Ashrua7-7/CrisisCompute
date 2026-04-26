# src/visualize.py

import json
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


class ResultsVisualizer:
    """
    Creates graphs and visualizations from training results
    """
    
    def __init__(self, results_data):
        """
        Args:
            results_data: List of episode results
        """
        self.results = results_data
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def plot_reward_curve(self, save_path="results/reward_curve.png"):
        """
        Plot reward progression over episodes.
        Shows raw per-episode reward (faded), a smoothed moving-average
        curve (solid, primary), and a trend line fit on the smoothed
        data so it reflects real learning, not single-episode noise.
        """

        if not self.results:
            print("❌ No results to plot")
            return

        episodes = []
        rewards = []
        for result in self.results:
            episodes.append(result.get("episode", 0))
            reward = result.get("metrics", {}).get("total_reward", 0)
            rewards.append(float(reward))

        if not episodes:
            print("❌ No episode data to plot")
            return

        episodes_arr = np.array(episodes, dtype=float)
        rewards_arr = np.array(rewards, dtype=float)

        # Moving average smoothing — window scales with run length
        n = len(rewards_arr)
        if n >= 80:
            window = 15
        elif n >= 50:
            window = 12
        elif n >= 30:
            window = 10
        else:
            window = max(3, n // 3)
        if n >= window:
            kernel = np.ones(window) / window
            smoothed = np.convolve(rewards_arr, kernel, mode="same")
            edge = window // 2
            for i in range(edge):
                smoothed[i] = rewards_arr[: i + edge + 1].mean()
                smoothed[-(i + 1)] = rewards_arr[-(i + edge + 1):].mean()
        else:
            smoothed = rewards_arr.copy()

        plt.figure(figsize=(12, 6))

        plt.plot(
            episodes_arr,
            rewards_arr,
            color="#7aa6ff",
            linewidth=1.0,
            alpha=0.45,
            marker="o",
            markersize=3,
            label=f"Raw per-episode reward",
        )

        plt.plot(
            episodes_arr,
            smoothed,
            color="#0b3d91",
            linewidth=3.0,
            label=f"Smoothed (window={window})",
        )

        # Trend line — fit ONLY on post-transient region (skip first 20% which
        # is the warmup-driven exploration phase). Linear fit on the early
        # transient is misleading; the steady-state slope is what reflects
        # actual learning.
        if len(episodes_arr) >= 10:
            transient_skip = max(1, n // 5)
            ep_steady = episodes_arr[transient_skip:]
            sm_steady = smoothed[transient_skip:]
            if len(ep_steady) >= 2:
                z = np.polyfit(ep_steady, sm_steady, 1)
                p = np.poly1d(z)
                slope_label = f"Steady-state trend (slope={z[0]:+.2f}/ep)"
                plt.plot(ep_steady, p(ep_steady), "r--", alpha=0.85, linewidth=2, label=slope_label)
                plt.axvline(x=transient_skip, color="gray", linestyle=":", alpha=0.5, linewidth=1)
                plt.text(
                    transient_skip + 0.5,
                    plt.gca().get_ylim()[0] if plt.gca().get_ylim() else min(rewards_arr),
                    "  ← warmup transient",
                    fontsize=8, alpha=0.6, color="gray",
                    verticalalignment="bottom",
                )

        plt.xlabel("Episode", fontsize=12, fontweight="bold")
        plt.ylabel("Total Reward (Points)", fontsize=12, fontweight="bold")
        plt.title(
            f"RL Training: Reward Curve ({n} Episodes)",
            fontsize=14,
            fontweight="bold",
        )
        plt.grid(True, alpha=0.3, linestyle="--")
        plt.legend(fontsize=10, loc="lower right")

        # Honest improvement metric: post-transient first-N vs last-N mean
        # (skip the first 20% so we measure learning, not warmup transient)
        if n >= 10:
            transient = max(1, n // 5)
            band = max(5, (n - transient) // 4)  # 25% of post-transient
            first_mean = float(rewards_arr[transient:transient + band].mean())
            last_mean = float(rewards_arr[-band:].mean())
            delta = last_mean - first_mean
            pct = (delta / first_mean * 100.0) if first_mean > 0 else 0.0
            std_first = float(rewards_arr[transient:transient + band].std())
            std_last = float(rewards_arr[-band:].std())
            peak = float(rewards_arr.max())
            peak_ep = int(episodes_arr[int(np.argmax(rewards_arr))])
            text = (
                f"Post-warmup first {band} ep: {first_mean:.1f}  (σ={std_first:.1f})\n"
                f"Last {band} ep mean:        {last_mean:.1f}  (σ={std_last:.1f})\n"
                f"Δ = {delta:+.1f}  ({pct:+.1f}%)\n"
                f"Peak: {peak:.1f} @ ep{peak_ep}\n"
                f"Variance reduction: {(1 - std_last / max(std_first, 1e-6)) * 100:+.0f}%"
            )
            plt.text(
                0.02, 0.98, text,
                transform=plt.gca().transAxes, fontsize=9,
                family="monospace",
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.85, edgecolor="gray"),
            )

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    def plot_metrics_dashboard(self, save_path="results/metrics_dashboard.png"):
        """
        Plot 4 metrics in 2x2 grid
        Shows all improvement areas
        """
        
        if not self.results:
            print("❌ No results to plot")
            return
        
        # Extract metrics
        episodes = []
        completion_rates = []
        on_time_rates = []
        resource_util = []
        cooperation_scores = []
        
        for result in self.results:
            episodes.append(result.get("episode", 0))
            metrics = result.get("metrics", {})
            
            completion_rates.append(metrics.get("completion_rate", 0))
            on_time_rates.append(metrics.get("on_time_rate", 0))
            resource_util.append(metrics.get("resource_utilization", 0))
            cooperation_scores.append(metrics.get("cooperation_score", 0))
        
        # Create 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Agent Performance Metrics Dashboard', fontsize=16, fontweight='bold', y=1.00)
        
        # Plot 1: Completion Rate
        axes[0, 0].plot(episodes, completion_rates, 'g-o', linewidth=2, markersize=5)
        axes[0, 0].set_ylabel('Completion %', fontsize=11, fontweight='bold')
        axes[0, 0].set_title('Task Completion Rate', fontsize=12, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0, 105])
        axes[0, 0].axhline(y=100, color='r', linestyle='--', alpha=0.3, label='Target: 100%')
        axes[0, 0].legend(fontsize=9)
        
        # Plot 2: On-Time Rate
        axes[0, 1].plot(episodes, on_time_rates, 'b-o', linewidth=2, markersize=5)
        axes[0, 1].set_ylabel('On-Time %', fontsize=11, fontweight='bold')
        axes[0, 1].set_title('On-Time Delivery Rate', fontsize=12, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].axhline(y=100, color='r', linestyle='--', alpha=0.3, label='Target: 100%')
        axes[0, 1].legend(fontsize=9)
        
        # Plot 3: Resource Utilization
        axes[1, 0].plot(episodes, resource_util, 'r-o', linewidth=2, markersize=5)
        axes[1, 0].set_ylabel('Utilization %', fontsize=11, fontweight='bold')
        axes[1, 0].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[1, 0].set_title('Resource Utilization', fontsize=12, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 105])
        axes[1, 0].axhline(y=70, color='orange', linestyle='--', alpha=0.3, label='Target: 70%+')
        axes[1, 0].legend(fontsize=9)
        
        # Plot 4: Cooperation Score
        axes[1, 1].plot(episodes, cooperation_scores, color='purple', marker='o', linewidth=2, markersize=5)
        axes[1, 1].set_ylabel('Cooperation %', fontsize=11, fontweight='bold')
        axes[1, 1].set_xlabel('Episode', fontsize=11, fontweight='bold')
        axes[1, 1].set_title('Agent Cooperation Score', fontsize=12, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0, 105])
        axes[1, 1].axhline(y=80, color='g', linestyle='--', alpha=0.3, label='Target: 80%+')
        axes[1, 1].legend(fontsize=9)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
        plt.close()
    
    @staticmethod
    def plot_mode_comparison(summaries, save_path="results/mode_comparison.png"):
        """
        Plot a side-by-side comparison of LLM / RL / Hybrid modes.

        Layout (2x2 grid):
          - Top-left:  Smoothed reward curves overlay (one line per mode)
          - Top-right: Mean reward bar chart
          - Bottom-left:  Completion rate bar chart
          - Bottom-right: Reward improvement (%) bar chart

        Args:
            summaries: list of mode-summary dicts as returned by train_agents().
                       Each must include "mode" and "per_episode_rewards".
            save_path: output PNG path
        """
        if not summaries:
            print("❌ No mode summaries to plot")
            return

        mode_colors = {
            "llm":    "#9b59b6",
            "rl":     "#2ecc71",
            "hybrid": "#e67e22",
        }
        mode_labels = {"llm": "LLM", "rl": "RL", "hybrid": "Hybrid"}

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Mode Comparison — LLM vs RL vs Hybrid", fontsize=15, fontweight="bold")

        # --- Top-left: smoothed reward curves overlay ---
        ax = axes[0, 0]
        for s in summaries:
            mode = str(s.get("mode", "?")).lower()
            color = mode_colors.get(mode, "#34495e")
            label = mode_labels.get(mode, mode.upper())
            rewards = s.get("per_episode_rewards") or []
            if not rewards:
                continue
            ep = np.arange(1, len(rewards) + 1)
            arr = np.array(rewards, dtype=float)
            n = len(arr)
            window = max(3, min(10, n // 4))
            if n >= window:
                kernel = np.ones(window) / window
                smoothed = np.convolve(arr, kernel, mode="same")
                edge = window // 2
                for i in range(edge):
                    smoothed[i] = arr[: i + edge + 1].mean()
                    smoothed[-(i + 1)] = arr[-(i + edge + 1):].mean()
            else:
                smoothed = arr
            ax.plot(ep, arr, color=color, alpha=0.25, linewidth=1)
            ax.plot(ep, smoothed, color=color, linewidth=2.5, label=f"{label} (smoothed)")
        ax.set_xlabel("Episode", fontweight="bold")
        ax.set_ylabel("Total Reward", fontweight="bold")
        ax.set_title("Reward Curves Overlay", fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(loc="lower right", fontsize=9)

        # --- Top-right: mean reward bar chart ---
        ax = axes[0, 1]
        modes = [mode_labels.get(str(s.get("mode", "?")).lower(), str(s.get("mode", "?"))) for s in summaries]
        means = [float(s.get("mean_reward", 0.0)) for s in summaries]
        colors = [mode_colors.get(str(s.get("mode", "?")).lower(), "#34495e") for s in summaries]
        bars = ax.bar(modes, means, color=colors, edgecolor="black", linewidth=1.2)
        for bar, val in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Mean Reward", fontweight="bold")
        ax.set_title("Mean Reward per Mode", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        if means:
            ax.set_ylim(0, max(means) * 1.15)

        # --- Bottom-left: completion-rate bar chart ---
        ax = axes[1, 0]
        completions = [float(s.get("mean_completion", 0.0)) for s in summaries]
        bars = ax.bar(modes, completions, color=colors, edgecolor="black", linewidth=1.2)
        for bar, val in zip(bars, completions):
            ax.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(y=100, color="gray", linestyle=":", alpha=0.5, label="Target 100%")
        ax.set_ylabel("Mean Completion Rate (%)", fontweight="bold")
        ax.set_title("Task Completion Rate", fontweight="bold")
        ax.set_ylim(0, 110)
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")
        ax.legend(fontsize=9, loc="lower right")

        # --- Bottom-right: improvement % bar chart ---
        ax = axes[1, 1]
        improvements = [float(s.get("reward_improvement_pct", 0.0)) for s in summaries]
        bars = ax.bar(modes, improvements, color=colors, edgecolor="black", linewidth=1.2)
        for bar, val in zip(bars, improvements):
            ypos = val if val >= 0 else 0
            ax.text(bar.get_x() + bar.get_width() / 2, ypos, f"{val:+.1f}%",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_ylabel("Reward Improvement (%)", fontweight="bold")
        ax.set_title("Stable-Phase Reward Improvement", fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y", linestyle="--")

        plt.tight_layout()
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"✅ Saved: {save_path}")
        plt.close()

    def extract_dialogues(self, save_path="results/dialogue_samples.txt"):
        """
        Extract run-derived negotiation snippets (no synthetic narration).
        """
        
        if not self.results or len(self.results) < 2:
            print("⚠️  Not enough episodes for dialogue comparison")
            return
        
        lines = []
        lines.append("AGENT NEGOTIATION LOG SAMPLES (RUN-DERIVED)")
        lines.append("=" * 60)
        lines.append(f"Episodes captured: {len(self.results)}")
        lines.append("")

        sample_count = min(3, len(self.results))
        for result in self.results[:sample_count]:
            episode = result.get("episode", "?")
            lines.append(f"Episode {episode}")
            lines.append("-" * 30)
            steps = result.get("steps", [])
            if not steps:
                lines.append("No step-level records captured.")
                lines.append("")
                continue
            for step in steps[:2]:
                lines.append(f"Hour {step.get('hour', '?')}:")
                actions = step.get("actions", {})
                for agent_id, action in actions.items():
                    lines.append(
                        f"  - {agent_id}: action={action.get('action')} task={action.get('task_id')} "
                        f"cpu={action.get('cores_needed', 0)} gpu={action.get('gpu_needed', 0)} mem={action.get('memory_needed', 0)}"
                    )
                lines.append(
                    f"    metrics: conflicts={step.get('conflict_count', 0)} coalitions={step.get('coalitions_formed', 0)} "
                    f"fairness={step.get('fairness_score', 0):.3f} belief={step.get('belief_accuracy', 0):.3f}"
                )
            lines.append("")

        dialogue = "\n".join(lines)
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(dialogue)
        
        print(f"✅ Saved: {save_path}")
    
    def generate_summary_stats(self, save_path="results/summary_statistics.txt"):
        """
        Generate numerical summary
        """
        
        if not self.results:
            print("❌ No results")
            return
        
        first = self.results[0]
        last = self.results[-1]

        all_rewards = [r.get("metrics", {}).get("total_reward", r.get("total_reward", 0)) for r in self.results]
        fairness_values = [r.get("avg_fairness_score", 0.0) for r in self.results]
        belief_values = [r.get("avg_belief_accuracy", 0.0) for r in self.results]
        conflict_values = [r.get("conflict_count", 0) for r in self.results]
        coalition_values = [r.get("coalitions_formed", 0) for r in self.results]
        contracts_kept = [r.get("contracts_kept", 0) for r in self.results]
        contracts_broken = [r.get("contracts_broken", 0) for r in self.results]

        summary = f"""
TRAINING SUMMARY STATISTICS (RUN-DERIVED)
=========================================

Training Configuration:
- Total Episodes: {len(self.results)}
- Training Date: {self.timestamp}

Reward Progression:
- Episode 1:    {first.get('metrics', {}).get('total_reward', first.get('total_reward', 0)):6.1f} points
- Episode Last: {last.get('metrics', {}).get('total_reward', last.get('total_reward', 0)):6.1f} points
- Average Reward: {np.mean(all_rewards):.2f} points
- Reward Std Dev: {np.std(all_rewards):.2f}

Performance Metrics (from run outputs):
- Completion Rate (first -> last): {first.get('metrics', {}).get('completion_rate', 0):.2f}% -> {last.get('metrics', {}).get('completion_rate', 0):.2f}%
- On-Time Rate (first -> last): {first.get('metrics', {}).get('on_time_rate', 0):.2f}% -> {last.get('metrics', {}).get('on_time_rate', 0):.2f}%
- Avg Fairness Score: {np.mean(fairness_values):.3f}
- Avg Belief Accuracy: {np.mean(belief_values):.3f}

Negotiation Health:
- Mean Conflicts per Episode: {np.mean(conflict_values):.2f}
- Mean Coalitions Formed: {np.mean(coalition_values):.2f}
- Mean Contracts Kept: {np.mean(contracts_kept):.2f}
- Mean Contracts Broken: {np.mean(contracts_broken):.2f}
"""
        
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(summary)
        
        print(f"✅ Saved: {save_path}")


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING VISUALIZER")
    print("="*60)

    sample_results = []
    try:
        with open("results/training_results.json", "r", encoding="utf-8") as f:
            sample_results = json.load(f)
        print("Loaded real training results.")
    except FileNotFoundError:
        print("No training results found. Using sample data.")
        for ep in range(1, 31):
            reward = 15 + (ep * 2) + (ep % 3)
            result = {
                "episode": ep,
                "metrics": {
                    "total_reward": reward,
                    "completion_rate": 40 + (ep * 1.8),
                    "on_time_rate": 30 + (ep * 2),
                    "resource_utilization": 25 + (ep * 2),
                    "cooperation_score": 10 + (ep * 2.5),
                },
            }
            sample_results.append(result)

    viz = ResultsVisualizer(sample_results)

    print("\n[TEST 1] Creating visualizations...")
    viz.plot_reward_curve()
    viz.plot_metrics_dashboard()
    viz.extract_dialogues()
    viz.generate_summary_stats()

    print("\n" + "="*60)
    print("ALL VISUALIZATIONS CREATED")
    print("="*60 + "\n")
    print("Check results/ folder for:")
    print("  - reward_curve.png")
    print("  - metrics_dashboard.png")
    print("  - dialogue_samples.txt")
    print("  - summary_statistics.txt")
    print("")