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
        Plot reward progression over episodes
        Shows agents learning!
        """
        
        if not self.results:
            print("❌ No results to plot")
            return
        
        # Extract data
        episodes = []
        rewards = []
        
        for result in self.results:
            episodes.append(result.get("episode", 0))
            reward = result.get("metrics", {}).get("total_reward", 0)
            rewards.append(reward)
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot main curve
        plt.plot(episodes, rewards, 'b-o', linewidth=2.5, markersize=6, label='Reward')
        
        # Add trend line
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), "r--", alpha=0.8, linewidth=2, label='Trend')
        
        # Styling
        plt.xlabel('Episode', fontsize=12, fontweight='bold')
        plt.ylabel('Total Reward (Points)', fontsize=12, fontweight='bold')
        plt.title('Agent Learning Progress - Reward Trajectory', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=11)
        
        # Add annotations
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            improvement = ((max_reward - min_reward) / min_reward * 100) if min_reward > 0 else 0
            
            plt.text(0.02, 0.98, f"Improvement: {improvement:.1f}%\nMin: {min_reward:.1f}, Max: {max_reward:.1f}", 
                    transform=plt.gca().transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
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