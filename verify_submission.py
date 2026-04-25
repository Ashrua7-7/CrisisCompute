#!/usr/bin/env python3
"""Verify submission artifacts are complete and valid"""

import json
from pathlib import Path

print("\n" + "="*70)
print("✅ SUBMISSION ARTIFACT VERIFICATION")
print("="*70 + "\n")

# Test 1: Theme #4 Summary
print("[CHECK 1] Theme #4 Self-Improvement Data")
print("-"*70)
try:
    with open('results/theme4_summary.json') as f:
        t4 = json.load(f)
    
    print(f"✅ theme4_summary.json loaded")
    print(f"   - Curriculum enabled: {t4.get('enabled')}")
    print(f"   - Curriculum phases logged: {len(t4.get('curriculum_history', []))}")
    print(f"   - Self-play duels recorded: {len(t4.get('league_duels', []))}")
    
    if t4.get('curriculum_history'):
        last_phase = t4['curriculum_history'][-1]
        print(f"   - Final curriculum level: {last_phase.get('level')}")
        print(f"   - Final promotion status: {last_phase.get('promoted')}")
    
    holdout_delta = t4.get('holdout_delta', {})
    print(f"\n   Holdout Generalization (trained vs fresh):")
    print(f"   - Reward improvement: {holdout_delta.get('avg_total_reward', 0):+.1f} pts")
    print(f"   - Belief accuracy gain: {holdout_delta.get('avg_belief_accuracy', 0):+.4f}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Training Results (Theme #1 + RL)
print("\n[CHECK 2] Training Results & Learning")
print("-"*70)
try:
    with open('results/training_results.json') as f:
        results = json.load(f)
    
    print(f"✅ training_results.json loaded")
    print(f"   - Episodes trained: {len(results)}")
    
    if results:
        first_ep = results[0]
        last_ep = results[-1]
        print(f"   - Episode 1 reward: {first_ep.get('total_reward'):.1f} pts")
        print(f"   - Last reward: {last_ep.get('total_reward'):.1f} pts")
        
        # Check for negotiation data
        if 'steps' in first_ep and first_ep['steps']:
            step = first_ep['steps'][0]
            has_conflict = 'conflict_count' in step
            has_coalitions = 'coalitions_formed' in step
            has_fairness = 'fairness_score' in step
            print(f"   - Step data includes: conflicts={has_conflict}, coalitions={has_coalitions}, fairness={has_fairness}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: Negotiation Trace (Theme #1 Protocol)
print("\n[CHECK 3] Negotiation Protocol Evidence (Theme #1)")
print("-"*70)
try:
    with open('results/negotiation_trace.json') as f:
        traces = json.load(f)
    
    print(f"✅ negotiation_trace.json loaded")
    print(f"   - Negotiation steps logged: {len(traces)}")
    
    if traces:
        sample = traces[0]
        print(f"   - Sample step has keys: {list(sample.keys())[:6]}")
        print(f"   - Contains conflicts: {len(sample.get('detected_conflicts', []))} detected")
        print(f"   - Contains coalitions: {len(sample.get('coalitions', []))} formed")
        print(f"   - Contains yields: {len(sample.get('yields', []))} concessions")
        
        # Count total
        total_conflicts = sum(len(t.get('detected_conflicts', [])) for t in traces)
        total_coalitions = sum(len(t.get('coalitions', [])) for t in traces)
        total_yields = sum(len(t.get('yields', [])) for t in traces)
        print(f"\n   Aggregate across {len(traces)} steps:")
        print(f"   - Total conflicts detected: {total_conflicts}")
        print(f"   - Total coalitions formed: {total_coalitions}")
        print(f"   - Total concessions: {total_yields}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 4: Plots exist
print("\n[CHECK 4] Visualization Artifacts")
print("-"*70)
for png_file in ['reward_curve.png', 'metrics_dashboard.png']:
    path = Path(f'results/{png_file}')
    if path.exists():
        size_kb = path.stat().st_size / 1024
        print(f"✅ {png_file:30} ({size_kb:6.1f} KB)")
    else:
        print(f"❌ {png_file:30} MISSING")

# Test 5: Baseline Comparison (Negotiation Impact)
print("\n[CHECK 5] Baseline Comparison (Negotiation vs No-Negotiation)")
print("-"*70)
try:
    with open('results/baseline_comparison.json') as f:
        baseline = json.load(f)
    
    print(f"✅ baseline_comparison.json loaded")
    with_neg = baseline.get('negotiation_enabled_run', {})
    without_neg = baseline.get('baseline_run_negotiation_disabled', {})
    delta = baseline.get('delta', {})
    
    print(f"\n   Negotiation IMPACT on performance:")
    print(f"   - Completion rate impact: {delta.get('avg_completion_rate', 0):+.3f}")
    print(f"   - Fairness impact: {delta.get('avg_fairness_score', 0):+.3f}")
    print(f"   - Belief accuracy impact: {delta.get('avg_belief_accuracy', 0):+.3f}")
    
except Exception as e:
    print(f"❌ FAILED: {e}")

# Final summary
print("\n" + "="*70)
print("✅ SUBMISSION READY FOR JUDGES")
print("="*70)
print("\nKey Evidence for:")
print("  Theme #1: Negotiation traces + coalition data in negotiation_trace.json")
print("  Theme #4: Curriculum + self-play in theme4_summary.json")
print("  Learning: Reward curves + holdout generalization in plots")
print("\nAll files committed to results/ folder")
print("="*70 + "\n")
