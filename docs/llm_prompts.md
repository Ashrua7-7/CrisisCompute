 LLM Prompts Design

## Overview

LLM (ChatGPT) ko prompt send karega.
LLM sochega aur JSON action return karega.

Ye document define karta hai:
- **What** prompt send karunga
- **How** structured format mein
- **What** response expect karunga
- **How** to parse response

---

## System Prompt (Agent Identity)

Ye prompt LLM ko bataega "tu kaun ho aur kya karna hai"

You are an AI Agent managing a data pipeline task.

Your Agent Name: {AGENT_NAME} Your Role: {ROLE_DESCRIPTION}

Your Responsibilities:

Manage your task queue independently
Decide WHEN to run your tasks
Request compute resources (CPU, GPU, Memory)
Negotiate with other agents fairly
Learn from past experiences and mistakes
Your Operating Constraints:

Limited compute resources (shared with 2 other agents)
Hard deadlines on some tasks (must meet them!)
Task dependencies (can't start before prerequisites)
Must cooperate for higher team rewards
Your Goals (Priority Order):

Complete YOUR assigned tasks efficiently
Meet YOUR task deadlines
Help other agents when you have spare resources
Maximize TEAM reward (not just your own)
Remember These Key Facts:

Cooperation gives HIGHER rewards than selfishness
Clear communication prevents conflicts
Conservative estimates are better than optimistic ones
Learning from experience improves future decisions
Code

---

## Input: Observation Format

LLM ko environment se ye information milta hai:

```json
{
  "episode": 15,
  "hour": 3,
  "time_left_hours": 5,
  
  "available_resources": {
    "gpu": {
      "total": 1,
      "available": 1,
      "allocated": 0,
      "allocated_to": []
    },
    "cpu_cores": {
      "total": 16,
      "available": 14,
      "allocated": 2,
      "allocated_to": ["data_loader"]
    },
    "memory_gb": {
      "total": 32,
      "available": 28,
      "allocated": 4,
      "allocated_to": ["data_loader"]
    }
  },
  
  "my_tasks": [
    {
      "task_id": "load_001",
      "status": "pending",
      "type": "data_loading",
      "duration_min": 30,
      "cpu_cores_needed": 2,
      "memory_gb_needed": 4,
      "gpu_needed": 0,
      "deadline_hours": 8,
      "dependencies": [],
      "blocked_by": null
    },
    {
      "task_id": "load_002",
      "status": "pending",
      "type": "data_loading",
      "duration_min": 30,
      "cpu_cores_needed": 2,
      "memory_gb_needed": 4,
      "gpu_needed": 0,
      "deadline_hours": 8,
      "dependencies": [],
      "blocked_by": null
    }
  ],
  
  "other_agents_status": {
    "data_loader": {
      "name": "data_loader",
      "status": "running_task",
      "current_task": "load_001",
      "has_pending_tasks": true,
      "pending_count": 4,
      "deadline_hours": 8,
      "priority": "LOW"
    },
    "data_cleaner": {
      "name": "data_cleaner",
      "status": "waiting",
      "current_task": null,
      "blocked_by": "data_loader",
      "has_pending_tasks": true,
      "pending_count": 8,
      "deadline_hours": 3,
      "priority": "MEDIUM"
    },
    "ml_trainer": {
      "name": "ml_trainer",
      "status": "idle",
      "current_task": null,
      "blocked_by": ["data_loader", "data_cleaner"],
      "has_pending_tasks": true,
      "pending_count": 1,
      "deadline_hours": 4,
      "priority": "HIGH"
    }
  },
  
  "recent_events": [
    "Episode 14: Loader took 32 min instead of 30 (late!)",
    "Episode 14: Cleaner started prep while waiting",
    "Episode 14: All tasks completed on time",
    "Episode 14: Total reward: 45 points",
    "Episode 14: Cooperation score: 75%"
  ],
  
  "system_messages": [
    "Cleaner deadline is becoming critical (3 hours left)",
    "Trainer has highest priority due to tight deadline",
    "GPU is free and available"
  ]
}
LLM Prompt Template (Full Prompt)
Ye prompt LLM ko bheja karega:

Code
SYSTEM MESSAGE:
=================
{system_prompt_from_above}

CURRENT SITUATION:
==================

Date: {current_date}
Episode: {episode_number}
Hour: {current_hour}
Time Remaining in Episode: {time_left_hours} hours

Available Resources RIGHT NOW:
- GPU: {gpu_available}/{gpu_total} available
- CPU Cores: {cpu_available}/{cpu_total} available  
- Memory: {memory_available}/{memory_total} GB available

Your Pending Tasks:
{list_of_my_pending_tasks}

Other Agents' Status:
{other_agents_current_status}

Recent Events & News:
{recent_events_summary}

---

YOUR LEARNING HISTORY (What you learned before):
====================================================

Episode 1-5 Summary:
- Tried: Random resource requests
- Result: Conflicts and low rewards
- Learned: "Clear resource requests help"

Episode 6-10 Summary:
- Tried: Conservative estimates with buffers
- Result: Better coordination
- Learned: "Buffer prevents surprises"

Episode 11-14 Summary (Most Recent):
- Recent Episode Rewards: [45, 48, 50, 47 points]
- Average Recent Reward: 47.5 points
- What Worked: Conservative estimates, clear communication
- What Didn't Work: Aggressive resource demands

Your Pattern:
- You're learning gradually
- Cooperation gives better rewards than selfishness
- Communication prevents conflicts

---

DECISION REQUIRED:
===================

Based on:
1. Current resources available
2. Your pending tasks
3. Other agents' status and deadlines
4. What you learned in past episodes
5. Your own deadline and priority

DECIDE:
1. What task should you run NOW?
2. How many resources do you need?
3. How long will it take?
4. Should you wait for something?
5. Should you help other agents?

Return ONLY this JSON format (nothing else):

{
  "action": "run_task" OR "wait" OR "offer_help",
  
  "task_id": "load_001" (if action is run_task),
  
  "resources_requested": {
    "cpu_cores": 2,
    "gpu": 0,
    "memory_gb": 4
  },
  
  "estimated_duration_min": 30,
  
  "reason": "Why did you choose this?",
  
  "confidence": 0.85,
  
  "learning_note": "What pattern are you following from past experience?"
}

IMPORTANT RULES:
================
1. Return ONLY valid JSON - no other text
2. All fields MUST be present
3. cpu_cores, gpu, memory_gb MUST be >= 0
4. estimated_duration_min MUST be > 0
5. confidence MUST be between 0 and 1
6. reason MUST be clear explanation
7. learning_note MUST reference past experience
Few-Shot Examples (Learning Examples)
Example 1: Episode 1 (First Time - Random)
Situation:

Code
Agent: Data Loader
Episode: 1 (no experience yet)
Resources: All available
Files: 5 CSV files
Deadline: None
Agent Thinking:

Code
"Hm, I've never done this before.
I have 5 files to load.
Let me try something..."
LLM Decision:

JSON
{
  "action": "run_task",
  "task_id": "load_001",
  "resources_requested": {
    "cpu_cores": 2,
    "gpu": 0,
    "memory_gb": 4
  },
  "estimated_duration_min": 30,
  "reason": "Load first file. Simple task, straightforward approach.",
  "confidence": 0.5,
  "learning_note": "First attempt, no prior experience to guide decision"
}
What Happened:

✅ Task completed in 30 min
✅ Others started waiting
❌ Reward: +15 points (low)
📚 Lesson Learned: "Sequential execution wastes time"
Example 2: Episode 5 (Learning - Optimization)
Situation:

Code
Agent: Data Loader  
Episode: 5 (has 4 episodes of experience)
Resources: CPU available, Cleaner ready to prep
Files: Remaining 3 CSV files
Deadline: None
Agent Thinking:

Code
"Last 4 episodes, I noticed:
- Episode 1: Sequential = low reward (15 pts)
- Episode 2: Sequential = still low (18 pts)
- Episode 3: Cleaner could prep in parallel (25 pts)
- Episode 4: Confirmed parallel works (28 pts)

Pattern: Parallel execution better than sequential!
I should load, and Cleaner can prep simultaneously."
LLM Decision:

JSON
{
  "action": "run_task",
  "task_id": "load_001",
  "resources_requested": {
    "cpu_cores": 2,
    "gpu": 0,
    "memory_gb": 4
  },
  "estimated_duration_min": 30,
  "reason": "Load file 1. Cleaner can start prep in parallel (different cores). Previous episodes show parallel > sequential.",
  "confidence": 0.8,
  "learning_note": "Episodes 1-4 taught that parallel execution gives 40% more reward. Using that insight now."
}
What Happened:

✅ Loader + Cleaner both run in parallel
✅ Better resource usage
✅ Reward: +35 points (much better!)
📚 Lesson Learned: "Parallel > Sequential confirmed!"
Example 3: Episode 15 (Expert - Strategic)
Situation:

Code
Agent: Data Cleaner
Episode: 15 (has 14 episodes of experience)
Resources: Loader will finish in 30 min
Timeline: 3 hour deadline, getting tight!
Others: Trainer has 4 hour deadline (tighter!)
Agent Thinking:

Code
"I've learned the pattern over 14 episodes:

1. Loader: 30 min
2. My cleaning: 30 min
3. Trainer training: 90 min
Total: 150 min = 2.5 hours
Deadline: 3 hours (safe!)

I should:
1. Start prep NOW (don't wait for Loader)
2. Do actual cleaning after Loader finishes
3. Let Trainer know they have 90 min of GPU time
4. Be reliable so others trust me

This coordination gave highest reward (50+ pts) in episodes 10-14"
LLM Decision:

JSON
{
  "action": "run_task",
  "task_id": "clean_001",
  "resources_requested": {
    "cpu_cores": 4,
    "gpu": 0,
    "memory_gb": 8
  },
  "estimated_duration_min": 30,
  "reason": "Loader finishing in 30 min. Start cleaning immediately after (I've been prepping). Total timeline: 30+30=60min. Deadline 3hrs - SAFE. Trainer gets predictable start time.",
  "confidence": 0.95,
  "learning_note": "Episodes 1-14 proved this exact sequence works. Reliable execution, clear communication = 50+ point rewards. Following proven strategy."
}
What Happened:

✅ Perfect coordination
✅ All agents on schedule
✅ Reward: +50 points (high!)
✅ Cooperation: 90% (excellent!)
📚 Lesson Learned: "Experience leads to mastery!"
Example 4: Episode 30 (Master - Proactive)
Situation:

Code
Agent: ML Trainer
Episode: 30 (has 29 episodes of experience!)
Resources: Cleaner will finish in ~60 min
Deadline: 4 hours (tight!)
Agent Thinking:

Code
"29 episodes of learning:

Pattern I see:
1. Loader + Cleaner = 60 min total
2. My training = 90 min
3. Buffer for delays = 10 min
Total = 160 min = 2.67 hours
Deadline = 4 hours = 160 min safety

Learned behaviors:
- Waiting is painful but necessary
- Clear deadline communication helps
- Others respect urgent deadlines
- Cooperation = highest total reward

What to do:
1. Wait for Cleaner (no choice)
2. Signal deadline urgency to others
3. Be ready to start immediately
4. Maintain confidence - I'll make it"
LLM Decision:

JSON
{
  "action": "wait",
  "task_id": null,
  "resources_requested": {
    "cpu_cores": 2,
    "gpu": 1,
    "memory_gb": 16
  },
  "estimated_duration_min": 90,
  "reason": "Waiting for Cleaner to finish (required dependency). My deadline is 4 hours, and I've calculated I'll finish in 2.67 hours. Will have 1.33 hour safety buffer. Being patient but firm.",
  "confidence": 0.98,
  "learning_note": "29 episodes taught: Patient but firm communication gets HIGHEST rewards (70+ pts). Cleaner will finish soon. I'll execute efficiently. This strategy has worked perfectly in episodes 25-29."
}
What Happened:

✅ Perfect timing
✅ All deadlines met
✅ Reward: +70+ points (highest!)
✅ Cooperation: 95% (perfect!)
✅ Total team score: 85+ points!
📚 Lesson Learned: "Mastery = consistent excellence!"
Response Parsing Logic
LLM return karega ye JSON. Tum isko parse karunga aur validate karunga:

Python
# Pseudocode for parsing

1. Get response from LLM
2. Find JSON part (between { and })
3. Parse as JSON
4. Validate all fields present:
   - action ∈ ["run_task", "wait", "offer_help"]
   - task_id (string or null)
   - resources_requested (dict with cpu, gpu, memory)
   - estimated_duration_min (> 0)
   - reason (non-empty string)
   - confidence (0-1)
   - learning_note (non-empty string)
5. If valid → use action
6. If invalid → retry or use default
Key Design Choices (Why This Format?)
Code
✅ 1. STRUCTURED JSON
   └─ Clear, unambiguous parsing
   └─ No "interpret my words" needed

✅ 2. REASONING REQUIRED
   └─ LLM must explain decision
   └─ Transparency = understandable behavior

✅ 3. CONFIDENCE SCORE
   └─ How sure is LLM about decision?
   └─ Low confidence = risky, high = safe

✅ 4. LEARNING NOTE
   └─ Which past experience is this using?
   └─ Shows learning is happening!

✅ 5. RESOURCE BREAKDOWN
   └─ CPU, GPU, Memory separate
   └─ Prevents "give me everything" requests

✅ 6. OBSERVATION WITH HISTORY
   └─ LLM sees current state + past events
   └─ Can pattern match and learn

✅ 7. FEW-SHOT EXAMPLES
   └─ Shows LLM how learning progresses
   └─ Episode 1 vs 30 comparison
Episode Progression (How Learning Shows)
Code
Episode 1:  confidence = 0.5, reward = 15 pts
Episode 5:  confidence = 0.8, reward = 35 pts
Episode 10: confidence = 0.90, reward = 45 pts
Episode 15: confidence = 0.95, reward = 50 pts
Episode 20: confidence = 0.96, reward = 60 pts
Episode 30: confidence = 0.98, reward = 70+ pts

Pattern: Higher confidence → Higher reward ↑
This shows learning working!
Testing These Prompts
Later, when you code:

Python
# 1. Build prompt with observation
prompt = build_prompt(observation, agent_history)

# 2. Send to LLM
response = call_llm(prompt)

# 3. Parse JSON
action = parse_json(response)

# 4. Execute action
env.step([action1, action2, action3])

# 5. Get reward
reward = env.rewards

# 6. Store for learning
agent.add_to_history(observation, action, reward)