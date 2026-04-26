
# Negotiation Protocol

## Overview

Agents negotiate with each other.
Ye protocol define karta hai:
- **HOW** agents communicate
- **WHAT** messages exchange karenge
- **WHEN** negotiation hota hai
- **HOW** conflicts resolve hote hain

---

## Communication Model

Agents don't speak free English.
**JSON messages** exchange karte hain.

┌─────────────────┐ │ Agent 1 │ │ (Proposes) │ └────────┬────────┘ │ │ JSON message ↓ ┌─────────────────────────────────┐ │ System (Judge/Referee) │ │ - Check conflicts │ │ - Validate resources │ │ - Make decisions │ └────────┬────────────────────────┘ │ │ JSON feedback ↓ ┌─────────────────┐ │ Agent 2 │ │ (Responds) │ └─────────────────┘

Code

---

## Message Types

### MESSAGE TYPE 1: PROPOSE

Agent proposes to run a task.

```json
{
  "message_type": "propose",
  "from_agent": "data_loader",
  "timestamp": "2024-01-15T10:30:00",
  "episode": 5,
  "hour": 2,
  
  "proposal": {
    "action": "run_task",
    "task_id": "load_001",
    "cpu_cores_needed": 2,
    "gpu_needed": 0,
    "memory_needed": 4,
    "estimated_duration_min": 30
  },
  
  "reasoning": "First file needs to load. No dependencies.",
  "urgency": "normal",
  "confidence": 0.85
}
When sent: Every hour/round By whom: All agents simultaneously What happens: System analyzes all proposals

MESSAGE TYPE 2: SYSTEM RESPONSE (Feedback)
System responds with analysis.

Case A: No Conflict

JSON
{
  "message_type": "system_response",
  "to_agents": ["data_loader", "data_cleaner", "ml_trainer"],
  "status": "approved",
  "reason": "No resource conflicts detected",
  
  "allocation": {
    "data_loader": {
      "cpu_cores": 2,
      "gpu": 0,
      "memory": 4,
      "assigned_cores": [0, 1]
    },
    "data_cleaner": {
      "cpu_cores": 4,
      "gpu": 0,
      "memory": 8,
      "assigned_cores": [2, 3, 4, 5]
    },
    "ml_trainer": {
      "cpu_cores": 2,
      "gpu": 1,
      "memory": 16,
      "assigned_cores": [6, 7]
    }
  },
  
  "execution_plan": "All tasks can run in parallel",
  "estimated_completion_min": 90
}
Case B: Conflict Detected

JSON
{
  "message_type": "system_response",
  "to_agents": ["data_loader", "data_cleaner", "ml_trainer"],
  "status": "conflict_detected",
  "reason": "GPU conflict: 2 agents want GPU, only 1 available",
  
  "conflict_details": {
    "resource": "gpu",
    "demanded_by": ["data_cleaner", "ml_trainer"],
    "available": 1,
    "needed": 2
  },
  
  "next_step": "re_negotiation",
  "max_rounds": 3,
  "message": "Please modify proposals. Agents: data_cleaner and ml_trainer, one of you needs to change request."
}
MESSAGE TYPE 3: COUNTER_OFFER
Agent counters with different proposal.

JSON
{
  "message_type": "counter_offer",
  "from_agent": "data_cleaner",
  "to_agents": ["data_loader", "ml_trainer"],
  "timestamp": "2024-01-15T10:35:00",
  
  "original_proposal": {
    "action": "run_task",
    "task_id": "clean_001",
    "cpu_cores_needed": 4,
    "gpu_needed": 0
  },
  
  "counter_proposal": {
    "action": "run_task",
    "task_id": "prep_001",  # Different task!
    "cpu_cores_needed": 2,   # Less cores!
    "gpu_needed": 0,
    "estimated_duration_min": 20
  },
  
  "reason": "Loader still running. I'll do prep work first with 2 cores instead of waiting.",
  "rationale": "This frees up 4 cores for Trainer's GPU task. Both can run!",
  "benefits": "Parallel execution, better resource usage, higher team reward"
}
MESSAGE TYPE 4: AGREEMENT
Agent agrees with proposal.

JSON
{
  "message_type": "agreement",
  "from_agent": "ml_trainer",
  "to_agents": ["data_cleaner", "data_loader"],
  "timestamp": "2024-01-15T10:37:00",
  
  "agrees_with": "data_cleaner's counter_offer",
  "message": "Good idea! You prep, I'll setup. Loader goes first. Perfect coordination.",
  
  "proposed_sequence": [
    {
      "order": 1,
      "agent": "data_loader",
      "action": "run_task",
      "duration": "0-30 min"
    },
    {
      "order": 2,
      "agent": "data_cleaner",
      "action": "prep_001",
      "duration": "0-20 min (parallel with loader)"
    },
    {
      "order": 3,
      "agent": "data_cleaner",
      "action": "run_task",
      "duration": "30-60 min (after loader done)"
    },
    {
      "order": 4,
      "agent": "ml_trainer",
      "action": "run_task",
      "duration": "60-150 min (after cleaner done)"
    }
  ],
  
  "confidence": 0.95,
  "learning_note": "This sequence worked in episodes 10-14. High reward strategy."
}
MESSAGE TYPE 5: ESCALATE
Agent declares deadline is critical.

JSON
{
  "message_type": "escalate",
  "from_agent": "ml_trainer",
  "to_agents": ["data_loader", "data_cleaner"],
  "timestamp": "2024-01-15T10:38:00",
  
  "urgency_level": "CRITICAL",
  "deadline_hours": 4,
  "time_left_hours": 3.5,
  
  "message": "My deadline is becoming CRITICAL! 3.5 hours left, need 90 min for training.",
  "calculation": "90 min training + 60 min before = 150 min total. Need to start in 60 min MAX.",
  
  "request": "Please prioritize finishing my data. I can't wait beyond 1 hour.",
  "consequences": "If delayed > 1 hour, I'll miss deadline. Total team reward will be penalized.",
  
  "tone": "Urgent but respectful. Acknowledging others are working hard too."
}
MESSAGE TYPE 6: HELP_OFFER
Agent offers to help.

JSON
{
  "message_type": "help_offer",
  "from_agent": "data_loader",
  "to_agents": ["data_cleaner", "ml_trainer"],
  "timestamp": "2024-01-15T10:45:00",
  
  "message": "I finished early! Some cores free.",
  
  "available_resources": {
    "cpu_cores": 2,
    "memory_gb": 4,
    "gpu": 0
  },
  
  "offer": "Can I help with setup or prep work?",
  "willing_to": [
    "Help Cleaner with data structure prep",
    "Help Trainer with GPU environment setup",
    "Run additional preprocessing tasks"
  ]
}
Negotiation Flow (Complete Process)
Code
┌──────────────────────────────────────────────────────┐
│ NEGOTIATION FLOW (Each Round/Hour)                   │
└──────────────────────────────────────────────────────┘

STEP 1: OBSERVATION
─────────────────────
System sends observation to all agents
├─ Current resources available
├─ Other agents' status
├─ Time remaining
└─ Task queues

        ↓↓↓

STEP 2: PROPOSAL PHASE (Round 1)
─────────────────────────────────
All agents propose SIMULTANEOUSLY
├─ Agent 1 (Loader): "Run load_001"
├─ Agent 2 (Cleaner): "Wait for Loader"
└─ Agent 3 (Trainer): "Wait for Cleaner"

        ↓↓↓

STEP 3: CONFLICT DETECTION
─────────────��────────────
System analyzes proposals
├─ Check resource conflicts
├─ Check dependency issues
├─ Check deadline safety
└─ Decision:
   ├─ IF no conflict → APPROVE all
   └─ IF conflict → Go to STEP 4

        ↓↓↓

STEP 4: RE-NEGOTIATION (Max 3 rounds)
──────────────────────────────────────

ROUND 1 of re-negotiation:
├─ System: "Conflict! Modify proposals"
├─ Agent 1: "OK, I can wait"
├─ Agent 2: "I can start prep work instead"
├─ Agent 3: "I'll wait longer"
└─ System checks again

IF still conflict:
├─ ROUND 2: Agents try again
├─ ROUND 3: Last chance
└─ If still conflict: System decides

        ↓↓↓

STEP 5: FINAL DECISION
──────────────────────
System approves final plan
├─ Allocate cores
├─ Allocate GPU
├─ Allocate memory
└─ Lock in schedule

        ↓↓↓

STEP 6: EXECUTION
─────────────────
Tasks run for 30-90 minutes
├─ Monitor progress
├─ Detect failures
├─ Track actual time
└─ Compare to estimates

        ↓↓↓

STEP 7: OUTCOME & LEARNING
──────────────────────────
├─ Calculate rewards
├─ Store experience
├─ Update history
└─ Ready for next round!
Example Negotiation (Episode 5)
Initial Observation
JSON
{
  "episode": 5,
  "hour": 1,
  "resources_available": {
    "gpu": 1,
    "cpu_cores": 16,
    "memory_gb": 32
  },
  "other_agents_status": {
    "data_loader": {"status": "ready", "pending_tasks": 5},
    "data_cleaner": {"status": "idle", "pending_tasks": 8},
    "ml_trainer": {"status": "idle", "pending_tasks": 1}
  }
}
STEP 2: Proposals
Loader proposes:

JSON
{
  "message_type": "propose",
  "from_agent": "data_loader",
  "proposal": {
    "action": "run_task",
    "task_id": "load_001",
    "cpu_cores_needed": 2,
    "gpu_needed": 0,
    "memory_needed": 4,
    "estimated_duration_min": 30
  }
}
Cleaner proposes:

JSON
{
  "message_type": "propose",
  "from_agent": "data_cleaner",
  "proposal": {
    "action": "run_task",
    "task_id": "prep_001",
    "cpu_cores_needed": 2,
    "gpu_needed": 0,
    "memory_needed": 4,
    "estimated_duration_min": 20,
    "reason": "Loader running. I'll prep while waiting."
  }
}
Trainer proposes:

JSON
{
  "message_type": "propose",
  "from_agent": "ml_trainer",
  "proposal": {
    "action": "run_task",
    "task_id": "setup_001",
    "cpu_cores_needed": 1,
    "gpu_needed": 0,
    "memory_needed": 2,
    "estimated_duration_min": 15,
    "reason": "Setup GPU environment while waiting for data."
  }
}
STEP 3: Conflict Check
Code
Total needed:
├─ CPU: 2 + 2 + 1 = 5 cores
├─ GPU: 0
└─ Memory: 4 + 4 + 2 = 10 GB

Available:
├─ CPU: 16 cores ✅
├─ GPU: 1 ✅
└─ Memory: 32 GB ✅

RESULT: NO CONFLICT! ✅
All proposals APPROVED!
STEP 4: Allocation
JSON
{
  "message_type": "system_response",
  "status": "approved",
  "allocation": {
    "data_loader": {
      "action": "run_task",
      "task_id": "load_001",
      "cores": [0, 1],
      "start_time": "now",
      "duration": 30
    },
    "data_cleaner": {
      "action": "run_task",
      "task_id": "prep_001",
      "cores": [2, 3],
      "start_time": "now",
      "duration": 20
    },
    "ml_trainer": {
      "action": "run_task",
      "task_id": "setup_001",
      "cores": [4],
      "start_time": "now",
      "duration": 15
    }
  }
}
STEP 6: Execution (30 minutes later)
All tasks complete:

Loader: DONE ✅ (load_001 complete)
Cleaner: DONE ✅ (prep_001 complete)
Trainer: DONE ✅ (setup_001 complete)
STEP 7: Rewards
Code
Loader: +10 (completed) = 10 pts
Cleaner: +10 (completed) = 10 pts
Trainer: +10 (completed) = 10 pts

Team reward: +30 (all done + cooperation) = +30
Shared equally: +10 each

Final:
Loader: 0.6*10 + 0.4*10 = 10 pts
Cleaner: 0.6*10 + 0.4*10 = 10 pts
Trainer: 0.6*10 + 0.4*10 = 10 pts

Total episode: 30 points ✅
Conflict Example (More Complex)
Setup
Code
Resources: GPU 1, CPU 16, RAM 32
Loader wants: 2 CPU (simple)
Cleaner wants: 4 CPU (medium)
Trainer wants: 8 CPU + 1 GPU (aggressive estimate!)

Total: 14 CPU + 1 GPU = OK so far...
STEP 2: Proposals
JSON
[
  {
    "agent": "data_loader",
    "task": "load_001",
    "cpu": 2,
    "gpu": 0
  },
  {
    "agent": "data_cleaner",
    "task": "clean_001",
    "cpu": 4,
    "gpu": 0
  },
  {
    "agent": "ml_trainer",
    "task": "train_001",
    "cpu": 8,
    "gpu": 1
  }
]
STEP 3: Check
Code
Total: 14 CPU + 1 GPU
Available: 16 CPU + 1 GPU

Fits! No conflict... BUT WAIT!

Check dependencies:
├─ Cleaner depends on Loader (OK)
├─ Trainer depends on Cleaner (OK)
└─ Timeline:
   ├─ Loader: 30 min
   ├─ Cleaner: 40 min (after Loader)
   ├─ Trainer: 90 min (after Cleaner)
   ├─ Total: 160 min
   └─ Available time: 480 min (8 hours) ✅

Everything OK! APPROVE!
STEP 5: Allocation
Code
Phase 1 (0-30 min): Loader runs
├─ Cores: [0, 1]
├─ Status: Working

Phase 2 (30-70 min): Cleaner runs
├─ Cores: [0, 1, 2, 3]  # Reuse Loader's cores!
├─ Status: Working

Phase 3 (70-160 min): Trainer runs
├─ GPU: 1
├─ Cores: [0, 1, 2, 3, 4, 5, 6, 7]  # All 8 cores
├─ Status: Training

Success! All sequential, but safe timeline.
Key Rules
Rule 1: Simultaneous Proposals
Code
ALL agents propose AT THE SAME TIME
(not one after another)

Why? No agent knows what others will do
→ Honest proposals
→ No gaming the system
Rule 2: System is Fair Referee
Code
System doesn't favor any agent
Decisions based on:
├─ Resource availability
├─ Deadline urgency
├─ Task dependencies
└─ Fairness

NO AGENT CAN OVERRIDE SYSTEM!
Rule 3: Re-negotiation Limited
Code
Max 3 rounds of re-negotiation
(not infinite)

Why? Prevents endless loops
Forces agents to compromise
Teaches patience
Rule 4: Learning Enabled
Code
Agents remember negotiation outcomes
Episode 1: Learn conflicts happen
Episode 5: Predict potential conflicts
Episode 10: Proactively avoid conflicts
Episode 30: Expert negotiators!
Communication Rules
Always Include:
JSON
{
  "message_type": "...",           # Required
  "from_agent": "...",              # Required
  "timestamp": "...",               # Required
  "episode": 5,                     # Required
  "reasoning": "...",               # Required (explain decision)
  "confidence": 0.85                # Required (0-1 score)
}
Never Do:
Code
❌ Demand resources (be respectful)
❌ Lie about needs
❌ Coordinate with other agents secretly
❌ Try to bypass system rules
❌ Send unstructured text messages

✅ DO: Communicate clearly in JSON
✅ DO: Be honest about needs
✅ DO: Propose alternatives
✅ DO: Respect deadline urgency
Episode Evolution (Learning Negotiation)
Code
Episode 1-5 (CHAOTIC):
├─ Frequent conflicts
├─ 2-3 re-negotiation rounds needed
├─ Random proposals
└─ Low cooperation

Episode 6-15 (LEARNING):
├─ Some conflicts
├─ 1-2 re-negotiation rounds needed
├─ Better proposals
└─ Growing cooperation

Episode 16-30 (EXPERT):
├─ Rare conflicts
├─ Often approve in Round 1
├─ Strategic proposals
├─ High cooperation
└─ Agents anticipate needs!
Summary
Code
Negotiation = Core of multi-agent system
├─ All agents communicate fairly
├─ System enforces rules
├─ Rewards cooperation
└─ Agents learn to coordinate!

This is what makes project INTERESTING!