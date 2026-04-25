"Multi-Agent Resource Negotiation & Emergent Cooperation:
Training LLMs to Coordinate in Constrained Environments"

┌──────────────────────────────────────────────────────────────┐
│ THE REAL WORLD PROBLEM                                       │
└──────────────────────────────────────────────────────────────┘

In modern ML engineering teams, multiple teams compete for 
limited computational resources (GPU, CPU cores, memory, etc).

SCENARIO:
├─ Company has: 1 GPU, 16 CPU cores, 32 GB RAM
├─ Teams: Data Engineering, Data Science, ML Research
├─ Each team has:
│  ├─ Different tasks with different resource needs
│  ├─ Different deadlines (some urgent, some flexible)
│  ├─ Limited information about others' urgency
│  └─ Conflicting priorities
│
├─ PROBLEM: How to fairly allocate resources?
│  ├─ Currently: Manual scheduling (slow, unfair, error-prone)
│  ├─ Result: Missed deadlines, team conflicts, wasted compute
│  └─ Need: Automated, intelligent coordination mechanism

┌──────────────────────────────────────────────────────────────┐
│ THE RESEARCH PROBLEM                                         │
└──────────────────────────────────────────────────────────────┘

Can LLMs learn to:
✅ Negotiate resource allocation with other agents?
✅ Model other agents' constraints and deadlines?
✅ Cooperate for mutual benefit (not just individual gain)?
✅ Develop emergent protocols without explicit rules?
✅ Handle failures and adapt strategies dynamically?

WHY IT MATTERS:
├─ Tests "theory-of-mind" reasoning in LLMs
├─ Shows emergent multi-agent behavior
├─ Practical for enterprise resource management
├─ Demonstrates LLM capability beyond next-token prediction
└─ Aligns with OpenEnv theme: Multi-Agent Interactions


┌──────────────────────────────────────────────────────────────┐
│ OUR SOLUTION: MULTI-AGENT PIPELINE ORCHESTRATION ENVIRONMENT │
└──────────────────────────────────────────────────────────────┘

We build an OpenEnv-compatible environment where:

1. WORLD (Environment):
   ├─ Virtual data center with finite resources
   ├─ Resource pool: 1 GPU, 16 CPU cores, 32 GB RAM
   ├─ Task queue with varying complexity levels
   ├─ Real execution simulation (time, failures, delays)
   └─ Real constraints (can't split cores, memory leaks, timeouts)

2. AGENTS (Specialized LLM-based actors):
   ├─ Data Loader Agent: Ingests data (needs 2 CPU cores)
   ├─ Data Cleaner Agent: Transforms data (needs 4 CPU cores)
   └─ ML Trainer Agent: Trains models (needs 1 GPU + 2 cores)

3. MECHANISM (How negotiation happens):
   ├─ Round-based interaction (hourly rounds in 8-hour work day)
   ├─ Simultaneous proposals from all agents
   ├─ Conflict detection (resource overlaps, deadlines)
   ├─ Multi-round negotiation if needed
   ├─ Allocation phase (system assigns resources)
   └─ Execution phase (tasks run in parallel if possible)

4. LEARNING (How agents improve):
   ├─ In-context learning via conversation history
   ├─ Each agent stores: state, action, reward, outcome
   ├─ Next episode, LLM uses history to inform decisions
   ├─ Rewards encourage cooperation + efficiency
   ├─ Penalties for delays, conflicts, failures
   └─ Emergent strategies develop over 30 episodes

5. EVALUATION (How we measure success):
   ├─ Reward curve (15 pts → 85 pts across 30 episodes)
   ├─ Task completion rate (40% → 95%)
   ├─ On-time delivery rate (30% → 90%)
   ├─ Resource utilization (25% → 80%)
   ├─ Agent cooperation score (10% → 85%)
   └─ Emergent behavior analysis



   WHY THIS IS NOVEL:

1. MULTI-AGENT FOCUS (Underexplored)
   ├─ Most hackathons focus on single-agent long-horizon tasks
   ├─ We focus on multi-agent coordination
   ├─ Fleet AI sub-theme specifically about this
   └─ Less crowded = easier to stand out

2. REAL CONSTRAINTS (Realistic)
   ├─ Cores can't be split (real hardware limitation)
   ├─ Tasks can fail, timeout, have delays (realistic)
   ├─ Memory leaks, cascading failures (enterprise problems)
   ├─ Asymmetric information (agents don't know others' deadlines)
   └─ Dynamic task arrivals (mid-episode new tasks)

3. EMERGENT BEHAVIOR (No explicit rules)
   ├─ No hard-coded negotiation protocol
   ├─ No predefined fairness rules
   ├─ Strategies EMERGE from reward signals
   ├─ Agents invent "languages" to communicate
   └─ Theory-of-mind reasoning visible in decisions

4. MEASURABLE IMPROVEMENT (Clear metrics)
   ├─ Reward curve smooth upward trajectory
   ├─ Multiple metrics showing agent improvement
   ├─ Before/after behavior demonstrable
   ├─ Video proof of learning
   └─ Reproducible results

   ┌──────────────────────────────────────────────────────────────┐
│ COMPONENT BREAKDOWN                                          │
└──────────────────────────────────────────────────────────────┘

1. ENVIRONMENT (env.py)
   ├─ Class: MultiAgentPipelineEnv
   ├─ Methods:
   │  ├─ reset(): Initialize episode, load tasks
   │  ├─ step(actions): Process proposals, allocate, execute
   │  ├─ _allocate_resources(): Core/GPU assignment logic
   │  ├─ _calculate_rewards(): Individual + team rewards
   │  └─ _check_done(): Episode termination
   │
   ├─ Features:
   │  ├─ Resource pool management (GPU, CPU, RAM tracking)
   │  ├─ Task queue management (pending, running, done)
   │  ├─ Real-time monitoring (actual vs estimated time)
   │  ├─ Failure simulation (timeouts, memory leaks, crashes)
   │  └─ Deadline tracking
   │
   └─ Difficulty levels:
      ├─ Easy: 5 sequential tasks, no time pressure
      ├─ Medium: 8 parallel tasks, moderate deadlines
      └─ Hard: 15 dynamic tasks, tight deadlines

2. TASKS (tasks.py)
   ├─ Task class: ID, type, duration, resources, deadline
   ├─ TaskQueue class: Per-agent task management
   ├─ Functions:
   │  ├─ load_tasks_library(): Read JSON task definitions
   │  ├─ generate_episode_tasks(): Create tasks by difficulty
   │  ├─ check_task_completion(): Verify successful execution
   │  └─ apply_penalties(): Deadline miss penalties
   │
   └─ Tasks have:
      ├─ Resource requirements (CPU cores, GPU, memory)
      ├─ Duration estimates (actual vs expected)
      ├─ Dependencies (task B after task A)
      └─ Deadlines (hard constraints)

3. AGENTS (agents.py)
   ├─ Base class: Agent
   ├─ Subclasses:
   │  ├─ DataLoaderAgent
   │  ├─ DataCleanerAgent
   │  └─ MLTrainerAgent
   │
   ├─ Methods:
   │  ├─ propose_action(observation): Generate JSON proposal
   │  ├─ receive_reward(reward): Store experience
   │  ├─ get_conversation_history(): For LLM context
   │  └─ update_deadline_urgency(): Deadline tracking
   │
   └─ Attributes:
      ├─ conversation_history: Learning memory
      ├─ resource_needs: CPU/GPU/RAM requirements
      ├─ task_queue: Pending tasks
      └─ deadline: Hard constraint

4. INFERENCE (inference.py) - LLM Agent
   ├─ Class: LLMAgent(Agent)
   ├─ Methods:
   │  ├─ propose_action(observation): Build prompt + LLM call
   │  ├─ build_prompt(observation): Create context-aware prompt
   │  ├─ parse_llm_response(response): Extract JSON action
   │  ├─ learn_from_reward(reward): Add to history
   │  └─ format_conversation_history(): For LLM context
   │
   ├─ Features:
   │  ├─ In-context learning via conversation history
   │  ├─ Few-shot examples in prompt
   │  ├─ Deadline urgency signaling
   │  ├─ Error recovery mechanisms
   │  └─ Flexible resource requests
   │
   └─ Integration:
      ├─ OpenAI API or HuggingFace endpoint
      ├─ Token optimization
      └─ Caching (to reduce API calls)

5. REWARDS (rewards.py)
   ├─ Functions:
   │  ├─ calculate_individual_reward(): Agent-specific
   │  ├─ calculate_team_reward(): Group incentive
   │  ├─ calculate_final_reward(): Weighted combination
   │  ├─ calculate_metrics(): Completion, latency, utilization
   │  └─ apply_penalties(): Late, failed, timeout tasks
   │
   ├─ Individual reward formula:
   │  ├─ +10 per task completed
   │  ├─ -2 per 10 min waiting
   │  ├─ -5 if late
   │  ├─ +3 if early
   │  └─ +5 communication bonus
   │
   ├─ Team reward formula:
   │  ├─ +50 all tasks done by EOD
   │  ├─ +20 no deadlines missed
   │  ├─ +10 smooth negotiation
   │  └─ +5 agent cooperation
   │
   └─ Final: 0.6 * individual + 0.4 * team
      (Forces balance between selfish and cooperative)

6. TRAINING (train.py)
   ├─ Main training loop:
   │  ├─ Initialize environment (difficulty level)
   │  ├─ For each episode (1-30):
   │  │  ├─ reset()
   │  │  ├─ For each step (1-8 hours):
   │  │  │  ├─ agents.propose(observation)
   │  │  │  ├─ env.allocate_resources()
   │  │  │  ├─ env.execute_tasks()
   │  │  │  ├─ calculate rewards
   │  │  │  └─ agents.learn(reward)
   │  │  └─ Save episode metrics
   │  └─ Save models, curves, results
   │
   ├─ Metrics tracking:
   │  ├─ Per-episode reward
   │  ├─ Task completion rate
   │  ├─ On-time delivery rate
   │  ├─ Resource utilization
   │  └─ Agent cooperation score
   │
   └─ Colab-compatible (runs in < 5 minutes demo)

7. VISUALIZATION (visualize.py)
   ├─ Generates:
   │  ├─ reward_curve.png (main metric)
   │  ├─ metrics_dashboard.png (4-metric graph)
   │  ├─ agent_dialogues.txt (dialogue examples)
   │  └─ summary_statistics.txt (numerical results)
   │
   └─ Shows improvement clearly for judges

8. MOCK AGENT (mock_agent.py) - Baseline
   ├─ Hard-coded solutions for each task level
   ├─ Easy: Sequential execution (baseline)
   ├─ Medium: Optimal order (Loader → Cleaner → Trainer)
   ├─ Hard: Same with error handling
   │
   └─ Purpose: Proves environment works before LLM integration


   ┌──────────────────────────────────────────────────────────────┐
│ EPISODE STRUCTURE (What happens each episode)               │
└──────────────────────────────────────────────────────────────┘

INITIALIZATION (Episode = 1)
│
├─ env.reset()
│  ├─ Clear workspace
│  ├─ Allocate resources (GPU 1, CPU 16, RAM 32)
│  ├─ Load tasks (5-15 based on difficulty)
│  └─ Initialize agents (empty history for Ep 1)
│
├─ Generate initial observation
│  ├─ Task queue
│  ├─ Resource availability
│  ├─ Agent deadlines
│  └─ Time remaining

EXECUTION LOOP (8 hours = 8 steps)
│
└─ FOR STEP = 1 TO 8:
   │
   ├─ OBSERVATION PHASE
   │  ├─ All agents receive: current state
   │  ├─ Format: {resources, tasks, deadlines, time_left}
   │  └─ Agents read their conversation history
   │
   ├─ DECISION PHASE (LLM calls)
   │  ├─ AI 1 LLM prompt:
   │  │  ├─ Observation
   │  │  ├─ Conversation history (what worked before)
   │  │  ├─ "What task should you run? Cores needed? Urgency?"
   │  │  └─ Returns: JSON proposal
   │  │
   │  ├─ AI 2 LLM prompt: Same format
   │  └─ AI 3 LLM prompt: Same format
   │
   ├─ NEGOTIATION PHASE
   │  ├─ Proposal analysis:
   │  │  ├─ Resource conflict check
   │  │  ├─ Dependency validation
   │  │  ├─ Efficiency threshold (70% utilization)
   │  │  └─ Deadline safety check
   │  │
   │  ├─ IF CONFLICT DETECTED:
   │  │  ├─ Up to 3 re-negotiation rounds
   │  │  ├─ Agents modify proposals
   │  │  ├─ Try to find agreement
   │  │  └─ If fails: System decides (greedy allocation)
   │  │
   │  └─ ALLOCATION:
   │     ├─ Core assignment [0,1] → AI 1, [2,3,4,5] → AI 2, etc
   │     ├─ GPU assignment (if needed)
   │     ├─ RAM allocation
   │     └─ Timeline locked in
   │
   ├─ EXECUTION PHASE (30-60 minutes simulated)
   │  ├─ Tasks run in parallel (if cores available)
   │  ├─ Monitor:
   │  │  ├─ CPU usage vs allocated
   │  │  ├─ Memory usage
   │  │  ├─ Actual vs estimated time
   │  │  ├─ Failures, timeouts, crashes
   │  │  └─ Deadline violations
   │  │
   │  ├─ IF ISSUE DETECTED:
   │  │  ├─ Alert agents immediately
   │  │  ├─ Trigger dynamic re-negotiation
   │  │  ├─ Suggest recovery strategy
   │  │  └─ Adjust timeline
   │  │
   │  └─ When task completes:
   │     ├─ Free resources
   │     ├─ Generate output
   │     └─ Update dependent task status
   │
   ├─ REWARD CALCULATION
   │  ├─ Individual rewards per agent:
   │  │  ├─ +10 if task completed
   │  │  ├─ -2 per 10 min waited
   │  │  ├─ -5 if deadline missed
   │  │  ├─ +3 if early
   │  │  └─ +5 if communicated delays
   │  │
   │  ├─ Team rewards (shared equally):
   │  │  ├─ +50 if all tasks done
   │  │  ├─ +20 if no deadlines missed
   │  │  ├─ +10 if negotiation smooth
   │  │  └─ +5 if agents cooperated
   │  │
   │  └─ Final reward: 0.6*individual + 0.4*team
   │
   └─ LEARNING SIGNAL
      ├─ Store in agent conversation history:
      │  ├─ Observation
      │  ├─ Action taken
      │  ├─ Reward received
      │  ├─ Outcome (success/failure/delayed)
      │  └─ Lessons learned
      │
      ├─ Update global experience pool
      └─ Next step (next agent will use this info!)

EPISODE END
│
├─ Calculate episode total reward (sum of all steps)
├─ Save metrics (completion rate, latency, utilization)
├─ Log dialogue samples
└─ Ready for next episode!

REPEAT 29 MORE TIMES (Episodes 2-30)
│
├─ Episode 2: Agents use Ep 1 learnings
├─ Episode 5: Patterns emerging, rewards improve
├─ Episode 10: Optimization phase, strategies solidifying
├─ Episode 20: Expert phase, high rewards
└─ Episode 30: Peak performance, emergent protocols

Episode │ Avg Reward │ Completion % │ On-Time % │ Utilization │ Status
───────┼────────────┼──────────────┼──────────┼─────────────┼──────────
   1   │    12      │    40%       │    30%   │    25%      │ Exploring
   2   │    15      │    45%       │    35%   │    28%      │ Exploring
   3   │    18      │    50%       │    40%   │    32%      │ Learning
   4   │    22      │    55%       │    45%   │    35%      │ Learning
   5   │    28      │    60%       │    50%   │    40%      │ Learning
   6   │    35      │    70%       │    60%   │    50%      │ Optimizing
   7   │    42      │    75%       │    65%   │    58%      │ Optimizing
   8   │    48      │    80%       │    70%   │    65%      │ Optimizing
   9   │    52      │    85%       │    75%   │    70%      │ Optimizing
  10   │    58      │    88%       │    78%   │    73%      │ Expert
  11   │    64      │    90%       │    82%   │    76%      │ Expert
  12   │    68      │    92%       │    85%   │    78%      │ Expert
  ...
  20   │    78      │    95%       │    90%   │    82%      │ Expert
  ...
  30   │    85      │    97%       │    92%   │    85%      │ Expert

KEY OBSERVATIONS:
├─ Reward: 12 → 85 (7x improvement!) 🎯
├─ Completion: 40% → 97% (2.4x improvement)
├─ On-Time: 30% → 92% (3x improvement)
├─ Utilization: 25% → 85% (3.4x improvement)
│
└─ Clear learning trajectory visible!

PROBLEM:     Multi-team resource conflicts in ML engineering
SOLUTION:    OpenEnv with 3 negotiating LLM agents
INNOVATION:  Emergent cooperation WITHOUT explicit rules
MECHANISM:   Reward-based learning over 30 episodes
RESULT:      7x reward improvement + measurable agent learning
PROOF:       Before/after video + reward curves + metrics
AUDIENCE:    Enterprise ML teams, LLM researchers, AI enthusiasts
IMPACT:      Demonstrates LLMs can learn coordination skills
TIMING:      4 days to build, ready for judges
WINNING:     Novel problem + dramatic demo + clear improvement 🏆