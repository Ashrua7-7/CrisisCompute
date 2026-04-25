# src/inference.py

import json
import os
import re
import time
from src.agents import Agent


class LLMAgent(Agent):
    """
    LLM-based Agent - uses ChatGPT/Ollama to decide actions
    Inherits from Agent base class
    """
    
    def __init__(self, name, resource_needs, llm_provider="ollama", model_name=None, base_url=None, timeout_s=180):
        """
        Initialize LLM Agent
        
        Args:
            name: Agent name
            resource_needs: Dict of resource requirements
            llm_provider: "ollama", "groq", or "openrouter"
            model_name: Optional model override for Ollama, Groq, or OpenRouter
            base_url: Optional Ollama base URL, e.g. http://localhost:11434 or a remote tunnel URL
            timeout_s: HTTP timeout for model calls
        """
        super().__init__(name, resource_needs)
        
        self.llm_provider = llm_provider
        self.model_name = model_name or os.getenv("LLM_MODEL", "mistral")
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")).rstrip("/")
        timeout_override = os.getenv("OLLAMA_TIMEOUT_S")
        self.timeout_s = int(timeout_override) if timeout_override else timeout_s
        self.ollama_retries = max(0, int(os.getenv("OLLAMA_RETRIES", "2")))
        self.ollama_num_predict = int(os.getenv("OLLAMA_NUM_PREDICT", "96"))
        self.llm_calls = 0
        self.llm_errors = 0
        self.last_raw_response = None
        
        # Setup LLM
        if llm_provider == "ollama":
            self.setup_ollama()
        elif llm_provider == "groq":
            self.setup_groq()
        elif llm_provider == "openrouter":
            self.setup_openrouter()
        else:
            print(f"❌ Unknown provider: {llm_provider}")
    
    def setup_ollama(self):
        """Setup local Ollama"""
        try:
            import requests
            # Test connection
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            response.raise_for_status()
            print(f"✅ {self.name}: Ollama connected at {self.base_url}")
            self.ollama_ready = True
        except Exception as e:
            print(f"❌ {self.name}: Ollama not available at {self.base_url} - {e}")
            self.ollama_ready = False
    
    def setup_groq(self):
        """Setup Groq API"""
        try:
            from groq import Groq
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                print(f"⚠️  {self.name}: GROQ_API_KEY not set")
                self.groq_client = None
                return
            self.groq_client = Groq(api_key=api_key)
            print(f"✅ {self.name}: Groq ready")
        except Exception as e:
            print(f"❌ {self.name}: Groq setup failed - {e}")
            self.groq_client = None
    
    def setup_openrouter(self):
        """Setup OpenRouter API"""
        try:
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                print(f"❌ {self.name}: OPENROUTER_API_KEY not set")
                print(f"   Please set: export OPENROUTER_API_KEY=your_api_key")
                self.openrouter_ready = False
                return
            
            import requests
            # Test connection with a simple models list request
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/gautam070205/Meta-Hack",
                "X-Title": "Meta-Hack"
            }
            response = requests.get("https://openrouter.ai/api/v1/models", headers=headers, timeout=5)
            response.raise_for_status()
            
            self.openrouter_api_key = api_key
            self.openrouter_ready = True
            print(f"✅ {self.name}: OpenRouter connected (Model: {self.model_name})")
        except Exception as e:
            print(f"❌ {self.name}: OpenRouter setup failed - {e}")
            self.openrouter_ready = False
    
    def build_prompt(self, observation, agent_type="loader"):
        """
        Build LLM prompt from observation + history
        
        Args:
            observation: Current state
            agent_type: "loader", "cleaner", or "trainer"
        
        Returns:
            Prompt string
        """
        
        # System message (agent personality)
        if agent_type == "loader":
            system_msg = """You are Data Loader Agent.
Your role: Load CSV files efficiently.
You have NO deadline. You're the foundation - others depend on you.
Be simple, straightforward, reliable.
Only choose tasks that belong to you."""
        elif agent_type == "cleaner":
            system_msg = """You are Data Cleaner Agent.
Your role: Clean and transform data.
You depend on Loader's output. Your deadline is 3 hours.
Be smart about waiting - do prep work while waiting.
Only choose tasks that belong to you."""
        else:  # trainer
            system_msg = """You are ML Trainer Agent.
Your role: Train machine learning models.
You depend on Cleaner's output. Your deadline is 4 hours (CRITICAL!).
This is the most important task. You can wait, but not forever.
Only choose tasks that belong to you."""
        
        # Current observation
        available_resources = observation.get('available_resources', {}) if isinstance(observation, dict) else {}

        def _resource_available(*resource_keys):
            for key in resource_keys:
                value = available_resources.get(key)
                if isinstance(value, dict):
                    return value.get('available', 0)
            return 0

        def _resource_total(*resource_keys):
            for key in resource_keys:
                value = available_resources.get(key)
                if isinstance(value, dict):
                    return value.get('total', 0)
            return 0

        other_agents_status = observation.get('other_agents_status', {}) if isinstance(observation, dict) else {}
        if isinstance(other_agents_status, list):
            other_agents_text = "\n".join(
                f"- {agent.get('agent_id', agent.get('name', 'unknown'))}: running={agent.get('running_task_id')}, completed={agent.get('completed_tasks', 0)}, missed_deadlines={agent.get('missed_deadlines', 0)}"
                for agent in other_agents_status
            ) or "- none"
        else:
            other_agents_text = f"""- Loader: {other_agents_status.get('data_loader', {}).get('status', 'unknown')}
- Cleaner: {other_agents_status.get('data_cleaner', {}).get('status', 'unknown')}
- Trainer: {other_agents_status.get('ml_trainer', {}).get('status', 'unknown')}"""

        obs_summary = f"""
Current Situation (Episode {observation.get('episode', 1)}, Hour {observation.get('hour', 1)}):
- Available CPU cores: {_resource_available('cpu_cores', 'cpu')}/{_resource_total('cpu_cores', 'cpu')}
- Available GPU: {_resource_available('gpu')}/{_resource_total('gpu')}
- Available Memory: {_resource_available('memory_gb', 'memory')}/{_resource_total('memory_gb', 'memory')} GB
- Time left: {observation.get('time_left_hours', 0)} hours

Other agents' status:
{other_agents_text}
"""
        
        # Learning history
        history = self.get_learning_summary()

        my_tasks = observation.get('my_tasks', {}) if isinstance(observation, dict) else {}
        pending_tasks = my_tasks.get('pending', []) if isinstance(my_tasks, dict) else []
        running_tasks = my_tasks.get('running', []) if isinstance(my_tasks, dict) else []
        done_tasks = my_tasks.get('done', []) if isinstance(my_tasks, dict) else []

        tasks_msg = f"""
    Your tasks (strict allow-list):
    - Pending task_ids you can run now: {pending_tasks}
    - Running task_ids: {running_tasks}
    - Done task_ids: {done_tasks}
    """
        
        # Decision prompt
        decision_msg = f"""
Based on above information and what you learned: {history}

DECIDE:
1. What task should you run?
2. How many resources do you need?
3. Should you wait for something?

Return ONLY this JSON (no other text):
{{
  "action": "run_task" or "wait",
  "task_id": "...",
  "cores_needed": number,
  "gpu_needed": 0 or 1,
  "memory_needed": number,
  "estimated_duration_min": number,
  "reasoning": "...",
  "confidence": 0.0
}}

Rules:
- If action is "run_task", task_id must exactly match one value from pending task_ids.
- Never invent task_id values like "1", "...", or placeholders.
- Use conservative resources close to your role's normal needs.

If you choose wait, set task_id to null and all resource fields to 0.
"""

        return system_msg + obs_summary + tasks_msg + decision_msg

    def _extract_json_candidate(self, response_text):
        """Extract a JSON object from raw model text."""
        if isinstance(response_text, dict):
            return response_text

        if not response_text:
            return None

        text = str(response_text).strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
        if fenced:
            return fenced.group(1)

        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            return text[first:last + 1]

        return None

    def _normalize_action(self, action):
        """Guarantee the environment receives a complete action payload."""
        normalized = {
            "action": action.get("action", "wait"),
            "task_id": action.get("task_id"),
            "cores_needed": int(action.get("cores_needed", 0) or 0),
            "gpu_needed": int(action.get("gpu_needed", 0) or 0),
            "memory_needed": int(action.get("memory_needed", 0) or 0),
            "estimated_duration_min": int(action.get("estimated_duration_min", 0) or 0),
            "reasoning": action.get("reasoning", ""),
        }
        if normalized["action"] == "wait":
            normalized.update({
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
            })
        return normalized

    def _sanitize_action_for_observation(self, action, observation):
        """Coerce model output into a valid action for the current observation."""
        if not isinstance(observation, dict):
            return action

        my_tasks = observation.get("my_tasks", {}) or {}
        pending_tasks = my_tasks.get("pending", []) if isinstance(my_tasks, dict) else []

        if not pending_tasks:
            return {
                "action": "wait",
                "task_id": None,
                "cores_needed": 0,
                "gpu_needed": 0,
                "memory_needed": 0,
                "estimated_duration_min": 0,
                "reasoning": "No pending tasks available for this agent.",
            }

        available_resources = observation.get("available_resources", {})
        cpu_info = available_resources.get("cpu") or available_resources.get("cpu_cores") or {}
        gpu_info = available_resources.get("gpu") or {}
        mem_info = available_resources.get("memory") or available_resources.get("memory_gb") or {}

        cpu_cap = int(cpu_info.get("available", cpu_info.get("total", self.resource_needs.get("cpu", 2))) or self.resource_needs.get("cpu", 2))
        gpu_cap = int(gpu_info.get("available", gpu_info.get("total", self.resource_needs.get("gpu", 0))) or self.resource_needs.get("gpu", 0))
        mem_cap = int(mem_info.get("available", mem_info.get("total", self.resource_needs.get("memory", 4))) or self.resource_needs.get("memory", 4))

        default_cpu = int(self.resource_needs.get("cpu", 2))
        default_gpu = int(self.resource_needs.get("gpu", 0))
        default_mem = int(self.resource_needs.get("memory", 4))

        if action.get("action") != "run_task":
            has_capacity = cpu_cap >= default_cpu and mem_cap >= default_mem and gpu_cap >= default_gpu
            if has_capacity:
                action = {
                    "action": "run_task",
                    "task_id": pending_tasks[0],
                    "cores_needed": default_cpu,
                    "gpu_needed": default_gpu,
                    "memory_needed": default_mem,
                    "estimated_duration_min": 60,
                    "reasoning": "Auto-switched from wait to run_task to avoid idle stall with available resources.",
                }
            else:
                return action

        if action.get("task_id") not in pending_tasks:
            action["task_id"] = pending_tasks[0]
            existing_reasoning = action.get("reasoning", "")
            action["reasoning"] = f"{existing_reasoning} Auto-corrected to valid pending task_id.".strip()

        action["cores_needed"] = max(1, min(cpu_cap, int(action.get("cores_needed", default_cpu) or default_cpu)))
        action["gpu_needed"] = max(0, min(gpu_cap, int(action.get("gpu_needed", default_gpu) or default_gpu)))
        action["memory_needed"] = max(1, min(mem_cap, int(action.get("memory_needed", default_mem) or default_mem)))

        if int(action.get("estimated_duration_min", 0) or 0) <= 0:
            action["estimated_duration_min"] = 60

        return action
    
    def call_ollama(self, prompt):
        """Call local Ollama"""
        try:
            import requests
            
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {
                    "temperature": 0.2,
                    "num_predict": self.ollama_num_predict
                }
            }

            for attempt in range(self.ollama_retries + 1):
                try:
                    print("Calling LLM...")
                    response = requests.post(url, json=payload, timeout=(5, self.timeout_s))
                    print("LLM responded")

                    response.raise_for_status()
                    result = response.json()
                    self.last_raw_response = result.get('response', '')
                    return self.last_raw_response
                except requests.exceptions.ReadTimeout as e:
                    if attempt >= self.ollama_retries:
                        raise e
                    print(f"⚠️  {self.name}: Ollama read timeout, retrying ({attempt + 1}/{self.ollama_retries})")
        except Exception as e:
            print(f"❌ {self.name}: Ollama error - {e}")
            self.llm_errors += 1
            return None
    
    def call_groq(self, prompt):
        """Call Groq API"""
        try:
            if not self.groq_client:
                return None
        
            import time, re
        
            for attempt in range(3):  # retry up to 3 times
                try:
                    response = self.groq_client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant. Respond with valid JSON only. No markdown, no code blocks, no extra text. Just the raw JSON object."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300,
                        temperature=0.2
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    err = str(e)
                    if '429' in err:
                        # Extract wait time from error message
                        match = re.search(r'try again in (\d+\.?\d*)s', err)
                        wait = float(match.group(1)) + 1 if match else 15
                        print(f"⏳ {self.name}: Rate limited, waiting {wait:.1f}s...")
                        time.sleep(wait)
                    else:
                        raise e
            return None
        
        except Exception as e:
            print(f"❌ {self.name}: Groq error - {e}")
            self.llm_errors += 1
            return None
    
    def call_openrouter(self, prompt):
        """Call OpenRouter API"""
        try:
            if not self.openrouter_ready:
                return None
            
            import requests
            
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "HTTP-Referer": "https://github.com/gautam070205/Meta-Hack",
                "X-Title": "Meta-Hack",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON only, no additional text."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 256,
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result and len(result["choices"]) > 0:
                self.last_raw_response = result["choices"][0]["message"]["content"]
                return self.last_raw_response
            else:
                return None
                
        except Exception as e:
            print(f"❌ {self.name}: OpenRouter error - {e}")
            self.llm_errors += 1
            return None
    
    def call_llm(self, prompt):
        """Call LLM based on provider"""
        self.llm_calls += 1
        
        if self.llm_provider == "ollama":
            return self.call_ollama(prompt)
        elif self.llm_provider == "groq":
            return self.call_groq(prompt)
        elif self.llm_provider == "openrouter":
            return self.call_openrouter(prompt)
        else:
            return None
    
    def parse_response(self, response_text):
        """
        Extract JSON from LLM response
        
        Args:
            response_text: Raw LLM response
        
        Returns:
            Parsed JSON dict
        """
        if not response_text:
            return {"action": "wait", "task_id": None}
        
        try:
            candidate = self._extract_json_candidate(response_text)
            if candidate is None:
                return {"action": "wait", "task_id": None}

            if isinstance(candidate, dict):
                action = candidate
            else:
                action = json.loads(candidate)

            return self._normalize_action(action)
        except json.JSONDecodeError as e:
            print(f"❌ {self.name}: JSON parse error - {e}")
            return {"action": "wait", "task_id": None}
        except Exception as e:
            print(f"❌ {self.name}: Response parse error - {e}")
            return {"action": "wait", "task_id": None}
    
    def propose_action(self, observation):
        """
        Main method: Use LLM to decide action
        
        Args:
            observation: Current state
        
        Returns:
            Action dict
        """
        
        # Determine agent type
        if "loader" in self.name:
            agent_type = "loader"
        elif "cleaner" in self.name:
            agent_type = "cleaner"
        else:
            agent_type = "trainer"
        
        # Build prompt
        prompt = self.build_prompt(observation, agent_type)
        
        # Call LLM
        response_text = self.call_llm(prompt)
        
        if not response_text:
            # LLM failed, use fallback
            print(f"⚠️  {self.name}: LLM failed, using fallback")
            return {
                "action": "wait",
                "task_id": None,
                "reasoning": "LLM unavailable, fallback"
            }
        
        # Parse response
        action = self.parse_response(response_text)
        action = self._sanitize_action_for_observation(action, observation)
        
        # Store in history for learning
        self.add_to_history(
            state=observation,
            action=action,
            reward=None,  # Will update later
            outcome="proposed"
        )
        
        return action


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING LLM AGENT CLASSES")
    print("="*60)
    
    # Test 1: Create LLM agents (with fallback if Ollama not available)
    print("\n[TEST 1] Creating LLM Agents")
    
    try:
        loader = LLMAgent("data_loader", {"cpu": 2, "memory": 4, "gpu": 0})
        cleaner = LLMAgent("data_cleaner", {"cpu": 4, "memory": 8, "gpu": 0})
        trainer = LLMAgent("ml_trainer", {"cpu": 2, "memory": 16, "gpu": 1})
        
        print(f"✅ Agents created")
    except Exception as e:
        print(f"❌ Agent creation failed: {e}")
        print("⚠️  Make sure Ollama is running!")
        exit(1)
    
    # Test 2: Try to propose action (may fail if Ollama not running)
    print("\n[TEST 2] Testing Proposals")
    
    observation = {
        "episode": 1,
        "hour": 1,
        "available_resources": {
            "cpu_cores": {"available": 16, "total": 16},
            "gpu": {"available": 1, "total": 1},
            "memory_gb": {"available": 32, "total": 32}
        },
        "other_agents_status": {
            "data_loader": {"status": "idle"},
            "data_cleaner": {"status": "idle"},
            "ml_trainer": {"status": "idle"}
        },
        "time_left_hours": 8
    }
    
    print(f"\n⏳ Calling Loader LLM... (may take a few seconds)")
    action = loader.propose_action(observation)
    
    if action:
        print(f"✅ Loader action: {action.get('action')}")
        if action.get('task_id'):
            print(f"   Task: {action.get('task_id')}")
        print(f"   Cores: {action.get('cores_needed', 'N/A')}")
    else:
        print("❌ Loader proposal failed (is Ollama running?)")
    
    print("\n" + "="*60)
    print("✅ LLM AGENT TESTS COMPLETE!")
    print("="*60 + "\n")
    print("⚠️  NOTE: If Ollama tests failed, make sure:")
    print("    1. Run: ollama serve")
    print("    2. In another terminal: ollama pull mistral")
    print("    3. Then run this test again")
    print("")