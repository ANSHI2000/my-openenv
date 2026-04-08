"""
Healthcare Appointment Scheduling - Baseline Inference Script
Uses OpenAI Client with environment variables for LLM calls
Implements [START]/[STEP]/[END] logging format
"""

import asyncio
import json
import os
import sys
from typing import List, Optional
from openai import OpenAI
import requests


# MANDATORY: Environment variables for LLM inference
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("API_KEY", os.getenv("HF_TOKEN", ""))  # Fallback to HF_TOKEN for backward compatibility
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "healthcare-scheduling")

# Environment constants
ENV_API_URL = os.getenv("ENV_API_BASE_URL", "http://localhost:8000")
TASK_NAME = os.getenv("TASK", "easy")
MAX_STEPS = 30
BENCHMARK = "healthcare-appointment-scheduling"


def log_start(task: str, env: str, model: str) -> None:
    """Log episode start in [START] format"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None) -> None:
    """Log step in [STEP] format"""
    error_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done}{error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Log end in [END] format"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def get_ai_action(client: OpenAI, observation: dict, step: int) -> Optional[dict]:
    """
    Get next action from AI model using OpenAI Client
    
    Args:
        client: OpenAI client instance
        observation: Current environment observation
        step: Current step number
        
    Returns:
        Parsed action dict or None
    """
    try:
        patients = observation.get("patients", [])
        appointments = observation.get("appointments", [])
        time_slots = observation.get("time_slots", [])
        recent_events = observation.get("recent_events", [])
        
        # Build scheduling state summary
        waiting_patients = [p for p in patients if p.get("status") == "waiting"]
        scheduled_count = len(appointments)
        
        prompt = f"""You are an expert healthcare scheduling AI agent.

CURRENT STATE:
- Task: {observation.get('task_name')}
- Objective: {observation.get('objective')}
- Step: {step}/{MAX_STEPS}
- Patients scheduled: {scheduled_count}/{len(patients)}
- Waiting patients: {len(waiting_patients)}
- Recent events: {[e.get('details') for e in recent_events[-2:]]}

AVAILABLE ACTIONS:
1. assign_patient: Assign waiting patient to available slot
2. escalate_urgent_case: Mark patient as emergency
3. reschedule_patient: Move scheduled patient (if allowed)
4. mark_no_show: Cancel patient appointment
5. close_schedule: Finalize and end session

STRATEGY:
- Schedule HIGH and EMERGENCY urgency patients first
- Assign urgent cases to priority slots (earliest times)
- Avoid double-booking conflicts
- Handle emergency walk-ins immediately

Choose the optimal action. Respond with ONLY valid JSON (no explanation):
{{"action_type": "assign_patient", "patient_id": 1, "slot_id": 0, "doctor_id": 1}}
OR
{{"action_type": "escalate_urgent_case", "patient_id": 2}}
OR
{{"action_type": "close_schedule"}}
"""
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        action = json.loads(response_text)
        return action
        
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON parse error: {e}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"[DEBUG] Model error: {e}", file=sys.stderr)
        return None


async def main() -> None:
    """Main inference loop"""
    
    # Validate API_KEY
    if not API_KEY:
        log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
        log_step(1, "<none>", 0.0, True, "API_KEY environment variable not set")
        log_end(False, 0, 0.0, [])
        return
    
    # Initialize OpenAI client with API_BASE_URL and API_KEY
    try:
        client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    except Exception as e:
        log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
        log_step(1, "<none>", 0.0, True, f"Failed to initialize OpenAI client: {e}")
        log_end(False, 0, 0.0, [])
        return
    
    # Log start
    log_start(TASK_NAME, BENCHMARK, MODEL_NAME)
    
    rewards: List[float] = []
    steps_taken = 0
    success = False
    
    try:
        # Reset environment
        reset_response = requests.post(
            f"{ENV_API_URL}/reset",
            json={"task": TASK_NAME},
            timeout=10
        )
        reset_response.raise_for_status()
        reset_data = reset_response.json()
        observation = reset_data.get("observation", {})
        
        # Main episode loop
        for step_num in range(1, MAX_STEPS + 1):
            # Get AI action
            action = await asyncio.to_thread(get_ai_action, client, observation, step_num)
            
            if not action or "action_type" not in action:
                log_step(step_num, "<none>", 0.0, False, "Model returned no action")
                break
            
            # Execute step
            try:
                step_response = requests.post(
                    f"{ENV_API_URL}/step",
                    json={"action": action},
                    timeout=10
                )
                step_response.raise_for_status()
                result = step_response.json()
            except Exception as e:
                log_step(step_num, str(action.get("action_type")), 0.0, False, f"Step failed: {e}")
                break
            
            # Extract results
            observation = result.get("observation", {})
            reward_obj = result.get("reward", {})
            step_reward = reward_obj.get("step_reward", 0.0)
            done = result.get("done", False)
            
            rewards.append(step_reward)
            steps_taken = step_num
            
            # Log step
            log_step(step_num, str(action.get("action_type")), step_reward, done)
            
            if done:
                break
        
        # Calculate final score from environment
        if observation and "episode_score" in observation:
            final_score = observation.get("episode_score", 0.0)
        else:
            final_score = sum(rewards) / len(rewards) if rewards else 0.0
        
        final_score = max(0.0, min(1.0, final_score))
        success = final_score > 0.0 and steps_taken > 0
    
    except Exception as e:
        print(f"[DEBUG] Episode error: {e}", file=sys.stderr)
        final_score = 0.0
    
    # Log end
    log_end(success, steps_taken, final_score, rewards)


if __name__ == "__main__":
    asyncio.run(main())