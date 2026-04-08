from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import httpx
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")

TASKS = [
    "easy",
    "medium",
    "hard",
]
BENCHMARK = "healthcare-appointment-scheduling"

SYSTEM_PROMPT = (
    "You are a healthcare scheduling AI agent. Return exactly one action in JSON with keys "
    "action_type, patient_id, slot_id, and doctor_id. Pick only from available_actions and known entities."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_val = str(done).lower()
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards) if rewards else ""
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _default_action(observation: Dict[str, object], completed: set[Tuple[str, str]]) -> Dict[str, str]:
    """Return a safe default action"""
    patients = observation.get("patients", []) if isinstance(observation, dict) else []
    
    waiting_patients = [p for p in patients if isinstance(p, dict) and p.get("status") == "waiting"]
    
    if waiting_patients:
        patient = waiting_patients[0]
        return {
            "action_type": "assign_patient",
            "patient_id": str(patient.get("id", 1)),
            "slot_id": "0",
            "doctor_id": "1"
        }
    
    return {"action_type": "close_schedule"}


def _parse_action(content: str) -> Optional[Dict[str, str]]:
    text = (content or "").strip()
    if not text:
        return None

    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "action_type" in obj:
            return {
                "action_type": str(obj.get("action_type", "")),
                "patient_id": str(obj.get("patient_id", "1")),
                "slot_id": str(obj.get("slot_id", "0")),
                "doctor_id": str(obj.get("doctor_id", "1")),
            }
    except json.JSONDecodeError:
        pass

    return None


def choose_action(
    client: OpenAI,
    task_name: str,
    observation: Dict[str, object],
    completed: set[Tuple[str, str]],
) -> Dict[str, str]:
    user_prompt = json.dumps(
        {
            "task": task_name,
            "objective": observation.get("objective"),
            "step_count": observation.get("step_count"),
            "max_steps": observation.get("max_steps"),
            "recent_events": observation.get("recent_events"),
            "available_actions": observation.get("available_actions"),
            "patients": observation.get("patients"),
            "response_format": {"action_type": "string", "patient_id": "int", "slot_id": "int", "doctor_id": "int"},
        }
    )

    try:
        res = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=200,
            stream=False,
        )
        text = (res.choices[0].message.content or "").strip()
        parsed = _parse_action(text)
        if parsed:
            return parsed
    except Exception:
        pass

    return _default_action(observation, completed)


def run_task(client: OpenAI, http_client: httpx.Client, task_name: str) -> float:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    completed: set[Tuple[str, str]] = set()

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_resp = http_client.post(f"{ENV_BASE_URL}/reset", json={"task": task_name}, timeout=20.0)
        reset_resp.raise_for_status()
        payload = reset_resp.json()
        observation = payload.get("observation", {})

        max_steps = int(observation.get("max_steps") or 10)
        if MAX_STEPS_OVERRIDE:
            max_steps = int(MAX_STEPS_OVERRIDE)

        done = bool(payload.get("done", False))

        for step in range(1, max_steps + 1):
            if done:
                break

            action = choose_action(client=client, task_name=task_name, observation=observation, completed=completed)
            action_type = action.get("action_type", "assign_patient")
            action_str = f"{action_type}({action.get('patient_id', '?')})"

            step_resp = http_client.post(
                f"{ENV_BASE_URL}/step",
                json={
                    "action": {
                        "action_type": action_type,
                        "patient_id": int(action.get("patient_id", 1)),
                        "slot_id": int(action.get("slot_id", 0)),
                        "doctor_id": int(action.get("doctor_id", 1)),
                    }
                },
                timeout=20.0,
            )
            step_resp.raise_for_status()
            step_payload = step_resp.json()

            reward = float(step_payload.get("reward", {}).get("step_reward", 0.0) or 0.0)
            done = bool(step_payload.get("done", False))
            observation = step_payload.get("observation", {})

            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break

        score = float(observation.get("episode_score", 0.0) or 0.0)
        score = max(0.0, min(1.0, score))
        success = score > 0.0 and steps_taken > 0

    except Exception:
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is required")

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    with httpx.Client() as http_client:
        scores = []
        for task in TASKS:
            scores.append(run_task(client=client, http_client=http_client, task_name=task))

    _ = sum(scores) / len(scores) if scores else 0.0


if __name__ == "__main__":
    main()
