#!/usr/bin/env python
"""
Healthcare Appointment Scheduling - Baseline Inference
Uses OpenAI client with LiteLLM proxy for intelligent scheduling
"""

import os
import re
from typing import Optional

from openai import OpenAI

from models import ResetRequest, Action, ActionType, UrgencyLevel
from env.healthcare_env import HealthcareSchedulingEnv
from graders import grade_easy, grade_medium, grade_hard


# Load configuration from environment - REQUIRED for LLM proxy
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")


def initialize_client() -> OpenAI:
    """Initialize OpenAI client using LiteLLM proxy with provided credentials"""
    return OpenAI(
        api_key=HF_TOKEN,
        base_url=API_BASE_URL,
    )


def validate_action(action: Action, observation: dict) -> bool:
    """
    Validate action IDs exist and constraints are met.
    Case 4-8: Prevent invalid patient/slot/doctor IDs and constraint violations
    """
    if action.action_type == ActionType.CLOSE_SCHEDULE:
        return True
    
    if action.action_type != ActionType.ASSIGN_PATIENT:
        return True
    
    patients = observation.get("patients", [])
    doctors = observation.get("doctors", [])
    time_slots = observation.get("time_slots", [])
    appointments = observation.get("appointments", [])
    
    # Case 4: Validate patient exists
    patient_ids = [p.get("id") for p in patients]
    if action.patient_id not in patient_ids:
        return False
    
    # Case 5: Validate slot exists
    slot_ids = [s.get("slot_id") for s in time_slots]
    if action.slot_id not in slot_ids:
        return False
    
    # Case 6: Validate doctor exists
    doctor_ids = [d.get("id") for d in doctors]
    if action.doctor_id not in doctor_ids:
        return False
    
    # Case 7: Check doctor workload not exceeded
    doctor = next((d for d in doctors if d.get("id") == action.doctor_id), None)
    if doctor:
        current_load = doctor.get("current_load", 0)
        max_load = doctor.get("max_patients_per_session", 5)
        if current_load >= max_load:
            return False
    
    # Case 8: Check slot not already assigned (conflict detection)
    slot_reserved = any(a.get("slot_id") == action.slot_id for a in appointments)
    if slot_reserved:
        return False
    
    return True


def get_llm_action(client: OpenAI, task_name: str, observation: dict, env: HealthcareSchedulingEnv) -> Action:
    """
    Use LLM via proxy to determine next scheduling action.
    This makes an actual API call through API_BASE_URL.
    """
    patients = observation.get("patients", [])
    doctors = observation.get("doctors", [])
    time_slots = observation.get("time_slots", [])
    scheduled = observation.get("patients_scheduled", 0)
    total = len(patients)
    
    # Build context
    prompt = f"""You are a healthcare scheduler. Task: {task_name}
Status: {scheduled}/{total} scheduled.
"""
    
    unscheduled = [p for p in patients if p.get("status") != "SCHEDULED"]
    if unscheduled:
        prompt += f"Next patient: ID {unscheduled[0].get('id')}, urgency {unscheduled[0].get('urgency')}\n"
    
    prompt += """If all scheduled, respond: CLOSE_SCHEDULE
Otherwise respond EXACTLY: ACTION ASSIGN_PATIENT|patient_id|slot_id|doctor_id
Available patients: [" + ", ".join(str(p.get('id')) for p in unscheduled[:3]) + "]
"""
    
    try:
        # API call goes through API_BASE_URL configured in client
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a healthcare scheduling expert. Be concise."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100,
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Case 9: Robust parsing - try multiple formats
        if "CLOSE" in response_text.upper():
            return Action(action_type=ActionType.CLOSE_SCHEDULE)
        
        # Try format: ACTION ASSIGN_PATIENT|1|2|3
        if "ACTION" in response_text and "ASSIGN" in response_text:
            parts = response_text.split("|")
            if len(parts) >= 4:
                try:
                    patient_id = int(parts[1].strip())
                    slot_id = int(parts[2].strip())
                    doctor_id = int(parts[3].strip())
                    candidate_action = Action(
                        action_type=ActionType.ASSIGN_PATIENT,
                        patient_id=patient_id,
                        slot_id=slot_id,
                        doctor_id=doctor_id
                    )
                    # Case 4-8: Validate before returning
                    if validate_action(candidate_action, observation):
                        return candidate_action
                except (ValueError, IndexError):
                    pass
        
        # Try alternate format without ACTION keyword
        if "ASSIGN" in response_text and "|" in response_text:
            try:
                # Extract numbers from: "assign patient 1 to slot 2 with doctor 3"
                numbers = re.findall(r'\d+', response_text)
                if len(numbers) >= 3:
                    candidate_action = Action(
                        action_type=ActionType.ASSIGN_PATIENT,
                        patient_id=int(numbers[0]),
                        slot_id=int(numbers[1]),
                        doctor_id=int(numbers[2])
                    )
                    if validate_action(candidate_action, observation):
                        return candidate_action
            except (ValueError, IndexError):
                pass
        
        # Fallback greedy assignment - with constraint checking
        unscheduled_patients = [p for p in patients if p.get("status") != "SCHEDULED"]
        if unscheduled_patients:
            available_slots = [s for s in time_slots if s.get("available")]
            available_doctors = [d for d in doctors if d.get("current_load", 0) < d.get("max_patients_per_session", 5)]
            # Filter out already-assigned slots (Case 8)
            appointments = observation.get("appointments", [])
            reserved_slot_ids = {a.get("slot_id") for a in appointments}
            available_slots = [s for s in available_slots if s.get("slot_id") not in reserved_slot_ids]
            
            if available_slots and available_doctors:
                fallback_action = Action(
                    action_type=ActionType.ASSIGN_PATIENT,
                    patient_id=unscheduled_patients[0].get("id"),
                    slot_id=available_slots[0].get("slot_id"),
                    doctor_id=available_doctors[0].get("id")
                )
                if validate_action(fallback_action, observation):
                    return fallback_action
        
        return Action(action_type=ActionType.CLOSE_SCHEDULE)
        
    except Exception as e:
        print(f"LLM call error: {e}")
        # Fallback to greedy without API - with constraint checking
        unscheduled = [p for p in patients if p.get("status") != "SCHEDULED"]
        if unscheduled:
            available_slots = [s for s in time_slots if s.get("available")]
            available_doctors = [d for d in doctors if d.get("current_load", 0) < d.get("max_patients_per_session", 5)]
            # Filter out already-assigned slots
            appointments = observation.get("appointments", [])
            reserved_slot_ids = {a.get("slot_id") for a in appointments}
            available_slots = [s for s in available_slots if s.get("slot_id") not in reserved_slot_ids]
            
            if available_slots and available_doctors:
                fallback_action = Action(
                    action_type=ActionType.ASSIGN_PATIENT,
                    patient_id=unscheduled[0].get("id"),
                    slot_id=available_slots[0].get("slot_id"),
                    doctor_id=available_doctors[0].get("id")
                )
                if validate_action(fallback_action, observation):
                    return fallback_action
        return Action(action_type=ActionType.CLOSE_SCHEDULE)


def run_episode(task: str = "easy") -> float:
    """Run episode with LLM-based scheduling"""
    print(f"[START] {task}")
    
    client = initialize_client()
    env = HealthcareSchedulingEnv()
    observation = env.reset(ResetRequest(task=task))
    
    step_count = 0
    while step_count < 100:
        step_count += 1
        action = get_llm_action(client, observation.task_name, observation.model_dump(), env)
        result = env.step(action)
        observation = result.observation
        print(f"[STEP] {step_count}: {action.action_type.value} - Score {result.reward.episode_score:.3f}")
        if result.done:
            break
    
    if task == "easy":
        score = grade_easy(env)
    elif task == "medium":
        score = grade_medium(env)
    else:
        score = grade_hard(env)
    
    print(f"[END] {task} - Final score {score:.3f}")
    return score


def main():
    """Main entry - run all tasks with LLM proxy"""
    print("="*70)
    print("HEALTHCARE SCHEDULING - LLM INFERENCE WITH API PROXY")
    print("="*70)
    
    # Case 2: Validate HF_TOKEN is set
    if not HF_TOKEN or HF_TOKEN.strip() == "":
        print("FATAL ERROR: HF_TOKEN environment variable not set or empty")
        print("Set it with: $env:HF_TOKEN='your-huggingface-token'")
        return 1
    
    if not API_BASE_URL or API_BASE_URL.strip() == "":
        print("FATAL ERROR: API_BASE_URL environment variable not set or empty")
        print("Set it with: $env:API_BASE_URL='https://router.huggingface.co/v1'")
        return 1
    
    print(f"Using LLM proxy at: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print()
    
    try:
        easy_score = run_episode("easy")
        medium_score = run_episode("medium")
        hard_score = run_episode("hard")
        
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print(f"Easy:   {easy_score:.3f}")
        print(f"Medium: {medium_score:.3f}")
        print(f"Hard:   {hard_score:.3f}")
        print(f"Average: {(easy_score + medium_score + hard_score) / 3:.3f}")
        print("="*70)
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main() if main() is not None else 0)
