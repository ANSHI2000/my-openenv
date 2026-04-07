"""
Simple test suite for healthcare appointment scheduling environment
"""

from models import ResetRequest, StepRequest, Action, ActionType, UrgencyLevel, PatientStatus
from env.healthcare_env import HealthcareSchedulingEnv
from graders import grade_easy, grade_medium, grade_hard


def test_easy_task():
    """Test Easy Task"""
    print("\n" + "="*70)
    print("EASY TASK: Schedule as many patients as possible")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="easy"))
    
    print(f"Initial state: {obs.patients_scheduled}/{len(obs.patients)} scheduled")
    
    # Schedule all patients
    actions = []
    for p in obs.patients:
        for s in obs.time_slots:
            if s.available:
                doctor = next((d for d in obs.doctors if p.specialty_required in d.specialties), obs.doctors[0])
                action = Action(
                    action_type=ActionType.ASSIGN_PATIENT,
                    patient_id=p.id,
                    slot_id=s.slot_id,
                    doctor_id=doctor.id
                )
                actions.append(action)
                break
    
    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        total_reward += result.reward.step_reward
        print(f"Step {i}: Patient {action.patient_id} >> Slot {action.slot_id}")
        print(f"  Reward: {result.reward.step_reward:.2f}")
    
    result = env.step(Action(action_type=ActionType.CLOSE_SCHEDULE))
    score = grade_easy(env)
    print(f"\n[PASS] EASY SCORE: {score:.2f}")
    return score


def test_medium_task():
    """Test Medium Task"""
    print("\n" + "="*70)
    print("MEDIUM TASK: Schedule with specialty constraints")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="medium"))
    
    print(f"Initial state: {obs.patients_scheduled}/{len(obs.patients)} scheduled")
    
    # Assign patients respecting specialties
    actions = []
    for p in obs.patients:
        doctor = next((d for d in obs.doctors if p.specialty_required in d.specialties), None)
        if doctor:
            slots = [s for s in obs.time_slots if s.available and s.doctor_id == doctor.id]
            if slots:
                action = Action(
                    action_type=ActionType.ASSIGN_PATIENT,
                    patient_id=p.id,
                    slot_id=slots[0].slot_id,
                    doctor_id=doctor.id
                )
                actions.append(action)
    
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        print(f"Step {i}: Patient {action.patient_id} >> Slot {action.slot_id}")
    
    result = env.step(Action(action_type=ActionType.CLOSE_SCHEDULE))
    score = grade_medium(env)
    print(f"\n[PASS] MEDIUM SCORE: {score:.2f}")
    return score


def test_hard_task():
    """Test Hard Task"""
    print("\n" + "="*70)
    print("HARD TASK: Emergency scheduling optimization")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="hard"))
    
    print(f"Initial state: {obs.patients_scheduled}/{len(obs.patients)} scheduled")
    print(f"Emergency walk-ins: {obs.emergency_cases_handled}")
    
    # Prioritize emergency patients
    emergency = [p for p in obs.patients if p.is_emergency_walkin]
    normal = [p for p in obs.patients if not p.is_emergency_walkin]
    sorted_patients = emergency + normal
    
    actions = []
    for p in sorted_patients:
        doctor = next((d for d in obs.doctors if p.specialty_required in d.specialties), obs.doctors[0])
        slots = [s for s in obs.time_slots if s.available and s.doctor_id == doctor.id]
        if slots:
            action = Action(
                action_type=ActionType.ASSIGN_PATIENT,
                patient_id=p.id,
                slot_id=slots[0].slot_id,
                doctor_id=doctor.id
            )
            actions.append(action)
    
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        if i % 2 == 0:
            print(f"Step {i}: Patient {action.patient_id} >> Scheduled: {result.observation.patients_scheduled}/{len(result.observation.patients)}")
    
    result = env.step(Action(action_type=ActionType.CLOSE_SCHEDULE))
    score = grade_hard(env)
    print(f"\n[PASS] HARD SCORE: {score:.2f}")
    return score


if __name__ == "__main__":
    print("\n" + "="*70)
    print("HEALTHCARE APPOINTMENT SCHEDULING - TEST SUITE")
    print("="*70)
    
    easy = test_easy_task()
    medium = test_medium_task()
    hard = test_hard_task()
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Easy:   {easy:.2f}")
    print(f"Medium: {medium:.2f}")
    print(f"Hard:   {hard:.2f}")
    print(f"Average: {(easy + medium + hard) / 3:.2f}")
    print("\n[SUCCESS] All tests completed!")
    print("="*70)
