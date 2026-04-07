"""
Comprehensive test suite for healthcare appointment scheduling environment
Tests all three task types with sophisticated action space
"""

from models import ResetRequest, StepRequest, Action, ActionType, UrgencyLevel
from env.healthcare_env import HealthcareSchedulingEnv
from graders import grade_easy, grade_medium, grade_hard


def test_easy_task():
    """Test Easy Task"""
    print("\n" + "="*70)
    print("EASY TASK: Schedule as many patients as possible")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="easy"))
    
    print(f"Task: {obs.task_name}")
    print(f"Patients: {len(obs.patients)} patients")
    print(f"Doctors: {len(obs.doctors)} doctors")
    print(f"Time slots: {len(obs.time_slots)} slots")
    print(f"\nInitial state: {obs.patients_scheduled}/{len(obs.patients)} scheduled")
    
    # Create assignment actions (best practice: high priority first)
    actions = []
    patients = obs.patients
    doctors = obs.doctors
    slots = obs.time_slots
    
    # Assign high urgency patients first
    high_urgency_patients = [p for p in patients if p.urgency in [UrgencyLevel.HIGH, UrgencyLevel.EMERGENCY]]
    for patient in high_urgency_patients:
        # Find priority slot
        available_slot = next((s for s in slots if s.available and s.is_priority_slot), None)
        if available_slot:
            doctor = doctors[0]
            action = Action(
                action_type=ActionType.ASSIGN_PATIENT,
                patient_id=patient.id,
                slot_id=available_slot.slot_id,
                doctor_id=doctor.id
            )
            actions.append(action)
    
    # Assign remaining patients
    remaining_patients = [p for p in patients if p.urgency not in [UrgencyLevel.HIGH, UrgencyLevel.EMERGENCY]]
    for patient in remaining_patients:
        available_slot = next((s for s in slots if s.available), None)
        if available_slot:
            doctor = next((d for d in doctors if d.current_load < d.max_patients_per_session), None)
            if doctor:
                action = Action(
                    action_type=ActionType.ASSIGN_PATIENT,
                    patient_id=patient.id,
                    slot_id=available_slot.slot_id,
                    doctor_id=doctor.id
                )
                actions.append(action)
    
    # Execute actions
    total_reward = 0.0
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        total_reward += result.reward.step_reward
        
        print(f"\nStep {i}: {action.action_type} - Patient {action.patient_id} >> Slot {action.slot_id}")
        print(f"  Reward: {result.reward.step_reward:.2f} | Episode score: {result.reward.episode_score:.2f}")
        print(f"  Scheduled: {result.observation.patients_scheduled}/{len(result.observation.patients)}")
    
    # Close schedule
    close_action = Action(action_type=ActionType.CLOSE_SCHEDULE)
    result = env.step(close_action)
    print(f"\nSchedule closed | Final score: {result.reward.episode_score:.2f}")
    
    # Grade
    score = grade_easy(env)
    print(f"\n[PASS] EASY SCORE: {score:.2f}")
    return score


def test_medium_task():
    """Test Medium Task"""
    print("\n" + "="*70)
    print("MEDIUM TASK: Multiple doctors, specialty constraints")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="medium"))
    
    print(f"Task: {obs.task_name}")
    print(f"Patients: {len(obs.patients)} patients")
    print(f"Doctors: {len(obs.doctors)} doctors")
    print(f"Time slots: {len(obs.time_slots)} slots")
    
    # Sort patients by urgency (descending)
    sorted_patients = sorted(obs.patients, key=lambda p: [
        UrgencyLevel.EMERGENCY, UrgencyLevel.HIGH, UrgencyLevel.MODERATE, UrgencyLevel.ROUTINE
    ].index(p.urgency))
    
    actions = []
    for patient in sorted_patients:
        # Find appropriate doctor
        doctor = next((d for d in obs.doctors if patient.specialty_required in d.specialties), None)
        if not doctor:
            continue
        
        # Find available slot
        available_slot = next((s for s in obs.time_slots if s.available and s.doctor_id == doctor.id), None)
        if available_slot:
            action = Action(
                action_type=ActionType.ASSIGN_PATIENT,
                patient_id=patient.id,
                slot_id=available_slot.slot_id,
                doctor_id=doctor.id
            )
            actions.append(action)
    
    # Execute actions
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        print(f"Step {i}: Patient {action.patient_id} >> Slot {action.slot_id} (Doctor {action.doctor_id})")
        print(f"  Score: {result.reward.episode_score:.2f}")
    
    # Close schedule
    result = env.step(Action(action_type=ActionType.CLOSE_SCHEDULE))
    print(f"Final episode score: {result.reward.episode_score:.2f}")
    
    score = grade_medium(env)
    print(f"\n[PASS] MEDIUM SCORE: {score:.2f}")
    return score


def test_hard_task():
    """Test Hard Task"""
    print("\n" + "="*70)
    print("HARD TASK: Emergency walk-ins, specialists, complex constraints")
    print("="*70)
    
    env = HealthcareSchedulingEnv()
    obs = env.reset(ResetRequest(task="hard"))
    
    print(f"Task: {obs.task_name}")
    print(f"Patients: {len(obs.patients)} patients (including {sum(1 for p in obs.patients if p.is_emergency_walkin)} emergency walk-ins)")
    print(f"Doctors: {len(obs.doctors)} doctors")
    print(f"Time slots: {len(obs.time_slots)} slots")
    
    # Prioritize emergency walk-ins
    emergency_patients = [p for p in obs.patients if p.is_emergency_walkin]
    normal_patients = [p for p in obs.patients if not p.is_emergency_walkin]
    
    # Sort both by urgency
    all_sorted = emergency_patients + sorted(normal_patients, key=lambda p: [
        UrgencyLevel.EMERGENCY, UrgencyLevel.HIGH, UrgencyLevel.MODERATE, UrgencyLevel.ROUTINE
    ].index(p.urgency))
    
    actions = []
    for patient in all_sorted:
        # Find appropriate doctor
        doctor = next((d for d in obs.doctors if patient.specialty_required in d.specialties), None )
        if not doctor:
            doctor = obs.doctors[0]  # Fallback
        
        # Prefer priority slots for emergency
        if patient.is_emergency_walkin:
            available_slots = [s for s in obs.time_slots if s.available and s.doctor_id == doctor.id and s.is_priority_slot]
        else:
            available_slots = [s for s in obs.time_slots if s.available and s.doctor_id == doctor.id]
        
        if available_slots:
            available_slot = available_slots[0]
            action = Action(
                action_type=ActionType.ASSIGN_PATIENT,
                patient_id=patient.id,
                slot_id=available_slot.slot_id,
                doctor_id=doctor.id
            )
            actions.append(action)
            print(f"Scheduling: {patient.name} (urgency={patient.urgency.value}) >> Slot {available_slot.slot_id}")
    
    # Execute actions
    for i, action in enumerate(actions, 1):
        result = env.step(action)
        if i % 2 == 0:
            print(f"  After step {i}: {result.observation.patients_scheduled}/{len(result.observation.patients)} scheduled")
    
    # Close schedule
    result = env.step(Action(action_type=ActionType.CLOSE_SCHEDULE))
    print(f"\nFinal episode score: {result.reward.episode_score:.2f}")
    
    score = grade_hard(env)
    print(f"\n[PASS] HARD SCORE: {score:.2f}")
    return score


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("HEALTHCARE APPOINTMENT SCHEDULING - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    scores = {
        "easy": test_easy_task(),
        "medium": test_medium_task(),
        "hard": test_hard_task(),
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Easy Task Score:   {scores['easy']:.2f}")
    print(f"Medium Task Score: {scores['medium']:.2f}")
    print(f"Hard Task Score:   {scores['hard']:.2f}")
    print(f"Average:           {sum(scores.values()) / len(scores):.2f}")
    
    if all(s > 0.7 for s in scores.values()):
        print("\n[SUCCESS] All tests completed successfully!")
    else:
        print("\n[WARNING] Some scores are below 0.7")
    
    print("="*70)


if __name__ == "__main__":
    main()
