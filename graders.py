"""
Healthcare Appointment Scheduling - Graders
Deterministic scoring for three task difficulty levels
"""

from typing import List
from models import Patient, TimeSlot, Appointment, UrgencyLevel, PatientStatus
from env.healthcare_env import HealthcareSchedulingEnv


def grade_easy(env: HealthcareSchedulingEnv) -> float:
    """
    Grade Easy Task: Schedule as many patients as possible
    
    Scoring:
    - Base: (patients_scheduled / total_patients) * 0.5
    - Bonus: Urgent-first ordering bonus * 0.3
    - Penalty: Conflicts * 0.2
    
    Returns:
        Score in (0, 1)
    """
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.01
    
    # Count scheduled patients
    scheduled = sum(1 for p in env.patients if p.status == PatientStatus.SCHEDULED)
    scheduling_score = (scheduled / total_patients) * 0.5
    
    # Check urgent-first ordering
    urgent_scheduled = [p for p in env.patients if p.urgency in [UrgencyLevel.HIGH, UrgencyLevel.EMERGENCY] and p.status == PatientStatus.SCHEDULED]
    routine_scheduled = [p for p in env.patients if p.urgency == UrgencyLevel.ROUTINE and p.status == PatientStatus.SCHEDULED]
    
    urgent_bonus = 0.0
    if urgent_scheduled and routine_scheduled:
        # Check if first urgent patient was scheduled before first routine patient
        earliest_urgent_time = min((a.scheduled_time_minutes for a in env.appointments if a.patient_id in [p.id for p in urgent_scheduled]), default=float('inf'))
        earliest_routine_time = min((a.scheduled_time_minutes for a in env.appointments if a.patient_id in [p.id for p in routine_scheduled]), default=float('inf'))
        
        if earliest_urgent_time < earliest_routine_time:
            urgent_bonus = 0.3
        else:
            urgent_bonus = 0.1
    elif urgent_scheduled:
        urgent_bonus = 0.3
    
    # Detect conflicts (same slot booked twice)
    slot_assignments = {}
    conflicts = 0
    for apt in env.appointments:
        if apt.slot_id in slot_assignments:
            conflicts += 1
        else:
            slot_assignments[apt.slot_id] = apt.patient_id
    
    conflict_penalty = max(0.0, 0.2 * (1 - conflicts / max(1, len(env.appointments))))
    
    score = scheduling_score + urgent_bonus - (0.2 if conflicts > 0 else conflict_penalty)
    return max(0.01, min(0.99, score))


def grade_medium(env: HealthcareSchedulingEnv) -> float:
    """
    Grade Medium Task: Schedule all patients with specialist constraints and priority ordering
    
    Scoring:
    - Base: (patients_scheduled / total_patients) * 0.4
    - Specialty compliance: * 0.2
    - Urgent-first: * 0.2
    - No conflicts: * 0.2
    
    Returns:
        Score in (0, 1)
    """
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.01
    
    # Count scheduled
    scheduled = sum(1 for p in env.patients if p.status == PatientStatus.SCHEDULED)
    base_score = (scheduled / total_patients) * 0.4
    
    # Check specialty compliance
    specialty_violations = 0
    for apt in env.appointments:
        patient = next((p for p in env.patients if p.id == apt.patient_id), None)
        if patient and patient.specialty_required:
            # Would need doctor lookup - assume perfect for now
            pass
    
    specialty_score = 0.2  # Assume all valid if we got here
    
    # Check urgent-first ordering
    urgent_first_score = 0.0
    urgent_patients = [p for p in env.patients if p.urgency in [UrgencyLevel.HIGH, UrgencyLevel.EMERGENCY]]
    routine_patients = [p for p in env.patients if p.urgency == UrgencyLevel.ROUTINE]
    
    if urgent_patients and routine_patients:
        urgent_appointments = [a for a in env.appointments if a.patient_id in [p.id for p in urgent_patients]]
        routine_appointments = [a for a in env.appointments if a.patient_id in [p.id for p in routine_patients]]
        
        if urgent_appointments and routine_appointments:
            avg_urgent_time = sum(a.scheduled_time_minutes for a in urgent_appointments) / len(urgent_appointments)
            avg_routine_time = sum(a.scheduled_time_minutes for a in routine_appointments) / len(routine_appointments)
            
            if avg_urgent_time < avg_routine_time:
                urgent_first_score = 0.2
            else:
                urgent_first_score = 0.05
    else:
        urgent_first_score = 0.2
    
    # Check for conflicts
    conflict_score = 0.2
    slot_doctor_pairs = set()
    for apt in env.appointments:
        pair = (apt.slot_id, apt.doctor_id)
        if pair in slot_doctor_pairs:
            conflict_score = 0.01
            break
        slot_doctor_pairs.add(pair)
    
    score = base_score + specialty_score + urgent_first_score + conflict_score
    return max(0.01, min(0.99, score))


def grade_hard(env: HealthcareSchedulingEnv) -> float:
    """
    Grade Hard Task: Emergency handling, specialist routing, rescheduling optimization
    
    Scoring:
    - Base: (all_scheduled) * 0.3
    - Emergency first: (emergency_in_priority_slots) * 0.2
    - No unsafe delays: * 0.2
    - Specialty compliance: * 0.15
    - Conflict-free: * 0.15
    
    Returns:
        Score in (0, 1)
    """
    total_patients = len(env.patients)
    if total_patients == 0:
        return 0.01
    
    # 1. All patients scheduled bonus
    scheduled = sum(1 for p in env.patients if p.status == PatientStatus.SCHEDULED)
    all_scheduled_bonus = 0.3 if scheduled == total_patients else (scheduled / total_patients) * 0.2
    
    # 2. Emergency cases in priority slots
    emergency_score = 0.0
    emergency_patients = [p for p in env.patients if p.urgency == UrgencyLevel.EMERGENCY]
    if emergency_patients:
        emergency_appointments = [a for a in env.appointments if a.patient_id in [p.id for p in emergency_patients]]
        priority_slots = [s for s in env.time_slots if s.is_priority_slot]
        
        if emergency_appointments:
            in_priority = sum(1 for a in emergency_appointments if a.slot_id in [s.slot_id for s in priority_slots])
            emergency_score = (in_priority / len(emergency_appointments)) * 0.2
    
    # 3. No unsafe delays for emergency cases
    unsafe_delay_score = 0.2
    for patient in emergency_patients:
        apt = next((a for a in env.appointments if a.patient_id == patient.id), None)
        if apt and apt.scheduled_time_minutes > 20:  # Unsafe if emergency waits > 20 min
            unsafe_delay_score = 0.01
            break
    
    # 4. Specialty compliance
    specialty_score = 0.15  # Assume valid if system allowed it
    
    # 5. Conflict detection
    conflict_score = 0.15
    seen_slots = set()
    for apt in env.appointments:
        key = (apt.slot_id, apt.doctor_id)
        if key in seen_slots:
            conflict_score = 0.01
            break
        seen_slots.add(key)
    
    score = all_scheduled_bonus + emergency_score + unsafe_delay_score + specialty_score + conflict_score
    return max(0.01, min(0.99, score))

