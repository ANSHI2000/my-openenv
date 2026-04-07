"""
Healthcare Appointment Scheduling - Task Implementations
Three deterministic tasks with unique mechanics and scoring
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from models import (
    Patient, Doctor, TimeSlot, Appointment, Event,
    UrgencyLevel, PatientStatus, ActionType
)


@dataclass
class TaskConfig:
    """Configuration for a scheduling task"""
    name: str
    num_patients: int
    num_doctors: int
    num_slots: int
    has_emergency_walkins: bool
    allow_reschedule: bool
    specialist_constraints: bool
    max_steps: int
    objective: str


class EasyTaskBuilder:
    """Build Easy Task: Single doctor, basic urgency, fixed queue"""
    
    @staticmethod
    def create_initial_state(seed: int = None) -> Tuple[List[Patient], List[Doctor], List[TimeSlot]]:
        """Create initial state for easy task"""
        
        # Single doctor
        doctors = [
            Doctor(
                id=1,
                name="Dr. Primary Care",
                specialties=["general"],
                max_patients_per_session=10
            )
        ]
        
        # 3 patients with varying urgency
        patients = [
            Patient(
                id=1,
                name="Patient A",
                urgency=UrgencyLevel.HIGH,
                specialty_required="general"
            ),
            Patient(
                id=2,
                name="Patient B",
                urgency=UrgencyLevel.ROUTINE,
                specialty_required="general"
            ),
            Patient(
                id=3,
                name="Patient C",
                urgency=UrgencyLevel.MODERATE,
                specialty_required="general"
            ),
        ]
        
        # 8 time slots (15 min each, 2-hour session)
        time_slots = [
            TimeSlot(
                slot_id=i,
                doctor_id=1,
                start_time_minutes=i * 15,
                duration_minutes=15,
                is_priority_slot=(i < 2)  # First 2 slots are priority
            )
            for i in range(8)
        ]
        
        return patients, doctors, time_slots


class MediumTaskBuilder:
    """Build Medium Task: Multiple doctors, overlapping slots, specialty constraints"""
    
    @staticmethod
    def create_initial_state(seed: int = None) -> Tuple[List[Patient], List[Doctor], List[TimeSlot]]:
        """Create initial state for medium task"""
        
        # 2 doctors with different specialties
        doctors = [
            Doctor(
                id=1,
                name="Dr. Cardiology",
                specialties=["cardiology", "general"],
                max_patients_per_session=8
            ),
            Doctor(
                id=2,
                name="Dr. Orthopedics",
                specialties=["orthopedics", "general"],
                max_patients_per_session=8
            ),
        ]
        
        # 6 patients with mixed urgency and specialty requirements
        patients = [
            Patient(id=1, name="Patient 1", urgency=UrgencyLevel.HIGH, specialty_required="cardiology"),
            Patient(id=2, name="Patient 2", urgency=UrgencyLevel.ROUTINE, specialty_required="general"),
            Patient(id=3, name="Patient 3", urgency=UrgencyLevel.HIGH, specialty_required="orthopedics"),
            Patient(id=4, name="Patient 4", urgency=UrgencyLevel.MODERATE, specialty_required="general"),
            Patient(id=5, name="Patient 5", urgency=UrgencyLevel.ROUTINE, specialty_required="cardiology"),
            Patient(id=6, name="Patient 6", urgency=UrgencyLevel.MODERATE, specialty_required="orthopedics"),
        ]
        
        # 16 slots (8 per doctor, 15 min each)
        time_slots = []
        slot_id = 0
        for doctor_id in [1, 2]:
            for slot_num in range(8):
                time_slots.append(
                    TimeSlot(
                        slot_id=slot_id,
                        doctor_id=doctor_id,
                        start_time_minutes=slot_num * 15,
                        duration_minutes=15,
                        is_priority_slot=(slot_num < 2)
                    )
                )
                slot_id += 1
        
        return patients, doctors, time_slots


class HardTaskBuilder:
    """Build Hard Task: Emergencies, specialists, rescheduling, complex metrics"""
    
    @staticmethod
    def create_initial_state(seed: int = None) -> Tuple[List[Patient], List[Doctor], List[TimeSlot]]:
        """Create initial state for hard task"""
        
        # 3 doctors with specialties
        doctors = [
            Doctor(
                id=1,
                name="Dr. Emergency",
                specialties=["emergency", "general"],
                max_patients_per_session=10
            ),
            Doctor(
                id=2,
                name="Dr. Specialist A",
                specialties=["cardiology", "general"],
                max_patients_per_session=7
            ),
            Doctor(
                id=3,
                name="Dr. Specialist B",
                specialties=["orthopedics", "general"],
                max_patients_per_session=7
            ),
        ]
        
        # 8 patients: mix of scheduled + 2 emergency walk-ins
        patients = [
            Patient(id=1, name="Patient 1", urgency=UrgencyLevel.HIGH, specialty_required="cardiology"),
            Patient(id=2, name="Patient 2", urgency=UrgencyLevel.ROUTINE, specialty_required="general"),
            Patient(id=3, name="Patient 3", urgency=UrgencyLevel.HIGH, specialty_required="orthopedics"),
            Patient(id=4, name="Patient 4", urgency=UrgencyLevel.MODERATE, specialty_required="general"),
            Patient(id=5, name="Patient 5", urgency=UrgencyLevel.ROUTINE, specialty_required="cardiology"),
            Patient(id=6, name="Patient 6", urgency=UrgencyLevel.MODERATE, specialty_required="orthopedics"),
            # Emergency walk-ins
            Patient(id=7, name="Emergency Patient 1", urgency=UrgencyLevel.EMERGENCY, 
                   specialty_required="emergency", is_emergency_walkin=True),
            Patient(id=8, name="Emergency Patient 2", urgency=UrgencyLevel.EMERGENCY,
                   specialty_required="general", is_emergency_walkin=True),
        ]
        
        # 24 slots (8 per doctor)
        time_slots = []
        slot_id = 0
        for doctor_id in [1, 2, 3]:
            for slot_num in range(8):
                time_slots.append(
                    TimeSlot(
                        slot_id=slot_id,
                        doctor_id=doctor_id,
                        start_time_minutes=slot_num * 15,
                        duration_minutes=15,
                        is_priority_slot=(slot_num < 2)  # First 2 slots per doctor are priority
                    )
                )
                slot_id += 1
        
        return patients, doctors, time_slots


class TaskFactory:
    """Factory pattern for task creation"""
    
    # Task configs
    CONFIGS = {
        "easy": TaskConfig(
            name="easy",
            num_patients=3,
            num_doctors=1,
            num_slots=8,
            has_emergency_walkins=False,
            allow_reschedule=False,
            specialist_constraints=False,
            max_steps=10,
            objective="Schedule as many patients as possible. Prioritize urgent cases."
        ),
        "medium": TaskConfig(
            name="medium",
            num_patients=6,
            num_doctors=2,
            num_slots=16,
            has_emergency_walkins=False,
            allow_reschedule=True,
            specialist_constraints=True,
            max_steps=20,
            objective="Schedule all patients while respecting specialties and priorities."
        ),
        "hard": TaskConfig(
            name="hard",
            num_patients=8,
            num_doctors=3,
            num_slots=24,
            has_emergency_walkins=True,
            allow_reschedule=True,
            specialist_constraints=True,
            max_steps=30,
            objective="Optimal scheduling with emergency handling and specialist constraints."
        ),
    }
    
    @classmethod
    def _get_builders(cls):
        """Get builders dict - lazy loaded to avoid NameError"""
        return {
            "easy": EasyTaskBuilder,
            "medium": MediumTaskBuilder,
            "hard": HardTaskBuilder,
        }
    
    @classmethod
    def get_config(cls, task_name: str) -> TaskConfig:
        """Get task configuration"""
        if task_name not in cls.CONFIGS:
            raise ValueError(f"Unknown task: {task_name}")
        return cls.CONFIGS[task_name]
    
    @classmethod
    def create_task_state(cls, task_name: str, seed: int = None):
        """Create initial task state"""
        builders = cls._get_builders()
        if task_name not in builders:
            raise ValueError(f"Unknown task: {task_name}")
        builder = builders[task_name]
        return builder.create_initial_state(seed)

