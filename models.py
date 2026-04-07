"""
Healthcare Appointment Scheduling - Data Models
Pydantic v2 models for strict type validation
"""

from pydantic import BaseModel, Field, model_validator
from typing import Optional, List, Dict, Literal
from enum import Enum


class UrgencyLevel(str, Enum):
    """Patient urgency classification"""
    ROUTINE = "routine"
    MODERATE = "moderate"
    HIGH = "high"
    EMERGENCY = "emergency"


class PatientStatus(str, Enum):
    """Current status of a patient"""
    WAITING = "waiting"
    SCHEDULED = "scheduled"
    NO_SHOW = "no_show"
    COMPLETED = "completed"
    ESCALATED = "escalated"


class ActionType(str, Enum):
    """Available action types"""
    ASSIGN_PATIENT = "assign_patient"
    RESCHEDULE_PATIENT = "reschedule_patient"
    ESCALATE_URGENT_CASE = "escalate_urgent_case"
    MARK_NO_SHOW = "mark_no_show"
    CLOSE_SCHEDULE = "close_schedule"


class Patient(BaseModel):
    """Patient entity"""
    id: int = Field(..., description="Unique patient ID")
    name: str = Field(..., description="Patient name")
    urgency: UrgencyLevel = Field(default=UrgencyLevel.ROUTINE)
    specialty_required: Optional[str] = Field(default=None, description="Required specialist type")
    status: PatientStatus = Field(default=PatientStatus.WAITING)
    appointment_time: Optional[int] = Field(default=None, description="Assigned slot number")
    wait_time_minutes: int = Field(default=0, description="Minutes spent waiting")
    is_emergency_walkin: bool = Field(default=False)
    
    @model_validator(mode='after')
    def validate_emergency(self):
        if self.is_emergency_walkin:
            self.urgency = UrgencyLevel.EMERGENCY
        return self


class Doctor(BaseModel):
    """Doctor/provider entity"""
    id: int = Field(..., description="Unique doctor ID")
    name: str = Field(..., description="Doctor name")
    specialties: List[str] = Field(default_factory=list, description="Areas of expertise")
    max_patients_per_session: int = Field(default=10)
    current_load: int = Field(default=0, description="Number of patients assigned")
    on_break: bool = Field(default=False)


class TimeSlot(BaseModel):
    """Appointment time slot"""
    slot_id: int = Field(..., description="Unique slot identifier")
    doctor_id: int = Field(..., description="Assigned doctor")
    start_time_minutes: int = Field(..., description="Minutes from session start")
    duration_minutes: int = Field(default=15)
    available: bool = Field(default=True)
    assigned_patient_id: Optional[int] = Field(default=None)
    is_priority_slot: bool = Field(default=False, description="Reserved for urgent cases")


class Action(BaseModel):
    """Agent action"""
    action_type: ActionType = Field(..., description="Type of action")
    patient_id: Optional[int] = Field(default=None)
    slot_id: Optional[int] = Field(default=None)
    doctor_id: Optional[int] = Field(default=None)
    reason: Optional[str] = Field(default=None)
    
    @model_validator(mode='after')
    def validate_action_params(self):
        """Validate required params for each action type"""
        if self.action_type == ActionType.ASSIGN_PATIENT:
            if self.patient_id is None or self.slot_id is None or self.doctor_id is None:
                raise ValueError("ASSIGN_PATIENT requires patient_id, slot_id, doctor_id")
        elif self.action_type == ActionType.RESCHEDULE_PATIENT:
            if self.patient_id is None or self.slot_id is None:
                raise ValueError("RESCHEDULE_PATIENT requires patient_id, slot_id")
        elif self.action_type == ActionType.ESCALATE_URGENT_CASE:
            if self.patient_id is None:
                raise ValueError("ESCALATE_URGENT_CASE requires patient_id")
        elif self.action_type == ActionType.MARK_NO_SHOW:
            if self.patient_id is None:
                raise ValueError("MARK_NO_SHOW requires patient_id")
        return self


class Appointment(BaseModel):
    """Scheduled appointment record"""
    appointment_id: int
    patient_id: int
    doctor_id: int
    slot_id: int
    scheduled_time_minutes: int
    status: PatientStatus = PatientStatus.SCHEDULED
    cancellation_reason: Optional[str] = None


class Event(BaseModel):
    """Recent event in the scheduling system"""
    step: int = Field(..., description="Step number when event occurred")
    event_type: str = Field(..., description="Type of event (assigned, conflict, escalated, etc)")
    details: str = Field(..., description="Human-readable event description")
    patient_id: Optional[int] = None
    impact_score: float = Field(..., description="Numerical impact on reward")


class Observation(BaseModel):
    """Complete environment observation"""
    task_name: str = Field(..., description="Name of current task (easy/medium/hard)")
    objective: str = Field(..., description="Task objective description")
    step_count: int = Field(..., description="Current step number")
    max_steps: int = Field(..., description="Maximum steps for this episode")
    
    # Entities
    patients: List[Patient]
    doctors: List[Doctor]
    time_slots: List[TimeSlot]
    appointments: List[Appointment]
    
    # State tracking
    recent_events: List[Event] = Field(default_factory=list, description="Last 5 events")
    available_actions: List[ActionType] = Field(default_factory=list)
    
    # Metrics
    patients_scheduled: int = Field(default=0)
    total_wait_time_minutes: int = Field(default=0)
    urgent_patients_scheduled: int = Field(default=0)
    scheduling_conflicts: int = Field(default=0)
    emergency_cases_handled: int = Field(default=0)


class Reward(BaseModel):
    """Reward structure with breakdowns"""
    step_reward: float = Field(..., description="Reward for this step")
    reward_components: Dict[str, float] = Field(default_factory=dict)
    episode_score: float = Field(default=0.0, description="Cumulative episode score [0,1]")
    done: bool = Field(default=False)
    info: str = Field(default="")
    
    @model_validator(mode='after')
    def validate_score_range(self):
        if not (0.0 <= self.episode_score <= 1.0):
            self.episode_score = max(0.0, min(1.0, self.episode_score))
        return self


class StepResult(BaseModel):
    """Complete result from a step"""
    observation: Observation
    reward: Reward
    done: bool
    step_info: Dict = Field(default_factory=dict)


class ResetRequest(BaseModel):
    """Reset endpoint request"""
    task: Literal["easy", "medium", "hard"] = Field(default="easy")
    seed: Optional[int] = Field(default=None)


class StepRequest(BaseModel):
    """Step endpoint request"""
    action: Action


class HealthcareEnvState(BaseModel):
    """Complete internal environment state (for serialization)"""
    current_task: str
    current_step: int
    episode_score: float
    patients: List[Patient]
    doctors: List[Doctor]
    time_slots: List[TimeSlot]
    appointments: List[Appointment]
    recent_events: List[Event]
