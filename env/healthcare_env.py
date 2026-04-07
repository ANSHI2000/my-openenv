"""
Healthcare Appointment Scheduling - Environment
Core scheduling logic with reward calculation and validation
"""

import random
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from models import (
    Patient, Doctor, TimeSlot, Appointment, Event, Observation, Reward, StepResult,
    Action, ActionType, PatientStatus, UrgencyLevel, ResetRequest
)
from tasks import TaskFactory, TaskConfig


class HealthcareSchedulingEnv:
    """Main environment for healthcare appointment scheduling"""
    
    def __init__(self):
        self.task_config: Optional[TaskConfig] = None
        self.current_step = 0
        self.max_steps = 10
        self.episode_score = 0.0
        
        # Entities
        self.patients: List[Patient] = []
        self.doctors: List[Doctor] = []
        self.time_slots: List[TimeSlot] = []
        self.appointments: List[Appointment] = []
        self.recent_events: List[Event] = []
        
        # Tracking
        self.urgent_patients_scheduled = 0
        self.scheduling_conflicts = 0
        self.emergency_cases_handled = 0
        self.total_wait_time = 0
        self.emergency_unsafe_delays = 0
        
    def reset(self, request: ResetRequest = None) -> Observation:
        """Reset environment to initial state"""
        if request is None:
            request = ResetRequest(task="easy")
        
        task_name = request.task
        self.task_config = TaskFactory.get_config(task_name)
        self.max_steps = self.task_config.max_steps
        self.current_step = 0
        self.episode_score = 0.0
        
        # Initialize state
        self.patients, self.doctors, self.time_slots = TaskFactory.create_task_state(
            task_name, seed=request.seed
        )
        self.appointments = []
        self.recent_events = []
        
        # Reset tracking metrics
        self.urgent_patients_scheduled = 0
        self.scheduling_conflicts = 0
        self.emergency_cases_handled = 0
        self.total_wait_time = 0
        self.emergency_unsafe_delays = 0
        
        return self._get_observation()
    
    def step(self, action: Action) -> StepResult:
        """Execute one step in the environment"""
        self.current_step += 1
        
        # Validate action
        validation_error = self._validate_action(action)
        if validation_error:
            reward = Reward(
                step_reward=0.0,
                reward_components={"error": -0.1},
                episode_score=self.episode_score,
                done=False,
                info=validation_error
            )
            return StepResult(
                observation=self._get_observation(),
                reward=reward,
                done=False,
                step_info={"action_valid": False}
            )
        
        # Execute action
        step_reward = 0.0
        action_result = None
        
        try:
            if action.action_type == ActionType.ASSIGN_PATIENT:
                step_reward, action_result = self._handle_assign_patient(action)
            elif action.action_type == ActionType.RESCHEDULE_PATIENT:
                step_reward, action_result = self._handle_reschedule_patient(action)
            elif action.action_type == ActionType.ESCALATE_URGENT_CASE:
                step_reward, action_result = self._handle_escalate(action)
            elif action.action_type == ActionType.MARK_NO_SHOW:
                step_reward, action_result = self._handle_no_show(action)
            elif action.action_type == ActionType.CLOSE_SCHEDULE:
                step_reward = self._finalize_schedule()
                action_result = "Schedule closed"
        except Exception as e:
            step_reward = -0.05
            action_result = f"Error: {str(e)}"
        
        # Apply per-step penalty
        step_reward -= 0.01
        
        # Update episode score
        self.episode_score += step_reward
        self.episode_score = max(0.0, min(1.0, self.episode_score))
        
        # Check if done
        done = self.current_step >= self.max_steps or action.action_type == ActionType.CLOSE_SCHEDULE
        
        reward = Reward(
            step_reward=step_reward,
            reward_components={
                "action_reward": step_reward + 0.01,  # Remove per-step penalty for display
                "per_step_penalty": -0.01
            },
            episode_score=self.episode_score,
            done=done,
            info=action_result or ""
        )
        
        observation = self._get_observation()
        
        return StepResult(
            observation=observation,
            reward=reward,
            done=done,
            step_info={
                "action_type": action.action_type,
                "action_valid": True
            }
        )
    
    def _validate_action(self, action: Action) -> Optional[str]:
        """Validate action feasibility"""
        if action.action_type == ActionType.ASSIGN_PATIENT:
            patient = self._get_patient(action.patient_id)
            if not patient:
                return f"Patient {action.patient_id} not found"
            if patient.status != PatientStatus.WAITING:
                return f"Patient already {patient.status.value}"
            
            slot = self._get_slot(action.slot_id)
            if not slot:
                return f"Slot {action.slot_id} not found"
            if not slot.available:
                return f"Slot {action.slot_id} already occupied"
            
            doctor = self._get_doctor(action.doctor_id)
            if not doctor:
                return f"Doctor {action.doctor_id} not found"
            if doctor.current_load >= doctor.max_patients_per_session:
                return f"Doctor {action.doctor_id} at capacity"
            
            # Check specialty if required
            if patient.specialty_required and patient.specialty_required not in doctor.specialties:
                return f"Doctor {action.doctor_id} doesn't have {patient.specialty_required} specialty"
        
        elif action.action_type == ActionType.RESCHEDULE_PATIENT:
            if not self.task_config.allow_reschedule:
                return "Rescheduling not allowed in this task"
            
            patient = self._get_patient(action.patient_id)
            if not patient or patient.status != PatientStatus.SCHEDULED:
                return f"Patient {action.patient_id} not scheduled"
            
            slot = self._get_slot(action.slot_id)
            if not slot or not slot.available:
                return f"Target slot {action.slot_id} not available"
        
        elif action.action_type == ActionType.ESCALATE_URGENT_CASE:
            patient = self._get_patient(action.patient_id)
            if not patient:
                return f"Patient {action.patient_id} not found"
            if patient.urgency == UrgencyLevel.EMERGENCY:
                return "Patient already at emergency level"
        
        elif action.action_type == ActionType.MARK_NO_SHOW:
            patient = self._get_patient(action.patient_id)
            if not patient or patient.status != PatientStatus.SCHEDULED:
                return f"Patient {action.patient_id} not scheduled"
        
        return None
    
    def _handle_assign_patient(self, action: Action) -> Tuple[float, str]:
        """Assign patient to a slot"""
        patient = self._get_patient(action.patient_id)
        slot = self._get_slot(action.slot_id)
        doctor = self._get_doctor(action.doctor_id)
        
        # Mark slot as taken
        slot.available = False
        slot.assigned_patient_id = action.patient_id
        
        # Mark patient as scheduled
        patient.status = PatientStatus.SCHEDULED
        patient.appointment_time = slot.start_time_minutes
        
        # Create appointment
        appointment = Appointment(
            appointment_id=len(self.appointments) + 1,
            patient_id=action.patient_id,
            doctor_id=action.doctor_id,
            slot_id=action.slot_id,
            scheduled_time_minutes=slot.start_time_minutes
        )
        self.appointments.append(appointment)
        
        # Update doctor load
        doctor.current_load += 1
        
        # Calculate reward
        reward = 0.0
        event_desc = f"Patient {patient.name} assigned to slot {slot.slot_id}"
        
        # Bonus for urgent cases in priority slots
        if patient.urgency in [UrgencyLevel.HIGH, UrgencyLevel.EMERGENCY]:
            if slot.is_priority_slot:
                reward += 0.2
                event_desc += " [PRIORITY]"
                self.urgent_patients_scheduled += 1
            else:
                reward += 0.1
                event_desc += " [URGENT]"
                self.urgent_patients_scheduled += 1
        else:
            reward += 0.05
        
        # Track emergency cases
        if patient.is_emergency_walkin:
            self.emergency_cases_handled += 1
            reward += 0.15
            event_desc += " [EMERGENCY WALKIN]"
        
        # Add event
        self.recent_events.append(Event(
            step=self.current_step,
            event_type="patient_assigned",
            details=event_desc,
            patient_id=action.patient_id,
            impact_score=reward
        ))
        
        return reward, event_desc
    
    def _handle_reschedule_patient(self, action: Action) -> Tuple[float, str]:
        """Reschedule patient to different slot"""
        patient = self._get_patient(action.patient_id)
        old_appointment = next((a for a in self.appointments if a.patient_id == action.patient_id), None)
        
        if not old_appointment:
            return -0.05, "No appointment found to reschedule"
        
        # Free old slot
        old_slot = self._get_slot(old_appointment.slot_id)
        old_slot.available = True
        old_slot.assigned_patient_id = None
        
        # Assign to new slot
        new_slot = self._get_slot(action.slot_id)
        new_slot.available = False
        new_slot.assigned_patient_id = action.patient_id
        
        # Update appointment
        old_appointment.slot_id = action.slot_id
        old_appointment.scheduled_time_minutes = new_slot.start_time_minutes
        
        patient.appointment_time = new_slot.start_time_minutes
        
        reward = 0.05  # Small reward for flexibility
        event_desc = f"Patient {patient.name} rescheduled from slot {old_appointment.slot_id} to {action.slot_id}"
        
        self.recent_events.append(Event(
            step=self.current_step,
            event_type="rescheduled",
            details=event_desc,
            patient_id=action.patient_id,
            impact_score=reward
        ))
        
        return reward, event_desc
    
    def _handle_escalate(self, action: Action) -> Tuple[float, str]:
        """Escalate patient urgency"""
        patient = self._get_patient(action.patient_id)
        old_urgency = patient.urgency
        patient.urgency = UrgencyLevel.EMERGENCY
        patient.status = PatientStatus.ESCALATED
        
        reward = 0.08
        event_desc = f"Patient {patient.name} escalated from {old_urgency.value} to emergency"
        
        self.recent_events.append(Event(
            step=self.current_step,
            event_type="escalated",
            details=event_desc,
            patient_id=action.patient_id,
            impact_score=reward
        ))
        
        return reward, event_desc
    
    def _handle_no_show(self, action: Action) -> Tuple[float, str]:
        """Mark patient as no-show"""
        patient = self._get_patient(action.patient_id)
        patient.status = PatientStatus.NO_SHOW
        
        # Free the slot
        appointment = next((a for a in self.appointments if a.patient_id == action.patient_id), None)
        if appointment:
            slot = self._get_slot(appointment.slot_id)
            slot.available = True
            slot.assigned_patient_id = None
            appointment.cancellation_reason = "No-show"
        
        reward = -0.05
        event_desc = f"Patient {patient.name} marked as no-show"
        
        self.recent_events.append(Event(
            step=self.current_step,
            event_type="no_show",
            details=event_desc,
            patient_id=action.patient_id,
            impact_score=reward
        ))
        
        return reward, event_desc
    
    def _finalize_schedule(self) -> float:
        """Finalize and score the schedule"""
        reward = 0.0
        
        # Count unscheduled patients
        unscheduled = sum(1 for p in self.patients if p.status == PatientStatus.WAITING)
        
        # Penalize unscheduled emergency cases heavily
        unscheduled_emergencies = sum(
            1 for p in self.patients 
            if p.status == PatientStatus.WAITING and p.urgency == UrgencyLevel.EMERGENCY
        )
        reward -= unscheduled_emergencies * 0.3
        
        # Penalize unscheduled patients
        reward -= unscheduled * 0.1
        
        # Bonus for scheduling all patients
        if unscheduled == 0:
            reward += 0.3
        
        # Penalize scheduling conflicts
        reward -= self.scheduling_conflicts * 0.2
        
        # Bonus for handling emergency cases
        reward += min(self.emergency_cases_handled * 0.15, 0.3)
        
        # Penalty for unsafe emergency delays
        reward -= self.emergency_unsafe_delays * 0.25
        
        event_desc = f"Schedule finalized: {len(self.appointments)} appointments scheduled"
        self.recent_events.append(Event(
            step=self.current_step,
            event_type="finalized",
            details=event_desc,
            impact_score=reward
        ))
        
        return reward
    
    def _get_observation(self) -> Observation:
        """Build current observation"""
        # Keep only last 5 events
        recent = self.recent_events[-5:] if self.recent_events else []
        
        # Available actions
        available_actions = [ActionType.ASSIGN_PATIENT, ActionType.ESCALATE_URGENT_CASE, ActionType.CLOSE_SCHEDULE]
        if self.task_config.allow_reschedule:
            available_actions.append(ActionType.RESCHEDULE_PATIENT)
        available_actions.append(ActionType.MARK_NO_SHOW)
        
        return Observation(
            task_name=self.task_config.name,
            objective=self.task_config.objective,
            step_count=self.current_step,
            max_steps=self.max_steps,
            patients=self.patients,
            doctors=self.doctors,
            time_slots=self.time_slots,
            appointments=self.appointments,
            recent_events=recent,
            available_actions=available_actions,
            patients_scheduled=sum(1 for p in self.patients if p.status == PatientStatus.SCHEDULED),
            total_wait_time_minutes=self.total_wait_time,
            urgent_patients_scheduled=self.urgent_patients_scheduled,
            scheduling_conflicts=self.scheduling_conflicts,
            emergency_cases_handled=self.emergency_cases_handled,
        )
    
    def _get_patient(self, patient_id: int) -> Optional[Patient]:
        """Get patient by ID"""
        return next((p for p in self.patients if p.id == patient_id), None)
    
    def _get_doctor(self, doctor_id: int) -> Optional[Doctor]:
        """Get doctor by ID"""
        return next((d for d in self.doctors if d.id == doctor_id), None)
    
    def _get_slot(self, slot_id: int) -> Optional[TimeSlot]:
        """Get time slot by ID"""
        return next((s for s in self.time_slots if s.slot_id == slot_id), None)
    
    def get_state(self) -> Dict:
        """Get current state for serialization"""
        return {
            "task": self.task_config.name if self.task_config else None,
            "step": self.current_step,
            "max_steps": self.max_steps,
            "episode_score": self.episode_score,
            "patients": [p.model_dump() for p in self.patients],
            "doctors": [d.model_dump() for d in self.doctors],
            "time_slots": [s.model_dump() for s in self.time_slots],
            "appointments": [a.model_dump() for a in self.appointments],
            "metrics": {
                "patients_scheduled": sum(1 for p in self.patients if p.status == PatientStatus.SCHEDULED),
                "urgent_scheduled": self.urgent_patients_scheduled,
                "emergencies_handled": self.emergency_cases_handled,
                "conflicts": self.scheduling_conflicts,
            }
        }
