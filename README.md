---
title: Healthcare Scheduling OpenEnv
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
---

# Healthcare Appointment Scheduling RL Environment

A healthcare appointment scheduling environment that simulates scheduler workflows:
- triage patient urgency,
- assign patients to doctors,
- manage specialty constraints,
- and handle emergency walk-ins.

It includes deterministic tasks, easy-medium-hard progression, and shaped rewards for partial progress.

## Problem Domain

The environment models appointment scheduling operations used by real healthcare systems.

## Tasks

1. `healthcare_easy_basic_scheduling`
- Assign one doctor to 3 patients in 8 available slots.

2. `healthcare_medium_specialty_routing`
- Assign 2 doctors with specialties to 6 patients respecting constraints.

3. `healthcare_hard_emergency_handling`
- Assign 3 doctors to 8 patients plus 2 emergency walk-ins with dynamic arrival.

## Action Space

- `assign_patient`
- `reschedule_patient`
- `escalate_urgent_case`
- `mark_no_show`
- `close_schedule`

Every action uses:
- `action_type` (enum)
- `patient_id` (int)
- `slot_id` (int, optional)
- `doctor_id` (int, optional)

## Observation Space

Each step returns:
- `task_name`
- `objective`
- `step`
- `max_steps`
- `patients` (with urgency levels)
- `doctors` (with specialties)
- `appointments`
- `time_slots`
- `recent_events`
- `episode_score`

## Reward Design

- Positive reward for correct patient assignment based on urgency.
- Extra reward for priority slot placement and emergency handling.
- Penalty for invalid, conflicting, or unsafe actions.
- Small per-step cost to discourage inefficiency.

Final normalized score is deterministic in `[0, 1]` and computed from scheduled patients and task objectives.

## Project Structure

- `models.py`: Typed action, observation, reward, and result models.
- `tasks.py`: Easy/medium/hard task definitions and objective plans.
- `env/healthcare_env.py`: Core simulator logic for `reset()`, `step()`, and `state()`.
- `app.py`: FastAPI service exposing `/reset`, `/step`, `/state`.
- `openenv.yaml`: OpenEnv metadata.
- `inference.py`: Baseline runner using the OpenAI client.

## Setup

```bash
pip install -r requirements.txt
```

Create environment variables:

On Windows PowerShell:

```powershell
$env:HF_TOKEN="your-huggingface-token"
$env:MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
$env:API_BASE_URL="https://router.huggingface.co/v1"
```

## Run Locally

Start environment server:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
python test_demo.py
```

Run baseline inference:

```bash
python inference.py
```

## Docker

```bash
docker build -t healthcare-scheduling .
docker run -p 8000:8000 healthcare-scheduling
```

## Validation

Run test suite:

```bash
python test_demo.py
python quick_validate.py http://localhost:8000
```

## License

Open source - available for research and educational use
