"""
Healthcare Appointment Scheduling - FastAPI Server
REST API endpoints for environment interaction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError

from models import Action, ResetRequest, StepRequest
from env.healthcare_env import HealthcareSchedulingEnv

# Initialize app and environment
app = FastAPI(
    title="Healthcare Appointment Scheduling RL Environment",
    description="OpenEnv compliant scheduling environment with patient prioritization",
    version="2.0"
)

# CORS configuration for HuggingFace Spaces and local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global environment instance
env = HealthcareSchedulingEnv()


@app.post("/reset")
def reset(request: ResetRequest = None):
    """
    Reset environment to initial state
    
    Args:
        request: ResetRequest with optional task ("easy", "medium", "hard") and seed
        
    Returns:
        Initial observation
    """
    try:
        if request is None:
            request = ResetRequest(task="easy")
        
        observation = env.reset(request)
        return {"observation": observation.model_dump()}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


@app.post("/step")
def step(request: StepRequest):
    """
    Execute one environment step with an action
    
    Args:
        request: StepRequest containing action
        
    Returns:
        StepResult with observation, reward, done flag
    """
    try:
        if not request or not request.action:
            raise ValueError("Action required in request")
        
        result = env.step(request.action)
        return result.model_dump()
        
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid action: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Step failed: {str(e)}")


@app.get("/state")
def get_state():
    """
    Get current environment state
    
    Returns:
        Complete environment state including all entities and metrics
    """
    try:
        state = env.get_state()
        return state
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"State retrieval failed: {str(e)}")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "healthcare-scheduling-env",
        "version": "2.0"
    }


@app.get("/")
def root():
    """API root with basic information"""
    return {
        "name": "Healthcare Appointment Scheduling RL Environment",
        "version": "2.0",
        "endpoints": {
            "reset": "POST /reset - Reset environment",
            "step": "POST /step - Execute step",
            "state": "GET /state - Get current state",
            "health": "GET /health - Health check",
            "docs": "GET /docs - API documentation"
        },
        "tasks": ["easy", "medium", "hard"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)