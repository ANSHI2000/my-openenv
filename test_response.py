from app import app
from models import ResetRequest
from env.healthcare_env import HealthcareSchedulingEnv
import json

env = HealthcareSchedulingEnv()
obs = env.reset(ResetRequest(task='easy'))
response = {"observation": obs.model_dump()}

# Try to serialize it
try:
    json_str = json.dumps(response)
    print("✓ JSON serialization works")
    print(f"Response size: {len(json_str)} bytes")
    # Try to parse it back
    parsed = json.loads(json_str)
    print("✓ JSON parse works")
    print(f"Keys in response: {list(parsed.keys())}")
    print(f"Keys in observation: {list(parsed['observation'].keys())}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
