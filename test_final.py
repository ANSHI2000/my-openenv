import uvicorn
import requests
import json
import threading
import time

def start_server():
    from app import app
    uvicorn.run(app, host='127.0.0.1', port=8002, log_level='error')

t = threading.Thread(target=start_server, daemon=True)
t.start()
time.sleep(3)

try:
    r = requests.post('http://127.0.0.1:8002/reset', json={}, timeout=10)
    data = json.loads(r.text)
    print(f'Status: {r.status_code}')
    print(f'Valid JSON: True')
    print(f'Has observation: {"observation" in data}')
    print(f'Has done: {"done" in data}')
    print(f'Done value: {data.get("done")}')
    if "observation" in data:
        print(f'Observation task: {data["observation"].get("task_name")}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
