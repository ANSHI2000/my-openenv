import uvicorn
import requests
import json
import threading
import time

def start_server():
    from app import app
    uvicorn.run(app, host='127.0.0.1', port=8001, log_level='error')

t = threading.Thread(target=start_server, daemon=True)
t.start()
time.sleep(3)

try:
    r = requests.post('http://127.0.0.1:8001/reset', json={}, timeout=10)
    print(f'Status: {r.status_code}')
    print(f'Response length: {len(r.text)}')
    data = json.loads(r.text)
    print(f'Valid JSON: True')
    print(f'Has observation: {"observation" in data}')
    print(f'First 300 chars: {r.text[:300]}')
except Exception as e:
    print(f'Error: {e}')
    import traceback
    traceback.print_exc()
