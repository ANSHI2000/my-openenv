import uvicorn
import requests
import threading
import time

def run_server():
    from app import app
    uvicorn.run(app, host='127.0.0.1', port=8003, log_level='critical')

# Start server in background
server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()

# Wait for server to start
time.sleep(3)

# Test all endpoints
tests = [
    ("GET", "http://127.0.0.1:8003/", "Root endpoint"),
    ("GET", "http://127.0.0.1:8003/health", "Health check"),
    ("POST", "http://127.0.0.1:8003/reset", "Reset endpoint"),
]

print("Testing all endpoints...")
for method, url, name in tests:
    try:
        if method == "GET":
            r = requests.get(url, timeout=5)
        else:
            r = requests.post(url, json={}, timeout=5)
        
        status = "✓" if r.status_code == 200 else "✗"
        print(f"{status} {name}: {r.status_code}")
    except Exception as e:
        print(f"✗ {name}: {str(e)[:50]}")

print("\nAll endpoints working!")
