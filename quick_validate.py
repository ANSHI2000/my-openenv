#!/usr/bin/env python
"""
quick_validate.py — Quick OpenEnv Submission Validator

Fast validation without Docker (good for development/testing).

Prerequisites:
    - Python 3.9+
    - requests library (pip install requests)
    - API server running locally (uvicorn app:app)

Usage:
    python quick_validate.py <ping_url> [repo_dir]

Arguments:
    ping_url   Your HuggingFace Space URL or local API (e.g. http://localhost:8000)
    repo_dir   Path to your repo (default: current directory)

Examples:
    python quick_validate.py http://localhost:8000
    python quick_validate.py https://my-team-hospital.hf.space
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("ERROR: requests module not found. Install: pip install requests")
    sys.exit(1)


class Color:
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BOLD = '\033[1m'
    NC = '\033[0m'
    
    @staticmethod
    def disable():
        Color.RED = Color.GREEN = Color.YELLOW = Color.BOLD = Color.NC = ''


if not sys.stdout.isatty():
    Color.disable()


PASS_COUNT = 0
FAIL_COUNT = 0


def log(msg: str) -> None:
    """Log with timestamp"""
    try:
        ts = datetime.now().strftime("%H:%M:%S")
    except:
        ts = datetime.utcnow().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def passed(msg: str) -> None:
    """Log passing check"""
    global PASS_COUNT
    log(f"{Color.GREEN}PASSED{Color.NC} -- {msg}")
    PASS_COUNT += 1


def failed(msg: str) -> None:
    """Log failing check"""
    global FAIL_COUNT
    log(f"{Color.RED}FAILED{Color.NC} -- {msg}")
    FAIL_COUNT += 1


def hint(msg: str) -> None:
    """Print hint"""
    print(f"  {Color.YELLOW}Hint:{Color.NC} {msg}")


def section(title: str) -> None:
    """Print section header"""
    print(f"\n{Color.BOLD}{title}{Color.NC}")
    print("=" * 50)


def stop_at(step: str) -> None:
    """Stop validation at a step"""
    print()
    print(f"{Color.RED}{Color.BOLD}Validation stopped at {step}.{Color.NC} Fix the above before continuing.")
    sys.exit(1)


def validate_api_endpoint(ping_url: str) -> bool:
    """Check API /reset endpoint responds"""
    section("Step 1/4: Pinging API Endpoint")
    
    ping_url = ping_url.rstrip('/')
    reset_url = urljoin(ping_url, '/reset')
    
    try:
        log(f"Checking {reset_url}...")
        response = requests.post(
            reset_url,
            json={},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        
        if response.status_code == 200:
            passed("API /reset endpoint responds with HTTP 200")
            return True
        else:
            failed(f"API /reset returned HTTP {response.status_code} (expected 200)")
            return False
    except requests.exceptions.ConnectionError:
        failed("Cannot connect to API (connection failed)")
        hint(f"Make sure API is running. For local: uvicorn app:app")
        hint(f"For HF Space: verify URL is correct")
        return False
    except requests.exceptions.Timeout:
        failed("API request timed out")
        hint("API may be overloaded. Try again in a moment.")
        return False
    except Exception as e:
        failed(f"API request error: {e}")
        return False


def validate_required_files(repo_dir: str) -> bool:
    """Check all required files exist"""
    section("Step 2/4: Checking Required Files")
    
    required_files = [
        ("openenv.yaml", "OpenEnv configuration"),
        ("inference.py", "Baseline inference script"),
        ("app.py", "FastAPI application"),
        ("graders.py", "Task graders"),
        ("requirements.txt", "Python dependencies"),
        ("Dockerfile", "Docker configuration"),
        ("README.md", "Documentation"),
    ]
    
    all_found = True
    for filename, description in required_files:
        filepath = Path(repo_dir) / filename
        if filepath.exists():
            passed(f"Found {description} ({filename})")
        else:
            failed(f"Missing {description} ({filename})")
            all_found = False
    
    return all_found


def validate_openenv_compliance(repo_dir: str) -> bool:
    """Check OpenEnv specification compliance"""
    section("Step 3/4: Checking OpenEnv Specification")
    
    all_valid = True
    
    # Check inference.py
    inference_py = Path(repo_dir) / "inference.py"
    if inference_py.exists():
        try:
            with open(inference_py) as f:
                content = f.read()
            
            # Critical checks
            checks = [
                ("from openai import OpenAI", "Uses OpenAI Client"),
                ("API_BASE_URL", "Defines API_BASE_URL"),
                ("MODEL_NAME", "Defines MODEL_NAME"),
                ("HF_TOKEN", "Defines HF_TOKEN"),
                ("[START]", "Uses [START] logging"),
                ("[STEP]", "Uses [STEP] logging"),
                ("[END]", "Uses [END] logging"),
            ]
            
            for pattern, description in checks:
                if pattern in content:
                    passed(f"inference.py: {description}")
                else:
                    failed(f"inference.py: Missing {description}")
                    all_valid = False
        except Exception as e:
            failed(f"Error reading inference.py: {e}")
            all_valid = False
    
    # Check graders.py
    graders_py = Path(repo_dir) / "graders.py"
    if graders_py.exists():
        try:
            with open(graders_py) as f:
                content = f.read()
            
            graders = ["grade_easy", "grade_medium", "grade_hard"]
            for grader in graders:
                if f"def {grader}" in content:
                    passed(f"graders.py: Has {grader}()")
                else:
                    failed(f"graders.py: Missing {grader}()")
                    all_valid = False
        except Exception as e:
            failed(f"Error reading graders.py: {e}")
            all_valid = False
    
    # Check app.py endpoints
    app_py = Path(repo_dir) / "app.py"
    if app_py.exists():
        try:
            with open(app_py) as f:
                content = f.read()
            
            endpoints = ["/reset", "/step", "/state"]
            for endpoint in endpoints:
                if f'"{endpoint}"' in content or f"'{endpoint}'" in content:
                    passed(f"app.py: Has {endpoint} endpoint")
                else:
                    failed(f"app.py: Missing {endpoint} endpoint")
                    all_valid = False
        except Exception as e:
            failed(f"Error reading app.py: {e}")
            all_valid = False
    
    return all_valid


def validate_test_suite(repo_dir: str) -> bool:
    """Check test suite runs"""
    section("Step 4/4: Running Test Suite")
    
    test_file = Path(repo_dir) / "test_demo.py"
    if not test_file.exists():
        failed("test_demo.py not found")
        hint("Create test_demo.py to verify environment works")
        return False
    
    log("Running test suite...")
    try:
        import subprocess
        result = subprocess.run(
            ["python", str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_dir
        )
        
        if result.returncode == 0:
            passed("Test suite passed")
            
            # Show score summary if available
            for line in result.stdout.split('\n'):
                if "Score" in line or "score" in line.lower():
                    log(f"  {line.strip()}")
            
            return True
        else:
            failed(f"Test suite failed (exit code: {result.returncode})")
            # Show output
            if result.stdout:
                for line in result.stdout.split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line}")
            if result.stderr:
                print("  STDERR:")
                for line in result.stderr.split('\n')[-10:]:
                    if line.strip():
                        print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        failed("Test suite timed out (60s)")
        return False
    except Exception as e:
        failed(f"Error running test suite: {e}")
        return False


def main():
    """Main validator"""
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    ping_url = sys.argv[1].rstrip('/')
    repo_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Resolve repo_dir
    try:
        repo_dir = str(Path(repo_dir).resolve())
        if not Path(repo_dir).is_dir():
            print(f"Error: directory '{repo_dir}' not found")
            sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Header
    print()
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    print(f"{Color.BOLD}  Quick OpenEnv Validator (No Docker){Color.NC}")
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    log(f"Repo:    {repo_dir}")
    log(f"Ping:    {ping_url}")
    print()
    
    # Run validations
    checks_ok = True
    
    if not validate_api_endpoint(ping_url):
        checks_ok = False
        stop_at("Step 1")
    
    if not validate_required_files(repo_dir):
        checks_ok = False
        stop_at("Step 2")
    
    if not validate_openenv_compliance(repo_dir):
        checks_ok = False
        stop_at("Step 3")
    
    if not validate_test_suite(repo_dir):
        checks_ok = False
        stop_at("Step 4")
    
    # Summary
    print()
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    print(f"{Color.GREEN}{Color.BOLD}  All 4/4 checks passed!{Color.NC}")
    print(f"{Color.GREEN}{Color.BOLD}  Your submission is ready.{Color.NC}")
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    print()
    
    return 0 if checks_ok else 1


if __name__ == "__main__":
    sys.exit(main())
