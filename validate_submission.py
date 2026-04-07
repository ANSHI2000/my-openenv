#!/usr/bin/env python
"""
validate-submission.py — OpenEnv Submission Validator

Checks that your HF Space is live, Docker image builds, and meets OpenEnv requirements.

Prerequisites:
    - Docker:       https://docs.docker.com/get-docker/
    - pip packages: requests (usually pre-installed)

Usage:
    python validate_submission.py <ping_url> [repo_dir]

Arguments:
    ping_url   Your HuggingFace Space URL (e.g. https://your-space.hf.space)
    repo_dir   Path to your repo (default: current directory)

Examples:
    python validate_submission.py https://my-team-hospital.hf.space
    python validate_submission.py https://my-team-hospital.hf.space ./hospital-rl-env

"""

import sys
import os
import subprocess
import json
import time
from pathlib import Path
from typing import Tuple, Optional
from datetime import datetime, timezone
from urllib.parse import urljoin

try:
    import requests
except ImportError:
    print("ERROR: requests module not found. Install it: pip install requests")
    sys.exit(1)


# Color codes for output
class Color:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BOLD = '\033[1m'
    NC = '\033[0m'  # No Color
    
    @staticmethod
    def disable():
        """Disable colors for non-TTY output"""
        Color.RED = ''
        Color.GREEN = ''
        Color.YELLOW = ''
        Color.BOLD = ''
        Color.NC = ''


# Check if output is to a terminal
if not sys.stdout.isatty():
    Color.disable()


DOCKER_BUILD_TIMEOUT = 600
PASS_COUNT = 0
FAIL_COUNT = 0


def log(msg: str) -> None:
    """Log with timestamp"""
    try:
        ts = datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
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


def validate_hf_space(ping_url: str) -> bool:
    """Step 1: Ping HF Space /reset endpoint"""
    section(f"Step 1/4: Pinging HF Space ({ping_url}/reset)")
    
    ping_url_full = urljoin(ping_url.rstrip('/'), '/reset')
    
    try:
        log(f"Making POST request to {ping_url_full}...")
        response = requests.post(
            ping_url_full,
            json={},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            passed("HF Space is live and responds to /reset")
            return True
        else:
            failed(f"HF Space /reset returned HTTP {response.status_code} (expected 200)")
            hint(f"Make sure your Space is running and the URL is correct.")
            hint(f"Try opening {ping_url} in your browser first.")
            return False
    except requests.exceptions.ConnectionError:
        failed("HF Space not reachable (connection failed)")
        hint("Check your network connection and that the Space is running.")
        hint(f"Try: curl -s -X POST {ping_url}/reset")
        return False
    except requests.exceptions.Timeout:
        failed("HF Space request timed out (30s)")
        hint("The Space may be loading. Try again in a moment.")
        return False
    except Exception as e:
        failed(f"HF Space request failed: {e}")
        return False


def validate_dockerfile(repo_dir: str) -> bool:
    """Step 2: Check and validate Dockerfile"""
    section("Step 2/4: Validating Dockerfile")
    
    # Check for Dockerfile
    dockerfile_paths = [
        Path(repo_dir) / "Dockerfile",
        Path(repo_dir) / "server" / "Dockerfile",
    ]
    
    dockerfile = None
    docker_context = None
    for path in dockerfile_paths:
        if path.exists():
            dockerfile = path
            docker_context = path.parent
            break
    
    if not dockerfile:
        failed("No Dockerfile found in repo root or server/ directory")
        hint(f"Expected Dockerfile at {repo_dir}/Dockerfile")
        return False
    
    log(f"Found Dockerfile at {dockerfile}")
    passed("Dockerfile found")
    
    # Check Docker is installed
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise RuntimeError("Docker not available")
        log(f"Found: {result.stdout.strip()}")
    except (FileNotFoundError, RuntimeError):
        failed("docker command not found")
        hint("Install Docker: https://docs.docker.com/get-docker/")
        return False
    
    # Try to build Docker image
    log(f"Building Docker image from {docker_context}...")
    try:
        result = subprocess.run(
            ["docker", "build", str(docker_context)],
            capture_output=True,
            text=True,
            timeout=DOCKER_BUILD_TIMEOUT
        )
        
        if result.returncode == 0:
            passed("Docker build succeeded")
            return True
        else:
            failed("Docker build failed")
            # Show last 20 lines of output
            lines = result.stderr.split('\n')
            for line in lines[-20:]:
                if line.strip():
                    print(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        failed(f"Docker build timed out (timeout={DOCKER_BUILD_TIMEOUT}s)")
        hint("Complex Dockerfile may take longer. Run locally: docker build .")
        return False
    except Exception as e:
        failed(f"Docker build error: {e}")
        return False


def validate_openenv_spec(repo_dir: str) -> bool:
    """Step 3: Validate OpenEnv specification compliance"""
    section("Step 3/4: Checking OpenEnv Specification Compliance")
    
    # Check required files
    required_files = [
        ("openenv.yaml", "OpenEnv config"),
        ("inference.py", "Baseline inference script"),
        ("app.py", "FastAPI application"),
        ("requirements.txt", "Python dependencies"),
    ]
    
    all_found = True
    for filename, description in required_files:
        filepath = Path(repo_dir) / filename
        if filepath.exists():
            passed(f"{description} ({filename}) found")
        else:
            failed(f"{description} ({filename}) not found")
            all_found = False
    
    if not all_found:
        return False
    
    # Check openenv.yaml structure
    openenv_yaml = Path(repo_dir) / "openenv.yaml"
    try:
        import yaml
        with open(openenv_yaml) as f:
            config = yaml.safe_load(f)
        
        required_keys = ["name", "endpoints", "reward_range"]
        for key in required_keys:
            if key in config:
                passed(f"openenv.yaml has '{key}' field")
            else:
                failed(f"openenv.yaml missing '{key}' field")
                all_found = False
    except ImportError:
        log("⚠️  yaml not installed (skipping detailed validation)")
        log("  Install: pip install pyyaml")
    except Exception as e:
        failed(f"Error reading openenv.yaml: {e}")
        all_found = False
    
    # Check inference.py format
    inference_py = Path(repo_dir) / "inference.py"
    try:
        with open(inference_py) as f:
            content = f.read()
        
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
                passed(f"inference.py {description}")
            else:
                failed(f"inference.py missing {description}")
                all_found = False
    except Exception as e:
        failed(f"Error reading inference.py: {e}")
        all_found = False
    
    # Check for graders
    graders_py = Path(repo_dir) / "graders.py"
    if graders_py.exists():
        try:
            with open(graders_py) as f:
                content = f.read()
            
            grader_functions = ["grade_easy", "grade_medium", "grade_hard"]
            for func in grader_functions:
                if f"def {func}" in content:
                    passed(f"graders.py has {func}()")
                else:
                    failed(f"graders.py missing {func}()")
                    all_found = False
        except Exception as e:
            failed(f"Error reading graders.py: {e}")
    
    return all_found


def validate_test_suite(repo_dir: str) -> bool:
    """Step 4: Validate test suite runs"""
    section("Step 4/4: Running Test Suite")
    
    test_file = Path(repo_dir) / "test_demo.py"
    if not test_file.exists():
        failed("test_demo.py not found")
        hint("Create a test file to verify your environment works")
        return False
    
    log("Running test suite...")
    try:
        result = subprocess.run(
            ["python", str(test_file)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_dir
        )
        
        if result.returncode == 0:
            passed("Test suite passed")
            # Show summary if available
            if "SUMMARY" in result.stdout:
                lines = result.stdout.split('\n')
                for i, line in enumerate(lines):
                    if "SUMMARY" in line:
                        for summary_line in lines[i:i+5]:
                            if summary_line.strip():
                                log(f"  {summary_line}")
            return True
        else:
            failed("Test suite failed")
            # Show last 10 lines
            lines = result.stdout.split('\n') if result.stdout else []
            for line in lines[-10:]:
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
    """Main validation runner"""
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"Usage: {sys.argv[0]} <ping_url> [repo_dir]")
        sys.exit(1)
    
    ping_url = sys.argv[1].rstrip('/')
    repo_dir = sys.argv[2] if len(sys.argv) > 2 else "."
    
    # Resolve repo_dir to absolute path
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
    print(f"{Color.BOLD}  OpenEnv Submission Validator{Color.NC}")
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    log(f"Repo:      {repo_dir}")
    log(f"Ping URL:  {ping_url}")
    print()
    
    # Run validations
    checks_passed = True
    
    if not validate_hf_space(ping_url):
        checks_passed = False
        stop_at("Step 1")
    
    if not validate_dockerfile(repo_dir):
        checks_passed = False
        stop_at("Step 2")
    
    if not validate_openenv_spec(repo_dir):
        checks_passed = False
        stop_at("Step 3")
    
    if not validate_test_suite(repo_dir):
        checks_passed = False
        stop_at("Step 4")
    
    # Summary
    print()
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    print(f"{Color.GREEN}{Color.BOLD}  All 4/4 checks passed!{Color.NC}")
    print(f"{Color.GREEN}{Color.BOLD}  Your submission is ready to submit.{Color.NC}")
    print(f"{Color.BOLD}{'='*50}{Color.NC}")
    print()
    
    return 0 if checks_passed else 1


if __name__ == "__main__":
    sys.exit(main())
