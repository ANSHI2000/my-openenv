#!/bin/bash
# Validation script for Healthcare Scheduling Environment

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get Space URL from argument
SPACE_URL="${1:-.}"

echo -e "${YELLOW}=== Healthcare Scheduling Environment Validator ===${NC}"
echo ""

# Check Dockerfile exists
if [ -f "Dockerfile" ]; then
    echo -e "${GREEN}✓${NC} Dockerfile found"
else
    echo -e "${RED}✗${NC} Dockerfile missing"
    exit 1
fi

# Check inference.py exists
if [ -f "inference.py" ]; then
    echo -e "${GREEN}✓${NC} inference.py found"
else
    echo -e "${RED}✗${NC} inference.py missing"
    exit 1
fi

# Check pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo -e "${GREEN}✓${NC} pyproject.toml found"
else
    echo -e "${RED}✗${NC} pyproject.toml missing"
    exit 1
fi

# Check server directory
if [ -d "server" ]; then
    echo -e "${GREEN}✓${NC} server/ directory found"
    if [ -f "server/app.py" ]; then
        echo -e "${GREEN}✓${NC} server/app.py found"
    else
        echo -e "${RED}✗${NC} server/app.py missing"
        exit 1
    fi
else
    echo -e "${RED}✗${NC} server/ directory missing"
    exit 1
fi

# Test API if URL provided
if [ "$SPACE_URL" != "." ]; then
    echo ""
    echo -e "${YELLOW}Testing API endpoints...${NC}"
    
    # Test /reset endpoint
    if curl -s -X POST "$SPACE_URL/reset" -H "Content-Type: application/json" -d '{}' > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} /reset endpoint responds"
    else
        echo -e "${RED}✗${NC} /reset endpoint not responding"
        exit 1
    fi
    
    # Test /health endpoint
    if curl -s -X GET "$SPACE_URL/health" > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} /health endpoint responds"
    else
        echo -e "${RED}✗${NC} /health endpoint not responding"
        exit 1
    fi
fi

echo ""
echo -e "${GREEN}✓ All checks passed!${NC}"
