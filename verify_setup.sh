#!/bin/bash

echo "🔍 Verifying Multimodal Video Analysis Setup"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo -n "Checking Python... "
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    echo -e "${GREEN}✓${NC} Python $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python not found"
fi

# Check Node.js
echo -n "Checking Node.js... "
if command -v node &> /dev/null; then
    NODE_VERSION=$(node --version)
    echo -e "${GREEN}✓${NC} Node $NODE_VERSION"
else
    echo -e "${RED}✗${NC} Node.js not found"
fi

# Check npm
echo -n "Checking npm... "
if command -v npm &> /dev/null; then
    NPM_VERSION=$(npm --version)
    echo -e "${GREEN}✓${NC} npm $NPM_VERSION"
else
    echo -e "${RED}✗${NC} npm not found"
fi

echo ""
echo "📂 Checking Directory Structure"
echo "--------------------------------"

# Check backend
if [ -d "backend" ]; then
    echo -e "${GREEN}✓${NC} backend/ directory exists"

    if [ -d "backend/venv" ]; then
        echo -e "  ${GREEN}✓${NC} Virtual environment exists"
    else
        echo -e "  ${YELLOW}!${NC} Virtual environment not found (run: cd backend && python -m venv venv)"
    fi

    if [ -f "backend/.env" ]; then
        echo -e "  ${GREEN}✓${NC} .env file exists"
        if grep -q "OPENAI_API_KEY=sk-" backend/.env 2>/dev/null; then
            echo -e "  ${GREEN}✓${NC} OpenAI API key configured"
        else
            echo -e "  ${YELLOW}!${NC} OpenAI API key not set in .env"
        fi
    else
        echo -e "  ${YELLOW}!${NC} .env file not found (copy from .env.example)"
    fi

    if [ -f "backend/requirements.txt" ]; then
        echo -e "  ${GREEN}✓${NC} requirements.txt exists"
    fi
else
    echo -e "${RED}✗${NC} backend/ directory not found"
fi

# Check frontend
if [ -d "frontend" ]; then
    echo -e "${GREEN}✓${NC} frontend/ directory exists"

    if [ -d "frontend/node_modules" ]; then
        echo -e "  ${GREEN}✓${NC} Dependencies installed"
    else
        echo -e "  ${YELLOW}!${NC} Dependencies not installed (run: cd frontend && npm install)"
    fi

    if [ -f "frontend/.env.local" ]; then
        echo -e "  ${GREEN}✓${NC} .env.local file exists"
    else
        echo -e "  ${YELLOW}!${NC} .env.local file not found (copy from .env.local.example)"
    fi

    if [ -f "frontend/package.json" ]; then
        echo -e "  ${GREEN}✓${NC} package.json exists"
    fi
else
    echo -e "${RED}✗${NC} frontend/ directory not found"
fi

# Check Redis (optional)
echo ""
echo "🔧 Optional Components"
echo "---------------------"
echo -n "Checking Redis... "
if command -v redis-server &> /dev/null; then
    echo -e "${GREEN}✓${NC} Redis installed"
    if pgrep redis-server > /dev/null; then
        echo -e "  ${GREEN}✓${NC} Redis is running"
    else
        echo -e "  ${YELLOW}!${NC} Redis not running (optional - app works without it)"
    fi
else
    echo -e "${YELLOW}!${NC} Redis not installed (optional - app will use in-memory cache)"
fi

echo ""
echo "📋 Summary"
echo "----------"

ISSUES=0

if [ ! -d "backend/venv" ]; then
    echo -e "${YELLOW}→${NC} Run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    ((ISSUES++))
fi

if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}→${NC} Run: cp backend/.env.example backend/.env (then add your OpenAI API key)"
    ((ISSUES++))
fi

if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}→${NC} Run: cd frontend && npm install"
    ((ISSUES++))
fi

if [ ! -f "frontend/.env.local" ]; then
    echo -e "${YELLOW}→${NC} Run: cp frontend/.env.local.example frontend/.env.local"
    ((ISSUES++))
fi

if [ $ISSUES -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Everything looks good! Run ./start.sh to start the application"
else
    echo -e "${YELLOW}!${NC} Please complete the steps above, then run ./verify_setup.sh again"
fi

echo ""
