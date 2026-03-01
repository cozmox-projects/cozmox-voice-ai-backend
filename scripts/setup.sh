#!/bin/bash
# ─────────────────────────────────────────────────────────────
#  scripts/setup.sh
#  One-command setup for the Voice AI Agent system
#  
#  Usage: chmod +x scripts/setup.sh && ./scripts/setup.sh
# ─────────────────────────────────────────────────────────────

set -e

BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Voice AI Agent — Setup Script        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Check prerequisites ───────────────────────────────────────────────
echo -e "${YELLOW}[1/6] Checking prerequisites...${NC}"

command -v docker >/dev/null 2>&1 || { echo -e "${RED}❌ Docker not found. Install Docker Desktop first.${NC}"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ Python 3 not found.${NC}"; exit 1; }
command -v pip3 >/dev/null 2>&1 || { echo -e "${RED}❌ pip3 not found.${NC}"; exit 1; }

echo -e "${GREEN}✅ Docker, Python3, pip3 found${NC}"

# ── Step 2: Copy env file ─────────────────────────────────────────────────────
echo -e "${YELLOW}[2/6] Setting up environment...${NC}"

if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✅ .env file created from .env.example${NC}"
    echo -e "${YELLOW}   ⚠️  Edit .env and fill in your API keys before continuing!${NC}"
    echo ""
    echo "   Required keys:"
    echo "     - DEEPGRAM_API_KEY"
    echo "     - AZURE_OPENAI_API_KEY + AZURE_OPENAI_ENDPOINT"
    echo "     - ELEVENLABS_API_KEY"
    echo ""
    read -p "Press Enter after filling in your .env file..."
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# ── Step 3: Create data directories ──────────────────────────────────────────
echo -e "${YELLOW}[3/6] Creating data directories...${NC}"
mkdir -p data/chromadb
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"

# ── Step 4: Install Python dependencies ──────────────────────────────────────
echo -e "${YELLOW}[4/6] Installing Python dependencies...${NC}"
pip3 install -r requirements.txt --quiet
echo -e "${GREEN}✅ Python packages installed${NC}"

# ── Step 5: Start infrastructure ─────────────────────────────────────────────
echo -e "${YELLOW}[5/6] Starting infrastructure (LiveKit, Prometheus, Grafana, Redis)...${NC}"
docker-compose up -d

echo "   Waiting for services to be ready..."
sleep 5

# Check services
docker ps | grep livekit >/dev/null 2>&1 && echo -e "${GREEN}   ✅ LiveKit running${NC}" || echo -e "${RED}   ❌ LiveKit failed to start${NC}"
docker ps | grep prometheus >/dev/null 2>&1 && echo -e "${GREEN}   ✅ Prometheus running${NC}" || echo -e "${RED}   ❌ Prometheus failed to start${NC}"
docker ps | grep grafana >/dev/null 2>&1 && echo -e "${GREEN}   ✅ Grafana running${NC}" || echo -e "${RED}   ❌ Grafana failed to start${NC}"
docker ps | grep redis >/dev/null 2>&1 && echo -e "${GREEN}   ✅ Redis running${NC}" || echo -e "${RED}   ❌ Redis failed to start${NC}"

# ── Step 6: Seed knowledge base ───────────────────────────────────────────────
echo -e "${YELLOW}[6/6] Seeding knowledge base...${NC}"
python3 scripts/seed_knowledge_base.py
echo -e "${GREEN}✅ Knowledge base ready${NC}"

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ✅ Setup Complete!                       ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════╝${NC}"
echo ""
echo "  Next steps:"
echo ""
echo "  1. Start the webhook service:"
echo "     python services/webhook/main.py"
echo ""
echo "  2. (Optional) For Twilio: expose webhook with ngrok:"
echo "     ngrok http 8000"
echo "     → Paste https URL into Twilio console as webhook"
echo ""
echo "  3. Test without Twilio (simulate a call):"
echo "     curl -X POST 'http://localhost:8000/calls/simulate?room_name=test-room-1'"
echo ""
echo "  4. View dashboards:"
echo "     Grafana:    http://localhost:3000  (admin/admin)"
echo "     Prometheus: http://localhost:9090"
echo "     LiveKit:    http://localhost:7880"
echo ""
