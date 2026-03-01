# 🎙️ Voice AI Agent
### Production-ready voice AI system — 100 concurrent PSTN calls

**Stack:** Twilio → LiveKit → Pipecat → Deepgram + Azure OpenAI + ElevenLabs → ChromaDB  
**Runs on:** AWS EC2 Ubuntu 22.04 (t3.large or better recommended)

---

## Table of Contents

1. [What This System Does](#what-this-system-does)
2. [AWS EC2 Setup](#1-aws-ec2-setup)
3. [Install System Dependencies](#2-install-system-dependencies)
4. [Clone & Configure](#3-clone--configure)
5. [Start Infrastructure (Docker)](#4-start-infrastructure-docker)
6. [Install Python Dependencies](#5-install-python-dependencies)
7. [Seed the Knowledge Base](#6-seed-the-knowledge-base)
8. [Start the Webhook Service](#7-start-the-webhook-service)
9. [Test Without Twilio (Simulate Calls)](#8-test-without-twilio-simulate-calls)
10. [Connect Twilio (Real Phone Calls)](#9-connect-twilio-real-phone-calls)
11. [View Metrics Dashboard](#10-view-metrics-dashboard)
12. [Run Concurrent Call Load Test](#11-run-concurrent-call-load-test)
13. [Run as a System Service (Production)](#12-run-as-a-system-service-production)
14. [Troubleshooting](#troubleshooting)
15. [Architecture & Trade-offs](#architecture--trade-offs)

---

## What This System Does

```
Caller's phone
      ↓
  Twilio (phone number → WebSocket audio)
      ↓
  Webhook Service (FastAPI, port 8000)
      ↓ creates LiveKit room + dispatches agent
  LiveKit Server (real-time audio, port 7880)
      ↓ one room per call
  Agent Worker Pool (one Pipecat pipeline per call)
      ↓
  Deepgram STT → Azure OpenAI GPT-4o → ElevenLabs TTS
      ↓ knowledge base lookup via ChromaDB before each LLM call
  Audio response → LiveKit → Twilio → Caller's ear

Target: <600ms end-to-end latency, 100 concurrent calls
```

---

## 1. AWS EC2 Setup

### Recommended Instance

| Use Case | Instance | vCPU | RAM |
|---|---|---|---|
| Development / demo | t3.large | 2 | 8 GB |
| 10–20 concurrent calls | t3.xlarge | 4 | 16 GB |
| 100 concurrent calls | t3.2xlarge | 8 | 32 GB |

### Launch Steps

1. Go to **AWS EC2 Console → Launch Instance**
2. Choose **Ubuntu Server 22.04 LTS (64-bit x86)**
3. Select instance type (t3.large minimum)
4. Under **Key pair** — create or select an existing key pair (you'll need this to SSH in)
5. Under **Network settings → Edit**, configure Security Group:

   | Type | Protocol | Port Range | Source | Why |
   |---|---|---|---|---|
   | SSH | TCP | 22 | Your IP | Admin access |
   | HTTP | TCP | 80 | 0.0.0.0/0 | Twilio webhooks + Nginx |
   | Custom TCP | TCP | 7880 | 0.0.0.0/0 | LiveKit HTTP/WebSocket |
   | Custom TCP | TCP | 7881 | 0.0.0.0/0 | LiveKit RTC TCP |
   | Custom UDP | UDP | 7882 | 0.0.0.0/0 | LiveKit RTC UDP (critical for audio!) |

6. Storage: **30 GB gp3** (minimum)
7. Launch instance and note the **Public IPv4 address**

### SSH into your instance

```bash
ssh -i your-key.pem ubuntu@<AWS-PUBLIC-IP>
```

---

## 2. Install System Dependencies

Run these commands on your EC2 instance:

```bash
# Update system
sudo apt-get update && sudo apt-get upgrade -y

# Install Python 3.11
sudo apt-get install -y python3.11 python3.11-venv python3.11-dev python3-pip

# Make python3 point to 3.11
sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install build tools (needed for some Python packages)
sudo apt-get install -y build-essential git curl wget unzip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker ubuntu

# Install Docker Compose v2
sudo apt-get install -y docker-compose-plugin

# Log out and back in so group membership takes effect
exit
# SSH back in
ssh -i your-key.pem ubuntu@<AWS-PUBLIC-IP>

# Verify Docker works
docker run hello-world
docker compose version
```

---

## 3. Clone & Configure

```bash
# Clone the repo
git clone https://github.com/your-org/voice-ai-agent.git /opt/voice-ai-agent
cd /opt/voice-ai-agent

# Create your environment file
cp .env.example .env
nano .env
```

Fill in `.env` — every field with `your_*` must be replaced:

```env
# ── Deepgram ─────────────────────────────────────
DEEPGRAM_API_KEY=your_deepgram_api_key

# ── Azure OpenAI ──────────────────────────────────
AZURE_OPENAI_API_KEY=your_azure_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01

# ── ElevenLabs ────────────────────────────────────
ELEVENLABS_API_KEY=your_elevenlabs_key
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM

# ── Twilio (leave blank if not using real calls yet)
TWILIO_ACCOUNT_SID=your_twilio_sid
TWILIO_AUTH_TOKEN=your_twilio_token
TWILIO_PHONE_NUMBER=+1xxxxxxxxxx

# ── LiveKit (these are fine as-is for single VM) ──
LIVEKIT_URL=ws://<AWS-PUBLIC-IP>:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=devsecret

# ── App ───────────────────────────────────────────
MAX_CONCURRENT_CALLS=100
WORKER_POOL_SIZE=100
LOG_LEVEL=INFO
WEBHOOK_PORT=8000

# ── ChromaDB ──────────────────────────────────────
CHROMA_PERSIST_DIR=/opt/voice-ai-agent/data/chromadb
```

> **Important:** Replace `<AWS-PUBLIC-IP>` with your actual EC2 public IP in `LIVEKIT_URL`.
> Find it with: `curl -s http://169.254.169.254/latest/meta-data/public-ipv4`

```bash
# Create data directories
mkdir -p /opt/voice-ai-agent/data/chromadb
mkdir -p /opt/voice-ai-agent/logs
```

---

## 4. Start Infrastructure (Docker)

```bash
cd /opt/voice-ai-agent

# Start all Docker services
docker compose up -d

# Verify all 4 containers are running
docker compose ps
```

Expected output:
```
NAME         STATUS    PORTS
livekit      running   (host network)
redis        running   127.0.0.1:6379->6379/tcp
prometheus   running   127.0.0.1:9090->9090/tcp
grafana      running   127.0.0.1:3000->3000/tcp
nginx        running   0.0.0.0:80->80/tcp
```

**Verify LiveKit is up:**
```bash
curl http://localhost:7880/
# Should return: LiveKit server info JSON
```

**View logs if something failed:**
```bash
docker compose logs livekit
docker compose logs nginx
```

---

## 5. Install Python Dependencies

```bash
cd /opt/voice-ai-agent

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all packages
pip install --upgrade pip
pip install -r requirements.txt

# This takes 3–5 minutes (sentence-transformers downloads a model)
```

---

## 6. Seed the Knowledge Base

This loads the documents (refund policy, FAQ, objection scripts) into ChromaDB.  
Run once. Re-run with `--force` if you update the documents.

```bash
cd /opt/voice-ai-agent
source venv/bin/activate

python scripts/seed_knowledge_base.py
```

Expected output:
```
[6 verification queries all returning results]
✅ Knowledge base ready!
```

---

## 7. Start the Webhook Service

```bash
cd /opt/voice-ai-agent
source venv/bin/activate

python -m services.webhook.main
```

You should see:
```
INFO  webhook_service_ready  port=8000  max_concurrent_calls=100  worker_pool_size=100
```

**Test it's running (from your laptop or another terminal):**
```bash
curl http://<AWS-PUBLIC-IP>/health
```

Expected:
```json
{
  "status": "healthy",
  "active_calls": 0,
  "max_concurrent_calls": 100,
  "available_slots": 100,
  "at_capacity": false
}
```

---

## 8. Test Without Twilio (Simulate Calls)

You don't need a real phone to test. The simulate endpoint creates a full AI agent session.

### Simulate a single call

```bash
curl -X POST "http://<AWS-PUBLIC-IP>/calls/simulate?room_name=demo-1"
```

Response includes a `join_url`. Open it in your browser to talk to the AI:
```json
{
  "call_id": "sim-demo-1-1234567890",
  "room_name": "demo-1",
  "join_url": "https://meet.livekit.io/custom?liveKitUrl=ws://...&token=...",
  "message": "Agent dispatched. Use join_url to connect as the caller."
}
```

Open `join_url` in your browser → you'll be in a WebRTC session with the AI agent.  
Click **Join** → speak → the AI should respond within ~600ms.

### Verify active calls

```bash
curl http://<AWS-PUBLIC-IP>/calls/active
```

### Run the automated integration test

```bash
cd /opt/voice-ai-agent
source venv/bin/activate

# Test with simulation only (no Twilio needed)
python tests/twilio_integration_test.py \
  --webhook-url http://<AWS-PUBLIC-IP> \
  --simulate-only
```

---

## 9. Connect Twilio (Real Phone Calls)

### Step 1: Get your Twilio credentials

1. Sign up at [twilio.com](https://www.twilio.com) (free trial gives you $15 credit)
2. Go to [console.twilio.com](https://console.twilio.com)
3. Copy **Account SID** and **Auth Token** → paste into `.env`
4. Click **Get a trial phone number** → copy the number → paste into `.env` as `TWILIO_PHONE_NUMBER`

### Step 2: Configure webhook URL in Twilio

Run the integration test — it auto-configures the webhook for you:

```bash
python tests/twilio_integration_test.py \
  --webhook-url http://<AWS-PUBLIC-IP>
```

Or configure it manually:
1. Go to **Twilio Console → Phone Numbers → Manage → Active Numbers**
2. Click your number
3. Under **Voice & Fax**:
   - **A call comes in**: Webhook → `http://<AWS-PUBLIC-IP>/twilio/incoming`
   - **Call status changes**: `http://<AWS-PUBLIC-IP>/twilio/status`
4. Click **Save**

### Step 3: Make a test call

Call your Twilio number from your mobile.  
You should hear the AI agent greet you: *"Hello! Thank you for calling Acme Corp. How can I help you today?"*

**Try these test phrases:**
- "What is your refund policy?" → AI answers from knowledge base
- "This is too expensive" → AI handles the objection
- "Can I track my order?" → AI answers from FAQ

### Twilio Free Trial Limitations

| Limitation | Workaround |
|---|---|
| Can only call verified numbers | Verify your personal mobile at console.twilio.com |
| Calls have "Twilio trial" announcement | Upgrade to paid ($20 minimum) |
| Limited concurrent calls | Fine for testing; paid tier supports 100+ |

---

## 10. View Metrics Dashboard

Open Grafana in your browser:

```
http://<AWS-PUBLIC-IP>/grafana
Username: admin
Password: admin
```

The **Voice AI Agent** dashboard is pre-loaded. It shows:

- **Active Concurrent Calls** — real-time call count
- **E2E Latency** — p50/p95/p99 in ms (target: avg < 600ms)
- **Per-stage breakdown** — STT / LLM / TTS separately
- **Agent error rates** — per service
- **Knowledge base hit rate** — % of queries answered from KB
- **Barge-in count** — interruption events

**Direct Prometheus queries** (at `http://<AWS-PUBLIC-IP>/prometheus`):

```promql
# Average E2E latency in ms
rate(voice_ai_e2e_latency_seconds_sum[5m]) /
rate(voice_ai_e2e_latency_seconds_count[5m]) * 1000

# 95th percentile E2E latency
histogram_quantile(0.95, rate(voice_ai_e2e_latency_seconds_bucket[5m])) * 1000

# Active calls right now
voice_ai_calls_active

# Failed call setup rate
rate(voice_ai_calls_failed_setup_total[5m])
```

---

## 11. Run Concurrent Call Load Test

Tests the worker pool with N simultaneous AI agent sessions:

```bash
cd /opt/voice-ai-agent
source venv/bin/activate

# Quick test: 10 concurrent calls
python tests/load_test.py --calls 10 --verbose

# Full test: 100 concurrent calls, hold 30s each
python tests/load_test.py --calls 100 --hold 30

# Watch Grafana during the test for live metrics
```

**What to check in the output:**
- `LiveKit connected` should be 100% (or close)
- `Agent responded` should be > 95%
- `avg agent response time` should be < 600ms

---

## 12. Run as a System Service (Production)

Keep the webhook running after you close your SSH session:

```bash
# Install the systemd service
sudo cp /opt/voice-ai-agent/infra/voice-ai-agent.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable voice-ai-agent
sudo systemctl start voice-ai-agent

# Check it's running
sudo systemctl status voice-ai-agent

# View logs
sudo journalctl -u voice-ai-agent -f

# Restart after config changes
sudo systemctl restart voice-ai-agent
```

**Auto-start Docker on reboot:**
```bash
sudo systemctl enable docker
```

---

## Troubleshooting

### "LiveKit not reachable"
```bash
# Check if container is running
docker compose ps livekit
docker compose logs livekit

# Check if ports are open (from your laptop)
nc -zv <AWS-IP> 7880
nc -zv <AWS-IP> 7882  # UDP

# Verify AWS Security Group has UDP 7882 open
```

### "Agent not responding to audio"
```bash
# Check webhook logs
sudo journalctl -u voice-ai-agent -f

# Check Deepgram key is valid
curl https://api.deepgram.com/v1/auth/token \
  -H "Authorization: Token $DEEPGRAM_API_KEY"

# Test Azure OpenAI directly
python3 -c "
from openai import AzureOpenAI
from config import get_settings
s = get_settings()
client = AzureOpenAI(api_key=s.azure_openai_api_key, azure_endpoint=s.azure_openai_endpoint, api_version=s.azure_openai_api_version)
r = client.chat.completions.create(model=s.azure_openai_deployment, messages=[{'role':'user','content':'say hello'}], max_tokens=10)
print(r.choices[0].message.content)
"
```

### "Twilio webhook not firing"
```bash
# Verify your EC2 port 80 is open in Security Group
# Verify nginx is running
docker compose ps nginx
curl http://localhost/health

# Check Twilio webhook logs in console
# https://console.twilio.com/us1/monitor/logs/requests
```

### "High latency (>600ms)"
The biggest bottleneck is usually Azure OpenAI time-to-first-token.

```bash
# Check per-stage breakdown in Grafana
# STT should be ~150ms, LLM ~200-300ms, TTS ~150ms

# Try gpt-4o-mini instead of gpt-4o for lower LLM latency
# In .env: AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

### "Out of memory / system slow"
```bash
# Check memory usage
free -h
docker stats

# If running 100 workers on t3.large (8GB), you may need t3.xlarge
# Each worker uses ~200MB RAM
# 100 workers = ~20GB RAM needed → use t3.2xlarge
```

---

## Architecture & Trade-offs

### LiveKit vs Pipecat

These operate at different layers and complement each other:

| | LiveKit | Pipecat |
|---|---|---|
| **What it is** | Real-time audio/video infrastructure (SFU) | AI voice pipeline framework |
| **Layer** | Network/Transport | Application/AI logic |
| **Responsibility** | Route audio between participants, WebRTC, 100 concurrent rooms | STT→LLM→TTS pipeline, barge-in, turn-taking |
| **Barge-in support** | No | Built-in |
| **AI service integrations** | None | Deepgram, OpenAI, ElevenLabs, 20+ others |
| **Scale** | Thousands of rooms natively | Depends on worker infra |
| **Self-hostable** | Yes (open source) | Yes (open source Python) |

**Analogy:** LiveKit is the highway. Pipecat is the car. You need both.

### What Breaks at 1,000 Calls

| Component | Why It Breaks | Fix |
|---|---|---|
| Single LiveKit server | ~200–500 room limit per node | LiveKit cluster (3–5 nodes + Redis) |
| Single VM workers | 1,000 Python processes = ~200GB RAM | Kubernetes HPA across 10+ nodes |
| Deepgram API | Rate limits on standard tier | Enterprise agreement or self-host Whisper |
| Azure OpenAI | Per-deployment token limits | Multiple deployments across regions + router |
| Single nginx | ~10k req/s max for single process | Multiple nginx + AWS ALB in front |

### Current Latency Bottleneck

LLM time-to-first-token is the dominant cost (~200–350ms of the 600ms budget).

Fix options (in order of impact):
1. Use `gpt-4o-mini` for simple queries → drops to ~100ms
2. Use streaming aggressively (already done) → saves 200ms vs. batch
3. Cache common responses (FAQs) → drops to ~0ms for cached
4. Route simple knowledge-base answers directly to TTS without LLM → eliminates LLM cost entirely for known questions

---

## Quick Reference

```bash
# Start everything
cd /opt/voice-ai-agent
docker compose up -d
source venv/bin/activate
python -m services.webhook.main

# Simulate a call
curl -X POST "http://<AWS-IP>/calls/simulate?room_name=test-1"

# Load test
python tests/load_test.py --calls 10 --verbose

# Twilio integration test
python tests/twilio_integration_test.py --webhook-url http://<AWS-IP>

# View dashboard
open http://<AWS-IP>/grafana

# Check logs
sudo journalctl -u voice-ai-agent -f
docker compose logs -f livekit

# Restart service
sudo systemctl restart voice-ai-agent
docker compose restart
```
