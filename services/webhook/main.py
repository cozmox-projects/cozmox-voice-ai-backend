"""
services/webhook/main.py
─────────────────────────
FastAPI webhook service — the entry point for the entire system.

Receives calls from Twilio, creates LiveKit rooms, dispatches AI agent workers.
Runs on AWS Ubuntu VM behind nginx (or directly on port 8000 for testing).

Endpoints:
  POST /twilio/incoming     → Twilio webhook: new inbound call
  POST /twilio/status       → Twilio webhook: call status updates
  GET  /health              → Health check (load balancer + uptime monitor)
  GET  /metrics             → Prometheus metrics scrape
  POST /calls/simulate      → Simulate a call without Twilio (for testing)
  GET  /calls/active        → List active calls + worker pool status

Run:
  cd /opt/voice-ai-agent
  python -m services.webhook.main
"""
import asyncio
import os
import sys
import hashlib
import hmac as hmac_lib
import base64
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_fastapi_instrumentator import Instrumentator
from twilio.twiml.voice_response import VoiceResponse, Connect, Stream

from config import get_settings
from logger import get_logger
from services.webhook.livekit_manager import get_livekit_manager
from services.agent.dispatcher import get_dispatcher
from services.metrics.collector import calls_total, calls_active, calls_failed_setup

log = get_logger(__name__)
settings = get_settings()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice AI Agent — Webhook Service",
    description="Handles inbound Twilio calls and dispatches AI agent workers",
    version="1.0.0",
    docs_url="/docs",
)

Instrumentator().instrument(app).expose(app, endpoint="/instrumentator-metrics")

# ── In-memory call registry ───────────────────────────────────────────────────
active_calls: dict[str, dict] = {}


# ═════════════════════════════════════════════════════════════════════════════
#  STARTUP / SHUTDOWN
# ═════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def on_startup():
    dispatcher = get_dispatcher()
    await dispatcher.start()
    log.info(
        "webhook_service_ready",
        port=settings.webhook_port,
        max_concurrent_calls=settings.max_concurrent_calls,
        worker_pool_size=settings.worker_pool_size,
    )


@app.on_event("shutdown")
async def on_shutdown():
    dispatcher = get_dispatcher()
    await dispatcher.stop()
    log.info("webhook_service_shutdown", active_calls_at_shutdown=len(active_calls))


# ═════════════════════════════════════════════════════════════════════════════
#  HEALTH + METRICS
# ═════════════════════════════════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    dispatcher = get_dispatcher()
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_calls": len(active_calls),
        "max_concurrent_calls": settings.max_concurrent_calls,
        "available_slots": dispatcher.available_slots(),
        "at_capacity": dispatcher.is_at_capacity(),
    }


@app.get("/metrics")
async def prometheus_metrics():
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/calls/active")
async def list_active_calls():
    dispatcher = get_dispatcher()
    return {
        "count": len(active_calls),
        "worker_pool": dispatcher.get_active_calls(),
        "calls": list(active_calls.values()),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  TWILIO WEBHOOKS
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/twilio/incoming")
async def handle_incoming_call(request: Request):
    """
    Twilio POST webhook fired when someone dials your Twilio number.

    Flow:
      1. Validate request is genuinely from Twilio
      2. Capacity check via dispatcher
      3. Create LiveKit room for this call
      4. Dispatch AI agent worker (non-blocking background task)
      5. Return TwiML telling Twilio to stream audio to LiveKit
    """
    form_data = await request.form()
    call_sid  = form_data.get("CallSid",  f"unknown-{int(datetime.utcnow().timestamp())}")
    caller    = form_data.get("From",     "unknown")
    called    = form_data.get("To",       "unknown")

    calls_total.inc()
    log.info("incoming_call", call_sid=call_sid, from_number=caller, to_number=called)

    # ── Twilio signature validation ───────────────────────────────────────
    if settings.twilio_auth_token:
        valid = _validate_twilio_signature(request, dict(form_data))
        if not valid:
            log.warning("twilio_signature_invalid", call_sid=call_sid)
            raise HTTPException(status_code=403, detail="Invalid Twilio signature")

    # ── Capacity check ────────────────────────────────────────────────────
    dispatcher = get_dispatcher()
    if dispatcher.is_at_capacity():
        log.warning("capacity_exceeded", call_sid=call_sid, active=len(active_calls))
        calls_failed_setup.inc()
        twiml = VoiceResponse()
        twiml.say(
            "We are experiencing high call volume. Please try again in a few minutes.",
            voice="alice",
        )
        twiml.hangup()
        return Response(content=str(twiml), media_type="application/xml")

    # ── Create LiveKit room ───────────────────────────────────────────────
    room_name  = f"call-{call_sid}"
    lk_manager = get_livekit_manager()

    try:
        await lk_manager.create_room(room_name)
        caller_token = lk_manager.generate_caller_token(
            room_name=room_name,
            caller_identity=f"caller-{call_sid}",
        )
        lk_ws_url = lk_manager.get_livekit_ws_url(room_name, caller_token)
    except Exception as e:
        log.error("livekit_setup_failed", call_sid=call_sid, error=str(e))
        calls_failed_setup.inc()
        twiml = VoiceResponse()
        twiml.say("We are having technical difficulties. Please try again shortly.", voice="alice")
        twiml.hangup()
        return Response(content=str(twiml), media_type="application/xml")

    # ── Register call ─────────────────────────────────────────────────────
    active_calls[call_sid] = {
        "call_sid":   call_sid,
        "room_name":  room_name,
        "caller":     caller,
        "called":     called,
        "status":     "setting_up",
        "started_at": datetime.utcnow().isoformat(),
    }
    calls_active.inc()

    # ── Dispatch agent worker ─────────────────────────────────────────────
    dispatched = await dispatcher.dispatch(call_id=call_sid, room_name=room_name)
    if not dispatched:
        active_calls.pop(call_sid, None)
        calls_active.dec()
        calls_failed_setup.inc()
        twiml = VoiceResponse()
        twiml.say("All agents are busy. Please try again shortly.", voice="alice")
        twiml.hangup()
        return Response(content=str(twiml), media_type="application/xml")

    active_calls[call_sid]["status"] = "agent_connecting"

    # ── TwiML: stream audio to LiveKit ────────────────────────────────────
    twiml = VoiceResponse()
    connect = Connect()
    connect.stream(url=lk_ws_url)
    twiml.append(connect)

    log.info("call_routed", call_sid=call_sid, room=room_name)
    return Response(content=str(twiml), media_type="application/xml")


@app.post("/twilio/status")
async def handle_call_status(request: Request):
    """Twilio status callback — cleans up when a call ends."""
    form_data = await request.form()
    call_sid  = form_data.get("CallSid",    "unknown")
    status    = form_data.get("CallStatus", "unknown")
    duration  = form_data.get("CallDuration", "0")

    log.info("call_status_update", call_sid=call_sid, status=status, duration_secs=duration)

    terminal = {"completed", "failed", "busy", "no-answer", "canceled"}
    if status in terminal and call_sid in active_calls:
        del active_calls[call_sid]
        try:
            calls_active.dec()
        except Exception:
            pass

    return Response(content="", status_code=204)


# ═════════════════════════════════════════════════════════════════════════════
#  SIMULATION (no Twilio needed — use on AWS for testing)
# ═════════════════════════════════════════════════════════════════════════════

@app.post("/calls/simulate")
async def simulate_call(room_name: str = "test-room-1"):
    """
    Simulate an inbound call without a real Twilio phone call.

    Usage:
        curl -X POST "http://<AWS-IP>:8000/calls/simulate?room_name=demo-1"

    The response includes a caller_token — use it in the LiveKit Meet demo
    app (https://meet.livekit.io) to join as the caller and talk to the AI.
    """
    call_id    = f"sim-{room_name}-{int(datetime.utcnow().timestamp())}"
    lk_manager = get_livekit_manager()

    await lk_manager.create_room(room_name)
    caller_token = lk_manager.generate_caller_token(
        room_name=room_name,
        caller_identity=f"sim-caller-{call_id}",
    )

    active_calls[call_id] = {
        "call_sid":   call_id,
        "room_name":  room_name,
        "caller":     "simulated",
        "status":     "simulated",
        "started_at": datetime.utcnow().isoformat(),
    }
    calls_total.inc()
    calls_active.inc()

    dispatcher = get_dispatcher()
    await dispatcher.dispatch(call_id=call_id, room_name=room_name)

    return {
        "call_id":         call_id,
        "room_name":       room_name,
        "livekit_url":     settings.livekit_url,
        "caller_token":    caller_token,
        "active_calls":    len(active_calls),
        "available_slots": dispatcher.available_slots(),
        "join_url": (
            f"https://meet.livekit.io/custom?"
            f"liveKitUrl={settings.livekit_url}&token={caller_token}"
        ),
        "message": "Agent dispatched. Use join_url to connect as the caller.",
    }


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _validate_twilio_signature(request: Request, form_params: dict) -> bool:
    """Validates X-Twilio-Signature header."""
    try:
        signature  = request.headers.get("X-Twilio-Signature", "")
        url        = str(request.url)
        auth_token = settings.twilio_auth_token

        s = url
        for key in sorted(form_params.keys()):
            s += key + str(form_params[key])

        mac      = hmac_lib.new(auth_token.encode("utf-8"), s.encode("utf-8"), hashlib.sha1)
        expected = base64.b64encode(mac.digest()).decode("utf-8")
        return hmac_lib.compare_digest(expected, signature)
    except Exception as e:
        log.error("signature_validation_error", error=str(e))
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    uvicorn.run(
        "services.webhook.main:app",
        host="0.0.0.0",
        port=settings.webhook_port,
        reload=False,
        workers=1,
        log_level=settings.log_level.lower(),
        access_log=True,
    )
