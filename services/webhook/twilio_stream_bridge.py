"""
services/webhook/twilio_stream_bridge.py
─────────────────────────────────────────
Twilio Media Streams → LiveKit Audio Bridge.

THE PROBLEM THIS SOLVES
───────────────────────
Twilio's <Connect><Stream> TwiML sends audio over a WebSocket using Twilio's
own JSON envelope protocol. Messages look like:

  {"event": "start",   "start": {"streamSid": "...", "callSid": "..."}}
  {"event": "media",   "media": {"payload": "<base64-mulaw-audio>"}}
  {"event": "stop",    "stop": {}}

LiveKit expects WebRTC participants — it has NO idea what Twilio's JSON
envelope means. So we need this bridge in the middle:

  Twilio PSTN call
      │  (WebSocket + Twilio JSON envelope)
      ▼
  TwilioStreamBridge  ← THIS FILE
      │  decodes JSON, extracts raw μ-law PCM bytes
      │  converts μ-law → int16 PCM
      ▼
  LiveKit Room
      │  (livekit-python rtc SDK, publishes audio track)
      ▼
  AgentWorker subscribes to that track → Deepgram STT → LLM → TTS

FLOW PER CALL
─────────────
1. Twilio dials your number → sends POST to /twilio/incoming
2. Webhook creates a LiveKit room, dispatches AgentWorker
3. Webhook returns TwiML pointing Twilio to /twilio/stream/<room_name>
4. Twilio opens WebSocket to /twilio/stream/<room_name>
5. This bridge accepts the WS, joins the same LiveKit room as "caller-bridge"
6. Forwards audio both ways:
     Twilio → bridge → LiveKit (for STT by the agent)
     LiveKit → bridge → Twilio (TTS audio from the agent back to caller)

MOUNTING
────────
This is a FastAPI WebSocket route. Mount it in services/webhook/main.py:

    from services.webhook.twilio_stream_bridge import router as bridge_router
    app.include_router(bridge_router)
"""

import asyncio
import base64
import json
from typing import Optional

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from livekit import rtc
from livekit.api import AccessToken, VideoGrants

from config import get_settings
from logger import get_logger
from services.metrics.collector import calls_active

log = get_logger(__name__)
settings = get_settings()

router = APIRouter()


# ── μ-law helpers ─────────────────────────────────────────────────────────────

def ulaw_to_pcm16(ulaw_bytes: bytes) -> np.ndarray:
    """
    Decodes μ-law (G.711) encoded bytes to int16 PCM samples.
    Twilio sends 8-bit μ-law at 8000 Hz mono.
    LiveKit AudioFrame requires int16 PCM.
    """
    ulaw = np.frombuffer(ulaw_bytes, dtype=np.uint8).astype(np.int16)
    ulaw_inv = ~ulaw
    sign = ulaw_inv & 0x80
    exponent = (ulaw_inv >> 4) & 0x07
    mantissa = ulaw_inv & 0x0F
    pcm = ((mantissa << 1) + 33) << exponent
    return np.where(sign != 0, -pcm, pcm).astype(np.int16)


def pcm16_to_ulaw(pcm_bytes: bytes) -> bytes:
    """
    Encodes int16 PCM samples to μ-law bytes.
    Used when sending TTS audio from LiveKit back to Twilio.
    """
    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
    # Clamp and take absolute value
    pcm_clamped = np.clip(pcm, -32767, 32767)
    sign = np.where(pcm_clamped < 0, 0x80, 0x00).astype(np.uint8)
    magnitude = np.abs(pcm_clamped).astype(np.int32)

    # Add bias
    magnitude = magnitude + 132
    magnitude = np.clip(magnitude, 0, 32767)

    # Find the exponent
    exp = np.zeros(len(magnitude), dtype=np.uint8)
    for i in range(7, 0, -1):
        exp = np.where((magnitude >> (i + 3)) > 0, i, exp)

    # Compute mantissa
    mantissa = (magnitude >> (exp + 3)).astype(np.uint8) & 0x0F

    ulaw = (~(sign | (exp << 4) | mantissa)).astype(np.uint8)
    return ulaw.tobytes()


# ── Bridge class ──────────────────────────────────────────────────────────────

class TwilioLiveKitBridge:
    """
    Manages one Twilio↔LiveKit bridge session for a single call.

    Lifecycle:
      1. __init__      — create objects
      2. connect()     — join LiveKit room as "caller-bridge-<room>"
      3. handle_twilio_message() — called for each WS message from Twilio
      4. disconnect()  — clean up on call end
    """

    def __init__(self, room_name: str, websocket: WebSocket):
        self.room_name = room_name
        self._ws = websocket
        self._room: Optional[rtc.Room] = None
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None
        self._stream_sid: Optional[str] = None
        self._call_sid: Optional[str] = None
        self._connected = False

    async def connect(self):
        """Join the LiveKit room as the caller-bridge participant."""
        self._room = rtc.Room()

        token = (
            AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(f"caller-bridge-{self.room_name}")
            .with_name("Twilio Caller")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,    # bridge publishes caller's voice
                    can_subscribe=True,  # bridge receives agent's TTS audio
                )
            )
            .to_jwt()
        )

        await self._room.connect(settings.livekit_url, token)

        # Create audio source — this is how we push caller audio into LiveKit
        self._audio_source = rtc.AudioSource(sample_rate=8000, num_channels=1)
        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "caller-audio", self._audio_source
        )
        await self._room.local_participant.publish_track(
            self._audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # Subscribe to agent's audio track to pipe it back to Twilio
        self._room.on("track_subscribed")(self._on_agent_track_subscribed)

        self._connected = True
        log.info("bridge_connected_to_livekit", room=self.room_name)

    def _on_agent_track_subscribed(
        self, track, publication, participant
    ):
        """
        When the AI agent publishes its TTS audio track, start piping
        that audio back to Twilio over the WebSocket.
        """
        if isinstance(track, rtc.RemoteAudioTrack):
            log.info(
                "bridge_subscribed_to_agent_audio",
                room=self.room_name,
                participant=participant.identity,
            )
            asyncio.ensure_future(self._forward_agent_audio_to_twilio(track))

    async def _forward_agent_audio_to_twilio(self, track: rtc.RemoteAudioTrack):
        """
        Reads audio frames from the agent's TTS track and sends them
        back to Twilio as base64-encoded μ-law payloads.
        """
        audio_stream = rtc.AudioStream(track)
        try:
            async for frame_event in audio_stream:
                if not self._connected or not self._stream_sid:
                    break

                pcm_bytes = bytes(frame_event.frame.data)
                ulaw_bytes = pcm16_to_ulaw(pcm_bytes)
                payload = base64.b64encode(ulaw_bytes).decode("utf-8")

                # Twilio expects this exact JSON structure for outbound audio
                message = json.dumps({
                    "event": "media",
                    "streamSid": self._stream_sid,
                    "media": {"payload": payload},
                })
                try:
                    await self._ws.send_text(message)
                except Exception:
                    break  # WebSocket closed
        except Exception as e:
            log.error(
                "bridge_agent_audio_forward_error",
                room=self.room_name,
                error=str(e),
            )

    async def handle_twilio_message(self, raw_message: str):
        """
        Processes one message from Twilio's WebSocket stream.

        Twilio sends three event types:
          - "start"  → call connected, contains streamSid + callSid
          - "media"  → audio chunk (base64 μ-law)
          - "stop"   → call ended

        Returns False if the call has ended (bridge should close).
        """
        try:
            msg = json.loads(raw_message)
        except json.JSONDecodeError:
            log.warning("bridge_invalid_json", room=self.room_name)
            return True  # keep going

        event = msg.get("event")

        if event == "start":
            start = msg.get("start", {})
            self._stream_sid = start.get("streamSid")
            self._call_sid = start.get("callSid")
            log.info(
                "bridge_stream_started",
                room=self.room_name,
                stream_sid=self._stream_sid,
                call_sid=self._call_sid,
            )
            return True

        elif event == "media":
            if not self._connected or not self._audio_source:
                return True

            # Decode Twilio's base64 μ-law payload → int16 PCM → LiveKit frame
            payload_b64 = msg.get("media", {}).get("payload", "")
            if not payload_b64:
                return True

            ulaw_bytes = base64.b64decode(payload_b64)
            pcm_samples = ulaw_to_pcm16(ulaw_bytes)

            frame = rtc.AudioFrame(
                data=pcm_samples.tobytes(),
                sample_rate=8000,
                num_channels=1,
                samples_per_channel=len(pcm_samples),
            )
            await self._audio_source.capture_frame(frame)
            return True

        elif event == "stop":
            log.info("bridge_stream_stopped", room=self.room_name)
            return False  # signal to close

        elif event == "mark":
            # Twilio sends "mark" events for synchronization — safe to ignore
            return True

        else:
            log.debug("bridge_unknown_event", event=event, room=self.room_name)
            return True

    async def disconnect(self):
        """Clean up LiveKit connection."""
        self._connected = False
        if self._room:
            try:
                await self._room.disconnect()
            except Exception:
                pass
        log.info("bridge_disconnected", room=self.room_name)


# ── FastAPI WebSocket route ────────────────────────────────────────────────────

@router.websocket("/twilio/stream/{room_name}")
async def twilio_media_stream(websocket: WebSocket, room_name: str):
    """
    WebSocket endpoint that Twilio connects to for media streaming.

    This URL is returned in the TwiML <Stream> tag from /twilio/incoming.
    Twilio opens this connection immediately after the call is answered.

    The bridge:
      1. Accepts the WebSocket
      2. Joins the LiveKit room (same room the AgentWorker is in)
      3. Forwards audio both ways until the call ends
    """
    await websocket.accept()
    log.info("twilio_ws_connected", room=room_name)

    bridge = TwilioLiveKitBridge(room_name=room_name, websocket=websocket)

    try:
        # Join LiveKit room as the caller participant
        await bridge.connect()

        # Process Twilio messages until call ends
        while True:
            try:
                raw_msg = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,  # 30s timeout — Twilio sends keepalives
                )
            except asyncio.TimeoutError:
                log.warning("twilio_ws_timeout", room=room_name)
                break

            should_continue = await bridge.handle_twilio_message(raw_msg)
            if not should_continue:
                break

    except WebSocketDisconnect:
        log.info("twilio_ws_disconnected", room=room_name)

    except Exception as e:
        log.error("twilio_ws_error", room=room_name, error=str(e))

    finally:
        await bridge.disconnect()
