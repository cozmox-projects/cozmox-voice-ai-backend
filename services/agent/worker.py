"""
services/agent/worker.py
─────────────────────────
Agent Worker — manages one complete call from LiveKit room join to hangup.

Each call gets its own worker instance. Workers are completely independent.
One crash does NOT affect other calls.

Usage (from dispatcher):
    worker = AgentWorker(call_id="call-CA123", room_name="call-CA123")
    await worker.run()  # blocks until call ends

Usage (standalone for testing):
    python services/agent/worker.py --room test-room-1
"""

import argparse
import asyncio
import os
import sys
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from livekit import rtc

from config import get_settings
from logger import get_logger
from services.agent.pipeline import VoiceAIPipeline
from services.metrics.collector import calls_active, calls_completed, calls_failed_setup

log = get_logger(__name__)
settings = get_settings()


class AgentWorker:
    """
    Manages one call's complete lifecycle:
      1. Connect to LiveKit room
      2. Create audio track + subscribe to caller's audio
      3. Start VoiceAIPipeline
      4. Feed caller audio into Deepgram
      5. On call end / error → clean up
    """

    def __init__(self, call_id: str, room_name: str):
        self.call_id = call_id
        self.room_name = room_name

        self._room: Optional[rtc.Room] = None
        self._pipeline: Optional[VoiceAIPipeline] = None
        self._dg_connection: Any = None
        self._audio_source: Optional[rtc.AudioSource] = None
        self._audio_track: Optional[rtc.LocalAudioTrack] = None

    async def run(self):
        """Main entry point. Blocks until the call ends."""
        # NOTE: workers_active is managed by the dispatcher (WorkerPoolDispatcher).
        # Do NOT increment/decrement it here — the dispatcher already does this
        # when it acquires/releases the semaphore slot, preventing double-counting.
        calls_active.inc()

        try:
            await self._setup()
            await self._main_loop()
            calls_completed.inc()
        except Exception as e:
            calls_failed_setup.inc()
            log.error("worker_fatal_error", call_id=self.call_id, error=str(e))
            raise
        finally:
            await self._teardown()
            calls_active.dec()

    async def _setup(self):
        """Connect to LiveKit and set up audio tracks."""
        log.info("worker_connecting", call_id=self.call_id, room=self.room_name)

        self._room = rtc.Room()

        # Connect to LiveKit room
        # auto_subscribe=True is REQUIRED — without it LiveKit will not deliver
        # remote audio tracks to this agent, causing Deepgram to receive no audio
        # and close with 1011 after ~12 seconds.
        token = self._generate_token()
        await self._room.connect(
            settings.livekit_url,
            token,
            options=rtc.RoomOptions(auto_subscribe=True),
        )
        log.info("worker_connected_to_room", call_id=self.call_id, room=self.room_name)

        # Create an audio source — this is how we send TTS audio to the caller
        self._audio_source = rtc.AudioSource(sample_rate=16000, num_channels=1)
        self._audio_track = rtc.LocalAudioTrack.create_audio_track(
            "agent-audio", self._audio_source
        )

        # Publish our audio track to the room
        await self._room.local_participant.publish_track(
            self._audio_track,
            rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE),
        )

        # Create the AI pipeline, passing our audio sender as callback
        self._pipeline = VoiceAIPipeline(
            call_id=self.call_id,
            room_name=self.room_name,
            send_audio_callback=self._send_audio_to_livekit,
        )

        # Open Deepgram connection for STT
        self._dg_connection = await self._pipeline.create_deepgram_connection()

        log.info("worker_setup_complete", call_id=self.call_id)

    async def _main_loop(self):
        """
        Main event loop for the call.
        - Subscribes to caller's audio track
        - Feeds audio chunks into Deepgram
        - Starts the AI pipeline greeting
        - Waits for call to end
        """
        # Guard: ensure pipeline.start() and _receive_caller_audio() are only
        # invoked once even if multiple audio tracks arrive (e.g. reconnects,
        # multiple participants in simulate mode).
        pipeline_started = False

        # Register the track handler BEFORE pipeline.start() and BEFORE checking
        # existing participants. This avoids a race where the event fires between
        # the two checks.
        @self._room.on("track_subscribed")
        def on_track_subscribed(track, publication, participant):
            nonlocal pipeline_started
            if isinstance(track, rtc.RemoteAudioTrack):
                log.info(
                    "caller_audio_subscribed",
                    call_id=self.call_id,
                    participant=participant.identity,
                )
                asyncio.ensure_future(self._receive_caller_audio(track))
                if not pipeline_started:
                    pipeline_started = True
                    asyncio.ensure_future(self._pipeline.start())

        # Also handle participants who were ALREADY in the room before we connected.
        # This happens in simulate mode where the caller joins before the agent,
        # meaning track_subscribed already fired before our handler was registered.
        for participant in self._room.remote_participants.values():
            for publication in participant.track_publications.values():
                if publication.track and isinstance(
                    publication.track, rtc.RemoteAudioTrack
                ):
                    log.info(
                        "subscribing_to_existing_audio_track",
                        call_id=self.call_id,
                        participant=participant.identity,
                    )
                    asyncio.ensure_future(self._receive_caller_audio(publication.track))
                    if not pipeline_started:
                        pipeline_started = True
                        asyncio.ensure_future(self._pipeline.start())

        # Wait for disconnect event (call ended)
        disconnect_event = asyncio.Event()

        @self._room.on("disconnected")
        def on_disconnect(reason=None):
            log.info("call_ended", call_id=self.call_id, reason=str(reason))
            disconnect_event.set()

        await disconnect_event.wait()

    async def _receive_caller_audio(self, track: rtc.RemoteAudioTrack):
        """
        Reads audio frames from the caller and sends them to Deepgram.
        Runs continuously for the duration of the call.

        rtc.AudioStream always outputs int16 PCM. We request 8000Hz explicitly
        to match Deepgram's expected sample_rate=8000 linear16 config.
        Without specifying sample_rate, LiveKit defaults to 48000Hz which
        would cause Deepgram to produce garbled/no transcripts.
        """
        audio_stream = rtc.AudioStream(track, sample_rate=8000, num_channels=1)
        log.debug("receiving_caller_audio", call_id=self.call_id)

        try:
            async for frame_event in audio_stream:
                audio_bytes = bytes(frame_event.frame.data)
                if self._dg_connection:
                    await self._dg_connection.send(audio_bytes)
        except Exception as e:
            log.error("audio_receive_error", call_id=self.call_id, error=str(e))

    async def _send_audio_to_livekit(self, audio_bytes: bytes):
        """
        Sends TTS audio bytes back to the caller via LiveKit.

        ElevenLabs returns pcm_16000 — raw int16 little-endian PCM at 16kHz.
        HTTP streaming chunks may arrive with an odd number of bytes, which
        cannot be interpreted as int16 samples (2 bytes each). We buffer any
        leftover odd byte across calls and prepend it to the next chunk.
        """
        if not self._audio_source or not audio_bytes:
            return
        try:
            # Prepend any leftover byte from previous chunk
            if getattr(self, "_pcm_remainder", b""):
                audio_bytes = self._pcm_remainder + audio_bytes
                self._pcm_remainder = b""

            # If odd length, hold the last byte for the next chunk
            if len(audio_bytes) % 2 != 0:
                self._pcm_remainder = audio_bytes[-1:]
                audio_bytes = audio_bytes[:-1]

            if not audio_bytes:
                return

            frame = rtc.AudioFrame(
                data=audio_bytes,
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(audio_bytes) // 2,
            )
            await self._audio_source.capture_frame(frame)
        except Exception as e:
            log.error("audio_send_error", call_id=self.call_id, error=str(e))

    async def _teardown(self):
        log.info("worker_teardown", call_id=self.call_id)
        if self._pipeline:
            await self._pipeline.stop()
        if self._dg_connection:
            try:
                await self._dg_connection.finish()
            except Exception:
                pass
        if self._room:
            await self._room.disconnect()

    def _generate_token(self) -> str:
        """Generates a LiveKit access token for this agent to join the room."""
        from livekit.api import AccessToken, VideoGrants

        return (
            AccessToken(settings.livekit_api_key, settings.livekit_api_secret)
            .with_identity(f"agent-{self.call_id}")
            .with_name("Voice AI Agent")
            .with_grants(
                VideoGrants(
                    room_join=True,
                    room=self.room_name,
                    can_publish=True,
                    can_subscribe=True,
                )
            )
            .to_jwt()
        )


# ── Standalone runner (for testing a single call) ─────────────────────────────
async def run_worker(room_name: str, call_id: str):
    worker = AgentWorker(call_id=call_id, room_name=room_name)
    await worker.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--room", required=True)
    parser.add_argument("--call-id", default=None)
    args = parser.parse_args()
    asyncio.run(run_worker(room_name=args.room, call_id=args.call_id or args.room))
