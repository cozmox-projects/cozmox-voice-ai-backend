"""
services/agent/barge_in.py
───────────────────────────
Barge-in controller — allows the caller to interrupt the AI mid-speech.

How it works:
1. While TTS is playing audio to the caller, VAD keeps running in parallel.
2. When VAD detects the caller speaking, this controller fires.
3. It sends a CancelFrame + StartInterruptionFrame into the Pipecat pipeline.
4. The pipeline stops TTS playback and starts listening again.

This is the key UX feature that makes AI voice calls feel natural instead of
like a bad IVR menu.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional
from logger import get_logger
from services.metrics.collector import barge_ins_total

log = get_logger(__name__)


@dataclass
class BargeInController:
    """
    Tracks whether TTS is currently playing, and handles interruptions.
    One instance per call.
    """
    call_id: str
    _tts_playing: bool = field(default=False, init=False)
    _interrupt_event: asyncio.Event = field(default_factory=asyncio.Event, init=False)
    _barge_in_count: int = field(default=0, init=False)

    def on_tts_started(self):
        """Call this when TTS starts sending audio."""
        self._tts_playing = True
        self._interrupt_event.clear()
        log.debug("tts_started", call_id=self.call_id)

    def on_tts_stopped(self):
        """Call this when TTS finishes or is cancelled."""
        self._tts_playing = False
        log.debug("tts_stopped", call_id=self.call_id)

    def on_speech_detected(self) -> bool:
        """
        Called by VAD when caller speech is detected.
        Returns True if this was a barge-in (TTS was playing).
        """
        if self._tts_playing:
            self._barge_in_count += 1
            barge_ins_total.inc()
            self._interrupt_event.set()
            log.info(
                "barge_in_detected",
                call_id=self.call_id,
                total_barge_ins=self._barge_in_count,
            )
            return True
        return False

    async def wait_for_interrupt(self) -> bool:
        """
        Awaitable — resolves when a barge-in occurs.
        Used in the pipeline to cancel TTS when interrupted.
        """
        await self._interrupt_event.wait()
        return True

    @property
    def is_tts_playing(self) -> bool:
        return self._tts_playing

    @property
    def barge_in_count(self) -> int:
        return self._barge_in_count
