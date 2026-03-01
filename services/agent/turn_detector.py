"""
services/agent/turn_detector.py
─────────────────────────────────
Turn detector — decides when the caller has FINISHED speaking.

This is harder than it sounds. People pause mid-sentence, cough, think.
A naive "silence = done" approach cuts people off constantly.

Strategy used here:
  1. VAD detects speech start → we're in an utterance
  2. VAD detects silence → start end-of-turn timer (400ms)
  3. If speech resumes before timer fires → cancel timer, still in utterance
  4. If timer fires → declare end-of-turn, send transcript to LLM

The 400ms silence window is tuned for natural phone conversations.
Too short (200ms) = cuts people off. Too long (800ms) = feels sluggish.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Optional, Callable, Awaitable
from logger import get_logger

log = get_logger(__name__)

# Silence duration (seconds) before declaring end of turn
END_OF_TURN_SILENCE_MS = 400
END_OF_TURN_SILENCE_SECS = END_OF_TURN_SILENCE_MS / 1000


@dataclass
class TurnDetector:
    call_id: str
    on_turn_end: Optional[Callable[[str], Awaitable[None]]] = None

    _in_utterance: bool = field(default=False, init=False)
    _silence_timer: Optional[asyncio.Task] = field(default=None, init=False)
    _current_transcript: str = field(default="", init=False)

    def on_speech_start(self):
        """Called by VAD when caller starts speaking."""
        if not self._in_utterance:
            self._in_utterance = True
            log.debug("turn_started", call_id=self.call_id)

        # Cancel any pending end-of-turn timer (they resumed speaking)
        if self._silence_timer and not self._silence_timer.done():
            self._silence_timer.cancel()
            self._silence_timer = None

    def on_speech_end(self):
        """
        Called by VAD when silence is detected.
        Starts the end-of-turn countdown timer.
        """
        if not self._in_utterance:
            return

        log.debug(
            "silence_detected_starting_timer",
            call_id=self.call_id,
            timeout_ms=END_OF_TURN_SILENCE_MS,
        )

        # Start the end-of-turn timer
        self._silence_timer = asyncio.create_task(
            self._end_of_turn_timer()
        )

    def on_transcript_update(self, partial_transcript: str):
        """Called by Deepgram as it streams partial transcripts."""
        self._current_transcript = partial_transcript

    async def _end_of_turn_timer(self):
        """
        Waits for the silence window. If not cancelled, fires end-of-turn.
        """
        try:
            await asyncio.sleep(END_OF_TURN_SILENCE_SECS)

            # Timer completed — declare end of turn
            transcript = self._current_transcript.strip()
            self._in_utterance = False
            self._current_transcript = ""
            self._silence_timer = None

            log.info(
                "turn_ended",
                call_id=self.call_id,
                transcript=transcript,
            )

            if self.on_turn_end and transcript:
                await self.on_turn_end(transcript)

        except asyncio.CancelledError:
            log.debug("end_of_turn_timer_cancelled", call_id=self.call_id)

    def reset(self):
        """Reset state between turns."""
        if self._silence_timer and not self._silence_timer.done():
            self._silence_timer.cancel()
        self._in_utterance = False
        self._current_transcript = ""
        self._silence_timer = None
