"""
services/agent/turn_detector.py
─────────────────────────────────
Turn Detector — decides when the caller has FINISHED speaking.

THE PROBLEM WITH A FIXED 400ms TIMER
──────────────────────────────────────
A naive "400ms silence = done" approach fails in real phone calls:

  Caller: "I want to... uh... cancel my subscription."
           ────────────^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           300ms pause here — fixed timer would fire mid-sentence!

  Caller: "Yes." [long pause while thinking]
           ──────^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           800ms pause but they HAVE finished — fixed timer would be too slow.

OUR STRATEGY: Hybrid ML + Adaptive Silence Timer
──────────────────────────────────────────────────
Layer 1 — Lightweight heuristic end-of-turn classifier
  Analyses the partial transcript text to determine if it LOOKS semantically
  complete (ends with "?" / "." / complete thought vs trailing "uh", "and", etc.)
  This runs on every transcript update — zero latency, no external API call.

Layer 2 — Adaptive silence timer
  Instead of a fixed 400ms, the silence window adapts based on:
    - Whether the transcript looks complete (Layer 1) → shorter window (300ms)
    - Whether transcript ends mid-thought / with filler → longer window (600ms)
    - Whether caller has been speaking a long time → shorter (they've said a lot)
    - Short utterances ("yes" / "no") → shortest window (250ms)
  This prevents cutting people off while staying responsive.

Layer 3 — Safety maximum
  Even if the silence timer never fires naturally, we declare turn-end
  after MAX_UTTERANCE_SECONDS (8s) to prevent a stuck state.

RESULT
──────
  Handles mid-sentence pauses ("I want to... uh... cancel")
  Handles fast complete answers ("Yes.", "No.", "Cancel it.")
  Handles long rambling utterances without cutting off
  Adapts to the caller's speaking style over the call
"""

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from logger import get_logger

log = get_logger(__name__)

# ── Timing constants ──────────────────────────────────────────────────────────

SILENCE_SHORT_MS = 250  # very short utterance completed ("yes", "no")
SILENCE_NORMAL_MS = 380  # typical complete sentence
SILENCE_EXTENDED_MS = 600  # incomplete / trailing thought detected
SILENCE_MAX_MS = 800  # absolute maximum

MAX_UTTERANCE_SECONDS = 8.0
SHORT_UTTERANCE_WORD_THRESHOLD = 3

# ── Completion signals (text-based heuristics) ────────────────────────────────

_COMPLETE_ENDINGS = re.compile(
    r"""
    (?:
        [.!?]\s*$              |
        \b(?:
            yes|no|okay|ok|sure|
            thanks|thank\s+you|
            bye|goodbye|
            please|correct|right|
            cancel|refund|help
        )\s*[.!?]?\s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_INCOMPLETE_SIGNALS = re.compile(
    r"""
    (?:
        \b(?:
            and|but|so|because|
            uh|um|er|like|
            actually|basically|
            i\s+mean|you\s+know|
            i\s+want\s+to|i\s+need\s+to|
            can\s+you|could\s+you|
            i\s+was\s+going\s+to
        )\s*$
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

_SHORT_COMPLETE = re.compile(
    r"^(?:yes|no|okay|ok|sure|cancel|refund|hello|hi|help)[.!?]?\s*$",
    re.IGNORECASE,
)


# ── Completeness classifier ───────────────────────────────────────────────────


def classify_transcript(transcript: str) -> tuple:
    """
    Lightweight heuristic classifier — decides how complete a transcript looks.

    Returns:
        (classification, silence_ms)
        classification: "complete" | "incomplete" | "short_complete" | "uncertain"
        silence_ms:     recommended silence window in milliseconds
    """
    t = transcript.strip()
    if not t:
        return "uncertain", SILENCE_NORMAL_MS

    words = t.split()
    word_count = len(words)

    if _SHORT_COMPLETE.match(t):
        return "short_complete", SILENCE_SHORT_MS

    if word_count < SHORT_UTTERANCE_WORD_THRESHOLD:
        return "uncertain", SILENCE_NORMAL_MS

    if _INCOMPLETE_SIGNALS.search(t):
        return "incomplete", SILENCE_EXTENDED_MS

    if _COMPLETE_ENDINGS.search(t):
        return "complete", SILENCE_NORMAL_MS

    if word_count > 15:
        return "complete", SILENCE_NORMAL_MS

    return "uncertain", SILENCE_NORMAL_MS


# ── Turn Detector ─────────────────────────────────────────────────────────────


@dataclass
class TurnDetector:
    """
    Hybrid turn detector combining text-based completion classification
    with an adaptive silence timer.

    One instance per call.
    """

    call_id: str
    on_turn_end: Optional[Callable[[str], Awaitable[None]]] = None

    _in_utterance: bool = field(default=False, init=False)
    _silence_timer: Optional[asyncio.Task] = field(default=None, init=False)
    _max_duration_timer: Optional[asyncio.Task] = field(default=None, init=False)
    _current_transcript: str = field(default="", init=False)
    _utterance_start_ts: float = field(default=0.0, init=False)
    _recent_silence_windows: list = field(default_factory=list, init=False)

    def on_speech_start(self):
        """Called by Deepgram SpeechStarted event."""
        if not self._in_utterance:
            self._in_utterance = True
            self._utterance_start_ts = time.perf_counter()
            log.debug("turn_started", call_id=self.call_id)
            self._max_duration_timer = asyncio.create_task(
                self._max_duration_watchdog()
            )

        # Cancel any pending silence timer — caller resumed speaking
        self._cancel_silence_timer()

    def on_speech_end(self):
        """Called by Deepgram UtteranceEnd event. Starts adaptive countdown."""
        if not self._in_utterance:
            return

        silence_ms = self._compute_silence_window()

        log.debug(
            "silence_detected",
            call_id=self.call_id,
            silence_window_ms=silence_ms,
            transcript_preview=self._current_transcript[:60],
        )

        self._silence_timer = asyncio.create_task(self._silence_countdown(silence_ms))

    def on_transcript_update(self, partial_transcript: str):
        """
        Called by Deepgram on every partial transcript update.
        Shrinks the active silence window if the transcript now looks complete.
        """
        self._current_transcript = partial_transcript

        # If silence timer is already running and transcript now looks clearly
        # complete, shrink the remaining window to respond faster
        if self._silence_timer and not self._silence_timer.done():
            classification, _ = classify_transcript(partial_transcript)
            if classification == "short_complete":
                self._cancel_silence_timer()
                self._silence_timer = asyncio.create_task(
                    self._silence_countdown(SILENCE_SHORT_MS)
                )
                log.debug(
                    "silence_window_shortened",
                    call_id=self.call_id,
                )

    def _compute_silence_window(self) -> int:
        """
        Computes the appropriate silence window for the current transcript.
        """
        transcript = self._current_transcript.strip()
        classification, base_silence_ms = classify_transcript(transcript)

        utterance_duration = time.perf_counter() - self._utterance_start_ts

        # Long utterance — caller has said a lot, likely done
        if utterance_duration > 4.0 and classification != "incomplete":
            silence_ms = SILENCE_SHORT_MS
        else:
            silence_ms = base_silence_ms

        # Adapt based on recent history for this caller (fast vs slow speaker)
        if len(self._recent_silence_windows) >= 3:
            recent_avg = sum(self._recent_silence_windows[-3:]) / 3
            if recent_avg < 350:
                silence_ms = max(SILENCE_SHORT_MS, silence_ms - 50)

        log.debug(
            "silence_window_computed",
            call_id=self.call_id,
            classification=classification,
            silence_ms=silence_ms,
            utterance_duration_ms=round(utterance_duration * 1000),
        )

        return silence_ms

    async def _silence_countdown(self, silence_ms: int):
        """Waits silence_ms. If not cancelled, fires end-of-turn."""
        try:
            await asyncio.sleep(silence_ms / 1000.0)

            transcript = self._current_transcript.strip()
            elapsed_ms = round((time.perf_counter() - self._utterance_start_ts) * 1000)

            self._recent_silence_windows.append(silence_ms)
            if len(self._recent_silence_windows) > 5:
                self._recent_silence_windows.pop(0)

            self._end_utterance(transcript, elapsed_ms, trigger="silence_timer")

        except asyncio.CancelledError:
            log.debug("silence_timer_cancelled", call_id=self.call_id)

    async def _max_duration_watchdog(self):
        """
        Safety: force end-of-turn after MAX_UTTERANCE_SECONDS.
        Prevents stuck state from continuous background noise.
        """
        try:
            await asyncio.sleep(MAX_UTTERANCE_SECONDS)
            if self._in_utterance:
                transcript = self._current_transcript.strip()
                log.warning(
                    "max_utterance_duration_exceeded",
                    call_id=self.call_id,
                    duration_s=MAX_UTTERANCE_SECONDS,
                    transcript_preview=transcript[:80],
                )
                self._cancel_silence_timer()
                self._end_utterance(
                    transcript,
                    MAX_UTTERANCE_SECONDS * 1000,
                    trigger="max_duration",
                )
        except asyncio.CancelledError:
            pass

    def _end_utterance(self, transcript: str, elapsed_ms: float, trigger: str):
        """Finalises the utterance and fires on_turn_end callback."""
        self._in_utterance = False
        self._cancel_max_duration_timer()
        self._silence_timer = None
        self._current_transcript = ""

        log.info(
            "turn_ended",
            call_id=self.call_id,
            transcript=transcript,
            elapsed_ms=round(elapsed_ms),
            trigger=trigger,
        )

        if self.on_turn_end and transcript:
            asyncio.create_task(self.on_turn_end(transcript))

    def _cancel_silence_timer(self):
        if self._silence_timer and not self._silence_timer.done():
            self._silence_timer.cancel()
        self._silence_timer = None

    def _cancel_max_duration_timer(self):
        if self._max_duration_timer and not self._max_duration_timer.done():
            self._max_duration_timer.cancel()
        self._max_duration_timer = None

    def reset(self):
        """Hard reset — call this if the pipeline restarts mid-call."""
        self._cancel_silence_timer()
        self._cancel_max_duration_timer()
        self._in_utterance = False
        self._current_transcript = ""
        self._utterance_start_ts = 0.0
        log.debug("turn_detector_reset", call_id=self.call_id)
