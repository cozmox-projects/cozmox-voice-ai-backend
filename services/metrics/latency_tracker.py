"""
services/metrics/latency_tracker.py
─────────────────────────────────────
Tracks per-call latency at each pipeline stage.

Usage:
    tracker = LatencyTracker(call_id="call-abc123")
    tracker.speech_ended()          # caller stops talking
    tracker.stt_completed("hello")  # deepgram returned transcript
    tracker.llm_first_token()       # first token from Azure OpenAI
    tracker.tts_first_audio()       # first audio chunk from ElevenLabs
    tracker.report()                # logs + records to Prometheus
"""
import time
from dataclasses import dataclass, field
from typing import Optional
from logger import get_logger
from services.metrics.collector import (
    e2e_latency_seconds,
    stt_latency_seconds,
    llm_latency_seconds,
    tts_latency_seconds,
)

log = get_logger(__name__)


@dataclass
class LatencyTracker:
    call_id: str

    # Timestamps (set via methods below)
    _speech_end_ts: Optional[float] = field(default=None, repr=False)
    _stt_done_ts: Optional[float] = field(default=None, repr=False)
    _llm_first_token_ts: Optional[float] = field(default=None, repr=False)
    _tts_first_audio_ts: Optional[float] = field(default=None, repr=False)

    # ── Event recording ──────────────────────────────────────────────────

    def speech_ended(self):
        """Call this when VAD detects end-of-speech."""
        self._speech_end_ts = time.perf_counter()

    def stt_completed(self, transcript: str = ""):
        """Call this when Deepgram returns the final transcript."""
        self._stt_done_ts = time.perf_counter()
        if self._speech_end_ts:
            duration = self._stt_done_ts - self._speech_end_ts
            stt_latency_seconds.observe(duration)
            log.debug(
                "stt_latency",
                call_id=self.call_id,
                ms=round(duration * 1000, 1),
                transcript_preview=transcript[:50],
            )

    def llm_first_token(self):
        """Call this when Azure OpenAI yields the first token."""
        self._llm_first_token_ts = time.perf_counter()
        if self._stt_done_ts:
            duration = self._llm_first_token_ts - self._stt_done_ts
            llm_latency_seconds.observe(duration)
            log.debug(
                "llm_first_token_latency",
                call_id=self.call_id,
                ms=round(duration * 1000, 1),
            )

    def tts_first_audio(self):
        """Call this when ElevenLabs returns the first audio chunk."""
        self._tts_first_audio_ts = time.perf_counter()
        if self._llm_first_token_ts:
            duration = self._tts_first_audio_ts - self._llm_first_token_ts
            tts_latency_seconds.observe(duration)
            log.debug(
                "tts_first_audio_latency",
                call_id=self.call_id,
                ms=round(duration * 1000, 1),
            )

    # ── E2E calculation ───────────────────────────────────────────────────

    def report(self) -> Optional[float]:
        """
        Records total e2e latency to Prometheus.
        Returns total ms or None if incomplete.
        """
        if not (self._speech_end_ts and self._tts_first_audio_ts):
            log.warning("latency_report_incomplete", call_id=self.call_id)
            return None

        total = self._tts_first_audio_ts - self._speech_end_ts
        e2e_latency_seconds.observe(total)

        status = "✅ PASS" if total < 0.6 else "⚠️  OVER TARGET"

        log.info(
            "e2e_latency_report",
            call_id=self.call_id,
            total_ms=round(total * 1000, 1),
            stt_ms=self._fmt(self._speech_end_ts, self._stt_done_ts),
            llm_ms=self._fmt(self._stt_done_ts, self._llm_first_token_ts),
            tts_ms=self._fmt(self._llm_first_token_ts, self._tts_first_audio_ts),
            target_600ms=status,
        )
        return total

    def _fmt(self, start: Optional[float], end: Optional[float]) -> str:
        if start and end:
            return f"{round((end - start) * 1000, 1)}ms"
        return "N/A"
