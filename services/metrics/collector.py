"""
services/metrics/collector.py
─────────────────────────────
Defines ALL Prometheus metrics for the system.
Import and use these from anywhere in the codebase.

Metrics exposed at /metrics endpoint by the webhook FastAPI app.
"""
from prometheus_client import Counter, Histogram, Gauge, Summary

# ── Call lifecycle counters ───────────────────────────────────────────────────

calls_total = Counter(
    "voice_ai_calls_total",
    "Total number of inbound calls received",
)

calls_failed_setup = Counter(
    "voice_ai_calls_failed_setup_total",
    "Calls that failed during setup (before agent connected)",
)

calls_completed = Counter(
    "voice_ai_calls_completed_total",
    "Calls that completed successfully",
)

# ── Concurrency gauge ─────────────────────────────────────────────────────────

calls_active = Gauge(
    "voice_ai_calls_active",
    "Number of currently active concurrent calls",
)

workers_active = Gauge(
    "voice_ai_workers_active",
    "Number of agent workers currently running",
)

# ── End-to-end latency (the KEY metric — must be <600ms avg) ──────────────────

e2e_latency_seconds = Histogram(
    "voice_ai_e2e_latency_seconds",
    "End-to-end latency: caller stops speaking → caller hears first audio byte",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.5, 2.0],
)

# ── Per-stage latency breakdowns ──────────────────────────────────────────────

stt_latency_seconds = Histogram(
    "voice_ai_stt_latency_seconds",
    "Deepgram STT latency: audio sent → transcript received",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
)

llm_latency_seconds = Histogram(
    "voice_ai_llm_latency_seconds",
    "Azure OpenAI latency: prompt sent → first token received",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0],
)

tts_latency_seconds = Histogram(
    "voice_ai_tts_latency_seconds",
    "ElevenLabs TTS latency: text sent → first audio chunk received",
    buckets=[0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5],
)

# ── Audio quality metrics ─────────────────────────────────────────────────────

mos_score = Gauge(
    "voice_ai_mos_score",
    "Estimated Mean Opinion Score (1-5) for call audio quality",
    ["call_id"],
)

packet_loss_percent = Gauge(
    "voice_ai_packet_loss_percent",
    "RTP packet loss percentage",
    ["call_id"],
)

jitter_ms = Gauge(
    "voice_ai_jitter_ms",
    "RTP jitter in milliseconds",
    ["call_id"],
)

# ── Barge-in counter ──────────────────────────────────────────────────────────

barge_ins_total = Counter(
    "voice_ai_barge_ins_total",
    "Number of times a caller interrupted the AI mid-speech",
)

# ── Knowledge base metrics ────────────────────────────────────────────────────

kb_queries_total = Counter(
    "voice_ai_kb_queries_total",
    "Total knowledge base lookups",
)

kb_hits_total = Counter(
    "voice_ai_kb_hits_total",
    "Knowledge base queries that returned a relevant result",
)

# ── AI API error counters ─────────────────────────────────────────────────────

stt_errors_total = Counter(
    "voice_ai_stt_errors_total",
    "Deepgram STT call failures",
)

llm_errors_total = Counter(
    "voice_ai_llm_errors_total",
    "Azure OpenAI call failures",
)

tts_errors_total = Counter(
    "voice_ai_tts_errors_total",
    "ElevenLabs TTS call failures",
)

# ── Worker health ─────────────────────────────────────────────────────────────

worker_restarts_total = Counter(
    "voice_ai_worker_restarts_total",
    "Number of times a crashed worker was restarted",
)
