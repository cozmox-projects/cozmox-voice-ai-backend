"""
services/metrics/mos_calculator.py
────────────────────────────────────
MOS (Mean Opinion Score) calculator for voice call quality.

MOS is the standard measure of call audio quality, rated 1-5:
  5.0  = Excellent (indistinguishable from face-to-face)
  4.0  = Good (noticeable but not annoying)
  3.0  = Fair (slightly annoying)
  2.0  = Poor (annoying)
  1.0  = Bad (very annoying, barely usable)

Target: > 3.5 MOS for acceptable voice AI calls.

Formula used: E-model (ITU-T G.107) simplified version.
Inputs: packet loss %, jitter (ms), latency (ms).
"""
import math
from dataclasses import dataclass
from logger import get_logger
from services.metrics.collector import mos_score, packet_loss_percent, jitter_ms

log = get_logger(__name__)


@dataclass
class CallQualityMetrics:
    call_id: str
    packet_loss_pct: float   # 0.0 to 100.0
    jitter_ms_val: float     # milliseconds
    rtt_ms: float            # round-trip time in ms
    mos: float = 0.0

    def is_acceptable(self) -> bool:
        return self.mos >= 3.5

    def quality_label(self) -> str:
        if self.mos >= 4.3:
            return "Excellent"
        elif self.mos >= 4.0:
            return "Good"
        elif self.mos >= 3.5:
            return "Fair"
        elif self.mos >= 2.5:
            return "Poor"
        else:
            return "Bad"


def calculate_mos(
    call_id: str,
    packet_loss_pct: float,
    jitter_ms_val: float,
    rtt_ms: float = 150.0,
) -> CallQualityMetrics:
    """
    Estimates MOS score using simplified E-model.

    Args:
        call_id:         Unique call identifier (for metrics labeling)
        packet_loss_pct: Packet loss percentage (e.g., 2.5 for 2.5%)
        jitter_ms_val:   Jitter in milliseconds
        rtt_ms:          Round-trip time in milliseconds (default 150ms)

    Returns:
        CallQualityMetrics with .mos populated
    """
    # ── E-model R-factor calculation ──────────────────────────────────────────

    # Base R-factor (perfect conditions = 93.2 for G.711)
    R0 = 93.2

    # Impairment from codec (G.711 μ-law = 0, we're using telephone quality)
    Is = 1.41

    # Delay impairment
    # One-way delay = RTT / 2
    one_way_ms = rtt_ms / 2.0
    if one_way_ms < 150:
        Id = 0.024 * one_way_ms + 0.11 * (one_way_ms - 177.3) * (one_way_ms > 177.3)
    else:
        Id = 0.024 * one_way_ms + 0.11 * (one_way_ms - 177.3)

    Id = max(0, Id)

    # Effective packet loss (jitter buffer can recover some loss)
    # Jitter buffer depth assumed ~2x jitter
    jitter_buffer_ms = min(jitter_ms_val * 2, 150)
    effective_loss_pct = max(
        0,
        packet_loss_pct * (1 - math.exp(-jitter_buffer_ms / 16.1))
    )

    # Packet loss impairment
    if effective_loss_pct == 0:
        Ie = 0
    elif effective_loss_pct < 1:
        Ie = effective_loss_pct * 2.5
    elif effective_loss_pct < 5:
        Ie = 2.5 + (effective_loss_pct - 1) * 4.0
    elif effective_loss_pct < 10:
        Ie = 18.5 + (effective_loss_pct - 5) * 2.5
    else:
        Ie = min(31.0, 31.0 * effective_loss_pct / 10.0)

    # Jitter impairment (separate from packet loss)
    if jitter_ms_val > 50:
        Ie += (jitter_ms_val - 50) * 0.1

    # R-factor
    R = R0 - Is - Id - Ie
    R = max(0, min(100, R))

    # Convert R-factor to MOS (ITU-T formula)
    if R < 0:
        mos_val = 1.0
    elif R > 100:
        mos_val = 4.5
    else:
        mos_val = 1 + 0.035 * R + R * (R - 60) * (100 - R) * 7e-6

    mos_val = round(max(1.0, min(4.5, mos_val)), 2)

    metrics = CallQualityMetrics(
        call_id=call_id,
        packet_loss_pct=packet_loss_pct,
        jitter_ms_val=jitter_ms_val,
        rtt_ms=rtt_ms,
        mos=mos_val,
    )

    # ── Record to Prometheus ──────────────────────────────────────────────────
    mos_score.labels(call_id=call_id).set(mos_val)
    packet_loss_percent.labels(call_id=call_id).set(packet_loss_pct)
    jitter_ms.labels(call_id=call_id).set(jitter_ms_val)

    log.info(
        "call_quality_metrics",
        call_id=call_id,
        mos=mos_val,
        quality=metrics.quality_label(),
        packet_loss_pct=packet_loss_pct,
        jitter_ms=jitter_ms_val,
        rtt_ms=rtt_ms,
        acceptable=metrics.is_acceptable(),
    )

    return metrics
