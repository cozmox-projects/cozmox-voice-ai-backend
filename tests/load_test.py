"""
tests/load_test.py
───────────────────
Concurrent call load test — proves the system handles N parallel calls.

What this does:
  1. Fires N simulate-call requests to the webhook service in parallel
  2. Connects N simulated LiveKit "callers" concurrently
  3. Waits for each agent to respond (greeting audio = first response)
  4. Reports aggregate stats: connection rate, response times, errors

This bypasses Twilio but exercises the full AI pipeline.

Usage:
  python tests/load_test.py --calls 10               # quick test
  python tests/load_test.py --calls 100 --hold 30    # full 100-call test
  python tests/load_test.py --calls 5 --verbose      # with detailed output
"""
import asyncio
import sys
import os
import argparse
import time
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import httpx
from livekit import rtc
from livekit.api import AccessToken, VideoGrants
from config import get_settings
from logger import get_logger

log = get_logger("load_test")
settings = get_settings()

WEBHOOK_BASE = f"http://localhost:{settings.webhook_port}"


@dataclass
class CallResult:
    call_id: str
    room_name: str
    connected: bool = False
    agent_responded: bool = False
    error: Optional[str] = None

    # Timings (ms)
    simulate_api_ms: float = 0.0
    livekit_connect_ms: float = 0.0
    agent_response_ms: float = 0.0     # time from connection to first agent audio


async def simulate_and_join(call_index: int, hold_seconds: int, verbose: bool) -> CallResult:
    """
    Full lifecycle for one simulated call:
      1. POST /calls/simulate  → spawns agent worker
      2. Connect LiveKit room  → as the "caller"
      3. Wait for agent greeting audio
      4. Hold for hold_seconds
      5. Disconnect
    """
    room_name = f"load-{call_index:04d}-{int(time.time())}"
    result    = CallResult(call_id=f"caller-{call_index:04d}", room_name=room_name)
    room      = rtc.Room()

    # ── Step 1: trigger agent via simulate endpoint ───────────────────────
    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                f"{WEBHOOK_BASE}/calls/simulate",
                params={"room_name": room_name},
            )
            resp.raise_for_status()
            data = resp.json()

        result.simulate_api_ms = (time.perf_counter() - t0) * 1000
        caller_token = data["caller_token"]

        if verbose:
            print(f"  [{call_index:03d}] simulate API → {result.simulate_api_ms:.0f}ms")

    except Exception as e:
        result.error = f"simulate_api_failed: {e}"
        log.error("simulate_api_error", call_index=call_index, error=str(e))
        return result

    # ── Step 2: connect to LiveKit room as caller ─────────────────────────
    t1 = time.perf_counter()
    try:
        await room.connect(settings.livekit_url, caller_token)
        result.connected = True
        result.livekit_connect_ms = (time.perf_counter() - t1) * 1000

        if verbose:
            print(f"  [{call_index:03d}] livekit connect → {result.livekit_connect_ms:.0f}ms")

    except Exception as e:
        result.error = f"livekit_connect_failed: {e}"
        return result

    # ── Step 3: wait for agent audio track (= agent has started responding) ─
    t2 = time.perf_counter()
    agent_audio_event = asyncio.Event()

    @room.on("track_subscribed")
    def on_track(track, publication, participant):
        if isinstance(track, rtc.RemoteAudioTrack):
            result.agent_response_ms = (time.perf_counter() - t2) * 1000
            result.agent_responded = True
            agent_audio_event.set()

    try:
        await asyncio.wait_for(agent_audio_event.wait(), timeout=15.0)
        if verbose:
            print(f"  [{call_index:03d}] agent responded → {result.agent_response_ms:.0f}ms ✅")
    except asyncio.TimeoutError:
        result.error = "agent_no_response_timeout"
        if verbose:
            print(f"  [{call_index:03d}] agent TIMEOUT ❌")

    # ── Step 4: hold the call ─────────────────────────────────────────────
    if result.connected:
        await asyncio.sleep(hold_seconds)

    # ── Step 5: disconnect ────────────────────────────────────────────────
    await room.disconnect()
    return result


async def run_load_test(n_calls: int, hold_seconds: int, verbose: bool):
    print(f"\n{'═'*65}")
    print(f"  🔥 Voice AI Agent — Concurrent Call Load Test")
    print(f"{'═'*65}")
    print(f"  Calls:           {n_calls}")
    print(f"  Hold per call:   {hold_seconds}s")
    print(f"  Webhook:         {WEBHOOK_BASE}")
    print(f"  LiveKit:         {settings.livekit_url}")
    print(f"{'─'*65}\n")

    # Verify webhook is up
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            health = await client.get(f"{WEBHOOK_BASE}/health")
            health_data = health.json()
            print(f"  Webhook health:  {health_data.get('status')} "
                  f"(slots available: {health_data.get('available_slots')})\n")
    except Exception as e:
        print(f"  ❌ Webhook not reachable: {e}")
        print(f"     Start it with:  python -m services.webhook.main\n")
        return

    # Fire all calls concurrently
    print(f"  Launching {n_calls} concurrent calls...\n")
    t_start = time.perf_counter()

    tasks = [
        simulate_and_join(i, hold_seconds, verbose)
        for i in range(n_calls)
    ]
    results: List[CallResult] = await asyncio.gather(*tasks, return_exceptions=False)

    total_elapsed = time.perf_counter() - t_start

    # ── Aggregate stats ───────────────────────────────────────────────────
    connected       = [r for r in results if r.connected]
    responded       = [r for r in results if r.agent_responded]
    failed          = [r for r in results if r.error]

    response_times  = [r.agent_response_ms for r in responded]
    connect_times   = [r.livekit_connect_ms for r in connected]

    def pct(n, d):
        return f"{n/d*100:.1f}%" if d else "0%"

    print(f"\n{'═'*65}")
    print(f"  📊 Results")
    print(f"{'─'*65}")
    print(f"  Total calls fired:        {n_calls}")
    print(f"  LiveKit connected:        {len(connected):>4}  ({pct(len(connected), n_calls)})")
    print(f"  Agent responded:          {len(responded):>4}  ({pct(len(responded), n_calls)})")
    print(f"  Failed:                   {len(failed):>4}  ({pct(len(failed), n_calls)})")
    print(f"  Total test time:          {total_elapsed:.1f}s")

    if connect_times:
        print(f"\n  LiveKit Connect Time:")
        print(f"    avg:  {statistics.mean(connect_times):.0f}ms")
        print(f"    p95:  {sorted(connect_times)[int(len(connect_times)*0.95)]:.0f}ms")

    if response_times:
        p95 = sorted(response_times)[int(len(response_times)*0.95)]
        avg = statistics.mean(response_times)
        print(f"\n  Agent First Response Time (connect → first audio):")
        print(f"    avg:  {avg:.0f}ms")
        print(f"    p50:  {statistics.median(response_times):.0f}ms")
        print(f"    p95:  {p95:.0f}ms")
        print(f"    min:  {min(response_times):.0f}ms")
        print(f"    max:  {max(response_times):.0f}ms")

    if failed:
        print(f"\n  Errors:")
        from collections import Counter
        error_counts = Counter(r.error for r in failed)
        for err, count in error_counts.most_common(5):
            print(f"    {count}x  {err}")

    print(f"\n  📈 Check Grafana for E2E latency breakdown:")
    print(f"     http://<your-ip>/grafana  (or http://localhost:3000)")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load test the Voice AI Agent")
    parser.add_argument("--calls",   type=int, default=10,  help="Number of concurrent calls (default: 10)")
    parser.add_argument("--hold",    type=int, default=15,  help="Seconds to hold each call (default: 15)")
    parser.add_argument("--verbose", action="store_true",   help="Show per-call timing")
    args = parser.parse_args()

    asyncio.run(run_load_test(
        n_calls=args.calls,
        hold_seconds=args.hold,
        verbose=args.verbose,
    ))
