"""
services/resilience/health_checker.py
───────────────────────────────────────
Worker health monitor with pre-restart LiveKit room validation.

WHY THE OLD VERSION WAS WRONG
───────────────────────────────
The old health checker restarted any crashed worker unconditionally.
But consider what happens after a crash:

  1. Call comes in at T=0. Worker starts. Deepgram connection opens.
  2. Worker crashes at T=30s (e.g. network blip to ElevenLabs).
  3. Health checker wakes up at T=40s (10s interval).
  4. OLD: spawns new AgentWorker for the same room_name.
  5. Problem: The Twilio <Stream> WebSocket / bridge may have already
     disconnected (Twilio times out ~60s). The LiveKit room may be
     empty. The caller has hung up. We're spawning a worker into
     an empty room — it will just sit there consuming a semaphore slot
     forever (or until empty_timeout=300s on LiveKit side).

THE FIX
────────
Before restarting a crashed worker, we ask LiveKit:
  "Is anyone still in this room?"
Using the LiveKit Server API (livekit.api.LiveKitAPI.room.list_participants).

If the room is empty (caller has disconnected) → do NOT restart,
just clean up the slot.

If the room still has participants (caller is still there, bridge is live)
→ restart is valid, proceed.

This also covers the "normally completed" case: a worker completes normally
because the caller hung up → room will be empty → health checker skips restart
(which is correct — the old version was also restarting normally-completed workers!)
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional

from livekit.api import LiveKitAPI

from config import get_settings
from logger import get_logger
from services.metrics.collector import worker_restarts_total

log = get_logger(__name__)
settings = get_settings()

CHECK_INTERVAL_SECONDS = 10


@dataclass
class WorkerRecord:
    call_id: str
    room_name: str
    task: asyncio.Task
    restart_count: int = 0
    max_restarts: int = 3


class WorkerHealthChecker:
    """
    Tracks all running agent workers and restarts them if they crash —
    but ONLY if the LiveKit room still has participants (i.e. the caller
    is still on the line).

    Usage:
        checker = WorkerHealthChecker()
        checker.register(call_id, room_name, task)
        await checker.start()  # runs forever in background
    """

    def __init__(self):
        self._workers: Dict[str, WorkerRecord] = {}
        self._running = False

    def register(self, call_id: str, room_name: str, task: asyncio.Task):
        """Register a worker task for health monitoring."""
        self._workers[call_id] = WorkerRecord(
            call_id=call_id,
            room_name=room_name,
            task=task,
        )
        log.debug("worker_registered", call_id=call_id)

    def unregister(self, call_id: str):
        """Remove a worker when the call ends normally."""
        if call_id in self._workers:
            del self._workers[call_id]
            log.debug("worker_unregistered", call_id=call_id)

    async def start(self):
        """Start the health check loop. Runs forever in background."""
        self._running = True
        log.info("health_checker_started", interval_seconds=CHECK_INTERVAL_SECONDS)

        while self._running:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            await self._check_all_workers()

    async def stop(self):
        self._running = False

    async def _check_all_workers(self):
        """Checks all registered workers. Restarts any that have crashed."""
        # Snapshot keys to avoid dict-changed-during-iteration
        records_to_check = list(self._workers.values())

        dead_workers = []
        for record in records_to_check:
            if record.task.done():
                exception = (
                    record.task.exception() if not record.task.cancelled() else None
                )
                if exception:
                    log.warning(
                        "worker_died",
                        call_id=record.call_id,
                        error=str(exception),
                        restart_count=record.restart_count,
                    )
                    dead_workers.append((record, "crashed"))
                else:
                    # Task completed normally — call probably ended cleanly
                    dead_workers.append((record, "completed"))

        for record, reason in dead_workers:
            await self._handle_dead_worker(record, reason)

    async def _room_has_participants(self, room_name: str) -> bool:
        """
        Asks LiveKit Server API if anyone is still in the room.

        Returns True if ≥1 participant is present (caller or bridge),
        False if the room is empty or doesn't exist.

        A False result means the caller has already disconnected — no
        point restarting the agent worker.
        """
        try:
            lk_http_url = settings.livekit_url.replace("ws://", "http://").replace(
                "wss://", "https://"
            )
            async with LiveKitAPI(
                url=lk_http_url,
                api_key=settings.livekit_api_key,
                api_secret=settings.livekit_api_secret,
            ) as lk_api:
                participants = await lk_api.room.list_participants(room=room_name)
                participant_count = len(participants.participants)

                log.debug(
                    "room_participant_check",
                    room=room_name,
                    participant_count=participant_count,
                )
                return participant_count > 0

        except Exception as e:
            # If we can't reach LiveKit, assume room is active (fail-safe:
            # better to attempt a restart than silently drop a live call)
            log.warning(
                "room_participant_check_failed",
                room=room_name,
                error=str(e),
                action="assuming_active_fail_safe",
            )
            return True

    async def _handle_dead_worker(self, record: WorkerRecord, reason: str):
        """
        Decides whether to restart a dead worker:
          1. If completed normally → check if room still has participants.
             If yes, it might have crashed before the caller hung up → restart.
             If no, call ended cleanly → just clean up.
          2. If crashed → same room check, then restart up to max_restarts.
        """
        # First, validate the room still has a caller
        room_active = await self._room_has_participants(record.room_name)

        if not room_active:
            log.info(
                "worker_ended_room_empty_no_restart",
                call_id=record.call_id,
                room=record.room_name,
                reason=reason,
            )
            self.unregister(record.call_id)
            return

        # Room still has participants — caller is still there
        if reason == "completed":
            # Worker completed but caller is still in the room
            # This is unexpected — restart to re-serve the caller
            log.warning(
                "worker_completed_but_caller_still_present",
                call_id=record.call_id,
                room=record.room_name,
            )

        # Check restart cap
        if record.restart_count >= record.max_restarts:
            log.error(
                "worker_max_restarts_exceeded",
                call_id=record.call_id,
                max_restarts=record.max_restarts,
            )
            self.unregister(record.call_id)
            return

        record.restart_count += 1
        worker_restarts_total.inc()

        log.info(
            "restarting_worker",
            call_id=record.call_id,
            room=record.room_name,
            attempt=record.restart_count,
            max=record.max_restarts,
            reason=reason,
        )

        from services.agent.worker import AgentWorker

        new_worker = AgentWorker(call_id=record.call_id, room_name=record.room_name)
        new_task = asyncio.create_task(new_worker.run())
        record.task = new_task

        log.info(
            "worker_restarted",
            call_id=record.call_id,
            room=record.room_name,
        )


# Singleton
_health_checker: Optional[WorkerHealthChecker] = None


def get_health_checker() -> WorkerHealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = WorkerHealthChecker()
    return _health_checker
