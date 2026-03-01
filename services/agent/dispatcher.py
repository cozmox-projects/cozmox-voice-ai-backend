"""
services/agent/dispatcher.py
──────────────────────────────
Worker Pool Dispatcher — manages a pool of agent workers for concurrent calls.

This is the key component for handling 100 concurrent calls.

Architecture:
  - Maintains a semaphore limiting concurrent workers to MAX_WORKERS
  - Each incoming call acquires a slot from the semaphore
  - Spawns an AgentWorker coroutine for that slot
  - Releases the slot when the call ends
  - Integrates with HealthChecker for automatic restart on crash

Think of it like a hotel: 100 rooms (semaphore), each guest (call) gets
one room. When the hotel is full, new guests wait or are turned away.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime

from config import get_settings
from logger import get_logger
from services.metrics.collector import workers_active, calls_failed_setup, worker_restarts_total
from services.resilience.health_checker import get_health_checker

log = get_logger(__name__)
settings = get_settings()


@dataclass
class WorkerSlot:
    call_id: str
    room_name: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    task: Optional[asyncio.Task] = None
    status: str = "starting"   # starting | running | completed | failed


class WorkerPoolDispatcher:
    """
    Central dispatcher that manages all agent worker lifecycles.

    Usage:
        dispatcher = WorkerPoolDispatcher(max_workers=100)
        await dispatcher.start()                          # start health checker
        await dispatcher.dispatch(call_id, room_name)    # spawn worker for call
        await dispatcher.stop()                           # graceful shutdown
    """

    def __init__(self, max_workers: int = None):
        self._max_workers = max_workers or settings.worker_pool_size
        # Semaphore: gates how many workers run concurrently
        self._semaphore = asyncio.Semaphore(self._max_workers)
        self._active_slots: Dict[str, WorkerSlot] = {}
        self._health_checker = get_health_checker()
        self._running = False
        log.info("dispatcher_initialized", max_workers=self._max_workers)

    async def start(self):
        """Start the dispatcher and health checker background task."""
        self._running = True
        asyncio.create_task(self._health_checker.start())
        log.info("dispatcher_started", max_workers=self._max_workers)

    async def stop(self):
        """Gracefully cancel all running workers."""
        self._running = False
        await self._health_checker.stop()

        cancel_tasks = []
        for slot in self._active_slots.values():
            if slot.task and not slot.task.done():
                slot.task.cancel()
                cancel_tasks.append(slot.task)

        if cancel_tasks:
            await asyncio.gather(*cancel_tasks, return_exceptions=True)

        log.info("dispatcher_stopped", cancelled_workers=len(cancel_tasks))

    def available_slots(self) -> int:
        """How many more calls we can accept right now."""
        return self._max_workers - len(self._active_slots)

    def is_at_capacity(self) -> bool:
        return len(self._active_slots) >= self._max_workers

    def get_active_calls(self) -> list:
        return [
            {
                "call_id": s.call_id,
                "room_name": s.room_name,
                "started_at": s.started_at.isoformat(),
                "status": s.status,
            }
            for s in self._active_slots.values()
        ]

    async def dispatch(self, call_id: str, room_name: str) -> bool:
        """
        Dispatches an agent worker for the given call.

        Returns True if successfully dispatched, False if at capacity.
        Non-blocking — the worker runs in the background.
        """
        if self.is_at_capacity():
            log.warning(
                "dispatcher_at_capacity",
                call_id=call_id,
                active=len(self._active_slots),
                max=self._max_workers,
            )
            return False

        slot = WorkerSlot(call_id=call_id, room_name=room_name)
        self._active_slots[call_id] = slot

        # Create the worker task
        task = asyncio.create_task(
            self._run_worker_with_slot(call_id, room_name, slot)
        )
        slot.task = task

        # Register with health checker for auto-restart
        self._health_checker.register(call_id, room_name, task)

        log.info(
            "worker_dispatched",
            call_id=call_id,
            room=room_name,
            active_workers=len(self._active_slots),
            available_slots=self.available_slots(),
        )
        return True

    async def _run_worker_with_slot(self, call_id: str, room_name: str, slot: WorkerSlot):
        """
        Wraps AgentWorker.run() with semaphore + slot lifecycle management.
        """
        async with self._semaphore:
            slot.status = "running"
            workers_active.inc()

            try:
                from services.agent.worker import AgentWorker
                worker = AgentWorker(call_id=call_id, room_name=room_name)
                await worker.run()
                slot.status = "completed"
                log.info("worker_completed_normally", call_id=call_id)

            except asyncio.CancelledError:
                slot.status = "cancelled"
                log.info("worker_cancelled", call_id=call_id)

            except Exception as e:
                slot.status = "failed"
                calls_failed_setup.inc()
                log.error("worker_failed", call_id=call_id, error=str(e), error_type=type(e).__name__)
                raise

            finally:
                workers_active.dec()
                self._health_checker.unregister(call_id)
                self._active_slots.pop(call_id, None)
                log.debug("worker_slot_released", call_id=call_id, remaining=len(self._active_slots))


# ── Singleton ─────────────────────────────────────────────────────────────────

_dispatcher: Optional[WorkerPoolDispatcher] = None


def get_dispatcher() -> WorkerPoolDispatcher:
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = WorkerPoolDispatcher()
    return _dispatcher
