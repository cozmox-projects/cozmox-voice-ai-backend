"""
services/resilience/health_checker.py
───────────────────────────────────────
Worker health monitor.

Runs as a background task alongside the webhook service.
Every 10 seconds, checks if all registered workers are still alive.
If a worker process has died, it restarts it.

This is the "reconnect dropped agent" resilience mechanism required
by the assessment.
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional
from logger import get_logger
from services.metrics.collector import worker_restarts_total, workers_active

log = get_logger(__name__)

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
    Tracks all running agent workers and restarts them if they crash.
    
    Usage:
        checker = WorkerHealthChecker()
        checker.register(call_id, room_name, task)
        await checker.start()  # runs forever
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
        dead_workers = []

        for call_id, record in self._workers.items():
            if record.task.done():
                exception = record.task.exception() if not record.task.cancelled() else None
                if exception:
                    log.warning(
                        "worker_died",
                        call_id=call_id,
                        error=str(exception),
                        restart_count=record.restart_count,
                    )
                    dead_workers.append(record)
                else:
                    # Task completed normally — call ended
                    dead_workers.append(record)

        for record in dead_workers:
            await self._handle_dead_worker(record)

    async def _handle_dead_worker(self, record: WorkerRecord):
        """
        Decides whether to restart a dead worker or give up.
        Restarts up to max_restarts times.
        """
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
            attempt=record.restart_count,
            max=record.max_restarts,
        )

        # Restart the worker
        from services.agent.worker import AgentWorker
        new_worker = AgentWorker(call_id=record.call_id, room_name=record.room_name)
        new_task = asyncio.create_task(new_worker.run())
        record.task = new_task

        log.info("worker_restarted", call_id=record.call_id)


# Singleton
_health_checker: Optional[WorkerHealthChecker] = None


def get_health_checker() -> WorkerHealthChecker:
    global _health_checker
    if _health_checker is None:
        _health_checker = WorkerHealthChecker()
    return _health_checker
