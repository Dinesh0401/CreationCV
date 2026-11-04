"""In-memory job/task queue for ReelMotion AI.

Manages rendering jobs with state tracking, progress,
and cancellation support. Foundation for Phase 5+ render queue.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class JobState(str, Enum):
    """Job lifecycle states."""

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Types of jobs the system can process."""

    RENDER = "render"
    EXPORT = "export"
    MODEL_DOWNLOAD = "model_download"


class Job(BaseModel):
    """A single job in the queue."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    type: JobType = JobType.RENDER
    state: JobState = JobState.QUEUED
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    message: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    started_at: str | None = None
    completed_at: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_terminal(self) -> bool:
        """Whether this job is in a terminal state."""
        return self.state in (
            JobState.COMPLETED,
            JobState.FAILED,
            JobState.CANCELLED,
        )


class JobSummary(BaseModel):
    """Lightweight job summary for listing."""

    id: str
    type: JobType
    state: JobState
    progress: float
    created_at: str


class JobManager:
    """In-memory job queue manager.

    Thread-safe for single-threaded async usage (FastAPI).
    For multi-worker setups, this would need to be backed by
    Redis or a database.
    """

    def __init__(self, max_history: int = 100) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_history = max_history

    def create_job(
        self,
        job_type: JobType = JobType.RENDER,
        metadata: dict[str, Any] | None = None,
    ) -> Job:
        """Create and enqueue a new job.

        Args:
            job_type: Type of job.
            metadata: Optional metadata for the job.

        Returns:
            The created Job.
        """
        job = Job(type=job_type, metadata=metadata or {})
        self._jobs[job.id] = job
        logger.info("Job created: %s (type=%s)", job.id, job_type.value)
        self._prune_history()
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(
        self,
        state: JobState | None = None,
        job_type: JobType | None = None,
    ) -> list[Job]:
        """List jobs, optionally filtered by state and/or type.

        Args:
            state: Filter by job state.
            job_type: Filter by job type.

        Returns:
            List of matching jobs, newest first.
        """
        jobs = list(self._jobs.values())

        if state is not None:
            jobs = [j for j in jobs if j.state == state]
        if job_type is not None:
            jobs = [j for j in jobs if j.type == job_type]

        # Sort by creation time, newest first
        jobs.sort(key=lambda j: j.created_at, reverse=True)
        return jobs

    def start_job(self, job_id: str) -> Job | None:
        """Transition a job to RUNNING state.

        Returns:
            The updated Job, or None if not found / invalid transition.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.state != JobState.QUEUED:
            logger.warning(
                "Cannot start job %s — state is %s", job_id, job.state.value
            )
            return None

        job.state = JobState.RUNNING
        job.started_at = datetime.now(timezone.utc).isoformat()
        logger.info("Job started: %s", job_id)
        return job

    def update_progress(
        self, job_id: str, progress: float, message: str = ""
    ) -> Job | None:
        """Update job progress.

        Args:
            job_id: Job ID.
            progress: Progress percentage (0-100).
            message: Optional status message.

        Returns:
            The updated Job, or None if not found.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.is_terminal:
            return None

        job.progress = min(max(progress, 0.0), 100.0)
        if message:
            job.message = message
        return job

    def complete_job(self, job_id: str, message: str = "") -> Job | None:
        """Mark a job as completed.

        Returns:
            The updated Job, or None if not found / invalid transition.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.is_terminal:
            return None

        job.state = JobState.COMPLETED
        job.progress = 100.0
        job.completed_at = datetime.now(timezone.utc).isoformat()
        if message:
            job.message = message
        logger.info("Job completed: %s", job_id)
        return job

    def fail_job(self, job_id: str, error: str) -> Job | None:
        """Mark a job as failed.

        Args:
            job_id: Job ID.
            error: Error message.

        Returns:
            The updated Job, or None if not found / invalid transition.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.is_terminal:
            return None

        job.state = JobState.FAILED
        job.error = error
        job.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error("Job failed: %s — %s", job_id, error)
        return job

    def cancel_job(self, job_id: str) -> Job | None:
        """Cancel a job.

        Only queued or running jobs can be cancelled.

        Returns:
            The updated Job, or None if not found / invalid transition.
        """
        job = self._jobs.get(job_id)
        if job is None:
            return None
        if job.is_terminal:
            logger.warning(
                "Cannot cancel job %s — already in terminal state %s",
                job_id,
                job.state.value,
            )
            return None

        job.state = JobState.CANCELLED
        job.completed_at = datetime.now(timezone.utc).isoformat()
        logger.info("Job cancelled: %s", job_id)
        return job

    def get_queue_stats(self) -> dict[str, int]:
        """Get queue statistics by state."""
        stats: dict[str, int] = {s.value: 0 for s in JobState}
        for job in self._jobs.values():
            stats[job.state.value] += 1
        return stats

    def _prune_history(self) -> None:
        """Remove oldest completed/failed/cancelled jobs if over limit."""
        terminal = [
            j
            for j in self._jobs.values()
            if j.is_terminal
        ]
        if len(terminal) <= self._max_history:
            return

        terminal.sort(key=lambda j: j.completed_at or j.created_at)
        to_remove = len(terminal) - self._max_history
        for job in terminal[:to_remove]:
            del self._jobs[job.id]
