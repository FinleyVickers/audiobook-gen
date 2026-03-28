import asyncio
import json
import uuid
from dataclasses import dataclass, field
from typing import AsyncGenerator


@dataclass
class Job:
    id: str
    status: str = "pending"  # pending | processing | complete | error
    total_chunks: int = 0
    completed_chunks: int = 0
    current_chapter: str = ""
    current_chapter_index: int = 0
    total_chapters: int = 0
    output_path: str = ""
    book_title: str = ""
    error: str = ""


jobs: dict[str, Job] = {}
_queues: dict[str, asyncio.Queue] = {}


def create_job(total_chunks: int, total_chapters: int, book_title: str = "") -> Job:
    job_id = str(uuid.uuid4())
    job = Job(
        id=job_id,
        total_chunks=total_chunks,
        total_chapters=total_chapters,
        book_title=book_title,
    )
    jobs[job_id] = job
    _queues[job_id] = asyncio.Queue()
    return job


def get_job(job_id: str) -> Job | None:
    return jobs.get(job_id)


def update_progress(job_id: str, **kwargs) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    for k, v in kwargs.items():
        setattr(job, k, v)
    _push_event(job_id, _job_event(job))


def increment_chunks(job_id: str) -> None:
    job = jobs.get(job_id)
    if not job:
        return
    job.completed_chunks += 1
    _push_event(job_id, _job_event(job))


def _push_event(job_id: str, data: dict) -> None:
    q = _queues.get(job_id)
    if q:
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def _job_event(job: Job) -> dict:
    return {
        "status": job.status,
        "total_chunks": job.total_chunks,
        "completed_chunks": job.completed_chunks,
        "current_chapter": job.current_chapter,
        "current_chapter_index": job.current_chapter_index,
        "total_chapters": job.total_chapters,
        "error": job.error,
    }


async def progress_stream(job_id: str) -> AsyncGenerator[str, None]:
    job = jobs.get(job_id)
    if not job:
        yield f"data: {json.dumps({'error': 'Job not found'})}\n\n"
        return

    q = _queues.get(job_id)
    if not q:
        yield f"data: {json.dumps({'error': 'No queue for job'})}\n\n"
        return

    # Send current state immediately
    yield f"data: {json.dumps(_job_event(job))}\n\n"

    while True:
        try:
            event = await asyncio.wait_for(q.get(), timeout=30.0)
            yield f"data: {json.dumps(event)}\n\n"
            if event.get("status") in ("complete", "error"):
                break
        except asyncio.TimeoutError:
            # Keepalive ping
            yield ": ping\n\n"

        # Re-check if job is done (in case we missed the event)
        current = jobs.get(job_id)
        if current and current.status in ("complete", "error"):
            yield f"data: {json.dumps(_job_event(current))}\n\n"
            break
