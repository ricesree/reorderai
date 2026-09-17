"""Single-worker background queue for async invoice PDF parsing.

Mirrors api/services/enrich_queue.py: one daemon thread processes jobs
strictly one after another in this process. In-memory job store — only
correct for a single worker process; move to a DB table if this ever runs
with --workers > 1 or multiple replicas.

Wraps invoice_agent's own parse_invoice_pdf/save_result unmodified — same
OpenAI calls, same results/ persistence, just off the request thread.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any

from openai import OpenAI

from invoice_agent.main import parse_invoice_pdf, save_result

logger = logging.getLogger(__name__)

_QUEUE: queue.Queue = queue.Queue()
_JOBS: dict[str, dict[str, Any]] = {}


def _worker() -> None:
    while True:
        job_id, tenant_id, filename, pdf_bytes = _QUEUE.get()
        _JOBS[job_id] = {"status": "running", "tenant_id": tenant_id}
        try:
            client = OpenAI(timeout=120)  # reads OPENAI_API_KEY from env
            result = parse_invoice_pdf(client, pdf_bytes)
            result["tenant_id"] = tenant_id
            record = save_result(filename, result)
            _JOBS[job_id] = {"status": "success", **result, "_id": record["id"]}
        except Exception as exc:  # noqa: BLE001 - job failure must not kill the worker
            logger.exception("invoice parse job %s failed", job_id)
            _JOBS[job_id] = {"status": "failed", "tenant_id": tenant_id, "error": str(exc)}
        finally:
            _QUEUE.task_done()


threading.Thread(target=_worker, daemon=True).start()


def enqueue(tenant_id: str | None, filename: str, pdf_bytes: bytes) -> str:
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"status": "queued", "tenant_id": tenant_id}
    _QUEUE.put((job_id, tenant_id, filename, pdf_bytes))
    return job_id


def get_status(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)
