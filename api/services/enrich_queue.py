"""Single-worker background queue for /api/products/enrich (no id → batch job).

Runs jobs strictly one after another in a daemon thread inside the same
uvicorn process. In-memory job store — only correct for a single worker
process; move to a DB table if this ever runs with --workers > 1 or
multiple replicas.
"""

from __future__ import annotations

import logging
import queue
import threading
import uuid
from typing import Any

from api.services import product_enrichment_service as service

logger = logging.getLogger(__name__)

_QUEUE: queue.Queue = queue.Queue()
_JOBS: dict[str, dict[str, Any]] = {}
_ACTIVE_TENANTS: dict[str | None, str] = {}  # tenant_id -> job_id currently queued/running
_lock = threading.Lock()


def _worker() -> None:
    while True:
        job_id, tenant_id, limit = _QUEUE.get()
        _JOBS[job_id] = {"status": "running"}
        try:
            result = service.enrich_missing(tenant_id=tenant_id, limit=limit)
            _JOBS[job_id] = {"status": "success", **result}
        except Exception as exc:  # noqa: BLE001 - job failure must not kill the worker
            logger.exception("enrich job %s failed", job_id)
            _JOBS[job_id] = {"status": "failed", "error": str(exc)}
        finally:
            with _lock:
                if _ACTIVE_TENANTS.get(tenant_id) == job_id:
                    del _ACTIVE_TENANTS[tenant_id]
            _QUEUE.task_done()


threading.Thread(target=_worker, daemon=True).start()


def enqueue(tenant_id: str | None, limit: int | None) -> tuple[str, bool]:
    """Returns (job_id, is_new). is_new=False means a job for this tenant is already in flight."""
    with _lock:
        existing = _ACTIVE_TENANTS.get(tenant_id)
        if existing is not None:
            return existing, False
        job_id = uuid.uuid4().hex
        _ACTIVE_TENANTS[tenant_id] = job_id
    _JOBS[job_id] = {"status": "queued"}
    _QUEUE.put((job_id, tenant_id, limit))
    return job_id, True


def get_status(job_id: str) -> dict[str, Any] | None:
    return _JOBS.get(job_id)
