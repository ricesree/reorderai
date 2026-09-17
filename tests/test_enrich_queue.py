from __future__ import annotations

import threading
import time

from api.services import enrich_queue


def test_second_request_for_same_tenant_is_rejected_while_first_runs(monkeypatch):
    started = threading.Event()
    release = threading.Event()

    def fake_enrich_missing(tenant_id=None, limit=None):
        started.set()
        release.wait(timeout=5)
        return {"updated": 0, "items": [], "failed": []}

    monkeypatch.setattr(enrich_queue.service, "enrich_missing", fake_enrich_missing)

    job_id, is_new = enrich_queue.enqueue("test-tenant-a", None)
    assert is_new is True
    assert started.wait(timeout=5), "worker never picked up the first job"

    dup_job_id, dup_is_new = enrich_queue.enqueue("test-tenant-a", None)
    assert dup_is_new is False
    assert dup_job_id == job_id

    other_job_id, other_is_new = enrich_queue.enqueue("test-tenant-b", None)
    assert other_is_new is True
    assert other_job_id != job_id

    release.set()
    for _ in range(50):
        if enrich_queue.get_status(job_id)["status"] == "success":
            break
        time.sleep(0.05)
    assert enrich_queue.get_status(job_id)["status"] == "success"

    freed_job_id, freed_is_new = enrich_queue.enqueue("test-tenant-a", None)
    assert freed_is_new is True
    assert freed_job_id != job_id
