"""Tests for the per-tenant model staleness / history gates."""

from datetime import date, datetime, timedelta, timezone

import pandas as pd

from v2.forecasting import tenant_sales_model as tsm


def test_is_stale_true_when_never_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    assert tsm.is_stale("tenant_a") is True


def test_is_stale_false_within_retrain_window(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    recent = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    tsm._save_meta("tenant_a", {"status": "trained", "trained_at": recent, "history_days": 90})
    assert tsm.is_stale("tenant_a") is False


def test_is_stale_true_after_retrain_window(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    old = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    tsm._save_meta("tenant_a", {"status": "trained", "trained_at": old, "history_days": 90})
    assert tsm.is_stale("tenant_a") is True


def test_history_span_days():
    sales = pd.DataFrame({"date": pd.date_range("2026-01-01", periods=60, freq="D")})
    assert tsm.history_span_days(sales) == 60
    assert tsm.history_span_days(pd.DataFrame({"date": []})) == 0


def test_can_predict_blocks_below_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    tsm._save_meta(
        "tenant_a",
        {"status": "trained", "trained_at": datetime.now(timezone.utc).isoformat(), "history_days": 59},
    )
    allowed, reason = tsm.can_predict("tenant_a")
    assert allowed is False
    assert reason == "insufficient_history"


def test_can_predict_allows_above_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    tsm._save_meta(
        "tenant_a",
        {"status": "trained", "trained_at": datetime.now(timezone.utc).isoformat(), "history_days": 60},
    )
    allowed, reason = tsm.can_predict("tenant_a")
    assert allowed is True
    assert reason == "ok"


def test_can_predict_blocks_when_untrained(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    assert tsm.can_predict("tenant_a") == (False, "no_model_trained_yet")


def _synthetic_daily_sales(days: int = 120, n_products: int = 5) -> pd.DataFrame:
    """Fabricated per-product daily sales panel, same shape as tenant_training.fetch_tenant_daily_sales."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for p in range(n_products):
        base = 2.0 + p
        for i, d in enumerate(dates):
            qty = max(base + (2.0 if d.dayofweek >= 5 else 0.0) + (i % 3) * 0.5, 0.0)
            rows.append(
                {
                    "date": d,
                    "product_id": str(p),
                    "category_id": str(p % 2),
                    "category_name": "produce" if p % 2 == 0 else "snacks",
                    "list_price": 3.0 + p,
                    "pack_size": 1.0,
                    "is_scale": 0,
                    "any_discount_flag": 0,
                    "target_sales": qty,
                    "is_perishable": int(p % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


def test_ensure_tenant_model_end_to_end(tmp_path, monkeypatch):
    """Fabricated 120-day sales -> real fetch is bypassed, real fit/save/predict runs."""
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(tsm, "_fetch_training_sales", lambda tenant_id: _synthetic_daily_sales())

    result = tsm.ensure_tenant_model("tenant_a")
    assert result["status"] == "trained"
    assert result["history_days"] == 120

    allowed, reason = tsm.can_predict("tenant_a")
    assert (allowed, reason) == (True, "ok")

    history = [
        {"date": d.date().isoformat(), "qty": 2.0 + (i % 4)}
        for i, d in enumerate(pd.date_range("2026-04-01", periods=35, freq="D"))
    ]
    forecast = tsm.predict_item_days(
        tenant_id="tenant_a",
        item_id="0",
        history=history,
        attrs={"category_id": "0", "list_price": 3.0, "pack_size": 1.0, "is_scale": False, "is_perishable": True},
        as_of=date(2026, 5, 6),
        horizon_days=10,
    )
    assert forecast is not None
    assert len(forecast) == 10
    assert all(f["qty"] >= 0 for f in forecast)
    assert [f["date"] for f in forecast] == [
        (date(2026, 5, 6) + timedelta(days=i + 1)).isoformat() for i in range(10)
    ]


def test_ensure_tenant_model_concurrent_requests_train_once(tmp_path, monkeypatch):
    """Two threads racing ensure_tenant_model for the same stale tenant must train only once."""
    import threading
    import time

    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    calls = {"n": 0}
    call_lock = threading.Lock()

    def _slow_fetch(tenant_id: str) -> pd.DataFrame:
        with call_lock:
            calls["n"] += 1
        time.sleep(0.2)  # widen the race window so both threads are mid-check before either finishes
        return _synthetic_daily_sales()

    monkeypatch.setattr(tsm, "_fetch_training_sales", _slow_fetch)

    results: list[dict] = []
    results_lock = threading.Lock()

    def _worker() -> None:
        r = tsm.ensure_tenant_model("tenant_racing")
        with results_lock:
            results.append(r)

    threads = [threading.Thread(target=_worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert calls["n"] == 1, f"expected exactly one training run, got {calls['n']}"
    assert sum(1 for r in results if r["status"] == "trained") == 1
    assert sum(1 for r in results if r["status"] == "skipped_fresh") == 4


def test_ensure_tenant_model_skips_when_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(tsm, "TENANT_MODELS_DIR", tmp_path)
    calls = {"n": 0}

    def _fetch(tenant_id: str) -> pd.DataFrame:
        calls["n"] += 1
        return _synthetic_daily_sales()

    monkeypatch.setattr(tsm, "_fetch_training_sales", _fetch)
    tsm.ensure_tenant_model("tenant_a")
    assert calls["n"] == 1
    tsm.ensure_tenant_model("tenant_a")
    assert calls["n"] == 1  # still fresh — second call must not refetch/retrain
