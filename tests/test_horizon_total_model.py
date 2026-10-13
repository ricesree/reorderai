"""Parallel window-total model: frame targets, feature parity, train/predict end to end."""

from datetime import date

import numpy as np
import pandas as pd

from v2.forecasting import horizon_total_model as hm


def _panel(days: int = 150) -> pd.DataFrame:
    """Dense synthetic panel, same shape as tenant_sales_cache.load_tenant_sales_panel."""
    dates = pd.date_range("2026-01-01", periods=days, freq="D")
    rows = []
    for pid, kind in (("1", "steady"), ("2", "weekend"), ("3", "sparse")):
        for i, d in enumerate(dates):
            if kind == "steady":
                q = 3.0
            elif kind == "weekend":
                q = 8.0 if d.dayofweek >= 5 else 1.0
            else:
                q = 4.0 if i % 9 == 0 else 0.0
            rows.append(
                {
                    "product_id": pid, "date": d, "target_sales": q, "category_id": "7",
                    "list_price": 2.5, "pack_size": 1.0, "is_scale": 0, "is_perishable": 0,
                }
            )
    return pd.DataFrame(rows)


def test_frame_targets_are_window_sums():
    panel = _panel()
    frame = hm.build_horizon_frame(panel, horizons=(7,))
    steady = frame[frame["product_id"] == "1"]
    assert (steady["target"] == 21.0).all()
    last = panel["date"].max()
    assert (frame["anchor"] + pd.to_timedelta(frame["horizon"], unit="D") <= last).all()


def test_history_features_padding_parity():
    """Same recent history => same features whether the series is long (train) or 120d (predict)."""
    rng = np.random.default_rng(0)
    long = rng.poisson(2, 300).astype(float)
    a = hm.history_features(long, np.array([299]))
    b = hm.history_features(long[-120:], np.array([119]))
    for k in a:
        assert np.isclose(a[k][0], b[k][0]), k


def test_train_and_predict_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "TENANT_MODELS_DIR", tmp_path)
    panel = _panel()
    monkeypatch.setattr(hm, "_fetch_training_panel", lambda tenant_id: panel)

    meta = hm.ensure_horizon_model("t1")
    assert meta["status"] == "trained"
    assert "wape_p50" in meta and "wape_ads" in meta
    assert hm.can_predict("t1") == (True, "ok")
    assert hm.ensure_horizon_model("t1")["status_check"] == "skipped_fresh"

    sparse = panel[panel["target_sales"] > 0][["product_id", "date", "target_sales"]]
    attrs = {p: {"category_id": "7", "list_price": 2.5, "pack_size": 1.0} for p in ("1", "2", "3")}
    as_of = panel["date"].max().date()
    out = hm.predict_window_totals(
        "t1", ["1", "2", "3", "999"], as_of=as_of, horizon_days=10, sales=sparse, attrs=attrs
    )
    assert set(out) == {"1", "2", "3"}  # 999 has no product row -> left out
    for v in out.values():
        assert v["horizon_days"] == 10
        assert v["p90_total"] >= v["p50_total"] >= 0
    assert 15 <= out["1"]["p50_total"] <= 45  # steady 3/day × 10d ≈ 30


def test_dead_item_predicts_zero(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "TENANT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(hm, "_fetch_training_panel", lambda tenant_id: _panel())
    hm.ensure_horizon_model("t2")
    empty = pd.DataFrame(columns=["product_id", "date", "target_sales"])
    out = hm.predict_window_totals(
        "t2", ["1"], as_of=date(2026, 6, 1), horizon_days=7, sales=empty, attrs={"1": {}}
    )
    assert out["1"]["p50_total"] == 0.0 and out["1"]["p90_total"] == 0.0


def test_insufficient_history_not_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "TENANT_MODELS_DIR", tmp_path)
    monkeypatch.setattr(hm, "_fetch_training_panel", lambda tenant_id: _panel(days=30))
    assert hm.ensure_horizon_model("t3")["status"] == "insufficient_history"
    assert hm.can_predict("t3")[0] is False
