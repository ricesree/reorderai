"""Per-tenant "window total" model: predicts ONE number per item — total units
sold over the next H days — instead of a day-by-day path.

Runs in parallel with the existing models (shadow output only; order math does
not read it). Same storage / staleness / history-gate pattern as
tenant_sales_model, under a separate key so the two never touch:
  tenants/<schema>/horizon_total/{model.joblib, meta.json}

Features are only what the DB gives us (order_items sales history, products /
categories attrs) plus pure calendar arithmetic on the window dates. Nothing
that is unknown at predict time (e.g. future discounts, stock-outs).
"""

from __future__ import annotations

import io
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.data_paths import TENANT_MODELS_DIR
from v2.forecasting import model_storage
from v2.forecasting.festival_calendar import calendar_labels, is_weekend
from v2.forecasting.tenant_sales_model import MIN_HISTORY_DAYS, RETRAIN_AFTER_HOURS, history_span_days

logger = logging.getLogger(__name__)

MODEL_NAME = "horizon_total_lgbm"
HORIZONS = (3, 7, 14, 21, 30, 45)
ANCHOR_STEP = 3  # anchor every 3rd day: overlapping windows are near-duplicates
WARMUP_DAYS = 28
HOLDOUT_DAYS = 30
MAX_GAP = 120  # days_since_last_sale cap == predict-time history length
QUANTILES = {"p50": 0.5, "p90": 0.9}

STATIC_COLS = ["product_id", "category_id", "list_price", "pack_size", "is_scale", "is_perishable"]
HISTORY_COLS = [
    "sum_7", "sum_28", "sum_90", "selling_days_28", "selling_days_90",
    "days_since_last_sale", "std_28", "mean_nonzero_90",
]
WINDOW_COLS = ["horizon", "weekend_days", "festival_days", "start_dow", "start_month"]
FEATURE_COLS = STATIC_COLS + HISTORY_COLS + WINDOW_COLS
CATEGORICAL_COLS = ["product_id", "category_id"]

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()
_model_cache: dict[str, Any] = {}


def _lock_for(tenant_id: str) -> threading.Lock:
    with _locks_guard:
        if tenant_id not in _locks:
            _locks[tenant_id] = threading.Lock()
        return _locks[tenant_id]


def _dir(tenant_id: str) -> Path:
    return TENANT_MODELS_DIR / tenant_id / "horizon_total"


def _blob_key(tenant_id: str, name: str) -> str:
    return f"tenants/{tenant_id}/horizon_total/{name}"


def load_meta(tenant_id: str) -> dict[str, Any] | None:
    try:
        raw = model_storage.read_bytes(_blob_key(tenant_id, "meta.json"), _dir(tenant_id) / "meta.json")
        return None if raw is None else json.loads(raw.decode("utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(tenant_id: str, meta: dict[str, Any]) -> None:
    data = json.dumps(meta, indent=2).encode("utf-8")
    model_storage.write_bytes(_blob_key(tenant_id, "meta.json"), _dir(tenant_id) / "meta.json", data)


def is_stale(tenant_id: str) -> bool:
    meta = load_meta(tenant_id)
    if meta is None:
        return True
    checked_at = pd.to_datetime(meta.get("trained_at"), errors="coerce", utc=True)
    if pd.isna(checked_at):
        return True
    age_hours = (datetime.now(timezone.utc) - checked_at.to_pydatetime()).total_seconds() / 3600
    return age_hours > RETRAIN_AFTER_HOURS


def can_predict(tenant_id: str) -> tuple[bool, str]:
    meta = load_meta(tenant_id)
    if meta is None:
        return False, "no_model_trained_yet"
    if meta.get("status") != "trained":
        return False, str(meta.get("status") or "not_trained")
    return True, "ok"


def load_model(tenant_id: str) -> dict[str, Any] | None:
    path = _dir(tenant_id) / "model.joblib"
    if str(path) in _model_cache:
        return _model_cache[str(path)]
    raw = model_storage.read_bytes(_blob_key(tenant_id, "model.joblib"), path)
    if raw is None:
        return None
    import joblib

    bundle = joblib.load(io.BytesIO(raw))
    _model_cache[str(path)] = bundle
    return bundle


def _save_model(tenant_id: str, bundle: dict[str, Any]) -> None:
    import joblib

    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    path = _dir(tenant_id) / "model.joblib"
    model_storage.write_bytes(_blob_key(tenant_id, "model.joblib"), path, buf.getvalue())
    _model_cache.pop(str(path), None)


# --- features shared by training and prediction ---


def history_features(s: np.ndarray, idx: np.ndarray) -> dict[str, np.ndarray]:
    """Sales-history features as of positions ``idx`` of daily series ``s`` (inclusive).

    Days before s[0] count as zero sales, so a 120-day predict window and a
    longer training series give the same numbers for the same recent history.
    """
    s = np.asarray(s, dtype=float)
    idx = np.asarray(idx, dtype=int)
    end = idx + 1
    cs = np.concatenate([[0.0], np.cumsum(s)])
    cs2 = np.concatenate([[0.0], np.cumsum(s * s)])
    cn = np.concatenate([[0], np.cumsum(s > 0)])

    def win(c: np.ndarray, w: int) -> np.ndarray:
        return c[end] - c[np.maximum(end - w, 0)]

    out = {f"sum_{w}": win(cs, w) for w in (7, 28, 90)}
    out["selling_days_28"] = win(cn, 28).astype(float)
    out["selling_days_90"] = win(cn, 90).astype(float)
    mean28 = out["sum_28"] / 28.0
    out["std_28"] = np.sqrt(np.maximum(win(cs2, 28) / 28.0 - mean28**2, 0.0))
    out["mean_nonzero_90"] = np.divide(
        out["sum_90"], out["selling_days_90"],
        out=np.zeros(len(idx)), where=out["selling_days_90"] > 0,
    )
    last_sale = np.maximum.accumulate(np.where(s > 0, np.arange(len(s)), -1))[idx]
    out["days_since_last_sale"] = np.where(
        last_sale >= 0, np.minimum(idx - last_sale, MAX_GAP), MAX_GAP
    ).astype(float)
    return out


@lru_cache(maxsize=8192)
def window_features(as_of: date, horizon: int) -> dict[str, int]:
    """Calendar composition of the window as_of+1 .. as_of+horizon."""
    days = [as_of + timedelta(days=i + 1) for i in range(horizon)]
    return {
        "horizon": horizon,
        "weekend_days": sum(is_weekend(d) for d in days),
        "festival_days": sum(any(t != "weekend" for t in calendar_labels(d)) for d in days),
        "start_dow": days[0].weekday(),
        "start_month": days[0].month,
    }


def build_horizon_frame(daily: pd.DataFrame, horizons=HORIZONS, step: int = ANCHOR_STEP) -> pd.DataFrame:
    """(product, anchor t, H) rows: features as of t, target = units sold in (t, t+H].

    ``daily`` is the dense zero-filled panel from tenant_sales_cache.load_tenant_sales_panel.
    Anchors with no sales in the prior 90 days are dropped (predicted as 0 directly).
    """
    if daily.empty:
        return pd.DataFrame(columns=FEATURE_COLS + ["anchor", "target"])
    all_dates = pd.DatetimeIndex(sorted(daily["date"].unique()))
    n = len(all_dates)
    parts: list[pd.DataFrame] = []
    # ponytail: Python loop over products; vectorise across products if catalogs reach 50k+ SKUs.
    for pid, g in daily.sort_values("date").groupby("product_id", sort=False):
        s = g["target_sales"].to_numpy(dtype=float)
        if len(s) != n or s.sum() <= 0:
            continue
        cs = np.concatenate([[0.0], np.cumsum(s)])
        static = g.iloc[0]
        for h in horizons:
            idx = np.arange(WARMUP_DAYS - 1, n - h, step)
            if idx.size == 0:
                continue
            feats = history_features(s, idx)
            keep = feats["sum_90"] > 0
            if not keep.any():
                continue
            idx = idx[keep]
            part = pd.DataFrame({k: v[keep] for k, v in feats.items()})
            part["anchor"] = all_dates[idx]
            part["target"] = cs[idx + 1 + h] - cs[idx + 1]
            part["horizon"] = h
            for c in STATIC_COLS:
                part[c] = static[c] if c != "product_id" else str(pid)
            parts.append(part)
    if not parts:
        return pd.DataFrame(columns=FEATURE_COLS + ["anchor", "target"])
    frame = pd.concat(parts, ignore_index=True)
    win = pd.DataFrame(
        [
            {"anchor": a, "horizon": h, **window_features(a.date(), int(h))}
            for a, h in frame[["anchor", "horizon"]].drop_duplicates().itertuples(index=False)
        ]
    )
    return frame.merge(win, on=["anchor", "horizon"], how="left")


def _as_model_input(frame: pd.DataFrame, products: list[str], categories: list[str]) -> pd.DataFrame:
    x = frame[FEATURE_COLS].copy()
    # Ids unseen at training time become NaN (LightGBM's "unknown" category).
    for col, cats in (("product_id", products), ("category_id", categories)):
        vals = x[col].astype(str)
        x[col] = pd.Categorical(vals.where(vals.isin(cats)), categories=cats)
    for c in FEATURE_COLS:
        if c not in CATEGORICAL_COLS:
            x[c] = pd.to_numeric(x[c], errors="coerce").fillna(0.0)
    return x


def _train(frame: pd.DataFrame) -> dict[str, Any]:
    import lightgbm as lgb

    products = sorted(frame["product_id"].astype(str).unique().tolist())
    categories = sorted(frame["category_id"].astype(str).unique().tolist())
    x = _as_model_input(frame, products, categories)
    models = {}
    for name, alpha in QUANTILES.items():
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=-1,
        )
        m.fit(x, frame["target"], categorical_feature=CATEGORICAL_COLS)
        models[name] = m
    return {
        "models": models,
        "features": FEATURE_COLS,
        "product_categories": products,
        "category_categories": categories,
    }


def _predict(bundle: dict[str, Any], frame: pd.DataFrame) -> dict[str, np.ndarray]:
    x = _as_model_input(frame, bundle["product_categories"], bundle["category_categories"])
    return {k: np.maximum(m.predict(x[bundle["features"]]), 0.0) for k, m in bundle["models"].items()}


def _holdout_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    """Train on anchors whose window ends before the cutoff, score windows after it.

    Reports WAPE of P50 vs the ADS baseline (sum_28/28 × H) so it is visible
    whether this model actually beats the current approach.
    """
    cutoff = frame["anchor"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    ends = frame["anchor"] + pd.to_timedelta(frame["horizon"], unit="D")
    train, test = frame[ends <= cutoff], frame[frame["anchor"] >= cutoff]
    if train.empty or test.empty or test["target"].sum() <= 0:
        return {"holdout_rows": int(len(test))}
    pred = _predict(_train(train), test)
    actual = test["target"].to_numpy(dtype=float)
    ads = test["sum_28"].to_numpy(dtype=float) / 28.0 * test["horizon"].to_numpy(dtype=float)
    denom = float(actual.sum())
    return {
        "holdout_rows": int(len(test)),
        "wape_p50": round(float(np.abs(actual - pred["p50"]).sum() / denom), 4),
        "wape_ads": round(float(np.abs(actual - ads).sum() / denom), 4),
        "p90_coverage": round(float((actual <= pred["p90"]).mean()), 4),
    }


def _fetch_training_panel(tenant_id: str) -> pd.DataFrame:
    from v2.forecasting.tenant_sales_cache import load_tenant_sales_panel

    return load_tenant_sales_panel(tenant_id)


def ensure_horizon_model(tenant_id: str) -> dict[str, Any]:
    """Train if missing/stale (inline, per-tenant lock); otherwise no-op."""
    with _lock_for(tenant_id):
        if not is_stale(tenant_id):
            return {**(load_meta(tenant_id) or {}), "status_check": "skipped_fresh"}

        t0 = time.monotonic()
        panel = _fetch_training_panel(tenant_id)
        history_days = history_span_days(panel)
        now = datetime.now(timezone.utc).isoformat()
        base = {"trained_at": now, "history_days": history_days, "min_history_days_required": MIN_HISTORY_DAYS}

        if history_days < MIN_HISTORY_DAYS:
            meta = {**base, "status": "insufficient_history"}
            _save_meta(tenant_id, meta)
            logger.info("tenant %s: horizon model — insufficient history (%d days)", tenant_id, history_days)
            return meta

        frame = build_horizon_frame(panel)
        if frame.empty:
            meta = {**base, "status": "no_trainable_rows"}
            _save_meta(tenant_id, meta)
            return meta

        metrics = _holdout_metrics(frame)
        bundle = _train(frame)
        # Model before meta: a failed upload never leaves meta saying "trained" without a model.
        _save_model(tenant_id, bundle)
        meta = {**base, "status": "trained", "rows": int(len(frame)), "horizons": list(HORIZONS), **metrics}
        _save_meta(tenant_id, meta)
        logger.info(
            "tenant %s: horizon model trained (%d rows, %.1fs) %s",
            tenant_id, len(frame), time.monotonic() - t0, metrics,
        )
        return meta


def predict_window_totals(
    tenant_id: str,
    item_ids: list[str],
    *,
    as_of: date,
    horizon_days: int,
    sales: pd.DataFrame | None = None,
    attrs: dict[str, dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """{item_id: {horizon_days, p50_total, p90_total, model, trained_at}} for the next H days.

    ``sales`` (sparse product_id/date/target_sales) and ``attrs`` default to the
    same DB sources used for training, so train and predict see identical data.
    Items with no product row are left out; items with no sales in 90 days get 0.
    """
    bundle = load_model(tenant_id)
    if bundle is None or not item_ids:
        return {}
    if sales is None:
        from v2.forecasting.tenant_sales_cache import update_sales_cache

        sales = update_sales_cache(tenant_id)
    if attrs is None:
        from v2.forecasting.tenant_training import fetch_product_attrs

        attrs = fetch_product_attrs(tenant_id, item_ids)

    h = max(int(horizon_days), 1)
    dates = pd.date_range(pd.Timestamp(as_of) - pd.Timedelta(days=MAX_GAP - 1), pd.Timestamp(as_of), freq="D")
    wanted = [str(i) for i in item_ids if str(i) in attrs]
    if not wanted:
        return {}
    recent = sales[sales["product_id"].isin(wanted) & sales["date"].isin(dates)]
    matrix = (
        recent.pivot_table(index="product_id", columns="date", values="target_sales", aggfunc="sum")
        .reindex(index=wanted, columns=dates)
        .fillna(0.0)
    )
    win = window_features(as_of, h)
    rows = []
    for iid, s in zip(wanted, matrix.to_numpy(dtype=float)):
        feats = {k: float(v[0]) for k, v in history_features(s, np.array([len(s) - 1])).items()}
        a = attrs[iid]
        rows.append(
            {
                "product_id": iid,
                "category_id": str(a.get("category_id") or ""),
                "list_price": float(a.get("list_price") or 0.0),
                "pack_size": float(a.get("pack_size") or 1.0),
                "is_scale": int(bool(a.get("is_scale"))),
                "is_perishable": int(bool(a.get("is_perishable"))),
                **feats,
                **win,
            }
        )
    frame = pd.DataFrame(rows)
    pred = _predict(bundle, frame)
    dead = frame["sum_90"].to_numpy() <= 0
    p50 = np.where(dead, 0.0, pred["p50"])
    p90 = np.where(dead, 0.0, np.maximum(pred["p90"], p50))
    trained_at = (load_meta(tenant_id) or {}).get("trained_at")
    return {
        iid: {
            "horizon_days": h,
            "p50_total": round(float(a), 2),
            "p90_total": round(float(b), 2),
            "model": MODEL_NAME,
            "trained_at": trained_at,
        }
        for iid, a, b in zip(wanted, p50, p90)
    }
