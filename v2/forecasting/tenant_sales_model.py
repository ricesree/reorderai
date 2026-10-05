"""Per-tenant sales prediction model: storage, staleness gate, 60-day history gate.

One model per tenant schema under data/models/tenants/<schema>/:
  model.joblib - trained model
  meta.json    - {status, trained_at, history_days, rows, min_history_days_required}

Training data is fetched into memory only for the duration of training and
discarded (never written to disk/DB) - see _fetch_training_sales().
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from config.data_paths import TENANT_MODELS_DIR

logger = logging.getLogger(__name__)

MIN_HISTORY_DAYS = int(os.getenv("FORECAST_MIN_HISTORY_DAYS", "60"))
RETRAIN_AFTER_HOURS = int(os.getenv("FORECAST_MODEL_RETRAIN_HOURS", "24"))

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(tenant_id: str) -> threading.Lock:
    """One lock per tenant so two concurrent requests can't both retrain at once."""
    with _locks_guard:
        if tenant_id not in _locks:
            _locks[tenant_id] = threading.Lock()
        return _locks[tenant_id]


def _tenant_dir(tenant_id: str) -> Path:
    return TENANT_MODELS_DIR / tenant_id


def _model_path(tenant_id: str) -> Path:
    return _tenant_dir(tenant_id) / "model.joblib"


def _meta_path(tenant_id: str) -> Path:
    return _tenant_dir(tenant_id) / "meta.json"


def load_meta(tenant_id: str) -> dict[str, Any] | None:
    path = _meta_path(tenant_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _save_meta(tenant_id: str, meta: dict[str, Any]) -> None:
    _tenant_dir(tenant_id).mkdir(parents=True, exist_ok=True)
    _meta_path(tenant_id).write_text(json.dumps(meta, indent=2), encoding="utf-8")


def is_stale(tenant_id: str) -> bool:
    """True if never checked, or last check is older than FORECAST_MODEL_RETRAIN_HOURS."""
    meta = load_meta(tenant_id)
    if meta is None:
        return True
    checked_at = pd.to_datetime(meta.get("trained_at"), errors="coerce", utc=True)
    if pd.isna(checked_at):
        return True
    age_hours = (datetime.now(timezone.utc) - checked_at.to_pydatetime()).total_seconds() / 3600
    return age_hours > RETRAIN_AFTER_HOURS


def history_span_days(sales: pd.DataFrame, date_col: str = "date") -> int:
    if sales.empty:
        return 0
    dates = pd.to_datetime(sales[date_col], errors="coerce").dropna()
    if dates.empty:
        return 0
    return int((dates.max() - dates.min()).days) + 1


def has_sufficient_history(sales: pd.DataFrame, date_col: str = "date") -> bool:
    return history_span_days(sales, date_col=date_col) >= MIN_HISTORY_DAYS


def can_predict(tenant_id: str) -> tuple[bool, str]:
    """Gate for serving predictions - independent of the retrain schedule."""
    meta = load_meta(tenant_id)
    if meta is None:
        return False, "no_model_trained_yet"
    if meta.get("status") != "trained":
        return False, str(meta.get("status") or "not_trained")
    if int(meta.get("history_days") or 0) < MIN_HISTORY_DAYS:
        return False, "insufficient_history"
    return True, "ok"


def load_model(tenant_id: str) -> Any | None:
    path = _model_path(tenant_id)
    if not path.exists():
        logger.debug("tenant %s: no model on disk at %s", tenant_id, path)
        return None
    import joblib

    logger.debug("tenant %s: loading model from %s", tenant_id, path)
    return joblib.load(path)


# --- Training data fetch / model fit ---


def _fetch_training_sales(tenant_id: str) -> pd.DataFrame:
    """Full per-product daily sales panel from order_items/orders/products (tenant_training.py)."""
    from v2.forecasting.tenant_training import fetch_tenant_daily_sales

    return fetch_tenant_daily_sales(tenant_id)


def _fit_model(sales: pd.DataFrame) -> dict[str, Any]:
    """Train one LightGBM model across the tenant's whole catalog. Returns a bundle
    dict (model + the exact feature/category lists needed to reproduce predictions).
    """
    import lightgbm as lgb

    from v2.forecasting.tenant_training import (
        CATEGORICAL_COLS,
        FEATURE_COLS,
        LAGS,
        ROLLING_WINDOWS,
        build_training_frame,
    )

    frame = build_training_frame(sales)
    if frame.empty:
        raise ValueError("No trainable rows after feature engineering (lag_28 needs 28+ days of history)")

    product_categories = sorted(frame["product_id"].unique().tolist())
    category_categories = sorted(frame["category_id"].astype(str).unique().tolist())
    frame = frame.copy()
    frame["product_id"] = pd.Categorical(frame["product_id"], categories=product_categories)
    frame["category_id"] = pd.Categorical(frame["category_id"].astype(str), categories=category_categories)

    model = lgb.LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=1.3,
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(frame[FEATURE_COLS], frame["target_sales"], categorical_feature=CATEGORICAL_COLS)

    return {
        "model": model,
        "features": FEATURE_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "product_categories": product_categories,
        "category_categories": category_categories,
        "lags": LAGS,
        "rolling_windows": ROLLING_WINDOWS,
    }


def ensure_tenant_model(tenant_id: str) -> dict[str, Any]:
    """Train tenant's model if stale; otherwise no-op. Called inline before serving a prediction.

    Sales fetched here are local to this call only - nothing is persisted,
    so there is no cleanup step once training finishes. Locked per-tenant so
    two concurrent requests for the same stale tenant don't both retrain.
    """
    with _lock_for(tenant_id):
        if not is_stale(tenant_id):
            logger.info("tenant %s: model is fresh (<%dh old) — skipping retrain", tenant_id, RETRAIN_AFTER_HOURS)
            return {**(load_meta(tenant_id) or {}), "status": "skipped_fresh"}

        logger.info("tenant %s: model missing/stale — fetching training data", tenant_id)
        t0 = time.monotonic()
        sales = _fetch_training_sales(tenant_id)
        history_days = history_span_days(sales)
        logger.info(
            "tenant %s: fetched %d rows, %d day(s) of history (%.1fs)",
            tenant_id, len(sales), history_days, time.monotonic() - t0,
        )
        now = datetime.now(timezone.utc).isoformat()

        if history_days < MIN_HISTORY_DAYS:
            logger.info(
                "tenant %s: insufficient history (%d/%d days required) — not training",
                tenant_id, history_days, MIN_HISTORY_DAYS,
            )
            meta = {
                "status": "insufficient_history",
                "trained_at": now,
                "history_days": history_days,
                "rows": int(len(sales)),
                "min_history_days_required": MIN_HISTORY_DAYS,
            }
            _save_meta(tenant_id, meta)
            return meta

        logger.info("tenant %s: training model...", tenant_id)
        t1 = time.monotonic()
        bundle = _fit_model(sales)
        import joblib

        _tenant_dir(tenant_id).mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, _model_path(tenant_id))
        logger.info(
            "tenant %s: trained and saved model (%d products, %.1fs)",
            tenant_id, len(bundle.get("product_categories") or []), time.monotonic() - t1,
        )
        meta = {
            "status": "trained",
            "trained_at": now,
            "history_days": history_days,
            "rows": int(len(sales)),
            "min_history_days_required": MIN_HISTORY_DAYS,
        }
        _save_meta(tenant_id, meta)
        return meta


def predict_item_days(
    *,
    tenant_id: str,
    item_id: str,
    history: list[dict[str, Any]],
    attrs: dict[str, Any],
    as_of: date,
    horizon_days: int,
) -> list[dict[str, Any]] | None:
    """Recursive day-by-day forecast for one item: predict day+1, feed it back
    into the lag/rolling features, predict day+2, and so on. Returns None if
    this tenant has no trained model yet.

    `history` is a list of {"date": ..., "qty": ...} dicts — the same shape
    ForecastStore.get_sales_history()/global_lightgbm_predictor already use,
    so callers can pass in sales history they've already fetched.
    """
    from v2.forecasting.tenant_training import predict_feature_row

    bundle = load_model(tenant_id)
    if bundle is None:
        return None
    horizon = max(int(horizon_days), 1)
    lags: list[int] = bundle["lags"]
    rolling_windows: list[int] = bundle["rolling_windows"]
    lookback = max(max(lags, default=0), max(rolling_windows, default=0)) + 5

    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(days=lookback)
    series = pd.Series(0.0, index=pd.date_range(start, end, freq="D"), dtype=float)
    for row in history or []:
        d = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(d):
            continue
        d = d.normalize()
        if d in series.index:
            series.loc[d] = float(row.get("qty") or 0.0)

    static = {**attrs, "product_id": str(item_id)}
    product_categories = bundle["product_categories"]
    category_categories = bundle["category_categories"]
    feature_cols = bundle["features"]

    out: list[dict[str, Any]] = []
    for i in range(horizon):
        target = pd.Timestamp(as_of + timedelta(days=i + 1))
        row = predict_feature_row(target_date=target, history=series, static=static)
        df = pd.DataFrame([row])
        df["product_id"] = pd.Categorical(df["product_id"].astype(str), categories=product_categories)
        df["category_id"] = pd.Categorical(df["category_id"].astype(str), categories=category_categories)
        qty = float(max(bundle["model"].predict(df[feature_cols])[0], 0.0))
        out.append({"date": target.date().isoformat(), "qty": round(qty, 4)})
        series.loc[target] = qty
    return out
