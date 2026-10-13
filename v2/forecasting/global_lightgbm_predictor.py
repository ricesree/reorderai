"""Global LightGBM daily sales forecast (Kevin joblib bundle).

Loads models/global_lightgbm_sales_model.joblib and recursively predicts
daily qty for the upcoming order window (sales_series.forecast).
"""

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.data_paths import PROJECT_ROOT
from v2.forecasting import model_storage

DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "global_lightgbm_sales_model.joblib"
# Need lag_56 + rolling_56 → keep at least this many history days
MIN_HISTORY_DAYS = 70


def global_lgbm_enabled() -> bool:
    return os.getenv("USE_GLOBAL_LIGHTGBM", "1").lower() in {"1", "true", "yes"}


def model_path() -> Path:
    raw = os.getenv("GLOBAL_LIGHTGBM_MODEL_PATH", "").strip()
    return Path(raw) if raw else DEFAULT_MODEL_PATH


@dataclass
class _Bundle:
    model: Any
    features: list[str]
    product_categories: list[str]
    category_categories: list[str]
    lags: list[int]
    rolling_windows: list[int]


@lru_cache(maxsize=1)
def _load_bundle() -> _Bundle | None:
    path = model_path()
    try:
        import joblib

        # Blob key global/<filename> when AZURE_STORAGE_CONNECTION_STRING is set, else local path
        raw = model_storage.read_bytes(f"global/{path.name}", path)
        if raw is None:
            return None
        obj = joblib.load(io.BytesIO(raw))
    except Exception:
        return None
    if not isinstance(obj, dict) or "model" not in obj:
        return None
    return _Bundle(
        model=obj["model"],
        features=list(obj["model_features"]),
        product_categories=[str(x) for x in obj.get("product_categories") or []],
        category_categories=[str(x) for x in obj.get("category_categories") or []],
        lags=[int(x) for x in (obj.get("lags") or [1, 2, 3, 7, 14, 21, 28, 56])],
        rolling_windows=[int(x) for x in (obj.get("rolling_windows") or [7, 14, 28, 56])],
    )


def model_ready() -> bool:
    return global_lgbm_enabled() and _load_bundle() is not None


def _history_to_series(
    history: list[dict[str, Any]],
    *,
    as_of: date,
    lookback_days: int = MIN_HISTORY_DAYS,
) -> pd.Series:
    """Zero-filled daily qty ending at as_of (inclusive if history has that day)."""
    end = pd.Timestamp(as_of)
    start = end - pd.Timedelta(days=max(int(lookback_days), MIN_HISTORY_DAYS) - 1)
    idx = pd.date_range(start, end, freq="D")
    s = pd.Series(0.0, index=idx, dtype=float)
    for row in history or []:
        d = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(d):
            continue
        d = d.normalize()
        if d in s.index:
            s.loc[d] = float(row.get("qty") or 0.0)
    return s


def _static_row(attrs: dict[str, Any] | None) -> dict[str, float | int | str]:
    a = attrs or {}
    return {
        "product_id": str(a.get("product_id") or ""),
        "category_id": str(a.get("category_id") if a.get("category_id") is not None else ""),
        "list_price": float(a.get("list_price") or 0.0),
        "purchase_price": float(a.get("purchase_price") or 0.0),
        "is_scale": int(bool(a.get("is_scale"))),
        "pack_size": float(a.get("pack_size") or 1.0),
        "any_discount_flag": int(bool(a.get("any_discount_flag"))),
        "product_age_days": float(a.get("product_age_days") or 0.0),
        "price_change_percent": float(a.get("price_change_percent") or 0.0),
    }


def _feature_row(
    *,
    target_date: pd.Timestamp,
    history: pd.Series,
    static: dict[str, float | int | str],
    lags: list[int],
    rolling_windows: list[int],
) -> dict[str, Any]:
    # history index is past days only; target_date is the day we predict
    # lag_k = sales on target_date - k days
    vals: dict[str, Any] = dict(static)
    d = target_date
    vals["dow"] = int(d.dayofweek)  # Mon=0 .. Sun=6 (pandas); model trained with Postgres DOW?
    # Training SQL used EXTRACT(DOW) Sunday=0. Pandas dayofweek Monday=0.
    # Remap to Postgres DOW so categorical calendar matches training.
    vals["dow"] = int((d.dayofweek + 1) % 7)  # Sun=0 .. Sat=6
    vals["day_of_month"] = int(d.day)
    vals["week_of_year"] = int(d.isocalendar().week)
    vals["month"] = int(d.month)
    vals["quarter"] = int(d.quarter)
    vals["year"] = int(d.year)
    vals["is_weekend"] = int(vals["dow"] in (0, 6))  # Sun/Sat under Postgres DOW
    vals["is_month_start"] = int(d.is_month_start)
    vals["is_month_end"] = int(d.is_month_end)

    for lag in lags:
        key = f"lag_{lag}"
        past = d - pd.Timedelta(days=lag)
        vals[key] = float(history.loc[past]) if past in history.index else 0.0

    # Rolling stats over history ending the day before target (shift-1 style)
    hist_before = history[history.index < d]
    for w in rolling_windows:
        window = hist_before.tail(w)
        vals[f"rolling_mean_{w}"] = float(window.mean()) if len(window) else 0.0
        vals[f"rolling_sum_{w}"] = float(window.sum()) if len(window) else 0.0
        vals[f"rolling_std_{w}"] = float(window.std(ddof=0)) if len(window) > 1 else 0.0
        vals[f"rolling_max_{w}"] = float(window.max()) if len(window) else 0.0
    w28 = hist_before.tail(28)
    if len(w28):
        vals["zero_sales_ratio_28"] = float((w28 <= 0).mean())
    else:
        vals["zero_sales_ratio_28"] = 1.0
    return vals


def _predict_frame(bundle: _Bundle, rows: list[dict[str, Any]]) -> np.ndarray:
    df = pd.DataFrame(rows)
    df["product_id"] = pd.Categorical(
        df["product_id"].astype(str), categories=bundle.product_categories
    )
    df["category_id"] = pd.Categorical(
        df["category_id"].astype(str), categories=bundle.category_categories
    )
    for c in bundle.features:
        if c in ("product_id", "category_id"):
            continue
        if c not in df.columns:
            df[c] = 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    pred = bundle.model.predict(df[bundle.features])
    return np.maximum(np.asarray(pred, dtype=float), 0.0)


def forecast_item_days(
    *,
    item_id: str,
    history: list[dict[str, Any]],
    attrs: dict[str, Any] | None,
    as_of: date,
    horizon_days: int,
) -> list[dict[str, Any]] | None:
    """Recursive daily forecast for one SKU. Returns None if model unavailable."""
    bundle = _load_bundle()
    if bundle is None or not global_lgbm_enabled():
        return None
    horizon = max(int(horizon_days), 1)
    series = _history_to_series(history, as_of=as_of)
    static = _static_row({**(attrs or {}), "product_id": str(item_id)})
    out: list[dict[str, Any]] = []
    for i in range(horizon):
        target = pd.Timestamp(as_of + timedelta(days=i + 1))
        row = _feature_row(
            target_date=target,
            history=series,
            static=static,
            lags=bundle.lags,
            rolling_windows=bundle.rolling_windows,
        )
        qty = float(_predict_frame(bundle, [row])[0])
        out.append({"date": target.date().isoformat(), "qty": round(qty, 4)})
        # append prediction so next-day lags see it
        series.loc[target] = qty
    return out


def forecast_batch(
    *,
    item_ids: list[str],
    sales_history: dict[str, list[dict[str, Any]]],
    product_attrs: dict[str, dict[str, Any]],
    as_of: date,
    horizon_days: int,
) -> dict[str, list[dict[str, Any]]]:
    """Recursive forecasts for many SKUs (one vectorized predict per future day)."""
    bundle = _load_bundle()
    if bundle is None or not global_lgbm_enabled():
        return {}
    horizon = max(int(horizon_days), 1)
    ids = [str(i) for i in item_ids]
    series_map = {
        iid: _history_to_series(sales_history.get(iid, []), as_of=as_of) for iid in ids
    }
    static_map = {
        iid: _static_row({**(product_attrs.get(iid) or {}), "product_id": iid}) for iid in ids
    }
    result: dict[str, list[dict[str, Any]]] = {iid: [] for iid in ids}
    for i in range(horizon):
        target = pd.Timestamp(as_of + timedelta(days=i + 1))
        rows = [
            _feature_row(
                target_date=target,
                history=series_map[iid],
                static=static_map[iid],
                lags=bundle.lags,
                rolling_windows=bundle.rolling_windows,
            )
            for iid in ids
        ]
        preds = _predict_frame(bundle, rows)
        date_s = target.date().isoformat()
        for iid, qty in zip(ids, preds):
            q = float(qty)
            result[iid].append({"date": date_s, "qty": round(q, 4)})
            series_map[iid].loc[target] = q
    return result
