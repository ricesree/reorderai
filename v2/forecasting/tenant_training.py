"""Per-tenant training data (order_items + orders + products) and shared
train/predict feature engineering for tenant_sales_model.

All SQL here is read-only and scoped to one tenant schema (wecomm_<tenant_id>).
Nothing is written back to the database or to disk — callers hold the
returned DataFrame only for the duration of training/prediction.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from database.connectors.wecomm import WecommDatabaseConnector
from database.tenant import q_ident

logger = logging.getLogger(__name__)

LAGS = [1, 7, 14, 28]
ROLLING_WINDOWS = [7, 14, 28]
LAG_COLS = [f"lag_{n}" for n in LAGS]
ROLLING_COLS = [f"rolling_mean_{w}" for w in ROLLING_WINDOWS] + [f"rolling_std_{w}" for w in ROLLING_WINDOWS]
CALENDAR_COLS = ["dow", "is_weekend", "day_of_month", "is_month_start", "month"]
STATIC_COLS = ["product_id", "category_id", "list_price", "pack_size", "is_scale", "is_perishable", "any_discount_flag"]
FEATURE_COLS = STATIC_COLS + CALENDAR_COLS + LAG_COLS + ROLLING_COLS
CATEGORICAL_COLS = ["product_id", "category_id"]

PERISHABLE_KEYWORDS = (
    "produce", "dairy", "bakery", "bread", "meat", "seafood", "deli", "frozen", "floral", "flower",
)


def schema_for_tenant(tenant_id: str) -> str:
    """Frontend sends the bare tenant_id; the DB schema is wecomm_<tenant_id>."""
    tid = tenant_id.strip().strip('"').strip("'")
    return tid if tid.startswith("wecomm_") else f"wecomm_{tid}"


def _is_perishable(category_name: str | None) -> int:
    name = (category_name or "").lower()
    return int(any(kw in name for kw in PERISHABLE_KEYWORDS))


def fetch_tenant_daily_sales(
    tenant_id: str,
    *,
    connector: WecommDatabaseConnector | None = None,
) -> pd.DataFrame:
    """Full daily per-product sales panel for training — one row per (product, day)
    since the tenant's first order, zero-filled on days with no sales.
    """
    schema = schema_for_tenant(tenant_id)
    logger.info("tenant %s: querying orders/order_items/products (schema %s)", tenant_id, schema)
    db = connector or WecommDatabaseConnector()
    sch = q_ident(schema)

    df = db.read_sql(
        f"""
        WITH bounds AS (
            SELECT MIN(o.created_at)::date AS d0, MAX(o.created_at)::date AS d1
            FROM {sch}.orders o
            WHERE o.deleted_at IS NULL AND COALESCE(o.is_return, FALSE) = FALSE
        ),
        dates AS (
            SELECT generate_series(d0, d1, INTERVAL '1 day')::date AS sale_date
            FROM bounds
        ),
        products AS (
            SELECT
                p.id AS product_id,
                p.category_id,
                c.name AS category_name,
                p.price AS list_price,
                p.purchase_price,
                COALESCE(p.scale, FALSE) AS is_scale,
                COALESCE(NULLIF(p.min_reorder_quantity, 0), 1) AS pack_size
            FROM {sch}.products p
            LEFT JOIN {sch}.categories c ON c.id = p.category_id AND c.deleted_at IS NULL
            WHERE p.deleted_at IS NULL
        ),
        daily_sales AS (
            SELECT
                o.created_at::date AS sale_date,
                oi.product_id,
                SUM(GREATEST(COALESCE(oi.quantity, 0) - COALESCE(oi.returned_quantity, 0), 0)) AS target_sales,
                BOOL_OR(
                    COALESCE(oi.discount_amount, 0) > 0
                    OR COALESCE(oi.total_allocated_discount, 0) > 0
                    OR COALESCE(oi.total_allocated_promotion_discount, 0) > 0
                ) AS any_discount_flag
            FROM {sch}.order_items oi
            JOIN {sch}.orders o ON o.id = oi.order_id
            WHERE o.deleted_at IS NULL
              AND oi.deleted_at IS NULL
              AND COALESCE(o.is_return, FALSE) = FALSE
            GROUP BY o.created_at::date, oi.product_id
        )
        SELECT
            d.sale_date, p.product_id, p.category_id, p.category_name,
            p.list_price, p.purchase_price, p.is_scale, p.pack_size,
            COALESCE(s.target_sales, 0) AS target_sales,
            COALESCE(s.any_discount_flag, FALSE) AS any_discount_flag
        FROM dates d
        CROSS JOIN products p
        LEFT JOIN daily_sales s ON s.sale_date = d.sale_date AND s.product_id = p.product_id
        ORDER BY p.product_id, d.sale_date
        """
    )
    empty_cols = [
        "date", "product_id", "category_id", "category_name", "list_price",
        "pack_size", "is_scale", "is_perishable", "any_discount_flag", "target_sales",
    ]
    if df.empty:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame(
        {
            "date": pd.to_datetime(df["sale_date"]),
            "product_id": df["product_id"].astype(str),
            "category_id": df["category_id"].apply(lambda x: str(int(x)) if pd.notna(x) else ""),
            "category_name": df["category_name"].fillna(""),
            "list_price": pd.to_numeric(df["list_price"], errors="coerce").fillna(0.0),
            "pack_size": pd.to_numeric(df["pack_size"], errors="coerce").fillna(1.0),
            "is_scale": df["is_scale"].astype(bool).astype(int),
            "any_discount_flag": df["any_discount_flag"].astype(bool).astype(int),
            "target_sales": pd.to_numeric(df["target_sales"], errors="coerce").fillna(0.0),
        }
    )
    out["is_perishable"] = df["category_name"].apply(_is_perishable)
    return out


def fetch_product_attrs(
    tenant_id: str,
    item_ids: list[str],
    *,
    connector: WecommDatabaseConnector | None = None,
) -> dict[str, dict[str, Any]]:
    """Current static attrs for a specific set of items — predict-time only."""
    if not item_ids:
        return {}
    db = connector or WecommDatabaseConnector()
    sch = q_ident(schema_for_tenant(tenant_id))
    ids = [int(x) for x in item_ids]
    df = db.read_sql(
        f"""
        SELECT
          p.id AS product_id, p.category_id, c.name AS category_name,
          p.price AS list_price,
          COALESCE(p.scale, FALSE) AS is_scale,
          COALESCE(NULLIF(p.min_reorder_quantity, 0), 1) AS pack_size
        FROM {sch}.products p
        LEFT JOIN {sch}.categories c ON c.id = p.category_id AND c.deleted_at IS NULL
        WHERE p.deleted_at IS NULL AND p.id = ANY(:ids)
        """,
        {"ids": ids},
    )
    out: dict[str, dict[str, Any]] = {}
    for r in df.itertuples(index=False):
        iid = str(int(r.product_id))
        out[iid] = {
            "product_id": iid,
            "category_id": str(int(r.category_id)) if pd.notna(r.category_id) else "",
            "list_price": float(r.list_price or 0.0),
            "pack_size": float(r.pack_size or 1.0),
            "is_scale": bool(r.is_scale),
            "is_perishable": _is_perishable(r.category_name),
        }
    return out


def fetch_recent_daily_sales(
    tenant_id: str,
    item_ids: list[str],
    *,
    lookback_days: int = 40,
    connector: WecommDatabaseConnector | None = None,
) -> pd.DataFrame:
    """Recent per-item daily qty — just enough history to seed lag/rolling features at predict time."""
    if not item_ids:
        return pd.DataFrame(columns=["product_id", "date", "target_sales"])
    db = connector or WecommDatabaseConnector()
    sch = q_ident(schema_for_tenant(tenant_id))
    ids = [int(x) for x in item_ids]
    df = db.read_sql(
        f"""
        SELECT
          oi.product_id,
          o.created_at::date AS sale_date,
          SUM(GREATEST(COALESCE(oi.quantity, 0) - COALESCE(oi.returned_quantity, 0), 0)) AS target_sales
        FROM {sch}.order_items oi
        JOIN {sch}.orders o ON o.id = oi.order_id
        WHERE o.deleted_at IS NULL
          AND oi.deleted_at IS NULL
          AND COALESCE(o.is_return, FALSE) = FALSE
          AND oi.product_id = ANY(:ids)
          AND o.created_at >= (NOW() - INTERVAL '{int(lookback_days)} days')
        GROUP BY oi.product_id, o.created_at::date
        """,
        {"ids": ids},
    )
    if df.empty:
        return pd.DataFrame(columns=["product_id", "date", "target_sales"])
    return pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str),
            "date": pd.to_datetime(df["sale_date"]),
            "target_sales": pd.to_numeric(df["target_sales"], errors="coerce").fillna(0.0),
        }
    )


# --- shared feature engineering: used by both training and recursive prediction ---


def _postgres_dow(d: pd.Timestamp) -> int:
    """Pandas Monday=0..Sunday=6 -> Postgres EXTRACT(DOW) Sunday=0..Saturday=6."""
    return int((d.dayofweek + 1) % 7)


def build_training_frame(daily: pd.DataFrame) -> pd.DataFrame:
    """Add calendar + lag/rolling features; drop rows without a full lag_28 window."""
    if daily.empty:
        return daily
    df = daily.sort_values(["product_id", "date"]).reset_index(drop=True)
    dow = df["date"].dt.dayofweek
    df["dow"] = (dow + 1) % 7
    df["is_weekend"] = df["dow"].isin([0, 6]).astype(int)
    df["day_of_month"] = df["date"].dt.day
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["month"] = df["date"].dt.month

    g = df.groupby("product_id", sort=False)["target_sales"]
    for lag in LAGS:
        df[f"lag_{lag}"] = g.shift(lag)
    for w in ROLLING_WINDOWS:
        df[f"rolling_mean_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
        df[f"rolling_std_{w}"] = g.transform(lambda s: s.shift(1).rolling(w, min_periods=2).std())
        df[f"rolling_std_{w}"] = df[f"rolling_std_{w}"].fillna(0.0)

    return df.dropna(subset=LAG_COLS).reset_index(drop=True)


def predict_feature_row(
    *, target_date: pd.Timestamp, history: pd.Series, static: dict[str, Any]
) -> dict[str, Any]:
    """One feature row for a single future day, recursed day-by-day by the caller."""
    vals: dict[str, Any] = {
        "product_id": str(static.get("product_id") or ""),
        "category_id": str(static.get("category_id") or ""),
        "list_price": float(static.get("list_price") or 0.0),
        "pack_size": float(static.get("pack_size") or 1.0),
        "is_scale": int(bool(static.get("is_scale"))),
        "is_perishable": int(bool(static.get("is_perishable"))),
        "any_discount_flag": int(bool(static.get("any_discount_flag"))),
    }
    d = target_date
    dow = _postgres_dow(d)
    vals["dow"] = dow
    vals["is_weekend"] = int(dow in (0, 6))
    vals["day_of_month"] = int(d.day)
    vals["is_month_start"] = int(d.is_month_start)
    vals["month"] = int(d.month)
    for lag in LAGS:
        past = d - pd.Timedelta(days=lag)
        vals[f"lag_{lag}"] = float(history.loc[past]) if past in history.index else 0.0
    hist_before = history[history.index < d]
    for w in ROLLING_WINDOWS:
        window = hist_before.tail(w)
        vals[f"rolling_mean_{w}"] = float(window.mean()) if len(window) else 0.0
        vals[f"rolling_std_{w}"] = float(window.std(ddof=0)) if len(window) > 1 else 0.0
    return vals
