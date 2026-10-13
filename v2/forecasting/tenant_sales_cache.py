"""Incremental per-tenant daily sales cache (used by horizon_total_model).

First call pulls the tenant's full sales history from order_items/orders once.
Later calls re-fetch only the last SALES_CACHE_REFETCH_DAYS days (catches late
returns / edited orders) and replace that tail, so retraining doesn't re-query
years of orders every time.

Stored per tenant via model_storage (Azure Blob, else local disk):
  tenants/<schema>/sales_cache/daily_sales.parquet
Sparse rows only (product_id, date, target_sales) for days with sales —
aggregated daily quantities, no customer/order detail. Zero-filling and
product attributes are applied in memory by load_tenant_sales_panel().
"""

from __future__ import annotations

import io
import logging
import os
import threading
from datetime import date

import pandas as pd

from config.data_paths import TENANT_MODELS_DIR
from database.connectors.wecomm import WecommDatabaseConnector
from database.tenant import q_ident
from v2.forecasting import model_storage
from v2.forecasting.tenant_training import _is_perishable, schema_for_tenant

logger = logging.getLogger(__name__)

REFETCH_DAYS = int(os.getenv("SALES_CACHE_REFETCH_DAYS", "7"))
CACHE_COLS = ["product_id", "date", "target_sales"]

_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(tenant_id: str) -> threading.Lock:
    with _locks_guard:
        if tenant_id not in _locks:
            _locks[tenant_id] = threading.Lock()
        return _locks[tenant_id]


def _cache_path(tenant_id: str):
    return TENANT_MODELS_DIR / tenant_id / "sales_cache" / "daily_sales.parquet"


def _blob_key(tenant_id: str) -> str:
    return f"tenants/{tenant_id}/sales_cache/daily_sales.parquet"


def fetch_daily_sales_since(
    tenant_id: str,
    since: date | None,
    *,
    connector: WecommDatabaseConnector | None = None,
) -> pd.DataFrame:
    """Sparse per-(product, day) net sales from order_items; all history when since is None."""
    db = connector or WecommDatabaseConnector()
    sch = q_ident(schema_for_tenant(tenant_id))
    since_clause = "AND o.created_at::date >= :since" if since is not None else ""
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
          {since_clause}
        GROUP BY oi.product_id, o.created_at::date
        """,
        {"since": since} if since is not None else None,
    )
    if df.empty:
        return pd.DataFrame(columns=CACHE_COLS)
    out = pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str),
            "date": pd.to_datetime(df["sale_date"]).dt.normalize(),
            "target_sales": pd.to_numeric(df["target_sales"], errors="coerce").fillna(0.0),
        }
    )
    return out[out["target_sales"] > 0].reset_index(drop=True)


def fetch_all_products(
    tenant_id: str, *, connector: WecommDatabaseConnector | None = None
) -> pd.DataFrame:
    """Current static attrs for every live product — only columns the DB actually has."""
    db = connector or WecommDatabaseConnector()
    sch = q_ident(schema_for_tenant(tenant_id))
    df = db.read_sql(
        f"""
        SELECT
          p.id AS product_id, p.category_id, c.name AS category_name,
          p.price AS list_price,
          COALESCE(p.scale, FALSE) AS is_scale,
          COALESCE(NULLIF(p.min_reorder_quantity, 0), 1) AS pack_size
        FROM {sch}.products p
        LEFT JOIN {sch}.categories c ON c.id = p.category_id AND c.deleted_at IS NULL
        WHERE p.deleted_at IS NULL
        """
    )
    cols = ["product_id", "category_id", "list_price", "pack_size", "is_scale", "is_perishable"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    return pd.DataFrame(
        {
            "product_id": df["product_id"].astype(str),
            "category_id": df["category_id"].apply(lambda x: str(int(x)) if pd.notna(x) else ""),
            "list_price": pd.to_numeric(df["list_price"], errors="coerce").fillna(0.0),
            "pack_size": pd.to_numeric(df["pack_size"], errors="coerce").fillna(1.0),
            "is_scale": df["is_scale"].astype(bool).astype(int),
            "is_perishable": df["category_name"].apply(_is_perishable),
        }
    )


def _read_cache(tenant_id: str) -> pd.DataFrame | None:
    try:
        raw = model_storage.read_bytes(_blob_key(tenant_id), _cache_path(tenant_id))
        if raw is None:
            return None
        df = pd.read_parquet(io.BytesIO(raw))
        if list(df.columns) != CACHE_COLS:
            raise ValueError(f"unexpected cache columns {list(df.columns)}")
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
        return df
    except Exception:
        # Corrupt/old-format cache must never block training — rebuild from scratch.
        logger.warning("tenant %s: sales cache unreadable — full refetch", tenant_id, exc_info=True)
        return None


def _write_cache(tenant_id: str, df: pd.DataFrame) -> None:
    buf = io.BytesIO()
    df[CACHE_COLS].to_parquet(buf, index=False)
    model_storage.write_bytes(_blob_key(tenant_id), _cache_path(tenant_id), buf.getvalue())


def update_sales_cache(
    tenant_id: str, *, connector: WecommDatabaseConnector | None = None
) -> pd.DataFrame:
    """Bring the cache up to date (full pull first time, tail refetch after) and return it."""
    with _lock_for(tenant_id):
        cached = _read_cache(tenant_id)
        if cached is None or cached.empty:
            since = None
        else:
            since = (cached["date"].max() - pd.Timedelta(days=REFETCH_DAYS)).date()
        fresh = fetch_daily_sales_since(tenant_id, since, connector=connector)
        if since is None:
            merged = fresh
        else:
            # Replace (not append) the refetched tail so edited/returned orders aren't double-counted.
            keep = cached[cached["date"] < pd.Timestamp(since)]
            merged = pd.concat([keep, fresh], ignore_index=True)
        merged = merged.sort_values(["product_id", "date"]).reset_index(drop=True)
        _write_cache(tenant_id, merged)
        logger.info(
            "tenant %s: sales cache %s — fetched %d row(s), %d cached",
            tenant_id, "full pull" if since is None else f"refetch since {since}", len(fresh), len(merged),
        )
        return merged


def load_tenant_sales_panel(
    tenant_id: str, *, connector: WecommDatabaseConnector | None = None
) -> pd.DataFrame:
    """Dense zero-filled (product × day) panel with current static attrs, for training."""
    sparse = update_sales_cache(tenant_id, connector=connector)
    products = fetch_all_products(tenant_id, connector=connector)
    if sparse.empty or products.empty:
        return pd.DataFrame(columns=["date", "target_sales", *products.columns])
    dates = pd.date_range(sparse["date"].min(), sparse["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [products["product_id"], dates], names=["product_id", "date"]
    ).to_frame(index=False)
    grid = grid.merge(sparse, on=["product_id", "date"], how="left")
    grid["target_sales"] = grid["target_sales"].fillna(0.0)
    return grid.merge(products, on="product_id", how="left")
