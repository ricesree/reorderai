"""Central paths for local data folders + runtime cache."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# All local / export inputs live under data/
DATA_ROOT = PROJECT_ROOT / "data"

SALES_DIR = DATA_ROOT / "sales"
INVENTORY_DIR = DATA_ROOT / "inventory"


def resolve_inventory_path() -> Path:
    """Prefer newest inventory count dump; else products.csv EOD export."""
    candidates = sorted(
        [
            p
            for p in INVENTORY_DIR.glob("*.csv")
            if "inventory" in p.name.lower() and "count" in p.name.lower()
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]
    products = INVENTORY_DIR / "products.csv"
    if products.exists():
        return products
    legacy = INVENTORY_DIR / "current inventory count.csv"
    return legacy


INVENTORY_PATH = resolve_inventory_path()
VENDORS_DIR = DATA_ROOT / "vendors"
PACK_OVERRIDES_PATH = VENDORS_DIR / "pack_overrides.csv"
# Lead time / cover days are dynamic per detect-order request (UI), not a local schedule file.
PAST_INVOICES_DIR = DATA_ROOT / "Past Invoices"
WASTE_DIR = DATA_ROOT / "waste"
SANDBOX_EXPORTS_DIR = DATA_ROOT / "sandbox_exports"
CACHE_DIR = DATA_ROOT / "cache"
ORDER_RUNS_DIR = CACHE_DIR / "order_runs"
FORECAST_STORE_DIR = DATA_ROOT / "forecast_store"
WEATHER_CACHE_DIR = CACHE_DIR
TENANT_MODELS_DIR = DATA_ROOT / "models" / "tenants"
