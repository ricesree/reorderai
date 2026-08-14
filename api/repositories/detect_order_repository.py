"""
DB reads for W-1 detect-order.

Live mode reads Wecomm tenant schema:
  vendors, products, product_vendor (fallback: vendor_order_products),
  product_locations, product_barcodes.
"""

from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd

from database.connectors.wecomm import WecommDatabaseConnector
from database.tenant import get_tenant_schema, q_ident
from v2.forecasting.local_pos_sales import normalize_upc
from v2.invoices.past_invoice_loader import last_pallet_qty_for_items
from config.data_paths import INVENTORY_DIR


def _live_enabled() -> bool:
    flag = os.getenv("DETECT_ORDER_USE_LIVE_SQL", "1").lower()
    return flag in {"1", "true", "yes"}


def _invoice_fallback_enabled() -> bool:
    flag = os.getenv("LAST_PALLET_FROM_INVOICES", "1").lower()
    return flag in {"1", "true", "yes"}


class DetectOrderRepository:
    def __init__(self, tenant_id: str | None = None) -> None:
        self.configured = bool(os.getenv("DB_HOST"))
        self.live = self.configured and _live_enabled()
        self.schema = get_tenant_schema(tenant_id)
        self._db: WecommDatabaseConnector | None = None
        self._vendor_name_cache: dict[str, str] = {}
        self._pack_by_upc: dict[str, int] | None = None

    @property
    def mode(self) -> str:
        return "live" if self.live else "stub"

    def _conn(self) -> WecommDatabaseConnector:
        if self._db is None:
            self._db = WecommDatabaseConnector()
        return self._db

    def list_vendors(self) -> list[dict[str, Any]]:
        if not self.live:
            return [
                {"vendor_id": "V001", "vendor_name": "OM PRODUCE"},
                {"vendor_id": "V002", "vendor_name": "JALARAM"},
                {"vendor_id": "V003", "vendor_name": "DEEP FOODS"},
            ]
        sch = q_ident(self.schema)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT id AS vendor_id, name AS vendor_name
                FROM {sch}.vendors
                WHERE deleted_at IS NULL
                ORDER BY name
                """
            )
        except Exception as exc:
            raise RuntimeError(
                "Cannot read tenant vendors — SSH tunnel may be pointing at the wrong "
                f"Postgres (missing schema {self.schema}). "
                f"Root: {type(exc).__name__}: {exc}"
            ) from exc
        return [
            {"vendor_id": str(int(r.vendor_id)), "vendor_name": str(r.vendor_name)}
            for r in df.itertuples(index=False)
        ]

    def detect_vendor(
        self,
        *,
        vendor_id: str | None = None,
        vendor_name: str | None = None,
    ) -> dict[str, Any] | None:
        vendors = self.list_vendors()
        if vendor_id:
            for v in vendors:
                if str(v["vendor_id"]).upper() == str(vendor_id).strip().upper():
                    return v
        if vendor_name:
            needle = vendor_name.strip().upper()
            for v in vendors:
                name = str(v["vendor_name"]).upper()
                if needle in name or name in needle:
                    return v
        return None

    def fetch_vendor_items(self, vendor_id: str) -> list[dict[str, Any]]:
        """Step 1 — catalog items for vendor."""
        if not self.live:
            return self._stub_items(vendor_id)

        sch = q_ident(self.schema)
        vid = int(vendor_id)

        # Preferred: product_vendor link (has lead_time_days)
        df = self._conn().read_sql(
            f"""
            SELECT
              p.id AS item_id,
              p.sku,
              p.name AS description,
              pv.vendor_id,
              pv.lead_time_days,
              COALESCE(pv.price, p.purchase_price, p.price) AS vendor_price,
              COALESCE(p.min_reorder_quantity, 1) AS box_qty,
              COALESCE(p.min_on_hand, 0) AS product_min_on_hand,
              (
                SELECT pb.barcode
                FROM {sch}.product_barcodes pb
                WHERE pb.product_id = p.id
                ORDER BY CASE WHEN pb.type = 'upc' THEN 0 ELSE 1 END, pb.id
                LIMIT 1
              ) AS upc
            FROM {sch}.product_vendor pv
            JOIN {sch}.products p ON p.id = pv.product_id
            WHERE pv.vendor_id = :vendor_id
              AND p.deleted_at IS NULL
              AND COALESCE(p.is_active, TRUE) = TRUE
            ORDER BY p.name
            """,
            {"vendor_id": vid},
        )

        source = "product_vendor"
        if df.empty:
            # Fallback while product_vendor is empty: items seen on that vendor's POs
            source = "vendor_order_products"
            df = self._conn().read_sql(
                f"""
                SELECT
                  p.id AS item_id,
                  p.sku,
                  p.name AS description,
                  vo.vendor_id,
                  NULL::integer AS lead_time_days,
                  COALESCE(p.purchase_price, p.price) AS vendor_price,
                  COALESCE(p.min_reorder_quantity, 1) AS box_qty,
                  COALESCE(p.min_on_hand, 0) AS product_min_on_hand,
                  (
                    SELECT pb.barcode
                    FROM {sch}.product_barcodes pb
                    WHERE pb.product_id = p.id
                    ORDER BY CASE WHEN pb.type = 'upc' THEN 0 ELSE 1 END, pb.id
                    LIMIT 1
                  ) AS upc
                FROM {sch}.vendor_order_products vop
                JOIN {sch}.vendor_orders vo ON vo.id = vop.vendor_order_id
                JOIN {sch}.products p ON p.id = vop.product_id
                WHERE vo.vendor_id = :vendor_id
                  AND vop.deleted_at IS NULL
                  AND p.deleted_at IS NULL
                GROUP BY p.id, p.sku, p.name, vo.vendor_id, p.purchase_price, p.price,
                         p.min_reorder_quantity, p.min_on_hand
                ORDER BY p.name
                """,
                {"vendor_id": vid},
            )

        item_ids = [int(x) for x in df["item_id"].tolist()] if not df.empty else []
        expiry_map = self._fetch_expiration_days(item_ids)
        pallet_map = self._fetch_last_pallet_qty(item_ids, vid)
        loc_min_map = self._fetch_location_min_quantity(item_ids)
        loc_max_map = self._fetch_location_max_quantity(item_ids)

        items: list[dict[str, Any]] = []
        for r in df.itertuples(index=False):
            iid = str(int(r.item_id))
            product_min = float(r.product_min_on_hand or 0)
            location_min = float(loc_min_map.get(iid, 0.0))
            wecomm_min = max(product_min, location_min)
            wecomm_max = float(loc_max_map.get(iid, 0.0))
            items.append(
                {
                    "item_id": iid,
                    "upc": str(r.upc) if r.upc is not None else None,
                    "sku": str(r.sku) if r.sku is not None else None,
                    "description": str(r.description or ""),
                    "vendor_id": str(int(r.vendor_id)),
                    "vendor_price": float(r.vendor_price) if r.vendor_price is not None else None,
                    "demand_class": None,
                    "box_qty": max(int(r.box_qty or 1), 1),
                    "expiration_days_remaining": expiry_map.get(iid),
                    "last_pallet_qty": pallet_map.get(iid),
                    "lead_time_days": int(r.lead_time_days)
                    if r.lead_time_days is not None
                    else None,
                    "catalog_source": source,
                    "wecomm_min_on_hand": wecomm_min if wecomm_min > 0 else 0.0,
                    "wecomm_max_on_hand": wecomm_max if wecomm_max > 0 else 0.0,
                }
            )

        # Paul rarely has vendor POs; fill gaps from Past Invoices workbook.
        if _invoice_fallback_enabled():
            missing = [it for it in items if it.get("last_pallet_qty") is None]
            if missing:
                try:
                    vname = self._vendor_name(str(vid))
                    inv_map = last_pallet_qty_for_items(missing, vendor_name=vname)
                    for it in items:
                        if it.get("last_pallet_qty") is None and it["item_id"] in inv_map:
                            it["last_pallet_qty"] = inv_map[it["item_id"]]
                except Exception:
                    # Never fail detect-order if invoice parse/match blows up.
                    pass

        # Prefer real case pack from local products.csv when DB min_reorder is 1/missing.
        self._enrich_pack_sizes(items)
        return items

    def _local_pack_by_upc(self) -> dict[str, int]:
        if self._pack_by_upc is not None:
            return self._pack_by_upc
        path = INVENTORY_DIR / "products.csv"
        out: dict[str, int] = {}
        if path.exists():
            try:
                df = pd.read_csv(path, dtype=str)
                for _, r in df.iterrows():
                    upc = normalize_upc(r.get("upc"))
                    pack = pd.to_numeric(r.get("pack"), errors="coerce")
                    if upc and pd.notna(pack) and float(pack) > 1:
                        out[upc] = int(float(pack))
            except Exception:
                out = {}
        self._pack_by_upc = out
        return out

    def _enrich_pack_sizes(self, items: list[dict[str, Any]]) -> None:
        packs = self._local_pack_by_upc()
        if not packs:
            return
        for it in items:
            if int(it.get("box_qty") or 1) > 1:
                continue
            upc_n = normalize_upc(it.get("upc"))
            if upc_n and upc_n in packs:
                it["box_qty"] = packs[upc_n]

    def _fetch_expiration_days(self, item_ids: list[int]) -> dict[str, float]:
        """Soonest batch expiration → days remaining (Step 5)."""
        if not item_ids:
            return {}
        sch = q_ident(self.schema)
        id_csv = ",".join(str(i) for i in item_ids)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT
                  product_id,
                  MIN(expiration_date) AS soonest_exp
                FROM {sch}.product_batches
                WHERE product_id IN ({id_csv})
                  AND remaining_quantity > 0
                  AND expiration_date IS NOT NULL
                GROUP BY product_id
                """
            )
        except Exception:
            return {}
        out: dict[str, float] = {}
        today = pd.Timestamp.utcnow().normalize()
        for r in df.itertuples(index=False):
            exp = pd.to_datetime(r.soonest_exp, errors="coerce")
            if pd.isna(exp):
                continue
            days = (exp.normalize() - today).days
            out[str(int(r.product_id))] = float(max(days, 0))
        return out

    def _vendor_name(self, vendor_id: str) -> str | None:
        key = str(vendor_id)
        if key in self._vendor_name_cache:
            return self._vendor_name_cache[key]
        for v in self.list_vendors():
            self._vendor_name_cache[str(v["vendor_id"])] = str(v["vendor_name"])
        return self._vendor_name_cache.get(key)

    def _fetch_last_pallet_qty(self, item_ids: list[int], vendor_id: int) -> dict[str, float]:
        """Latest vendor_order_products.quantity for reference (Step 5)."""
        if not item_ids:
            return {}
        sch = q_ident(self.schema)
        id_csv = ",".join(str(i) for i in item_ids)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT DISTINCT ON (vop.product_id)
                  vop.product_id,
                  vop.quantity
                FROM {sch}.vendor_order_products vop
                JOIN {sch}.vendor_orders vo ON vo.id = vop.vendor_order_id
                WHERE vo.vendor_id = :vendor_id
                  AND vop.product_id IN ({id_csv})
                  AND vop.deleted_at IS NULL
                ORDER BY vop.product_id, vop.created_at DESC NULLS LAST, vop.id DESC
                """,
                {"vendor_id": vendor_id},
            )
        except Exception:
            return {}
        return {
            str(int(r.product_id)): float(r.quantity)
            for r in df.itertuples(index=False)
            if r.quantity is not None
        }

    def fetch_all_vendor_prices(self, item_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
        """All vendor offers (vendor_id, vendor_name, price) per product_id — for the
        'cheaper by other vendor' flag. Keyed by item_id (str)."""
        if not self.live or not item_ids:
            return {}

        sch = q_ident(self.schema)
        ids = [int(x) for x in item_ids]
        id_csv = ",".join(str(i) for i in ids)
        df = self._conn().read_sql(
            f"""
            SELECT
              pv.product_id,
              pv.vendor_id,
              v.name AS vendor_name,
              COALESCE(pv.price, p.purchase_price, p.price) AS price
            FROM {sch}.product_vendor pv
            JOIN {sch}.vendors v ON v.id = pv.vendor_id AND v.deleted_at IS NULL
            JOIN {sch}.products p ON p.id = pv.product_id
            WHERE pv.product_id IN ({id_csv})
            """
        )
        out: dict[str, list[dict[str, Any]]] = {}
        for r in df.itertuples(index=False):
            if r.price is None:
                continue
            pid = str(int(r.product_id))
            out.setdefault(pid, []).append(
                {
                    "vendor_id": str(int(r.vendor_id)),
                    "vendor_name": str(r.vendor_name),
                    "price": float(r.price),
                }
            )
        return out

    # Location types that don't count as sellable on-hand stock (mirrors the main
    # app's excludeReturnsLocations/excludeVirtualLocations/excludePackingLocations
    # + scrape scopes).
    _EXCLUDED_LOCATION_TYPES = ("returns", "virtual", "scrape", "packing")

    def fetch_available_stock(self, item_ids: list[str]) -> dict[str, float]:
        """Step 2 — sum on-hand qty from product_locations (raw; negatives allowed).

        Excludes returns / virtual / scrape / packing locations (not sellable stock).
        Detect-order keeps negatives on the line (oversold). Order math floors at 0
        and counts |OH| toward sold / ADS.
        """
        if not self.live:
            stock = {
                "I1001": 12.0,
                "I1002": 35.0,
                "I1003": 5.0,
                "I2001": 10.0,
                "I3001": 8.0,
            }
            return {i: float(stock.get(i, 0.0)) for i in item_ids}

        if not item_ids:
            return {}

        sch = q_ident(self.schema)
        ids = [int(x) for x in item_ids]
        id_csv = ",".join(str(i) for i in ids)
        excluded_csv = ",".join(f"'{t}'" for t in self._EXCLUDED_LOCATION_TYPES)
        df = self._conn().read_sql(
            f"""
            SELECT product_id, COALESCE(SUM(quantity), 0) AS qty
            FROM {sch}.product_locations
            WHERE deleted_at IS NULL
              AND product_id IN ({id_csv})
              AND COALESCE(location_type, '') NOT IN ({excluded_csv})
            GROUP BY product_id
            """
        )
        found = {str(int(r.product_id)): float(r.qty) for r in df.itertuples(index=False)}
        return {i: float(found.get(i, 0.0)) for i in item_ids}

    def _fetch_location_min_quantity(self, item_ids: list[int]) -> dict[str, float]:
        """Sum of location min_quantity per product across all locations (Wecomm shelf/warehouse min)."""
        if not item_ids:
            return {}
        sch = q_ident(self.schema)
        id_csv = ",".join(str(i) for i in item_ids)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT product_id, COALESCE(SUM(min_quantity), 0) AS min_qty
                FROM {sch}.product_locations
                WHERE deleted_at IS NULL
                  AND product_id IN ({id_csv})
                GROUP BY product_id
                """
            )
        except Exception:
            return {}
        return {
            str(int(r.product_id)): float(r.min_qty)
            for r in df.itertuples(index=False)
            if r.min_qty is not None and float(r.min_qty) > 0
        }

    def _fetch_location_max_quantity(self, item_ids: list[int]) -> dict[str, float]:
        """Sum of location max_quantity per product, storage + picking locations only

        (anti-overstock cap — shown in the API/UI as stock_capacity / wecomm_max_on_hand).
        """
        if not item_ids:
            return {}
        sch = q_ident(self.schema)
        id_csv = ",".join(str(i) for i in item_ids)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT product_id, SUM(max_quantity) AS max_qty
                FROM {sch}.product_locations
                WHERE deleted_at IS NULL
                  AND product_id IN ({id_csv})
                  AND max_quantity IS NOT NULL
                  AND max_quantity > 0
                  AND location_type IN ('storage', 'picking')
                GROUP BY product_id
                """
            )
        except Exception:
            return {}
        return {
            str(int(r.product_id)): float(r.max_qty)
            for r in df.itertuples(index=False)
            if r.max_qty is not None and float(r.max_qty) > 0
        }

    def fetch_all_products_on_hand(self) -> list[dict[str, Any]]:
        """All active products with barcode + total on-hand — for same-item brand matching."""
        if not self.live:
            return []
        sch = q_ident(self.schema)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT
                  p.id::text AS item_id,
                  p.name AS description,
                  p.product_name AS product_name,
                  (
                    SELECT pb.barcode
                    FROM {sch}.product_barcodes pb
                    WHERE pb.product_id = p.id
                    ORDER BY pb.id
                    LIMIT 1
                  ) AS upc,
                  COALESCE((
                    SELECT SUM(pl.quantity)
                    FROM {sch}.product_locations pl
                    WHERE pl.product_id = p.id
                      AND pl.deleted_at IS NULL
                      AND COALESCE(pl.location_type, '') NOT IN ('returns', 'virtual', 'scrape', 'packing')
                  ), 0) AS quantity
                FROM {sch}.products p
                WHERE p.deleted_at IS NULL
                  AND COALESCE(p.is_active, TRUE) = TRUE
                """
            )
        except Exception:
            return []
        if df.empty:
            return []
        return [
            {
                "item_id": str(r.item_id),
                "description": str(r.description or ""),
                "product_name": str(r.product_name or "") if r.product_name is not None else "",
                "upc": str(r.upc or "") if r.upc is not None else "",
                "quantity": float(r.quantity or 0.0),
            }
            for r in df.itertuples(index=False)
        ]

    def fetch_product_forecast_attrs(
        self, item_ids: list[str], *, as_of: date | None = None
    ) -> dict[str, dict[str, Any]]:
        """Static product features for global LightGBM inference."""
        if not self.live or not item_ids:
            return {}
        as_of_d = as_of or date.today()
        sch = q_ident(self.schema)
        ids = [int(x) for x in item_ids]
        id_csv = ",".join(str(i) for i in ids)
        try:
            df = self._conn().read_sql(
                f"""
                SELECT
                  p.id::text AS product_id,
                  p.category_id::text AS category_id,
                  COALESCE(p.price, 0) AS list_price,
                  COALESCE(p.purchase_price, 0) AS purchase_price,
                  COALESCE(p.scale, FALSE) AS is_scale,
                  COALESCE(NULLIF(p.min_reorder_quantity, 0), 1) AS pack_size,
                  p.created_at::date AS created_on
                FROM {sch}.products p
                WHERE p.id IN ({id_csv})
                """
            )
        except Exception:
            return {}
        out: dict[str, dict[str, Any]] = {}
        for r in df.itertuples(index=False):
            created = getattr(r, "created_on", None)
            age = 0
            if created is not None and not pd.isna(created):
                try:
                    age = max((as_of_d - pd.Timestamp(created).date()).days, 0)
                except Exception:
                    age = 0
            out[str(r.product_id)] = {
                "product_id": str(r.product_id),
                "category_id": str(r.category_id) if r.category_id is not None else "",
                "list_price": float(r.list_price or 0.0),
                "purchase_price": float(r.purchase_price or 0.0),
                "is_scale": bool(r.is_scale),
                "pack_size": float(r.pack_size or 1.0),
                "any_discount_flag": False,
                "product_age_days": float(age),
                "price_change_percent": 0.0,
            }
        return out

    def _stub_items(self, vendor_id: str) -> list[dict[str, Any]]:
        catalog = {
            "V001": [
                {
                    "item_id": "I1001",
                    "upc": "0000000000042",
                    "sku": "OKRA-25",
                    "description": "CHINESE OKRA 25-30 LB",
                    "vendor_id": "V001",
                    "demand_class": "intermittent",
                    "box_qty": 25,
                    "expiration_days_remaining": 10,
                    "last_pallet_qty": 50,
                    "wecomm_min_on_hand": 0.0,
                    "wecomm_max_on_hand": 0.0,
                },
                {
                    "item_id": "I1002",
                    "upc": "0000000000043",
                    "sku": "GUVAR-20",
                    "description": "GUVAR BEANS 20 LB",
                    "vendor_id": "V001",
                    "demand_class": "smooth",
                    "box_qty": 20,
                    "expiration_days_remaining": 45,
                    "last_pallet_qty": 40,
                    "wecomm_min_on_hand": 30.0,
                    "wecomm_max_on_hand": 80.0,
                },
            ],
            "V002": [
                {
                    "item_id": "I2001",
                    "upc": "8901030865482",
                    "sku": "AMUL-BUTTER",
                    "description": "AMUL BUTTER 500G",
                    "vendor_id": "V002",
                    "demand_class": "smooth",
                    "box_qty": 12,
                    "expiration_days_remaining": 60,
                    "last_pallet_qty": 48,
                    "wecomm_min_on_hand": 0.0,
                    "wecomm_max_on_hand": 0.0,
                },
            ],
        }
        return list(catalog.get(str(vendor_id).upper(), []))
