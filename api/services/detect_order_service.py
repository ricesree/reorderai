"""
W-1 Detect Order — vendor reorder sheet.

Select vendor + L + C → for each SKU:
  ADS, safety stock, ROP (AI min), uplifted P50/P90,
  order-up-to target, units + cases.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import numpy as np

from api.repositories.detect_order_repository import DetectOrderRepository
from api.repositories.forecast_store import ForecastStore
from api.schemas.detect_order import (
    DetectOrderItem,
    DetectOrderRequest,
    DetectOrderResponse,
    SalesSeries,
    VendorInfo,
)
from api.services.order_run_store import new_run_id, save_order_run
from api.services.reorder_engine import compute_line_reorder
from v2.forecasting.festival_calendar import (
    format_festivals_for_display,
    festivals_in_horizon,
    reorder_as_of_date,
)
from v2.forecasting.global_lightgbm_predictor import (
    forecast_batch as lgbm_forecast_batch,
    model_ready as lgbm_model_ready,
)
from v2.products.same_item_brands import build_other_brands_map, build_same_product_name_map


def _ads_lookback_days() -> float:
    try:
        return max(float(os.getenv("ADS_LOOKBACK_DAYS", "90")), 1.0)
    except (TypeError, ValueError):
        return 90.0


def risk_to_service_level(risk_factor: int) -> float:
    """Map risk 0-100 -> target percentile: 0->P50, 50->P75, 100->P100.

    P100 needs infinite safety stock under a normal model, so it's capped at
    P99.9 (see calculate_safety_stock).
    """
    r = max(0, min(100, int(risk_factor)))
    percentile = min(50.0 + r / 2.0, 99.9)
    return round(percentile / 100.0, 4)


def _forecast_series(
    store: ForecastStore,
    item_id: str,
    alt_ids: list[str],
    ads: float,
    x_days: int,
    as_of: date,
    uplift_types: list[str] | None,
    *,
    lgbm_series: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Prefer global LightGBM daily path; else ADS × per-day learned uplift."""
    if lgbm_series:
        return lgbm_series
    ads_f = max(float(ads), 0.0)
    dates = [as_of + timedelta(days=i + 1) for i in range(max(int(x_days), 1))]
    multipliers = store.daily_uplift_multipliers(
        item_id, dates, alt_ids=alt_ids, allowed_types=uplift_types
    )
    if ads_f <= 0:
        return [{"date": d.isoformat(), "qty": 0.0} for d in dates]

    history = store.get_daily_series(item_id).to_numpy(dtype=float)
    if history.size == 0:
        history = np.array([ads_f])
    # ponytail: seeded by item_id so the chart is stable across repeated requests,
    # not a fresh random draw every refresh — upgrade to a proper simulation model
    # (e.g. Croston/TSB per-day) if the business wants a "most likely path" instead.
    rng = np.random.default_rng(abs(hash(str(item_id))) % (2**32))
    draws = rng.choice(history, size=len(dates), replace=True)
    return [
        {"date": d.isoformat(), "qty": float(round(q * m))}
        for d, q, m in zip(dates, draws, multipliers)
    ]


def _vendors(repo: DetectOrderRepository) -> list[VendorInfo]:
    return [
        VendorInfo(vendor_id=str(v["vendor_id"]), vendor_name=str(v["vendor_name"]))
        for v in repo.list_vendors()
    ]


def _template_justification(
    item: DetectOrderItem,
    *,
    lead: int,
    cover: int,
    as_of: str,
) -> str:
    """Report-style why-order text from computed outputs only (no GPT)."""
    name = item.description or item.item_id
    parts: list[str] = []

    if item.line_action == "ORDER":
        parts.append(
            f"ORDER: buy {item.qty_to_order:g} units "
            f"({item.cases_to_order:g} case(s) x pack {item.box_qty}) of {name}."
        )
    elif item.line_action == "WATCH":
        parts.append(
            f"WATCH: {name} is below the reorder point, but stock at arrival "
            f"already covers the next {cover} day(s) — no buy now."
        )
    else:
        parts.append(
            f"SKIP: {name} — enough stock or no recent demand."
        )

    if item.below_reorder_point:
        parts.append(
            f"Reason: on-hand {item.available_stock:g} is below reorder point "
            f"{item.reorder_point:g} (may run short during {lead}-day lead time)."
        )
    else:
        parts.append(
            f"Stock check: on-hand {item.available_stock:g} vs reorder point "
            f"{item.reorder_point:g} (not below reorder point)."
        )

    parts.append(
        f"Cover need after arrival ({cover}d): desired stock {item.desired_stock:g}; "
        f"expected on hand when truck arrives ~{item.projected_stock_at_arrival:g}; "
        f"raw need {item.raw_qty_to_order:g}, rounded up to full cases."
    )

    lookback = int(item.sales_lookback_days or 90)
    if lookback > 0:
        sales_bit = (
            f"Sales report (last {lookback}d): sold on {item.selling_days} day(s), "
            f"no sale on {item.zero_sales_days} day(s); "
            f"total {item.total_units_sold:g} units "
            f"(~{item.avg_units_on_selling_day:g}/selling day; "
            f"avg daily sales {item.ads:g})"
        )
        if item.available_stock < 0:
            sales_bit += (
                f"; includes oversold |on-hand|={abs(item.available_stock):g} as sold"
            )
        parts.append(sales_bit + ".")

    fest = (item.upcoming_festivals or "").strip()
    # Split named festivals vs weekend note for clearer wording
    fest_named = ""
    weekend_bit = ""
    if fest:
        chunks = [c.strip() for c in fest.split(";") if c.strip()]
        named = [c for c in chunks if not c.lower().startswith("weekend")]
        weekends = [c for c in chunks if c.lower().startswith("weekend")]
        fest_named = "; ".join(named)
        weekend_bit = "; ".join(weekends)

    window_end_note = f"as of {as_of}" if as_of else "as of today"
    if fest_named:
        cal = (
            f"Festivals in next {item.horizon_days} days ({window_end_note}): {fest_named}."
        )
    else:
        cal = (
            f"Festivals in next {item.horizon_days} days ({window_end_note}): "
            f"none on the holiday calendar."
        )
    if weekend_bit:
        cal += f" Also {weekend_bit}."
    if item.festival_uplift_applied and item.uplift_multiplier > 1.0:
        cal += (
            f" This item historically spikes — cover raised by x{item.uplift_multiplier:g}"
            f"{f' ({item.uplift_rule})' if item.uplift_rule else ''}."
        )
    elif fest_named or weekend_bit:
        cal += " No festival uplift applied for this item."
    parts.append(cal)

    if item.urgency and item.urgency not in ("ok", "skip"):
        parts.append(f"Urgency: {item.urgency}.")

    return " ".join(parts)


def detect_order(req: DetectOrderRequest) -> DetectOrderResponse:
    repo = DetectOrderRepository(tenant_id=req.tenant_id)
    store = ForecastStore(tenant_id=req.tenant_id)

    lead = int(req.lead_time_days)
    cover = int(req.time_to_cover_days)
    x_days = max(lead + cover, 1)

    try:
        vendors = _vendors(repo)
    except Exception as exc:
        return DetectOrderResponse(
            ok=False,
            vendors=[],
            lead_time_days=lead,
            time_to_cover_days=cover,
            x_days=x_days,
            db_mode=repo.mode,  # type: ignore[arg-type]
            forecast_mode="stub",  # type: ignore[arg-type]
            message=(
                "Database error — reconnect the SSH tunnel to the Wecomm *store* Postgres "
                f"(tenant schema with vendors/products). Details: {exc}"
            ),
        )

    if not (req.vendor_id or req.vendor_name):
        return DetectOrderResponse(
            ok=True,
            vendors=vendors,
            lead_time_days=lead,
            time_to_cover_days=cover,
            x_days=x_days,
            db_mode=repo.mode,  # type: ignore[arg-type]
            forecast_mode=store.mode,  # type: ignore[arg-type]
            message=f"{len(vendors)} vendors available. Pass vendor_id or vendor_name.",
        )

    vendor = repo.detect_vendor(vendor_id=req.vendor_id, vendor_name=req.vendor_name)
    if vendor is None:
        return DetectOrderResponse(
            ok=False,
            vendors=vendors,
            vendor=VendorInfo(
                vendor_id=req.vendor_id or "",
                vendor_name=req.vendor_name or "",
                detected=False,
            ),
            lead_time_days=lead,
            time_to_cover_days=cover,
            x_days=x_days,
            db_mode=repo.mode,  # type: ignore[arg-type]
            forecast_mode=store.mode,  # type: ignore[arg-type]
            message="Vendor not found.",
        )

    vendor_id = str(vendor["vendor_id"])
    vendor_name = str(vendor["vendor_name"])
    vendor_info = VendorInfo(vendor_id=vendor_id, vendor_name=vendor_name, detected=True)

    raw_items = repo.fetch_vendor_items(vendor_id)
    if not raw_items:
        return DetectOrderResponse(
            ok=True,
            vendors=vendors,
            vendor=vendor_info,
            lead_time_days=lead,
            time_to_cover_days=cover,
            x_days=x_days,
            db_mode=repo.mode,  # type: ignore[arg-type]
            forecast_mode=store.mode,  # type: ignore[arg-type]
            message=(
                f"No catalog items for vendor {vendor_name}. "
                "Need rows in product_vendor."
            ),
        )

    item_ids = [str(it["item_id"]) for it in raw_items]
    available_map = repo.fetch_available_stock(item_ids)
    other_vendor_prices = repo.fetch_all_vendor_prices(item_ids)
    all_products_on_hand = repo.fetch_all_products_on_hand()
    other_brands_by_item = build_other_brands_map(all_products_on_hand)
    same_product_name_by_item = build_same_product_name_map(all_products_on_hand)
    # Warm ADS/std once for the whole catalog
    demand_stats = store.get_demand_stats(item_ids)
    # Longer history for LightGBM lags (56d) + rolling windows
    sales_history = store.get_sales_history(item_ids, days=120)

    as_of = reorder_as_of_date()
    as_of_s = as_of.isoformat()
    lgbm_by_item: dict[str, list[dict[str, Any]]] = {}
    if lgbm_model_ready():
        product_attrs = repo.fetch_product_forecast_attrs(item_ids, as_of=as_of)
        lgbm_by_item = lgbm_forecast_batch(
            item_ids=item_ids,
            sales_history=sales_history,
            product_attrs=product_attrs,
            as_of=as_of,
            horizon_days=x_days,
        )
    fest_rows = festivals_in_horizon(x_days, as_of=as_of)
    upcoming_festivals = format_festivals_for_display(fest_rows)
    weekend_row = next((r for r in fest_rows if r.get("name") == "weekend"), None)
    if weekend_row:
        weekend_note = f"weekends ({weekend_row['days_in_window']}d in window)"
        upcoming_festivals = (
            f"{upcoming_festivals}; {weekend_note}" if upcoming_festivals else weekend_note
        )

    uplift_types = list(req.uplift_types or [])
    service_level = risk_to_service_level(int(req.risk_factor))
    lines: list[DetectOrderItem] = []

    for it in raw_items:
        item_id = str(it["item_id"])
        raw_on_hand = float(available_map.get(item_id, 0.0))
        oversold = max(0.0, -raw_on_hand)
        # Display keeps the real (possibly negative) OH. Order math floors at 0.
        available_for_order = max(0.0, raw_on_hand)
        box_qty = max(int(it.get("box_qty") or 1), 1)
        exp_days = it.get("expiration_days_remaining")
        exp_days_f = float(exp_days) if exp_days is not None else None
        last_pallet = it.get("last_pallet_qty")
        alt_ids = [str(x) for x in (it.get("upc"), it.get("sku")) if x]
        demand_class = it.get("demand_class") or store.demand_class_for(
            item_id, alt_ids=alt_ids
        )

        fc = store.get_forecast(
            item_id,
            horizon_days=x_days,
            demand_class=str(demand_class) if demand_class else None,
            alt_ids=alt_ids,
            uplift_types=uplift_types,
        )
        if demand_class is None and fc.get("demand_class"):
            demand_class = fc.get("demand_class")

        st = demand_stats.get(item_id) or {}
        # Live 90d ADS wins. Never let batch P50 invent a fake daily rate.
        ads = float(st.get("ads") if st.get("ads") is not None else fc.get("ads") or 0.0)
        demand_std = float(
            st.get("demand_std")
            if st.get("demand_std") is not None
            else fc.get("demand_std")
            or 0.0
        )
        if demand_std <= 0 and ads > 0:
            demand_std = ads * 0.3

        lookback_days = int(st.get("sales_lookback_days") or _ads_lookback_days())
        selling_days = int(st.get("selling_days") or 0)
        total_units_sold = float(st.get("total_units_sold") or 0.0)

        # Negative OH: show the minus in On Hand, but count |OH| as sold for ADS
        # and the sales report (so totals are not stuck at 0 while ADS rises).
        # Do not add |negative| onto the PO qty — order from floored stock = 0.
        if oversold > 0:
            ads = ads + oversold / max(float(lookback_days), 1.0)
            total_units_sold += oversold
            selling_days = max(selling_days, 1)

        zero_sales_days = max(lookback_days - selling_days, 0)
        avg_units_on_selling_day = (
            (total_units_sold / selling_days) if selling_days > 0 else 0.0
        )

        p50_full = float(fc["p50"])
        p90_full = float(fc["p90"])
        stored_h = int(fc["horizon_days"])
        uplift_m = float(fc.get("uplift_multiplier") or 1.0)
        uplift_rule = fc.get("uplift_rule")

        notes: list[str] = []
        if oversold > 0:
            notes.append(
                f"On-hand {raw_on_hand:g} (oversold). "
                f"{oversold:g} counted as sold in ADS and sales totals; "
                f"order math uses stock 0 (deficit not added to PO)."
            )

        # Order sizes to cover C after arrival; OH burns during L.
        effective_days = float(x_days)
        expiry_capped = False
        if exp_days_f is not None and effective_days > 0 and exp_days_f < effective_days:
            effective_days = max(exp_days_f, 0.0)
            expiry_capped = True
            notes.append(
                f"Window capped to {effective_days:g}d by expiration "
                f"(requested X={x_days}d = L{lead}+C{cover})."
            )

        wecomm_min = float(it.get("wecomm_min_on_hand") or 0.0)
        wecomm_max = float(it.get("wecomm_max_on_hand") or 0.0)

        calc = compute_line_reorder(
            available=available_for_order,
            ads=ads,
            demand_std=demand_std,
            lead_days=lead,
            cover_days=cover,
            x_days=x_days,
            p50_full=p50_full,
            p90_full=p90_full,
            stored_horizon=stored_h,
            box_qty=box_qty,
            effective_days=effective_days,
            uplift_multiplier=uplift_m,
            service_level=service_level,
            wecomm_min_on_hand=wecomm_min,
            wecomm_max_on_hand=wecomm_max,
        )
        if calc.get("skip_dead_stock"):
            notes.append("No recent demand (ADS≈0) — skipped to avoid overstock.")
        if uplift_m and float(uplift_m) > 1.0:
            notes.append(
                f"Cover sales uplift×{float(uplift_m):g}"
                f"{f' ({uplift_rule})' if uplift_rule else ''}."
            )
        if calc.get("max_capped_target"):
            notes.append(
                f"Wecomm max {wecomm_max:g} capped desired (anti-overstock)."
            )
        if calc.get("box_rounded") and float(calc.get("qty_to_order") or 0) > 0:
            notes.append(
                f"Rounded up to full case(s): raw {float(calc.get('raw_qty_to_order') or 0):g} "
                f"→ {float(calc.get('qty_to_order') or 0):g} units "
                f"({float(calc.get('cases_to_order') or 0):g} cases)."
            )
        if calc.get("line_action") == "WATCH" and float(calc.get("qty_to_order") or 0) <= 0:
            notes.append(
                "WATCH: below ROP but cover need is already met at arrival."
            )

        qty_box = float(calc["qty_to_order"])
        # Expiry re-cap after pack round vs desired stock at arrival
        if expiry_capped and qty_box > 0 and effective_days > 0:
            max_sellable = float(calc["projected_stock_required"])
            at_arrival = float(calc["projected_stock_at_arrival"])
            if at_arrival + qty_box > max_sellable + 1e-6:
                from v2.inventory_math.pack_size import normalize_pack_size, round_up_to_pack

                pack = normalize_pack_size(box_qty)
                capped_need = max(0.0, max_sellable - at_arrival)
                new_qty = float(round_up_to_pack(capped_need, pack))
                while new_qty > 0 and (at_arrival + new_qty) > max_sellable + 1e-6:
                    new_qty = max(0.0, new_qty - pack)
                if new_qty != qty_box:
                    notes.append(f"Expiry re-cap after pack round: {qty_box:g} → {new_qty:g}.")
                    qty_box = new_qty
                    calc["qty_to_order"] = qty_box
                    calc["cases_to_order"] = (
                        round(qty_box / pack, 2) if pack > 1 and qty_box > 0 else qty_box
                    )

        vendor_price = float(it["vendor_price"]) if it.get("vendor_price") is not None else None
        offers = [o for o in other_vendor_prices.get(item_id, []) if o["vendor_id"] != vendor_id]
        cheaper_offers = (
            [o for o in offers if o["price"] < vendor_price] if vendor_price is not None else []
        )
        cheapest_offer = min(cheaper_offers, key=lambda o: o["price"], default=None)
        if cheapest_offer:
            notes.append(
                f"Cheaper elsewhere: {cheapest_offer['vendor_name']} @ ${cheapest_offer['price']:.2f} "
                f"vs ${vendor_price:.2f} here."
            )

        item = DetectOrderItem(
            item_id=item_id,
            upc=it.get("upc"),
            sku=it.get("sku"),
            description=str(it.get("description") or ""),
            vendor_id=vendor_id,
            vendor_price=vendor_price,
            other_brands_stock=str(
                (other_brands_by_item.get(item_id) or {}).get("other_brands_stock") or ""
            ),
            same_item_brand_count=int(
                (other_brands_by_item.get(item_id) or {}).get("same_item_brand_count") or 0
            ),
            same_product_name_qty=same_product_name_by_item.get(item_id) or [],
            other_vendor_prices=offers,
            cheaper_elsewhere=bool(cheapest_offer),
            cheapest_vendor=cheapest_offer,
            sales_series=SalesSeries(
                history=sales_history.get(item_id, []),
                forecast=_forecast_series(
                    store,
                    item_id,
                    alt_ids,
                    ads,
                    x_days,
                    as_of,
                    uplift_types,
                    lgbm_series=lgbm_by_item.get(item_id),
                ),
            ),
            demand_class=str(demand_class) if demand_class else None,
            forecast_source=(
                "global_lightgbm"
                if item_id in lgbm_by_item
                else str(fc.get("source") or "")
            ),
            available_stock=raw_on_hand,
            last_pallet_qty=float(last_pallet) if last_pallet is not None else None,
            expiration_days_remaining=exp_days_f,
            box_qty=box_qty,
            ads=float(calc["ads"]),
            demand_std=float(calc["demand_std"]),
            safety_stock=float(calc["safety_stock"]),
            safety_stock_cover=float(calc.get("safety_stock_x") or 0.0),
            reorder_point=float(calc["reorder_point"]),
            below_reorder_point=bool(calc["below_reorder_point"]),
            wecomm_min_on_hand=float(calc.get("wecomm_min_on_hand") or 0.0),
            wecomm_max_on_hand=float(calc.get("wecomm_max_on_hand") or 0.0),
            min_on_hand=float(calc.get("min_on_hand") or 0.0),
            min_on_hand_source=str(calc.get("min_on_hand_source") or "none"),
            below_min_on_hand=bool(calc.get("below_min_on_hand")),
            desired_stock=float(calc.get("desired_stock") or calc["ai_target_qty"]),
            days_of_supply=calc.get("days_of_supply"),
            days_of_supply_after_order=calc.get("days_of_supply_after_order"),
            urgency=str(calc.get("urgency") or "ok"),
            line_action=str(calc.get("line_action") or "SKIP"),
            lead_demand_ads=float(calc["lead_demand_ads"]),
            lead_demand_p50=float(calc["lead_demand_p50"]),
            cover_demand_ads=float(calc["cover_demand_ads"]),
            cover_demand_p90=float(calc["cover_demand_p90"]),
            ads_cover_qty=float(calc["ads_cover_qty"]),
            ads_times_x=float(calc.get("ads_times_x") or 0.0),
            uplift_multiplier=float(calc["uplift_multiplier"]),
            uplift_rule=str(uplift_rule) if uplift_rule else None,
            upcoming_festivals=upcoming_festivals,
            festival_uplift_applied=bool(uplift_m and float(uplift_m) > 1.0),
            sales_lookback_days=lookback_days,
            selling_days=selling_days,
            zero_sales_days=zero_sales_days,
            total_units_sold=round(total_units_sold, 4),
            avg_units_on_selling_day=round(avg_units_on_selling_day, 4),
            p50_demand=float(calc["p50_demand"]),
            p90_demand=float(calc["p90_demand"]),
            ai_target_qty=float(calc["ai_target_qty"]),
            horizon_days=x_days,
            forecast_horizon_used=stored_h,
            projected_stock_required=float(calc["projected_stock_required"]),
            projected_stock_at_arrival=float(calc["projected_stock_at_arrival"]),
            raw_qty_to_order=float(calc["raw_qty_to_order"]),
            qty_before_box_round=float(calc["qty_before_box_round"]),
            qty_to_order=float(calc["qty_to_order"]),
            cases_to_order=float(calc["cases_to_order"]),
            expiry_capped=expiry_capped,
            expiry_cap_days=effective_days if expiry_capped else None,
            box_rounded=bool(calc["box_rounded"]),
            validation_notes=notes,
        )
        item.justification = _template_justification(
            item, lead=lead, cover=cover, as_of=as_of_s
        )
        lines.append(item)

    # Sort: stockout/critical first, then ORDER qty desc, then WATCH
    _urg_rank = {
        "stockout": 0,
        "critical": 1,
        "high": 2,
        "medium": 3,
        "ok": 4,
        "skip": 5,
    }
    lines.sort(
        key=lambda x: (
            _urg_rank.get(x.urgency, 9),
            0 if x.line_action == "ORDER" else 1 if x.line_action == "WATCH" else 2,
            -x.qty_to_order,
            x.description or "",
        )
    )

    order_lines = [x for x in lines if x.qty_to_order > 0]
    watch_lines = [x for x in lines if x.line_action == "WATCH"]
    # Always return actionable lines only (ORDER + WATCH). SKIP / zero lines omitted.
    out_items = [x for x in lines if x.line_action in ("ORDER", "WATCH")]
    total_units = round(sum(x.qty_to_order for x in order_lines), 2)
    total_cases = round(sum(x.cases_to_order for x in order_lines), 2)

    run_id = new_run_id()
    response = DetectOrderResponse(
        ok=True,
        run_id=run_id,
        tenant_id=req.tenant_id,
        vendors=vendors,
        vendor=vendor_info,
        lead_time_days=lead,
        time_to_cover_days=cover,
        x_days=x_days,
        uplift_types=list(uplift_types),  # type: ignore[arg-type]
        risk_factor=int(req.risk_factor),
        service_level=float(service_level),
        as_of_date=as_of_s,
        upcoming_festivals=upcoming_festivals,
        catalog_item_count=len(lines),
        item_count=len(out_items),
        order_line_count=len(order_lines),
        total_units_to_order=total_units,
        total_cases_to_order=total_cases,
        items=out_items,
        db_mode=repo.mode,  # type: ignore[arg-type]
        forecast_mode=store.mode,  # type: ignore[arg-type]
        message=(
            f"{vendor_name}: catalog {len(lines)} → ORDER {len(order_lines)} + "
            f"WATCH {len(watch_lines)} for X={x_days}d (L={lead}/C={cover}): "
            f"{total_units:g} units / {total_cases:g} cases. "
            f"ROP=trigger only; qty=cover C (+ Wecomm min/max). run_id={run_id}"
        ),
    )

    save_order_run(response.model_dump())
    return response


def get_order_run(run_id: str) -> dict[str, Any] | None:
    from api.services.order_run_store import load_order_run

    return load_order_run(run_id)


def get_item_model_comparison(run_id: str, item_id: str) -> dict[str, Any] | None:
    """Live per-model forecast comparison for one item from a saved run (on-demand, not for full-catalog use)."""
    from api.services.order_run_store import load_order_run

    run = load_order_run(run_id)
    if not run:
        return None
    item = next((it for it in run.get("items", []) if str(it.get("item_id")) == str(item_id)), None)
    if item is None:
        return None

    store = ForecastStore(tenant_id=run.get("tenant_id"))
    alt_ids = [str(x) for x in (item.get("upc"), item.get("sku")) if x]
    horizon = int(run.get("x_days") or item.get("horizon_days") or 7)
    models = store.compare_models(item_id, alt_ids, horizon)
    return {
        "run_id": run_id,
        "item_id": item_id,
        "description": item.get("description"),
        "horizon_days": horizon,
        "models": models,
    }
