"""
Pro vendor reorder math for a periodic PO sheet.

Goals (replace gut-feel):
  - Don't miss items that will stock out during lead (ROP trigger)
  - Don't overstock (order for cover C after arrival, not gut pallets; Wecomm max cap)
  - Clear action list: ORDER / WATCH / SKIP

Core:
  ROP      = ADS×L + SS(L)          → trigger / urgency only (NOT an order floor)
  Cover    = ceil((ADS×C×uplift) + SS(C))  → whole units after arrival
  Arrival  = max(0, OH − ADS×L)            → burn on-hand during lead
  Desired  = Cover (integer); optional Wecomm max cap if set
  Order    = round UP to full cases (pack multiples); cases is a whole number

Dead stock (ADS≈0): skip. ML P50/P90: reference only.
"""

from __future__ import annotations

import math
import os
from typing import Any

from v2.inventory_math.pack_size import normalize_pack_size
from v2.inventory_math.reorder_point import calculate_dynamic_reorder_point
from v2.inventory_math.safety_stock import calculate_safety_stock

DEFAULT_MAX_DEMAND_CV = 2.0
# Treat ADS below this as "no demand" for ordering (noise / one-off scans).
ADS_DEAD_EPS = 1e-6


def _max_demand_cv() -> float:
    raw = os.getenv("MAX_DEMAND_CV", str(DEFAULT_MAX_DEMAND_CV))
    try:
        return max(float(raw), 0.3)
    except (TypeError, ValueError):
        return DEFAULT_MAX_DEMAND_CV


def _capped_demand_std(ads: float, demand_std: float) -> float:
    ads_f = max(float(ads), 0.0)
    std_f = max(float(demand_std), 0.0)
    if ads_f <= 0:
        return 0.0
    return min(std_f, ads_f * _max_demand_cv())


def scale_forecast(value: float, from_days: int, to_days: int) -> float:
    if from_days <= 0 or to_days <= 0:
        return 0.0
    if to_days == from_days:
        return float(value)
    return float(value) * (float(to_days) / float(from_days))


def _days_of_supply(on_hand: float, ads: float) -> float | None:
    if ads <= ADS_DEAD_EPS:
        return None
    return round(max(on_hand, 0.0) / ads, 2)


def _urgency(
    *,
    on_hand: float,
    ads: float,
    rop: float,
    below_rop: bool,
    qty: float,
    wecomm_min: float,
    skip_dead: bool,
) -> str:
    if skip_dead:
        return "skip"
    if on_hand <= 0 and (ads > ADS_DEAD_EPS or wecomm_min > 0):
        return "stockout"
    if rop > 0 and on_hand < 0.5 * rop:
        return "critical"
    if below_rop:
        return "high"
    if qty > 0:
        return "medium"
    return "ok"


def _line_action(
    *,
    qty: float,
    below_rop: bool,
    skip_dead: bool,
    ads: float,
) -> str:
    if qty > 0:
        return "ORDER"
    if skip_dead:
        return "SKIP"
    if below_rop:
        return "WATCH"
    if ads <= ADS_DEAD_EPS:
        return "SKIP"
    return "SKIP"


def _ceil_units(value: float) -> float:
    """Round up to a whole unit (no fractional cover / order display)."""
    if value <= 1e-9:
        return 0.0
    return float(math.ceil(value - 1e-9))


def _round_up_full_cases(raw_need: float, pack: int) -> tuple[float, float]:
    """Round need up to whole cases. Returns (qty_units, cases)."""
    if raw_need <= 1e-9:
        return 0.0, 0.0
    pack = max(int(pack), 1)
    cases = int(math.ceil(raw_need / pack - 1e-9))
    return float(cases * pack), float(cases)


def compute_line_reorder(
    *,
    available: float,
    ads: float,
    demand_std: float,
    lead_days: int,
    cover_days: int,
    x_days: int,
    p50_full: float,
    p90_full: float,
    stored_horizon: int,
    box_qty: int,
    effective_days: float,
    uplift_multiplier: float = 1.0,
    service_level: float = 0.95,
    case_fill_ratio: float | None = None,  # unused — units suggested, no case gate
    ml_ads_cap_ratio: float | None = None,
    wecomm_min_on_hand: float = 0.0,
    wecomm_max_on_hand: float = 0.0,
) -> dict[str, Any]:
    """Return sizing + action fields for one SKU."""
    del ml_ads_cap_ratio
    del case_fill_ratio
    lead = max(int(lead_days), 0)
    cover = max(int(cover_days), 0)
    x = max(int(x_days), 1)
    eff = max(float(effective_days), 0.0)
    pack = normalize_pack_size(box_qty)
    ads_f = max(float(ads), 0.0)
    std_raw = max(float(demand_std), 0.0)
    std_f = _capped_demand_std(ads_f, std_raw)
    uplift = max(float(uplift_multiplier or 1.0), 0.0)
    on_hand = max(float(available), 0.0)
    wecomm_min = max(float(wecomm_min_on_hand or 0.0), 0.0)
    wecomm_max = max(float(wecomm_max_on_hand or 0.0), 0.0)

    # --- ROP (trigger only) ---
    ss_lead = (
        calculate_safety_stock(ads_f, std_f, max(lead, 1), service_level=service_level)
        if ads_f > ADS_DEAD_EPS and lead > 0
        else 0.0
    )
    rop = (
        calculate_dynamic_reorder_point(ads_f, float(lead), ss_lead)
        if ads_f > ADS_DEAD_EPS and lead > 0
        else 0.0
    )
    lead_demand_ads = round(ads_f * lead, 4) if lead > 0 else 0.0
    lead_demand_p50 = (
        scale_forecast(p50_full, stored_horizon, max(lead, 1)) if lead > 0 else 0.0
    )
    stock_at_arrival = max(0.0, on_hand - ads_f * lead) if lead > 0 else on_hand

    # --- Cover C (order sizes to this) ---
    # When effective window X is set, cover days = X − L (arrival → end of window).
    # cover_days is the fallback only when effective_days is not provided.
    if eff > 0:
        cover_eff = max(eff - float(lead), 0.0)
    else:
        cover_eff = float(cover)

    has_demand = ads_f > ADS_DEAD_EPS
    ss_c = (
        calculate_safety_stock(ads_f, std_f, max(cover_eff, 1.0), service_level=service_level)
        if has_demand and cover_eff > 0
        else 0.0
    )
    base_expected = ads_f * cover_eff if has_demand and cover_eff > 0 else 0.0
    uplifted_expected = base_expected * uplift
    # Whole units only (ceil) — no fractional cover days / targets
    ads_cover = _ceil_units(base_expected + ss_c)
    ai_target = _ceil_units(uplifted_expected + ss_c)

    # Dead stock: no sales → no order
    skip_dead = not has_demand
    if skip_dead:
        ai_target = 0.0
        ads_cover = 0.0

    desired = float(ai_target)
    max_capped = False
    if wecomm_max > 0 and not skip_dead:
        capped = min(desired, wecomm_max)
        if capped + 1e-9 < desired:
            max_capped = True
        desired = _ceil_units(capped) if capped > 0 else 0.0
        if stock_at_arrival >= wecomm_max:
            desired = _ceil_units(stock_at_arrival)  # raw need → 0

    # ML reference (full X) — always the real model forecast over the full requested
    # window, independent of expiry-capping. Expiry only shrinks the order QUANTITY
    # (cover_eff below), not the reference sales prediction shown to the buyer.
    h = int(max(round(float(x)), 1))
    p50_x = scale_forecast(p50_full, stored_horizon, h)
    p90_x = scale_forecast(p90_full, stored_horizon, h)

    if cover_eff > 0 and has_demand:
        cover_h = int(max(round(cover_eff), 1))
        cover_demand_p90 = scale_forecast(p90_full, stored_horizon, cover_h)
        cover_demand_ads = _ceil_units(ads_f * cover_eff)
    else:
        cover_demand_p90 = 0.0
        cover_demand_ads = 0.0

    below_min = wecomm_min > 0 and on_hand < wecomm_min
    below_rop = below_min or (
        bool(on_hand < rop) if rop > 0 else bool(on_hand <= 0 and ai_target > 0)
    )

    raw_need = 0.0 if skip_dead else max(0.0, desired - stock_at_arrival)
    qty_units, cases = _round_up_full_cases(raw_need, pack)
    # Optional Wecomm max: trim down to whole cases that still fit under max
    if wecomm_max > 0 and qty_units > 0:
        room = max(0.0, wecomm_max - stock_at_arrival)
        max_cases = int(math.floor(room / pack + 1e-9)) if pack > 0 else 0
        if max_cases <= 0:
            qty_units, cases = 0.0, 0.0
        elif cases > max_cases:
            cases = float(max_cases)
            qty_units = float(max_cases * pack)

    did_round = abs(qty_units - raw_need) > 1e-6

    dos = _days_of_supply(on_hand, ads_f)
    dos_after = _days_of_supply(stock_at_arrival + qty_units, ads_f)
    urgency = _urgency(
        on_hand=on_hand,
        ads=ads_f,
        rop=float(rop),
        below_rop=below_rop,
        qty=qty_units,
        wecomm_min=wecomm_min,
        skip_dead=skip_dead,
    )
    action = _line_action(
        qty=qty_units,
        below_rop=below_rop,
        skip_dead=skip_dead,
        ads=ads_f,
    )

    return {
        "ads": round(ads_f, 4),
        "demand_std": round(std_f, 4),
        "demand_std_raw": round(std_raw, 4),
        "safety_stock": float(ss_lead),
        "safety_stock_x": float(ss_c),
        "safety_stock_cover": float(ss_c),
        "reorder_point": float(rop),
        "below_reorder_point": bool(below_rop),
        "wecomm_min_on_hand": round(wecomm_min, 2),
        "wecomm_max_on_hand": round(wecomm_max, 2),
        "min_on_hand": round(wecomm_min, 2),
        "min_on_hand_source": "wecomm" if wecomm_min > 0 else "none",
        "below_min_on_hand": bool(below_min),
        "min_raised_target": False,
        "max_capped_target": bool(max_capped),
        "lead_demand_ads": float(lead_demand_ads),
        "lead_demand_p50": round(float(lead_demand_p50), 2),
        "cover_demand_ads": float(cover_demand_ads),
        "cover_demand_p90": round(float(cover_demand_p90), 2),
        "ads_cover_qty": float(ads_cover),
        # Sanity baseline: ADS × full window X (before SS / uplift / ML polish)
        "ads_times_x": round(ads_f * float(x), 4) if has_demand else 0.0,
        "expected_sales_x": float(_ceil_units(base_expected)),
        "uplifted_expected_x": float(_ceil_units(uplifted_expected)),
        "uplift_multiplier": float(uplift),
        "p50_demand": round(p50_x, 2),
        "p90_demand": round(p90_x, 2),
        "p50_raw": round(p50_x, 2),
        "p90_raw": round(p90_x, 2),
        "ml_capped": False,
        "ml_ads_cap_ratio": 1.0,
        "ai_target_qty": float(ai_target),
        "desired_stock": float(desired),
        "projected_stock_required": float(desired),
        "projected_stock_at_arrival": round(stock_at_arrival, 2),
        "raw_qty_to_order": round(raw_need, 2),
        "qty_before_box_round": round(raw_need, 2),
        "qty_to_order": float(qty_units),
        "cases_to_order": float(cases),
        "box_rounded": bool(did_round),
        "p90_eff": float(p90_x),
        "p50_eff": float(p50_x),
        "cover_days_used": float(cover_eff),
        "x_days": float(x),
        "days_of_supply": dos,
        "days_of_supply_after_order": dos_after,
        "urgency": urgency,
        "line_action": action,
        "skip_dead_stock": bool(skip_dead),
    }
