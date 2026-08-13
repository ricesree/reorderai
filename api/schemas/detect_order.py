"""W-1 Detect Order — request / response schemas (design doc)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# Multi-select uplift patterns the client can enable (Swagger / UI dropdown).
UpliftType = Literal["weekend", "festival", "trend"]


class DetectOrderRequest(BaseModel):
    """
    Workflow W-1 inputs:
      - Vendor
      - Lead time (days until delivery)
      - Time to cover (extra days the next pallet should cover)
      - Which uplift patterns to apply (multi-select; empty = none)
      - Risk factor 0–100 (scales safety-stock buffer)
    """

    tenant_id: str | None = Field(
        None,
        pattern=r"^[A-Za-z0-9_-]+$",
        description=(
            "Tenant schema override (e.g. wecomm_<uuid>). "
            "Omit to use the server's default TENANT_SCHEMA."
        ),
    )
    vendor_id: str | None = None
    vendor_name: str | None = None
    lead_time_days: int = Field(ge=0, description="Days from order to receipt")
    time_to_cover_days: int = Field(ge=0, description="Extra days of cover beyond lead time")
    uplift_types: list[UpliftType] = Field(
        default_factory=lambda: ["weekend", "festival"],
        description=(
            "Multi-select uplift patterns to apply for this run. "
            "Allowed: weekend, festival, trend. "
            "Select all, some, or [] for none (multiplier stays 1.0). "
            "trend is reserved (no effect until trend uplift is trained)."
        ),
        examples=[["weekend", "festival"], ["weekend"], []],
    )
    risk_factor: int = Field(
        default=50,
        ge=0,
        le=100,
        description=(
            "Risk level 0–100. Higher → larger safety-stock buffer "
            "(target percentile: 0→P50, 50→P75, 100→P100, capped at P99.9)."
        ),
    )
    generate_justification: bool = Field(
        default=True,
        description=(
            "Deprecated / ignored. Justification is always filled from order math "
            "(report-style template; no GPT)."
        ),
    )


class VendorInfo(BaseModel):
    vendor_id: str
    vendor_name: str
    detected: bool = True


class VendorPriceOffer(BaseModel):
    vendor_id: str
    vendor_name: str
    price: float


class SameProductNameQty(BaseModel):
    item_id: str
    description: str
    qty: float


class SalesPoint(BaseModel):
    date: str
    qty: float


class SalesSeries(BaseModel):
    history: list[SalesPoint] = Field(
        default_factory=list, description="Actual daily sales, most recent lookback window"
    )
    forecast: list[SalesPoint] = Field(
        default_factory=list,
        description="Projected daily sales for the upcoming order window (ADS × uplift)",
    )


class DetectOrderItem(BaseModel):
    item_id: str
    upc: str | None = None
    sku: str | None = None
    description: str
    vendor_id: str
    vendor_price: float | None = Field(None, description="Vendor's price for this item (product_vendor.price)")
    other_brands_stock: str = Field(
        "",
        description=(
            "Same product under other brands with on-hand qty, e.g. "
            "'Vadilal Drumstick - 7 | Ashoka Drumstick 310G - 1'. "
            "Matched by core item name (not Wecomm category)."
        ),
    )
    same_item_brand_count: int = Field(
        0,
        description="How many brand SKUs of this same item are in stock (incl. this line); 0 if alone",
    )
    same_product_name_qty: list[SameProductNameQty] = Field(
        default_factory=list,
        description=(
            "Other products sharing this line's exact products.product_name, "
            "with each one's current on-hand qty."
        ),
    )
    other_vendor_prices: list[VendorPriceOffer] = Field(
        default_factory=list,
        description="All other vendors selling this product (product_vendor), for price comparison",
    )
    cheaper_elsewhere: bool = Field(
        False, description="True if another vendor sells this item for less than vendor_price"
    )
    cheapest_vendor: VendorPriceOffer | None = Field(
        None, description="Lowest-price offer among other vendors, only set when cheaper_elsewhere"
    )
    sales_series: SalesSeries = Field(
        default_factory=SalesSeries,
        description="Historical + predicted daily sales for this item, for UI charting",
    )

    demand_class: str | None = None
    forecast_source: str | None = None

    # Stock (raw Wecomm OH — negatives kept; order math floors at 0)
    available_stock: float
    last_pallet_qty: float | None = None
    expiration_days_remaining: float | None = None
    box_qty: int = 1

    # Demand drivers
    ads: float = Field(0.0, description="Average daily sales")
    demand_std: float = Field(0.0, description="Daily demand std used for safety stock")
    safety_stock: float = Field(
        0.0, description="SS(L) — lead-time buffer for ROP only (Z×σ×√L)"
    )
    safety_stock_cover: float = Field(
        0.0, description="SS(C) — cover buffer used in ads_cover / AI target (Z×σ×√C)"
    )
    reorder_point: float = Field(
        0.0, description="ROP trigger = ADS×L + SS(L) — urgency only, not order floor"
    )
    below_reorder_point: bool = False
    wecomm_min_on_hand: float = Field(
        0.0, description="Raw Wecomm min (product min_on_hand / location min_quantity)"
    )
    wecomm_max_on_hand: float = Field(
        0.0, description="Wecomm max_quantity cap (0 = no cap)"
    )
    min_on_hand: float = Field(
        0.0, description="Effective min floor = Wecomm min if > 0, else 0 (no ROP floor)"
    )
    min_on_hand_source: str = Field(
        "none", description="wecomm | none — where min_on_hand came from"
    )
    below_min_on_hand: bool = False
    desired_stock: float = Field(
        0.0, description="Order-up-to after arrival = max(cover, min), capped by max"
    )
    days_of_supply: float | None = Field(
        None, description="OH / ADS (None if no demand)"
    )
    days_of_supply_after_order: float | None = Field(
        None, description="(stock at arrival + qty) / ADS"
    )
    urgency: str = Field(
        "ok", description="stockout | critical | high | medium | ok | skip"
    )
    line_action: str = Field(
        "SKIP", description="ORDER | WATCH | SKIP — what the buyer should do"
    )

    # Lead burn (not ordered) + cover C (order sizes to this)
    lead_demand_ads: float = Field(0.0, description="ADS × L (burned from on-hand before arrival)")
    lead_demand_p50: float = Field(0.0, description="ML P50 over lead days")
    cover_demand_ads: float = Field(0.0, description="ADS × C (after arrival)")
    cover_demand_p90: float = Field(0.0, description="ML P90 over cover days")
    ads_cover_qty: float = Field(0.0, description="ADS × C + SS(C) without uplift")
    ads_times_x: float = Field(
        0.0,
        description="Sanity baseline = ADS × X (before SS / uplift / final target)",
    )
    uplift_multiplier: float = 1.0
    uplift_rule: str | None = None
    upcoming_festivals: str = Field(
        "",
        description="Festival/weekend tags in the next X days from the calendar",
    )
    festival_uplift_applied: bool = False
    sales_lookback_days: int = 90
    selling_days: int = Field(
        0, description="Days with sales > 0 in the ADS lookback (e.g. 5 of 90)"
    )
    zero_sales_days: int = Field(
        0, description="Days with no sale in the ADS lookback"
    )
    total_units_sold: float = 0.0
    avg_units_on_selling_day: float = 0.0
    p50_demand: float = Field(0.0, description="ML P50 for full window X=L+C (reference)")
    p90_demand: float = Field(0.0, description="ML P90 for full window X=L+C (reference)")
    ai_target_qty: float = Field(
        0.0, description="Cover need after arrival = (ADS×C×uplift) + SS(C)"
    )

    horizon_days: int
    forecast_horizon_used: int
    projected_stock_required: float = Field(
        description="Desired stock after arrival = max(ai_target, min_on_hand)"
    )
    projected_stock_at_arrival: float = Field(
        description="Expected on-hand when order arrives = max(0, OH − ADS×L)"
    )

    # Order qty
    raw_qty_to_order: float
    qty_before_box_round: float
    qty_to_order: float
    cases_to_order: float = 0.0
    expiry_capped: bool = False
    expiry_cap_days: float | None = None
    box_rounded: bool = False
    validation_notes: list[str] = Field(default_factory=list)

    justification: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class DetectOrderResponse(BaseModel):
    ok: bool = True
    run_id: str | None = None
    tenant_id: str | None = Field(
        None, description="Tenant schema used for this run (persisted so per-item model lookups can reconnect)"
    )
    vendors: list[VendorInfo] = Field(default_factory=list)
    vendor: VendorInfo | None = None
    lead_time_days: int = 0
    time_to_cover_days: int = 0
    x_days: int = Field(0, description="Lead Time + Time to Cover")
    uplift_types: list[UpliftType] = Field(
        default_factory=list,
        description="Uplift patterns enabled for this run (from request)",
    )
    risk_factor: int = Field(50, description="Risk 0–100 used for this run")
    service_level: float = Field(
        0.95, description="Target percentile derived from risk_factor (e.g. 0.75 = P75), used for safety stock"
    )
    as_of_date: str = Field(
        "",
        description=(
            "Calendar 'today' used for festival/weekend scan "
            "(API host clock in REORDER_TZ, default America/Detroit — Michigan)"
        ),
    )
    upcoming_festivals: str = Field(
        "",
        description="Named festivals in the next X days from as_of_date (shared for the run)",
    )
    catalog_item_count: int = 0
    item_count: int = 0
    order_line_count: int = 0
    total_units_to_order: float = 0.0
    total_cases_to_order: float = 0.0
    items: list[DetectOrderItem] = Field(default_factory=list)
    db_mode: Literal["stub", "live"] = "stub"
    forecast_mode: Literal["stub", "live", "batch"] = "stub"
    message: str = ""


class OrderRunSummary(BaseModel):
    run_id: str
    vendor_id: str
    vendor_name: str
    created_at: str
    x_days: int
    order_line_count: int
