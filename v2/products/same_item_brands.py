"""Same product across brands — on-hand list for detect-order UI.

Groups by brand-stripped core item name (not Wecomm category), e.g.:
  Vadilal Drumstick | Ashoka Drumstick 310G | Daily Delight Drumstick
Mango ice cream will NOT match vanilla ice cream.
"""

from __future__ import annotations

from typing import Any

from v2.products.product_normalization import product_signature

# Single-token cores that are too generic to group alone
_GENERIC_SINGLE = frozenset(
    {
        "RICE",
        "OIL",
        "FLOUR",
        "SALT",
        "SUGAR",
        "TEA",
        "MILK",
        "WATER",
        "SNACK",
        "SNACKS",
        "MIX",
        "POWDER",
        "LEAVES",
        "LEAF",
        "BEANS",
        "DAL",
        "ATTA",
        "GHEE",
        "BUTTER",
        "CHEESE",
        "BREAD",
        "JUICE",
        "SODA",
        "WATER",
        "ICE",
        "CREAM",
        "ICECREAM",
    }
)


def _stock_int(val: Any) -> int:
    try:
        return int(round(float(val or 0)))
    except (TypeError, ValueError):
        return 0


def _fmt_alt(description: str, stock: Any) -> str:
    return f"{str(description).strip()} - {_stock_int(stock)}"


def _usable_key(sig: dict[str, Any]) -> str | None:
    """Return item_key if distinctive enough (flavor kept; brand stripped)."""
    tokens = list(sig.get("core_tokens") or [])
    key = str(sig.get("item_key") or "").strip()
    if not key or not tokens:
        return None
    if len(tokens) >= 2:
        return key
    # Single core token OK if brand was stripped and token isn't generic
    tok = tokens[0].upper()
    if len(tok) >= 4 and tok not in _GENERIC_SINGLE and sig.get("brand"):
        return key
    if len(tok) >= 6 and tok not in _GENERIC_SINGLE:
        return key
    return None


def build_other_brands_map(
    catalog: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """
    From full product catalog rows ({item_id, upc, description, quantity}),
    return per item_id:
      other_brands_stock: "Name - qty | Name - qty"
      same_item_brand_count: int (# of distinct brand SKUs in group including self)
    """
    by_key: dict[str, list[dict[str, Any]]] = {}
    id_to_key: dict[str, str] = {}

    for row in catalog:
        item_id = str(row.get("item_id") or "").strip()
        desc = str(row.get("description") or "").strip()
        if not item_id or not desc:
            continue
        sig = product_signature(desc)
        key = _usable_key(sig)
        if not key:
            continue
        by_key.setdefault(key, []).append(
            {
                "item_id": item_id,
                "upc": str(row.get("upc") or "").strip(),
                "description": desc,
                "quantity": float(row.get("quantity") or 0.0),
                "brand": str(sig.get("brand") or ""),
            }
        )
        id_to_key[item_id] = key

    out: dict[str, dict[str, Any]] = {}
    for item_id, key in id_to_key.items():
        members = by_key.get(key) or []
        # Deduplicate by description
        seen: set[str] = set()
        others: list[tuple[str, float]] = []
        for m in members:
            if m["item_id"] == item_id:
                continue
            du = m["description"].upper()
            if du in seen:
                continue
            seen.add(du)
            others.append((m["description"], float(m["quantity"])))
        others.sort(key=lambda x: x[0].upper())
        text = " | ".join(_fmt_alt(d, q) for d, q in others)
        # brand count = self + distinct other descriptions
        brand_count = (1 if others or any(m["item_id"] == item_id for m in members) else 0) + len(
            others
        )
        if others:
            brand_count = len(others) + 1
        else:
            brand_count = 0  # alone — nothing useful to show
        out[item_id] = {
            "other_brands_stock": text,
            "same_item_brand_count": brand_count,
        }
    return out


def build_same_product_name_map(
    catalog: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """
    Group by exact products.product_name (not the fuzzy brand-stripped signature above).

    From full product catalog rows ({item_id, product_name, description, quantity}),
    return per item_id: the other rows sharing the same product_name, each as
    {item_id, description, qty}.
    """
    by_name: dict[str, list[dict[str, Any]]] = {}
    for row in catalog:
        name = str(row.get("product_name") or "").strip()
        if not name:
            continue
        by_name.setdefault(name.upper(), []).append(row)

    out: dict[str, list[dict[str, Any]]] = {}
    for rows in by_name.values():
        for row in rows:
            item_id = str(row.get("item_id") or "").strip()
            if not item_id:
                continue
            out[item_id] = [
                {
                    "item_id": str(r.get("item_id") or ""),
                    "description": str(r.get("description") or ""),
                    "qty": float(r.get("quantity") or 0.0),
                }
                for r in rows
                if str(r.get("item_id") or "") != item_id
            ]
    return out
