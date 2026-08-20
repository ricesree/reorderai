"""
Fill products.product_name / brand / size from products.slug via an LLM.

Single id -> looked up + enriched, returned as JSON only (no DB write).
No id -> every product missing one of these fields is enriched and written
back immediately (so a mid-batch failure keeps prior progress).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from openai import OpenAI

from api.repositories.product_repository import ProductRepository

logger = logging.getLogger(__name__)

_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = (
    "You extract product_name, brand, and size from a retail item name. "
    "product_name is the item name with the brand stripped out. "
    "brand is the manufacturer/brand name. "
    "size is the pack/weight/volume (e.g. '12 oz', '6-pack', '1 lb'). "
    "If the item is generic with no identifiable brand or size (e.g. loose "
    "produce like 'BANANA' or 'RED ONION'), return 'n/a' for that field. "
    "Respond with JSON only: {\"product_name\": ..., \"brand\": ..., \"size\": ...}"
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def enrich_from_name(name: str) -> dict[str, str]:
    resp = _get_client().chat.completions.create(
        model=_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": name},
        ],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    return {
        "product_name": str(data.get("product_name") or "n/a"),
        "brand": str(data.get("brand") or "n/a"),
        "size": str(data.get("size") or "n/a"),
    }


def enrich_one(product_id: int, tenant_id: str | None = None) -> dict[str, Any]:
    repo = ProductRepository(tenant_id)
    row = repo.get_by_id(product_id)
    if row is None:
        raise ValueError(f"Product {product_id} not found")
    fields = enrich_from_name(str(row["slug"]))
    return {"id": product_id, **fields}


def enrich_missing(tenant_id: str | None = None, limit: int | None = None) -> dict[str, Any]:
    repo = ProductRepository(tenant_id)
    rows = repo.find_missing(limit=limit)
    logger.info("enrich_missing: tenant=%s starting, %d product(s) to process", tenant_id, len(rows))
    updated: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    for i, row in enumerate(rows, start=1):
        try:
            fields = enrich_from_name(str(row["slug"]))
            repo.update_fields(row["id"], **fields)
            updated.append({"id": row["id"], **fields})
            logger.info("enrich_missing: [%d/%d] id=%s ok", i, len(rows), row["id"])
        except Exception as exc:  # noqa: BLE001 - one bad item must not kill the batch
            failed.append({"id": row["id"], "error": str(exc)})
            logger.warning("enrich_missing: [%d/%d] id=%s failed: %s", i, len(rows), row["id"], exc)
    logger.info("enrich_missing: tenant=%s done, updated=%d failed=%d", tenant_id, len(updated), len(failed))
    return {"updated": len(updated), "items": updated, "failed": failed}
