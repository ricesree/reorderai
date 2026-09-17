"""Smoke tests - no network calls (OpenAI client is faked)."""
import json
from types import SimpleNamespace

from main import extract_json_output, parse_invoice_pdf


def _response(payload: dict, total_tokens: int = 0):
    return SimpleNamespace(output_text=json.dumps(payload), usage=SimpleNamespace(total_tokens=total_tokens))


def test_extract_json_output_parses_output_text():
    assert extract_json_output(_response({"po_number": "123"})) == {"po_number": "123"}


def test_parse_invoice_pdf_merges_raw_and_normalized_items():
    raw = {
        "po_number": "PO-1",
        "vendor_name": "ACME",
        "order_items": [
            {
                "name": "Water 24ct", "raw_quantity": "3 cases", "raw_units_per_pack": None,
                "raw_price": "$12.00/case", "raw_total": None, "barcode": None,
            },
        ],
        "tax_amount": 5.0,
        "shipping_handling_amount": None,
        "price_adjustment": None,
        "invoice_total": 41.0,
    }
    normalized = {"name": "Water 24ct", "quantity": 72, "price": 0.5, "total": 36.0, "barcode": None, "quantity_confident": True}

    class FakeClient:
        class responses:
            @staticmethod
            def create(**kwargs):
                schema_name = kwargs["text"]["format"]["name"]
                if schema_name == "extract_invoice_raw":
                    return _response(raw, total_tokens=100)
                return _response(normalized, total_tokens=50)

    result = parse_invoice_pdf(FakeClient(), b"%PDF-fake")

    assert result["po_number"] == "PO-1"
    assert result["vendor_id"] is None
    assert result["order_items"] == [
        {
            "name": "Water 24ct", "quantity": 72, "price": 0.5, "total": 36.0, "barcode": None,
            "quantity_confident": True, "product_id": None, "unit_of_measurement_id": None,
        },
    ]
    assert result["invoice_total"] == 41.0
    assert result["tokens_used"] == 150  # 100 (raw extraction) + 50 (one normalized item)


if __name__ == "__main__":
    test_extract_json_output_parses_output_text()
    test_parse_invoice_pdf_merges_raw_and_normalized_items()
    print("ok")
