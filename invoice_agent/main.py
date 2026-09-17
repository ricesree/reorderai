"""Invoice-parsing API: upload a PDF, get back structured invoice JSON.

Uses OpenAI's Responses API - native PDF input + structured outputs
(json_schema, strict mode) so the model literally cannot return anything
that doesn't match the schema.

Two-phase pipeline:
  1. One call reads the PDF and extracts everything raw - line items keep
     their as-printed quantity/price text (case counts, pack sizes, case
     prices and all), plus the invoice-level totals.
  2. For each extracted line item, a separate call normalizes it into the
     target format (total units, per-unit price).
The two results are then merged into the final response.

No database lookups - vendor_id / product_id / unit_of_measurement_id are
always null (there's nothing here to match them against yet).
"""
import base64
import difflib
import json
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from openai import OpenAI

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

UOM_REFERENCE_PATH = Path(__file__).parent / "uom_reference.json"
UOM_REFERENCE = json.loads(UOM_REFERENCE_PATH.read_text()) if UOM_REFERENCE_PATH.exists() else {}


def normalize_product_name(name: str) -> str:
    return re.sub(r"\s+", " ", name).strip().upper()


def lookup_uom(product_name: str) -> tuple[str, str] | None:
    """Find this product in the reference list. Returns (matched_name, 'Unit'|'Weight') or None."""
    key = normalize_product_name(product_name)
    if key in UOM_REFERENCE:
        return key, UOM_REFERENCE[key]
    close = difflib.get_close_matches(key, UOM_REFERENCE.keys(), n=1, cutoff=0.85)
    if close:
        return close[0], UOM_REFERENCE[close[0]]
    return None


MODEL = os.environ.get("OPENAI_MODEL", "gpt-5")
# normalize_item is pure arithmetic on already-extracted text, not document reasoning -
# a small fast model handles it fine and is far cheaper/quicker than the main model.
NORMALIZE_MODEL = os.environ.get("OPENAI_NORMALIZE_MODEL", "gpt-4.1-mini")

# --- Phase 1: raw extraction from the PDF ----------------------------------

RAW_EXTRACTION_INSTRUCTIONS = """You're an expert invoice reader. Read the given PDF invoice and record everything you find, exactly as printed - do NOT do any unit/price math yet.

Fields to extract:
- po_number: The primary reference number for this invoice/order. Look for labels such as "Invoice Number", "Invoice #", "PO Number", "PO #", "Order Number", "Reference Number", "Document Number", or similar, usually near the top. Prefer a PO Number over an Invoice Number if both are present.
- vendor_name: Vendor/seller name. If not explicitly labeled, infer it from the invoice branding/letterhead.
- order_items: Every line item, as printed:
  - name: ONLY the product name/description text itself, including any pack-size text that's actually part of that description (e.g. "12 x 16oz", "24ct"). Do NOT append values from other columns (quantity, weight, price, port of origin, etc) onto the name - each of those goes in its own field below, even if the PDF's layout puts them visually close to the description.
  - raw_quantity: the quantity exactly as shown in the quantity/QTY column (e.g. "3", "3 cases", "20", "72 ea", "1"). This is the box/case count when the invoice bills by case or box, not necessarily the final unit/weight count.
  - raw_units_per_pack: if the invoice has a SEPARATE column/field stating how much is in each case/box/pack, put just the number here as a string - whether that column expresses a unit count (e.g. "12") or a weight per box/case (e.g. a column showing "10 (LB)" - extract just "10"). Otherwise null - even if a pack size is embedded in the name text like "24ct" (that's covered by `name`, don't duplicate it here).
  - raw_price: the price exactly as shown (often an "EA" or "Unit Price" column), including whether it's labeled as a case/box price or a per-unit/per-weight price if that's indicated.
  - raw_total: the line's total/extended amount exactly as shown (e.g. an "Amount" or "Ext Price" column), else null.
  - barcode: UPC/barcode/SKU if present anywhere for this item (product name, description, or a dedicated field), else null.
- tax_amount: Total tax amount.
- shipping_handling_amount: Total shipping/handling/freight amount (see rule below).
- price_adjustment: Any discount/price adjustment applied to the invoice.
- invoice_total: Total amount of the invoice.

# Shipping & Handling Rule

Vendors rarely use the exact label "Shipping & Handling". Look through the totals/summary section for ANY extra charge line that isn't the item subtotal, tax, or a discount - e.g. "Freight", "Freight per Pallet", "Delivery Fee", "Handling Fee", "Fuel Surcharge", "Pallet Charge", "Misc Fee", or similar. Sum all such lines into shipping_handling_amount. Do not include tax or discounts in this sum. Only null if no such line exists.

Return null for any field you cannot find on the invoice, don't guess.
"""

RAW_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "po_number": {"type": ["string", "null"]},
        "vendor_name": {"type": ["string", "null"]},
        "order_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Original product name/description as printed"},
                    "raw_quantity": {"type": "string", "description": "Quantity exactly as printed, e.g. '3 cases'"},
                    "raw_units_per_pack": {"type": ["string", "null"], "description": "Units-per-case/box from a separate invoice field, if shown"},
                    "raw_price": {"type": "string", "description": "Price exactly as printed, e.g. '$47.88/case'"},
                    "raw_total": {"type": ["string", "null"], "description": "Line's total/extended amount as printed, if shown"},
                    "barcode": {"type": ["string", "null"]},
                },
                "required": ["name", "raw_quantity", "raw_units_per_pack", "raw_price", "raw_total", "barcode"],
                "additionalProperties": False,
            },
        },
        "tax_amount": {"type": ["number", "null"]},
        "shipping_handling_amount": {"type": ["number", "null"]},
        "price_adjustment": {"type": ["number", "null"]},
        "invoice_total": {"type": ["number", "null"]},
    },
    "required": [
        "po_number", "vendor_name", "order_items",
        "tax_amount", "shipping_handling_amount", "price_adjustment", "invoice_total",
    ],
    "additionalProperties": False,
}

# --- Phase 2: per-item normalization ----------------------------------------

NORMALIZE_ITEM_INSTRUCTIONS = """You're given one raw invoice line item as JSON. Normalize it into the target format.

Rules:
- Figure out the units-per-pack multiplier first, in this priority order:
  1. If raw_units_per_pack is given, use that number.
  2. Otherwise, look for a pack size embedded in the name (e.g. "12 x 16oz" -> 12, "24ct" -> 24, "6 pack" -> 6).
  3. Otherwise, there is no multiplier - raw_quantity already IS the total unit count (multiplier = 1).
- When the name contains a "N x M <size>" pattern (e.g. "12 x 2 lb", "20 x 170 g", "10 x 800 Gms/Case"), N (the case pack count) is the ENTIRE multiplier. M is only the size/weight of each individual unit, describing what one unit is - it is never a second multiplier, and must NOT be multiplied into N. This holds even if reference_unit_of_measure is "Weight": that means each unit is sold and counted individually (a 2 lb bag, a 170 g pack), not that the case's combined weight becomes the quantity.
  Example: name "Anand Muruku Mini - 20 x 170 g", raw_quantity "1" -> multiplier 20 (NOT 20*170=3400) -> quantity: 20
  Example: name "Dharti Star Anise (Flower) 20 x 50Gms/Case", raw_quantity "1" -> multiplier 20 (NOT 20*50=1000) -> quantity: 20
- quantity = raw_quantity * multiplier. This must be the actual total unit count or weight, NEVER the case/box count.
  Example: raw_quantity "3 cases", raw_units_per_pack "24" -> quantity: 72
  Example: name "Water 24ct", raw_quantity "3 cases", raw_units_per_pack null -> quantity: 72 (multiplier taken from the name)
- price = the per-unit (or per-weight-unit) price = raw_price / multiplier. If raw_price already describes a single unit (multiplier is 1), pass it through as a number unchanged.
  Example: raw_price "$12.00/case", multiplier 24 -> price: 0.50
- total = the line's total dollar amount. If raw_total is given, convert it straight to a number. Otherwise compute quantity * price, rounded to 2 decimals.
- Keep barcode as given (or null).
- Never output a case/box quantity as quantity, or a case/box price as price.
- quantity_confident: true only if the multiplier above is directly supported - raw_units_per_pack was given, or the name has an unambiguous "N x M <size>" pattern. Set to false whenever you had to guess: a vague unit-of-sale word with no stated count (e.g. "case", "bunch", "box", "lot", "pallet"), conflicting signals between raw_quantity/raw_units_per_pack/name, or reference_unit_of_measure disagreeing with your own read of the item.

# Reference product catalog

If a `reference_unit_of_measure` field is present, it comes from our product master and takes priority over your own guess about this product:
- "Unit" means this product is sold and counted as discrete units (each, box, pack, etc) - quantity must be a whole/discrete count, never a weight.
- "Weight" means this product is sold by weight - quantity must be the TOTAL WEIGHT (lb, oz, kg, g - whatever unit the invoice uses), never a discrete box/each count.
  Example: raw_quantity "1", raw_units_per_pack "10" (a "10 (LB)" weight-per-box column), reference_unit_of_measure "Weight" -> quantity: 10 (total lb), price = raw_price / 10.
If `reference_unit_of_measure` is absent or null, no catalog match was found - fall back to your own judgment from the name/price/quantity text.
"""

NORMALIZE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "quantity": {"type": "number", "description": "Total unit count, or total weight (e.g. total lbs) for items sold by weight - never a case/box count"},
        "price": {"type": "number", "description": "Per-unit price, normalized from case price if applicable"},
        "total": {"type": "number", "description": "Line total dollar amount for this item"},
        "barcode": {"type": ["string", "null"]},
        "quantity_confident": {"type": "boolean", "description": "False if the unit-of-sale/multiplier was ambiguous (e.g. case, bunch, lot) and quantity may be wrong"},
    },
    "required": ["name", "quantity", "price", "total", "barcode", "quantity_confident"],
    "additionalProperties": False,
}


def extract_json_output(response) -> dict:
    """Parse the structured JSON text out of an OpenAI Responses API result."""
    return json.loads(response.output_text)


def response_tokens(response) -> int:
    """Total tokens billed for one Responses API call (0 if usage is unavailable, e.g. in tests)."""
    usage = getattr(response, "usage", None)
    return getattr(usage, "total_tokens", 0) or 0


def extract_raw_invoice(client: OpenAI, pdf_bytes: bytes) -> tuple[dict, int]:
    b64 = base64.standard_b64encode(pdf_bytes).decode("utf-8")
    response = client.responses.create(
        model=MODEL,
        instructions=RAW_EXTRACTION_INSTRUCTIONS,
        input=[{
            "role": "user",
            "content": [
                {
                    "type": "input_file",
                    "filename": "invoice.pdf",
                    "file_data": f"data:application/pdf;base64,{b64}",
                },
                {"type": "input_text", "text": "Extract this invoice, raw."},
            ],
        }],
        text={"format": {
            "type": "json_schema",
            "name": "extract_invoice_raw",
            "schema": RAW_EXTRACTION_SCHEMA,
            "strict": True,
        }},
        reasoning={"effort": "low"},
    )
    return extract_json_output(response), response_tokens(response)


def normalize_item(client: OpenAI, raw_item: dict) -> tuple[dict, int]:
    match = lookup_uom(raw_item["name"])
    enriched = {
        **raw_item,
        "reference_unit_of_measure": match[1] if match else None,
    }

    response = client.responses.create(
        model=NORMALIZE_MODEL,
        instructions=NORMALIZE_ITEM_INSTRUCTIONS,
        input=json.dumps(enriched),
        text={"format": {
            "type": "json_schema",
            "name": "normalize_item",
            "schema": NORMALIZE_ITEM_SCHEMA,
            "strict": True,
        }},
    )
    return extract_json_output(response), response_tokens(response)


def _normalize_item_logged(client: OpenAI, item: dict, i: int, n: int) -> tuple[dict, int]:
    t1 = time.monotonic()
    normalized, tokens = normalize_item(client, item)
    print(f"[parse] normalized item {i}/{n} in {time.monotonic()-t1:.1f}s: {item['name']!r}", flush=True)
    return {**normalized, "product_id": None, "unit_of_measurement_id": None}, tokens


def parse_invoice_pdf(client: OpenAI, pdf_bytes: bytes) -> dict:
    t0 = time.monotonic()
    print(f"[parse] extracting raw invoice ({len(pdf_bytes)} bytes)...", flush=True)
    raw, raw_tokens = extract_raw_invoice(client, pdf_bytes)
    n = len(raw["order_items"])
    print(f"[parse] raw extraction done in {time.monotonic()-t0:.1f}s - {n} item(s)", flush=True)

    with ThreadPoolExecutor(max_workers=8) as pool:
        normalized = list(pool.map(
            lambda args: _normalize_item_logged(client, *args),
            [(item, i, n) for i, item in enumerate(raw["order_items"], 1)],
        ))
    normalized_items = [item for item, _tokens in normalized]
    tokens_used = raw_tokens + sum(tokens for _item, tokens in normalized)

    return {
        "po_number": raw["po_number"],
        "vendor_name": raw["vendor_name"],
        "vendor_id": None,
        "order_items": normalized_items,
        "tax_amount": raw["tax_amount"],
        "shipping_handling_amount": raw["shipping_handling_amount"],
        "price_adjustment": raw["price_adjustment"],
        "invoice_total": raw["invoice_total"],
        "tokens_used": tokens_used,
    }


def save_result(filename: str, result: dict) -> dict:
    record = {
        "id": f"{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}",
        "filename": filename,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "result": result,
    }
    (RESULTS_DIR / f"{record['id']}.json").write_text(json.dumps(record, indent=2))
    return record


def list_results() -> list:
    summaries = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        record = json.loads(path.read_text())
        r = record["result"]
        summaries.append({
            "id": record["id"],
            "filename": record["filename"],
            "saved_at": record["saved_at"],
            "po_number": r.get("po_number"),
            "vendor_name": r.get("vendor_name"),
            "invoice_total": r.get("invoice_total"),
        })
    return summaries


def load_result(result_id: str) -> dict | None:
    path = (RESULTS_DIR / f"{result_id}.json").resolve()
    if not path.is_file() or path.parent != RESULTS_DIR.resolve():
        return None
    return json.loads(path.read_text())


app = FastAPI(title="Invoice Parsing Agent")

INDEX_HTML = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Invoice Parsing Agent</title>
<style>
  html, body { width: 100vw; min-height: 100vh; margin: 0; }
  body { font-family: system-ui, sans-serif; padding: 40px 32px; box-sizing: border-box; }
  #drop { border: 2px dashed #999; border-radius: 8px; padding: 40px; text-align: center; cursor: pointer; }
  #drop.hover { border-color: #333; background: #f5f5f5; }
  button { margin-top: 16px; padding: 8px 16px; font-size: 1rem; }
  #status { margin-top: 12px; color: #555; }
  #history { margin-top: 40px; }
  #history table { width: 100%; border-collapse: collapse; }
  #history th, #history td { text-align: left; padding: 6px 8px; border-bottom: 1px solid #ddd; font-size: 0.9rem; }
  #history tr { cursor: pointer; }
  #history tr:hover { background: #f5f5f5; }

  .po-card { border: 1px solid #e3e6ea; border-radius: 8px; overflow: hidden; }
  .po-header { display: flex; align-items: center; gap: 16px; padding: 20px 24px; }
  .po-header .icon { width: 44px; height: 44px; border-radius: 50%; background: #e6f4ea; display: flex; align-items: center; justify-content: center; font-size: 20px; flex-shrink: 0; }
  .po-header h2 { margin: 0; font-size: 1.05rem; }
  .po-header p { margin: 2px 0 0; color: #6b7280; font-size: 0.85rem; }
  .po-table-wrap { overflow-x: auto; }
  table.po-table { width: 100%; border-collapse: collapse; min-width: 900px; }
  table.po-table thead th { background: #f4f6f9; text-align: left; padding: 12px 16px; font-size: 0.8rem; color: #4b5563; border-bottom: 1px solid #e3e6ea; white-space: nowrap; }
  table.po-table tbody td { padding: 12px 16px; border-bottom: 1px solid #eef0f2; font-size: 0.9rem; vertical-align: middle; }
  table.po-table tbody tr:last-child td { border-bottom: none; }
  .prod-cell { display: flex; align-items: center; gap: 12px; }
  .prod-thumb { width: 40px; height: 40px; border-radius: 6px; background: #f1f3f5; display: flex; align-items: center; justify-content: center; font-size: 18px; flex-shrink: 0; }
  .prod-name { font-weight: 600; }
  .prod-cat { color: #9ca3af; font-size: 0.78rem; }
  .num { text-align: left; white-space: nowrap; }
  .po-summary { display: flex; flex-direction: column; align-items: flex-end; gap: 6px; padding: 16px 24px 24px; }
  .po-summary-row { display: flex; gap: 40px; min-width: 220px; justify-content: space-between; font-size: 0.9rem; color: #4b5563; }
  .po-summary-total { border-top: 1px solid #e3e6ea; margin-top: 4px; padding-top: 8px; font-weight: 700; font-size: 1rem; color: #111; }
</style>
</head>
<body>
  <h1>Invoice Parsing Agent</h1>
  <div id="drop">Drop a PDF here or click to choose one</div>
  <input type="file" id="file" accept="application/pdf" hidden>
  <button id="go" disabled>Parse invoice</button>
  <div id="status"></div>
  <div id="result" hidden></div>

  <div id="history">
    <h2>Past results</h2>
    <table id="historyTable">
      <thead><tr><th>Saved</th><th>File</th><th>Vendor</th><th>PO #</th><th>Total</th></tr></thead>
      <tbody></tbody>
    </table>
  </div>

<script>
const drop = document.getElementById('drop');
const fileInput = document.getElementById('file');
const goBtn = document.getElementById('go');
const status = document.getElementById('status');
const result = document.getElementById('result');
let selected = null;

drop.onclick = () => fileInput.click();
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add('hover'); };
drop.ondragleave = () => drop.classList.remove('hover');
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove('hover');
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
};
fileInput.onchange = () => { if (fileInput.files.length) setFile(fileInput.files[0]); };

function setFile(f) {
  selected = f;
  drop.textContent = f.name;
  goBtn.disabled = false;
}

function esc(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function money(n) {
  return '$' + Number(n ?? 0).toFixed(2);
}

function renderResult(data) {
  const items = data.order_items || [];
  const rows = items.map(item => `
    <tr>
      <td>
        <div class="prod-cell">
          <div class="prod-thumb">📦</div>
          <div>
            <div class="prod-name">${esc(item.name)}</div>
            <div class="prod-cat">SKU: ${esc(item.barcode) || '—'}</div>
          </div>
        </div>
      </td>
      <td>${esc(item.barcode) || '—'}</td>
      <td>${esc(item.unit_of_measurement_id) || '—'}</td>
      <td class="num">${item.quantity}</td>
      <td class="num">0</td>
      <td class="num">${money(item.price)}</td>
      <td class="num">${money(item.total)}</td>
      <td class="num">0%</td>
      <td class="num">${money(0)}</td>
      <td class="num">${money(item.total)}</td>
    </tr>
  `).join('');

  const subtotal = items.reduce((sum, item) => sum + Number(item.total ?? 0), 0);
  const summaryRows = [
    ['Subtotal', subtotal],
    ['Tax', data.tax_amount],
    ['Shipping & Handling', data.shipping_handling_amount],
    ['Price Adjustment', data.price_adjustment],
  ].filter(([, v]) => v != null);

  result.innerHTML = `
    <div class="po-card">
      <div class="po-header">
        <div class="icon">🛒</div>
        <div>
          <h2>Products</h2>
          <p>PO ${esc(data.po_number) || '—'} · ${esc(data.vendor_name) || 'Unknown vendor'} · Total ${money(data.invoice_total)}</p>
        </div>
      </div>
      <div class="po-table-wrap">
        <table class="po-table">
          <thead>
            <tr>
              <th>Product</th><th>SKU</th><th>UoM</th><th>Ordered Qty</th><th>Received QTY</th>
              <th>Unit Price</th><th>Sub Total</th><th>Tax Rate</th><th>Tax Amount</th><th>Total Price</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      </div>
      <div class="po-summary">
        ${summaryRows.map(([label, v]) => `<div class="po-summary-row"><span>${label}</span><span>${money(v)}</span></div>`).join('')}
        <div class="po-summary-row po-summary-total"><span>Total</span><span>${money(data.invoice_total)}</span></div>
      </div>
    </div>
  `;
}

goBtn.onclick = async () => {
  if (!selected) return;
  goBtn.disabled = true;
  status.textContent = 'Parsing... this can take a bit.';
  result.hidden = true;

  const formData = new FormData();
  formData.append('file', selected);

  try {
    const res = await fetch('/parse-invoice', { method: 'POST', body: formData });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || res.statusText);
    renderResult(data);
    result.hidden = false;
    status.textContent = 'Done.';
    loadHistory();
  } catch (err) {
    status.textContent = 'Error: ' + err.message;
  } finally {
    goBtn.disabled = false;
  }
};

async function loadHistory() {
  const rows = await (await fetch('/invoices')).json();
  const tbody = document.querySelector('#historyTable tbody');
  tbody.innerHTML = rows.map(r => `
    <tr data-id="${r.id}">
      <td>${new Date(r.saved_at).toLocaleString()}</td>
      <td>${r.filename}</td>
      <td>${r.vendor_name ?? ''}</td>
      <td>${r.po_number ?? ''}</td>
      <td>${r.invoice_total ?? ''}</td>
    </tr>
  `).join('');
  tbody.querySelectorAll('tr').forEach(tr => tr.onclick = async () => {
    const record = await (await fetch('/invoices/' + tr.dataset.id)).json();
    renderResult(record.result);
    result.hidden = false;
    status.textContent = 'Showing saved result from ' + new Date(record.saved_at).toLocaleString();
  });
}
loadHistory();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return INDEX_HTML


@app.post("/parse-invoice")
async def parse_invoice(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        raise HTTPException(400, "Only application/pdf uploads are supported")

    pdf_bytes = await file.read()
    client = OpenAI(timeout=120)  # reads OPENAI_API_KEY from env

    try:
        result = parse_invoice_pdf(client, pdf_bytes)
    except (KeyError, json.JSONDecodeError) as e:
        raise HTTPException(502, f"Model response didn't match the expected format: {e}")

    record = save_result(file.filename or "invoice.pdf", result)
    return JSONResponse({**result, "_id": record["id"]})


@app.get("/invoices")
async def get_invoices():
    return JSONResponse(list_results())


@app.get("/invoices/{result_id}")
async def get_invoice(result_id: str):
    record = load_result(result_id)
    if record is None:
        raise HTTPException(404, "No saved result with that id")
    return JSONResponse(record)
