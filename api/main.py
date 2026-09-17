"""
Reorder AI API — detect-order, forecast_store, chatbot.

  uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=True)

from api.routes import chatbot, detect_order, invoice_agent_queue, products, system  # noqa: E402

app = FastAPI(
    title="Reorder AI",
    description=(
        "Wecomm detect-order + forecast_store + investigate chatbot. APIs only."
    ),
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system.router)
app.include_router(detect_order.router)
app.include_router(chatbot.router)
app.include_router(products.router)
app.include_router(invoice_agent_queue.router)


@app.get("/")
async def root() -> dict:
    return {
        "service": "reorder-ai",
        "version": "3.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "GET /api/health",
            "db_health": "GET /api/db-health",
            "list_vendors": "GET /api/detect-order",
            "detect_order": "POST /api/detect-order",
            "order_run": "GET /api/detect-order/runs/{run_id}",
            "order_excel": "GET /api/detect-order/runs/{run_id}/export.xlsx",
            "chatbot_tools": "GET /api/chatbot/tools",
            "chatbot_ask": "POST /api/chatbot/ask",
            "chatbot_tool": "POST /api/chatbot/tool",
            "products_enrich": "POST /api/products/enrich",
            "parse_invoice": "POST /api/invoice-agent/parse-invoice",
            "parse_invoice_status": "GET /api/invoice-agent/parse-invoice/status/{job_id}",
            "nightly_batch": "python scripts/run_nightly_forecast.py",
        },
    }
