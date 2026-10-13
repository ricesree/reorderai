"""Pytest configuration — add project root to Python path."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_real_blob_storage(monkeypatch):
    """A real connection string in .env must never send test models to Azure."""
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
