"""Model file storage: Azure Blob when AZURE_STORAGE_CONNECTION_STRING is set, else local disk.

Callers pass both a blob key (e.g. "tenants/<schema>/model.joblib") and the
local path used when Blob is not configured (dev / tests).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _connection_string() -> str:
    return os.getenv("AZURE_STORAGE_CONNECTION_STRING", "").strip()


def blob_enabled() -> bool:
    return bool(_connection_string())


@lru_cache(maxsize=1)
def _container():
    from azure.storage.blob import BlobServiceClient

    container = os.getenv("AZURE_STORAGE_CONTAINER", "models").strip()
    return BlobServiceClient.from_connection_string(_connection_string()).get_container_client(container)


def read_bytes(key: str, local_path: Path) -> bytes | None:
    """File contents, or None if it doesn't exist."""
    if blob_enabled():
        from azure.core.exceptions import ResourceNotFoundError

        try:
            return _container().download_blob(key).readall()
        except ResourceNotFoundError:
            return None
    return local_path.read_bytes() if local_path.exists() else None


def write_bytes(key: str, local_path: Path, data: bytes) -> None:
    if blob_enabled():
        _container().upload_blob(key, data, overwrite=True)
        return
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(data)
