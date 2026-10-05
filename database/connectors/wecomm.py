"""Wecomm Azure PostgreSQL connector — via SSH tunnel or direct."""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from config.settings import DatabaseSettings, get_settings

logger = logging.getLogger(__name__)

_tunnel: Any | None = None  # process-wide singleton — one tunnel, reused by every connector


def _tunnel_enabled() -> bool:
    return os.getenv("USE_SSH_TUNNEL", "false").strip().lower() in {"1", "true", "yes"}


def _get_tunnel(remote_host: str, remote_port: int) -> Any | None:
    """Lazily start (once per process) an SSH tunnel to (remote_host, remote_port)."""
    global _tunnel
    if not _tunnel_enabled():
        return None
    if _tunnel is not None:
        return _tunnel
    from sshtunnel import SSHTunnelForwarder

    ssh_host = os.getenv("SSH_HOST", "")
    ssh_port = int(os.getenv("SSH_PORT", "22"))
    ssh_user = os.getenv("SSH_USER", "")
    ssh_pkey = os.getenv("SSH_PKEY_PATH", "")
    if not ssh_host or not ssh_user or not ssh_pkey:
        raise RuntimeError(
            "USE_SSH_TUNNEL=1 but SSH_HOST/SSH_USER/SSH_PKEY_PATH are not all set"
        )
    tunnel = SSHTunnelForwarder(
        (ssh_host, ssh_port),
        ssh_username=ssh_user,
        ssh_pkey=ssh_pkey,
        remote_bind_address=(remote_host, remote_port),
        local_bind_address=("127.0.0.1", 0),
    )
    tunnel.start()
    atexit.register(tunnel.stop)
    logger.info(
        "SSH tunnel up: 127.0.0.1:%s -> %s via %s", tunnel.local_bind_port, remote_host, ssh_host
    )
    _tunnel = tunnel
    return _tunnel


class WecommDatabaseConnector:
    """
    Shared connector for Wecomm Postgres.

    Locally: DB_HOST=127.0.0.1 DB_PORT=5433 (SSH tunnel started externally), or
    USE_SSH_TUNNEL=1 with SSH_HOST/SSH_PORT/SSH_USER/SSH_PKEY_PATH set, in which
    case this connector opens the tunnel itself and DB_HOST/DB_PORT are the
    Postgres host/port as reached from the SSH server's side.
    Password is passed through SQLAlchemy URL.create (not hand-encoded).
    """

    def __init__(self, settings: DatabaseSettings | None = None) -> None:
        self._settings = settings or get_settings().database
        self._engine: Engine | None = None
        logger.info("WecommDatabaseConnector init: %s", self._settings.safe_debug())

    @property
    def engine(self) -> Engine:
        if self._engine is None:
            url = self._settings.sqlalchemy_url_obj()
            tunnel = _get_tunnel(self._settings.host, self._settings.port)
            if tunnel is not None:
                url = url.set(host="127.0.0.1", port=tunnel.local_bind_port)
            self._engine = create_engine(
                url,
                pool_pre_ping=True,
                connect_args=self._settings.connect_args(),
            )
        return self._engine

    @property
    def is_postgresql(self) -> bool:
        return True

    def execute(self, query: str, params: dict[str, Any] | None = None) -> None:
        with self.engine.begin() as conn:
            conn.execute(text(query), params or {})

    def read_sql(self, query: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        with self.engine.connect() as conn:
            return pd.read_sql(text(query), conn, params=params or {})

    def write_dataframe(
        self,
        df: pd.DataFrame,
        table: str,
        *,
        schema: str = "public",
        if_exists: str = "append",
        chunksize: int = 1000,
    ) -> int:
        if df.empty:
            return 0
        df.to_sql(
            table,
            self.engine,
            schema=schema,
            if_exists=if_exists,
            index=False,
            chunksize=chunksize,
        )
        return len(df)

    def list_tables(self, schema: str = "public") -> pd.DataFrame:
        return self.read_sql(
            """
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_type = 'BASE TABLE'
              AND table_schema NOT IN ('pg_catalog', 'information_schema')
            ORDER BY table_schema, table_name
            """,
        )

    def ping(self) -> dict[str, Any]:
        """Lightweight auth check."""
        df = self.read_sql("SELECT current_user AS username, current_database() AS database")
        return {
            "username": str(df.iloc[0]["username"]),
            "database": str(df.iloc[0]["database"]),
        }

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None


class SandboxDatabaseConnector(WecommDatabaseConnector):
    """Alias — sandbox removed; everything uses Wecomm DB."""


class ProductionDatabaseConnector(WecommDatabaseConnector):
    """Alias — old production connector removed; everything uses Wecomm DB."""
