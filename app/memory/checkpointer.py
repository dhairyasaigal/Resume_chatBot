"""
PostgreSQL checkpointer for LangGraph persistent conversation threads.
Uses langgraph-checkpoint-postgres with psycopg3.
"""

import logging
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg_pool import ConnectionPool
from app.config import get_settings

logger = logging.getLogger(__name__)

_pool: ConnectionPool | None = None


def _clean_url(url: str) -> str:
    """Remove params unsupported by psycopg3 (e.g. channel_binding)."""
    import urllib.parse as up
    parsed = up.urlparse(url)
    qs = up.parse_qs(parsed.query, keep_blank_values=True)
    qs.pop("channel_binding", None)
    new_query = up.urlencode({k: v[0] for k, v in qs.items()})
    return parsed._replace(query=new_query).geturl()


def get_connection_pool() -> ConnectionPool:
    """Returns a shared psycopg3 connection pool, initialised on first call."""
    global _pool
    if _pool is None:
        settings = get_settings()
        conninfo = _clean_url(settings.postgres_url)
        _pool = ConnectionPool(conninfo=conninfo, max_size=10, open=True)
    return _pool


def get_checkpointer() -> PostgresSaver:
    """
    Returns a PostgresSaver checkpointer.
    Uses a direct connection for setup() to avoid the
    'CREATE INDEX CONCURRENTLY cannot run inside a transaction block' error
    that occurs when using a connection pool on Neon/managed Postgres.
    """
    settings = get_settings()
    conninfo = _clean_url(settings.postgres_url)

    # Run setup with autocommit=True outside of any transaction
    with psycopg.connect(conninfo, autocommit=True) as conn:
        PostgresSaver(conn).setup()

    # Use pool for actual runtime operations
    pool = get_connection_pool()
    return PostgresSaver(pool)


def close_pool():
    """Gracefully close the connection pool (call on app shutdown)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
