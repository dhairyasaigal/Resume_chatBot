"""
Redis cache for RAG retrieval and LLM responses.
Falls back gracefully to no-cache if Redis is unavailable.
"""

import json
import hashlib
import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

# Cache TTLs
RETRIEVAL_TTL = 3600      # 1 hour — retrieved docs for a query
LLM_TTL = 1800            # 30 minutes — LLM answers


@lru_cache(maxsize=1)
def get_redis_client():
    """
    Returns a Redis client. Returns None if Redis is not configured or unavailable.
    """
    from app.config import get_settings
    settings = get_settings()
    redis_url = getattr(settings, "redis_url", None)

    if not redis_url:
        return None

    try:
        import redis
        client = redis.from_url(redis_url, decode_responses=True, socket_timeout=2)
        client.ping()
        logger.info("Redis cache connected.")
        return client
    except Exception as e:
        logger.warning(f"Redis unavailable, running without cache: {e}")
        return None


def _make_key(prefix: str, *parts: str) -> str:
    """SHA256-based cache key to handle long or special-char queries."""
    raw = ":".join(parts)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:16]
    return f"interview_agent:{prefix}:{digest}"


# ── Retrieval cache ───────────────────────────────────────────────────────────

def get_cached_retrieval(query: str) -> dict | None:
    """
    Returns cached retrieval result for a query, or None if not cached.
    Cached value: {"context": str, "sources": list, "docs": list}
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        key = _make_key("retrieval", query.lower().strip())
        value = client.get(key)
        if value:
            logger.debug(f"Cache HIT retrieval: {query[:50]}")
            return json.loads(value)
    except Exception as e:
        logger.warning(f"Redis retrieval get error: {e}")
    return None


def set_cached_retrieval(query: str, context: str, sources: list, docs: list) -> None:
    """Store retrieval results in Redis with TTL."""
    client = get_redis_client()
    if client is None:
        return
    try:
        key = _make_key("retrieval", query.lower().strip())
        value = json.dumps({"context": context, "sources": sources, "docs": docs})
        client.setex(key, RETRIEVAL_TTL, value)
        logger.debug(f"Cache SET retrieval: {query[:50]}")
    except Exception as e:
        logger.warning(f"Redis retrieval set error: {e}")


# ── LLM response cache ────────────────────────────────────────────────────────

def get_cached_answer(query: str, context: str) -> str | None:
    """
    Returns a cached LLM answer for (query, context) pair, or None.
    Context is included in the key so different retrieved docs give different answers.
    """
    client = get_redis_client()
    if client is None:
        return None
    try:
        key = _make_key("llm", query.lower().strip(), context[:500])
        value = client.get(key)
        if value:
            logger.debug(f"Cache HIT llm: {query[:50]}")
            return value
    except Exception as e:
        logger.warning(f"Redis llm get error: {e}")
    return None


def set_cached_answer(query: str, context: str, answer: str) -> None:
    """Store LLM answer in Redis with TTL."""
    client = get_redis_client()
    if client is None:
        return
    try:
        key = _make_key("llm", query.lower().strip(), context[:500])
        client.setex(key, LLM_TTL, answer)
        logger.debug(f"Cache SET llm: {query[:50]}")
    except Exception as e:
        logger.warning(f"Redis llm set error: {e}")


def invalidate_all() -> int:
    """Flush all interview_agent cache keys. Returns count deleted."""
    client = get_redis_client()
    if client is None:
        return 0
    try:
        keys = client.keys("interview_agent:*")
        if keys:
            return client.delete(*keys)
    except Exception as e:
        logger.warning(f"Redis flush error: {e}")
    return 0


def cache_stats() -> dict:
    """Returns basic cache stats."""
    client = get_redis_client()
    if client is None:
        return {"status": "disabled"}
    try:
        retrieval_keys = len(client.keys("interview_agent:retrieval:*"))
        llm_keys = len(client.keys("interview_agent:llm:*"))
        return {
            "status": "connected",
            "retrieval_entries": retrieval_keys,
            "llm_entries": llm_keys,
            "total": retrieval_keys + llm_keys,
        }
    except Exception as e:
        return {"status": f"error: {e}"}
