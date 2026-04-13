# RAG/cache.py

import redis
import hashlib
import json

# Connect to local Redis server
client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

CACHE_TTL = 60 * 60 * 24  # 24 hours in seconds


def make_cache_key(topic: str) -> str:
    """
    Creates a consistent hash key from the debate topic.
    Same topic always maps to same key regardless of whitespace/case.
    """
    normalized = topic.strip().lower()
    return "debate:" + hashlib.md5(normalized.encode()).hexdigest()


def get_cached_debate(topic: str) -> dict | None:
    """
    Returns cached debate result if it exists, None otherwise.
    """
    key  = make_cache_key(topic)
    data = client.get(key)

    if data:
        print(f"[Cache] HIT for topic: '{topic}'")
        return json.loads(data)

    print(f"[Cache] MISS for topic: '{topic}'")
    return None


def cache_debate(topic: str, result: dict) -> None:
    """
    Stores debate result in Redis with 24hr expiry.
    """
    key = make_cache_key(topic)
    client.setex(key, CACHE_TTL, json.dumps(result))
    print(f"[Cache] Stored debate for topic: '{topic}'")


def is_redis_available() -> bool:
    """
    Health check — returns False if Redis is down so app doesn't crash.
    """
    try:
        client.ping()
        return True
    except redis.ConnectionError:
        return False