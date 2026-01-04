"""
Tool implementations for Agentic MCP Server.
"""
from .mongodb_tools import (
    mongodb_query_sessions,
    mongodb_upsert_insights,
    mongodb_upsert_trends,
    mongodb_upsert_baseline,
    mongodb_upsert_alert,
    mongodb_get_baseline,
    mongodb_get_drift_flag,
)

from .redis_tools import (
    redis_send_to_queue,
    redis_listen_to_queue,
    redis_broadcast_message,
)

from .websocket_tools import (
    websocket_send_alert,
    websocket_send_followup,
)

from .cloudflare_tools import (
    cloudflare_get_stream_url,
    cloudflare_upload_clip,
)

from .retrieval_tools import (
    retrieval_extract_insights,
    retrieval_track_trends,
    retrieval_establish_baseline,
    retrieval_detect_drift,
)

__all__ = [
    # MongoDB
    "mongodb_query_sessions",
    "mongodb_upsert_insights",
    "mongodb_upsert_trends",
    "mongodb_upsert_baseline",
    "mongodb_upsert_alert",
    "mongodb_get_baseline",
    "mongodb_get_drift_flag",
    # Redis
    "redis_send_to_queue",
    "redis_listen_to_queue",
    "redis_broadcast_message",
    # WebSocket
    "websocket_send_alert",
    "websocket_send_followup",
    # Cloudflare
    "cloudflare_get_stream_url",
    "cloudflare_upload_clip",
    # Retrieval
    "retrieval_extract_insights",
    "retrieval_track_trends",
    "retrieval_establish_baseline",
    "retrieval_detect_drift",
]


