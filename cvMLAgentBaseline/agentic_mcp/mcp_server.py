"""
MCP Server for Agentic Technical Drift Detection.

Registers all services as tools for the LangChain agent.
"""
from typing import List
from langchain_core.tools import BaseTool

from config import Config
from tools import (
    # MongoDB
    mongodb_query_sessions,
    mongodb_upsert_insights,
    mongodb_upsert_trends,
    mongodb_upsert_baseline,
    mongodb_upsert_alert,
    mongodb_get_baseline,
    mongodb_get_drift_flag,
    # Redis
    redis_send_to_queue,
    redis_listen_to_queue,
    redis_broadcast_message,
    # WebSocket
    websocket_send_alert,
    websocket_send_followup,
    # Cloudflare
    cloudflare_get_stream_url,
    cloudflare_upload_clip,
    # Retrieval
    retrieval_extract_insights,
    retrieval_track_trends,
    retrieval_establish_baseline,
    retrieval_detect_drift,
)


class TechnicalDriftMCPServer:
    """
    MCP Server that registers all services as tools for the agent.
    """
    
    def __init__(self):
        """Initialize MCP server and register all tools."""
        self.tools: List[BaseTool] = []
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all services as tools."""
        # MongoDB Tools
        self.tools.extend([
            mongodb_query_sessions,
            mongodb_upsert_insights,
            mongodb_upsert_trends,
            mongodb_upsert_baseline,
            mongodb_upsert_alert,
            mongodb_get_baseline,
            mongodb_get_drift_flag,
        ])
        
        # Redis Tools
        self.tools.extend([
            redis_send_to_queue,
            redis_listen_to_queue,
            redis_broadcast_message,
        ])
        
        # WebSocket Tools
        self.tools.extend([
            websocket_send_alert,
            websocket_send_followup,
        ])
        
        # Cloudflare Tools
        self.tools.extend([
            cloudflare_get_stream_url,
            cloudflare_upload_clip,
        ])
        
        # Retrieval Agent Tools
        self.tools.extend([
            retrieval_extract_insights,
            retrieval_track_trends,
            retrieval_establish_baseline,
            retrieval_detect_drift,
        ])
    
    def get_tools(self) -> List[BaseTool]:
        """Get all registered tools."""
        return self.tools
    
    def get_tool_names(self) -> List[str]:
        """Get names of all registered tools."""
        return [tool.name for tool in self.tools]

