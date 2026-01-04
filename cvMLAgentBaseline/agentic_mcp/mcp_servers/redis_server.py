#!/usr/bin/env python3
"""
MCP Server for Redis Operations.

This server exposes Redis queue tools via the Model Context Protocol.
Run with: python redis_server.py
"""
import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directories to path
# When run as subprocess, need agentic_mcp directory in path
agentic_mcp_dir = Path(__file__).parent.parent
sys.path.insert(0, str(agentic_mcp_dir))
sys.path.insert(0, str(agentic_mcp_dir.parent))

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from config import Config

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None


# Create MCP server
server = Server("redis-server")

# Global Redis connection (initialized on startup)
redis_client: Optional[redis.Redis] = None


# Note: MCP Server doesn't support on_initialize in this version
# Using lazy initialization in call_tool instead


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available Redis tools."""
    return [
        Tool(
            name="redis_send_to_queue",
            description="Send message to Redis queue for inter-service communication.",
            inputSchema={
                "type": "object",
                "properties": {
                    "queue_name": {"type": "string", "description": "Queue name (e.g., 'retrievalQueue', 'drift_alerts_queue')"},
                    "message": {"type": "object", "description": "Message dictionary to send"}
                },
                "required": ["queue_name", "message"]
            }
        ),
        Tool(
            name="redis_listen_to_queue",
            description="Listen to Redis queue and retrieve messages.",
            inputSchema={
                "type": "object",
                "properties": {
                    "queue_name": {"type": "string", "description": "Queue name to listen to"},
                    "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 5},
                    "max_messages": {"type": "integer", "description": "Maximum messages to retrieve", "default": 1}
                },
                "required": ["queue_name"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global redis_client
    
    # Lazy initialization
    if redis_client is None:
        if not REDIS_AVAILABLE:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Redis not available", "success": False})
            )]
        try:
            Config.validate()
            redis_client = redis.Redis(
                host=Config.REDIS_HOST,
                port=Config.REDIS_PORT,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_keepalive=True
            )
            redis_client.ping()  # Test connection
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Redis initialization failed: {str(e)}", "success": False})
            )]
    
    try:
        # Use the initialized connection
        client = redis_client
        
        if name == "redis_send_to_queue":
            message_json = json.dumps(arguments["message"])
            client.lpush(arguments["queue_name"], message_json)
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "queue_name": arguments["queue_name"],
                "message_sent": True
            }))]
        
        elif name == "redis_listen_to_queue":
            messages = []
            max_messages = arguments.get("max_messages", 1)
            timeout = arguments.get("timeout", 5)
            
            for _ in range(max_messages):
                result = client.brpop(arguments["queue_name"], timeout=timeout)
                if result:
                    queue, message_json = result
                    messages.append(json.loads(message_json))
                else:
                    break
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "count": len(messages),
                "messages": messages
            }))]
        
        else:
            return [TextContent(type="text", text=json.dumps({
                "error": f"Unknown tool: {name}",
                "success": False
            }))]
    
    except Exception as e:
        return [TextContent(type="text", text=json.dumps({
            "error": str(e),
            "success": False
        }))]


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

