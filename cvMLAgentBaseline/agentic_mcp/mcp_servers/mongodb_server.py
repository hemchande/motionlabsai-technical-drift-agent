#!/usr/bin/env python3
"""
MCP Server for MongoDB Operations.

This server exposes MongoDB tools via the Model Context Protocol.
Run with: python mongodb_server.py
"""
import sys
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import sys

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
    from videoAgent.mongodb_service import MongoDBService
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoDBService = None


# Create MCP server
server = Server("mongodb-server")

# Global MongoDB connection (initialized on startup)
mongodb_service: Optional[MongoDBService] = None


# Note: MCP Server doesn't support on_initialize in this version
# Using lazy initialization in call_tool instead


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available MongoDB tools."""
    return [
        Tool(
            name="mongodb_query_sessions",
            description="Query sessions collection from MongoDB. Returns sessions matching filters.",
            inputSchema={
                "type": "object",
                "properties": {
                    "athlete_id": {"type": "string", "description": "Athlete ID filter"},
                    "activity": {"type": "string", "description": "Activity filter (e.g., 'gymnastics')"},
                    "technique": {"type": "string", "description": "Technique filter (e.g., 'back_handspring')"},
                    "date_from": {"type": "string", "description": "Start date (ISO format)"},
                    "date_to": {"type": "string", "description": "End date (ISO format)"},
                    "min_confidence": {"type": "number", "description": "Minimum capture_confidence_score", "default": 0.7},
                    "baseline_eligible": {"type": "boolean", "description": "Filter by baseline_eligible flag", "default": True}
                }
            }
        ),
        Tool(
            name="mongodb_upsert_insights",
            description="Upsert insights to MongoDB insights collection.",
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {"type": "string", "description": "Session identifier"},
                    "athlete_id": {"type": "string", "description": "Athlete identifier"},
                    "insights": {"type": "array", "description": "List of insight objects"},
                    "activity": {"type": "string", "description": "Activity type"},
                    "technique": {"type": "string", "description": "Technique performed"}
                },
                "required": ["session_id", "athlete_id", "insights", "activity", "technique"]
            }
        ),
        Tool(
            name="mongodb_get_baseline",
            description="Get active baseline for an athlete from MongoDB.",
            inputSchema={
                "type": "object",
                "properties": {
                    "athlete_id": {"type": "string", "description": "Athlete identifier"}
                },
                "required": ["athlete_id"]
            }
        ),
        Tool(
            name="mongodb_get_drift_flag",
            description="Get drift detection flag for an athlete from MongoDB.",
            inputSchema={
                "type": "object",
                "properties": {
                    "athlete_id": {"type": "string", "description": "Athlete identifier"}
                },
                "required": ["athlete_id"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global mongodb_service
    
    # Lazy initialization
    if mongodb_service is None:
        if not MONGODB_AVAILABLE:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "MongoDB service not available", "success": False})
            )]
        try:
            Config.validate()
            mongodb_service = MongoDBService()
            mongodb_service.connect()
            mongodb_service.get_sessions_collection().find_one()  # Test connection
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"MongoDB initialization failed: {str(e)}", "success": False})
            )]
    
    try:
        # Use the initialized connection
        mongodb = mongodb_service
        
        if name == "mongodb_query_sessions":
            query = {}
            if arguments.get("athlete_id"):
                query["athlete_id"] = arguments["athlete_id"]
            if arguments.get("activity"):
                query["activity"] = arguments["activity"]
            if arguments.get("technique"):
                query["technique"] = arguments["technique"]
            if arguments.get("min_confidence"):
                query["capture_confidence_score"] = {"$gte": arguments["min_confidence"]}
            if arguments.get("baseline_eligible", True):
                query["baseline_eligible"] = True
            
            if arguments.get("date_from") or arguments.get("date_to"):
                query["timestamp"] = {}
                if arguments.get("date_from"):
                    query["timestamp"]["$gte"] = arguments["date_from"]
                if arguments.get("date_to"):
                    query["timestamp"]["$lte"] = arguments["date_to"]
            
            sessions = list(mongodb.get_sessions_collection().find(query).sort("timestamp", -1))
            
            result = {
                "success": True,
                "count": len(sessions),
                "sessions": [
                    {
                        "session_id": str(s.get("_id")),
                        "athlete_id": s.get("athlete_id"),
                        "athlete_name": s.get("athlete_name"),
                        "activity": s.get("activity"),
                        "technique": s.get("technique"),
                        "metrics": s.get("metrics", {}),
                        "timestamp": s.get("timestamp"),
                        "confidence_score": s.get("capture_confidence_score")
                    }
                    for s in sessions
                ]
            }
            
            # Don't close - reuse connection
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        
        elif name == "mongodb_upsert_insights":
            result = mongodb.upsert_insights(
                session_id=arguments["session_id"],
                athlete_id=arguments["athlete_id"],
                insights=arguments["insights"],
                activity=arguments["activity"],
                technique=arguments["technique"]
            )
            # Don't close - reuse connection
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "session_id": arguments["session_id"],
                "insight_count": len(arguments["insights"])
            }))]
        
        elif name == "mongodb_get_baseline":
            baseline = mongodb.get_baseline(athlete_id=arguments["athlete_id"])
            # Don't close - reuse connection
            
            if baseline:
                result = {
                    "success": True,
                    "baseline_exists": True,
                    "baseline_id": str(baseline.get("_id")),
                    "baseline_type": baseline.get("baseline_type"),
                    "status": baseline.get("status"),
                    "metric_count": len(baseline.get("baseline_vector", {}))
                }
            else:
                result = {
                    "success": True,
                    "baseline_exists": False
                }
            
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        
        elif name == "mongodb_get_drift_flag":
            flag = mongodb.get_drift_detection_flag(athlete_id=arguments["athlete_id"])
            # Don't close - reuse connection
            
            if flag:
                result = {
                    "success": True,
                    "flag_exists": True,
                    "drift_detection_enabled": flag.get("drift_detection_enabled", False),
                    "drift_threshold": flag.get("drift_threshold", 2.0),
                    "baseline_id": str(flag.get("baseline_id")) if flag.get("baseline_id") else None
                }
            else:
                result = {
                    "success": True,
                    "flag_exists": False
                }
            
            return [TextContent(type="text", text=json.dumps(result, default=str))]
        
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

