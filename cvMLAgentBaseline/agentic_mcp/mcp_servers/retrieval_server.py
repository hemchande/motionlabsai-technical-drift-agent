#!/usr/bin/env python3
"""
MCP Server for Retrieval Agent Operations.

This server exposes retrieval agent tools via the Model Context Protocol.
Run with: python retrieval_server.py
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
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    FormCorrectionRetrievalAgent = None


# Create MCP server
server = Server("retrieval-server")

# Global retrieval agent (initialized on startup)
retrieval_agent: Optional[FormCorrectionRetrievalAgent] = None


# Note: MCP Server doesn't support on_initialize in this version
# Using lazy initialization in call_tool instead


@server.list_tools()
async def list_tools() -> List[Tool]:
    """List all available retrieval agent tools."""
    return [
        Tool(
            name="retrieval_extract_insights",
            description="Extract form issues/insights from sessions. Only saves insights for issues appearing in min_sessions_per_issue or more sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "activity": {"type": "string", "description": "Activity type (e.g., 'gymnastics')"},
                    "technique": {"type": "string", "description": "Technique (e.g., 'back_handspring')"},
                    "athlete_id": {"type": "string", "description": "Optional athlete ID filter"},
                    "min_sessions_per_issue": {"type": "integer", "description": "Minimum sessions for issue to be saved", "default": 3}
                },
                "required": ["activity", "technique"]
            }
        ),
        Tool(
            name="retrieval_track_trends",
            description="Track trends across sessions. Analyzes how form issues change over time.",
            inputSchema={
                "type": "object",
                "properties": {
                    "sessions": {"type": "array", "description": "List of session dictionaries"},
                    "activity": {"type": "string", "description": "Activity type"},
                    "technique": {"type": "string", "description": "Technique"},
                    "athlete_name": {"type": "string", "description": "Optional athlete name"}
                },
                "required": ["sessions", "activity", "technique"]
            }
        ),
        Tool(
            name="retrieval_establish_baseline",
            description="Establish baseline for athlete. Requires min_sessions eligible sessions.",
            inputSchema={
                "type": "object",
                "properties": {
                    "athlete_id": {"type": "string", "description": "Athlete identifier"},
                    "baseline_type": {"type": "string", "description": "Baseline type: 'pre_injury' | 'pre_rehab' | 'post_rehab'", "default": "pre_injury"},
                    "min_sessions": {"type": "integer", "description": "Minimum sessions required", "default": 8},
                    "min_confidence_score": {"type": "number", "description": "Minimum capture confidence", "default": 0.7}
                },
                "required": ["athlete_id"]
            }
        ),
        Tool(
            name="retrieval_detect_drift",
            description="Detect technical drift from baseline. Compares current session metrics against baseline and creates alerts if drift is detected.",
            inputSchema={
                "type": "object",
                "properties": {
                    "athlete_id": {"type": "string", "description": "Athlete identifier"},
                    "session_id": {"type": "string", "description": "Optional specific session ID"},
                    "drift_threshold": {"type": "number", "description": "Z-score threshold (default: 2.0σ)", "default": 2.0},
                    "analyze_multiple_sessions": {"type": "boolean", "description": "Analyze multiple sessions if True", "default": True},
                    "max_sessions": {"type": "integer", "description": "Maximum sessions to analyze", "default": 10}
                },
                "required": ["athlete_id"]
            }
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handle tool calls."""
    global retrieval_agent
    
    # Lazy initialization
    if retrieval_agent is None:
        if not RETRIEVAL_AVAILABLE:
            return [TextContent(
                type="text",
                text=json.dumps({"error": "Retrieval agent not available", "success": False})
            )]
        try:
            Config.validate()
            retrieval_agent = FormCorrectionRetrievalAgent()
        except Exception as e:
            return [TextContent(
                type="text",
                text=json.dumps({"error": f"Retrieval agent initialization failed: {str(e)}", "success": False})
            )]
    
    try:
        # Use the initialized agent
        agent = retrieval_agent
        
        if name == "retrieval_extract_insights":
            sessions = agent.find_sessions_with_form_issues(
                activity=arguments["activity"],
                technique=arguments["technique"],
                athlete_name=None,
                min_sessions_per_issue=arguments.get("min_sessions_per_issue", 3)
            )
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "sessions_found": len(sessions),
                "sessions": [
                    {
                        "session_id": s.get("session_id"),
                        "athlete_id": s.get("athlete_id"),
                        "form_issues": s.get("form_issues", [])
                    }
                    for s in sessions
                ]
            }, default=str))]
        
        elif name == "retrieval_track_trends":
            trends = agent.track_form_issue_trends(
                sessions=arguments["sessions"],
                activity=arguments["activity"],
                technique=arguments["technique"],
                athlete_name=arguments.get("athlete_name"),
                min_sessions_per_issue=3
            )
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "trends_identified": len(trends) if trends else 0,
                "trends": trends or []
            }, default=str))]
        
        elif name == "retrieval_establish_baseline":
            result = agent.establish_baseline(
                athlete_id=arguments["athlete_id"],
                baseline_type=arguments.get("baseline_type", "pre_injury"),
                min_sessions=arguments.get("min_sessions", 8),
                min_confidence_score=arguments.get("min_confidence_score", 0.7)
            )
            
            return [TextContent(type="text", text=json.dumps({
                "success": True,
                "baseline_id": result.get("baseline_id"),
                "metric_count": result.get("metric_count"),
                "session_count": result.get("session_count")
            }, default=str))]
        
        elif name == "retrieval_detect_drift":
            result = agent.detect_technical_drift(
                athlete_id=arguments["athlete_id"],
                session_id=arguments.get("session_id"),
                drift_threshold=arguments.get("drift_threshold", 2.0),
                analyze_multiple_sessions=arguments.get("analyze_multiple_sessions", True),
                max_sessions=arguments.get("max_sessions", 10)
            )
            
            if result:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "drift_detected": True,
                    "drift_count": result.get("drift_count", 0),
                    "alert_id": result.get("alert_id"),
                    "insights": result.get("insights", [])
                }, default=str))]
            else:
                return [TextContent(type="text", text=json.dumps({
                    "success": True,
                    "drift_detected": False
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

