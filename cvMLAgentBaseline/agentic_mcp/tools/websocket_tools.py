"""
WebSocket Tools for Agentic MCP Server.
"""
import json
import asyncio
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import Config

try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None


class SendAlertInput(BaseModel):
    """Input schema for send_alert tool."""
    alert_data: Dict[str, Any] = Field(..., description="Alert dictionary with athlete_id, session_id, drift_metrics")
    websocket_port: int = Field(8765, description="WebSocket server port")


@tool(args_schema=SendAlertInput)
def websocket_send_alert(
    alert_data: Dict[str, Any],
    websocket_port: int = 8765
) -> str:
    """
    Send drift alert via WebSocket to connected clients.
    
    Use this tool to broadcast alerts in real-time to coaches/PTs.
    """
    if not WEBSOCKETS_AVAILABLE:
        return json.dumps({"error": "WebSockets not available"})
    
    try:
        async def send_alert():
            uri = f"ws://{Config.WEBSOCKET_HOST}:{websocket_port}"
            try:
                async with websockets.connect(uri) as websocket:
                    await websocket.send(json.dumps({
                        "type": "drift_alert",
                        "data": alert_data
                    }))
                    response = await websocket.recv()
                    return json.loads(response)
            except Exception as e:
                return {"error": str(e)}
        
        result = asyncio.run(send_alert())
        return json.dumps({
            "success": "error" not in result,
            "alert_sent": "error" not in result,
            "result": result
        })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def websocket_send_followup(
    followup_data: Dict[str, Any]
) -> str:
    """
    Send coach follow-up action via WebSocket.
    
    Use this tool to send coach decisions (Monitor, Adjust Training, etc.) to the system.
    """
    return json.dumps({"error": "Not yet implemented", "success": False})


