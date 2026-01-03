"""
Cloudflare Stream Tools for Agentic MCP Server.
"""
import json
import requests
from typing import Dict, Any
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from config import Config


class GetStreamUrlInput(BaseModel):
    """Input schema for get_stream_url tool."""
    video_id: str = Field(..., description="Cloudflare Stream video ID")


@tool(args_schema=GetStreamUrlInput)
def cloudflare_get_stream_url(video_id: str) -> str:
    """
    Get Cloudflare Stream URL for video.
    
    Use this tool to retrieve stream URLs for video playback.
    """
    if not Config.CLOUDFLARE_ACCOUNT_ID or not Config.CLOUDFLARE_API_TOKEN:
        return json.dumps({"error": "Cloudflare credentials not configured"})
    
    try:
        url = f"https://api.cloudflare.com/client/v4/accounts/{Config.CLOUDFLARE_ACCOUNT_ID}/stream/{video_id}"
        headers = {
            "Authorization": f"Bearer {Config.CLOUDFLARE_API_TOKEN}"
        }
        
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if response.status_code == 200:
            result_data = data.get("result", {})
            return json.dumps({
                "success": True,
                "video_id": video_id,
                "stream_url": result_data.get("playback", {}).get("hls"),
                "thumbnail_url": result_data.get("thumbnail")
            })
        else:
            return json.dumps({
                "error": data.get("errors", [{}])[0].get("message", "Unknown error"),
                "success": False
            })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


@tool
def cloudflare_upload_clip(
    clip_data: Dict[str, Any]
) -> str:
    """
    Upload clip to Cloudflare Stream.
    
    Use this tool to upload video clips for streaming.
    """
    return json.dumps({"error": "Not yet implemented", "success": False})

