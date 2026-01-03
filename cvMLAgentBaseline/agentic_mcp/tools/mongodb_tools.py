"""
MongoDB Tools for Agentic MCP Server.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config

try:
    from videoAgent.mongodb_service import MongoDBService
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoDBService = None


class QuerySessionsInput(BaseModel):
    """Input schema for query_sessions tool."""
    athlete_id: Optional[str] = Field(None, description="Athlete ID filter")
    activity: Optional[str] = Field(None, description="Activity filter (e.g., 'gymnastics')")
    technique: Optional[str] = Field(None, description="Technique filter (e.g., 'back_handspring')")
    date_from: Optional[str] = Field(None, description="Start date (ISO format)")
    date_to: Optional[str] = Field(None, description="End date (ISO format)")
    min_confidence: float = Field(0.7, description="Minimum capture_confidence_score")
    baseline_eligible: bool = Field(True, description="Filter by baseline_eligible flag")


@tool(args_schema=QuerySessionsInput)
def mongodb_query_sessions(
    athlete_id: Optional[str] = None,
    activity: Optional[str] = None,
    technique: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_confidence: float = 0.7,
    baseline_eligible: bool = True
) -> str:
    """
    Query sessions collection from MongoDB.
    
    Use this tool to find sessions for an athlete, activity, or technique.
    Returns JSON string with sessions list and metadata.
    """
    if not MONGODB_AVAILABLE:
        return json.dumps({"error": "MongoDB service not available"})
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        query = {}
        if athlete_id:
            query["athlete_id"] = athlete_id
        if activity:
            query["activity"] = activity
        if technique:
            query["technique"] = technique
        if min_confidence:
            query["capture_confidence_score"] = {"$gte": min_confidence}
        if baseline_eligible:
            query["baseline_eligible"] = baseline_eligible
        
        if date_from or date_to:
            query["timestamp"] = {}
            if date_from:
                query["timestamp"]["$gte"] = date_from
            if date_to:
                query["timestamp"]["$lte"] = date_to
        
        sessions = list(mongodb.get_sessions_collection().find(query).sort("timestamp", -1))
        
        mongodb.close()
        
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
        
        return json.dumps(result, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class UpsertInsightsInput(BaseModel):
    """Input schema for upsert_insights tool."""
    session_id: str = Field(..., description="Session identifier")
    athlete_id: str = Field(..., description="Athlete identifier")
    insights: List[Dict[str, Any]] = Field(..., description="List of insight objects")
    activity: str = Field(..., description="Activity type")
    technique: str = Field(..., description="Technique performed")


@tool(args_schema=UpsertInsightsInput)
def mongodb_upsert_insights(
    session_id: str,
    athlete_id: str,
    insights: List[Dict[str, Any]],
    activity: str,
    technique: str
) -> str:
    """
    Upsert insights to MongoDB insights collection.
    
    Use this tool to save extracted form issues/insights for a session.
    """
    if not MONGODB_AVAILABLE:
        return json.dumps({"error": "MongoDB service not available"})
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        result = mongodb.upsert_insights(
            session_id=session_id,
            athlete_id=athlete_id,
            insights=insights,
            activity=activity,
            technique=technique
        )
        
        mongodb.close()
        
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "insight_count": len(insights)
        })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class GetBaselineInput(BaseModel):
    """Input schema for get_baseline tool."""
    athlete_id: str = Field(..., description="Athlete identifier")


@tool(args_schema=GetBaselineInput)
def mongodb_get_baseline(athlete_id: str) -> str:
    """
    Get active baseline for an athlete from MongoDB.
    
    Use this tool to check if an athlete has an established baseline.
    """
    if not MONGODB_AVAILABLE:
        return json.dumps({"error": "MongoDB service not available"})
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        baseline = mongodb.get_baseline(athlete_id=athlete_id)
        
        mongodb.close()
        
        if baseline:
            return json.dumps({
                "success": True,
                "baseline_exists": True,
                "baseline_id": str(baseline.get("_id")),
                "baseline_type": baseline.get("baseline_type"),
                "status": baseline.get("status"),
                "metric_count": len(baseline.get("baseline_vector", {}))
            }, default=str)
        else:
            return json.dumps({
                "success": True,
                "baseline_exists": False
            })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class GetDriftFlagInput(BaseModel):
    """Input schema for get_drift_flag tool."""
    athlete_id: str = Field(..., description="Athlete identifier")


@tool(args_schema=GetDriftFlagInput)
def mongodb_get_drift_flag(athlete_id: str) -> str:
    """
    Get drift detection flag for an athlete from MongoDB.
    
    Use this tool to check if drift detection is enabled for an athlete.
    """
    if not MONGODB_AVAILABLE:
        return json.dumps({"error": "MongoDB service not available"})
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        flag = mongodb.get_drift_detection_flag(athlete_id=athlete_id)
        
        mongodb.close()
        
        if flag:
            return json.dumps({
                "success": True,
                "flag_exists": True,
                "drift_detection_enabled": flag.get("drift_detection_enabled", False),
                "drift_threshold": flag.get("drift_threshold", 2.0),
                "baseline_id": str(flag.get("baseline_id")) if flag.get("baseline_id") else None
            }, default=str)
        else:
            return json.dumps({
                "success": True,
                "flag_exists": False
            })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


# Placeholder tools (to be implemented)
@tool
def mongodb_upsert_trends(
    trend_data: Dict[str, Any]
) -> str:
    """Upsert trends to MongoDB trends collection."""
    return json.dumps({"error": "Not yet implemented", "success": False})


@tool
def mongodb_upsert_baseline(
    baseline_data: Dict[str, Any]
) -> str:
    """Upsert baseline to MongoDB baselines collection."""
    return json.dumps({"error": "Not yet implemented", "success": False})


@tool
def mongodb_upsert_alert(
    alert_data: Dict[str, Any]
) -> str:
    """Upsert alert to MongoDB alerts collection."""
    return json.dumps({"error": "Not yet implemented", "success": False})

