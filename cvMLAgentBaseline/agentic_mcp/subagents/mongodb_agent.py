"""
MongoDB Sub-Agent for Technical Drift Detection.

Specialized agent that handles all MongoDB operations.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from config import Config

try:
    from videoAgent.mongodb_service import MongoDBService
    MONGODB_AVAILABLE = True
except ImportError:
    MONGODB_AVAILABLE = False
    MongoDBService = None

from langchain.tools import tool


# MongoDB Tools
@tool
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
    
    Returns JSON string with sessions list and metadata.
    Use this to find sessions for an athlete, activity, or technique.
    """
    if not MONGODB_AVAILABLE:
        return '{"error": "MongoDB service not available", "success": false}'
    
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
        
        mongodb.close()
        import json
        return json.dumps(result, default=str)
    except Exception as e:
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def mongodb_upsert_insights(
    session_id: str,
    athlete_id: str,
    insights: List[Dict[str, Any]],
    activity: str,
    technique: str
) -> str:
    """
    Upsert insights to MongoDB insights collection.
    
    Saves extracted form issues/insights for a session.
    """
    if not MONGODB_AVAILABLE:
        return '{"error": "MongoDB service not available", "success": false}'
    
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
        
        import json
        return json.dumps({
            "success": True,
            "session_id": session_id,
            "insight_count": len(insights)
        })
    except Exception as e:
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def mongodb_get_baseline(athlete_id: str) -> str:
    """
    Get active baseline for an athlete from MongoDB.
    
    Use this to check if an athlete has an established baseline.
    Returns JSON with baseline_exists flag and baseline details if found.
    """
    if not MONGODB_AVAILABLE:
        return '{"error": "MongoDB service not available", "success": false}'
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        baseline = mongodb.get_baseline(athlete_id=athlete_id)
        
        mongodb.close()
        
        import json
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
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def mongodb_get_drift_flag(athlete_id: str) -> str:
    """
    Get drift detection flag for an athlete from MongoDB.
    
    Use this to check if drift detection is enabled for an athlete.
    Returns JSON with flag_exists and drift_detection_enabled flags.
    """
    if not MONGODB_AVAILABLE:
        return '{"error": "MongoDB service not available", "success": false}'
    
    try:
        mongodb = MongoDBService()
        mongodb.connect()
        
        flag = mongodb.get_drift_detection_flag(athlete_id=athlete_id)
        
        mongodb.close()
        
        import json
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
        import json
        return json.dumps({"error": str(e), "success": False})


# MongoDB Sub-Agent Prompt
MONGODB_AGENT_PROMPT = (
    "You are a MongoDB database assistant for technical drift detection. "
    "Your role is to query and manage data in MongoDB collections. "
    "You handle: sessions, insights, baselines, and drift detection flags. "
    "Always parse JSON responses from tools and provide clear summaries. "
    "When querying sessions, use appropriate filters (athlete_id, activity, technique). "
    "When checking baselines, verify if baseline_exists is true before proceeding. "
    "Always confirm what data was found or saved in your final response."
)


def create_mongodb_agent(model):
    """Create the MongoDB sub-agent."""
    return create_agent(
        model,
        tools=[
            mongodb_query_sessions,
            mongodb_upsert_insights,
            mongodb_get_baseline,
            mongodb_get_drift_flag,
        ],
        prompt=MONGODB_AGENT_PROMPT,
    )

