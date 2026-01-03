"""
Retrieval Agent Tools for Agentic MCP Server.
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import Config

try:
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    FormCorrectionRetrievalAgent = None


class ExtractInsightsInput(BaseModel):
    """Input schema for extract_insights tool."""
    activity: str = Field(..., description="Activity type (e.g., 'gymnastics')")
    technique: str = Field(..., description="Technique (e.g., 'back_handspring')")
    athlete_id: Optional[str] = Field(None, description="Optional athlete ID filter")
    min_sessions_per_issue: int = Field(3, description="Minimum sessions for issue to be saved")


@tool(args_schema=ExtractInsightsInput)
def retrieval_extract_insights(
    activity: str,
    technique: str,
    athlete_id: Optional[str] = None,
    min_sessions_per_issue: int = 3
) -> str:
    """
    Extract form issues/insights from sessions.
    
    Use this tool to identify form issues from session metrics.
    Only saves insights for issues appearing in min_sessions_per_issue or more sessions.
    """
    if not RETRIEVAL_AVAILABLE:
        return json.dumps({"error": "Retrieval agent not available"})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        sessions = agent.find_sessions_with_form_issues(
            activity=activity,
            technique=technique,
            athlete_name=None,
            min_sessions_per_issue=min_sessions_per_issue
        )
        
        return json.dumps({
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
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class TrackTrendsInput(BaseModel):
    """Input schema for track_trends tool."""
    sessions: List[Dict[str, Any]] = Field(..., description="List of session dictionaries")
    activity: str = Field(..., description="Activity type")
    technique: str = Field(..., description="Technique")
    athlete_name: Optional[str] = Field(None, description="Optional athlete name")


@tool(args_schema=TrackTrendsInput)
def retrieval_track_trends(
    sessions: List[Dict[str, Any]],
    activity: str,
    technique: str,
    athlete_name: Optional[str] = None
) -> str:
    """
    Track trends across sessions.
    
    Use this tool to analyze how form issues change over time across multiple sessions.
    """
    if not RETRIEVAL_AVAILABLE:
        return json.dumps({"error": "Retrieval agent not available"})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        trends = agent.track_form_issue_trends(
            sessions=sessions,
            activity=activity,
            technique=technique,
            athlete_name=athlete_name,
            min_sessions_per_issue=3
        )
        
        return json.dumps({
            "success": True,
            "trends_identified": len(trends) if trends else 0,
            "trends": trends or []
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class EstablishBaselineInput(BaseModel):
    """Input schema for establish_baseline tool."""
    athlete_id: str = Field(..., description="Athlete identifier")
    baseline_type: str = Field("pre_injury", description="Baseline type: 'pre_injury' | 'pre_rehab' | 'post_rehab'")
    min_sessions: int = Field(8, description="Minimum sessions required")
    min_confidence_score: float = Field(0.7, description="Minimum capture confidence")


@tool(args_schema=EstablishBaselineInput)
def retrieval_establish_baseline(
    athlete_id: str,
    baseline_type: str = "pre_injury",
    min_sessions: int = 8,
    min_confidence_score: float = 0.7
) -> str:
    """
    Establish baseline for athlete.
    
    Use this tool to calculate and save a movement baseline for an athlete.
    Requires min_sessions eligible sessions.
    """
    if not RETRIEVAL_AVAILABLE:
        return json.dumps({"error": "Retrieval agent not available"})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        result = agent.establish_baseline(
            athlete_id=athlete_id,
            baseline_type=baseline_type,
            min_sessions=min_sessions,
            min_confidence_score=min_confidence_score
        )
        
        return json.dumps({
            "success": True,
            "baseline_id": result.get("baseline_id"),
            "metric_count": result.get("metric_count"),
            "session_count": result.get("session_count")
        }, default=str)
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})


class DetectDriftInput(BaseModel):
    """Input schema for detect_drift tool."""
    athlete_id: str = Field(..., description="Athlete identifier")
    session_id: Optional[str] = Field(None, description="Optional specific session ID")
    drift_threshold: float = Field(2.0, description="Z-score threshold (default: 2.0σ)")
    analyze_multiple_sessions: bool = Field(True, description="Analyze multiple sessions if True")
    max_sessions: int = Field(10, description="Maximum sessions to analyze")


@tool(args_schema=DetectDriftInput)
def retrieval_detect_drift(
    athlete_id: str,
    session_id: Optional[str] = None,
    drift_threshold: float = 2.0,
    analyze_multiple_sessions: bool = True,
    max_sessions: int = 10
) -> str:
    """
    Detect technical drift from baseline.
    
    Use this tool to compare current session metrics against baseline and detect drift.
    Returns drift results and creates alerts if drift is detected.
    """
    if not RETRIEVAL_AVAILABLE:
        return json.dumps({"error": "Retrieval agent not available"})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        result = agent.detect_technical_drift(
            athlete_id=athlete_id,
            session_id=session_id,
            drift_threshold=drift_threshold,
            analyze_multiple_sessions=analyze_multiple_sessions,
            max_sessions=max_sessions
        )
        
        if result:
            return json.dumps({
                "success": True,
                "drift_detected": True,
                "drift_count": result.get("drift_count", 0),
                "alert_id": result.get("alert_id"),
                "insights": result.get("insights", [])
            }, default=str)
        else:
            return json.dumps({
                "success": True,
                "drift_detected": False
            })
    except Exception as e:
        return json.dumps({"error": str(e), "success": False})

