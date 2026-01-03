"""
Retrieval Sub-Agent for Technical Drift Detection.

Specialized agent that handles insights extraction, trend tracking, baseline establishment, and drift detection.
"""
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from langchain.agents import create_agent
from langchain.tools import tool

from config import Config

try:
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    RETRIEVAL_AVAILABLE = True
except ImportError:
    RETRIEVAL_AVAILABLE = False
    FormCorrectionRetrievalAgent = None


# Retrieval Tools
@tool
def retrieval_extract_insights(
    activity: str,
    technique: str,
    athlete_id: Optional[str] = None,
    min_sessions_per_issue: int = 3
) -> str:
    """
    Extract form issues/insights from sessions.
    
    Identifies form issues from session metrics.
    Only saves insights for issues appearing in min_sessions_per_issue or more sessions.
    Returns JSON with sessions_found and sessions array.
    """
    if not RETRIEVAL_AVAILABLE:
        import json
        return json.dumps({"error": "Retrieval agent not available", "success": False})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        sessions = agent.find_sessions_with_form_issues(
            activity=activity,
            technique=technique,
            athlete_name=None,
            min_sessions_per_issue=min_sessions_per_issue
        )
        
        import json
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
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def retrieval_track_trends(
    sessions: List[Dict[str, Any]],
    activity: str,
    technique: str,
    athlete_name: Optional[str] = None
) -> str:
    """
    Track trends across sessions.
    
    Analyzes how form issues change over time across multiple sessions.
    Returns JSON with trends_identified and trends array.
    """
    if not RETRIEVAL_AVAILABLE:
        import json
        return json.dumps({"error": "Retrieval agent not available", "success": False})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        trends = agent.track_form_issue_trends(
            sessions=sessions,
            activity=activity,
            technique=technique,
            athlete_name=athlete_name,
            min_sessions_per_issue=3
        )
        
        import json
        return json.dumps({
            "success": True,
            "trends_identified": len(trends) if trends else 0,
            "trends": trends or []
        }, default=str)
    except Exception as e:
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def retrieval_establish_baseline(
    athlete_id: str,
    baseline_type: str = "pre_injury",
    min_sessions: int = 8,
    min_confidence_score: float = 0.7
) -> str:
    """
    Establish baseline for athlete.
    
    Calculates and saves a movement baseline for an athlete.
    Requires min_sessions eligible sessions.
    Returns JSON with baseline_id, metric_count, and session_count.
    """
    if not RETRIEVAL_AVAILABLE:
        import json
        return json.dumps({"error": "Retrieval agent not available", "success": False})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        result = agent.establish_baseline(
            athlete_id=athlete_id,
            baseline_type=baseline_type,
            min_sessions=min_sessions,
            min_confidence_score=min_confidence_score
        )
        
        import json
        return json.dumps({
            "success": True,
            "baseline_id": result.get("baseline_id"),
            "metric_count": result.get("metric_count"),
            "session_count": result.get("session_count")
        }, default=str)
    except Exception as e:
        import json
        return json.dumps({"error": str(e), "success": False})


@tool
def retrieval_detect_drift(
    athlete_id: str,
    session_id: Optional[str] = None,
    drift_threshold: float = 2.0,
    analyze_multiple_sessions: bool = True,
    max_sessions: int = 10
) -> str:
    """
    Detect technical drift from baseline.
    
    Compares current session metrics against baseline and detects drift.
    Creates alerts if drift is detected.
    Returns JSON with drift_detected flag, drift_count, and alert_id if drift found.
    """
    if not RETRIEVAL_AVAILABLE:
        import json
        return json.dumps({"error": "Retrieval agent not available", "success": False})
    
    try:
        agent = FormCorrectionRetrievalAgent()
        result = agent.detect_technical_drift(
            athlete_id=athlete_id,
            session_id=session_id,
            drift_threshold=drift_threshold,
            analyze_multiple_sessions=analyze_multiple_sessions,
            max_sessions=max_sessions
        )
        
        import json
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
        import json
        return json.dumps({"error": str(e), "success": False})


# Retrieval Sub-Agent Prompt
RETRIEVAL_AGENT_PROMPT = (
    "You are a technical drift detection and analysis assistant. "
    "Your role is to extract insights, track trends, establish baselines, and detect drift. "
    "You handle: form issue extraction, trend analysis, baseline calculation, and drift detection. "
    "When extracting insights, only issues appearing in 3+ sessions are saved. "
    "When establishing baselines, you need 8+ eligible sessions. "
    "When detecting drift, compare current metrics to baseline using z-scores. "
    "Always confirm what was found or created in your final response."
)


def create_retrieval_agent(model):
    """Create the retrieval sub-agent."""
    return create_agent(
        model,
        tools=[
            retrieval_extract_insights,
            retrieval_track_trends,
            retrieval_establish_baseline,
            retrieval_detect_drift,
        ],
        prompt=RETRIEVAL_AGENT_PROMPT,
    )

