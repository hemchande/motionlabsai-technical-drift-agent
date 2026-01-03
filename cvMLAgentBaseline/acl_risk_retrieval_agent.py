#!/usr/bin/env python3
"""
ACL Risk Retrieval Agent

Retrieves and analyzes ACL risk patterns from MongoDB sessions.
Detects patterns of MODERATE risk (0.4-0.7) and HIGH risk (>= 0.7).
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import MongoDBService
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

# Import trend tracker from same directory
try:
    from trend_tracker import TrendTracker
    TREND_TRACKER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  TrendTracker not available: {e}")
    TREND_TRACKER_AVAILABLE = False
    TrendTracker = None

# Import guardrails
try:
    from guardrails import Guardrails
    GUARDRAILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️  Guardrails not available: {e}")
    GUARDRAILS_AVAILABLE = False
    Guardrails = None

# Load environment variables
load_dotenv()

# Try to import LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class ACLRiskRetrievalAgent:
    """
    Retrieves and analyzes ACL risk patterns from MongoDB sessions.
    Supports both MODERATE (0.4-0.7) and HIGH (>= 0.7) risk detection.
    """
    
    def __init__(self):
        """Initialize the retrieval agent."""
        self.mongodb = MongoDBService()
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        
        # Initialize trend tracker
        if TREND_TRACKER_AVAILABLE:
            try:
                self.trend_tracker = TrendTracker()
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize TrendTracker: {e}")
                self.trend_tracker = None
        else:
            self.trend_tracker = None
        
        # Initialize guardrails
        if GUARDRAILS_AVAILABLE:
            try:
                self.guardrails = Guardrails()
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Guardrails: {e}")
                self.guardrails = None
        else:
            self.guardrails = None
        
        logger.info("✅ Initialized ACLRiskRetrievalAgent")
    
    def find_risk_sessions(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_risk_score: float = 0.4,
        include_moderate: bool = True,
        include_high: bool = True,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions with moderate or high-risk ACL moments.
        
        Args:
            activity: Filter by activity (e.g., "gymnastics")
            technique: Filter by technique (e.g., "back_handspring")
            min_risk_score: Minimum risk score threshold (default: 0.4 for MODERATE+ risk)
            include_moderate: Include MODERATE risk (0.4-0.7) sessions
            include_high: Include HIGH risk (>= 0.7) sessions
            date_from: Filter sessions from this date
            date_to: Filter sessions to this date
        
        Returns:
            List of session documents with risk ACL moments
        """
        collection = self.mongodb.get_sessions_collection()
        
        # Build query - look for sessions with ACL risk data
        # We'll check both flagged timesteps and overall risk scores
        query = {}
        
        # Build risk level filter
        risk_levels = []
        if include_moderate:
            risk_levels.append("MODERATE")
        if include_high:
            risk_levels.append("HIGH")
        
        if risk_levels:
            query["acl_risk_level"] = {"$in": risk_levels}
        
        # Also check for sessions with flagged timesteps or risk scores
        # This catches sessions that may have risk but weren't flagged as high-risk
        if min_risk_score > 0:
            # Include sessions with overall risk score >= threshold
            query["$or"] = [
                {"acl_risk_score": {"$gte": min_risk_score}},
                {"acl_risk_level": {"$in": risk_levels}},
                {"acl_flagged_timesteps": {"$exists": True, "$ne": []}}
            ]
        
        if activity:
            query["activity"] = activity
        
        if technique:
            query["technique"] = technique
        
        if date_from or date_to:
            query["timestamp"] = {}
            if date_from:
                query["timestamp"]["$gte"] = date_from.isoformat()
            if date_to:
                query["timestamp"]["$lte"] = date_to.isoformat()
        
        # Execute query
        sessions = list(collection.find(query))
        
        # Filter by risk score threshold and categorize moments
        filtered_sessions = []
        for session in sessions:
            acl_flagged = session.get("acl_flagged_timesteps", [])
            metrics = session.get("metrics", {})
            overall_risk_score = session.get("acl_risk_score") or metrics.get("acl_tear_risk_score", 0.0)
            risk_level = session.get("acl_risk_level") or metrics.get("acl_risk_level", "MINIMAL")
            
            # Check if session meets criteria
            meets_criteria = False
            if overall_risk_score >= min_risk_score:
                if include_high and overall_risk_score >= 0.7:
                    meets_criteria = True
                elif include_moderate and 0.4 <= overall_risk_score < 0.7:
                    meets_criteria = True
            
            # Also check if risk level matches
            if risk_level in risk_levels:
                meets_criteria = True
            
            # Filter flagged timesteps by threshold
            risk_moments = [
                ts for ts in acl_flagged
                if ts.get("risk_score", 0.0) >= min_risk_score
            ]
            
            # Include session if it meets criteria or has risk moments
            if meets_criteria or risk_moments:
                # Categorize moments
                high_risk_moments = [ts for ts in risk_moments if ts.get("risk_score", 0.0) >= 0.7]
                moderate_risk_moments = [ts for ts in risk_moments if 0.4 <= ts.get("risk_score", 0.0) < 0.7]
                
                session["risk_moments"] = risk_moments
                session["high_risk_moments"] = high_risk_moments
                session["moderate_risk_moments"] = moderate_risk_moments
                session["high_risk_count"] = len(high_risk_moments)
                session["moderate_risk_count"] = len(moderate_risk_moments)
                session["total_risk_count"] = len(risk_moments)
                session["overall_risk_score"] = overall_risk_score
                session["overall_risk_level"] = risk_level
                
                # Convert ObjectId to string
                session["_id"] = str(session["_id"])
                filtered_sessions.append(session)
        
        risk_type = []
        if include_moderate:
            risk_type.append("moderate")
        if include_high:
            risk_type.append("high")
        risk_type_str = " and ".join(risk_type) if risk_type else "risk"
        
        logger.info(f"📊 Found {len(filtered_sessions)} sessions with {risk_type_str}-risk ACL moments (threshold: >= {min_risk_score})")
        return filtered_sessions
    
    def find_high_risk_sessions(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_risk_score: float = 0.7,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions with high-risk ACL moments (convenience method).
        
        Args:
            activity: Filter by activity (e.g., "gymnastics")
            technique: Filter by technique (e.g., "back_handspring")
            min_risk_score: Minimum risk score threshold (default: 0.7 for HIGH risk)
            date_from: Filter sessions from this date
            date_to: Filter sessions to this date
        
        Returns:
            List of session documents with high-risk ACL moments
        """
        return self.find_risk_sessions(
            activity=activity,
            technique=technique,
            min_risk_score=min_risk_score,
            include_moderate=False,
            include_high=True,
            date_from=date_from,
            date_to=date_to
        )
    
    def analyze_risk_patterns(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        include_moderate: bool = True,
        include_high: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze moderate and high-risk ACL patterns across sessions.
        
        Args:
            sessions: Optional list of sessions to analyze (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
            include_moderate: Include MODERATE risk (0.4-0.7) in analysis
            include_high: Include HIGH risk (>= 0.7) in analysis
        
        Returns:
            Dictionary with pattern analysis results
        """
        if sessions is None:
            sessions = self.find_risk_sessions(
                activity=activity,
                technique=technique,
                include_moderate=include_moderate,
                include_high=include_high
            )
        
        if not sessions:
            return {
                "total_sessions": 0,
                "total_risk_moments": 0,
                "total_high_risk_moments": 0,
                "total_moderate_risk_moments": 0,
                "patterns": {},
                "summary": "No risk ACL sessions found"
            }
        
        # Aggregate patterns
        patterns = {
            "by_activity": defaultdict(int),
            "by_technique": defaultdict(int),
            "by_risk_factor": defaultdict(int),
            "by_risk_level": defaultdict(int),  # Track MODERATE vs HIGH
            "risk_score_distribution": [],
            "valgus_angle_distribution": [],
            "flexion_risk_distribution": [],
            "impact_risk_distribution": [],
            "temporal_patterns": []
        }
        
        total_risk_moments = 0
        total_high_risk_moments = 0
        total_moderate_risk_moments = 0
        
        for session in sessions:
            activity_name = session.get("activity", "unknown")
            technique_name = session.get("technique", "unknown")
            risk_moments = session.get("risk_moments", [])
            high_risk_moments = session.get("high_risk_moments", [])
            moderate_risk_moments = session.get("moderate_risk_moments", [])
            risk_level = session.get("overall_risk_level", "MINIMAL")
            
            patterns["by_activity"][activity_name] += len(risk_moments)
            patterns["by_technique"][technique_name] += len(risk_moments)
            patterns["by_risk_level"][risk_level] += 1  # Count sessions by risk level
            total_risk_moments += len(risk_moments)
            total_high_risk_moments += len(high_risk_moments)
            total_moderate_risk_moments += len(moderate_risk_moments)
            
            # Analyze each risk moment (both moderate and high)
            for moment in risk_moments:
                risk_score = moment.get("risk_score", 0.0)
                patterns["risk_score_distribution"].append(risk_score)
                
                # Extract risk factors
                risk_factors = moment.get("primary_risk_factors", [])
                for factor in risk_factors:
                    patterns["by_risk_factor"][factor] += 1
                
                # Extract metric values
                metrics = session.get("metrics", {})
                if "acl_max_valgus_angle" in metrics:
                    patterns["valgus_angle_distribution"].append(metrics["acl_max_valgus_angle"])
                if "acl_insufficient_flexion_risk" in metrics:
                    patterns["flexion_risk_distribution"].append(metrics["acl_insufficient_flexion_risk"])
                if "acl_high_impact_risk" in metrics:
                    patterns["impact_risk_distribution"].append(metrics["acl_high_impact_risk"])
                
                # Temporal pattern (timestamp)
                timestamp = moment.get("timestamp")
                if timestamp:
                    patterns["temporal_patterns"].append({
                        "timestamp": timestamp,
                        "risk_score": risk_score,
                        "session_id": session.get("session_id"),
                        "activity": activity_name,
                        "technique": technique_name
                    })
        
        # Calculate statistics
        def calc_stats(values):
            if not values:
                return {}
            return {
                "count": len(values),
                "mean": sum(values) / len(values),
                "min": min(values),
                "max": max(values)
            }
        
        analysis = {
            "total_sessions": len(sessions),
            "total_risk_moments": total_risk_moments,
            "total_high_risk_moments": total_high_risk_moments,
            "total_moderate_risk_moments": total_moderate_risk_moments,
            "patterns": {
                "by_activity": dict(patterns["by_activity"]),
                "by_technique": dict(patterns["by_technique"]),
                "by_risk_factor": dict(patterns["by_risk_factor"]),
                "by_risk_level": dict(patterns["by_risk_level"]),
                "risk_score_stats": calc_stats(patterns["risk_score_distribution"]),
                "valgus_angle_stats": calc_stats(patterns["valgus_angle_distribution"]),
                "flexion_risk_stats": calc_stats(patterns["flexion_risk_distribution"]),
                "impact_risk_stats": calc_stats(patterns["impact_risk_distribution"])
            },
            "temporal_patterns": sorted(
                patterns["temporal_patterns"],
                key=lambda x: x.get("timestamp", 0)
            ),
            "summary": self._generate_pattern_summary(patterns, total_risk_moments, total_high_risk_moments, total_moderate_risk_moments)
        }
        
        return analysis
    
    def analyze_high_risk_patterns(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze high-risk ACL patterns across sessions (convenience method).
        
        Args:
            sessions: Optional list of sessions to analyze (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
        
        Returns:
            Dictionary with pattern analysis results
        """
        return self.analyze_risk_patterns(
            sessions=sessions,
            activity=activity,
            technique=technique,
            include_moderate=False,
            include_high=True
        )
    
    def _generate_pattern_summary(
        self,
        patterns: Dict[str, Any],
        total_moments: int,
        total_high: int,
        total_moderate: int
    ) -> str:
        """Generate human-readable pattern summary."""
        if total_moments == 0:
            return "No risk ACL patterns detected."
        
        summary_parts = [
            f"Found {total_moments} risk ACL moments across sessions"
        ]
        
        if total_high > 0 and total_moderate > 0:
            summary_parts.append(f"({total_high} HIGH risk, {total_moderate} MODERATE risk).")
        elif total_high > 0:
            summary_parts.append(f"({total_high} HIGH risk).")
        elif total_moderate > 0:
            summary_parts.append(f"({total_moderate} MODERATE risk).")
        else:
            summary_parts.append(".")
        
        # Most common risk factors
        risk_factors = patterns["by_risk_factor"]
        if risk_factors:
            top_factor = max(risk_factors.items(), key=lambda x: x[1])
            summary_parts.append(f"Most common risk factor: {top_factor[0]} ({top_factor[1]} occurrences)")
        
        # Activity distribution
        activities = patterns["by_activity"]
        if activities:
            top_activity = max(activities.items(), key=lambda x: x[1])
            summary_parts.append(f"Most affected activity: {top_activity[0]} ({top_activity[1]} moments)")
        
        return " ".join(summary_parts)
    
    def get_session_details(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get detailed session information including ACL risk data.
        
        Args:
            session_id: Session identifier or MongoDB _id
        
        Returns:
            Session document with full details
        """
        collection = self.mongodb.get_sessions_collection()
        
        # Try to find by session_id first
        session = collection.find_one({"session_id": session_id})
        
        # If not found, try MongoDB _id
        if not session:
            from bson import ObjectId
            try:
                session = collection.find_one({"_id": ObjectId(session_id)})
            except:
                pass
        
        if session:
            session["_id"] = str(session["_id"])
            return session
        
        return None
    
    def get_risk_metadata(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        include_moderate: bool = True,
        include_high: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get metadata for sessions with moderate or high-risk ACL moments.
        
        Args:
            sessions: Optional list of sessions (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
            include_moderate: Include MODERATE risk sessions
            include_high: Include HIGH risk sessions
        
        Returns:
            List of metadata dictionaries
        """
        if sessions is None:
            sessions = self.find_risk_sessions(
                activity=activity,
                technique=technique,
                include_moderate=include_moderate,
                include_high=include_high
            )
        
        metadata_list = []
        
        for session in sessions:
            metadata = {
                "session_id": session.get("session_id"),
                "activity": session.get("activity"),
                "technique": session.get("technique"),
                "timestamp": session.get("timestamp"),
                "call_id": session.get("call_id"),
                "high_risk_count": session.get("high_risk_count", 0),
                "moderate_risk_count": session.get("moderate_risk_count", 0),
                "total_risk_count": session.get("total_risk_count", 0),
                "acl_risk_score": session.get("overall_risk_score", session.get("acl_risk_score", 0.0)),
                "acl_risk_level": session.get("overall_risk_level", session.get("acl_risk_level", "MINIMAL")),
                "acl_max_valgus_angle": session.get("acl_max_valgus_angle", 0.0),
                "has_transcript": session.get("transcript") is not None,
                "metrics_file": session.get("metrics_file"),
                "transcript_file": session.get("transcript_file"),
                "risk_moments_summary": [
                    {
                        "frame_number": m.get("frame_number"),
                        "timestamp": m.get("timestamp"),
                        "risk_score": m.get("risk_score"),
                        "risk_level": "HIGH" if m.get("risk_score", 0.0) >= 0.7 else "MODERATE",
                        "risk_factors": m.get("primary_risk_factors", [])
                    }
                    for m in session.get("risk_moments", [])
                ]
            }
            metadata_list.append(metadata)
        
        return metadata_list
    
    def get_high_risk_metadata(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get metadata for sessions with high-risk ACL moments (convenience method).
        
        Args:
            sessions: Optional list of sessions (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
        
        Returns:
            List of metadata dictionaries
        """
        return self.get_risk_metadata(
            sessions=sessions,
            activity=activity,
            technique=technique,
            include_moderate=False,
            include_high=True
        )
    
    def reason_about_acl_patterns(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        include_moderate: bool = True,
        include_high: bool = True,
        include_recommendations: bool = True
    ) -> Dict[str, Any]:
        """
        Use LLM reasoning to analyze moderate and high-risk ACL patterns and provide insights.
        
        Args:
            sessions: Optional list of sessions to analyze (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
            include_moderate: Include MODERATE risk (0.4-0.7) in analysis
            include_high: Include HIGH risk (>= 0.7) in analysis
            include_recommendations: Whether to include actionable recommendations
        
        Returns:
            Dictionary with LLM-generated analysis and insights
        """
        if sessions is None:
            sessions = self.find_risk_sessions(
                activity=activity,
                technique=technique,
                include_moderate=include_moderate,
                include_high=include_high
            )
        
        if not sessions:
            return {
                "analysis": "No risk ACL sessions found to analyze.",
                "insights": [],
                "recommendations": []
            }
        
        # Get pattern analysis data
        pattern_analysis = self.analyze_risk_patterns(
            sessions=sessions,
            activity=activity,
            technique=technique,
            include_moderate=include_moderate,
            include_high=include_high
        )
        
        # Build comprehensive prompt for LLM reasoning
        prompt = self._build_acl_analysis_prompt(
            sessions=sessions,
            pattern_analysis=pattern_analysis,
            include_moderate=include_moderate,
            include_high=include_high,
            include_recommendations=include_recommendations
        )
        
        # Call LLM for reasoning
        llm_response = self._call_llm_for_reasoning(prompt)
        
        # Parse and structure response
        reasoning_result = self._parse_llm_reasoning_response(llm_response)
        
        # Apply guardrails validation if available
        if self.guardrails and reasoning_result:
            source_data_for_validation = {
                "pattern_analysis": pattern_analysis,
                "session_count": len(sessions),
                "total_risk_moments": pattern_analysis.get("total_risk_moments", 0)
            }
            
            guardrails_result = self.guardrails.apply_guardrails(
                reasoning_result,
                source_data_for_validation,
                prompt_context=prompt[:500],  # First 500 chars of prompt
                auto_fix=True
            )
            
            # Use improved response if available
            if guardrails_result.get("improved", False):
                reasoning_result = guardrails_result["final_response"]
                logger.info(f"✅ Guardrails improved reasoning response")
            
            # Log validation results
            validation = guardrails_result.get("validation", {})
            if validation.get("confidence_score", 1.0) < 0.6:
                logger.warning(f"⚠️  Low confidence reasoning ({validation.get('confidence_score', 0):.2f})")
            
            if validation.get("hallucination_detected", False):
                logger.warning("⚠️  Potential hallucination detected in reasoning")
        
        # Combine with pattern analysis
        result = {
            "pattern_analysis": pattern_analysis,
            "llm_reasoning": reasoning_result,
            "session_count": len(sessions),
            "total_risk_moments": pattern_analysis.get("total_risk_moments", 0),
            "total_high_risk_moments": pattern_analysis.get("total_high_risk_moments", 0),
            "total_moderate_risk_moments": pattern_analysis.get("total_moderate_risk_moments", 0)
        }
        
        return result
    
    def _build_acl_analysis_prompt(
        self,
        sessions: List[Dict[str, Any]],
        pattern_analysis: Dict[str, Any],
        include_moderate: bool = True,
        include_high: bool = True,
        include_recommendations: bool = True
    ) -> str:
        """
        Build a comprehensive prompt for LLM analysis of ACL risk patterns.
        
        Args:
            sessions: List of high-risk sessions
            pattern_analysis: Pattern analysis results
            include_recommendations: Whether to request recommendations
        
        Returns:
            Formatted prompt string
        """
        # Extract key data for prompt
        total_sessions = len(sessions)
        total_moments = pattern_analysis.get("total_risk_moments", 0)
        total_high = pattern_analysis.get("total_high_risk_moments", 0)
        total_moderate = pattern_analysis.get("total_moderate_risk_moments", 0)
        risk_factors = pattern_analysis.get("patterns", {}).get("by_risk_factor", {})
        activities = pattern_analysis.get("patterns", {}).get("by_activity", {})
        techniques = pattern_analysis.get("patterns", {}).get("by_technique", {})
        risk_levels = pattern_analysis.get("patterns", {}).get("by_risk_level", {})
        risk_stats = pattern_analysis.get("patterns", {}).get("risk_score_stats", {})
        valgus_stats = pattern_analysis.get("patterns", {}).get("valgus_angle_stats", {})
        flexion_stats = pattern_analysis.get("patterns", {}).get("flexion_risk_stats", {})
        
        # Build session summaries
        session_summaries = []
        for i, session in enumerate(sessions[:10], 1):  # Limit to first 10 for prompt size
            high_risk_moments = session.get("high_risk_moments", [])
            session_summaries.append({
                "session_id": session.get("session_id", "unknown"),
                "activity": session.get("activity", "unknown"),
                "technique": session.get("technique", "unknown"),
                "timestamp": session.get("timestamp", ""),
                "high_risk_count": len(high_risk_moments),
                "moderate_risk_count": len(session.get("moderate_risk_moments", [])),
                "total_risk_count": len(session.get("risk_moments", [])),
                "acl_risk_score": session.get("overall_risk_score", session.get("acl_risk_score", 0.0)),
                "acl_risk_level": session.get("overall_risk_level", session.get("acl_risk_level", "MINIMAL")),
                "max_valgus_angle": session.get("acl_max_valgus_angle", 0.0),
                "primary_risk_factors": list(set([
                    factor
                    for moment in session.get("risk_moments", [])
                    for factor in moment.get("primary_risk_factors", [])
                ]))
            })
        
        # Build risk level description
        risk_level_desc = []
        if include_high and total_high > 0:
            risk_level_desc.append(f"{total_high} HIGH-risk (score >= 0.7)")
        if include_moderate and total_moderate > 0:
            risk_level_desc.append(f"{total_moderate} MODERATE-risk (0.4-0.7)")
        risk_level_str = " and ".join(risk_level_desc) if risk_level_desc else "risk"
        
        prompt = f"""You are an expert sports biomechanist and ACL injury prevention specialist analyzing ACL injury risk patterns across multiple training sessions.

## Context
You are analyzing {total_sessions} training sessions with {total_moments} ACL risk moments ({risk_level_str}). These risk indicators require attention and intervention to prevent injury progression.

## Data Summary

### Risk Factor Distribution
{json.dumps(risk_factors, indent=2)}

### Activity Distribution
{json.dumps(activities, indent=2)}

### Technique Distribution
{json.dumps(techniques, indent=2)}

### Risk Level Distribution
{json.dumps(risk_levels, indent=2)}

### Risk Score Statistics
- Mean: {risk_stats.get('mean', 0.0):.2f}
- Min: {risk_stats.get('min', 0.0):.2f}
- Max: {risk_stats.get('max', 0.0):.2f}
- Count: {risk_stats.get('count', 0)}

### Valgus Angle Statistics
- Mean: {valgus_stats.get('mean', 0.0):.1f}°
- Min: {valgus_stats.get('min', 0.0):.1f}°
- Max: {valgus_stats.get('max', 0.0):.1f}°
- Count: {valgus_stats.get('count', 0)}

### Knee Flexion Risk Statistics
- Mean: {flexion_stats.get('mean', 0.0):.2f}
- Min: {flexion_stats.get('min', 0.0):.2f}
- Max: {flexion_stats.get('max', 0.0):.2f}
- Count: {flexion_stats.get('count', 0)}

### Sample Session Details
{json.dumps(session_summaries, indent=2)}

## Your Analysis Task

Provide a comprehensive analysis in JSON format with the following structure:

{{
  "executive_summary": "A 2-3 sentence summary of the most critical findings",
  "key_insights": [
    "Insight 1: Specific pattern or finding with data support",
    "Insight 2: Another key finding",
    ...
  ],
  "root_cause_analysis": {{
    "primary_causes": [
      {{
        "cause": "Specific biomechanical issue",
        "evidence": "Data supporting this cause",
        "severity": "high/medium/low",
        "affected_sessions": "Number or percentage"
      }},
      ...
    ],
    "contributing_factors": [
      "Factor 1 with explanation",
      "Factor 2 with explanation",
      ...
    ]
  }},
  "pattern_identification": {{
    "recurring_patterns": [
      {{
        "pattern": "Description of pattern",
        "frequency": "How often it occurs",
        "implications": "Why this is concerning"
      }},
      ...
    ],
    "technique_specific_risks": {{
      "technique_name": {{
        "risk_level": "high/medium/low",
        "primary_issues": ["Issue 1", "Issue 2"],
        "recommendation_priority": "immediate/high/medium"
      }},
      ...
    }}
  }},
  "biomechanical_analysis": {{
    "valgus_collapse": {{
      "prevalence": "How common",
      "severity": "Average and max angles",
      "biomechanical_implications": "Why this increases ACL risk"
    }},
    "knee_flexion": {{
      "prevalence": "How common",
      "severity": "Average flexion degrees",
      "biomechanical_implications": "Why insufficient flexion is dangerous"
    }},
    "impact_forces": {{
      "prevalence": "How common",
      "severity": "Force levels",
      "biomechanical_implications": "Why high impact increases risk"
    }}
  }},
  "risk_assessment": {{
    "overall_risk_level": "critical/high/medium",
    "immediate_concerns": [
      "Concern 1 with explanation",
      "Concern 2 with explanation",
      ...
    ],
    "long_term_implications": "What could happen if patterns continue"
  }},
  "recommendations": [
    {{
      "priority": "immediate/high/medium",
      "category": "technique_correction/strength_training/landing_mechanics/etc",
      "recommendation": "Specific, actionable recommendation",
      "rationale": "Why this recommendation addresses the root cause",
      "expected_impact": "How this should reduce ACL risk"
    }},
    ...
  ],
  "coaching_priorities": [
    "Priority 1: Most urgent coaching focus",
    "Priority 2: Next most important",
    ...
  ]
}}

## Analysis Guidelines

1. **Focus on risk patterns**: Analyze both MODERATE (0.4-0.7) and HIGH (>= 0.7) risk moments
   - HIGH risk (>= 0.7): Serious injury risk requiring immediate intervention
   - MODERATE risk (0.4-0.7): Concerning patterns that may progress to high risk if not addressed
2. **Root cause analysis**: Don't just describe symptoms - identify WHY these patterns are occurring
3. **Evidence-based**: Support all insights with specific data from the analysis
4. **Actionable**: Recommendations must be specific and implementable
5. **Biomechanical accuracy**: Use correct biomechanical terminology and explain mechanisms
6. **Prioritization**: Rank recommendations by urgency and impact (HIGH risk first, then MODERATE)
7. **Pattern recognition**: Identify recurring patterns across sessions, not just single instances
8. **Risk progression**: Note if MODERATE risk patterns could progress to HIGH risk

## Critical Considerations

- **Valgus collapse >10°**: This is a PRIMARY ACL risk factor - knee collapsing inward
- **Insufficient flexion <20°**: Landing with <20° knee flexion is dangerous - insufficient shock absorption
- **High impact forces >3000N**: Excessive landing forces increase ACL stress
- **Recurring patterns**: Multiple risk moments in same technique indicate systematic issues
- **Technique-specific risks**: Some techniques may have inherent risk factors that need addressing
- **MODERATE risk significance**: MODERATE risk (0.4-0.7) indicates concerning patterns that should be addressed before they progress to HIGH risk

Provide your analysis now in valid JSON format:"""

        return prompt
    
    def _call_llm_for_reasoning(self, prompt: str) -> str:
        """
        Call LLM (Gemini or OpenAI) for reasoning analysis.
        
        Args:
            prompt: Analysis prompt
        
        Returns:
            LLM response text
        """
        # Try Gemini first
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.3,  # Lower temperature for more focused analysis
                            top_p=0.95,
                        )
                    )
                    return response.text
            except Exception as e:
                logger.warning(f"⚠️  Gemini API call failed: {e}, trying OpenAI...")
        
        # Fallback to OpenAI
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    # Try with json_object format first (for supported models)
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",  # Use gpt-4o which supports json_object
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert sports biomechanist and ACL injury prevention specialist. Always respond with valid JSON only, no additional text."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.3
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        # Fallback: try without json_object format
                        logger.debug(f"JSON format not supported, trying without: {e}")
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert sports biomechanist and ACL injury prevention specialist. Always respond with valid JSON only, no additional text or markdown."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.3
                        )
                        content = response.choices[0].message.content
                        # Try to extract JSON if wrapped in markdown
                        if content.strip().startswith("```"):
                            import re
                            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
                            if json_match:
                                return json_match.group(1)
                        return content
            except Exception as e:
                logger.warning(f"⚠️  OpenAI API call failed: {e}")
        
        logger.error("❌ No LLM available for reasoning")
        return "{}"
    
    def _parse_llm_reasoning_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM reasoning response into structured dictionary.
        
        Args:
            response: LLM response text (should be JSON)
        
        Returns:
            Parsed reasoning dictionary
        """
        try:
            # Try to extract JSON from response (in case LLM adds markdown or text)
            response = response.strip()
            
            # Remove markdown code blocks if present
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            response = response.strip()
            
            # Parse JSON
            reasoning = json.loads(response)
            return reasoning
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse LLM response as JSON: {e}")
            logger.debug(f"Response text: {response[:500]}")
            # Return fallback structure
            return {
                "executive_summary": "LLM analysis unavailable - response parsing failed",
                "key_insights": [],
                "root_cause_analysis": {},
                "recommendations": []
            }
        except Exception as e:
            logger.error(f"❌ Error parsing LLM response: {e}", exc_info=True)
            return {
                "executive_summary": "LLM analysis unavailable - error occurred",
                "key_insights": [],
                "root_cause_analysis": {},
                "recommendations": []
            }
    
    def track_trends_from_sessions(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_risk_score: float = 0.4,
        include_moderate: bool = True,
        include_high: bool = True,
        min_sessions: int = 3,
        athlete_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track trends from sessions with insights and store in MongoDB.
        
        Args:
            sessions: Optional list of sessions (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
            min_risk_score: Minimum risk score threshold
            include_moderate: Include MODERATE risk sessions
            include_high: Include HIGH risk sessions
            min_sessions: Minimum sessions required to identify trends
            athlete_name: Optional athlete name filter
            
        Returns:
            Dictionary with identified trends and metadata
        """
        if not self.trend_tracker:
            return {
                "success": False,
                "error": "TrendTracker not available",
                "trends": []
            }
        
        # Get sessions if not provided
        if sessions is None:
            sessions = self.find_risk_sessions(
                activity=activity,
                technique=technique,
                min_risk_score=min_risk_score,
                include_moderate=include_moderate,
                include_high=include_high
            )
        
        if len(sessions) < min_sessions:
            return {
                "success": True,
                "trends": [],
                "message": f"Insufficient sessions ({len(sessions)} < {min_sessions}) to identify trends",
                "session_count": len(sessions)
            }
        
        # Identify trends
        trends = self.trend_tracker.identify_trends_from_sessions(
            sessions=sessions,
            min_sessions=min_sessions,
            athlete_name=athlete_name
        )
        
        # Upsert trends to MongoDB
        trend_ids = []
        if trends:
            trend_ids = self.trend_tracker.upsert_trends(trends)
        
        return {
            "success": True,
            "trends": trends,
            "trend_ids": trend_ids,
            "trend_count": len(trends),
            "session_count": len(sessions)
        }
    
    def get_tracked_trends(
        self,
        athlete_name: Optional[str] = None,
        technique: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get tracked trends from MongoDB.
        
        Args:
            athlete_name: Filter by athlete name
            technique: Filter by technique
            status: Filter by status (improving/unchanged/worsening/insufficient_data)
            limit: Maximum number of results
            
        Returns:
            List of trend documents
        """
        if not self.trend_tracker:
            logger.warning("⚠️  TrendTracker not available")
            return []
        
        return self.trend_tracker.get_trends(
            athlete_name=athlete_name,
            technique=technique,
            status=status,
            limit=limit
        )
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongodb:
            self.mongodb.close()
        if self.trend_tracker:
            self.trend_tracker.close()


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Retrieve and analyze high-risk ACL patterns from MongoDB")
    parser.add_argument(
        "--activity",
        type=str,
        default=None,
        help="Filter by activity (e.g., 'gymnastics')"
    )
    parser.add_argument(
        "--technique",
        type=str,
        default=None,
        help="Filter by technique (e.g., 'back_handspring')"
    )
    parser.add_argument(
        "--min-risk",
        type=float,
        default=0.4,
        help="Minimum risk score threshold (default: 0.4 for MODERATE+ risk, use 0.7 for HIGH only)"
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Perform pattern analysis"
    )
    parser.add_argument(
        "--metadata",
        action="store_true",
        help="Show session metadata"
    )
    parser.add_argument(
        "--reason",
        action="store_true",
        help="Use LLM reasoning for pattern analysis"
    )
    
    args = parser.parse_args()
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Find risk sessions (moderate and/or high)
        include_moderate = args.min_risk < 0.7  # Include moderate if threshold is below 0.7
        include_high = True  # Always include high risk
        
        sessions = agent.find_risk_sessions(
            activity=args.activity,
            technique=args.technique,
            min_risk_score=args.min_risk,
            include_moderate=include_moderate,
            include_high=include_high
        )
        
        risk_type = "moderate and high" if include_moderate else "high"
        print(f"\n📊 Found {len(sessions)} sessions with {risk_type}-risk ACL moments (threshold: >= {args.min_risk})")
        
        if args.analyze:
            print("\n" + "="*60)
            print("🔍 PATTERN ANALYSIS")
            print("="*60)
            analysis = agent.analyze_risk_patterns(
                sessions=sessions,
                activity=args.activity,
                technique=args.technique,
                include_moderate=include_moderate,
                include_high=include_high
            )
            print(f"Total sessions: {analysis['total_sessions']}")
            print(f"Total risk moments: {analysis['total_risk_moments']}")
            print(f"  - HIGH risk: {analysis['total_high_risk_moments']}")
            print(f"  - MODERATE risk: {analysis['total_moderate_risk_moments']}")
            print(f"\nSummary: {analysis['summary']}")
            print(f"\nRisk Factors: {analysis['patterns']['by_risk_factor']}")
            print(f"Risk Levels: {analysis['patterns'].get('by_risk_level', {})}")
            print(f"Activities: {analysis['patterns']['by_activity']}")
            print(f"Techniques: {analysis['patterns']['by_technique']}")
            print("="*60)
        
        if args.metadata:
            print("\n" + "="*60)
            print("📋 SESSION METADATA")
            print("="*60)
            metadata = agent.get_risk_metadata(
                sessions=sessions,
                include_moderate=include_moderate,
                include_high=include_high
            )
            for i, meta in enumerate(metadata[:10], 1):  # Show first 10
                print(f"\n{i}. Session: {meta['session_id']}")
                print(f"   Activity: {meta['activity']}, Technique: {meta['technique']}")
                print(f"   Risk level: {meta['acl_risk_level']}, Score: {meta['acl_risk_score']:.2f}")
                print(f"   HIGH-risk moments: {meta['high_risk_count']}")
                if include_moderate:
                    print(f"   MODERATE-risk moments: {meta['moderate_risk_count']}")
                print(f"   Total risk moments: {meta['total_risk_count']}")
                print(f"   Max valgus: {meta['acl_max_valgus_angle']:.1f}°")
            if len(metadata) > 10:
                print(f"\n... and {len(metadata) - 10} more sessions")
            print("="*60)
        
        if args.reason:
            print("\n" + "="*60)
            print("🧠 LLM REASONING ANALYSIS")
            print("="*60)
            reasoning_result = agent.reason_about_acl_patterns(
                sessions=sessions,
                activity=args.activity,
                technique=args.technique,
                include_moderate=include_moderate,
                include_high=include_high
            )
            
            llm_reasoning = reasoning_result.get("llm_reasoning", {})
            
            print(f"\n📊 Executive Summary:")
            print(f"   {llm_reasoning.get('executive_summary', 'N/A')}")
            
            print(f"\n🔍 Key Insights:")
            for i, insight in enumerate(llm_reasoning.get("key_insights", [])[:5], 1):
                print(f"   {i}. {insight}")
            
            root_causes = llm_reasoning.get("root_cause_analysis", {})
            if root_causes.get("primary_causes"):
                print(f"\n🔬 Root Cause Analysis:")
                for i, cause in enumerate(root_causes["primary_causes"][:3], 1):
                    print(f"   {i}. {cause.get('cause', 'N/A')}")
                    print(f"      Evidence: {cause.get('evidence', 'N/A')}")
                    print(f"      Severity: {cause.get('severity', 'N/A')}")
            
            recommendations = llm_reasoning.get("recommendations", [])
            if recommendations:
                print(f"\n💡 Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. [{rec.get('priority', 'N/A').upper()}] {rec.get('recommendation', 'N/A')}")
                    print(f"      Rationale: {rec.get('rationale', 'N/A')}")
            
            coaching_priorities = llm_reasoning.get("coaching_priorities", [])
            if coaching_priorities:
                print(f"\n🎯 Coaching Priorities:")
                for i, priority in enumerate(coaching_priorities[:5], 1):
                    print(f"   {i}. {priority}")
            
            print("="*60)
        
        agent.close()
        
    except Exception as e:
        logger.error(f"❌ Retrieval failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


