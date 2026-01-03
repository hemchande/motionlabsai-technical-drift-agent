#!/usr/bin/env python3
"""
Form Correction Retrieval Agent

Extracts form issues from session metrics and tracks trends across sessions.
Focuses on actual data extraction, not diagnosis from prompts.

Form Issues Tracked:
- Insufficient height
- Landing knee bend
- Inward collapse of knees (valgus)
- Not enough hip flexion
- Lack of straight knees during flight
- Bad alignment
- And more...
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to import numpy (optional)
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

# Try to import bson ObjectId
try:
    from bson import ObjectId
    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    ObjectId = None

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

# Import trend tracker and guardrails
try:
    from trend_tracker import TrendTracker
    TREND_TRACKER_AVAILABLE = True
except ImportError:
    TREND_TRACKER_AVAILABLE = False
    TrendTracker = None

try:
    from guardrails import Guardrails
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guardrails = None

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class FormCorrectionRetrievalAgent:
    """
    Retrieves and analyzes form correction issues from MongoDB sessions.
    Extracts actual issues from metrics, not from prompts.
    """
    
    # Form issue definitions with metric keys and thresholds
    FORM_ISSUES = {
        "insufficient_height": {
            "metric_keys": [
                "height_off_floor_meters",
                "height_off_floor_pixels",
                "height_from_beam",
                "max_height",
                "leap_height"
            ],
            "threshold": 0.3,  # meters (adjust based on technique)
            "comparison": "below",
            "description": "Insufficient height off floor/beam"
        },
        "landing_knee_bend": {
            "metric_keys": [
                "landing_knee_bend_min",
                "landing_knee_bend_avg",
                "landing_knee_bend_left",
                "landing_knee_bend_right"
            ],
            "threshold": 150.0,  # degrees (180 = straight, <150 = too bent)
            "comparison": "below",
            "description": "Insufficient landing knee extension"
        },
        "knee_valgus_collapse": {
            "metric_keys": [
                "acl_max_valgus_angle",
                "valgus_angle",
                "knee_valgus_angle"
            ],
            "threshold": 10.0,  # degrees (valgus > 10° is concerning)
            "comparison": "above",
            "description": "Inward collapse of knees (valgus)"
        },
        "insufficient_hip_flexion": {
            "metric_keys": [
                "hip_angle",
                "hip_angle_left",
                "hip_angle_right",
                "hip_flexion_angle"
            ],
            "threshold": 120.0,  # degrees (adjust based on technique)
            "comparison": "below",
            "description": "Not enough hip flexion"
        },
        "bent_knees_in_flight": {
            "metric_keys": [
                "flight_knee_bend",
                "bent_knees_in_flight",
                "knee_bend_in_tuck",
                "knee_angle_left",
                "knee_angle_right"
            ],
            "threshold": 170.0,  # degrees (should be >170° for straight)
            "comparison": "below",
            "description": "Lack of straight knees during flight"
        },
        "poor_alignment": {
            "metric_keys": [
                "body_alignment",
                "spinal_alignment",
                "hip_alignment",
                "shoulder_alignment",
                "beam_alignment"
            ],
            "threshold": 5.0,  # degrees deviation
            "comparison": "above",
            "description": "Poor body/alignment"
        },
        "insufficient_split_angle": {
            "metric_keys": [
                "split_angle",
                "leg_split_angle",
                "leg_separation_angle"
            ],
            "threshold": 160.0,  # degrees (ideal is 180°)
            "comparison": "below",
            "description": "Insufficient split angle"
        },
        "poor_landing_quality": {
            "metric_keys": [
                "landing_quality",
                "landing_stability",
                "landing_angle"
            ],
            "threshold": 0.6,  # score (0-1)
            "comparison": "below",
            "description": "Poor landing quality"
        }
    }
    
    def __init__(self):
        """Initialize the form correction retrieval agent."""
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
        
        logger.info("✅ Initialized FormCorrectionRetrievalAgent")
    
    def extract_form_issues_from_session(
        self,
        session: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract actual form issues from session metrics.
        
        Args:
            session: Session document from MongoDB
            
        Returns:
            List of form issues found in the session
        """
        issues = []
        metrics = session.get("metrics", {})
        
        if not metrics:
            return issues
        
        # Check each form issue type
        for issue_type, issue_config in self.FORM_ISSUES.items():
            metric_keys = issue_config["metric_keys"]
            threshold = issue_config["threshold"]
            comparison = issue_config["comparison"]
            
            # Find the metric value in session
            metric_value = None
            metric_key_used = None
            
            for key in metric_keys:
                # Check in metrics dict
                if key in metrics:
                    metric_value = metrics[key]
                    metric_key_used = key
                    break
                
                # Check nested in metrics
                if isinstance(metrics, dict):
                    for nested_key, nested_value in metrics.items():
                        if isinstance(nested_value, dict) and key in nested_value:
                            metric_value = nested_value[key]
                            metric_key_used = f"{nested_key}.{key}"
                            break
                
                # Check at session root level
                if key in session:
                    metric_value = session[key]
                    metric_key_used = key
                    break
            
            if metric_value is None:
                continue
            
            # Check if issue exists based on threshold
            is_issue = False
            if comparison == "below" and metric_value < threshold:
                is_issue = True
            elif comparison == "above" and metric_value > threshold:
                is_issue = True
            
            if is_issue:
                issues.append({
                    "issue_type": issue_type,
                    "description": issue_config["description"],
                    "metric_key": metric_key_used,
                    "metric_value": metric_value,
                    "threshold": threshold,
                    "deviation": abs(metric_value - threshold),
                    "severity": self._calculate_severity(metric_value, threshold, comparison)
                })
        
        return issues
    
    def _calculate_severity(
        self,
        value: float,
        threshold: float,
        comparison: str
    ) -> str:
        """
        Calculate severity of form issue.
        
        Args:
            value: Actual metric value
            threshold: Threshold value
            comparison: "below" or "above"
            
        Returns:
            Severity: "minor", "moderate", or "severe"
        """
        if comparison == "below":
            deviation_percent = ((threshold - value) / threshold) * 100
        else:  # above
            deviation_percent = ((value - threshold) / threshold) * 100
        
        if deviation_percent > 30:
            return "severe"
        elif deviation_percent > 15:
            return "moderate"
        else:
            return "minor"
    
    def find_sessions_with_form_issues(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        issue_types: Optional[List[str]] = None,
        min_severity: str = "minor",
        min_sessions_per_issue: int = 3,
        date_from: Optional[datetime] = None,
        date_to: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions with form issues.
        Only includes issues that appear in 3+ sessions (configurable via min_sessions_per_issue).
        
        Args:
            activity: Filter by activity
            technique: Filter by technique
            issue_types: Filter by specific issue types (None = all)
            min_severity: Minimum severity ("minor", "moderate", "severe")
            min_sessions_per_issue: Minimum number of sessions an issue must appear in (default: 3)
            date_from: Filter sessions from this date
            date_to: Filter sessions to this date
            
        Returns:
            List of sessions with form issues (only issues appearing in 3+ sessions)
        """
        collection = self.mongodb.get_sessions_collection()
        
        # Build query
        query = {}
        
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
        
        # Get all sessions (we'll filter by issues after)
        sessions = list(collection.find(query))
        
        # Step 1: Extract issues from all sessions first
        all_sessions_with_potential_issues = []
        severity_order = {"minor": 1, "moderate": 2, "severe": 3}
        min_severity_level = severity_order.get(min_severity, 1)
        
        for session in sessions:
            issues = self.extract_form_issues_from_session(session)
            
            if not issues:
                continue
            
            # Filter by issue types if specified
            if issue_types:
                issues = [i for i in issues if i["issue_type"] in issue_types]
            
            if not issues:
                continue
            
            # Filter by severity
            filtered_issues = [
                i for i in issues
                if severity_order.get(i["severity"], 0) >= min_severity_level
            ]
            
            if not filtered_issues:
                continue
            
            # Store session with all potential issues
            session["_form_issues_all"] = filtered_issues
            all_sessions_with_potential_issues.append(session)
        
        if not all_sessions_with_potential_issues:
            logger.info("📊 No sessions with form issues found")
            return []
        
        # Step 2: Count how many sessions have each issue type
        issue_type_counts = defaultdict(int)
        for session in all_sessions_with_potential_issues:
            issue_types_in_session = set([i["issue_type"] for i in session["_form_issues_all"]])
            for issue_type in issue_types_in_session:
                issue_type_counts[issue_type] += 1
        
        # Step 3: Filter to only include issues that appear in min_sessions_per_issue+ sessions
        recurring_issue_types = {
            issue_type for issue_type, count in issue_type_counts.items()
            if count >= min_sessions_per_issue
        }
        
        logger.info(f"📊 Issue type counts: {dict(issue_type_counts)}")
        logger.info(f"📊 Recurring issues (appear in {min_sessions_per_issue}+ sessions): {recurring_issue_types}")
        
        # Step 4: Filter sessions to only include recurring issues
        sessions_with_issues = []
        for session in all_sessions_with_potential_issues:
            # Filter issues to only recurring ones
            recurring_issues = [
                i for i in session["_form_issues_all"]
                if i["issue_type"] in recurring_issue_types
            ]
            
            if not recurring_issues:
                continue
            
            # Add filtered issues to session
            session["form_issues"] = recurring_issues
            session["form_issue_count"] = len(recurring_issues)
            session["form_issue_types"] = list(set([i["issue_type"] for i in recurring_issues]))
            session["severest_issue"] = max(
                recurring_issues,
                key=lambda x: severity_order.get(x["severity"], 0)
            )
            
            # Remove temporary field
            del session["_form_issues_all"]
            
            # Convert ObjectId to string
            session["_id"] = str(session["_id"])
            sessions_with_issues.append(session)
        
        logger.info(f"📊 Found {len(sessions_with_issues)} sessions with recurring form issues (appearing in {min_sessions_per_issue}+ sessions)")
        
        # Step 5: Save insights to MongoDB collection
        self._save_insights_to_mongodb(sessions_with_issues)
        
        return sessions_with_issues
    
    def _save_insights_to_mongodb(self, sessions: List[Dict[str, Any]]) -> None:
        """
        Save extracted insights to MongoDB insights collection.
        
        Args:
            sessions: List of sessions with form issues
        """
        try:
            for session in sessions:
                session_id = session.get("session_id") or str(session.get("_id", ""))
                issues = session.get("form_issues", [])
                
                if not issues:
                    continue
                
                # Extract insight descriptions as a list
                insights = [issue["description"] for issue in issues]
                
                # Prepare metadata
                metadata = {
                    "activity": session.get("activity"),
                    "technique": session.get("technique"),
                    "athlete_name": session.get("athlete_name"),
                    "timestamp": session.get("timestamp"),
                    "form_issue_count": len(issues),
                    "form_issue_types": session.get("form_issue_types", [])
                }
                
                # Save to MongoDB
                self.mongodb.upsert_insights(
                    session_id=session_id,
                    insights=insights,
                    metadata=metadata
                )
            
            logger.info(f"✅ Saved insights for {len(sessions)} sessions to MongoDB")
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to save insights to MongoDB: {e}")
    
    def analyze_form_issues_across_sessions(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_sessions_per_issue: int = 3
    ) -> Dict[str, Any]:
        """
        Analyze form issues across multiple sessions to identify trends.
        
        Args:
            sessions: Optional list of sessions (if None, queries all)
            activity: Filter by activity
            technique: Filter by technique
            
        Returns:
            Analysis dictionary with issue patterns and trends
        """
        if sessions is None:
            sessions = self.find_sessions_with_form_issues(
                activity=activity,
                technique=technique,
                min_sessions_per_issue=min_sessions_per_issue
            )
        
        if not sessions:
            return {
                "total_sessions": 0,
                "total_issues": 0,
                "issue_patterns": {},
                "summary": "No sessions with form issues found"
            }
        
        # Aggregate issue patterns
        issue_patterns = defaultdict(lambda: {
            "count": 0,
            "sessions": [],
            "severity_distribution": defaultdict(int),
            "values": []
        })
        
        total_issues = 0
        
        for session in sessions:
            issues = session.get("form_issues", [])
            total_issues += len(issues)
            
            for issue in issues:
                issue_type = issue["issue_type"]
                issue_patterns[issue_type]["count"] += 1
                issue_patterns[issue_type]["sessions"].append(session.get("session_id", "unknown"))
                issue_patterns[issue_type]["severity_distribution"][issue["severity"]] += 1
                issue_patterns[issue_type]["values"].append(issue["metric_value"])
        
        # Calculate statistics for each issue type
        issue_statistics = {}
        for issue_type, pattern in issue_patterns.items():
            values = pattern["values"]
            if values:
                issue_statistics[issue_type] = {
                    "occurrence_count": pattern["count"],
                    "affected_sessions": len(set(pattern["sessions"])),
                    "severity_distribution": dict(pattern["severity_distribution"]),
                    "mean_value": sum(values) / len(values),
                    "min_value": min(values),
                    "max_value": max(values),
                    "description": self.FORM_ISSUES[issue_type]["description"]
                }
        
        # Identify most common issues
        most_common = sorted(
            issue_statistics.items(),
            key=lambda x: x[1]["occurrence_count"],
            reverse=True
        )[:5]
        
        return {
            "total_sessions": len(sessions),
            "total_issues": total_issues,
            "issue_patterns": issue_statistics,
            "most_common_issues": [
                {
                    "issue_type": issue_type,
                    "count": stats["occurrence_count"],
                    "affected_sessions": stats["affected_sessions"],
                    "description": stats["description"]
                }
                for issue_type, stats in most_common
            ],
            "summary": self._generate_analysis_summary(issue_statistics, total_issues)
        }
    
    def _generate_analysis_summary(
        self,
        issue_statistics: Dict[str, Any],
        total_issues: int
    ) -> str:
        """Generate human-readable summary of form issues."""
        if total_issues == 0:
            return "No form issues detected across sessions."
        
        summary_parts = [f"Found {total_issues} form issues across sessions."]
        
        if issue_statistics:
            top_issue = max(
                issue_statistics.items(),
                key=lambda x: x[1]["occurrence_count"]
            )
            summary_parts.append(
                f"Most common issue: {top_issue[1]['description']} "
                f"({top_issue[1]['occurrence_count']} occurrences)"
            )
        
        return " ".join(summary_parts)
    
    def track_form_issue_trends(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_sessions: int = 3,
        athlete_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Track trends for form issues across sessions.
        
        Args:
            sessions: Optional list of sessions
            activity: Filter by activity
            technique: Filter by technique
            min_sessions: Minimum sessions required
            athlete_name: Optional athlete name filter
            
        Returns:
            Dictionary with identified trends
        """
        if not self.trend_tracker:
            return {
                "success": False,
                "error": "TrendTracker not available",
                "trends": []
            }
        
        if sessions is None:
            sessions = self.find_sessions_with_form_issues(
                activity=activity,
                technique=technique,
                min_sessions_per_issue=min_sessions_per_issue
            )
        
        if len(sessions) < min_sessions:
            return {
                "success": True,
                "trends": [],
                "message": f"Insufficient sessions ({len(sessions)} < {min_sessions}) for trend tracking",
                "session_count": len(sessions)
            }
        
        # Prepare sessions for trend tracking by extracting metric data
        # Group by athlete
        athlete_sessions = defaultdict(list)
        for session in sessions:
            athlete = session.get("athlete_name") or session.get("activity") or "unknown"
            athlete_sessions[athlete].append(session)
        
        all_trends = []
        
        for athlete, athlete_sess in athlete_sessions.items():
            if len(athlete_sess) < min_sessions:
                continue
            
            # Sort by timestamp
            athlete_sess.sort(key=lambda s: self._parse_timestamp(s.get("timestamp", "")))
            
            # Extract trends for each form issue type
            for issue_type, issue_config in self.FORM_ISSUES.items():
                metric_keys = issue_config["metric_keys"]
                
                # Collect metric values across sessions
                metric_data_points = []
                
                for session in athlete_sess:
                    timestamp = self._parse_timestamp(session.get("timestamp", ""))
                    metrics = session.get("metrics", {})
                    
                    # Find metric value
                    metric_value = None
                    for key in metric_keys:
                        if key in metrics:
                            metric_value = metrics[key]
                            break
                        elif key in session:
                            metric_value = session[key]
                            break
                    
                    if metric_value is not None:
                        metric_data_points.append({
                            "timestamp": timestamp.isoformat() if isinstance(timestamp, datetime) else str(timestamp),
                            "value": metric_value,
                            "session_id": session.get("session_id") or str(session.get("_id", "")),
                            "issue_type": issue_type
                        })
                
                # If we have enough data points, create trend
                if len(metric_data_points) >= min_sessions:
                    # Use trend tracker to analyze
                    trend = self.trend_tracker._calculate_trend_statistics(
                        metric_data_points,
                        issue_type
                    )
                    
                    if trend:
                        trend["athlete_name"] = athlete
                        trend["technique"] = athlete_sess[0].get("technique", "unknown")
                        trend["metric_signature"] = f"{athlete}_{issue_type}_{trend['technique']}"
                        trend["data_points"] = metric_data_points
                        trend["session_count"] = len(metric_data_points)
                        trend["issue_description"] = issue_config["description"]
                        
                        # Generate three-layer output
                        three_layer = self.trend_tracker._generate_three_layer_output(
                            trend, metric_data_points, athlete_sess
                        )
                        trend.update(three_layer)
                        
                        # Determine status
                        trend["status"] = self.trend_tracker._determine_trend_status(
                            trend, metric_data_points
                        )
                        
                        all_trends.append(trend)
        
        # Upsert trends to MongoDB
        trend_ids = []
        if all_trends and self.trend_tracker:
            trend_ids = self.trend_tracker.upsert_trends(all_trends)
        
        logger.info(f"📊 Identified {len(all_trends)} form issue trends")
        
        return {
            "success": True,
            "trends": all_trends,
            "trend_ids": trend_ids,
            "trend_count": len(all_trends),
            "session_count": len(sessions)
        }
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse timestamp from various formats."""
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                    try:
                        return datetime.strptime(timestamp, fmt)
                    except:
                        continue
        
        return datetime.utcnow()
    
    def get_form_issue_metadata(
        self,
        sessions: Optional[List[Dict[str, Any]]] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_sessions_per_issue: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Get metadata for sessions with form issues.
        
        Args:
            sessions: Optional list of sessions
            activity: Filter by activity
            technique: Filter by technique
            
        Returns:
            List of metadata dictionaries
        """
        if sessions is None:
            sessions = self.find_sessions_with_form_issues(
                activity=activity,
                technique=technique,
                min_sessions_per_issue=min_sessions_per_issue
            )
        
        metadata_list = []
        
        for session in sessions:
            issues = session.get("form_issues", [])
            
            metadata = {
                "session_id": session.get("session_id"),
                "activity": session.get("activity"),
                "technique": session.get("technique"),
                "timestamp": session.get("timestamp"),
                "athlete_name": session.get("athlete_name"),
                "form_issue_count": len(issues),
                "form_issue_types": session.get("form_issue_types", []),
                "severest_issue": session.get("severest_issue", {}),
                "issues": [
                    {
                        "issue_type": issue["issue_type"],
                        "description": issue["description"],
                        "metric_value": issue["metric_value"],
                        "threshold": issue["threshold"],
                        "deviation": issue["deviation"],
                        "severity": issue["severity"]
                    }
                    for issue in issues
                ]
            }
            metadata_list.append(metadata)
        
        return metadata_list
    
    def establish_baseline(
        self,
        athlete_id: str,
        session_ids: Optional[List[str]] = None,
        baseline_type: str = "pre_injury",
        min_sessions: int = 8,
        min_confidence_score: float = 0.7,
        program_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Establish baseline from sessions for an athlete.
        
        Args:
            athlete_id: Athlete identifier
            session_ids: Optional list of specific session IDs to use (if None, uses most recent clean sessions)
            baseline_type: "pre_injury" | "pre_rehab" | "post_rehab"
            min_sessions: Minimum number of sessions required (default: 8)
            min_confidence_score: Minimum capture confidence score (default: 0.7)
            program_id: Optional program identifier
            
        Returns:
            Baseline document with baseline_vector, or None if insufficient sessions
        """
        try:
            collection = self.mongodb.get_sessions_collection()
            
            # If session_ids not provided, find eligible sessions
            if session_ids is None:
                query = {
                    "athlete_id": athlete_id,
                    "capture_confidence_score": {"$gte": min_confidence_score},
                    "baseline_eligible": True
                }
                
                # Get most recent eligible sessions
                sessions = list(collection.find(query).sort("timestamp", -1).limit(min_sessions * 2))
                
                # Filter to last 2 weeks or min_sessions, whichever is more
                if sessions:
                    latest_date = sessions[0].get("timestamp")
                    if isinstance(latest_date, str):
                        latest_date = datetime.fromisoformat(latest_date.replace('Z', '+00:00'))
                    
                    two_weeks_ago = latest_date - timedelta(days=14)
                    
                    eligible_sessions = [
                        s for s in sessions
                        if self._parse_timestamp(s.get("timestamp")) >= two_weeks_ago
                    ][:min_sessions]
                    
                    session_ids = [str(s.get("_id")) for s in eligible_sessions]
                else:
                    session_ids = []
            
            if len(session_ids) < min_sessions:
                logger.warning(f"⚠️  Insufficient sessions for baseline: {len(session_ids)} < {min_sessions}")
                return None
            
            # Get session documents
            sessions = []
            for session_id in session_ids:
                # Try multiple ways to find the session
                session = None
                
                # Try as ObjectId
                if BSON_AVAILABLE and ObjectId:
                    try:
                        session = collection.find_one({"_id": ObjectId(session_id)})
                    except:
                        pass
                
                # Try as session_id field
                if not session:
                    session = collection.find_one({"session_id": session_id})
                
                # Try as _id string
                if not session:
                    try:
                        if BSON_AVAILABLE and ObjectId:
                            session = collection.find_one({"_id": ObjectId(session_id)})
                    except:
                        pass
                
                if session:
                    sessions.append(session)
                else:
                    logger.warning(f"⚠️  Session not found: {session_id}")
            
            if len(sessions) < min_sessions:
                logger.warning(f"⚠️  Could not retrieve enough sessions: {len(sessions)} < {min_sessions}")
                return None
            
            # Extract all metrics from sessions
            all_metrics = defaultdict(list)
            capture_quality_scores = []
            session_timestamps = []
            
            for session in sessions:
                metrics = session.get("metrics", {})
                capture_confidence = session.get("capture_confidence_score", 0.85)
                capture_quality_scores.append(capture_confidence)
                
                timestamp = session.get("timestamp")
                if timestamp:
                    session_timestamps.append(timestamp)
                
                # Extract all numeric metrics
                self._extract_metrics_recursive(metrics, all_metrics)
            
            # Calculate baseline vector (mean, sd, min, max for each metric)
            baseline_vector = {}
            for metric_key, values in all_metrics.items():
                if len(values) > 0:
                    if NUMPY_AVAILABLE:
                        baseline_vector[metric_key] = {
                            "mean": float(np.mean(values)),
                            "sd": float(np.std(values)) if len(values) > 1 else 0.0,
                            "min": float(np.min(values)),
                            "max": float(np.max(values)),
                            "percentile_rank": 50.0  # TODO: Calculate vs sport cohort
                        }
                    else:
                        # Fallback without numpy
                        baseline_vector[metric_key] = {
                            "mean": float(sum(values) / len(values)),
                            "sd": float((sum((x - sum(values) / len(values))**2 for x in values) / len(values))**0.5) if len(values) > 1 else 0.0,
                            "min": float(min(values)),
                            "max": float(max(values)),
                            "percentile_rank": 50.0
                        }
            
            if not baseline_vector:
                logger.warning("⚠️  No metrics found in sessions for baseline")
                return None
            
            # Generate signature ID
            import hashlib
            signature_data = json.dumps(baseline_vector, sort_keys=True) + str(min(session_timestamps))
            signature_id = hashlib.sha256(signature_data.encode()).hexdigest()[:16]
            
            # Create baseline document
            baseline_window = {
                "start_date": min(session_timestamps) if session_timestamps else datetime.utcnow(),
                "end_date": max(session_timestamps) if session_timestamps else datetime.utcnow(),
                "session_count": len(sessions),
                "session_ids": session_ids
            }
            
            baseline_doc = {
                "athlete_id": athlete_id,
                "program_id": program_id,
                "baseline_type": baseline_type,
                "baseline_window": baseline_window,
                "baseline_vector": baseline_vector,
                "signature_id": signature_id,
                "capture_quality_scores": capture_quality_scores,
                "established_at": datetime.utcnow(),
                "established_by": "system",
                "status": "active",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Save to MongoDB
            baseline_id = self._save_baseline(baseline_doc)
            
            if baseline_id:
                logger.info(f"✅ Established baseline for athlete {athlete_id} ({len(baseline_vector)} metrics)")
                
                # Create drift detection flag
                self._create_drift_detection_flag(athlete_id, baseline_id, baseline_window["end_date"])
                
                return {
                    "baseline_id": baseline_id,
                    "athlete_id": athlete_id,
                    "baseline_vector": baseline_vector,
                    "baseline_window": baseline_window,
                    "signature_id": signature_id,
                    "metric_count": len(baseline_vector)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error establishing baseline: {e}", exc_info=True)
            return None
    
    def detect_technical_drift(
        self,
        athlete_id: str,
        session_id: Optional[str] = None,
        new_session_metrics: Optional[Dict[str, float]] = None,
        drift_threshold: float = 2.0,
        analyze_multiple_sessions: bool = True,
        max_sessions: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Detect technical drift from baseline.
        
        Can analyze a single session or multiple sessions after baseline to show
        deviations across sessions for multiple insights.
        
        Args:
            athlete_id: Athlete identifier
            session_id: Optional single session ID to check (if None and analyze_multiple_sessions=True, analyzes all sessions after baseline)
            new_session_metrics: Optional metrics dict for single session (if None, fetches from session)
            drift_threshold: Z-score threshold for drift (default: 2.0 sigma)
            analyze_multiple_sessions: If True, analyzes multiple sessions after baseline (default: True)
            max_sessions: Maximum number of sessions to analyze (default: 10)
            
        Returns:
            Drift detection results with insights and deviations across sessions, or None if no drift or no baseline
        """
        try:
            # Get active baseline
            baseline = self._get_active_baseline(athlete_id)
            if not baseline:
                logger.debug(f"No active baseline found for athlete {athlete_id}")
                return None
            
            # Get drift detection flag
            drift_flag = self._get_drift_detection_flag(athlete_id)
            if not drift_flag or not drift_flag.get("drift_detection_enabled"):
                logger.debug(f"Drift detection not enabled for athlete {athlete_id}")
                return None
            
            # Check if drift detection should be active (past start date)
            start_date = drift_flag.get("drift_detection_start_date")
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                if datetime.utcnow() < start_date:
                    logger.debug(f"Drift detection not yet active (starts {start_date})")
                    return None
            
            # If analyze_multiple_sessions is True and session_id is None, analyze all sessions after baseline
            if analyze_multiple_sessions and session_id is None:
                return self._detect_drift_across_sessions(
                    athlete_id=athlete_id,
                    baseline=baseline,
                    drift_flag=drift_flag,
                    drift_threshold=drift_threshold,
                    max_sessions=max_sessions
                )
            
            # Get new session metrics if not provided
            if new_session_metrics is None:
                collection = self.mongodb.get_sessions_collection()
                # Try multiple ways to find the session
                session = None
                
                # Try as ObjectId
                if BSON_AVAILABLE and ObjectId:
                    try:
                        session = collection.find_one({"_id": ObjectId(session_id)})
                    except:
                        pass
                
                # Try as session_id field
                if not session:
                    session = collection.find_one({"session_id": session_id})
                
                # Try as _id string (if it's already an ObjectId string)
                if not session and BSON_AVAILABLE and ObjectId:
                    try:
                        # Check if it looks like an ObjectId
                        if len(session_id) == 24:
                            session = collection.find_one({"_id": ObjectId(session_id)})
                    except:
                        pass
                
                if not session:
                    logger.warning(f"Session {session_id} not found")
                    return None
                
                metrics = session.get("metrics", {})
                temp_metrics = defaultdict(list)
                self._extract_metrics_recursive(metrics, temp_metrics)
                # Flatten to single values (use mean if multiple)
                new_session_metrics = {}
                for key, values in temp_metrics.items():
                    if values:
                        new_session_metrics[key] = values[0] if len(values) == 1 else sum(values) / len(values)
            
            # Compare metrics to baseline
            baseline_vector = baseline.get("baseline_vector", {})
            drift_results = {}
            drift_metrics = {}
            
            for metric_key, new_value in new_session_metrics.items():
                if metric_key not in baseline_vector:
                    continue
                
                baseline_stats = baseline_vector[metric_key]
                baseline_mean = baseline_stats.get("mean", 0)
                baseline_sd = baseline_stats.get("sd", 0)
                
                if baseline_sd == 0:
                    continue  # Skip if no variance
                
                # Calculate z-score
                z_score = (new_value - baseline_mean) / baseline_sd
                
                # Check if drift exceeds threshold
                if abs(z_score) > drift_threshold:
                    # Determine severity
                    if abs(z_score) > 3.0:
                        severity = "severe"
                    elif abs(z_score) > 2.5:
                        severity = "moderate"
                    else:
                        severity = "minor"
                    
                    # Determine direction
                    if z_score > 0:
                        direction = "worsening" if metric_key in ["height_off_floor_meters", "landing_knee_bend_min"] else "improving"
                    else:
                        direction = "improving" if metric_key in ["height_off_floor_meters", "landing_knee_bend_min"] else "worsening"
                    
                    drift_metrics[metric_key] = {
                        "baseline_value": baseline_mean,
                        "current_value": new_value,
                        "z_score": float(z_score),
                        "drift_magnitude": abs(z_score),
                        "direction": direction,
                        "severity": severity,
                        "coach_follow_up": None,  # Will be set by coach
                        "is_monitored": False,  # True if coach_follow_up is "Monitor"
                        "monitored_at": None  # Timestamp when monitoring started
                    }
            
            if not drift_metrics:
                return None  # No drift detected
            
            # Create drift alert
            alert_id = self._create_drift_alert(athlete_id, session_id, drift_metrics, baseline.get("_id"))
            
            drift_results = {
                "athlete_id": athlete_id,
                "session_id": session_id,
                "baseline_id": str(baseline.get("_id")),
                "drift_metrics": drift_metrics,
                "drift_count": len(drift_metrics),
                "alert_id": alert_id,
                "detected_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Detected drift for athlete {athlete_id} in session {session_id} ({len(drift_metrics)} metrics)")
            
            return drift_results
            
        except Exception as e:
            logger.error(f"❌ Error detecting drift: {e}", exc_info=True)
            return None
    
    def _detect_drift_across_sessions(
        self,
        athlete_id: str,
        baseline: Dict[str, Any],
        drift_flag: Dict[str, Any],
        drift_threshold: float,
        max_sessions: int
    ) -> Optional[Dict[str, Any]]:
        """
        Detect drift across multiple sessions after baseline, showing deviations for each insight.
        
        Args:
            athlete_id: Athlete identifier
            baseline: Baseline document
            drift_flag: Drift detection flag document
            drift_threshold: Z-score threshold
            max_sessions: Maximum sessions to analyze
            
        Returns:
            Dictionary with insights and deviations across sessions
        """
        try:
            collection = self.mongodb.get_sessions_collection()
            baseline_vector = baseline.get("baseline_vector", {})
            baseline_window = baseline.get("baseline_window", {})
            baseline_end = baseline_window.get("end_date")
            
            if isinstance(baseline_end, str):
                baseline_end = datetime.fromisoformat(baseline_end.replace('Z', '+00:00'))
            elif not isinstance(baseline_end, datetime):
                baseline_end = datetime.utcnow() - timedelta(days=1)
            
            # Get start date for drift detection
            start_date = drift_flag.get("drift_detection_start_date")
            if start_date:
                if isinstance(start_date, str):
                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                # Use later of baseline_end or start_date
                if start_date > baseline_end:
                    baseline_end = start_date
            
            # Get all sessions after baseline
            query = {
                "athlete_id": athlete_id,
                "timestamp": {"$gte": baseline_end.isoformat()}
            }
            
            sessions = list(collection.find(query).sort("timestamp", 1).limit(max_sessions))
            
            if not sessions:
                logger.debug(f"No sessions found after baseline for athlete {athlete_id}")
                return None
            
            # Map metric keys to insight descriptions
            metric_to_insight = {}
            for issue_key, issue_data in self.FORM_ISSUES.items():
                metric_keys = issue_data.get("metric_keys", [])
                description = issue_data.get("description", issue_key)
                for metric_key in metric_keys:
                    if metric_key not in metric_to_insight:
                        metric_to_insight[metric_key] = description
            
            # Track insights with deviations across sessions
            insights_data = {}
            
            for session in sessions:
                session_id = str(session.get("_id"))
                session_timestamp = session.get("timestamp")
                metrics = session.get("metrics", {})
                
                # Extract metrics
                temp_metrics = defaultdict(list)
                self._extract_metrics_recursive(metrics, temp_metrics)
                session_metrics = {}
                for key, values in temp_metrics.items():
                    if values:
                        session_metrics[key] = values[0] if len(values) == 1 else sum(values) / len(values)
                
                # Calculate deviations for each metric
                for metric_key, current_value in session_metrics.items():
                    if metric_key not in baseline_vector:
                        continue
                    
                    baseline_stats = baseline_vector[metric_key]
                    baseline_mean = baseline_stats.get("mean", 0)
                    baseline_sd = baseline_stats.get("sd", 0)
                    
                    if baseline_sd == 0:
                        continue
                    
                    # Calculate z-score
                    z_score = (current_value - baseline_mean) / baseline_sd
                    
                    # Only track if exceeds threshold
                    if abs(z_score) > drift_threshold:
                        # Initialize insight if not exists
                        if metric_key not in insights_data:
                            insight_description = metric_to_insight.get(metric_key, metric_key.replace("_", " ").title())
                            insights_data[metric_key] = {
                                "metric_key": metric_key,
                                "insight_description": insight_description,
                                "baseline_value": baseline_mean,
                                "baseline_sd": baseline_sd,
                                "deviations": [],
                                "coach_follow_up": None,
                                "is_monitored": False
                            }
                        
                        # Determine severity
                        if abs(z_score) > 3.0:
                            severity = "severe"
                        elif abs(z_score) > 2.5:
                            severity = "moderate"
                        else:
                            severity = "minor"
                        
                        # Determine direction
                        higher_is_better = metric_key in ["height_off_floor_meters", "landing_knee_bend_min", "hip_angle"]
                        if z_score > 0:
                            direction = "worsening" if higher_is_better else "improving"
                        else:
                            direction = "improving" if higher_is_better else "worsening"
                        
                        # Calculate deviation percentage
                        if baseline_mean != 0:
                            deviation_percent = ((current_value - baseline_mean) / abs(baseline_mean)) * 100
                        else:
                            deviation_percent = 0
                        
                        # Add deviation
                        insights_data[metric_key]["deviations"].append({
                            "session_id": session_id,
                            "session_timestamp": session_timestamp,
                            "current_value": float(current_value),
                            "z_score": float(z_score),
                            "deviation_percent": float(deviation_percent),
                            "direction": direction,
                            "severity": severity
                        })
            
            if not insights_data:
                return None
            
            # Calculate trends for each insight
            insights_list = []
            for metric_key, insight_data in insights_data.items():
                deviations = insight_data["deviations"]
                if len(deviations) < 2:
                    # Single deviation or insufficient data
                    trend = "insufficient_data"
                    trend_strength = 0.0
                    change_in_deviation = 0.0
                else:
                    # Calculate trend
                    first_z = deviations[0]["z_score"]
                    last_z = deviations[-1]["z_score"]
                    
                    if abs(last_z) < abs(first_z):
                        trend = "improving"
                    elif abs(last_z) > abs(first_z):
                        trend = "worsening"
                    else:
                        trend = "unchanged"
                    
                    change_in_deviation = last_z - first_z
                    
                    # Calculate trend strength (rate of change per session)
                    if NUMPY_AVAILABLE and len(deviations) >= 2:
                        z_scores = [d["z_score"] for d in deviations]
                        x = np.arange(len(z_scores))
                        y = np.array(z_scores)
                        slope = np.polyfit(x, y, 1)[0]
                        trend_strength = abs(slope)
                    else:
                        trend_strength = abs(change_in_deviation) / len(deviations) if len(deviations) > 0 else 0.0
                
                # Determine overall severity
                severities = [d["severity"] for d in deviations]
                overall_severity = "severe" if "severe" in severities else ("moderate" if "moderate" in severities else "minor")
                
                insights_list.append({
                    **insight_data,
                    "trend": trend,
                    "trend_strength": float(trend_strength),
                    "overall_severity": overall_severity,
                    "first_deviation": float(deviations[0]["z_score"]) if deviations else 0.0,
                    "latest_deviation": float(deviations[-1]["z_score"]) if deviations else 0.0,
                    "change_in_deviation": float(change_in_deviation),
                    "session_count": len(deviations)
                })
            
            # Sort by overall severity and trend
            severity_order = {"severe": 3, "moderate": 2, "minor": 1}
            insights_list.sort(key=lambda x: (severity_order.get(x["overall_severity"], 0), x["change_in_deviation"]), reverse=True)
            
            # Calculate summary
            worsening_count = sum(1 for i in insights_list if i["trend"] == "worsening")
            improving_count = sum(1 for i in insights_list if i["trend"] == "improving")
            unchanged_count = sum(1 for i in insights_list if i["trend"] == "unchanged")
            
            session_ids = [str(s.get("_id")) for s in sessions]
            
            result = {
                "athlete_id": athlete_id,
                "baseline_id": str(baseline.get("_id")),
                "baseline_end_date": baseline_end.isoformat(),
                "sessions_analyzed": session_ids,
                "session_count": len(sessions),
                "insights": insights_list,
                "summary": {
                    "total_insights": len(insights_list),
                    "worsening_insights": worsening_count,
                    "improving_insights": improving_count,
                    "unchanged_insights": unchanged_count,
                    "sessions_with_drift": len(sessions),
                    "most_severe_insight": insights_list[0]["metric_key"] if insights_list else None
                },
                "detected_at": datetime.utcnow().isoformat()
            }
            
            logger.info(f"✅ Analyzed drift across {len(sessions)} sessions for athlete {athlete_id}: {len(insights_list)} insights")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error detecting drift across sessions: {e}", exc_info=True)
            return None
    
    def analyze_treatment_effectiveness(
        self,
        treatment_action_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze treatment action effectiveness by comparing pre/post drift patterns.
        
        Args:
            treatment_action_id: Treatment action ID to analyze
            
        Returns:
            Outcome metrics with effectiveness analysis
        """
        try:
            # Get treatment action
            treatment_action = self._get_treatment_action(treatment_action_id)
            if not treatment_action:
                logger.warning(f"Treatment action {treatment_action_id} not found")
                return None
            
            athlete_id = treatment_action.get("athlete_id")
            action_timestamp = treatment_action.get("action_timestamp")
            baseline_id = treatment_action.get("baseline_id")
            validation_n_sessions = treatment_action.get("validation_after_n_sessions", 3)
            
            if isinstance(action_timestamp, str):
                action_timestamp = datetime.fromisoformat(action_timestamp.replace('Z', '+00:00'))
            
            # Get sessions before treatment
            pre_sessions = self._get_sessions_before(athlete_id, action_timestamp)
            
            # Get sessions after treatment
            post_sessions = self._get_sessions_after(athlete_id, action_timestamp, n_sessions=validation_n_sessions)
            
            if len(pre_sessions) == 0 or len(post_sessions) == 0:
                logger.warning(f"Insufficient sessions for analysis: {len(pre_sessions)} pre, {len(post_sessions)} post")
                return None
            
            # Get baseline for comparison
            baseline = self._get_baseline_by_id(baseline_id) if baseline_id else None
            
            # Calculate drift patterns
            pre_drift = self._calculate_drift_pattern(pre_sessions, baseline)
            post_drift = self._calculate_drift_pattern(post_sessions, baseline)
            
            # Determine outcome
            outcome = {
                "drift_resolved": post_drift.get("magnitude", 0) < pre_drift.get("magnitude", 0),
                "improvement_percentage": self._calculate_improvement_percentage(pre_drift, post_drift),
                "sessions_to_resolution": self._find_resolution_session(post_sessions, baseline),
                "correlation_strength": 0.85  # TODO: Calculate actual correlation
            }
            
            # Update treatment action
            self._update_treatment_action_outcome(treatment_action_id, outcome)
            
            logger.info(f"✅ Analyzed treatment effectiveness for action {treatment_action_id}")
            
            return outcome
            
        except Exception as e:
            logger.error(f"❌ Error analyzing treatment effectiveness: {e}", exc_info=True)
            return None
    
    # Helper methods for baseline and drift
    
    def _extract_metrics_recursive(self, metrics: Any, all_metrics: Dict[str, List[float]]) -> Dict[str, List[float]]:
        """Recursively extract all numeric metrics from nested structure."""
        if isinstance(metrics, dict):
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    all_metrics[key].append(float(value))
                elif isinstance(value, dict):
                    self._extract_metrics_recursive(value, all_metrics)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, (int, float)):
                            all_metrics[key].append(float(item))
                        elif isinstance(item, dict):
                            self._extract_metrics_recursive(item, all_metrics)
        elif isinstance(metrics, list):
            for item in metrics:
                self._extract_metrics_recursive(item, all_metrics)
        
        return all_metrics
    
    def _parse_timestamp(self, timestamp_str: Any) -> datetime:
        """Parse timestamp string to datetime."""
        if isinstance(timestamp_str, datetime):
            return timestamp_str
        if isinstance(timestamp_str, str):
            try:
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            except:
                return datetime.utcnow()
        return datetime.utcnow()
    
    def _save_baseline(self, baseline_doc: Dict[str, Any]) -> Optional[str]:
        """Save baseline to MongoDB."""
        try:
            collection = self.mongodb.database.get_collection("baselines")
            
            # Check for existing active baseline
            existing = collection.find_one({
                "athlete_id": baseline_doc["athlete_id"],
                "status": "active"
            })
            
            if existing:
                # Supersede existing baseline
                collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": {"status": "superseded", "updated_at": datetime.utcnow()}}
                )
            
            # Insert new baseline
            result = collection.insert_one(baseline_doc)
            return str(result.inserted_id)
            
        except Exception as e:
            logger.error(f"❌ Error saving baseline: {e}", exc_info=True)
            return None
    
    def _get_active_baseline(self, athlete_id: str) -> Optional[Dict[str, Any]]:
        """Get active baseline for athlete."""
        try:
            collection = self.mongodb.database.get_collection("baselines")
            baseline = collection.find_one({
                "athlete_id": athlete_id,
                "status": "active"
            })
            return baseline
            
        except Exception as e:
            logger.error(f"❌ Error getting baseline: {e}", exc_info=True)
            return None
    
    def _create_drift_detection_flag(
        self,
        athlete_id: str,
        baseline_id: str,
        start_date: Optional[datetime] = None
    ) -> Optional[str]:
        """Create drift detection flag."""
        try:
            collection = self.mongodb.database.get_collection("drift_detection_flags")
            
            # Use baseline end date + 1 day as start date if not provided
            if start_date is None:
                start_date = datetime.utcnow() + timedelta(days=1)
            
            flag_doc = {
                "athlete_id": athlete_id,
                "baseline_id": baseline_id,
                "drift_detection_enabled": True,
                "drift_detection_start_date": start_date,
                "drift_threshold": 2.0,
                "alert_on_drift": True,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Upsert
            existing = collection.find_one({"athlete_id": athlete_id})
            if existing:
                collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": flag_doc}
                )
                return str(existing["_id"])
            else:
                result = collection.insert_one(flag_doc)
                return str(result.inserted_id)
                
        except Exception as e:
            logger.error(f"❌ Error creating drift flag: {e}", exc_info=True)
            return None
    
    def _get_drift_detection_flag(self, athlete_id: str) -> Optional[Dict[str, Any]]:
        """Get drift detection flag for athlete."""
        try:
            collection = self.mongodb.database.get_collection("drift_detection_flags")
            flag = collection.find_one({"athlete_id": athlete_id})
            return flag
            
        except Exception as e:
            logger.error(f"❌ Error getting drift flag: {e}", exc_info=True)
            return None
    
    def _create_drift_alert(
        self,
        athlete_id: str,
        session_id: str,
        drift_metrics: Dict[str, Any],
        baseline_id: Any
    ) -> Optional[str]:
        """Create drift alert in MongoDB."""
        try:
            collection = self.mongodb.database.get_collection("alerts")
            
            # Determine overall severity
            severities = [m.get("severity") for m in drift_metrics.values()]
            overall_severity = "severe" if "severe" in severities else ("moderate" if "moderate" in severities else "minor")
            
            alert_doc = {
                "alert_id": f"drift_{athlete_id}_{session_id}_{int(datetime.utcnow().timestamp())}",
                "alert_type": "technical_drift",
                "alert_created_at": datetime.utcnow(),
                "alert_confidence": 0.92,
                "athlete_id": athlete_id,
                "session_id": session_id,
                "sessions_affected": [session_id],
                "reps_affected": 0,  # TODO: Calculate from session
                "drift_metrics": drift_metrics,
                "top_clip_ids": [],  # TODO: Get clip IDs
                "alert_payload_summary": f"Technical drift detected: {len(drift_metrics)} metrics deviating from baseline",
                "status": "new",
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            result = collection.insert_one(alert_doc)
            alert_id_str = str(result.inserted_id)
            
            # Send alert to Redis queue for delivery
            self._send_alert_to_queue(alert_doc, alert_id_str)
            
            return alert_id_str
            
        except Exception as e:
            logger.error(f"❌ Error creating drift alert: {e}", exc_info=True)
            return None
    
    def _send_alert_to_queue(self, alert_doc: Dict[str, Any], alert_id: str) -> bool:
        """
        Send drift alert to Redis queue for delivery.
        
        Args:
            alert_doc: Alert document
            alert_id: Alert ID
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Try to import Redis
            try:
                import redis
                REDIS_AVAILABLE = True
            except ImportError:
                REDIS_AVAILABLE = False
                logger.debug("Redis not available for alert queue")
                return False
            
            if not REDIS_AVAILABLE:
                return False
            
            # Connect to Redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            
            try:
                redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                redis_client.ping()
            except Exception as e:
                logger.debug(f"Redis not available: {e}")
                return False
            
            # Prepare queue message
            queue_message = {
                "alert_type": "technical_drift",
                "alert_id": alert_doc.get("alert_id", alert_id),
                "athlete_id": alert_doc.get("athlete_id"),
                "session_id": alert_doc.get("session_id"),
                "severity": "severe" if "severe" in [m.get("severity") for m in alert_doc.get("drift_metrics", {}).values()] else "moderate",
                "drift_metrics": alert_doc.get("drift_metrics", {}),
                "drift_count": len(alert_doc.get("drift_metrics", {})),
                "alert_confidence": alert_doc.get("alert_confidence", 0.92),
                "created_at": alert_doc.get("alert_created_at", datetime.utcnow()).isoformat() if isinstance(alert_doc.get("alert_created_at"), datetime) else alert_doc.get("alert_created_at"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to drift_alerts_queue
            queue_name = "drift_alerts_queue"
            message_json = json.dumps(queue_message)
            redis_client.lpush(queue_name, message_json)
            
            logger.info(f"✅ Sent drift alert to queue: {queue_name} (alert_id: {alert_doc.get('alert_id', alert_id)})")
            redis_client.close()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to send alert to queue: {e}")
            return False
    
    def update_drift_alert_coach_follow_up(
        self,
        alert_id: str,
        metric_key: str,
        coach_follow_up: str
    ) -> bool:
        """
        Update coach follow-up for a specific drift metric in an alert.
        
        Args:
            alert_id: Alert ID
            metric_key: The metric key (e.g., "height_off_floor_meters")
            coach_follow_up: One of "Monitor", "Escalate to AT/PT"
            
        Returns:
            True if successful, False otherwise
        """
        try:
            collection = self.mongodb.database.get_collection("alerts")
            
            # Validate coach_follow_up
            valid_follow_ups = ["Monitor", "Escalate to AT/PT"]
            if coach_follow_up not in valid_follow_ups:
                logger.error(f"❌ Invalid coach_follow_up for drift: {coach_follow_up}. Must be one of {valid_follow_ups}")
                return False
            
            # Find the alert
            alert = collection.find_one({"alert_id": alert_id})
            if not alert:
                # Try finding by _id
                if BSON_AVAILABLE and ObjectId:
                    try:
                        alert = collection.find_one({"_id": ObjectId(alert_id)})
                    except:
                        pass
                
                if not alert:
                    logger.warning(f"⚠️  Alert not found: {alert_id}")
                    return False
            
            # Update the specific drift metric
            drift_metrics = alert.get("drift_metrics", {})
            if metric_key not in drift_metrics:
                logger.warning(f"⚠️  Metric '{metric_key}' not found in alert {alert_id}")
                return False
            
            # Update coach follow-up
            drift_metrics[metric_key]["coach_follow_up"] = coach_follow_up
            drift_metrics[metric_key]["is_monitored"] = (coach_follow_up == "Monitor")
            
            if coach_follow_up == "Monitor":
                drift_metrics[metric_key]["monitored_at"] = datetime.utcnow()
            else:
                drift_metrics[metric_key]["monitored_at"] = None
            
            # Update alert
            result = collection.update_one(
                {"_id": alert["_id"]},
                {
                    "$set": {
                        "drift_metrics": drift_metrics,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            if result.modified_count > 0:
                logger.info(f"✅ Updated coach follow-up for drift metric '{metric_key}' in alert {alert_id}: {coach_follow_up}")
                
                # Send follow-up action to queue
                self._send_followup_to_queue(alert_id, metric_key, coach_follow_up, athlete_id)
                
                return True
            else:
                logger.warning(f"⚠️  No changes made to alert {alert_id}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error updating drift alert coach follow-up: {e}", exc_info=True)
            return False
    
    def _send_followup_to_queue(self, alert_id: str, metric_key: str, coach_follow_up: str, athlete_id: str) -> bool:
        """
        Send coach follow-up action to Redis queue.
        
        Args:
            alert_id: Alert ID
            metric_key: Metric key
            coach_follow_up: Follow-up action
            athlete_id: Athlete ID
            
        Returns:
            True if sent successfully, False otherwise
        """
        try:
            # Try to import Redis
            try:
                import redis
                REDIS_AVAILABLE = True
            except ImportError:
                REDIS_AVAILABLE = False
                logger.debug("Redis not available for follow-up queue")
                return False
            
            if not REDIS_AVAILABLE:
                return False
            
            # Connect to Redis
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            
            try:
                redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    decode_responses=True
                )
                redis_client.ping()
            except Exception as e:
                logger.debug(f"Redis not available: {e}")
                return False
            
            # Prepare queue message
            queue_message = {
                "event_type": "coach_followup",
                "alert_id": alert_id,
                "athlete_id": athlete_id,
                "metric_key": metric_key,
                "coach_follow_up": coach_follow_up,
                "is_monitored": (coach_follow_up == "Monitor"),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Send to coach_followup_queue
            queue_name = "coach_followup_queue"
            message_json = json.dumps(queue_message)
            redis_client.lpush(queue_name, message_json)
            
            logger.info(f"✅ Sent coach follow-up to queue: {queue_name} (alert_id: {alert_id}, action: {coach_follow_up})")
            redis_client.close()
            return True
            
        except Exception as e:
            logger.warning(f"⚠️  Failed to send follow-up to queue: {e}")
            return False
    
    def track_monitored_drift_insights(
        self,
        athlete_id: str,
        metric_key: str,
        monitored_since: Optional[datetime] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Track how a monitored drift insight is worsening or improving over time.
        
        Args:
            athlete_id: Athlete identifier
            metric_key: The metric key to track (e.g., "height_off_floor_meters")
            monitored_since: Timestamp when monitoring started (if None, finds from alerts)
            
        Returns:
            Dictionary with trend analysis (worsening/improving/unchanged) or None
        """
        try:
            alerts_collection = self.mongodb.database.get_collection("alerts")
            sessions_collection = self.mongodb.get_sessions_collection()
            
            # Find all alerts with this metric being monitored
            query = {
                "athlete_id": athlete_id,
                "alert_type": "technical_drift",
                f"drift_metrics.{metric_key}.is_monitored": True
            }
            
            alerts = list(alerts_collection.find(query).sort("alert_created_at", 1))
            
            if not alerts:
                logger.debug(f"No monitored drift alerts found for metric {metric_key}")
                return None
            
            # Get monitored_since from first alert if not provided
            if monitored_since is None:
                first_alert = alerts[0]
                monitored_at = first_alert.get("drift_metrics", {}).get(metric_key, {}).get("monitored_at")
                if monitored_at:
                    if isinstance(monitored_at, str):
                        monitored_since = datetime.fromisoformat(monitored_at.replace('Z', '+00:00'))
                    else:
                        monitored_since = monitored_at
                else:
                    monitored_since = first_alert.get("alert_created_at", datetime.utcnow())
                    if isinstance(monitored_since, str):
                        monitored_since = datetime.fromisoformat(monitored_since.replace('Z', '+00:00'))
            
            # Get baseline for comparison
            baseline = self._get_active_baseline(athlete_id)
            if not baseline:
                logger.warning(f"No baseline found for athlete {athlete_id}")
                return None
            
            baseline_vector = baseline.get("baseline_vector", {})
            baseline_stats = baseline_vector.get(metric_key)
            if not baseline_stats:
                logger.warning(f"Metric {metric_key} not found in baseline")
                return None
            
            baseline_mean = baseline_stats.get("mean", 0)
            baseline_sd = baseline_stats.get("sd", 0.001)
            
            # Get sessions after monitoring started
            sessions = list(sessions_collection.find({
                "athlete_id": athlete_id,
                "timestamp": {"$gte": monitored_since.isoformat()}
            }).sort("timestamp", 1))
            
            if len(sessions) < 2:
                logger.debug(f"Insufficient sessions for trend analysis: {len(sessions)} < 2")
                return {
                    "metric_key": metric_key,
                    "athlete_id": athlete_id,
                    "trend": "insufficient_data",
                    "sessions_analyzed": len(sessions),
                    "monitored_since": monitored_since.isoformat(),
                    "message": "Need at least 2 sessions to determine trend"
                }
            
            # Extract metric values from sessions
            metric_values = []
            session_timestamps = []
            
            for session in sessions:
                metrics = session.get("metrics", {})
                temp_metrics = defaultdict(list)
                self._extract_metrics_recursive(metrics, temp_metrics)
                
                if metric_key in temp_metrics:
                    values = temp_metrics[metric_key]
                    if values:
                        # Use mean if multiple values
                        metric_value = sum(values) / len(values) if len(values) > 1 else values[0]
                        metric_values.append(metric_value)
                        session_timestamps.append(session.get("timestamp"))
            
            if len(metric_values) < 2:
                return {
                    "metric_key": metric_key,
                    "athlete_id": athlete_id,
                    "trend": "insufficient_data",
                    "sessions_analyzed": len(sessions),
                    "monitored_since": monitored_since.isoformat(),
                    "message": "Metric not found in enough sessions"
                }
            
            # Calculate trend
            # Compare first and last values, and overall pattern
            first_value = metric_values[0]
            last_value = metric_values[-1]
            
            # Calculate z-scores
            first_z = (first_value - baseline_mean) / baseline_sd if baseline_sd > 0 else 0
            last_z = (last_value - baseline_mean) / baseline_sd if baseline_sd > 0 else 0
            
            # Determine if improving or worsening
            # For metrics where higher is better (e.g., height_off_floor_meters):
            # - Improving: z-score increasing (getting closer to/better than baseline)
            # - Worsening: z-score decreasing (getting further from baseline)
            # For metrics where lower is better (e.g., valgus_angle):
            # - Improving: z-score decreasing (getting closer to/better than baseline)
            # - Worsening: z-score increasing (getting further from baseline)
            
            # Determine metric direction (higher is better or lower is better)
            higher_is_better = metric_key in ["height_off_floor_meters", "landing_knee_bend_min", "hip_angle"]
            
            if higher_is_better:
                # Higher is better
                if abs(last_z) < abs(first_z):
                    trend = "improving"  # Getting closer to baseline
                elif abs(last_z) > abs(first_z):
                    trend = "worsening"  # Getting further from baseline
                else:
                    trend = "unchanged"
            else:
                # Lower is better (e.g., valgus_angle)
                if abs(last_z) < abs(first_z):
                    trend = "improving"  # Getting closer to baseline
                elif abs(last_z) > abs(first_z):
                    trend = "worsening"  # Getting further from baseline
                else:
                    trend = "unchanged"
            
            # Calculate change percentage
            if first_value != 0:
                change_percent = ((last_value - first_value) / abs(first_value)) * 100
            else:
                change_percent = 0
            
            # Calculate linear regression slope for trend strength
            if NUMPY_AVAILABLE and len(metric_values) >= 2:
                x = np.arange(len(metric_values))
                y = np.array(metric_values)
                slope = np.polyfit(x, y, 1)[0]
                trend_strength = abs(slope) / baseline_sd if baseline_sd > 0 else 0
            else:
                slope = (last_value - first_value) / len(metric_values) if len(metric_values) > 1 else 0
                trend_strength = abs(slope) / baseline_sd if baseline_sd > 0 else 0
            
            # Save monitoring trend
            trend_doc = {
                "metric_key": metric_key,
                "athlete_id": athlete_id,
                "trend": trend,
                "trend_strength": float(trend_strength),
                "change_percent": float(change_percent),
                "first_value": float(first_value),
                "last_value": float(last_value),
                "first_z_score": float(first_z),
                "last_z_score": float(last_z),
                "baseline_mean": float(baseline_mean),
                "baseline_sd": float(baseline_sd),
                "sessions_analyzed": len(sessions),
                "metric_values": [float(v) for v in metric_values],
                "session_timestamps": session_timestamps,
                "monitored_since": monitored_since.isoformat(),
                "analyzed_at": datetime.utcnow().isoformat(),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Save to monitoring_trends collection
            monitoring_collection = self.mongodb.database.get_collection("monitoring_trends")
            monitoring_collection.insert_one(trend_doc)
            
            logger.info(f"✅ Tracked drift trend for {metric_key}: {trend} ({change_percent:.1f}% change)")
            
            return {
                "metric_key": metric_key,
                "athlete_id": athlete_id,
                "trend": trend,
                "trend_strength": float(trend_strength),
                "change_percent": float(change_percent),
                "first_value": float(first_value),
                "last_value": float(last_value),
                "first_z_score": float(first_z),
                "last_z_score": float(last_z),
                "sessions_analyzed": len(sessions),
                "monitored_since": monitored_since.isoformat(),
                "message": f"Trend: {trend} ({change_percent:.1f}% change over {len(sessions)} sessions)"
            }
            
        except Exception as e:
            logger.error(f"❌ Error tracking monitored drift insights: {e}", exc_info=True)
            return None
    
    def _get_treatment_action(self, action_id: str) -> Optional[Dict[str, Any]]:
        """Get treatment action by ID."""
        try:
            collection = self.mongodb.database.get_collection("treatment_actions")
            action = collection.find_one({"action_id": action_id})
            return action
            
        except Exception as e:
            logger.error(f"❌ Error getting treatment action: {e}", exc_info=True)
            return None
    
    def _get_sessions_before(self, athlete_id: str, timestamp: datetime, limit: int = 10) -> List[Dict[str, Any]]:
        """Get sessions before timestamp."""
        try:
            collection = self.mongodb.get_sessions_collection()
            sessions = list(collection.find({
                "athlete_id": athlete_id,
                "timestamp": {"$lt": timestamp.isoformat()}
            }).sort("timestamp", -1).limit(limit))
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error getting sessions before: {e}", exc_info=True)
            return []
    
    def _get_sessions_after(self, athlete_id: str, timestamp: datetime, n_sessions: int = 3) -> List[Dict[str, Any]]:
        """Get sessions after timestamp."""
        try:
            collection = self.mongodb.get_sessions_collection()
            sessions = list(collection.find({
                "athlete_id": athlete_id,
                "timestamp": {"$gt": timestamp.isoformat()}
            }).sort("timestamp", 1).limit(n_sessions))
            return sessions
            
        except Exception as e:
            logger.error(f"❌ Error getting sessions after: {e}", exc_info=True)
            return []
    
    def _calculate_drift_pattern(self, sessions: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate drift pattern from sessions."""
        if not sessions or not baseline:
            return {"magnitude": 0, "metrics": {}}
        
        baseline_vector = baseline.get("baseline_vector", {})
        all_drifts = []
        
        for session in sessions:
            metrics = session.get("metrics", {})
            temp_metrics = defaultdict(list)
            self._extract_metrics_recursive(metrics, temp_metrics)
            
            session_metrics = {}
            for key, values in temp_metrics.items():
                if values and key in baseline_vector:
                    session_metrics[key] = values[0] if len(values) == 1 else sum(values) / len(values)
            
            for metric_key, value in session_metrics.items():
                baseline_stats = baseline_vector[metric_key]
                baseline_mean = baseline_stats.get("mean", 0)
                baseline_sd = baseline_stats.get("sd", 0)
                if baseline_sd > 0:
                    z_score = abs((value - baseline_mean) / baseline_sd)
                    all_drifts.append(z_score)
        
        return {
            "magnitude": sum(all_drifts) / len(all_drifts) if all_drifts else 0,
            "metrics": {}
        }
    
    def _calculate_improvement_percentage(self, pre_drift: Dict[str, Any], post_drift: Dict[str, Any]) -> float:
        """Calculate improvement percentage."""
        pre_mag = pre_drift.get("magnitude", 0)
        post_mag = post_drift.get("magnitude", 0)
        
        if pre_mag == 0:
            return 0.0
        
        improvement = ((pre_mag - post_mag) / pre_mag) * 100
        return max(0.0, improvement)
    
    def _find_resolution_session(self, post_sessions: List[Dict[str, Any]], baseline: Optional[Dict[str, Any]]) -> Optional[int]:
        """Find session number where drift was resolved."""
        if not baseline:
            return None
        
        for i, session in enumerate(post_sessions, 1):
            drift = self._calculate_drift_pattern([session], baseline)
            if drift.get("magnitude", 0) < 1.0:  # Resolved if drift < 1 sigma
                return i
        
        return None
    
    def _update_treatment_action_outcome(self, action_id: str, outcome: Dict[str, Any]) -> bool:
        """Update treatment action with outcome metrics."""
        try:
            collection = self.mongodb.database.get_collection("treatment_actions")
            result = collection.update_one(
                {"action_id": action_id},
                {
                    "$set": {
                        "outcome_metrics": outcome,
                        "validation_status": "validated",
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            return result.modified_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error updating treatment action: {e}", exc_info=True)
            return False
    
    def _get_baseline_by_id(self, baseline_id: Any) -> Optional[Dict[str, Any]]:
        """Get baseline by ID."""
        try:
            collection = self.mongodb.database.get_collection("baselines")
            if isinstance(baseline_id, str):
                from bson import ObjectId
                baseline_id = ObjectId(baseline_id)
            baseline = collection.find_one({"_id": baseline_id})
            return baseline
            
        except Exception as e:
            logger.error(f"❌ Error getting baseline by ID: {e}", exc_info=True)
            return None
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongodb:
            self.mongodb.close()
        if self.trend_tracker:
            self.trend_tracker.close()

