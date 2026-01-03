#!/usr/bin/env python3
"""
Monitor Flagged Insights - Track trends for monitored insights over time

Monitors flagged insights (especially those with "Monitor" follow-up) and
identifies trends from the timestamp when they were flagged forward.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService
from retrieve_flagged_insights import FlaggedInsightsRetriever
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlaggedInsightsMonitor:
    """
    Monitors flagged insights and tracks trends from the flag timestamp forward.
    """
    
    def __init__(self):
        """Initialize flagged insights monitor."""
        self.mongodb = MongoDBService()
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        
        self.retriever = FlaggedInsightsRetriever()
        self.retrieval_agent = FormCorrectionRetrievalAgent()
        
        logger.info("✅ Initialized FlaggedInsightsMonitor")
    
    def monitor_flagged_insights(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        min_new_sessions: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Monitor flagged insights and identify trends from flag timestamp forward.
        
        Args:
            activity: Filter by activity
            technique: Filter by technique
            min_new_sessions: Minimum new sessions required to identify trends
            
        Returns:
            List of monitoring results with trends
        """
        try:
            logger.info("🔍 Starting flagged insights monitoring...")
            
            # Step 1: Get all monitored insights
            monitored_insights = self.retriever.get_monitored_insights(
                activity=activity,
                technique=technique
            )
            
            logger.info(f"📊 Found {len(monitored_insights)} sessions with monitored insights")
            
            if not monitored_insights:
                logger.info("ℹ️  No monitored insights found")
                return []
            
            monitoring_results = []
            
            # Step 2: For each monitored insight, find new sessions and track trends
            for session_data in monitored_insights:
                session_id = session_data["session_id"]
                flagged_insights = session_data["flagged_insights"]
                
                for insight in flagged_insights:
                    if not insight.get("is_monitored"):
                        continue
                    
                    monitored_at_str = insight.get("monitored_at")
                    if not monitored_at_str:
                        logger.warning(f"⚠️  Insight '{insight['insight']}' has no monitored_at timestamp")
                        continue
                    
                    # Parse timestamp
                    try:
                        if isinstance(monitored_at_str, str):
                            monitored_at = datetime.fromisoformat(monitored_at_str.replace('Z', '+00:00'))
                        else:
                            monitored_at = monitored_at_str
                    except Exception as e:
                        logger.warning(f"⚠️  Failed to parse monitored_at: {e}")
                        continue
                    
                    insight_text = insight["insight"]
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📈 Monitoring insight: {insight_text}")
                    logger.info(f"   Session: {session_id}")
                    logger.info(f"   Flagged at: {monitored_at}")
                    
                    # Step 3: Find new sessions after the flag timestamp
                    new_sessions = self._find_new_sessions_after_timestamp(
                        monitored_at=monitored_at,
                        insight_text=insight_text,
                        activity=activity or session_data.get("activity"),
                        technique=technique or session_data.get("technique"),
                        exclude_session_id=session_id
                    )
                    
                    logger.info(f"   📊 Found {len(new_sessions)} new sessions after flag timestamp")
                    
                    if len(new_sessions) < min_new_sessions:
                        logger.info(f"   ℹ️  Insufficient new sessions ({len(new_sessions)} < {min_new_sessions}) for trend analysis")
                        monitoring_results.append({
                            "insight": insight_text,
                            "session_id": session_id,
                            "monitored_at": monitored_at_str,
                            "new_sessions_count": len(new_sessions),
                            "trends": [],
                            "status": "insufficient_data"
                        })
                        continue
                    
                    # Step 4: Analyze trends from new sessions
                    trends = self._analyze_trends_from_new_sessions(
                        insight_text=insight_text,
                        new_sessions=new_sessions,
                        monitored_at=monitored_at,
                        activity=activity or session_data.get("activity"),
                        technique=technique or session_data.get("technique")
                    )
                    
                    # Step 5: Store trends in MongoDB
                    trend_id = self._store_monitoring_trends(
                        insight_text=insight_text,
                        session_id=session_id,
                        monitored_at=monitored_at,
                        trends=trends,
                        new_sessions_count=len(new_sessions)
                    )
                    
                    monitoring_results.append({
                        "insight": insight_text,
                        "session_id": session_id,
                        "monitored_at": monitored_at_str,
                        "new_sessions_count": len(new_sessions),
                        "trends": trends,
                        "trend_id": trend_id,
                        "status": "trends_identified" if trends else "no_trends"
                    })
                    
                    logger.info(f"   ✅ Identified {len(trends)} trends")
            
            logger.info(f"\n✅ Monitoring complete. Processed {len(monitoring_results)} insights")
            return monitoring_results
            
        except Exception as e:
            logger.error(f"❌ Error monitoring flagged insights: {e}", exc_info=True)
            return []
    
    def _find_new_sessions_after_timestamp(
        self,
        monitored_at: datetime,
        insight_text: str,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        exclude_session_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find sessions that occurred after the monitored_at timestamp and contain the same insight.
        
        Args:
            monitored_at: Timestamp when insight was flagged
            insight_text: The insight text to match
            activity: Filter by activity
            technique: Filter by technique
            exclude_session_id: Session ID to exclude (the original flagged session)
            
        Returns:
            List of sessions after the timestamp
        """
        try:
            # Get all sessions with form issues
            all_sessions = self.retrieval_agent.find_sessions_with_form_issues(
                activity=activity,
                technique=technique,
                min_sessions_per_issue=1  # Get all sessions, not just recurring
            )
            
            new_sessions = []
            
            for session in all_sessions:
                session_id = session.get("session_id") or str(session.get("_id", ""))
                
                # Exclude the original session
                if exclude_session_id and session_id == exclude_session_id:
                    continue
                
                # Get session timestamp
                session_timestamp_str = session.get("timestamp")
                if not session_timestamp_str:
                    continue
                
                try:
                    if isinstance(session_timestamp_str, str):
                        session_timestamp = datetime.fromisoformat(session_timestamp_str.replace('Z', '+00:00'))
                    else:
                        session_timestamp = session_timestamp_str
                except Exception:
                    continue
                
                # Check if session is after monitored_at
                if session_timestamp <= monitored_at:
                    continue
                
                # Check if this session has the same insight
                form_issues = session.get("form_issues", [])
                has_insight = False
                
                for issue in form_issues:
                    if issue.get("description") == insight_text:
                        has_insight = True
                        break
                
                if has_insight:
                    new_sessions.append(session)
            
            # Sort by timestamp
            new_sessions.sort(key=lambda s: self._parse_timestamp(s.get("timestamp", "")))
            
            return new_sessions
            
        except Exception as e:
            logger.error(f"❌ Error finding new sessions: {e}", exc_info=True)
            return []
    
    def _analyze_trends_from_new_sessions(
        self,
        insight_text: str,
        new_sessions: List[Dict[str, Any]],
        monitored_at: datetime,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze trends from new sessions after the flag timestamp.
        
        Args:
            insight_text: The insight being monitored
            new_sessions: List of new sessions after timestamp
            monitored_at: Timestamp when insight was flagged
            activity: Activity filter
            technique: Technique filter
            
        Returns:
            List of identified trends
        """
        try:
            if len(new_sessions) < 3:
                return []
            
            # Extract metric data for the specific insight
            metric_data = []
            
            for session in new_sessions:
                form_issues = session.get("form_issues", [])
                
                for issue in form_issues:
                    if issue.get("description") == insight_text:
                        # Get metric value for this issue
                        metric_key = issue.get("metric_key")
                        metric_value = issue.get("metric_value")
                        
                        if metric_value is not None:
                            metric_data.append({
                                "session_id": session.get("session_id"),
                                "timestamp": session.get("timestamp"),
                                "metric_key": metric_key,
                                "metric_value": metric_value,
                                "severity": issue.get("severity")
                            })
                        break
            
            if len(metric_data) < 3:
                return []
            
            # Group by metric key
            metric_groups = {}
            for data in metric_data:
                metric_key = data["metric_key"]
                if metric_key not in metric_groups:
                    metric_groups[metric_key] = []
                metric_groups[metric_key].append(data)
            
            trends = []
            
            for metric_key, data_points in metric_groups.items():
                if len(data_points) < 3:
                    continue
                
                # Calculate trend statistics
                values = [d["metric_value"] for d in data_points]
                mean_value = sum(values) / len(values)
                
                # Determine trend direction
                first_half = values[:len(values)//2]
                second_half = values[len(values)//2:]
                first_mean = sum(first_half) / len(first_half)
                second_mean = sum(second_half) / len(second_half)
                
                if second_mean > first_mean * 1.1:
                    direction = "worsening"
                elif second_mean < first_mean * 0.9:
                    direction = "improving"
                else:
                    direction = "unchanged"
                
                trend = {
                    "insight": insight_text,
                    "metric_key": metric_key,
                    "metric_type": metric_key.replace("_min", "").replace("_max", "").replace("_score", ""),
                    "data_points": len(data_points),
                    "mean_value": mean_value,
                    "first_half_mean": first_mean,
                    "second_half_mean": second_mean,
                    "direction": direction,
                    "monitored_since": monitored_at.isoformat(),
                    "sessions_analyzed": [d["session_id"] for d in data_points]
                }
                
                trends.append(trend)
            
            return trends
            
        except Exception as e:
            logger.error(f"❌ Error analyzing trends: {e}", exc_info=True)
            return []
    
    def _store_monitoring_trends(
        self,
        insight_text: str,
        session_id: str,
        monitored_at: datetime,
        trends: List[Dict[str, Any]],
        new_sessions_count: int
    ) -> Optional[str]:
        """
        Store monitoring trends in MongoDB.
        
        Args:
            insight_text: The insight being monitored
            session_id: Original session ID
            monitored_at: When monitoring started
            trends: List of identified trends
            new_sessions_count: Number of new sessions analyzed
            
        Returns:
            Trend document ID if successful
        """
        try:
            collection = self.mongodb.database.get_collection("monitoring_trends")
            
            trend_doc = {
                "insight": insight_text,
                "original_session_id": session_id,
                "monitored_at": monitored_at,
                "new_sessions_count": new_sessions_count,
                "trends": trends,
                "trend_count": len(trends),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Upsert based on insight and original session
            existing = collection.find_one({
                "insight": insight_text,
                "original_session_id": session_id
            })
            
            if existing:
                trend_doc["created_at"] = existing.get("created_at", datetime.utcnow())
                result = collection.update_one(
                    {"_id": existing["_id"]},
                    {"$set": trend_doc}
                )
                trend_id = str(existing["_id"])
            else:
                result = collection.insert_one(trend_doc)
                trend_id = str(result.inserted_id)
            
            logger.info(f"   💾 Stored monitoring trends (ID: {trend_id})")
            return trend_id
            
        except Exception as e:
            logger.error(f"❌ Error storing monitoring trends: {e}", exc_info=True)
            return None
    
    def _parse_timestamp(self, timestamp_str: str) -> datetime:
        """Parse timestamp string to datetime."""
        try:
            if isinstance(timestamp_str, str):
                return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            return timestamp_str
        except Exception:
            return datetime.min
    
    def close(self):
        """Close connections."""
        if self.retriever:
            self.retriever.close()
        if self.retrieval_agent:
            self.retrieval_agent.close()
        if self.mongodb:
            self.mongodb.close()
        logger.info("✅ Connections closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor flagged insights and track trends from flag timestamp forward"
    )
    parser.add_argument(
        "--activity",
        type=str,
        help="Filter by activity"
    )
    parser.add_argument(
        "--technique",
        type=str,
        help="Filter by technique"
    )
    parser.add_argument(
        "--min-sessions",
        type=int,
        default=3,
        help="Minimum new sessions required for trend analysis (default: 3)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        monitor = FlaggedInsightsMonitor()
        
        results = monitor.monitor_flagged_insights(
            activity=args.activity,
            technique=args.technique,
            min_new_sessions=args.min_sessions
        )
        
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"\n📊 Monitoring Results ({len(results)} insights)\n")
            for i, result in enumerate(results, 1):
                print(f"{'='*60}")
                print(f"Insight {i}: {result['insight']}")
                print(f"  Session: {result['session_id']}")
                print(f"  Monitored Since: {result['monitored_at']}")
                print(f"  New Sessions: {result['new_sessions_count']}")
                print(f"  Trends Identified: {len(result['trends'])}")
                print(f"  Status: {result['status']}")
                
                if result['trends']:
                    print(f"\n  Trends:")
                    for trend in result['trends']:
                        print(f"    - {trend['metric_type']}: {trend['direction']}")
                        print(f"      Mean: {trend['mean_value']:.2f}")
                        print(f"      Sessions: {trend['data_points']}")
                print()
        
        monitor.close()
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

