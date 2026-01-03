#!/usr/bin/env python3
"""
Retrieve Flagged Insights from MongoDB

Retrieves insights that have been flagged with coach follow-up actions,
particularly those being monitored or requiring attention.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FlaggedInsightsRetriever:
    """
    Retrieves flagged insights from MongoDB insights collection.
    """
    
    def __init__(self):
        """Initialize flagged insights retriever."""
        self.mongodb = MongoDBService()
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        logger.info("✅ Connected to MongoDB")
    
    def get_flagged_insights(
        self,
        coach_follow_up: Optional[str] = None,
        is_monitored: Optional[bool] = None,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Retrieve flagged insights from MongoDB.
        
        Args:
            coach_follow_up: Filter by specific follow-up action (Monitor, Adjust Training, etc.)
            is_monitored: Filter by monitoring status (True for monitored, False for not monitored)
            activity: Filter by activity
            technique: Filter by technique
            limit: Maximum number of documents to return
            
        Returns:
            List of insight documents with flagged insights
        """
        try:
            collection = self.mongodb.get_insights_collection()
            
            # Build query to find documents with flagged insights
            query = {}
            
            if activity:
                query["activity"] = activity
            if technique:
                query["technique"] = technique
            
            # Get all matching documents
            documents = list(collection.find(query).limit(limit))
            
            flagged_results = []
            
            for doc in documents:
                session_id = doc.get("session_id")
                insights = doc.get("insights", [])
                
                # Filter insights based on criteria
                flagged_insights = []
                
                for insight_item in insights:
                    # Handle both old (string) and new (object) formats
                    if isinstance(insight_item, dict):
                        insight_text = insight_item.get("insight", "")
                        insight_coach_follow_up = insight_item.get("coach_follow_up")
                        insight_is_monitored = insight_item.get("is_monitored", False)
                        
                        # Check if this insight matches our filters
                        matches = True
                        
                        if coach_follow_up is not None:
                            if insight_coach_follow_up != coach_follow_up:
                                matches = False
                        
                        if is_monitored is not None:
                            if insight_is_monitored != is_monitored:
                                matches = False
                        
                        if matches and insight_coach_follow_up:
                            # This insight is flagged
                            flagged_insights.append({
                                "insight": insight_text,
                                "coach_follow_up": insight_coach_follow_up,
                                "is_monitored": insight_is_monitored,
                                "monitored_at": insight_item.get("monitored_at"),
                                "session_id": session_id,
                                "activity": doc.get("activity"),
                                "technique": doc.get("technique"),
                                "athlete_name": doc.get("athlete_name"),
                                "timestamp": doc.get("timestamp"),
                                "updated_at": doc.get("updated_at")
                            })
                
                # If this document has flagged insights, add it to results
                if flagged_insights:
                    flagged_results.append({
                        "session_id": session_id,
                        "activity": doc.get("activity"),
                        "technique": doc.get("technique"),
                        "athlete_name": doc.get("athlete_name"),
                        "flagged_insights": flagged_insights,
                        "total_flagged": len(flagged_insights),
                        "total_insights": len(insights),
                        "updated_at": doc.get("updated_at")
                    })
            
            logger.info(f"✅ Found {len(flagged_results)} sessions with flagged insights")
            return flagged_results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving flagged insights: {e}", exc_info=True)
            return []
    
    def get_monitored_insights(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get insights that are currently being monitored.
        
        Args:
            activity: Filter by activity
            technique: Filter by technique
            
        Returns:
            List of monitored insights
        """
        return self.get_flagged_insights(
            coach_follow_up="Monitor",
            is_monitored=True,
            activity=activity,
            technique=technique
        )
    
    def get_all_flagged_insights(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all insights with any coach follow-up action.
        
        Args:
            activity: Filter by activity
            technique: Filter by technique
            
        Returns:
            List of all flagged insights
        """
        return self.get_flagged_insights(
            coach_follow_up=None,  # Any follow-up
            is_monitored=None,  # Any monitoring status
            activity=activity,
            technique=technique
        )
    
    def get_insights_by_follow_up(
        self,
        coach_follow_up: str,
        activity: Optional[str] = None,
        technique: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get insights by specific coach follow-up action.
        
        Args:
            coach_follow_up: One of "Monitor", "Adjust Training", "Escalate to AT/PT", "Dismiss"
            activity: Filter by activity
            technique: Filter by technique
            
        Returns:
            List of insights with the specified follow-up action
        """
        return self.get_flagged_insights(
            coach_follow_up=coach_follow_up,
            activity=activity,
            technique=technique
        )
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongodb:
            self.mongodb.close()
            logger.info("✅ MongoDB connection closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retrieve flagged insights from MongoDB"
    )
    parser.add_argument(
        "--monitored",
        action="store_true",
        help="Get only monitored insights"
    )
    parser.add_argument(
        "--follow-up",
        type=str,
        choices=["Monitor", "Adjust Training", "Escalate to AT/PT", "Dismiss"],
        help="Filter by specific coach follow-up action"
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
        "--all",
        action="store_true",
        help="Get all flagged insights (any follow-up action)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    
    args = parser.parse_args()
    
    try:
        retriever = FlaggedInsightsRetriever()
        
        # Determine which method to call
        if args.monitored:
            results = retriever.get_monitored_insights(
                activity=args.activity,
                technique=args.technique
            )
            print(f"\n📊 Monitored Insights ({len(results)} sessions)\n")
        elif args.follow_up:
            results = retriever.get_insights_by_follow_up(
                coach_follow_up=args.follow_up,
                activity=args.activity,
                technique=args.technique
            )
            print(f"\n📊 Insights with follow-up '{args.follow_up}' ({len(results)} sessions)\n")
        elif args.all:
            results = retriever.get_all_flagged_insights(
                activity=args.activity,
                technique=args.technique
            )
            print(f"\n📊 All Flagged Insights ({len(results)} sessions)\n")
        else:
            # Default: get all flagged insights
            results = retriever.get_all_flagged_insights(
                activity=args.activity,
                technique=args.technique
            )
            print(f"\n📊 All Flagged Insights ({len(results)} sessions)\n")
        
        if args.json:
            # Output as JSON
            print(json.dumps(results, indent=2, default=str))
        else:
            # Output as formatted text
            if not results:
                print("ℹ️  No flagged insights found")
            else:
                for i, result in enumerate(results, 1):
                    print(f"{'='*60}")
                    print(f"Session {i}: {result['session_id']}")
                    print(f"  Activity: {result.get('activity', 'N/A')}")
                    print(f"  Technique: {result.get('technique', 'N/A')}")
                    print(f"  Athlete: {result.get('athlete_name', 'N/A')}")
                    print(f"  Flagged Insights: {result['total_flagged']}/{result['total_insights']}")
                    print()
                    
                    for insight in result['flagged_insights']:
                        print(f"  ✅ {insight['insight']}")
                        print(f"     Follow-up: {insight['coach_follow_up']}")
                        print(f"     Monitored: {insight['is_monitored']}")
                        if insight.get('monitored_at'):
                            print(f"     Monitored At: {insight['monitored_at']}")
                        print()
        
        retriever.close()
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

