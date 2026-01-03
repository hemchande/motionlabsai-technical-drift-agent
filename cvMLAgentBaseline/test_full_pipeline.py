#!/usr/bin/env python3
"""
Comprehensive Full Pipeline Test
Runs mock video agent → retrieval agent → logs all outputs (insights, trends, baseline, drift)
"""

import json
import logging
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoAgent.mongodb_service import MongoDBService
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'pipeline_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ComprehensivePipelineTest:
    """Comprehensive test of the full pipeline."""
    
    def __init__(self):
        """Initialize test."""
        self.mongodb = MongoDBService()
        self.mongodb.connect()
        self.retrieval_agent = FormCorrectionRetrievalAgent()
        
    def print_section_header(self, title: str, char: str = "="):
        """Print a formatted section header."""
        width = 80
        logger.info("")
        logger.info(char * width)
        logger.info(f"  {title}")
        logger.info(char * width)
        logger.info("")
    
    def print_insights_summary(self, athlete_id: Optional[str] = None, activity: Optional[str] = None, technique: Optional[str] = None):
        """Print comprehensive insights summary."""
        self.print_section_header("📊 INSIGHTS SUMMARY", "=")
        
        try:
            insights_collection = self.mongodb.get_insights_collection()
            
            # Build query
            query = {}
            if athlete_id:
                query["athlete_id"] = athlete_id
            if activity:
                query["activity"] = activity
            if technique:
                query["technique"] = technique
            
            # Get recent insights
            recent_insights = list(insights_collection.find(query).sort("updated_at", -1).limit(20))
            
            if not recent_insights:
                logger.info("   ℹ️  No insights found")
                return
            
            logger.info(f"   Found {len(recent_insights)} recent insight documents")
            logger.info("")
            
            # Group by session
            for idx, insight_doc in enumerate(recent_insights[:10], 1):
                session_id = insight_doc.get("session_id", "N/A")
                insights_list = insight_doc.get("insights", [])
                athlete_id_doc = insight_doc.get("athlete_id", "N/A")
                activity_doc = insight_doc.get("activity", "N/A")
                technique_doc = insight_doc.get("technique", "N/A")
                updated_at = insight_doc.get("updated_at", "N/A")
                
                logger.info(f"   📋 Insight Document #{idx}")
                logger.info(f"      Session ID: {str(session_id)[:30]}...")
                logger.info(f"      Athlete ID: {athlete_id_doc}")
                logger.info(f"      Activity: {activity_doc} | Technique: {technique_doc}")
                logger.info(f"      Updated: {updated_at}")
                logger.info(f"      Total Insights: {len(insights_list)}")
                
                if insights_list:
                    logger.info("      Insights:")
                    for i, insight in enumerate(insights_list[:5], 1):
                        desc = insight.get("description", "N/A")
                        monitored = insight.get("is_monitored", False)
                        follow_up = insight.get("coach_follow_up", "None")
                        logger.info(f"         {i}. {desc}")
                        if monitored or follow_up:
                            logger.info(f"            [Monitored: {monitored}, Follow-up: {follow_up}]")
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Error retrieving insights: {e}", exc_info=True)
    
    def print_trends_summary(self, athlete_id: Optional[str] = None):
        """Print comprehensive trends summary."""
        self.print_section_header("📈 TRENDS SUMMARY", "=")
        
        try:
            trends_collection = self.mongodb.get_trends_collection()
            
            # Build query
            query = {}
            if athlete_id:
                query["athlete_id"] = athlete_id
            
            # Get recent trends
            recent_trends = list(trends_collection.find(query).sort("updated_at", -1).limit(20))
            
            if not recent_trends:
                logger.info("   ℹ️  No trends found")
                return
            
            logger.info(f"   Found {len(recent_trends)} recent trends")
            logger.info("")
            
            for idx, trend in enumerate(recent_trends[:10], 1):
                trend_id = trend.get("trend_id", "N/A")
                athlete_name = trend.get("athlete_name", "N/A")
                issue_type = trend.get("issue_type", "N/A")
                status = trend.get("status", "N/A")
                observation = trend.get("observation", "N/A")
                evidence = trend.get("evidence", "N/A")
                coaching_options = trend.get("coaching_options", [])
                updated_at = trend.get("updated_at", "N/A")
                
                logger.info(f"   📊 Trend #{idx}: {issue_type}")
                logger.info(f"      Trend ID: {trend_id}")
                logger.info(f"      Athlete: {athlete_name}")
                logger.info(f"      Status: {status}")
                logger.info(f"      Updated: {updated_at}")
                logger.info(f"      Observation: {observation[:100]}..." if len(str(observation)) > 100 else f"      Observation: {observation}")
                logger.info(f"      Evidence: {evidence[:100]}..." if len(str(evidence)) > 100 else f"      Evidence: {evidence}")
                
                if coaching_options:
                    logger.info(f"      Coaching Options ({len(coaching_options)}):")
                    for i, option in enumerate(coaching_options[:3], 1):
                        logger.info(f"         {i}. {option[:80]}..." if len(option) > 80 else f"         {i}. {option}")
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Error retrieving trends: {e}", exc_info=True)
    
    def print_baseline_summary(self, athlete_id: Optional[str] = None):
        """Print comprehensive baseline summary."""
        self.print_section_header("📊 BASELINE SUMMARY", "=")
        
        try:
            baselines_collection = self.mongodb.database.get_collection("baselines")
            
            # Build query
            query = {"status": "active"}
            if athlete_id:
                query["athlete_id"] = athlete_id
            
            baselines = list(baselines_collection.find(query).sort("established_at", -1).limit(10))
            
            if not baselines:
                logger.info("   ℹ️  No active baselines found")
                return
            
            logger.info(f"   Found {len(baselines)} active baseline(s)")
            logger.info("")
            
            for idx, baseline in enumerate(baselines, 1):
                baseline_id = str(baseline.get("_id", "N/A"))[:20]
                athlete_id_doc = baseline.get("athlete_id", "N/A")
                baseline_type = baseline.get("baseline_type", "N/A")
                baseline_vector = baseline.get("baseline_vector", {})
                baseline_window = baseline.get("baseline_window", {})
                session_count = baseline_window.get("session_count", 0)
                established_at = baseline.get("established_at", "N/A")
                
                logger.info(f"   📊 Baseline #{idx}")
                logger.info(f"      Baseline ID: {baseline_id}...")
                logger.info(f"      Athlete ID: {athlete_id_doc}")
                logger.info(f"      Type: {baseline_type}")
                logger.info(f"      Sessions Used: {session_count}")
                logger.info(f"      Established: {established_at}")
                logger.info(f"      Metrics in Baseline: {len(baseline_vector)}")
                
                # Show sample metrics
                if baseline_vector:
                    logger.info("      Sample Metrics:")
                    for i, (metric_key, metric_data) in enumerate(list(baseline_vector.items())[:5], 1):
                        mean = metric_data.get("mean", "N/A")
                        sd = metric_data.get("sd", "N/A")
                        logger.info(f"         {i}. {metric_key}: μ={mean:.3f}, σ={sd:.3f}")
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Error retrieving baselines: {e}", exc_info=True)
    
    def print_drift_summary(self, athlete_id: Optional[str] = None):
        """Print comprehensive drift detection summary."""
        self.print_section_header("🔍 DRIFT DETECTION SUMMARY", "=")
        
        try:
            alerts_collection = self.mongodb.database.get_collection("alerts")
            
            # Build query
            query = {"alert_type": "technical_drift"}
            if athlete_id:
                query["athlete_id"] = athlete_id
            
            alerts = list(alerts_collection.find(query).sort("created_at", -1).limit(10))
            
            if not alerts:
                logger.info("   ℹ️  No drift alerts found")
                return
            
            logger.info(f"   Found {len(alerts)} drift alert(s)")
            logger.info("")
            
            for idx, alert in enumerate(alerts, 1):
                alert_id = alert.get("alert_id", "N/A")
                athlete_id_doc = alert.get("athlete_id", "N/A")
                session_id = alert.get("session_id", "N/A")
                drift_metrics = alert.get("drift_metrics", {})
                status = alert.get("status", "N/A")
                created_at = alert.get("created_at", "N/A")
                
                logger.info(f"   🔍 Drift Alert #{idx}")
                logger.info(f"      Alert ID: {alert_id}")
                logger.info(f"      Athlete ID: {athlete_id_doc}")
                logger.info(f"      Session ID: {str(session_id)[:30]}...")
                logger.info(f"      Status: {status}")
                logger.info(f"      Created: {created_at}")
                logger.info(f"      Metrics with Drift: {len(drift_metrics)}")
                logger.info("")
                
                # Show drift metrics
                if drift_metrics:
                    logger.info("      Drift Metrics:")
                    for i, (metric_key, metric_data) in enumerate(list(drift_metrics.items())[:5], 1):
                        baseline_value = metric_data.get("baseline_value", "N/A")
                        current_value = metric_data.get("current_value", "N/A")
                        z_score = metric_data.get("z_score", "N/A")
                        severity = metric_data.get("severity", "N/A")
                        direction = metric_data.get("direction", "N/A")
                        coach_follow_up = metric_data.get("coach_follow_up", "None")
                        
                        logger.info(f"         {i}. {metric_key}")
                        logger.info(f"            Baseline: {baseline_value:.3f} | Current: {current_value:.3f}")
                        logger.info(f"            Z-score: {z_score:.2f}σ | Severity: {severity.upper()} | Direction: {direction.upper()}")
                        if coach_follow_up:
                            logger.info(f"            Coach Follow-up: {coach_follow_up}")
                
                logger.info("")
                
        except Exception as e:
            logger.error(f"   ❌ Error retrieving drift alerts: {e}", exc_info=True)
    
    def run_full_pipeline_test(
        self,
        athlete_id: str,
        activity: str = "gymnastics",
        technique: str = "back_handspring",
        wait_time: int = 90
    ):
        """Run the full pipeline test."""
        self.print_section_header("🚀 FULL PIPELINE TEST", "=")
        
        session_id = f"test_pipeline_{int(time.time())}"
        
        logger.info(f"   Test Configuration:")
        logger.info(f"      Session ID: {session_id}")
        logger.info(f"      Athlete ID: {athlete_id}")
        logger.info(f"      Activity: {activity}")
        logger.info(f"      Technique: {technique}")
        logger.info("")
        
        # Step 1: Send message via mock video agent
        self.print_section_header("STEP 1: Sending Message via Mock Video Agent", "-")
        
        try:
            from mock_video_agent import MockVideoAgent
            
            video_agent = MockVideoAgent()
            if not video_agent.redis_client:
                logger.error("   ❌ Redis not available. Cannot run test.")
                return False
            
            success = video_agent.simulate_video_call_ended(
                session_id=session_id,
                athlete_id=athlete_id,
                activity=activity,
                technique=technique,
                athlete_name="Test Athlete"
            )
            
            if success:
                logger.info(f"   ✅ Message sent successfully to queue")
            else:
                logger.error(f"   ❌ Failed to send message")
                return False
            
            video_agent.close()
            
        except Exception as e:
            logger.error(f"   ❌ Error sending message: {e}", exc_info=True)
            return False
        
        # Step 2: Wait for processing
        self.print_section_header("STEP 2: Waiting for Retrieval Agent Processing", "-")
        logger.info(f"   Waiting {wait_time} seconds for processing...")
        logger.info("")
        
        time.sleep(wait_time)
        
        # Step 3: Retrieve and print all outputs
        self.print_section_header("STEP 3: Retrieving Pipeline Outputs", "-")
        
        # Print insights
        self.print_insights_summary(athlete_id=athlete_id, activity=activity, technique=technique)
        
        # Print trends
        self.print_trends_summary(athlete_id=athlete_id)
        
        # Print baseline
        self.print_baseline_summary(athlete_id=athlete_id)
        
        # Print drift
        self.print_drift_summary(athlete_id=athlete_id)
        
        self.print_section_header("✅ PIPELINE TEST COMPLETE", "=")
        
        return True
    
    def close(self):
        """Close connections."""
        if self.retrieval_agent:
            self.retrieval_agent.close()
        if self.mongodb:
            self.mongodb.close()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Full Pipeline Test")
    parser.add_argument("--athlete-id", type=str, default="test_athlete_8plus_001", help="Athlete ID")
    parser.add_argument("--activity", type=str, default="gymnastics", help="Activity")
    parser.add_argument("--technique", type=str, default="back_handspring", help="Technique")
    parser.add_argument("--wait", type=int, default=90, help="Wait time for processing (seconds)")
    
    args = parser.parse_args()
    
    test = ComprehensivePipelineTest()
    
    try:
        success = test.run_full_pipeline_test(
            athlete_id=args.athlete_id,
            activity=args.activity,
            technique=args.technique,
            wait_time=args.wait
        )
        
        if success:
            logger.info("✅ Test completed successfully")
        else:
            logger.error("❌ Test failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
    except Exception as e:
        logger.error(f"❌ Test error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        test.close()


if __name__ == "__main__":
    main()

