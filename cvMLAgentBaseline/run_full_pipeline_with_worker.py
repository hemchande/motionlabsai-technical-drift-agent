#!/usr/bin/env python3
"""
Run Full Pipeline with Worker
Starts retrieval worker, sends message, captures all outputs
"""

import json
import logging
import os
import sys
import time
import subprocess
import signal
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoAgent.mongodb_service import MongoDBService
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section_header(title: str, char: str = "="):
    """Print a formatted section header."""
    width = 80
    print("")
    print(char * width)
    print(f"  {title}")
    print(char * width)
    print("")


def print_insights_summary(mongodb, athlete_id=None, activity=None, technique=None):
    """Print comprehensive insights summary."""
    print_section_header("📊 INSIGHTS SUMMARY", "=")
    
    try:
        insights_collection = mongodb.get_insights_collection()
        
        query = {}
        if athlete_id:
            query["athlete_id"] = athlete_id
        if activity:
            query["activity"] = activity
        if technique:
            query["technique"] = technique
        
        recent_insights = list(insights_collection.find(query).sort("updated_at", -1).limit(20))
        
        if not recent_insights:
            print("   ℹ️  No insights found")
            return
        
        print(f"   Found {len(recent_insights)} recent insight documents\n")
        
        for idx, insight_doc in enumerate(recent_insights[:10], 1):
            session_id = insight_doc.get("session_id", "N/A")
            insights_list = insight_doc.get("insights", [])
            athlete_id_doc = insight_doc.get("athlete_id", "N/A")
            activity_doc = insight_doc.get("activity", "N/A")
            technique_doc = insight_doc.get("technique", "N/A")
            updated_at = insight_doc.get("updated_at", "N/A")
            
            print(f"   📋 Insight Document #{idx}")
            print(f"      Session ID: {str(session_id)[:30]}...")
            print(f"      Athlete ID: {athlete_id_doc}")
            print(f"      Activity: {activity_doc} | Technique: {technique_doc}")
            print(f"      Updated: {updated_at}")
            print(f"      Total Insights: {len(insights_list)}")
            
            if insights_list:
                print("      Insights:")
                for i, insight in enumerate(insights_list[:5], 1):
                    desc = insight.get("description", "N/A")
                    monitored = insight.get("is_monitored", False)
                    follow_up = insight.get("coach_follow_up", "None")
                    print(f"         {i}. {desc}")
                    if monitored or follow_up:
                        print(f"            [Monitored: {monitored}, Follow-up: {follow_up}]")
            
            print("")
            
    except Exception as e:
        print(f"   ❌ Error retrieving insights: {e}")


def print_trends_summary(mongodb, athlete_id=None):
    """Print comprehensive trends summary."""
    print_section_header("📈 TRENDS SUMMARY", "=")
    
    try:
        trends_collection = mongodb.get_trends_collection()
        
        query = {}
        if athlete_id:
            query["athlete_id"] = athlete_id
        
        recent_trends = list(trends_collection.find(query).sort("updated_at", -1).limit(20))
        
        if not recent_trends:
            print("   ℹ️  No trends found")
            return
        
        print(f"   Found {len(recent_trends)} recent trends\n")
        
        for idx, trend in enumerate(recent_trends[:10], 1):
            trend_id = trend.get("trend_id", "N/A")
            athlete_name = trend.get("athlete_name", "N/A")
            issue_type = trend.get("issue_type", "N/A")
            status = trend.get("status", "N/A")
            observation = trend.get("observation", "N/A")
            evidence = trend.get("evidence", "N/A")
            coaching_options = trend.get("coaching_options", [])
            updated_at = trend.get("updated_at", "N/A")
            
            print(f"   📊 Trend #{idx}: {issue_type}")
            print(f"      Trend ID: {trend_id}")
            print(f"      Athlete: {athlete_name}")
            print(f"      Status: {status}")
            print(f"      Updated: {updated_at}")
            obs_str = str(observation)
            print(f"      Observation: {obs_str[:100]}..." if len(obs_str) > 100 else f"      Observation: {obs_str}")
            ev_str = str(evidence)
            print(f"      Evidence: {ev_str[:100]}..." if len(ev_str) > 100 else f"      Evidence: {ev_str}")
            
            if coaching_options:
                print(f"      Coaching Options ({len(coaching_options)}):")
                for i, option in enumerate(coaching_options[:3], 1):
                    opt_str = str(option)
                    print(f"         {i}. {opt_str[:80]}..." if len(opt_str) > 80 else f"         {i}. {opt_str}")
            
            print("")
            
    except Exception as e:
        print(f"   ❌ Error retrieving trends: {e}")


def print_baseline_summary(mongodb, athlete_id=None):
    """Print comprehensive baseline summary."""
    print_section_header("📊 BASELINE SUMMARY", "=")
    
    try:
        baselines_collection = mongodb.database.get_collection("baselines")
        
        query = {"status": "active"}
        if athlete_id:
            query["athlete_id"] = athlete_id
        
        baselines = list(baselines_collection.find(query).sort("established_at", -1).limit(10))
        
        if not baselines:
            print("   ℹ️  No active baselines found")
            return
        
        print(f"   Found {len(baselines)} active baseline(s)\n")
        
        for idx, baseline in enumerate(baselines, 1):
            baseline_id = str(baseline.get("_id", "N/A"))[:20]
            athlete_id_doc = baseline.get("athlete_id", "N/A")
            baseline_type = baseline.get("baseline_type", "N/A")
            baseline_vector = baseline.get("baseline_vector", {})
            baseline_window = baseline.get("baseline_window", {})
            session_count = baseline_window.get("session_count", 0)
            established_at = baseline.get("established_at", "N/A")
            
            print(f"   📊 Baseline #{idx}")
            print(f"      Baseline ID: {baseline_id}...")
            print(f"      Athlete ID: {athlete_id_doc}")
            print(f"      Type: {baseline_type}")
            print(f"      Sessions Used: {session_count}")
            print(f"      Established: {established_at}")
            print(f"      Metrics in Baseline: {len(baseline_vector)}")
            
            if baseline_vector:
                print("      Sample Metrics:")
                for i, (metric_key, metric_data) in enumerate(list(baseline_vector.items())[:5], 1):
                    mean = metric_data.get("mean", "N/A")
                    sd = metric_data.get("sd", "N/A")
                    if isinstance(mean, (int, float)) and isinstance(sd, (int, float)):
                        print(f"         {i}. {metric_key}: μ={mean:.3f}, σ={sd:.3f}")
                    else:
                        print(f"         {i}. {metric_key}: mean={mean}, sd={sd}")
            
            print("")
            
    except Exception as e:
        print(f"   ❌ Error retrieving baselines: {e}")


def print_drift_summary(mongodb, athlete_id=None):
    """Print comprehensive drift detection summary."""
    print_section_header("🔍 DRIFT DETECTION SUMMARY", "=")
    
    try:
        alerts_collection = mongodb.database.get_collection("alerts")
        
        query = {"alert_type": "technical_drift"}
        if athlete_id:
            query["athlete_id"] = athlete_id
        
        alerts = list(alerts_collection.find(query).sort("created_at", -1).limit(10))
        
        if not alerts:
            print("   ℹ️  No drift alerts found")
            return
        
        print(f"   Found {len(alerts)} drift alert(s)\n")
        
        for idx, alert in enumerate(alerts, 1):
            alert_id = alert.get("alert_id", "N/A")
            athlete_id_doc = alert.get("athlete_id", "N/A")
            session_id = alert.get("session_id", "N/A")
            drift_metrics = alert.get("drift_metrics", {})
            status = alert.get("status", "N/A")
            created_at = alert.get("created_at", "N/A")
            
            print(f"   🔍 Drift Alert #{idx}")
            print(f"      Alert ID: {alert_id}")
            print(f"      Athlete ID: {athlete_id_doc}")
            print(f"      Session ID: {str(session_id)[:30]}...")
            print(f"      Status: {status}")
            print(f"      Created: {created_at}")
            print(f"      Metrics with Drift: {len(drift_metrics)}")
            print("")
            
            if drift_metrics:
                print("      Drift Metrics:")
                for i, (metric_key, metric_data) in enumerate(list(drift_metrics.items())[:5], 1):
                    baseline_value = metric_data.get("baseline_value", "N/A")
                    current_value = metric_data.get("current_value", "N/A")
                    z_score = metric_data.get("z_score", "N/A")
                    severity = metric_data.get("severity", "N/A")
                    direction = metric_data.get("direction", "N/A")
                    coach_follow_up = metric_data.get("coach_follow_up", "None")
                    
                    print(f"         {i}. {metric_key}")
                    if isinstance(baseline_value, (int, float)) and isinstance(current_value, (int, float)):
                        print(f"            Baseline: {baseline_value:.3f} | Current: {current_value:.3f}")
                    else:
                        print(f"            Baseline: {baseline_value} | Current: {current_value}")
                    if isinstance(z_score, (int, float)):
                        print(f"            Z-score: {z_score:.2f}σ | Severity: {severity.upper()} | Direction: {direction.upper()}")
                    else:
                        print(f"            Z-score: {z_score} | Severity: {severity.upper()} | Direction: {direction.upper()}")
                    if coach_follow_up:
                        print(f"            Coach Follow-up: {coach_follow_up}")
            
            print("")
            
    except Exception as e:
        print(f"   ❌ Error retrieving drift alerts: {e}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Full Pipeline with Worker")
    parser.add_argument("--athlete-id", type=str, default="test_athlete_8plus_001", help="Athlete ID")
    parser.add_argument("--activity", type=str, default="gymnastics", help="Activity")
    parser.add_argument("--technique", type=str, default="back_handspring", help="Technique")
    parser.add_argument("--wait", type=int, default=120, help="Wait time for processing (seconds)")
    
    args = parser.parse_args()
    
    session_id = f"test_pipeline_{int(time.time())}"
    worker_process = None
    
    print_section_header("🚀 FULL PIPELINE TEST WITH WORKER", "=")
    print(f"   Test Configuration:")
    print(f"      Session ID: {session_id}")
    print(f"      Athlete ID: {args.athlete_id}")
    print(f"      Activity: {args.activity}")
    print(f"      Technique: {args.technique}")
    print("")
    
    try:
        # Step 1: Start retrieval worker
        print_section_header("STEP 1: Starting Retrieval Queue Worker", "-")
        
        worker_process = subprocess.Popen(
            ["python3", "retrieval_queue_worker.py", "--max-messages", "1", "--timeout", "2"],
            cwd=Path(__file__).parent,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        print("   ✅ Worker started (PID: {})".format(worker_process.pid))
        print("   Waiting 5 seconds for initialization...")
        time.sleep(5)
        print("")
        
        # Step 2: Send message via mock video agent
        print_section_header("STEP 2: Sending Message via Mock Video Agent", "-")
        
        from mock_video_agent import MockVideoAgent
        
        video_agent = MockVideoAgent()
        if not video_agent.redis_client:
            print("   ❌ Redis not available. Cannot run test.")
            return
        
        success = video_agent.simulate_video_call_ended(
            session_id=session_id,
            athlete_id=args.athlete_id,
            activity=args.activity,
            technique=args.technique,
            athlete_name="Test Athlete"
        )
        
        if success:
            print(f"   ✅ Message sent successfully to queue")
        else:
            print(f"   ❌ Failed to send message")
            return
        
        video_agent.close()
        print("")
        
        # Step 3: Monitor worker output
        print_section_header("STEP 3: Monitoring Worker Processing", "-")
        print(f"   Waiting {args.wait} seconds for processing...")
        print("   Worker output:")
        print("   " + "-" * 76)
        
        start_time = time.time()
        output_lines = []
        
        while time.time() - start_time < args.wait:
            line = worker_process.stdout.readline()
            if not line:
                time.sleep(0.1)
                continue
            
            line = line.rstrip()
            output_lines.append(line)
            print(f"   {line}")
            
            # Check for completion
            if "Processed 1 messages" in line or "Stopping" in line:
                print("   ✅ Worker completed processing")
                break
            
            if len(output_lines) > 500:
                print("   ⚠️  Output truncated (too many lines)")
                break
        
        print("   " + "-" * 76)
        print("")
        
        # Wait a bit more for any final processing
        time.sleep(2)
        
        # Step 4: Retrieve and print all outputs
        print_section_header("STEP 4: Retrieving Pipeline Outputs from MongoDB", "-")
        
        mongodb = MongoDBService()
        mongodb.connect()
        
        # Print insights
        print_insights_summary(mongodb, athlete_id=args.athlete_id, activity=args.activity, technique=args.technique)
        
        # Print trends
        print_trends_summary(mongodb, athlete_id=args.athlete_id)
        
        # Print baseline
        print_baseline_summary(mongodb, athlete_id=args.athlete_id)
        
        # Print drift
        print_drift_summary(mongodb, athlete_id=args.athlete_id)
        
        mongodb.close()
        
        print_section_header("✅ PIPELINE TEST COMPLETE", "=")
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"❌ Test error: {e}", exc_info=True)
    finally:
        # Clean up worker
        if worker_process:
            try:
                worker_process.terminate()
                worker_process.wait(timeout=5)
                print("   ✅ Worker stopped")
            except:
                try:
                    worker_process.kill()
                except:
                    pass


if __name__ == "__main__":
    main()

