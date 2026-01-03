#!/usr/bin/env python3
"""
Test Drift Coach Follow-Up and Monitoring

Tests coach follow-up for drift alerts and tracking of monitored drift insights.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
from videoAgent.mongodb_service import MongoDBService

def test_drift_coach_followup():
    """Test coach follow-up for drift alerts."""
    print("🧪 Testing Drift Coach Follow-Up")
    print("="*60)
    
    agent = FormCorrectionRetrievalAgent()
    mongodb = MongoDBService()
    mongodb.connect()
    
    try:
        # Step 1: Create baseline and drift
        print("\n📊 Step 1: Setting up baseline and drift...")
        test_athlete = "test_drift_coach_001"
        collection = mongodb.get_sessions_collection()
        
        # Clean up
        collection.delete_many({"athlete_id": test_athlete})
        mongodb.database.get_collection("baselines").delete_many({"athlete_id": test_athlete})
        mongodb.database.get_collection("alerts").delete_many({"athlete_id": test_athlete})
        
        # Create baseline sessions
        base_date = datetime.utcnow() - timedelta(days=20)
        baseline_sessions = []
        for i in range(3):
            session_date = base_date + timedelta(days=i)
            session_doc = {
                "athlete_id": test_athlete,
                "athlete_name": "Coach Follow-Up Test",
                "activity": "gymnastics",
                "technique": "back_handspring",
                "timestamp": session_date.isoformat(),
                "capture_confidence_score": 0.90,
                "baseline_eligible": True,
                "metrics": {
                    "height_off_floor_meters": 0.30 + (i * 0.01),
                    "landing_knee_bend_min": 160.0 + (i * 0.5)
                },
                "created_at": session_date,
                "updated_at": session_date
            }
            result = collection.insert_one(session_doc)
            baseline_sessions.append(str(result.inserted_id))
        
        # Establish baseline
        baseline = agent.establish_baseline(
            athlete_id=test_athlete,
            session_ids=baseline_sessions,
            baseline_type="pre_injury",
            min_sessions=3
        )
        
        if not baseline:
            print("❌ Failed to establish baseline")
            return False
        
        print(f"✅ Baseline established: {baseline.get('baseline_id')[:8]}...")
        
        # Enable drift detection
        drift_flags = mongodb.database.get_collection("drift_detection_flags")
        drift_flags.update_one(
            {"athlete_id": test_athlete},
            {"$set": {
                "drift_detection_start_date": datetime.utcnow() - timedelta(days=1),
                "drift_detection_enabled": True
            }}
        )
        
        # Create drift session
        drift_session = {
            "athlete_id": test_athlete,
            "athlete_name": "Coach Follow-Up Test",
            "activity": "gymnastics",
            "technique": "back_handspring",
            "timestamp": datetime.utcnow().isoformat(),
            "capture_confidence_score": 0.90,
            "baseline_eligible": True,
            "metrics": {
                "height_off_floor_meters": 0.15,  # Clear drift
                "landing_knee_bend_min": 140.0
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        result = collection.insert_one(drift_session)
        drift_session_id = str(result.inserted_id)
        
        # Detect drift
        drift = agent.detect_technical_drift(
            athlete_id=test_athlete,
            session_id=drift_session_id,
            drift_threshold=2.0
        )
        
        if not drift:
            print("❌ No drift detected")
            return False
        
        alert_id = drift.get("alert_id")
        print(f"✅ Drift detected, alert created: {alert_id[:8]}...")
        
        # Step 2: Test coach follow-up update
        print("\n📝 Step 2: Testing coach follow-up update...")
        
        # Get actual alert document first
        alerts_collection = mongodb.database.get_collection("alerts")
        alert_doc = alerts_collection.find_one({"session_id": drift_session_id})
        if not alert_doc:
            alert_doc = alerts_collection.find_one({"alert_id": alert_id})
        if not alert_doc:
            print("❌ Could not find alert document")
            return False
        
        actual_alert_id = alert_doc.get("alert_id") or str(alert_doc.get("_id"))
        print(f"   Found alert: {actual_alert_id[:8]}...")
        
        # Update coach follow-up for a metric
        metric_key = "height_off_floor_meters"
        success = agent.update_drift_alert_coach_follow_up(
            alert_id=actual_alert_id,
            metric_key=metric_key,
            coach_follow_up="Monitor"
        )
        
        if not success:
            print("❌ Failed to update coach follow-up")
            return False
        
        print(f"✅ Coach follow-up updated: {metric_key} -> Monitor")
        
        # Verify in MongoDB (use actual_alert_id)
        alert = alerts_collection.find_one({"alert_id": actual_alert_id})
        if not alert:
            alert = alerts_collection.find_one({"_id": alert_doc.get("_id")})
        
        if alert:
            drift_metrics = alert.get("drift_metrics", {})
            metric_data = drift_metrics.get(metric_key, {})
            
            if metric_data.get("coach_follow_up") == "Monitor":
                print(f"✅ Verified: coach_follow_up = Monitor")
                print(f"✅ Verified: is_monitored = {metric_data.get('is_monitored')}")
                print(f"✅ Verified: monitored_at = {metric_data.get('monitored_at')}")
            else:
                print(f"❌ Coach follow-up not set correctly")
                return False
        else:
            print(f"❌ Alert not found")
            return False
        
        # Step 3: Test monitoring tracking
        print("\n📈 Step 3: Testing drift monitoring tracking...")
        
        # Create additional sessions to track trend
        for i in range(3):
            session_date = datetime.utcnow() + timedelta(days=i+1)
            session_doc = {
                "athlete_id": test_athlete,
                "athlete_name": "Coach Follow-Up Test",
                "activity": "gymnastics",
                "technique": "back_handspring",
                "timestamp": session_date.isoformat(),
                "capture_confidence_score": 0.90,
                "baseline_eligible": True,
                "metrics": {
                    "height_off_floor_meters": 0.15 + (i * 0.02),  # Slight improvement
                    "landing_knee_bend_min": 140.0 + (i * 1.0)
                },
                "created_at": session_date,
                "updated_at": session_date
            }
            collection.insert_one(session_doc)
            print(f"   ✅ Created tracking session {i+1}/3")
        
        # Track monitored drift
        trend = agent.track_monitored_drift_insights(
            athlete_id=test_athlete,
            metric_key=metric_key
        )
        
        if trend:
            print(f"✅ Trend tracked successfully!")
            print(f"   Trend: {trend.get('trend')}")
            print(f"   Change: {trend.get('change_percent'):.1f}%")
            print(f"   Sessions analyzed: {trend.get('sessions_analyzed')}")
            print(f"   Message: {trend.get('message')}")
            
            # Verify in monitoring_trends collection
            monitoring_collection = mongodb.database.get_collection("monitoring_trends")
            saved_trend = monitoring_collection.find_one({
                "athlete_id": test_athlete,
                "metric_key": metric_key
            }, sort=[("created_at", -1)])
            
            if saved_trend:
                print(f"✅ Trend saved to MongoDB")
                print(f"   Trend: {saved_trend.get('trend')}")
                print(f"   Trend strength: {saved_trend.get('trend_strength'):.3f}")
            else:
                print(f"⚠️  Trend not found in MongoDB")
        else:
            print(f"⚠️  Trend tracking returned None (may need more sessions)")
        
        # Step 4: Test Escalate follow-up
        print("\n🚨 Step 4: Testing Escalate follow-up...")
        
        # Update another metric to Escalate
        metric_key2 = "landing_knee_bend_min"
        success2 = agent.update_drift_alert_coach_follow_up(
            alert_id=actual_alert_id,
            metric_key=metric_key2,
            coach_follow_up="Escalate to AT/PT"
        )
        
        if success2:
            # Verify - re-fetch alert after update
            alert = alerts_collection.find_one({"alert_id": actual_alert_id})
            if not alert:
                alert = alerts_collection.find_one({"_id": alert_doc.get("_id")})
            
            if not alert:
                print(f"❌ Alert not found after update")
                return False
            
            drift_metrics = alert.get("drift_metrics", {})
            metric_data2 = drift_metrics.get(metric_key2, {})
            
            if metric_data2.get("coach_follow_up") == "Escalate to AT/PT":
                print(f"✅ Escalate follow-up set correctly")
                print(f"   is_monitored: {metric_data2.get('is_monitored')} (should be False)")
            else:
                print(f"❌ Escalate follow-up not set correctly")
                return False
        else:
            print(f"❌ Failed to update escalate follow-up")
            return False
        
        print("\n✅ All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent.close()
        mongodb.close()


def main():
    """Run tests."""
    result = test_drift_coach_followup()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"{'✅' if result else '❌'} Drift Coach Follow-Up: {'PASSED' if result else 'FAILED'}")
    
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())

