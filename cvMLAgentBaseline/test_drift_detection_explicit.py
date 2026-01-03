#!/usr/bin/env python3
"""
Explicit Drift Detection Test

Creates a baseline and then tests drift detection with clear deviations.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
from videoAgent.mongodb_service import MongoDBService

def setup_baseline_and_drift_test():
    """Set up baseline and test drift detection."""
    print("🧪 Explicit Drift Detection Test")
    print("="*60)
    
    agent = FormCorrectionRetrievalAgent()
    mongodb = MongoDBService()
    mongodb.connect()
    
    try:
        collection = mongodb.get_sessions_collection()
        test_athlete = "test_drift_athlete_001"
        
        # Clean up any existing test data
        print("\n📝 Setting up test data...")
        collection.delete_many({"athlete_id": test_athlete})
        mongodb.database.get_collection("baselines").delete_many({"athlete_id": test_athlete})
        mongodb.database.get_collection("drift_detection_flags").delete_many({"athlete_id": test_athlete})
        mongodb.database.get_collection("alerts").delete_many({"athlete_id": test_athlete})
        
        # Step 1: Create baseline sessions (normal values)
        print("\n📊 Step 1: Creating baseline sessions...")
        base_date = datetime.utcnow() - timedelta(days=20)
        baseline_sessions = []
        
        baseline_metrics = {
            "height_off_floor_meters": 0.30,  # Good height
            "landing_knee_bend_min": 160.0,   # Good extension
            "hip_angle": 130.0,                # Good flexion
            "acl_max_valgus_angle": 5.0        # Good alignment
        }
        
        for i in range(5):
            session_date = base_date + timedelta(days=i)
            session_doc = {
                "athlete_id": test_athlete,
                "athlete_name": "Drift Test Athlete",
                "activity": "gymnastics",
                "technique": "back_handspring",
                "timestamp": session_date.isoformat(),
                "capture_confidence_score": 0.90,
                "baseline_eligible": True,
                "metrics": {
                    "height_off_floor_meters": baseline_metrics["height_off_floor_meters"] + (i * 0.01),
                    "landing_knee_bend_min": baseline_metrics["landing_knee_bend_min"] + (i * 0.5),
                    "hip_angle": baseline_metrics["hip_angle"] + (i * 1.0),
                    "acl_max_valgus_angle": baseline_metrics["acl_max_valgus_angle"] + (i * 0.2)
                },
                "created_at": session_date,
                "updated_at": session_date
            }
            result = collection.insert_one(session_doc)
            baseline_sessions.append(str(result.inserted_id))
            print(f"   ✅ Created baseline session {i+1}/5")
        
        # Step 2: Establish baseline
        print("\n🔄 Step 2: Establishing baseline...")
        baseline = agent.establish_baseline(
            athlete_id=test_athlete,
            session_ids=baseline_sessions[:3],  # Use 3 for baseline
            baseline_type="pre_injury",
            min_sessions=3
        )
        
        if not baseline:
            print("❌ Failed to establish baseline")
            return False
        
        print(f"✅ Baseline established!")
        print(f"   Baseline ID: {baseline.get('baseline_id')}")
        baseline_vector = baseline.get('baseline_vector', {})
        
        # Show baseline values
        print(f"\n   Baseline Values:")
        for metric, stats in baseline_vector.items():
            print(f"     {metric}:")
            print(f"       Mean: {stats.get('mean'):.3f}")
            print(f"       SD: {stats.get('sd'):.3f}")
        
        # Step 3: Enable drift detection immediately
        print("\n🔧 Step 3: Enabling drift detection...")
        drift_flags = mongodb.database.get_collection("drift_detection_flags")
        drift_flags.update_one(
            {"athlete_id": test_athlete},
            {"$set": {
                "drift_detection_start_date": datetime.utcnow() - timedelta(days=1),
                "drift_detection_enabled": True
            }}
        )
        print("   ✅ Drift detection enabled")
        
        # Step 4: Create session with CLEAR drift (worsening)
        print("\n📉 Step 4: Creating session with drift (worsening)...")
        drift_date = datetime.utcnow()
        
        # Create metrics that are clearly worse (3+ sigma deviation)
        drift_metrics = {
            "height_off_floor_meters": 0.15,  # Much lower (baseline ~0.30)
            "landing_knee_bend_min": 140.0,    # Much lower (baseline ~160)
            "hip_angle": 100.0,                # Much lower (baseline ~130)
            "acl_max_valgus_angle": 15.0       # Much higher (baseline ~5)
        }
        
        drift_session = {
            "athlete_id": test_athlete,
            "athlete_name": "Drift Test Athlete",
            "activity": "gymnastics",
            "technique": "back_handspring",
            "timestamp": drift_date.isoformat(),
            "capture_confidence_score": 0.90,
            "baseline_eligible": True,
            "metrics": drift_metrics,
            "created_at": drift_date,
            "updated_at": drift_date
        }
        
        result = collection.insert_one(drift_session)
        drift_session_id = str(result.inserted_id)
        print(f"   ✅ Created drift session: {drift_session_id[:8]}...")
        print(f"\n   Drift Metrics (vs Baseline):")
        for metric, value in drift_metrics.items():
            baseline_stats = baseline_vector.get(metric, {})
            baseline_mean = baseline_stats.get('mean', 0)
            baseline_sd = baseline_stats.get('sd', 0.001)
            z_score = (value - baseline_mean) / baseline_sd if baseline_sd > 0 else 0
            print(f"     {metric}:")
            print(f"       Current: {value:.3f}")
            print(f"       Baseline: {baseline_mean:.3f} ± {baseline_sd:.3f}")
            print(f"       Z-score: {z_score:.2f}σ")
            if abs(z_score) > 2.0:
                print(f"       ⚠️  DRIFT DETECTED (|{z_score:.2f}| > 2.0σ)")
        
        # Step 5: Test drift detection
        print("\n🔍 Step 5: Testing drift detection...")
        drift_result = agent.detect_technical_drift(
            athlete_id=test_athlete,
            session_id=drift_session_id,
            drift_threshold=2.0
        )
        
        if drift_result:
            print(f"✅ DRIFT DETECTED!")
            print(f"   Drift metrics: {drift_result.get('drift_count')}")
            print(f"   Alert ID: {drift_result.get('alert_id')}")
            
            drift_metrics_found = drift_result.get('drift_metrics', {})
            print(f"\n   Drift Details:")
            for metric, data in drift_metrics_found.items():
                print(f"\n     📊 {metric}:")
                print(f"       Baseline: {data.get('baseline_value'):.3f}")
                print(f"       Current: {data.get('current_value'):.3f}")
                print(f"       Z-score: {data.get('z_score'):.2f}σ")
                print(f"       Severity: {data.get('severity')}")
                print(f"       Direction: {data.get('direction')}")
            
            # Verify alert
            alerts_collection = mongodb.database.get_collection("alerts")
            alert = alerts_collection.find_one({"session_id": drift_session_id})
            
            if alert:
                print(f"\n✅ Drift alert created in MongoDB")
                print(f"   Alert type: {alert.get('alert_type')}")
                print(f"   Status: {alert.get('status')}")
                print(f"   Confidence: {alert.get('alert_confidence')}")
            else:
                # Try finding by athlete_id
                alert = alerts_collection.find_one({"athlete_id": test_athlete}, sort=[("created_at", -1)])
                if alert:
                    print(f"\n✅ Drift alert found in MongoDB")
                    print(f"   Alert ID: {alert.get('alert_id')}")
                else:
                    print(f"\n⚠️  Alert not found (may use different ID format)")
            
            return True
        else:
            print("❌ No drift detected (unexpected - metrics should trigger drift)")
            print("\n   Debugging...")
            
            # Check baseline
            baseline_collection = mongodb.database.get_collection("baselines")
            saved_baseline = baseline_collection.find_one({"athlete_id": test_athlete, "status": "active"})
            if saved_baseline:
                print(f"   ✅ Baseline found in MongoDB")
            else:
                print(f"   ❌ Baseline not found")
            
            # Check drift flag
            flag = drift_flags.find_one({"athlete_id": test_athlete})
            if flag:
                print(f"   ✅ Drift flag found")
                print(f"      Enabled: {flag.get('drift_detection_enabled')}")
                print(f"      Start date: {flag.get('drift_detection_start_date')}")
            else:
                print(f"   ❌ Drift flag not found")
            
            # Check session
            session = collection.find_one({"_id": drift_session_id})
            if session:
                print(f"   ✅ Drift session found")
                print(f"      Metrics: {list(session.get('metrics', {}).keys())}")
            else:
                print(f"   ❌ Drift session not found")
            
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        agent.close()
        mongodb.close()


def test_drift_with_improving_metrics():
    """Test drift detection with improving metrics."""
    print("\n" + "="*60)
    print("TEST: Drift Detection - Improving Metrics")
    print("="*60)
    
    agent = FormCorrectionRetrievalAgent()
    mongodb = MongoDBService()
    mongodb.connect()
    
    try:
        collection = mongodb.get_sessions_collection()
        test_athlete = "test_drift_athlete_001"
        
        # Get baseline
        baseline_collection = mongodb.database.get_collection("baselines")
        baseline = baseline_collection.find_one({"athlete_id": test_athlete, "status": "active"})
        
        if not baseline:
            print("⚠️  No baseline found - run setup first")
            return False
        
        baseline_vector = baseline.get('baseline_vector', {})
        
        # Create session with IMPROVING metrics (better than baseline)
        print("\n📈 Creating session with improving metrics...")
        improving_session = {
            "athlete_id": test_athlete,
            "athlete_name": "Drift Test Athlete",
            "activity": "gymnastics",
            "technique": "back_handspring",
            "timestamp": datetime.utcnow().isoformat(),
            "capture_confidence_score": 0.90,
            "baseline_eligible": True,
            "metrics": {
                "height_off_floor_meters": 0.35,  # Better than baseline
                "landing_knee_bend_min": 170.0,   # Better than baseline
                "hip_angle": 140.0,                # Better than baseline
                "acl_max_valgus_angle": 3.0        # Better than baseline
            },
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        result = collection.insert_one(improving_session)
        session_id = str(result.inserted_id)
        print(f"   ✅ Created improving session: {session_id[:8]}...")
        
        # Test drift detection
        print("\n🔍 Testing drift detection for improving metrics...")
        drift = agent.detect_technical_drift(
            athlete_id=test_athlete,
            session_id=session_id,
            drift_threshold=2.0
        )
        
        if drift:
            print(f"✅ Drift detected (improving direction)")
            drift_metrics = drift.get('drift_metrics', {})
            for metric, data in drift_metrics.items():
                print(f"   {metric}: {data.get('direction')} ({data.get('severity')})")
        else:
            print("ℹ️  No drift detected (improving metrics may not trigger alerts)")
        
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
    """Run drift detection tests."""
    # Test 1: Worsening drift
    result1 = setup_baseline_and_drift_test()
    
    # Test 2: Improving metrics
    result2 = test_drift_with_improving_metrics()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"{'✅' if result1 else '❌'} Worsening Drift Detection: {'PASSED' if result1 else 'FAILED'}")
    print(f"{'✅' if result2 else '❌'} Improving Metrics Test: {'PASSED' if result2 else 'FAILED'}")
    
    if result1:
        print("\n✅ Drift detection system is working!")
        return 0
    else:
        print("\n⚠️  Drift detection needs review")
        return 1


if __name__ == "__main__":
    sys.exit(main())

