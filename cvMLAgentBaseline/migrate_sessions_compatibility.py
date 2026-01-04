#!/usr/bin/env python3
"""
Migrate sessions collection to be compatible with retrieval agent.

Fixes:
1. Transform metrics_results array → metrics dictionary (aggregate per-frame data)
2. Populate activity field (from session_type or default)
3. Populate technique field (from metrics_results or classification_results)
4. Add baseline_eligible field
5. Add capture_confidence_score field
6. Add timestamp field (from session_date)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoAgent.mongodb_service import MongoDBService
from datetime import datetime
import statistics

mongodb = MongoDBService()
mongodb.connect()

sessions_collection = mongodb.get_sessions_collection()

print("=" * 70)
print("🔄 MIGRATING SESSIONS COLLECTION FOR COMPATIBILITY")
print("=" * 70)

# Get all sessions
sessions = list(sessions_collection.find({}))
total = len(sessions)
print(f"\n📊 Found {total} sessions to migrate\n")

updated_count = 0
skipped_count = 0

for idx, session in enumerate(sessions, 1):
    session_id = session.get("_id")
    athlete_id = session.get("athlete_id", "unknown")
    
    print(f"[{idx}/{total}] Processing session: {session_id} (athlete: {athlete_id})")
    
    update_fields = {}
    
    # 1. Transform metrics_results array → metrics dictionary
    if "metrics_results" in session and session["metrics_results"]:
        metrics_results = session["metrics_results"]
        
        # Aggregate metrics from all frames
        metrics_dict = {}
        
        # Collect all values for each metric key
        metric_values = {}
        for frame_data in metrics_results:
            for key, value in frame_data.items():
                if key not in ["frame", "timestamp", "technique", "in_flight"]:
                    if isinstance(value, (int, float)):
                        if key not in metric_values:
                            metric_values[key] = []
                        metric_values[key].append(value)
        
        # Calculate statistics for each metric
        for key, values in metric_values.items():
            if values:
                # Use mean for most metrics, min/max for specific ones
                if "min" in key.lower():
                    metrics_dict[key] = min(values)
                elif "max" in key.lower():
                    metrics_dict[key] = max(values)
                elif "avg" in key.lower() or "average" in key.lower():
                    metrics_dict[key] = statistics.mean(values)
                else:
                    # Default to mean
                    metrics_dict[key] = statistics.mean(values)
        
        if metrics_dict:
            update_fields["metrics"] = metrics_dict
            print(f"   ✅ Created metrics dict with {len(metrics_dict)} metrics")
    
    # 2. Populate activity field
    if not session.get("activity"):
        # Try to infer from session_type or default
        session_type = session.get("session_type", "").lower()
        if "training" in session_type or "practice" in session_type:
            activity = "gymnastics"
        else:
            activity = "gymnastics"  # Default
        
        update_fields["activity"] = activity
        print(f"   ✅ Set activity: {activity}")
    
    # 3. Populate technique field
    if not session.get("technique"):
        technique = None
        
        # Try to get from metrics_results
        if "metrics_results" in session and session["metrics_results"]:
            for frame_data in session["metrics_results"]:
                if "technique" in frame_data and frame_data["technique"]:
                    technique = frame_data["technique"]
                    break
        
        # Try to get from classification_results
        if not technique and "classification_results" in session:
            class_results = session["classification_results"]
            if isinstance(class_results, dict) and "frame_mappings" in class_results:
                frame_mappings = class_results["frame_mappings"]
                if frame_mappings:
                    # Get first frame's technique
                    first_frame = list(frame_mappings.values())[0]
                    if isinstance(first_frame, dict) and "action" in first_frame:
                        action = first_frame["action"]
                        # Extract technique from action (e.g., "bb_split_leap_forward" -> "split_leap")
                        if action:
                            technique = action.replace("bb_", "").replace("_forward", "").replace("_backward", "")
        
        if technique:
            update_fields["technique"] = technique
            print(f"   ✅ Set technique: {technique}")
        else:
            update_fields["technique"] = "unknown"
            print(f"   ⚠️  Could not determine technique, set to 'unknown'")
    
    # 4. Add timestamp field (from session_date)
    if not session.get("timestamp") and session.get("session_date"):
        session_date = session["session_date"]
        try:
            # Parse session_date and convert to ISO format
            if isinstance(session_date, str):
                # Try different date formats
                try:
                    dt = datetime.strptime(session_date, "%Y-%m-%d")
                except:
                    try:
                        dt = datetime.strptime(session_date, "%Y-%m-%d %H:%M:%S")
                    except:
                        dt = datetime.now()
            else:
                dt = datetime.now()
            
            update_fields["timestamp"] = dt.isoformat()
            print(f"   ✅ Set timestamp: {update_fields['timestamp']}")
        except Exception as e:
            print(f"   ⚠️  Could not parse session_date: {e}")
    
    # 5. Add capture_confidence_score (default to 0.8 if not present)
    if "capture_confidence_score" not in session:
        # Try to infer from other fields or use default
        confidence = 0.8  # Default
        update_fields["capture_confidence_score"] = confidence
        print(f"   ✅ Set capture_confidence_score: {confidence}")
    
    # 6. Add baseline_eligible field
    if "baseline_eligible" not in session:
        confidence = update_fields.get("capture_confidence_score", session.get("capture_confidence_score", 0.8))
        baseline_eligible = confidence >= 0.7
        update_fields["baseline_eligible"] = baseline_eligible
        print(f"   ✅ Set baseline_eligible: {baseline_eligible}")
    
    # 7. Add session_id if not present
    if "session_id" not in session:
        update_fields["session_id"] = str(session_id)
        print(f"   ✅ Set session_id: {update_fields['session_id']}")
    
    # Update session if we have changes
    if update_fields:
        try:
            sessions_collection.update_one(
                {"_id": session_id},
                {"$set": update_fields}
            )
            updated_count += 1
            print(f"   ✅ Updated session")
        except Exception as e:
            print(f"   ❌ Error updating session: {e}")
            skipped_count += 1
    else:
        skipped_count += 1
        print(f"   ⏭️  No updates needed")
    
    print()

print("=" * 70)
print("📊 MIGRATION SUMMARY")
print("=" * 70)
print(f"   Total sessions: {total}")
print(f"   ✅ Updated: {updated_count}")
print(f"   ⏭️  Skipped: {skipped_count}")
print()

# Verify migration
print("🔍 Verifying migration...")
sessions_with_metrics = sessions_collection.count_documents({"metrics": {"$exists": True, "$ne": {}}})
sessions_with_activity = sessions_collection.count_documents({"activity": {"$exists": True, "$ne": None}})
sessions_with_technique = sessions_collection.count_documents({"technique": {"$exists": True, "$ne": None}})
sessions_eligible = sessions_collection.count_documents({
    "baseline_eligible": True,
    "capture_confidence_score": {"$gte": 0.7}
})

print(f"   Sessions with metrics: {sessions_with_metrics}/{total}")
print(f"   Sessions with activity: {sessions_with_activity}/{total}")
print(f"   Sessions with technique: {sessions_with_technique}/{total}")
print(f"   Baseline eligible sessions: {sessions_eligible}/{total}")

if sessions_with_metrics == total and sessions_with_activity == total and sessions_with_technique == total:
    print("\n✅ Migration successful! All sessions are now compatible.")
else:
    print("\n⚠️  Migration incomplete. Some sessions may still need manual fixes.")

mongodb.close()

