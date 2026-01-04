#!/usr/bin/env python3
"""
Analyze MongoDB collection compatibility with retrieval agent expectations.

Checks:
1. sessions collection - required fields for form issue extraction
2. insights collection - structure matches expectations
3. trends collection - structure matches expectations
4. baselines collection - structure matches expectations
5. alerts collection - structure matches expectations
6. drift_detection_flags collection - structure matches expectations
"""
import sys
from pathlib import Path

# Add parent directory to path (same as retrieval_queue_worker.py)
sys.path.insert(0, str(Path(__file__).parent.parent))

from videoAgent.mongodb_service import MongoDBService
import json
from collections import defaultdict

mongodb = MongoDBService()
mongodb.connect()

db = mongodb.database

print("=" * 70)
print("🔍 MONGODB COLLECTION COMPATIBILITY ANALYSIS")
print("=" * 70)

# Expected fields for each collection
EXPECTED_FIELDS = {
    "sessions": {
        "required": ["_id", "athlete_id", "athlete_name", "activity", "technique", "metrics", "timestamp"],
        "optional": ["capture_confidence_score", "baseline_eligible", "form_issues", "session_id"]
    },
    "insights": {
        "required": ["_id", "session_id", "athlete_id", "insights", "activity", "technique"],
        "optional": ["insight_count", "form_issue_types", "created_at", "updated_at"]
    },
    "trends": {
        "required": ["_id", "trend_id", "athlete_id", "issue_type", "activity", "technique", "trend_status"],
        "optional": ["baseline_id", "observation", "coaching_options", "created_at", "updated_at"]
    },
    "baselines": {
        "required": ["_id", "athlete_id", "baseline_type", "baseline_vector", "baseline_window", "status"],
        "optional": ["established_at", "created_at", "updated_at"]
    },
    "alerts": {
        "required": ["_id", "alert_id", "alert_type", "athlete_id", "session_id", "drift_metrics"],
        "optional": ["baseline_id", "status", "created_at", "updated_at"]
    },
    "drift_detection_flags": {
        "required": ["_id", "athlete_id", "drift_detection_enabled"],
        "optional": ["baseline_id", "drift_detection_start_date", "drift_threshold", "created_at", "updated_at"]
    }
}

# Key collections to check
KEY_COLLECTIONS = ["sessions", "insights", "trends", "baselines", "alerts", "drift_detection_flags"]

results = {}

for collection_name in KEY_COLLECTIONS:
    print(f"\n{'=' * 70}")
    print(f"📋 Collection: {collection_name}")
    print('=' * 70)
    
    collection = db.get_collection(collection_name)
    count = collection.count_documents({})
    print(f"   Total documents: {count}")
    
    if count == 0:
        print("   ⚠️  Collection is empty - no data to analyze")
        results[collection_name] = {"status": "empty", "issues": []}
        continue
    
    # Get sample documents
    samples = list(collection.find().limit(5))
    
    # Analyze field structure
    all_fields = set()
    field_types = defaultdict(set)
    field_presence = defaultdict(int)
    
    for doc in samples:
        def analyze_fields(obj, prefix=''):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f'{prefix}.{key}' if prefix else key
                    all_fields.add(full_key)
                    field_types[full_key].add(type(value).__name__)
                    field_presence[full_key] += 1
                    if isinstance(value, dict):
                        analyze_fields(value, full_key)
                    elif isinstance(value, list) and value and isinstance(value[0], dict):
                        analyze_fields(value[0], f'{full_key}[]')
    
        analyze_fields(doc)
    
    # Check expected fields
    expected = EXPECTED_FIELDS.get(collection_name, {})
    required_fields = expected.get("required", [])
    optional_fields = expected.get("optional", [])
    
    print(f"\n   ✅ Fields found: {len(all_fields)}")
    print(f"   📝 Expected required: {len(required_fields)}")
    print(f"   📝 Expected optional: {len(optional_fields)}")
    
    # Check required fields
    missing_required = []
    present_required = []
    
    for field in required_fields:
        # Check if field exists (could be nested)
        found = False
        for actual_field in all_fields:
            if field == actual_field or actual_field.startswith(field + ".") or actual_field.endswith("." + field):
                found = True
                break
        
        if found:
            present_required.append(field)
        else:
            missing_required.append(field)
    
    # Check optional fields
    missing_optional = []
    present_optional = []
    
    for field in optional_fields:
        found = False
        for actual_field in all_fields:
            if field == actual_field or actual_field.startswith(field + ".") or actual_field.endswith("." + field):
                found = True
                break
        
        if found:
            present_optional.append(field)
        else:
            missing_optional.append(field)
    
    # Report
    issues = []
    
    if missing_required:
        print(f"\n   ❌ MISSING REQUIRED FIELDS:")
        for field in missing_required:
            print(f"      - {field}")
            issues.append(f"Missing required field: {field}")
    else:
        print(f"\n   ✅ All required fields present")
    
    if present_required:
        print(f"\n   ✅ Required fields present:")
        for field in present_required[:10]:
            print(f"      - {field}")
        if len(present_required) > 10:
            print(f"      ... and {len(present_required) - 10} more")
    
    if present_optional:
        print(f"\n   ✅ Optional fields present ({len(present_optional)}):")
        for field in present_optional[:5]:
            print(f"      - {field}")
        if len(present_optional) > 5:
            print(f"      ... and {len(present_optional) - 5} more")
    
    # Special checks for sessions collection
    if collection_name == "sessions":
        print(f"\n   🔍 Special checks for sessions collection:")
        
        # Check for metrics field
        sessions_with_metrics = collection.count_documents({"metrics": {"$exists": True, "$ne": {}}})
        print(f"      Sessions with metrics field: {sessions_with_metrics}/{count}")
        if sessions_with_metrics == 0:
            issues.append("No sessions have metrics field - form issues cannot be extracted")
        
        # Check for activity field
        sessions_with_activity = collection.count_documents({"activity": {"$exists": True, "$ne": None}})
        print(f"      Sessions with activity field: {sessions_with_activity}/{count}")
        if sessions_with_activity == 0:
            issues.append("No sessions have activity field - cannot filter by activity")
        
        # Check for technique field
        sessions_with_technique = collection.count_documents({"technique": {"$exists": True, "$ne": None}})
        print(f"      Sessions with technique field: {sessions_with_technique}/{count}")
        if sessions_with_technique == 0:
            issues.append("No sessions have technique field - cannot filter by technique")
        
        # Check for baseline_eligible
        sessions_eligible = collection.count_documents({
            "baseline_eligible": True,
            "capture_confidence_score": {"$gte": 0.7}
        })
        print(f"      Baseline eligible sessions: {sessions_eligible}/{count}")
        if sessions_eligible == 0:
            issues.append("No baseline eligible sessions - baselines cannot be established")
        
        # Check for form_issues field (should be populated by retrieval agent)
        sessions_with_form_issues = collection.count_documents({"form_issues": {"$exists": True, "$ne": []}})
        print(f"      Sessions with form_issues field: {sessions_with_form_issues}/{count}")
        if sessions_with_form_issues == 0:
            print(f"      ⚠️  No sessions have form_issues - retrieval agent needs to extract them")
    
    # Special checks for insights collection
    if collection_name == "insights":
        print(f"\n   🔍 Special checks for insights collection:")
        
        # Check insights array structure
        sample_insight = samples[0] if samples else None
        if sample_insight and "insights" in sample_insight:
            insights_list = sample_insight["insights"]
            if insights_list and isinstance(insights_list, list) and len(insights_list) > 0:
                first_insight = insights_list[0]
                print(f"      Sample insight structure: {list(first_insight.keys())}")
                
                # Check for required insight fields
                if "insight" not in first_insight and "description" not in first_insight:
                    issues.append("Insights missing 'insight' or 'description' field")
                if "is_monitored" not in first_insight:
                    issues.append("Insights missing 'is_monitored' field")
                if "coach_follow_up" not in first_insight:
                    issues.append("Insights missing 'coach_follow_up' field")
    
    results[collection_name] = {
        "status": "ok" if not issues else "issues",
        "issues": issues,
        "missing_required": missing_required,
        "present_required": present_required,
        "field_count": len(all_fields)
    }
    
    # Show sample document structure (simplified)
    if samples:
        sample = samples[0]
        print(f"\n   📄 Sample document (top-level fields only):")
        top_level_fields = {k: type(v).__name__ for k, v in sample.items() if not isinstance(v, (dict, list)) or k in ["_id", "session_id", "athlete_id"]}
        for key, value_type in list(top_level_fields.items())[:15]:
            print(f"      {key}: {value_type}")
        if len(top_level_fields) > 15:
            print(f"      ... and {len(top_level_fields) - 15} more top-level fields")

# Summary
print(f"\n{'=' * 70}")
print("📊 COMPATIBILITY SUMMARY")
print('=' * 70)

for collection_name, result in results.items():
    status_icon = "✅" if result["status"] == "ok" else "❌"
    print(f"\n{status_icon} {collection_name}:")
    if result["status"] == "empty":
        print(f"   ⚠️  Collection is empty")
    elif result["issues"]:
        print(f"   ❌ Issues found ({len(result['issues'])}):")
        for issue in result["issues"]:
            print(f"      - {issue}")
    else:
        print(f"   ✅ Compatible - all required fields present")
        print(f"   📝 Fields: {result['field_count']} total")

mongodb.close()

