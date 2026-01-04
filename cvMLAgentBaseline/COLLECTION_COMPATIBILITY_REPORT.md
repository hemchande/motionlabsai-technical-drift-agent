# MongoDB Collection Compatibility Report

## 🔍 Analysis Date
2026-01-03

## 📊 Summary

### Collections Status

| Collection | Documents | Status | Issues |
|------------|-----------|--------|--------|
| `sessions` | 49 | ❌ **INCOMPATIBLE** | Missing required fields |
| `insights` | 0 | ⚠️ Empty | No data yet |
| `trends` | 0 | ⚠️ Empty | No data yet |
| `baselines` | 0 | ⚠️ Empty | No data yet |
| `alerts` | 0 | ⚠️ Empty | No data yet |
| `drift_detection_flags` | 0 | ⚠️ Empty | No data yet |

---

## ❌ CRITICAL ISSUES: `sessions` Collection

### Missing Required Fields

1. **`activity` field**
   - **Expected**: String field (e.g., "gymnastics")
   - **Actual**: Field doesn't exist or is `None`
   - **Impact**: Cannot filter sessions by activity
   - **Fix**: Populate `activity` field in all sessions

2. **`metrics` field**
   - **Expected**: Dictionary with metric keys (e.g., `height_off_floor_meters`, `landing_knee_bend_min`)
   - **Actual**: Field doesn't exist
   - **Impact**: **Cannot extract form issues** - retrieval agent requires `metrics` field
   - **Fix**: Transform `metrics_results` array to `metrics` dictionary OR update retrieval agent to read from `metrics_results`

3. **`technique` field**
   - **Expected**: String field (e.g., "back_handspring")
   - **Actual**: Field is `None` or doesn't exist
   - **Impact**: Cannot filter sessions by technique
   - **Fix**: Populate `technique` field in all sessions

### Additional Issues

4. **No baseline eligible sessions**
   - **Expected**: Sessions with `baseline_eligible: true` and `capture_confidence_score >= 0.7`
   - **Actual**: 0/49 sessions meet criteria
   - **Impact**: Cannot establish baselines
   - **Fix**: Set `baseline_eligible: true` and ensure `capture_confidence_score >= 0.7`

5. **No `form_issues` field**
   - **Expected**: Array of form issues (populated by retrieval agent)
   - **Actual**: Field doesn't exist
   - **Impact**: Cannot find sessions with form issues
   - **Note**: This is expected - retrieval agent will populate this

---

## 📋 Actual Session Structure

### Current Fields (from analysis)

**Top-level fields found:**
- `_id`: ObjectId ✅
- `athlete_id`: string ✅
- `athlete_name`: string ✅
- `session_date`: string (instead of `timestamp`)
- `session_type`: string
- `filename`: string
- `video_filename`: string
- `status`: string
- `classification_completed_at`: datetime
- `metrics_completed_at`: datetime
- `metrics_results`: **array** (instead of `metrics` dict)
- `classification_results`: dict
- `insights`: array (different structure than expected)
- `agent_processing_status`: string
- `agent_error`: string
- `agent_failed_at`: datetime
- `agent_processed_at`: datetime
- `agent_report`: string

### Key Differences

| Expected Field | Actual Field | Issue |
|----------------|--------------|-------|
| `metrics` (dict) | `metrics_results` (array) | **Structure mismatch** |
| `timestamp` | `session_date` | Field name difference |
| `activity` | ❌ Missing | **Required field missing** |
| `technique` | ❌ Missing/None | **Required field missing** |
| `baseline_eligible` | ❌ Missing | Cannot establish baselines |
| `capture_confidence_score` | ❌ Missing | Cannot establish baselines |

---

## 🔧 Required Fixes

### Option 1: Transform Existing Data (Recommended)

Create a migration script to:
1. **Extract metrics from `metrics_results` array** → convert to `metrics` dictionary
2. **Populate `activity` field** (from `session_type` or default to "gymnastics")
3. **Populate `technique` field** (from `classification_results` or default)
4. **Add `baseline_eligible` field** (set to `true` if `capture_confidence_score >= 0.7`)
5. **Add `capture_confidence_score` field** (calculate or set default)
6. **Add `timestamp` field** (from `session_date`)

### Option 2: Update Retrieval Agent

Modify `form_correction_retrieval_agent.py` to:
1. **Read from `metrics_results` array** instead of `metrics` dict
2. **Handle missing `activity`/`technique`** fields gracefully
3. **Use `session_date`** instead of `timestamp`

**Recommendation**: Option 1 is better - keeps retrieval agent simple and ensures data consistency.

---

## 📝 Expected vs Actual Structure

### Expected `sessions` Document

```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "athlete_001",
  "athlete_name": "Jordan Chiles",
  "activity": "gymnastics",  // ❌ MISSING
  "technique": "back_handspring",  // ❌ MISSING
  "timestamp": "2024-10-15T12:30:00Z",
  "metrics": {  // ❌ MISSING (has metrics_results array instead)
    "height_off_floor_meters": 0.25,
    "landing_knee_bend_min": 155.0,
    "hip_angle": 125.0
  },
  "capture_confidence_score": 0.85,  // ❌ MISSING
  "baseline_eligible": true,  // ❌ MISSING
  "form_issues": []  // Will be populated by retrieval agent
}
```

### Actual `sessions` Document

```json
{
  "_id": ObjectId,
  "athlete_id": "athlete_001",
  "athlete_name": "Jordan Chiles",
  "session_date": "2024-10-15",  // ✅ Has this
  "session_type": "Training",
  "metrics_results": [  // ✅ Has this (but array, not dict)
    {
      "frame": 0,
      "timestamp": 0.0,
      "technique": "back_handspring",
      "flight_time": 0.5,
      "height_from_beam": 0.25,
      "landing_stability": 0.8
    }
  ],
  "classification_results": {...},
  "insights": [...]  // Different structure
}
```

---

## 🎯 Compatibility Checklist

### `sessions` Collection

- [ ] Add `activity` field to all sessions
- [ ] Add `technique` field to all sessions
- [ ] Transform `metrics_results` array → `metrics` dictionary
- [ ] Add `capture_confidence_score` field
- [ ] Add `baseline_eligible` field
- [ ] Add `timestamp` field (from `session_date`)
- [ ] Add `session_id` field (if not present)

### Other Collections

- [ ] `insights` - Will be created by retrieval agent (no action needed)
- [ ] `trends` - Will be created by retrieval agent (no action needed)
- [ ] `baselines` - Will be created when 8+ eligible sessions exist
- [ ] `alerts` - Will be created when drift is detected
- [ ] `drift_detection_flags` - Will be created when baseline is established

---

## 🚀 Next Steps

1. **Create migration script** to transform existing sessions
2. **Run migration** on all 49 sessions
3. **Verify compatibility** by re-running analysis
4. **Test retrieval agent** with transformed data

---

## 📚 References

- `COLLECTION_SCHEMAS_AND_PIPELINE.md` - Expected schemas
- `form_correction_retrieval_agent.py` - Retrieval agent implementation
- `analyze_collection_compatibility.py` - Analysis script

