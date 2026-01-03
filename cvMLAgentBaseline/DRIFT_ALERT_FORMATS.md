# Drift Alert Formats - Complete Reference

This document shows how drift alerts look in all formats: MongoDB, console output, Redis queue, and WebSocket.

---

## 1. MongoDB Alert Document

**Collection**: `alerts`  
**Document Structure**:

```json
{
  "_id": ObjectId("695954d54c28ce6abc013a25"),
  "alert_id": "drift_test_drift_athlete_001_695954d54c28ce6abc013a24_1767483701",
  "alert_type": "technical_drift",
  "alert_created_at": ISODate("2026-01-03T17:41:41.886Z"),
  "alert_confidence": 0.92,
  "athlete_id": "test_drift_athlete_001",
  "session_id": "695954d54c28ce6abc013a24",
  "sessions_affected": ["695954d54c28ce6abc013a24"],
  "reps_affected": 0,
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.31,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "improving",
      "severity": "severe",
      "coach_follow_up": null,
      "is_monitored": false,
      "monitored_at": null
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.5,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "improving",
      "severity": "severe",
      "coach_follow_up": null,
      "is_monitored": false,
      "monitored_at": null
    },
    "hip_angle": {
      "baseline_value": 131.0,
      "current_value": 100.0,
      "z_score": -37.97,
      "drift_magnitude": 37.97,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": null,
      "is_monitored": false,
      "monitored_at": null
    },
    "acl_max_valgus_angle": {
      "baseline_value": 5.2,
      "current_value": 15.0,
      "z_score": 60.01,
      "drift_magnitude": 60.01,
      "direction": "improving",
      "severity": "severe",
      "coach_follow_up": null,
      "is_monitored": false,
      "monitored_at": null
    }
  },
  "top_clip_ids": [],
  "alert_payload_summary": "Technical drift detected: 4 metrics deviating from baseline",
  "status": "new",
  "created_at": ISODate("2026-01-03T17:41:41.886Z"),
  "updated_at": ISODate("2026-01-03T17:41:41.886Z")
}
```

**Key Fields**:
- `alert_id`: Unique identifier (format: `drift_{athlete_id}_{session_id}_{timestamp}`)
- `alert_type`: Always `"technical_drift"`
- `drift_metrics`: Dictionary of metric_key → drift data
- `status`: `"new"`, `"acknowledged"`, or `"resolved"`
- Each metric in `drift_metrics` has:
  - `baseline_value`: Mean from baseline
  - `current_value`: Current session value
  - `z_score`: Standard deviations from baseline
  - `drift_magnitude`: Absolute z-score
  - `direction`: `"worsening"` or `"improving"`
  - `severity`: `"minor"`, `"moderate"`, or `"severe"`
  - `coach_follow_up`: `null`, `"Monitor"`, `"Escalate to AT/PT"`, etc.
  - `is_monitored`: `true` if coach_follow_up is "Monitor"
  - `monitored_at`: Timestamp when monitoring started

---

## 2. Console Output (Formatted Print)

**Method**: `retrieval_queue_worker.py` → `_print_drift_insights()`

**Output Format**:

```
======================================================================
🔍 DRIFT DETECTION RESULTS
======================================================================
Athlete ID: test_drift_athlete_001
Sessions Analyzed: 1
Total Insights: 4
  - Worsening: 0
  - Improving: 0
  - Unchanged: 0
======================================================================

📊 Insight #1: Insufficient height off floor/beam
   Metric: height_off_floor_meters
   Trend: INSUFFICIENT_DATA
   Overall Severity: SEVERE
   Baseline Value: 0.310
   Change in Deviation: +0.00σ
   Sessions with Drift: 1

   Deviations by Session:
      Session 1: 695954d5...
         Timestamp: 2026-01-03T17:41:41.886000
         Value: 0.150 (baseline: 0.310)
         Z-score: -19.60σ
         Deviation: -51.6%
         Severity: SEVERE
         Direction: IMPROVING
   ------------------------------------------------------------

📊 Insight #2: Insufficient landing knee extension
   Metric: landing_knee_bend_min
   Trend: INSUFFICIENT_DATA
   Overall Severity: SEVERE
   Baseline Value: 160.500
   Change in Deviation: +0.00σ
   Sessions with Drift: 1

   Deviations by Session:
      Session 1: 695954d5...
         Timestamp: 2026-01-03T17:41:41.886000
         Value: 140.000 (baseline: 160.500)
         Z-score: -50.21σ
         Deviation: -12.8%
         Severity: SEVERE
         Direction: IMPROVING
   ------------------------------------------------------------

📊 Insight #3: Not enough hip flexion
   Metric: hip_angle
   Trend: INSUFFICIENT_DATA
   Overall Severity: SEVERE
   Baseline Value: 131.000
   Change in Deviation: +0.00σ
   Sessions with Drift: 1

   Deviations by Session:
      Session 1: 695954d5...
         Timestamp: 2026-01-03T17:41:41.886000
         Value: 100.000 (baseline: 131.000)
         Z-score: -37.97σ
         Deviation: -23.7%
         Severity: SEVERE
         Direction: WORSENING
   ------------------------------------------------------------

📊 Insight #4: Inward collapse of knees (valgus)
   Metric: acl_max_valgus_angle
   Trend: INSUFFICIENT_DATA
   Overall Severity: SEVERE
   Baseline Value: 5.200
   Change in Deviation: +0.00σ
   Sessions with Drift: 1

   Deviations by Session:
      Session 1: 695954d5...
         Timestamp: 2026-01-03T17:41:41.886000
         Value: 15.000 (baseline: 5.200)
         Z-score: +60.01σ
         Deviation: +188.5%
         Severity: SEVERE
         Direction: IMPROVING
   ------------------------------------------------------------

======================================================================
```

**Note**: For multiple sessions, the output shows trend (improving/worsening/unchanged) and change in deviation across sessions.

---

## 3. Redis Queue Message

**Queue**: `drift_alerts_queue`  
**Format** (JSON string):

```json
{
  "alert_type": "technical_drift",
  "alert_id": "drift_test_drift_athlete_001_695954d54c28ce6abc013a24_1767483701",
  "athlete_id": "test_drift_athlete_001",
  "session_id": "695954d54c28ce6abc013a24",
  "severity": "severe",
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.31,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "improving",
      "severity": "severe"
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.5,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "improving",
      "severity": "severe"
    },
    "hip_angle": {
      "baseline_value": 131.0,
      "current_value": 100.0,
      "z_score": -37.97,
      "drift_magnitude": 37.97,
      "direction": "worsening",
      "severity": "severe"
    },
    "acl_max_valgus_angle": {
      "baseline_value": 5.2,
      "current_value": 15.0,
      "z_score": 60.01,
      "drift_magnitude": 60.01,
      "direction": "improving",
      "severity": "severe"
    }
  },
  "drift_count": 4,
  "alert_confidence": 0.92,
  "created_at": "2026-01-03T17:41:41.886000",
  "timestamp": "2026-01-03T17:41:41.886000"
}
```

**Differences from MongoDB**:
- No `_id` field
- No `coach_follow_up`, `is_monitored`, `monitored_at` fields (these are added after coach action)
- `severity` is overall severity (most severe metric)
- `created_at` is ISO string (not ISODate)

---

## 4. WebSocket Broadcast Message

**Endpoint**: `ws://localhost:8766`  
**Event Type**: `drift_alert`  
**Format** (JSON):

```json
{
  "type": "drift_alert",
  "alert_id": "drift_test_drift_athlete_001_695954d54c28ce6abc013a24_1767483701",
  "athlete_id": "test_drift_athlete_001",
  "session_id": "695954d54c28ce6abc013a24",
  "severity": "severe",
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.31,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "improving",
      "severity": "severe"
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.5,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "improving",
      "severity": "severe"
    },
    "hip_angle": {
      "baseline_value": 131.0,
      "current_value": 100.0,
      "z_score": -37.97,
      "drift_magnitude": 37.97,
      "direction": "worsening",
      "severity": "severe"
    },
    "acl_max_valgus_angle": {
      "baseline_value": 5.2,
      "current_value": 15.0,
      "z_score": 60.01,
      "drift_magnitude": 60.01,
      "direction": "improving",
      "severity": "severe"
    }
  },
  "drift_count": 4,
  "alert_confidence": 0.92,
  "timestamp": "2026-01-03T17:41:41.886000"
}
```

**Same format as Redis queue** - WebSocket worker broadcasts the queue message directly.

---

## 5. Multi-Session Drift Result (from `_detect_drift_across_sessions`)

**Format** (returned by `detect_technical_drift()` when `analyze_multiple_sessions=True`):

```json
{
  "athlete_id": "test_drift_athlete_001",
  "session_count": 3,
  "insights": [
    {
      "metric_key": "height_off_floor_meters",
      "insight_description": "Insufficient height off floor/beam",
      "baseline_value": 0.31,
      "baseline_sd": 0.008,
      "deviations": [
        {
          "session_id": "session_1",
          "session_timestamp": "2026-01-03T10:00:00Z",
          "current_value": 0.15,
          "z_score": -19.60,
          "deviation_percent": -51.6,
          "direction": "improving",
          "severity": "severe"
        },
        {
          "session_id": "session_2",
          "session_timestamp": "2026-01-03T11:00:00Z",
          "current_value": 0.18,
          "z_score": -16.25,
          "deviation_percent": -41.9,
          "direction": "improving",
          "severity": "severe"
        },
        {
          "session_id": "session_3",
          "session_timestamp": "2026-01-03T12:00:00Z",
          "current_value": 0.20,
          "z_score": -13.75,
          "deviation_percent": -35.5,
          "direction": "improving",
          "severity": "severe"
        }
      ],
      "trend": "improving",
      "trend_strength": 2.93,
      "overall_severity": "severe",
      "first_deviation": -19.60,
      "latest_deviation": -13.75,
      "change_in_deviation": 5.85,
      "session_count": 3,
      "coach_follow_up": null,
      "is_monitored": false
    }
  ],
  "summary": {
    "total_insights": 1,
    "worsening_insights": 0,
    "improving_insights": 1,
    "unchanged_insights": 0,
    "insufficient_data_insights": 0
  }
}
```

**Key Differences**:
- `insights` array instead of `drift_metrics` dictionary
- Each insight has `deviations` array (one per session)
- `trend` field: `"improving"`, `"worsening"`, `"unchanged"`, or `"insufficient_data"`
- `trend_strength`: Rate of change per session
- `change_in_deviation`: Change in z-score from first to last session

---

## 6. Alert After Coach Follow-Up

**After coach sets follow-up action** (e.g., "Monitor"):

```json
{
  "alert_id": "drift_test_drift_athlete_001_695954d54c28ce6abc013a24_1767483701",
  "alert_type": "technical_drift",
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.31,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "improving",
      "severity": "severe",
      "coach_follow_up": "Monitor",  // ← Updated
      "is_monitored": true,           // ← Updated
      "monitored_at": "2026-01-03T18:00:00Z"  // ← Updated
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.5,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "improving",
      "severity": "severe",
      "coach_follow_up": null,  // ← Not set yet
      "is_monitored": false,
      "monitored_at": null
    }
  }
}
```

---

## Summary

| Format | Location | Use Case |
|--------|----------|----------|
| **MongoDB Document** | `alerts` collection | Persistent storage, includes coach follow-up flags |
| **Console Output** | Terminal logs | Human-readable formatted output for debugging |
| **Redis Queue** | `drift_alerts_queue` | Message queue for async processing |
| **WebSocket Broadcast** | `ws://localhost:8766` | Real-time delivery to PT/Instructor clients |
| **Multi-Session Result** | Return value from `detect_technical_drift()` | Trend analysis across multiple sessions |

**All formats contain**:
- Alert ID
- Athlete ID
- Session ID(s)
- Drift metrics with z-scores, severity, direction
- Timestamps

**MongoDB format is the source of truth** - all other formats are derived from it.

