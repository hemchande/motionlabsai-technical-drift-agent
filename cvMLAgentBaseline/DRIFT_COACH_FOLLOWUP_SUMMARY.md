# Drift Coach Follow-Up & Monitoring System

## ✅ Implementation Complete

**Date**: 2026-01-03  
**Status**: Fully functional

---

## Overview

After technical drift is detected for a specific insight/metric, coaches can:
1. **Monitor**: Track how the insight is worsening or improving over time
2. **Escalate**: Escalate to AT/PT for immediate attention

If "Monitor" is selected, the system automatically tracks the trend (worsening/improving/unchanged) across subsequent sessions.

---

## Features

### 1. Coach Follow-Up for Drift Alerts

Each drift metric in an alert can have a coach follow-up action:
- **Monitor**: Track the metric over time
- **Escalate to AT/PT**: Immediate escalation

### 2. Monitoring Tracking

When a drift metric is set to "Monitor":
- System tracks the metric across new sessions
- Calculates trend: **worsening**, **improving**, or **unchanged**
- Stores trend data in `monitoring_trends` collection
- Provides trend strength and change percentage

---

## API Methods

### `update_drift_alert_coach_follow_up()`

Update coach follow-up for a specific drift metric in an alert.

```python
agent.update_drift_alert_coach_follow_up(
    alert_id="drift_athlete_001_session_123_1234567890",
    metric_key="height_off_floor_meters",
    coach_follow_up="Monitor"  # or "Escalate to AT/PT"
)
```

**Returns**: `True` if successful, `False` otherwise

---

### `track_monitored_drift_insights()`

Track how a monitored drift insight is worsening or improving over time.

```python
trend = agent.track_monitored_drift_insights(
    athlete_id="athlete_001",
    metric_key="height_off_floor_meters",
    monitored_since=None  # Optional: timestamp when monitoring started
)
```

**Returns**: Dictionary with trend analysis:
```python
{
    "metric_key": "height_off_floor_meters",
    "athlete_id": "athlete_001",
    "trend": "improving",  # or "worsening", "unchanged", "insufficient_data"
    "trend_strength": 2.449,
    "change_percent": 26.7,
    "first_value": 0.15,
    "last_value": 0.19,
    "first_z_score": -19.60,
    "last_z_score": -12.50,
    "sessions_analyzed": 3,
    "monitored_since": "2026-01-03T17:46:34.020000",
    "message": "Trend: improving (26.7% change over 3 sessions)"
}
```

---

## Data Structures

### Alert Document (Updated)

```json
{
  "_id": ObjectId,
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "alert_type": "technical_drift",
  "athlete_id": "athlete_001",
  "session_id": "session_123",
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.30,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": "Monitor",  // NEW
      "is_monitored": true,          // NEW
      "monitored_at": "2026-01-03T17:46:34.020000"  // NEW
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.0,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": "Escalate to AT/PT",  // NEW
      "is_monitored": false,                    // NEW
      "monitored_at": null                      // NEW
    }
  },
  "status": "new",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

### Monitoring Trends Collection

```json
{
  "_id": ObjectId,
  "metric_key": "height_off_floor_meters",
  "athlete_id": "athlete_001",
  "trend": "improving",  // or "worsening", "unchanged", "insufficient_data"
  "trend_strength": 2.449,
  "change_percent": 26.7,
  "first_value": 0.15,
  "last_value": 0.19,
  "first_z_score": -19.60,
  "last_z_score": -12.50,
  "baseline_mean": 0.30,
  "baseline_sd": 0.008,
  "sessions_analyzed": 3,
  "metric_values": [0.15, 0.17, 0.19],
  "session_timestamps": ["2026-01-03T...", "2026-01-04T...", "2026-01-05T..."],
  "monitored_since": "2026-01-03T17:46:34.020000",
  "analyzed_at": "2026-01-05T10:00:00.000000",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

---

## Trend Calculation

### Trend Determination

1. **Extract metric values** from all sessions after `monitored_since`
2. **Calculate z-scores** for first and last values
3. **Compare z-scores**:
   - **Improving**: Getting closer to baseline (|z-score| decreasing)
   - **Worsening**: Getting further from baseline (|z-score| increasing)
   - **Unchanged**: No significant change

### Metric Direction

- **Higher is better** (e.g., `height_off_floor_meters`, `landing_knee_bend_min`, `hip_angle`):
  - Improving: z-score increasing (toward positive)
  - Worsening: z-score decreasing (toward negative)

- **Lower is better** (e.g., `acl_max_valgus_angle`):
  - Improving: z-score decreasing (toward negative)
  - Worsening: z-score increasing (toward positive)

### Trend Strength

Calculated using linear regression slope normalized by baseline standard deviation:
```
trend_strength = |slope| / baseline_sd
```

---

## Test Results

### ✅ All Tests Passed

**Test 1: Coach Follow-Up Update**
- ✅ Update "Monitor" for drift metric
- ✅ Verify `coach_follow_up`, `is_monitored`, `monitored_at` set correctly

**Test 2: Monitoring Tracking**
- ✅ Track trend across 3 sessions
- ✅ Calculate trend (improving/worsening/unchanged)
- ✅ Save to `monitoring_trends` collection
- ✅ Provide trend strength and change percentage

**Test 3: Escalate Follow-Up**
- ✅ Update "Escalate to AT/PT" for drift metric
- ✅ Verify `is_monitored` set to `false`

---

## Usage Example

```python
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

agent = FormCorrectionRetrievalAgent()

# 1. Detect drift (creates alert)
drift = agent.detect_technical_drift(
    athlete_id="athlete_001",
    session_id="session_123",
    drift_threshold=2.0
)

alert_id = drift.get("alert_id")

# 2. Coach sets follow-up to Monitor
agent.update_drift_alert_coach_follow_up(
    alert_id=alert_id,
    metric_key="height_off_floor_meters",
    coach_follow_up="Monitor"
)

# 3. After new sessions, track trend
trend = agent.track_monitored_drift_insights(
    athlete_id="athlete_001",
    metric_key="height_off_floor_meters"
)

print(f"Trend: {trend.get('trend')}")  # "improving", "worsening", or "unchanged"
print(f"Change: {trend.get('change_percent'):.1f}%")
```

---

## Integration Points

### WebSocket Server (Future)

Can extend `websocket_insights_server.py` to handle drift alert follow-ups:

```json
{
  "type": "drift_alert_followup",
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "metric_key": "height_off_floor_meters",
  "coach_follow_up": "Monitor"
}
```

### Queue Worker (Future)

Can integrate into `retrieval_queue_worker.py` to:
1. Check for monitored drift metrics after each session
2. Automatically track trends for monitored metrics
3. Generate alerts if trend worsens significantly

---

## Collections Used

1. **`alerts`**: Stores drift alerts with coach follow-up
2. **`monitoring_trends`**: Stores trend analysis for monitored metrics
3. **`sessions`**: Source of metric data for trend calculation
4. **`baselines`**: Baseline values for z-score calculation

---

## Next Steps

1. ✅ **Coach Follow-Up**: Complete
2. ✅ **Monitoring Tracking**: Complete
3. 🔄 **WebSocket Integration**: Ready for implementation
4. 🔄 **Queue Worker Integration**: Ready for implementation
5. 🔄 **Dashboard/UI**: Display trends and follow-up status

---

## Test File

- `test_drift_coach_followup.py` - Comprehensive test suite

**Run Test**:
```bash
cd cvMLAgentBaseline
python3 test_drift_coach_followup.py
```

