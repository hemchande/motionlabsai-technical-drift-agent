# Drift Detection Analysis - Current vs. Enhanced

## Current Implementation

### What's Included

Currently, drift detection:
1. **Checks a single session** against baseline
2. **Calculates z-scores** for each metric
3. **Identifies metrics** that exceed threshold (default: 2.0σ)
4. **Creates an alert** with drift metrics

### Formula

For each metric:
```
z_score = (current_value - baseline_mean) / baseline_sd

If |z_score| > drift_threshold (2.0σ):
    - Calculate severity (minor/moderate/severe)
    - Determine direction (worsening/improving)
    - Add to drift_metrics
```

### Current Output

```python
{
    "athlete_id": "athlete_001",
    "session_id": "session_123",
    "drift_metrics": {
        "height_off_floor_meters": {
            "baseline_value": 0.30,
            "current_value": 0.15,
            "z_score": -19.60,
            "drift_magnitude": 19.60,
            "direction": "worsening",
            "severity": "severe"
        }
    },
    "drift_count": 1
}
```

## What's Missing

The user wants:
1. **Multiple insights** with deviations
2. **Deviations across sessions** (not just one session)
3. **Trend analysis** showing how metrics change over time

## Enhanced Implementation Needed

### New Approach

1. **Analyze multiple sessions** after baseline
2. **Track each insight/metric** across sessions
3. **Calculate deviations** for each session
4. **Show trend** (improving/worsening/unchanged)
5. **Aggregate insights** with session-by-session deviations

### Enhanced Output Structure

```python
{
    "athlete_id": "athlete_001",
    "baseline_id": "baseline_123",
    "sessions_analyzed": ["session_1", "session_2", "session_3"],
    "insights": [
        {
            "metric_key": "height_off_floor_meters",
            "insight_description": "Insufficient height off floor/beam",
            "baseline_value": 0.30,
            "baseline_sd": 0.008,
            "deviations": [
                {
                    "session_id": "session_1",
                    "session_timestamp": "2026-01-01T10:00:00Z",
                    "current_value": 0.25,
                    "z_score": -6.25,
                    "deviation_percent": -16.7,
                    "direction": "worsening",
                    "severity": "moderate"
                },
                {
                    "session_id": "session_2",
                    "session_timestamp": "2026-01-02T10:00:00Z",
                    "current_value": 0.20,
                    "z_score": -12.50,
                    "deviation_percent": -33.3,
                    "direction": "worsening",
                    "severity": "severe"
                },
                {
                    "session_id": "session_3",
                    "session_timestamp": "2026-01-03T10:00:00Z",
                    "current_value": 0.15,
                    "z_score": -18.75,
                    "deviation_percent": -50.0,
                    "direction": "worsening",
                    "severity": "severe"
                }
            ],
            "trend": "worsening",
            "trend_strength": 0.05,  # Rate of change per session
            "overall_severity": "severe",
            "first_deviation": -6.25,
            "latest_deviation": -18.75,
            "change_in_deviation": -12.50
        },
        {
            "metric_key": "landing_knee_bend_min",
            "insight_description": "Insufficient landing knee extension",
            "baseline_value": 160.0,
            "baseline_sd": 0.5,
            "deviations": [...],
            "trend": "improving",
            ...
        }
    ],
    "summary": {
        "total_insights": 2,
        "worsening_insights": 1,
        "improving_insights": 1,
        "unchanged_insights": 0,
        "sessions_with_drift": 3,
        "most_severe_insight": "height_off_floor_meters"
    }
}
```

