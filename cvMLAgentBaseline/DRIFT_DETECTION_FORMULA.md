# Drift Detection Formula & Implementation

## Current Implementation

### What's Included in Drift Detection

1. **Baseline Comparison**: Compares current session metrics against baseline
2. **Z-Score Calculation**: Statistical measure of deviation
3. **Multiple Insights**: Tracks all metrics that exceed threshold
4. **Severity Classification**: Minor/Moderate/Severe
5. **Direction Detection**: Worsening/Improving

### Formula

For each metric in a session:

```
z_score = (current_value - baseline_mean) / baseline_sd

If |z_score| > drift_threshold (default: 2.0σ):
    - Calculate severity based on |z_score|
    - Determine direction (worsening/improving)
    - Add to drift_metrics
```

**Severity Classification**:
- **Minor**: 2.0σ ≤ |z-score| < 2.5σ
- **Moderate**: 2.5σ ≤ |z-score| < 3.0σ
- **Severe**: |z-score| ≥ 3.0σ

**Direction**:
- For metrics where **higher is better** (e.g., height, knee extension):
  - Positive z-score → Worsening (below baseline)
  - Negative z-score → Improving (above baseline)
- For metrics where **lower is better** (e.g., valgus angle):
  - Positive z-score → Worsening (above baseline)
  - Negative z-score → Improving (below baseline)

---

## Enhanced Implementation: Multiple Sessions Analysis

### New Feature: Analyze Across Sessions

When `analyze_multiple_sessions=True` and `session_id=None`, the system:

1. **Finds all sessions** after baseline end date
2. **Tracks each insight/metric** across those sessions
3. **Calculates deviations** for each session
4. **Shows trend** (improving/worsening/unchanged)

### Enhanced Formula

For each metric across multiple sessions:

```
For each session after baseline:
    z_score = (session_value - baseline_mean) / baseline_sd
    
    If |z_score| > drift_threshold:
        deviation = {
            session_id: session_id,
            current_value: session_value,
            z_score: z_score,
            deviation_percent: ((session_value - baseline_mean) / baseline_mean) * 100,
            direction: worsening/improving,
            severity: minor/moderate/severe
        }
        Add to insight.deviations[]

For each insight:
    trend = Calculate from first_z_score to last_z_score
    trend_strength = |slope| of z-score progression
    change_in_deviation = last_z_score - first_z_score
```

### Trend Calculation

```
first_z = deviations[0].z_score
last_z = deviations[-1].z_score

if |last_z| < |first_z|:
    trend = "improving"  # Getting closer to baseline
elif |last_z| > |first_z|:
    trend = "worsening"  # Getting further from baseline
else:
    trend = "unchanged"

# Trend strength (rate of change per session)
slope = linear_regression(z_scores)
trend_strength = |slope|
```

---

## Output Structure

### Single Session (Current)

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

### Multiple Sessions (Enhanced)

```python
{
    "athlete_id": "athlete_001",
    "baseline_id": "baseline_123",
    "baseline_end_date": "2026-01-01T10:00:00Z",
    "sessions_analyzed": ["session_1", "session_2", "session_3"],
    "session_count": 3,
    "insights": [
        {
            "metric_key": "height_off_floor_meters",
            "insight_description": "Insufficient height off floor/beam",
            "baseline_value": 0.30,
            "baseline_sd": 0.008,
            "deviations": [
                {
                    "session_id": "session_1",
                    "session_timestamp": "2026-01-02T10:00:00Z",
                    "current_value": 0.25,
                    "z_score": -6.25,
                    "deviation_percent": -16.7,
                    "direction": "worsening",
                    "severity": "moderate"
                },
                {
                    "session_id": "session_2",
                    "session_timestamp": "2026-01-03T10:00:00Z",
                    "current_value": 0.20,
                    "z_score": -12.50,
                    "deviation_percent": -33.3,
                    "direction": "worsening",
                    "severity": "severe"
                },
                {
                    "session_id": "session_3",
                    "session_timestamp": "2026-01-04T10:00:00Z",
                    "current_value": 0.15,
                    "z_score": -18.75,
                    "deviation_percent": -50.0,
                    "direction": "worsening",
                    "severity": "severe"
                }
            ],
            "trend": "worsening",
            "trend_strength": 6.25,  # Rate of change per session
            "overall_severity": "severe",
            "first_deviation": -6.25,
            "latest_deviation": -18.75,
            "change_in_deviation": -12.50,
            "session_count": 3
        }
    ],
    "summary": {
        "total_insights": 1,
        "worsening_insights": 1,
        "improving_insights": 0,
        "unchanged_insights": 0,
        "sessions_with_drift": 3,
        "most_severe_insight": "height_off_floor_meters"
    }
}
```

---

## Usage

### Single Session Analysis

```python
drift = agent.detect_technical_drift(
    athlete_id="athlete_001",
    session_id="session_123",
    drift_threshold=2.0
)
```

### Multiple Sessions Analysis

```python
drift = agent.detect_technical_drift(
    athlete_id="athlete_001",
    session_id=None,  # None triggers multi-session analysis
    analyze_multiple_sessions=True,
    max_sessions=10,
    drift_threshold=2.0
)
```

---

## Key Features

1. **Multiple Insights**: Each metric that exceeds threshold becomes an insight
2. **Session-by-Session Deviations**: Shows how each insight changes across sessions
3. **Trend Analysis**: Calculates if insights are improving, worsening, or unchanged
4. **Trend Strength**: Quantifies rate of change
5. **Summary Statistics**: Aggregates insights by trend direction

