# Insights vs Trends - Key Differences

## Quick Summary

| Aspect | Insights | Trends |
|--------|----------|--------|
| **What** | Form issues found in individual sessions | How issues change across multiple sessions |
| **Scope** | Per-session | Cross-session analysis |
| **Calculation** | Compare metrics to thresholds | Analyze metric changes over time |
| **Output** | List of issues (e.g., "Insufficient height") | Trend direction (improving/worsening/unchanged) |
| **Created** | Step 1 (every session) | Step 2 (when 3+ sessions) |
| **Collection** | `insights` | `trends` |
| **Requires Baseline** | ❌ No | ⚠️ Optional |

---

## Detailed Comparison

### 1. **Insights** - What Are They?

**Definition**: Form issues extracted from a single session's metrics

**Purpose**: Identify problems in movement form (e.g., insufficient height, knee valgus)

**Calculation**:
```python
# For each session:
for issue_type in FORM_ISSUES:
    metric_value = session.metrics[metric_key]
    threshold = FORM_ISSUES[issue_type]["threshold"]
    
    if metric_value < threshold:  # or > threshold
        # Issue found!
        insights.append({
            "description": "Insufficient height off floor/beam",
            "metric_value": 0.15,
            "threshold": 0.30,
            "severity": "severe"
        })
```

**Example**:
```
Session 1:
  - Height: 0.15m (threshold: 0.30m) → "Insufficient height"
  - Knee bend: 140° (threshold: 150°) → "Insufficient landing knee extension"

Session 2:
  - Height: 0.18m (threshold: 0.30m) → "Insufficient height"
  - Knee bend: 145° (threshold: 150°) → "Insufficient landing knee extension"
```

**Key Characteristics**:
- ✅ **Per-session**: One insight document per session
- ✅ **Threshold-based**: Compares metric to fixed threshold
- ✅ **Immediate**: Identifies issues right away
- ✅ **No baseline needed**: Works independently
- ✅ **Binary**: Issue exists or doesn't exist

**Schema**:
```json
{
  "session_id": "session_123",
  "insights": [
    {
      "insight": "Insufficient height off floor/beam",
      "is_monitored": false,
      "coach_follow_up": null
    }
  ],
  "timestamp": "2026-01-03T10:00:00Z"
}
```

---

### 2. **Trends** - What Are They?

**Definition**: Analysis of how form issues change across multiple sessions

**Purpose**: Show if issues are improving, worsening, or staying the same over time

**Calculation**:
```python
# Across multiple sessions:
sessions = [session1, session2, session3, ...]

# Extract metric values for same issue
heights = [0.15, 0.18, 0.22, 0.25, 0.28]  # Over 5 sessions

# Calculate trend
first_half_mean = mean([0.15, 0.18, 0.22]) = 0.183
second_half_mean = mean([0.22, 0.25, 0.28]) = 0.250

if second_half_mean > first_half_mean * 1.1:
    trend = "improving"  # Getting better!
elif second_half_mean < first_half_mean * 0.9:
    trend = "worsening"  # Getting worse
else:
    trend = "unchanged"
```

**Example**:
```
Sessions 1-5: Height values [0.15, 0.18, 0.22, 0.25, 0.28]
  → Trend: "improving" (height increasing over time)
  → Observation: "Height increased from 0.15m to 0.28m across 5 sessions"
  → Coaching: "Consider maintaining current training approach"

Sessions 6-10: Height values [0.28, 0.25, 0.22, 0.20, 0.18]
  → Trend: "worsening" (height decreasing over time)
  → Observation: "Height decreased from 0.28m to 0.18m across 5 sessions"
  → Coaching: "Consider reducing high-impact volume"
```

**Key Characteristics**:
- ✅ **Cross-session**: Requires 3+ sessions
- ✅ **Time-based**: Analyzes changes over time
- ✅ **Directional**: Shows improving/worsening/unchanged
- ⚠️ **Optional baseline**: Can reference baseline if exists
- ✅ **Three-layer output**: Observation, Evidence, Coaching Options

**Schema**:
```json
{
  "trend_id": "athlete_001_insufficient_height_back_handspring",
  "issue_type": "insufficient_height",
  "trend_status": "improving",
  "observation": "Height increased from 0.15m to 0.28m across 5 sessions",
  "evidence_reasoning": "Non-clinical explanation...",
  "coaching_options": [
    "Consider maintaining current training approach",
    "If pain/symptoms are present, consult your AT/PT."
  ],
  "sessions_analyzed": 5
}
```

---

## Side-by-Side Comparison

### Calculation Method

**Insights**:
```
Single Session → Compare to Threshold → Issue Found?
```

**Trends**:
```
Multiple Sessions → Calculate Changes → Trend Direction?
```

---

### Data Source

**Insights**:
- Uses: `sessions.metrics` (single session)
- Compares: Metric value vs fixed threshold
- Example: `height_off_floor_meters = 0.15` vs `threshold = 0.30`

**Trends**:
- Uses: `sessions.metrics` (multiple sessions)
- Compares: Metric values across sessions
- Example: `[0.15, 0.18, 0.22, 0.25, 0.28]` → increasing trend

---

### Output Format

**Insights**:
```json
{
  "insights": [
    "Insufficient height off floor/beam",
    "Insufficient landing knee extension"
  ]
}
```
- Simple list of issues
- No direction or change information

**Trends**:
```json
{
  "observation": "Height increased from 0.15m to 0.28m across 5 sessions",
  "evidence_reasoning": "This pattern often appears with...",
  "coaching_options": ["Consider maintaining...", "If pain..."],
  "trend_status": "improving"
}
```
- Three-layer structured output
- Includes direction and coaching recommendations

---

### When Created

**Insights**:
- ✅ **Every session** with form issues
- ✅ **Step 1** of pipeline
- ✅ **Before** baseline establishment
- ✅ **Before** drift detection

**Trends**:
- ⚠️ **Only when 3+ sessions** exist
- ✅ **Step 2** of pipeline
- ⚠️ **After** insights are created
- ⚠️ **Can reference** baseline (optional)

---

### Dependencies

**Insights**:
- ✅ **No dependencies**: Works with just session metrics
- ✅ **No baseline needed**: Independent analysis
- ✅ **Immediate**: Available right after session

**Trends**:
- ⚠️ **Requires 3+ sessions**: Needs historical data
- ⚠️ **Optional baseline**: Can enhance with baseline comparison
- ⚠️ **Delayed**: Only available after multiple sessions

---

## Real-World Example

### Scenario: Athlete with Height Issues

**Session 1** (Jan 1):
```
Insights: ["Insufficient height off floor/beam"]
  - Height: 0.15m (threshold: 0.30m)
  - Issue identified immediately

Trends: None (only 1 session)
```

**Session 2** (Jan 3):
```
Insights: ["Insufficient height off floor/beam"]
  - Height: 0.18m (threshold: 0.30m)
  - Issue still present

Trends: None (only 2 sessions, need 3+)
```

**Session 3** (Jan 5):
```
Insights: ["Insufficient height off floor/beam"]
  - Height: 0.22m (threshold: 0.30m)
  - Issue still present

Trends: Created!
  - Trend: "improving"
  - Observation: "Height increased from 0.15m to 0.22m across 3 sessions"
  - Coaching: "Consider maintaining current training approach"
```

**Session 4** (Jan 7):
```
Insights: ["Insufficient height off floor/beam"]
  - Height: 0.25m (threshold: 0.30m)
  - Issue still present (but getting better!)

Trends: Updated
  - Trend: "improving" (confirmed)
  - Observation: "Height increased from 0.15m to 0.25m across 4 sessions"
```

---

## Use Cases

### When to Use Insights

✅ **Immediate feedback**: "This session has insufficient height"  
✅ **Per-session analysis**: Identify issues right away  
✅ **Coaching decisions**: "Fix this in next session"  
✅ **No history needed**: Works from first session

### When to Use Trends

✅ **Long-term tracking**: "Is the athlete improving?"  
✅ **Training effectiveness**: "Did coaching changes work?"  
✅ **Pattern recognition**: "This issue appears in 80% of sessions"  
✅ **Requires history**: Needs 3+ sessions

---

## Relationship Between Insights and Trends

```
Insights (Step 1)
    ↓
    Identifies which issues to track
    ↓
Trends (Step 2)
    ↓
    Analyzes how those issues change over time
```

**Flow**:
1. **Insights** identify: "Insufficient height" in Session 1, 2, 3
2. **Trends** analyze: "Height is improving across Sessions 1-3"

**They work together**:
- Insights provide the "what" (what issues exist)
- Trends provide the "how" (how issues are changing)

---

## Summary

| Question | Insights | Trends |
|----------|----------|--------|
| **What do they identify?** | Form issues in a session | How issues change over time |
| **When are they created?** | Every session | After 3+ sessions |
| **What do they compare?** | Metric vs threshold | Metrics across sessions |
| **What's the output?** | List of issues | Trend direction + coaching |
| **Do they need baseline?** | No | Optional |
| **Are they per-session?** | Yes | No (cross-session) |

**Key Takeaway**: 
- **Insights** = "What's wrong in this session?"
- **Trends** = "Is it getting better or worse over time?"

Both are important:
- **Insights** for immediate coaching feedback
- **Trends** for long-term progress tracking

