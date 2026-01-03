# Drift Detection Logic & Collection Structures - Deep Dive

## Table of Contents
1. [Complete Data Flow](#complete-data-flow)
2. [Drift Detection Logic](#drift-detection-logic)
3. [Collection Structures](#collection-structures)
4. [Insight Calculation Flow](#insight-calculation-flow)
5. [Formulas & Calculations](#formulas--calculations)

---

## Complete Data Flow

### High-Level Flow

```
Video Agent → Sessions Collection
    ↓
Retrieval Agent → Extract Form Issues → Insights Collection
    ↓
Baseline Establishment → Baselines Collection + Drift Detection Flags
    ↓
Drift Detection → Alerts Collection
    ↓
Coach Follow-Up → Update Alerts/Insights
    ↓
Monitoring → Monitoring Trends Collection
```

---

## Drift Detection Logic

### Step-by-Step Process

#### 1. **Prerequisites Check**

```python
# Step 1.1: Get Active Baseline
baseline = _get_active_baseline(athlete_id)
# Returns: Baseline document with baseline_vector, baseline_window, etc.

# Step 1.2: Get Drift Detection Flag
drift_flag = _get_drift_detection_flag(athlete_id)
# Returns: Flag document with drift_detection_enabled, drift_detection_start_date, drift_threshold

# Step 1.3: Validate Flag Status
if not drift_flag or not drift_flag.get("drift_detection_enabled"):
    return None  # Drift detection not enabled

# Step 1.4: Check Start Date
start_date = drift_flag.get("drift_detection_start_date")
if datetime.utcnow() < start_date:
    return None  # Not yet time to start monitoring
```

**Logic**: Drift detection only runs if:
- Baseline exists for athlete
- Drift detection is enabled
- Current date >= start date

---

#### 2. **Session Selection**

**Single Session Mode**:
```python
if session_id is provided:
    # Get specific session
    session = collection.find_one({"_id": ObjectId(session_id)})
    metrics = session.get("metrics", {})
```

**Multiple Sessions Mode** (Default):
```python
if session_id is None and analyze_multiple_sessions is True:
    # Get all sessions after baseline end date
    baseline_end = baseline.get("baseline_window", {}).get("end_date")
    query = {
        "athlete_id": athlete_id,
        "timestamp": {"$gte": baseline_end.isoformat()}
    }
    sessions = collection.find(query).sort("timestamp", 1).limit(max_sessions)
```

**Logic**: 
- Single session: Check one session against baseline
- Multiple sessions: Analyze all sessions after baseline to show trends

---

#### 3. **Metric Extraction**

```python
# Extract all metrics recursively from session
temp_metrics = defaultdict(list)
_extract_metrics_recursive(metrics, temp_metrics)

# Flatten to single values
session_metrics = {}
for key, values in temp_metrics.items():
    if values:
        # Use mean if multiple values
        session_metrics[key] = values[0] if len(values) == 1 else sum(values) / len(values)
```

**Logic**: 
- Recursively extracts all numeric metrics from nested structures
- Handles arrays, nested objects, and flat structures
- Flattens multiple values to single value (mean if multiple)

---

#### 4. **Z-Score Calculation**

For each metric in session:

```python
# Get baseline statistics
baseline_stats = baseline_vector[metric_key]
baseline_mean = baseline_stats.get("mean", 0)
baseline_sd = baseline_stats.get("sd", 0)

# Skip if no variance
if baseline_sd == 0:
    continue

# Calculate z-score
z_score = (current_value - baseline_mean) / baseline_sd
```

**Formula**:
```
z_score = (current_value - baseline_mean) / baseline_sd
```

**Interpretation**:
- `z_score = 0`: Exactly at baseline mean
- `z_score = +2.0`: 2 standard deviations above baseline
- `z_score = -2.0`: 2 standard deviations below baseline

---

#### 5. **Drift Threshold Check**

```python
if abs(z_score) > drift_threshold:  # Default: 2.0σ
    # Drift detected - process this metric
```

**Logic**: Only metrics exceeding threshold are considered "drift"

---

#### 6. **Severity Classification**

```python
if abs(z_score) > 3.0:
    severity = "severe"
elif abs(z_score) > 2.5:
    severity = "moderate"
else:
    severity = "minor"  # 2.0 ≤ |z| < 2.5
```

**Classification**:
- **Minor**: 2.0σ ≤ |z-score| < 2.5σ
- **Moderate**: 2.5σ ≤ |z-score| < 3.0σ
- **Severe**: |z-score| ≥ 3.0σ

---

#### 7. **Direction Determination**

```python
# Determine if metric is "higher is better" or "lower is better"
higher_is_better = metric_key in [
    "height_off_floor_meters",
    "landing_knee_bend_min",
    "hip_angle"
]

if z_score > 0:
    direction = "worsening" if higher_is_better else "improving"
else:
    direction = "improving" if higher_is_better else "worsening"
```

**Logic**:
- For "higher is better" metrics (e.g., height):
  - Positive z-score → Below baseline → Worsening
  - Negative z-score → Above baseline → Improving
- For "lower is better" metrics (e.g., valgus angle):
  - Positive z-score → Above baseline → Worsening
  - Negative z-score → Below baseline → Improving

---

#### 8. **Trend Calculation** (Multiple Sessions Only)

```python
# Compare first and last z-scores
first_z = deviations[0]["z_score"]
last_z = deviations[-1]["z_score"]

if abs(last_z) < abs(first_z):
    trend = "improving"  # Getting closer to baseline
elif abs(last_z) > abs(first_z):
    trend = "worsening"  # Getting further from baseline
else:
    trend = "unchanged"

# Calculate trend strength (rate of change per session)
z_scores = [d["z_score"] for d in deviations]
slope = np.polyfit(x, y, 1)[0]  # Linear regression
trend_strength = abs(slope)
```

**Formula**:
```
trend = "improving" if |last_z| < |first_z| else "worsening"
trend_strength = |slope| from linear regression of z-scores
```

---

## Collection Structures

### 1. `sessions` Collection

**Purpose**: Stores raw session data with metrics

**Schema**:
```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "string",  // REQUIRED for drift detection
  "athlete_name": "string",
  "activity": "gymnastics",
  "technique": "back_handspring",
  "timestamp": "2026-01-03T10:00:00Z",
  
  // Metrics (nested structure)
  "metrics": {
    "height_off_floor_meters": 0.25,
    "landing_knee_bend_min": 155.0,
    "hip_angle": 125.0,
    "acl_max_valgus_angle": 8.0,
    // ... can be nested
    "nested": {
      "landing_knee_bend": 155.0
    }
  },
  
  // Quality control
  "capture_confidence_score": 0.85,  // REQUIRED for baseline (>= 0.7)
  "baseline_eligible": true,         // REQUIRED for baseline
  
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields for Drift Detection**:
- `athlete_id`: Links session to athlete
- `metrics`: Source of all metric values
- `timestamp`: Used to filter sessions after baseline
- `capture_confidence_score`: Quality threshold (>= 0.7 for baseline)
- `baseline_eligible`: Marks if session qualifies for baseline

**Indexes**:
- `{ athlete_id: 1, timestamp: 1 }` - Find sessions by athlete and date
- `{ athlete_id: 1, capture_confidence_score: 1 }` - Find eligible sessions

---

### 2. `baselines` Collection

**Purpose**: Stores calculated baseline vectors for athletes

**Schema**:
```json
{
  "_id": ObjectId,
  "athlete_id": "string",
  "baseline_type": "pre_injury" | "pre_rehab" | "post_rehab",
  
  // Baseline window (which sessions were used)
  "baseline_window": {
    "start_date": "2025-12-20T10:00:00Z",
    "end_date": "2026-01-01T10:00:00Z",
    "session_count": 8,
    "session_ids": ["session_1", "session_2", ...]
  },
  
  // Baseline vector (statistics for each metric)
  "baseline_vector": {
    "height_off_floor_meters": {
      "mean": 0.30,
      "sd": 0.008,
      "min": 0.28,
      "max": 0.32,
      "percentile_rank": 65.5
    },
    "landing_knee_bend_min": {
      "mean": 160.0,
      "sd": 0.5,
      "min": 158.0,
      "max": 162.0,
      "percentile_rank": 72.3
    }
    // ... all metrics from baseline sessions
  },
  
  // Signature (deterministic hash)
  "signature_id": "eabac9f8faac6a80...",
  
  // Quality scores
  "capture_quality_scores": [0.85, 0.92, 0.88, ...],
  
  "established_at": ISODate,
  "status": "active" | "superseded" | "invalidated",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `baseline_vector`: Mean, SD, min, max for each metric (used for z-score calculation)
- `baseline_window.end_date`: Used to find sessions after baseline
- `status`: Only "active" baselines are used for drift detection

**Calculation**:
```python
# For each metric across baseline sessions:
mean = sum(values) / len(values)
sd = sqrt(sum((x - mean)^2) / len(values))
min = min(values)
max = max(values)
```

**Indexes**:
- `{ athlete_id: 1, status: 1 }` - Find active baseline
- `{ signature_id: 1 }` - Unique signature lookup

---

### 3. `drift_detection_flags` Collection

**Purpose**: Controls when and how drift detection runs

**Schema**:
```json
{
  "_id": ObjectId,
  "athlete_id": "string",  // ONE flag per athlete
  "baseline_id": ObjectId,  // Link to baseline
  
  // Control flags
  "drift_detection_enabled": true,
  "drift_detection_start_date": "2026-01-02T10:00:00Z",  // When to start monitoring
  "drift_threshold": 2.0,  // Global threshold (sigma)
  "alert_on_drift": true,
  
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `drift_detection_enabled`: Master on/off switch
- `drift_detection_start_date`: Prevents drift detection before this date
- `drift_threshold`: Global threshold for all metrics (default: 2.0σ)

**Logic**:
- Created automatically when baseline is established
- One flag per athlete (not per insight)
- Controls global drift detection behavior

**Indexes**:
- `{ athlete_id: 1 }` - Find flag for athlete
- `{ drift_detection_enabled: 1, drift_detection_start_date: 1 }` - Find active flags

---

### 4. `insights` Collection

**Purpose**: Stores extracted form issues/insights per session

**Schema**:
```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "string",  // For filtering
  
  // List of insights (can be strings or objects)
  "insights": [
    {
      "description": "Insufficient height off floor/beam",
      "is_monitored": false,
      "coach_follow_up": null,
      "monitored_at": null
    },
    {
      "description": "Insufficient landing knee extension",
      "is_monitored": true,
      "coach_follow_up": "Monitor",
      "monitored_at": "2026-01-03T10:00:00Z"
    }
  ],
  
  "insight_count": 2,
  "activity": "gymnastics",
  "technique": "back_handspring",
  "athlete_name": "Test Athlete",
  "timestamp": "2026-01-03T10:00:00Z",
  "form_issue_count": 2,
  "form_issue_types": ["insufficient_height", "landing_knee_bend"],
  
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `insights`: Array of insight objects with coach follow-up flags
- `session_id`: Links to session
- `athlete_id`: For filtering by athlete

**Insight Extraction Logic**:
1. Check each `FORM_ISSUES` definition
2. Find metric value in session
3. Compare against threshold
4. If exceeds threshold → Add to insights

**Indexes**:
- `{ session_id: 1 }` - Find insights for session
- `{ athlete_id: 1 }` - Find insights for athlete
- `{ "insights.is_monitored": 1 }` - Find monitored insights

---

### 5. `alerts` Collection

**Purpose**: Stores drift alerts with per-insight coach follow-up

**Schema**:
```json
{
  "_id": ObjectId,
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "alert_type": "technical_drift",
  "athlete_id": "athlete_001",
  "session_id": "session_123",
  "baseline_id": ObjectId,
  
  // Per-insight drift metrics
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.30,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": "Monitor",  // Per-insight
      "is_monitored": true,          // Per-insight
      "monitored_at": "2026-01-03T10:00:00Z"  // Per-insight
    },
    "landing_knee_bend_min": {
      "baseline_value": 160.0,
      "current_value": 140.0,
      "z_score": -50.21,
      "drift_magnitude": 50.21,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": "Escalate to AT/PT",  // Different per insight
      "is_monitored": false,
      "monitored_at": null
    }
  },
  
  "sessions_affected": ["session_123"],
  "alert_confidence": 0.92,
  "status": "new" | "acknowledged" | "resolved",
  "alert_created_at": ISODate,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `drift_metrics`: Dictionary of metric_key → drift data
- Each metric has its own `coach_follow_up` and `is_monitored` flags
- `status`: Alert lifecycle state

**Indexes**:
- `{ athlete_id: 1 }` - Find alerts for athlete
- `{ session_id: 1 }` - Find alerts for session
- `{ alert_type: 1, status: 1 }` - Find active drift alerts

---

### 6. `monitoring_trends` Collection

**Purpose**: Stores trend analysis for monitored insights

**Schema**:
```json
{
  "_id": ObjectId,
  "metric_key": "height_off_floor_meters",
  "athlete_id": "athlete_001",
  
  // Trend analysis
  "trend": "improving" | "worsening" | "unchanged" | "insufficient_data",
  "trend_strength": 2.449,  // Rate of change per session
  "change_percent": 26.7,   // Percentage change
  
  // Values
  "first_value": 0.15,
  "last_value": 0.19,
  "first_z_score": -19.60,
  "last_z_score": -12.50,
  
  // Baseline reference
  "baseline_mean": 0.30,
  "baseline_sd": 0.008,
  
  // Session data
  "sessions_analyzed": 3,
  "metric_values": [0.15, 0.17, 0.19],
  "session_timestamps": ["2026-01-02T...", "2026-01-03T...", "2026-01-04T..."],
  
  "monitored_since": "2026-01-02T10:00:00Z",
  "analyzed_at": "2026-01-05T10:00:00Z",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend`: Overall direction (improving/worsening/unchanged)
- `trend_strength`: Quantifies rate of change
- `metric_values`: Historical values for visualization

**Indexes**:
- `{ athlete_id: 1, metric_key: 1 }` - Find trends for athlete/metric
- `{ monitored_since: 1 }` - Find recent trends

---

### 7. `trends` Collection

**Purpose**: Stores form issue trends (separate from drift trends)

**Schema**:
```json
{
  "_id": ObjectId,
  "trend_id": "athlete_001_insufficient_height_back_handspring",
  "athlete_name": "Test Athlete",
  "issue_type": "insufficient_height",
  "activity": "gymnastics",
  "technique": "back_handspring",
  
  // Trend data
  "observation": "Height decreased from 0.30m to 0.25m across 5 sessions",
  "evidence_reasoning": "Non-clinical explanation...",
  "coaching_options": [
    "Consider reducing high-impact volume",
    "If pain/symptoms are present, consult your AT/PT."
  ],
  "trend_status": "improving" | "unchanged" | "worsening" | "insufficient_data",
  "change_percent": -16.7,
  
  "sessions_analyzed": 5,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend_status`: Overall trend direction
- `observation`: Measurement summary
- `coaching_options`: LLM-generated recommendations

---

## Insight Calculation Flow

### Step 1: Extract Form Issues from Session

```python
def extract_form_issues_from_session(session):
    issues = []
    metrics = session.get("metrics", {})
    
    # For each form issue type
    for issue_type, issue_config in FORM_ISSUES.items():
        metric_keys = issue_config["metric_keys"]
        threshold = issue_config["threshold"]
        comparison = issue_config["comparison"]  # "below" or "above"
        
        # Find metric value in session
        metric_value = find_metric_in_session(metrics, metric_keys)
        
        # Check if exceeds threshold
        if comparison == "below" and metric_value < threshold:
            issues.append({
                "issue_type": issue_type,
                "description": issue_config["description"],
                "metric_key": metric_key_used,
                "metric_value": metric_value,
                "threshold": threshold,
                "severity": calculate_severity(metric_value, threshold, comparison)
            })
    
    return issues
```

**FORM_ISSUES Definition**:
```python
FORM_ISSUES = {
    "insufficient_height": {
        "description": "Insufficient height off floor/beam",
        "metric_keys": ["height_off_floor_meters", "height_off_ground", "max_height"],
        "threshold": 0.3,  # meters
        "comparison": "below"  # Issue if value < threshold
    },
    "landing_knee_bend": {
        "description": "Insufficient landing knee extension",
        "metric_keys": ["landing_knee_bend_min", "landing_knee_bend"],
        "threshold": 150.0,  # degrees (180 = straight)
        "comparison": "below"  # Issue if value < threshold
    },
    "knee_valgus_collapse": {
        "description": "Knee valgus collapse (inward collapse)",
        "metric_keys": ["acl_max_valgus_angle", "valgus_angle"],
        "threshold": 10.0,  # degrees
        "comparison": "above"  # Issue if value > threshold
    }
    // ... more issues
}
```

---

### Step 2: Find Sessions with Recurring Issues

```python
def find_sessions_with_form_issues(min_sessions_per_issue=3):
    # Step 1: Extract issues from ALL sessions
    all_sessions_with_issues = []
    for session in sessions:
        issues = extract_form_issues_from_session(session)
        if issues:
            all_sessions_with_issues.append({
                "session": session,
                "form_issues": issues,
                "form_issue_types": [i["issue_type"] for i in issues]
            })
    
    # Step 2: Count issue occurrences
    issue_counts = defaultdict(int)
    for session_data in all_sessions_with_issues:
        for issue_type in session_data["form_issue_types"]:
            issue_counts[issue_type] += 1
    
    # Step 3: Filter to recurring issues (3+ sessions)
    recurring_issues = {
        issue_type for issue_type, count in issue_counts.items()
        if count >= min_sessions_per_issue
    }
    
    # Step 4: Filter sessions to only include recurring issues
    sessions_with_recurring_issues = []
    for session_data in all_sessions_with_issues:
        session_issues = [
            issue for issue in session_data["form_issues"]
            if issue["issue_type"] in recurring_issues
        ]
        if session_issues:
            session_data["form_issues"] = session_issues
            sessions_with_recurring_issues.append(session_data)
    
    return sessions_with_recurring_issues
```

**Logic**: Only issues appearing in 3+ sessions are considered "recurring" and saved to insights

---

### Step 3: Save Insights to MongoDB

```python
def _save_insights_to_mongodb(sessions):
    for session in sessions:
        session_id = session.get("session_id")
        issues = session.get("form_issues", [])
        
        # Convert to insight format
        insights = [
            {
                "description": issue["description"],
                "is_monitored": False,
                "coach_follow_up": None,
                "monitored_at": None
            }
            for issue in issues
        ]
        
        # Save to insights collection
        mongodb.upsert_insights(
            session_id=session_id,
            insights=insights,
            metadata={
                "activity": session.get("activity"),
                "technique": session.get("technique"),
                "athlete_name": session.get("athlete_name"),
                "timestamp": session.get("timestamp"),
                "form_issue_count": len(issues),
                "form_issue_types": [i["issue_type"] for i in issues]
            }
        )
```

---

## Formulas & Calculations

### 1. Baseline Calculation

```python
# For each metric across baseline sessions:
values = [session1[metric], session2[metric], ..., sessionN[metric]]

mean = sum(values) / len(values)
variance = sum((x - mean)^2 for x in values) / len(values)
sd = sqrt(variance)
min = min(values)
max = max(values)
```

**Example**:
```
Sessions: [0.28, 0.30, 0.29, 0.31, 0.30, 0.29, 0.30, 0.28]
mean = 0.29375
sd = 0.0103
min = 0.28
max = 0.31
```

---

### 2. Z-Score Calculation

```python
z_score = (current_value - baseline_mean) / baseline_sd
```

**Example**:
```
Baseline: mean=0.30, sd=0.008
Current: 0.15
z_score = (0.15 - 0.30) / 0.008 = -18.75σ
```

**Interpretation**:
- Current value is 18.75 standard deviations below baseline
- This is severe drift

---

### 3. Deviation Percentage

```python
deviation_percent = ((current_value - baseline_mean) / abs(baseline_mean)) * 100
```

**Example**:
```
Baseline: 0.30
Current: 0.15
deviation_percent = ((0.15 - 0.30) / 0.30) * 100 = -50.0%
```

**Interpretation**: 50% below baseline

---

### 4. Trend Calculation

```python
# Compare first and last z-scores
first_z = deviations[0]["z_score"]  # -6.25σ
last_z = deviations[-1]["z_score"]  # -18.75σ

# Trend determination
if abs(last_z) < abs(first_z):
    trend = "improving"  # Getting closer to baseline
elif abs(last_z) > abs(first_z):
    trend = "worsening"  # Getting further from baseline
else:
    trend = "unchanged"

# Trend strength (linear regression slope)
z_scores = [-6.25, -12.50, -18.75]
x = [0, 1, 2]  # Session indices
y = z_scores
slope = np.polyfit(x, y, 1)[0]  # -6.25 per session
trend_strength = abs(slope)  # 6.25
```

**Interpretation**:
- Z-score changing by -6.25σ per session
- Strong worsening trend

---

### 5. Severity Classification

```python
if abs(z_score) > 3.0:
    severity = "severe"
elif abs(z_score) > 2.5:
    severity = "moderate"
else:
    severity = "minor"  # 2.0 ≤ |z| < 2.5
```

**Thresholds**:
- **Minor**: 2.0σ ≤ |z-score| < 2.5σ (68% confidence interval)
- **Moderate**: 2.5σ ≤ |z-score| < 3.0σ (95% confidence interval)
- **Severe**: |z-score| ≥ 3.0σ (99.7% confidence interval)

---

## Complete Workflow Example

### Scenario: Athlete with Baseline, New Session Arrives

1. **Baseline Exists**:
   ```json
   {
     "athlete_id": "athlete_001",
     "baseline_vector": {
       "height_off_floor_meters": {"mean": 0.30, "sd": 0.008}
     },
     "baseline_window": {"end_date": "2026-01-01T10:00:00Z"}
   }
   ```

2. **Drift Flag Active**:
   ```json
   {
     "athlete_id": "athlete_001",
     "drift_detection_enabled": true,
     "drift_detection_start_date": "2026-01-02T10:00:00Z",
     "drift_threshold": 2.0
   }
   ```

3. **New Session Arrives**:
   ```json
   {
     "session_id": "session_123",
     "athlete_id": "athlete_001",
     "timestamp": "2026-01-03T10:00:00Z",
     "metrics": {
       "height_off_floor_meters": 0.15
     }
   }
   ```

4. **Drift Detection Runs**:
   ```python
   # Calculate z-score
   z_score = (0.15 - 0.30) / 0.008 = -18.75σ
   
   # Check threshold
   abs(-18.75) > 2.0  # True → Drift detected
   
   # Classify severity
   abs(-18.75) > 3.0  # True → Severe
   
   # Determine direction
   higher_is_better = True  # height_off_floor_meters
   z_score < 0  # True
   direction = "worsening"  # Below baseline for "higher is better"
   ```

5. **Alert Created**:
   ```json
   {
     "alert_type": "technical_drift",
     "drift_metrics": {
       "height_off_floor_meters": {
         "z_score": -18.75,
         "severity": "severe",
         "direction": "worsening",
         "coach_follow_up": null,
         "is_monitored": false
       }
     }
   }
   ```

6. **Coach Sets Follow-Up**:
   ```python
   update_drift_alert_coach_follow_up(
       alert_id="alert_123",
       metric_key="height_off_floor_meters",
       coach_follow_up="Monitor"
   )
   ```

7. **Monitoring Starts**:
   - `is_monitored` = true
   - `monitored_at` = timestamp
   - Future sessions tracked for this metric

8. **Trend Calculated** (after 3+ sessions):
   ```json
   {
     "metric_key": "height_off_floor_meters",
     "trend": "worsening",
     "trend_strength": 6.25,
     "change_percent": -50.0
   }
   ```

---

## Key Design Decisions

### 1. **One Drift Flag Per Athlete**
- **Rationale**: Simpler management, global control
- **Alternative**: Per-insight flags (more granular but complex)

### 2. **Per-Insight Coach Follow-Up**
- **Rationale**: Coaches can monitor some insights, escalate others
- **Implementation**: Stored in `alerts.drift_metrics[metric_key].coach_follow_up`

### 3. **Baseline from 8 Sessions**
- **Rationale**: Statistical significance, reduces noise
- **Configurable**: Can be adjusted via `min_sessions` parameter

### 4. **Z-Score Threshold of 2.0σ**
- **Rationale**: Standard statistical threshold (95% confidence)
- **Configurable**: Can be adjusted per athlete or globally

### 5. **Recurring Issues Only (3+ sessions)**
- **Rationale**: Filters out one-off anomalies
- **Configurable**: Via `min_sessions_per_issue` parameter

---

## Data Relationships

```
athlete_id
    ↓
baselines (one active per athlete)
    ↓
drift_detection_flags (one per athlete)
    ↓
sessions (many per athlete)
    ↓
insights (one per session, multiple insights per session)
    ↓
alerts (one per drift detection, multiple metrics per alert)
    ↓
monitoring_trends (one per monitored metric)
```

---

## Summary

**Drift Detection Logic**:
1. Check baseline exists and drift detection enabled
2. Get sessions after baseline end date
3. Extract metrics from sessions
4. Calculate z-scores vs baseline
5. Identify metrics exceeding threshold
6. Classify severity and direction
7. Create alerts with per-insight coach follow-up flags
8. Track trends for monitored insights

**Collections**:
- `sessions`: Raw data source
- `baselines`: Statistical reference
- `drift_detection_flags`: Control mechanism
- `insights`: Extracted form issues
- `alerts`: Drift detections with coach follow-up
- `monitoring_trends`: Trend analysis for monitored insights
- `trends`: Form issue trends (separate system)

