# Collection Schemas & Pipeline Interactions

Complete field schemas for all collections and how they interact in the processing pipeline.

---

## Collection Field Schemas

### 1. `insights` Collection

**Purpose**: Stores extracted form issues/insights per session

**Complete Schema**:
```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "string",  // For filtering by athlete
  
  // List of insights (array of objects)
  "insights": [
    {
      "insight": "Insufficient height off floor/beam",  // or "description"
      "is_monitored": false,
      "coach_follow_up": null,  // "Monitor" | "Adjust Training" | "Escalate to AT/PT" | "Dismiss"
      "monitored_at": null  // ISODate when monitoring started
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
- `session_id`: Links to `sessions` collection
- `athlete_id`: Links to athlete (for filtering)

**Created in Pipeline**: Step 1 (before drift detection)

---

### 2. `trends` Collection

**Purpose**: Stores form issue trends (separate from drift trends)

**Complete Schema**:
```json
{
  "_id": ObjectId,
  "trend_id": "athlete_001_insufficient_height_back_handspring",
  "athlete_name": "Test Athlete",
  "athlete_id": "athlete_001",  // For baseline linking
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
  
  // Baseline integration (optional)
  "baseline_id": ObjectId,  // Link to baseline if trend compares to baseline
  "baseline_referenced": false,  // Indicates if trend uses baseline comparison
  "drift_related": false,  // Indicates if trend is from drift detection
  
  "sessions_analyzed": 5,
  "metric_signature": "height_off_floor_meters",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend_status`: Overall trend direction
- `observation`: Measurement summary
- `coaching_options`: LLM-generated recommendations
- `baseline_id`: Links to `baselines` collection (if baseline-referenced)

**Created in Pipeline**: Step 2 (after insights)

---

### 3. `monitoring_trends` Collection

**Purpose**: Stores trend analysis for monitored insights (after coach flags them)

**Complete Schema**:
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
  "baseline_id": ObjectId,  // Link to baseline
  
  // Session data
  "sessions_analyzed": 3,
  "metric_values": [0.15, 0.17, 0.19],
  "session_timestamps": ["2026-01-02T...", "2026-01-03T...", "2026-01-04T..."],
  "session_ids": ["session_1", "session_2", "session_3"],
  
  "monitored_since": "2026-01-02T10:00:00Z",  // When coach set "Monitor"
  "analyzed_at": "2026-01-05T10:00:00Z",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend`: Overall direction (improving/worsening/unchanged)
- `trend_strength`: Quantifies rate of change
- `metric_values`: Historical values for visualization
- `baseline_id`: Links to `baselines` collection

**Created in Pipeline**: Step 4 (monitoring flagged insights)

---

### 4. `baselines` Collection

**Purpose**: Stores calculated baseline vectors for athletes

**Complete Schema**:
```json
{
  "_id": ObjectId,
  "athlete_id": "string",
  "program_id": "string",  // Optional
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
  "established_by": "system" | "pt_id",
  "status": "active" | "superseded" | "invalidated",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `baseline_vector`: Mean, SD, min, max for each metric (used for z-score calculation)
- `baseline_window.end_date`: Used to find sessions after baseline
- `status`: Only "active" baselines are used for drift detection
- `athlete_id`: Links to athlete

**Created in Pipeline**: Step 5 (when 8+ eligible sessions exist)

---

### 5. `drift_detection_flags` Collection

**Purpose**: Controls when and how drift detection runs

**Complete Schema**:
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
- `baseline_id`: Links to `baselines` collection

**Created in Pipeline**: Step 5 (automatically when baseline is established, or manually if baseline exists but flag doesn't)

---

### 6. `alerts` Collection

**Purpose**: Stores drift alerts with per-insight coach follow-up

**Complete Schema**:
```json
{
  "_id": ObjectId,
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "alert_type": "technical_drift",
  "athlete_id": "athlete_001",
  "session_id": "session_123",
  "baseline_id": ObjectId,  // Link to baseline used for comparison
  
  // Per-insight drift metrics
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.30,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "worsening" | "improving",
      "severity": "minor" | "moderate" | "severe",
      "coach_follow_up": null | "Monitor" | "Escalate to AT/PT" | "Adjust Training" | "Dismiss",
      "is_monitored": false,  // True if coach_follow_up is "Monitor"
      "monitored_at": null  // ISODate when monitoring started
    }
  },
  
  "sessions_affected": ["session_123"],
  "reps_affected": 0,  // TODO: Calculate from session
  "top_clip_ids": [],  // TODO: Get clip IDs
  "alert_confidence": 0.92,
  "alert_payload_summary": "Technical drift detected: 1 metrics deviating from baseline",
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
- `baseline_id`: Links to `baselines` collection

**Created in Pipeline**: Step 6 (when drift detected)

---

## Pipeline Interactions

### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Video Agent Completes                    │
│                    Session Processing                       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Send Message to Redis Queue                    │
│              {session_id, athlete_id, ...}                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              Retrieval Queue Worker Receives                │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Find Sessions with Form Issues                      │
│   • Query sessions collection                               │
│   • Extract form issues from metrics                        │
│   • Filter to issues in 3+ sessions                         │
│   ↓                                                          │
│   ✅ SAVE TO: insights collection                           │
│      - session_id → links to sessions                      │
│      - athlete_id → links to athlete                       │
│      - insights[] → list of form issues                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Track Trends                                        │
│   • Analyze trends across sessions                          │
│   • Generate three-layer output                             │
│   ↓                                                          │
│   ✅ SAVE TO: trends collection                             │
│      - athlete_id → links to athlete                       │
│      - baseline_id → links to baselines (if referenced)    │
│      - trend_status → improving/worsening/unchanged        │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Analyze Form Patterns                               │
│   • Identify recurring patterns                             │
│   • No collection writes (analysis only)                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Monitor Flagged Insights                           │
│   • Query insights collection for monitored insights        │
│   • Track trends after monitored_at timestamp               │
│   ↓                                                          │
│   ✅ SAVE TO: monitoring_trends collection                 │
│      - metric_key → links to metric                        │
│      - athlete_id → links to athlete                       │
│      - baseline_id → links to baselines                     │
│      - trend → improving/worsening/unchanged              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Check Baseline Eligibility                          │
│   • Count eligible sessions (8+ required)                  │
│   • If threshold met and no baseline:                       │
│     ↓                                                        │
│     ✅ CREATE: baselines collection                         │
│        - athlete_id → links to athlete                     │
│        - baseline_vector → stats for each metric           │
│        - baseline_window → session range                    │
│     ↓                                                        │
│     ✅ CREATE: drift_detection_flags collection            │
│        - athlete_id → links to athlete                      │
│        - baseline_id → links to baselines                   │
│        - drift_detection_enabled → true                    │
│   • If baseline exists but no flag:                         │
│     ↓                                                        │
│     ✅ CREATE: drift_detection_flags collection            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Detect Technical Drift                              │
│   • Query baselines collection (active baseline)            │
│   • Query drift_detection_flags collection                  │
│   • Compare new session metrics to baseline                 │
│   • Calculate z-scores                                      │
│   • If drift > threshold:                                   │
│     ↓                                                        │
│     ✅ CREATE: alerts collection                            │
│        - athlete_id → links to athlete                     │
│        - session_id → links to sessions                     │
│        - baseline_id → links to baselines                   │
│        - drift_metrics → per-metric drift data             │
└─────────────────────────────────────────────────────────────┘
```

---

## Data Relationships

### Foreign Key Relationships

```
athlete_id
    ↓
    ├─→ sessions (many)
    ├─→ baselines (one active)
    ├─→ drift_detection_flags (one)
    ├─→ insights (many, via sessions)
    ├─→ trends (many)
    ├─→ monitoring_trends (many)
    └─→ alerts (many)

session_id
    ↓
    ├─→ insights (one)
    └─→ alerts (many)

baseline_id
    ↓
    ├─→ drift_detection_flags (one)
    ├─→ alerts (many)
    ├─→ trends (many, if baseline_referenced)
    └─→ monitoring_trends (many)

alert_id
    ↓
    └─→ (referenced by coach follow-up actions)
```

---

## Collection Interactions by Step

### Step 1: Insights Production

**Reads from**:
- `sessions` collection (query by activity, technique, athlete)

**Writes to**:
- `insights` collection (one document per session with form issues)

**Fields Used**:
- `sessions.metrics` → extracted to form issues
- `sessions.session_id` → saved to `insights.session_id`
- `sessions.athlete_id` → saved to `insights.athlete_id`

---

### Step 2: Trend Tracking

**Reads from**:
- `sessions` collection (multiple sessions)
- `insights` collection (to identify recurring issues)

**Writes to**:
- `trends` collection (one document per trend)

**Fields Used**:
- `sessions.metrics` → analyzed for trends
- `insights.form_issue_types` → identifies issue types
- `trends.baseline_id` → set if baseline exists (optional)

---

### Step 4: Monitoring Flagged Insights

**Reads from**:
- `insights` collection (query for `is_monitored = true`)
- `sessions` collection (query for sessions after `monitored_at`)
- `baselines` collection (for z-score calculation)

**Writes to**:
- `monitoring_trends` collection (one document per monitored metric)

**Fields Used**:
- `insights.is_monitored` → identifies which insights to monitor
- `insights.monitored_at` → timestamp to filter sessions
- `baselines.baseline_vector` → for z-score calculation
- `sessions.metrics` → current values for trend calculation

---

### Step 5: Baseline Establishment

**Reads from**:
- `sessions` collection (query for eligible sessions: `capture_confidence_score >= 0.7`, `baseline_eligible = true`)

**Writes to**:
- `baselines` collection (one document per athlete)
- `drift_detection_flags` collection (one document per athlete)

**Fields Used**:
- `sessions.athlete_id` → groups sessions
- `sessions.metrics` → calculates baseline_vector (mean, sd, min, max)
- `sessions.timestamp` → determines baseline_window
- `baselines.baseline_window.end_date` → used to set `drift_detection_start_date`

---

### Step 6: Drift Detection

**Reads from**:
- `baselines` collection (active baseline for athlete)
- `drift_detection_flags` collection (check if enabled and start date)
- `sessions` collection (new session metrics)

**Writes to**:
- `alerts` collection (one document per drift detection)

**Fields Used**:
- `baselines.baseline_vector` → mean and sd for z-score calculation
- `drift_detection_flags.drift_detection_enabled` → check if enabled
- `drift_detection_flags.drift_detection_start_date` → check if active
- `drift_detection_flags.drift_threshold` → threshold for drift (default: 2.0σ)
- `sessions.metrics` → current values to compare
- `alerts.baseline_id` → links to baseline used
- `alerts.drift_metrics` → per-metric drift data

---

## Cross-Collection Queries

### Example 1: Get All Insights for Athlete with Drift Alerts

```javascript
// 1. Get athlete's insights
insights = db.insights.find({ athlete_id: "athlete_001" })

// 2. Get athlete's drift alerts
alerts = db.alerts.find({ 
  athlete_id: "athlete_001",
  alert_type: "technical_drift"
})

// 3. Match insights to alerts by metric_key
// (insights have form_issue_types, alerts have drift_metrics keys)
```

### Example 2: Get Monitoring Trends for Monitored Drift Metrics

```javascript
// 1. Get alerts with monitored metrics
alerts = db.alerts.find({
  athlete_id: "athlete_001",
  "drift_metrics.$*.is_monitored": true
})

// 2. Get monitoring trends
monitoring_trends = db.monitoring_trends.find({
  athlete_id: "athlete_001"
})

// 3. Match by metric_key
```

### Example 3: Get Baseline and All Related Data

```javascript
// 1. Get baseline
baseline = db.baselines.findOne({
  athlete_id: "athlete_001",
  status: "active"
})

// 2. Get drift flag
drift_flag = db.drift_detection_flags.findOne({
  athlete_id: "athlete_001"
})

// 3. Get alerts using this baseline
alerts = db.alerts.find({
  baseline_id: baseline._id
})

// 4. Get monitoring trends using this baseline
monitoring_trends = db.monitoring_trends.find({
  baseline_id: baseline._id
})
```

---

## Summary Table

| Collection | Created In | Reads From | Links To | Key Linking Field |
|------------|-----------|------------|----------|-------------------|
| `insights` | Step 1 | `sessions` | `sessions`, athlete | `session_id`, `athlete_id` |
| `trends` | Step 2 | `sessions`, `insights` | athlete, `baselines` (optional) | `athlete_id`, `baseline_id` |
| `monitoring_trends` | Step 4 | `insights`, `sessions`, `baselines` | athlete, `baselines` | `athlete_id`, `baseline_id` |
| `baselines` | Step 5 | `sessions` | athlete | `athlete_id` |
| `drift_detection_flags` | Step 5 | (created with baseline) | athlete, `baselines` | `athlete_id`, `baseline_id` |
| `alerts` | Step 6 | `baselines`, `drift_detection_flags`, `sessions` | athlete, `sessions`, `baselines` | `athlete_id`, `session_id`, `baseline_id` |

---

## Key Takeaways

1. **Insights are independent** - Created from session metrics, don't require baseline
2. **Baseline enables drift detection** - Must exist before drift detection can run
3. **Drift flag controls activation** - Even with baseline, drift detection only runs if flag is enabled
4. **Alerts link everything** - Connect athlete, session, baseline, and drift metrics
5. **Monitoring trends track progress** - Follow monitored insights over time using baseline for comparison

