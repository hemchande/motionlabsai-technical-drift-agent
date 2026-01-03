# MotionLabs AI - Technical Drift Detection Agent

A comprehensive microservices-based system for detecting technical drift in athlete movement patterns, establishing baselines, tracking trends, and providing actionable insights for coaches and physical therapists.

## 🎯 Overview

This system processes video sessions through a distributed pipeline to:
- **Extract form issues** from movement metrics (insights)
- **Track trends** across multiple sessions
- **Establish baselines** for pre-injury movement patterns
- **Detect technical drift** from established baselines
- **Generate alerts** for coaches and PTs via queues and WebSocket
- **Monitor flagged insights** over time

## 🏗️ System Architecture

### Microservices Architecture

```
┌─────────────────┐
│  Video Agent    │ (External - processes video)
└────────┬────────┘
         │ Sends message to Redis Queue
         ↓
┌─────────────────────────────────────────┐
│         Redis Queue (retrievalQueue)     │
└────────┬─────────────────────────────────┘
         │
         ↓
┌─────────────────────────────────────────┐
│    Retrieval Queue Worker                │
│    (retrieval_queue_worker.py)           │
│    - Processes queue messages            │
│    - Runs 6-step pipeline                │
│    - Saves to MongoDB                    │
└────────┬─────────────────────────────────┘
         │
         ├─→ MongoDB (sessions, insights, trends, baselines, alerts)
         │
         ├─→ Redis Queue (drift_alerts_queue)
         │        ↓
         │   ┌─────────────────────────────┐
         │   │  Drift Alert Worker          │
         │   │  (drift_alert_worker.py)    │
         │   │  - Listens to alerts queue   │
         │   │  - Broadcasts via WebSocket  │
         │   └─────────────────────────────┘
         │
         └─→ Redis Queue (coach_followup_queue)
                  ↓
             ┌─────────────────────────────┐
             │  Coach Follow-up Worker      │
             │  (coach_followup_worker.py)  │
             │  - Listens to follow-up queue│
             │  - Updates MongoDB           │
             │  - Triggers monitoring       │
             └─────────────────────────────┘
```

### Pipeline Flow

```
Video Agent → Redis Queue → Retrieval Agent → MongoDB → WebSocket/Queue Alerts
```

**Step-by-Step Processing**:

1. **Video Agent** processes video and sends session data to `retrievalQueue`
2. **Retrieval Queue Worker** processes message:
   - **Step 1**: Extract insights from sessions → `insights` collection
   - **Step 2**: Track trends across sessions → `trends` collection
   - **Step 3**: Analyze form patterns (analysis only)
   - **Step 4**: Monitor flagged insights → `monitoring_trends` collection
   - **Step 5**: Check baseline eligibility → `baselines` + `drift_detection_flags` collections
   - **Step 6**: Detect technical drift → `alerts` collection + `drift_alerts_queue`
3. **Alert Workers** process alerts:
   - **Drift Alert Worker**: Broadcasts alerts via WebSocket
   - **Coach Follow-up Worker**: Processes coach actions

## 🔧 Microservices

### 1. Retrieval Queue Worker (`retrieval_queue_worker.py`)

**Purpose**: Main processing worker that handles the complete pipeline

**Responsibilities**:
- Listens to `retrievalQueue` Redis queue
- Processes messages from video agent
- Executes 6-step pipeline (insights → trends → baseline → drift)
- Saves all results to MongoDB
- Sends alerts to `drift_alerts_queue`

**Usage**:
```bash
python3 retrieval_queue_worker.py --max-messages 10 --timeout 5
```

**Configuration**:
- `--queue`: Redis queue name (default: `retrievalQueue`)
- `--max-messages`: Maximum messages to process before stopping
- `--timeout`: Redis BRPOP timeout in seconds

**Dependencies**:
- Redis connection
- MongoDB connection
- OpenAI API (for LLM reasoning in trends)

---

### 2. Drift Alert Worker (`drift_alert_worker.py`)

**Purpose**: Broadcasts drift alerts via WebSocket

**Responsibilities**:
- Listens to `drift_alerts_queue` Redis queue
- Processes drift alert messages
- Broadcasts alerts to connected WebSocket clients
- Formats alerts for real-time display

**Usage**:
```bash
python3 drift_alert_worker.py --port 8765
```

**Configuration**:
- `--queue`: Redis queue name (default: `drift_alerts_queue`)
- `--port`: WebSocket server port (default: 8765)
- `--host`: WebSocket server host (default: localhost)

**Message Format**:
```json
{
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "athlete_id": "athlete_001",
  "session_id": "session_123",
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.30,
      "current_value": 0.15,
      "z_score": -19.60,
      "severity": "severe",
      "direction": "worsening"
    }
  },
  "created_at": "2026-01-03T10:00:00Z"
}
```

---

### 3. Coach Follow-up Worker (`coach_followup_worker.py`)

**Purpose**: Processes coach follow-up actions on insights/alerts

**Responsibilities**:
- Listens to `coach_followup_queue` Redis queue
- Processes coach actions (Monitor, Adjust Training, Escalate, Dismiss)
- Updates MongoDB with coach decisions
- Triggers monitoring for "Monitor" actions

**Usage**:
```bash
python3 coach_followup_worker.py
```

**Message Format**:
```json
{
  "insight": "Insufficient height off floor/beam",
  "coach_follow_up": "Monitor",
  "session_id": "session_123",
  "athlete_id": "athlete_001",
  "metric_key": "height_off_floor_meters"
}
```

**Actions**:
- `"Monitor"`: Sets `is_monitored = true`, triggers trend tracking
- `"Adjust Training"`: Logs training adjustment
- `"Escalate to AT/PT"`: Flags for PT review
- `"Dismiss"`: Marks as dismissed

---

### 4. Mock Video Agent (`mock_video_agent.py`)

**Purpose**: Simulates video agent for testing

**Responsibilities**:
- Sends test messages to `retrievalQueue`
- Simulates video processing completion
- Used for testing and development

**Usage**:
```bash
python3 mock_video_agent.py \
  --athlete-id test_athlete_001 \
  --session-id test_session_$(date +%s) \
  --activity gymnastics \
  --technique back_handspring
```

---

### 5. WebSocket Insights Server (`websocket_insights_server.py`)

**Purpose**: WebSocket server for real-time coach follow-up

**Responsibilities**:
- Receives coach follow-up messages via WebSocket
- Updates MongoDB with coach decisions
- Sends follow-up actions to `coach_followup_queue`

**Usage**:
```bash
python3 websocket_insights_server.py --port 8765
```

---

## 📊 MongoDB Collections

### Complete Collection Schemas

#### 1. `sessions` Collection

**Purpose**: Stores video session data with movement metrics

**Schema**:
```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "string",
  "athlete_name": "string",
  "activity": "gymnastics",
  "technique": "back_handspring",
  "timestamp": "2026-01-03T10:00:00Z",
  "metrics": {
    "height_off_floor_meters": 0.25,
    "landing_knee_bend_min": 155.0,
    "hip_angle": 125.0,
    "acl_max_valgus_angle": 8.0
  },
  "capture_confidence_score": 0.85,
  "baseline_eligible": true,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `metrics`: Movement metrics extracted from video
- `capture_confidence_score`: Quality score (≥0.7 for baseline eligibility)
- `baseline_eligible`: Whether session can be used for baseline

---

#### 2. `insights` Collection

**Purpose**: Stores extracted form issues per session

**Schema**:
```json
{
  "_id": ObjectId,
  "session_id": "string",
  "athlete_id": "string",
  "activity": "gymnastics",
  "technique": "back_handspring",
  "insights": [
    {
      "insight": "Insufficient height off floor/beam",
      "is_monitored": false,
      "coach_follow_up": null,
      "monitored_at": null
    }
  ],
  "insight_count": 1,
  "form_issue_types": ["insufficient_height"],
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `insights`: Array of insight objects with coach flags
- `is_monitored`: True if coach set to "Monitor"
- `coach_follow_up`: "Monitor" | "Adjust Training" | "Escalate to AT/PT" | "Dismiss"

**Created In**: Step 1 (before drift detection)

---

#### 3. `trends` Collection

**Purpose**: Stores form issue trends across sessions

**Schema**:
```json
{
  "_id": ObjectId,
  "trend_id": "athlete_001_insufficient_height_back_handspring",
  "athlete_name": "Test Athlete",
  "athlete_id": "athlete_001",
  "issue_type": "insufficient_height",
  "activity": "gymnastics",
  "technique": "back_handspring",
  "observation": "Height decreased from 0.30m to 0.25m across 5 sessions",
  "evidence_reasoning": "Non-clinical explanation...",
  "coaching_options": [
    "Consider reducing high-impact volume",
    "If pain/symptoms are present, consult your AT/PT."
  ],
  "trend_status": "worsening",
  "change_percent": -16.7,
  "sessions_analyzed": 5,
  "metric_signature": "height_off_floor_meters",
  "baseline_id": ObjectId,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend_status`: "improving" | "unchanged" | "worsening" | "insufficient_data"
- `observation`: Measurement summary
- `coaching_options`: LLM-generated recommendations

**Created In**: Step 2 (after insights)

---

#### 4. `baselines` Collection

**Purpose**: Stores movement baselines for athletes

**Schema**:
```json
{
  "_id": ObjectId,
  "athlete_id": "athlete_001",
  "baseline_type": "pre_injury",
  "baseline_window": {
    "start_date": "2025-12-20T10:00:00Z",
    "end_date": "2026-01-01T10:00:00Z",
    "session_count": 8,
    "session_ids": ["session_1", "session_2", ...]
  },
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
  },
  "status": "active",
  "established_at": ISODate,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `baseline_vector`: Mean, SD, min, max for each metric (used for z-score calculation)
- `baseline_window.end_date`: Used to find sessions after baseline
- `status`: Only "active" baselines are used for drift detection

**Created In**: Step 5 (when 8+ eligible sessions exist)

---

#### 5. `drift_detection_flags` Collection

**Purpose**: Controls when and how drift detection runs

**Schema**:
```json
{
  "_id": ObjectId,
  "athlete_id": "athlete_001",
  "baseline_id": ObjectId,
  "drift_detection_enabled": true,
  "drift_detection_start_date": "2026-01-02T10:00:00Z",
  "drift_threshold": 2.0,
  "alert_on_drift": true,
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `drift_detection_enabled`: Master on/off switch
- `drift_detection_start_date`: Prevents drift detection before this date
- `drift_threshold`: Global threshold for all metrics (default: 2.0σ)

**Created In**: Step 5 (automatically when baseline is established)

---

#### 6. `alerts` Collection

**Purpose**: Stores drift detection alerts with per-metric coach follow-up

**Schema**:
```json
{
  "_id": ObjectId,
  "alert_id": "drift_athlete_001_session_123_1234567890",
  "alert_type": "technical_drift",
  "athlete_id": "athlete_001",
  "session_id": "session_123",
  "baseline_id": ObjectId,
  "drift_metrics": {
    "height_off_floor_meters": {
      "baseline_value": 0.30,
      "current_value": 0.15,
      "z_score": -19.60,
      "drift_magnitude": 19.60,
      "direction": "worsening",
      "severity": "severe",
      "coach_follow_up": null,
      "is_monitored": false,
      "monitored_at": null
    }
  },
  "status": "new",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `drift_metrics`: Dictionary of metric_key → drift data
- Each metric has its own `coach_follow_up` and `is_monitored` flags
- `status`: "new" | "acknowledged" | "resolved"

**Created In**: Step 6 (when drift detected)

---

#### 7. `monitoring_trends` Collection

**Purpose**: Stores trend analysis for monitored insights

**Schema**:
```json
{
  "_id": ObjectId,
  "metric_key": "height_off_floor_meters",
  "athlete_id": "athlete_001",
  "trend": "improving",
  "trend_strength": 2.449,
  "change_percent": 26.7,
  "first_value": 0.15,
  "last_value": 0.19,
  "first_z_score": -19.60,
  "last_z_score": -12.50,
  "baseline_mean": 0.30,
  "baseline_sd": 0.008,
  "baseline_id": ObjectId,
  "sessions_analyzed": 3,
  "metric_values": [0.15, 0.17, 0.19],
  "session_timestamps": ["2026-01-02T...", "2026-01-03T...", "2026-01-04T..."],
  "monitored_since": "2026-01-02T10:00:00Z",
  "created_at": ISODate,
  "updated_at": ISODate
}
```

**Key Fields**:
- `trend`: "improving" | "worsening" | "unchanged" | "insufficient_data"
- `trend_strength`: Quantifies rate of change
- `metric_values`: Historical values for visualization

**Created In**: Step 4 (monitoring flagged insights)

---

## 🔄 Processing Pipeline Logic

### Step-by-Step Processing Flow

#### Step 1: Find Sessions with Form Issues ✅ **INSIGHTS PRODUCED HERE**

**Logic**:
1. Query `sessions` collection by activity, technique, athlete
2. For each session, extract form issues by comparing metrics to thresholds:
   ```python
   for issue_type in FORM_ISSUES:
       metric_value = session.metrics[metric_key]
       threshold = FORM_ISSUES[issue_type]["threshold"]
       
       if metric_value < threshold:  # or > threshold
           issues.append(issue_type)
   ```
3. Count issue occurrences across all sessions
4. Filter to issues appearing in 3+ sessions (recurring issues)
5. Save to `insights` collection

**Form Issues Detected**:
- `insufficient_height`: Height off floor/beam < 0.20m
- `landing_knee_bend`: Landing knee bend < 150°
- `knee_valgus_collapse`: Valgus angle > 10°
- `insufficient_hip_flexion`: Hip angle < 120°
- `bent_knees_in_flight`: Knee extension < 170° during flight
- `bad_alignment`: Body alignment issues

**Output**: `insights` collection documents

---

#### Step 2: Track Trends

**Logic**:
1. Group sessions by athlete, issue type, activity, technique
2. Extract metric values for each issue type across sessions
3. Calculate trend:
   ```python
   first_half_mean = mean(values[:len(values)//2])
   second_half_mean = mean(values[len(values)//2:])
   
   if second_half_mean > first_half_mean * 1.1:
       trend = "improving"
   elif second_half_mean < first_half_mean * 0.9:
       trend = "worsening"
   else:
       trend = "unchanged"
   ```
4. Generate three-layer output using LLM:
   - **Observation**: Measurement summary
   - **Evidence & Reasoning**: Non-clinical explanation
   - **Coaching Options**: 2-5 actionable recommendations
5. Save to `trends` collection

**Output**: `trends` collection documents

---

#### Step 3: Analyze Form Patterns

**Logic**:
1. Analyze form issue patterns across sessions
2. Identify recurring issue types
3. No collection writes (analysis only)

**Output**: Logged analysis (no database writes)

---

#### Step 4: Monitor Flagged Insights

**Logic**:
1. Query `insights` collection for `is_monitored = true`
2. For each monitored insight:
   - Query `sessions` collection for sessions after `monitored_at` timestamp
   - Calculate trends using baseline (if exists) for z-score calculation
   - Track how monitored metrics change over time
3. Save to `monitoring_trends` collection

**Output**: `monitoring_trends` collection documents

---

#### Step 5: Check Baseline Eligibility

**Logic**:
1. For each athlete from Step 1 sessions:
   - Count eligible sessions:
     ```python
     eligible_count = count_documents({
         "athlete_id": athlete_id,
         "capture_confidence_score": {"$gte": 0.7},
         "baseline_eligible": True
     })
     ```
   - If `eligible_count >= 8` and no baseline exists:
     - Establish baseline:
       ```python
       baseline = establish_baseline(
           athlete_id=athlete_id,
           min_sessions=8,
           min_confidence_score=0.7
       )
       ```
     - Calculate baseline vector (mean, SD, min, max for each metric)
     - Save to `baselines` collection
     - Create `drift_detection_flag` automatically
   - If baseline exists but no flag:
     - Create `drift_detection_flag`

**Output**: `baselines` + `drift_detection_flags` collection documents

---

#### Step 6: Detect Technical Drift ✅ **DRIFT DETECTION HERE**

**Logic**:
1. For athlete with baseline:
   - Get active baseline from `baselines` collection
   - Get drift detection flag from `drift_detection_flags` collection
   - Check if drift detection is enabled and active
2. Compare current session metrics to baseline:
   ```python
   for metric_key, metric_value in session.metrics.items():
       baseline_mean = baseline.baseline_vector[metric_key]["mean"]
       baseline_sd = baseline.baseline_vector[metric_key]["sd"]
       
       z_score = (metric_value - baseline_mean) / baseline_sd
       
       if abs(z_score) > drift_threshold:
           # Drift detected!
           severity = classify_severity(abs(z_score))
           direction = determine_direction(z_score, metric_key)
   ```
3. If drift detected:
   - Create alert in `alerts` collection
   - Send alert to `drift_alerts_queue`

**Drift Detection Formula**:
- **Z-score**: `(current_value - baseline_mean) / baseline_sd`
- **Severity**:
  - Minor: 2.0σ ≤ |z-score| < 2.5σ
  - Moderate: 2.5σ ≤ |z-score| < 3.0σ
  - Severe: |z-score| ≥ 3.0σ
- **Direction**: Worsening or Improving (based on metric type)

**Output**: `alerts` collection documents + Redis queue message

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- MongoDB (Atlas or local instance)
- Redis server
- OpenAI API key (for LLM reasoning)

### Step-by-Step Setup

1. **Clone the repository:**
```bash
git clone https://github.com/hemchande/motionlabsai-technical-drift-agent.git
cd motionlabsai-technical-drift-agent/cvMLAgentBaseline
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp env_template.txt .env
# Edit .env with your credentials
```

**Required environment variables**:
```env
# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=gymnastics_analytics

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# OpenAI (for LLM reasoning)
OPENAI_API_KEY=sk-...

# Optional: WebSocket
WEBSOCKET_PORT=8765
WEBSOCKET_HOST=localhost
```

4. **Verify MongoDB connection:**
```bash
python3 -c "from videoAgent.mongodb_service import MongoDBService; m = MongoDBService(); m.connect(); print('✅ Connected')"
```

5. **Verify Redis connection:**
```bash
redis-cli ping
# Should return: PONG
```

6. **Test the system:**
```bash
python3 test_full_pipeline_with_session.py --athlete-id test_athlete_001 --drift
```

---

## 📖 Usage

### Running the Complete Pipeline

#### Option 1: Full Pipeline Test (Recommended for Testing)

```bash
python3 test_full_pipeline_with_session.py \
  --athlete-id test_athlete_8plus_001 \
  --drift \
  --wait 120
```

This script:
1. Creates a test session in MongoDB
2. Starts the retrieval queue worker
3. Sends a message via mock video agent
4. Monitors worker processing
5. Retrieves and displays all outputs (insights, trends, baseline, drift)

#### Option 2: Production Setup

**1. Start the retrieval queue worker:**
```bash
python3 retrieval_queue_worker.py --max-messages 100 --timeout 5
```

**2. Start the drift alert worker (optional):**
```bash
python3 drift_alert_worker.py --port 8765
```

**3. Start the coach follow-up worker (optional):**
```bash
python3 coach_followup_worker.py
```

**4. Video agent sends messages:**
```bash
# Your video agent should send messages to retrievalQueue:
python3 mock_video_agent.py \
  --athlete-id athlete_001 \
  --session-id session_$(date +%s) \
  --activity gymnastics \
  --technique back_handspring
```

---

### Running Individual Components

#### Insights Extraction
```python
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

agent = FormCorrectionRetrievalAgent()
sessions = agent.find_sessions_with_form_issues(
    activity="gymnastics",
    technique="back_handspring",
    min_sessions_per_issue=3
)
```

#### Trend Tracking
```python
trends = agent.track_form_issue_trends(
    sessions=sessions,
    activity="gymnastics",
    technique="back_handspring",
    min_sessions=3
)
```

#### Baseline Establishment
```python
baseline = agent.establish_baseline(
    athlete_id="athlete_001",
    baseline_type="pre_injury",
    min_sessions=8,
    min_confidence_score=0.7
)
```

#### Drift Detection
```python
drift_result = agent.detect_technical_drift(
    athlete_id="athlete_001",
    session_id="session_123",
    drift_threshold=2.0,
    analyze_multiple_sessions=True,
    max_sessions=10
)
```

---

## 🔍 Key Concepts

### Insights vs Trends

**Insights**:
- **What**: Form issues found in individual sessions
- **When**: Created in Step 1 (every session)
- **Calculation**: Compare metrics to thresholds
- **Output**: List of issues (e.g., "Insufficient height")
- **Requires Baseline**: ❌ No

**Trends**:
- **What**: How issues change across multiple sessions
- **When**: Created in Step 2 (when 3+ sessions)
- **Calculation**: Analyze metric changes over time
- **Output**: Trend direction (improving/worsening/unchanged) + coaching options
- **Requires Baseline**: ⚠️ Optional

See `INSIGHTS_VS_TRENDS.md` for detailed comparison.

---

### Baseline Establishment

**Requirements**:
- 8+ sessions with `capture_confidence_score >= 0.7`
- Sessions with `baseline_eligible = true`
- Same `athlete_id`, `activity`, `technique`

**What's Calculated**:
- Mean, SD, min, max for each metric
- Percentile rank (if cohort data available)
- Baseline window (session range)

**Automatic Actions**:
- Creates `drift_detection_flag` automatically
- Sets `drift_detection_start_date` to baseline end date

---

### Drift Detection Logic

**Formula**:
```
z_score = (current_value - baseline_mean) / baseline_sd

If |z_score| > drift_threshold (default: 2.0σ):
    severity = classify_severity(|z_score|)
    direction = determine_direction(z_score, metric_key)
    Create alert
```

**Severity Classification**:
- **Minor**: 2.0σ ≤ |z-score| < 2.5σ
- **Moderate**: 2.5σ ≤ |z-score| < 3.0σ
- **Severe**: |z-score| ≥ 3.0σ

**Direction**:
- For metrics where **higher is better** (height, knee extension):
  - Positive z-score → Worsening (below baseline)
  - Negative z-score → Improving (above baseline)
- For metrics where **lower is better** (valgus angle):
  - Positive z-score → Worsening (above baseline)
  - Negative z-score → Improving (below baseline)

**Multi-Session Analysis**:
- When `analyze_multiple_sessions=True`:
  - Analyzes all sessions after baseline end date
  - Tracks each metric across sessions
  - Calculates trend (improving/worsening/unchanged)
  - Shows deviations per session

See `DRIFT_DETECTION_FORMULA.md` for detailed formula.

---

## 🔧 Configuration

### Form Issue Thresholds

Edit `form_correction_retrieval_agent.py` → `FORM_ISSUES` dictionary:

```python
FORM_ISSUES = {
    "insufficient_height": {
        "metric_keys": ["height_off_floor_meters"],
        "threshold": 0.20,  # meters
        "comparison": "<",
        "description": "Insufficient height off floor/beam"
    },
    "landing_knee_bend": {
        "metric_keys": ["landing_knee_bend_min"],
        "threshold": 150.0,  # degrees
        "comparison": "<",
        "description": "Insufficient landing knee extension"
    },
    # ... more issues
}
```

### Baseline Parameters

```python
baseline = agent.establish_baseline(
    athlete_id="athlete_001",
    min_sessions=8,              # Minimum sessions required
    min_confidence_score=0.7,    # Minimum capture confidence
    baseline_type="pre_injury"   # Type of baseline
)
```

### Drift Detection Parameters

```python
drift_result = agent.detect_technical_drift(
    athlete_id="athlete_001",
    session_id="session_123",
    drift_threshold=2.0,         # Z-score threshold
    analyze_multiple_sessions=True,  # Analyze multiple sessions
    max_sessions=10              # Max sessions to analyze
)
```

---

## 🧪 Testing

### Run Full Pipeline Test
```bash
python3 test_full_pipeline_with_session.py \
  --athlete-id test_athlete_8plus_001 \
  --drift \
  --wait 120
```

### Test Individual Components
```bash
# Test baseline establishment
python3 test_baseline_drift_system.py

# Test drift detection
python3 test_drift_detection_explicit.py

# Test coach follow-up
python3 test_drift_coach_followup.py

# Test queue integration
python3 test_queue_integration.py
```

---

## 📁 Project Structure

```
cvMLAgentBaseline/
├── form_correction_retrieval_agent.py  # Main retrieval agent
├── retrieval_queue_worker.py          # Redis queue worker (main pipeline)
├── trend_tracker.py                    # Trend analysis
├── guardrails.py                       # LLM guardrails
├── drift_alert_worker.py              # Alert broadcasting worker
├── coach_followup_worker.py           # Coach follow-up worker
├── websocket_insights_server.py       # WebSocket server
├── monitor_flagged_insights.py        # Flagged insights monitoring
├── retrieve_flagged_insights.py       # Retrieve flagged insights
├── mock_video_agent.py                 # Mock video agent for testing
├── test_full_pipeline_with_session.py # Full pipeline test
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
├── PROCESSING_ORDER.md                 # Detailed processing steps
├── COLLECTION_SCHEMAS_AND_PIPELINE.md  # Complete collection schemas
├── DRIFT_DETECTION_FORMULA.md         # Drift detection formula details
└── INSIGHTS_VS_TRENDS.md              # Insights vs trends comparison
```

---

## 🔐 Security

- **Never commit `.env` files** - Already in `.gitignore`
- **Use environment variables** for all credentials
- **Rotate API keys** regularly
- **Use MongoDB connection strings** with proper authentication
- **Restrict Redis access** to localhost or VPN

---

## 🐛 Troubleshooting

### MongoDB Connection Issues
```bash
# Test connection
python3 -c "from videoAgent.mongodb_service import MongoDBService; m = MongoDBService(); m.connect()"
```

### Redis Connection Issues
```bash
# Check Redis is running
redis-cli ping

# Check queue contents
redis-cli LLEN retrievalQueue
```

### No Insights/Trends Showing
- Check that sessions have `capture_confidence_score >= 0.7`
- Verify sessions have `baseline_eligible = true`
- Ensure form issues appear in 3+ sessions (for insights)
- Check MongoDB collections directly:
```python
from videoAgent.mongodb_service import MongoDBService
m = MongoDBService()
m.connect()
insights = list(m.get_insights_collection().find().limit(5))
print(insights)
```

### Drift Detection Not Running
- Verify baseline exists: `agent._get_active_baseline(athlete_id)`
- Check drift flag is enabled: `agent._get_drift_detection_flag(athlete_id)`
- Ensure `drift_detection_start_date` has passed (if set)

---

## 📚 Additional Documentation

- `PROCESSING_ORDER.md`: Detailed processing steps and order
- `COLLECTION_SCHEMAS_AND_PIPELINE.md`: Complete collection schemas and relationships
- `DRIFT_DETECTION_FORMULA.md`: Drift detection formula and implementation details
- `INSIGHTS_VS_TRENDS.md`: Detailed comparison of insights vs trends

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

---

## 📄 License

[Your License Here]

---

## 🙏 Acknowledgments

Built for MotionLabs AI to support athlete movement analysis and injury prevention.
