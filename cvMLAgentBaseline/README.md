# MotionLabs AI - Technical Drift Detection Agent

A comprehensive system for detecting technical drift in athlete movement patterns, establishing baselines, tracking trends, and providing actionable insights for coaches and physical therapists.

## 🎯 Overview

This system processes video sessions to:
- **Extract form issues** from movement metrics
- **Track trends** across multiple sessions
- **Establish baselines** for pre-injury movement patterns
- **Detect technical drift** from established baselines
- **Generate alerts** for coaches and PTs
- **Monitor flagged insights** over time

## 🏗️ Architecture

### Pipeline Flow

```
Video Agent → Redis Queue → Retrieval Agent → MongoDB → WebSocket/Queue Alerts
```

1. **Video Agent** processes video and sends session data to Redis queue
2. **Retrieval Agent** (worker) processes queue messages:
   - Step 1: Extract insights from sessions
   - Step 2: Track trends across sessions
   - Step 3: Analyze form patterns
   - Step 4: Monitor flagged insights
   - Step 5: Check baseline eligibility & establish baselines
   - Step 6: Detect technical drift (if baseline exists)
3. **Alerts** sent to Redis queues and WebSocket for real-time notifications

## 📊 MongoDB Collections

### Core Collections

#### 1. `sessions`
Stores video session data with metrics
- `session_id`, `athlete_id`, `activity`, `technique`
- `metrics`: Movement metrics (height, knee angles, valgus, etc.)
- `capture_confidence_score`, `baseline_eligible`
- `timestamp`, `created_at`, `updated_at`

#### 2. `insights`
Stores extracted form issues per session
- `session_id`, `athlete_id`, `activity`, `technique`
- `insights`: Array of insight objects with:
  - `insight`: Description (e.g., "Insufficient height off floor/beam")
  - `is_monitored`: Boolean flag
  - `coach_follow_up`: "Monitor" | "Adjust Training" | "Escalate to AT/PT" | "Dismiss"
  - `monitored_at`: Timestamp when monitoring started
- `insight_count`, `form_issue_types`

#### 3. `trends`
Stores form issue trends across sessions
- `trend_id`, `athlete_name`, `athlete_id`, `issue_type`
- `observation`: Measurement summary
- `evidence_reasoning`: Non-clinical explanation
- `coaching_options`: Array of coaching recommendations
- `trend_status`: "improving" | "unchanged" | "worsening" | "insufficient_data"
- `sessions_analyzed`, `metric_signature`

#### 4. `baselines`
Stores movement baselines for athletes
- `athlete_id`, `baseline_type`: "pre_injury" | "pre_rehab" | "post_rehab"
- `baseline_vector`: Dictionary of metric_key → {mean, sd, min, max, percentile_rank}
- `baseline_window`: {session_count, start_date, end_date, session_ids}
- `status`: "active" | "superseded"
- `established_at`, `created_at`, `updated_at`

#### 5. `drift_detection_flags`
Controls drift detection per athlete
- `athlete_id`, `baseline_id`
- `drift_detection_enabled`: Boolean
- `drift_detection_start_date`: When drift detection should begin
- `created_at`, `updated_at`

#### 6. `alerts`
Stores drift detection alerts
- `alert_id`, `athlete_id`, `session_id`, `baseline_id`
- `alert_type`: "technical_drift"
- `drift_metrics`: Dictionary of metric_key → drift data:
  - `baseline_value`, `current_value`, `z_score`
  - `severity`: "minor" | "moderate" | "severe"
  - `direction`: "worsening" | "improving"
  - `coach_follow_up`, `is_monitored`, `monitored_at`
- `status`: "new" | "acknowledged" | "resolved"
- `created_at`, `updated_at`

#### 7. `monitoring_trends`
Stores trends for monitored insights
- `insight_id`, `athlete_id`, `session_id`
- `metric_key`, `insight_description`
- `trend`: "improving" | "unchanged" | "worsening" | "insufficient_data"
- `observation`, `evidence_reasoning`, `coaching_options`
- `sessions_analyzed`, `monitored_since`

### Collection Relationships

```
sessions → insights (1:1)
sessions → trends (1:many, via athlete_name)
sessions → baselines (many:1, via athlete_id)
baselines → drift_detection_flags (1:1)
baselines → alerts (1:many)
insights → monitoring_trends (1:many)
alerts → monitoring_trends (1:many)
```

## 🚀 Installation

### Prerequisites

- Python 3.11+
- MongoDB (Atlas or local)
- Redis server
- OpenAI API key (for LLM reasoning)

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/hemchande/motionlabsai-technical-drift-agent.git
cd motionlabsai-technical-drift-agent
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your credentials
```

Required environment variables:
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

## 📖 Usage

### Running the Full Pipeline

1. **Start the retrieval queue worker:**
```bash
cd cvMLAgentBaseline
python3 retrieval_queue_worker.py --max-messages 10 --timeout 5
```

2. **Send a test message (mock video agent):**
```bash
python3 mock_video_agent.py \
  --athlete-id test_athlete_001 \
  --session-id test_session_$(date +%s) \
  --activity gymnastics \
  --technique back_handspring \
  --queue retrievalQueue
```

3. **Run full pipeline test:**
```bash
python3 test_full_pipeline_with_session.py \
  --athlete-id test_athlete_8plus_001 \
  --drift \
  --wait 120
```

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
    drift_threshold=2.0
)
```

## 🔄 Processing Pipeline

### Step-by-Step Flow

1. **Video Agent** → Sends message to `retrievalQueue` when video processing completes

2. **Retrieval Worker** processes message:
   - **Step 1**: Find sessions with form issues (appearing in 3+ sessions)
     - Extracts issues by comparing metrics to thresholds
     - Saves to `insights` collection
   
   - **Step 2**: Track trends across sessions
     - Analyzes how issues change over time
     - Generates three-layer output (Observation, Evidence, Coaching Options)
     - Saves to `trends` collection
   
   - **Step 3**: Analyze form patterns
     - Identifies recurring issue types
   
   - **Step 4**: Monitor flagged insights
     - Tracks insights marked for monitoring
     - Saves to `monitoring_trends` collection
   
   - **Step 5**: Check baseline eligibility
     - If athlete has 8+ eligible sessions → establish baseline
     - Creates `drift_detection_flag` automatically
   
   - **Step 6**: Detect technical drift (if baseline exists)
     - Compares current session metrics to baseline
     - Calculates z-scores for each metric
     - Creates alert if drift exceeds threshold (default: 2.0σ)
     - Saves to `alerts` collection
     - Sends alert to `drift_alerts_queue`

3. **Alert Workers**:
   - `drift_alert_worker.py`: Broadcasts alerts via WebSocket
   - `coach_followup_worker.py`: Processes coach follow-up actions

## 📈 Key Features

### Form Issue Detection
- **Insufficient height**: Height off floor/beam below threshold
- **Landing knee bend**: Insufficient knee extension during landing
- **Knee valgus collapse**: Inward collapse of knees
- **Insufficient hip flexion**: Hip angle below threshold
- **Bent knees in flight**: Lack of straight knees during flight
- **Bad alignment**: Poor body alignment

### Trend Analysis
- **Three-layer output**:
  - Observation: Measurement summary (e.g., "Height decreased from 0.30m to 0.25m")
  - Evidence & Reasoning: Non-clinical explanation
  - Coaching Options: 2-5 actionable recommendations
- **Trend status**: Improving / Unchanged / Worsening / Insufficient data

### Baseline Establishment
- **Requirements**: 8+ sessions with `capture_confidence_score >= 0.7` and `baseline_eligible = true`
- **Baseline window**: Most recent 8 sessions or last 2 weeks
- **Metrics calculated**: Mean, SD, min, max, percentile rank for all relevant metrics

### Drift Detection
- **Z-score calculation**: `(current_value - baseline_mean) / baseline_sd`
- **Severity classification**:
  - Minor: |z-score| > 2.0σ
  - Moderate: |z-score| > 2.5σ
  - Severe: |z-score| > 3.0σ
- **Direction**: Worsening or Improving (based on metric type)
- **Multi-session analysis**: Tracks deviations across multiple sessions after baseline

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

## 📁 Project Structure

```
cvMLAgentBaseline/
├── form_correction_retrieval_agent.py  # Main retrieval agent
├── retrieval_queue_worker.py          # Redis queue worker
├── trend_tracker.py                    # Trend analysis
├── guardrails.py                       # LLM guardrails
├── mock_video_agent.py                 # Mock video agent for testing
├── drift_alert_worker.py              # Alert broadcasting worker
├── coach_followup_worker.py           # Coach follow-up worker
├── monitor_flagged_insights.py        # Flagged insights monitoring
├── retrieve_flagged_insights.py       # Retrieve flagged insights
├── test_full_pipeline_with_session.py # Full pipeline test
├── requirements.txt                    # Python dependencies
└── README.md                          # This file
```

## 🔐 Security

- **Never commit `.env` files** - Already in `.gitignore`
- **Use environment variables** for all credentials
- **Rotate API keys** regularly
- **Use MongoDB connection strings** with proper authentication
- **Restrict Redis access** to localhost or VPN

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

## 📚 Additional Documentation

- `PROCESSING_ORDER.md`: Detailed processing steps
- `COLLECTION_SCHEMAS_AND_PIPELINE.md`: Complete collection schemas
- `DRIFT_DETECTION_FORMULA.md`: Drift detection formula details
- `INSIGHTS_VS_TRENDS.md`: Differences between insights and trends

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

Built for MotionLabs AI to support athlete movement analysis and injury prevention.
