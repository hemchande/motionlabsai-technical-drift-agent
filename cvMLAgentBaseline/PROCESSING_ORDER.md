# Processing Order in Retrieval Queue Worker

## Current Flow

When a message is received from the queue, the retrieval worker processes in this order:

### Step 1: Find Sessions with Form Issues ✅ **INSIGHTS PRODUCED HERE**
```
📊 Step 1: Finding sessions with form issues...
   ✅ Found X sessions with form issues
```

**What happens**:
- Calls `find_sessions_with_form_issues()`
- Extracts form issues from session metrics
- **Automatically saves insights to MongoDB** via `_save_insights_to_mongodb()`
- Only saves insights for issues appearing in 3+ sessions (recurring issues)

**Insights saved to**: `insights` collection in MongoDB

---

### Step 2: Track Trends
```
📈 Step 2: Tracking trends across sessions...
   ✅ Identified X trends
```

**What happens**:
- Analyzes trends across multiple sessions
- Generates three-layer output (Observation, Evidence, Coaching Options)
- Saves trends to `trends` collection

---

### Step 3: Analyze Form Patterns
```
🔍 Step 3: Analyzing form patterns...
   ✅ Analyzed X issue types
```

**What happens**:
- Analyzes form issue patterns across sessions
- Identifies recurring issue types

---

### Step 4: Monitor Flagged Insights
```
📈 Step 4: Monitoring flagged insights...
   ✅ Monitored X flagged insights
```

**What happens**:
- Monitors insights that coaches have flagged for follow-up
- Tracks trends after the flag timestamp
- Saves to `monitoring_trends` collection

---

### Step 5: Check Baseline Eligibility
```
📊 Step 5: Checking baseline eligibility...
   ✅ Baseline established successfully!
```

**What happens**:
- Checks if athlete has enough sessions (8 default)
- Establishes baseline if threshold met
- Creates drift detection flag if baseline exists but flag doesn't

---

### Step 6: Detect Technical Drift ✅ **DRIFT DETECTION HERE**
```
🔍 Step 6: Detecting technical drift...
   ✅ Drift detected!
```

**What happens**:
- Compares new session metrics to baseline
- Calculates z-scores
- Creates drift alerts if drift exceeds threshold
- Saves alerts to `alerts` collection

---

## Summary

**YES - Insights are produced BEFORE drift detection**

1. **Step 1** → Insights extracted and saved to MongoDB
2. **Steps 2-4** → Trend analysis and monitoring
3. **Step 5** → Baseline establishment (if needed)
4. **Step 6** → Drift detection (runs AFTER insights are already saved)

---

## Code Reference

**Insights saved in**:
```python
# form_correction_retrieval_agent.py, line 426
def find_sessions_with_form_issues(...):
    # ... extract issues ...
    self._save_insights_to_mongodb(sessions_with_issues)  # ← Saves here
    return sessions_with_issues
```

**Drift detection runs in**:
```python
# retrieval_queue_worker.py, Step 6
drift_result = self.retrieval_agent.detect_technical_drift(...)  # ← Runs here
```

---

## Why This Order?

1. **Insights are independent** of baseline/drift - they identify form issues from metrics
2. **Drift detection requires baseline** - can only run after baseline is established
3. **Both serve different purposes**:
   - **Insights**: Identify form issues (e.g., "Insufficient height")
   - **Drift**: Detect deviation from baseline (e.g., "Height decreased from baseline")

---

## Example Timeline

```
Session 1-7:
  Step 1: ✅ Insights saved
  Step 5: ℹ️  Not enough sessions for baseline
  Step 6: ⏭️  Skipped (no baseline)

Session 8:
  Step 1: ✅ Insights saved
  Step 5: ✅ Baseline established
  Step 6: ⏭️  Skipped (drift detection starts after baseline)

Session 9:
  Step 1: ✅ Insights saved
  Step 5: ✅ Baseline exists
  Step 6: ✅ Drift detection runs
```

---

## Collections Updated

| Step | Collection | When |
|------|-----------|------|
| Step 1 | `insights` | Every session with form issues |
| Step 2 | `trends` | When 3+ sessions with trends |
| Step 4 | `monitoring_trends` | When flagged insights exist |
| Step 5 | `baselines` | When 8+ eligible sessions |
| Step 5 | `drift_detection_flags` | When baseline established |
| Step 6 | `alerts` | When drift detected |

