# Agentic MCP Server Architecture with LangChain

## 🎯 Overview

This document outlines an agentic approach to the Technical Drift Detection system using LangChain with an MCP (Model Context Protocol) server. Services (MongoDB, Redis, WebSocket, Cloudflare Stream, etc.) are registered as tools that an intelligent agent orchestrates.

## 🏗️ Architecture Design

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              LangChain Agent (Orchestrator)                  │
│              - Uses LLM for decision making                  │
│              - Orchestrates pipeline steps                  │
│              - Calls tools based on context                 │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Uses MCP Tools
               ↓
┌─────────────────────────────────────────────────────────────┐
│              MCP Server (Tool Registry)                     │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ MongoDB Tool │  │  Redis Tool  │  │ WebSocket Tool│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Cloudflare   │  │ Retrieval    │  │ Video Agent  │    │
│  │ Stream Tool  │  │ Agent Tool   │  │ Tool         │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
               │
               │ Tool Calls
               ↓
┌─────────────────────────────────────────────────────────────┐
│              External Services                               │
│  - MongoDB Atlas                                             │
│  - Redis Server                                              │
│  - WebSocket Server                                          │
│  - Cloudflare Stream                                         │
│  - Video Processing Agent                                    │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 MCP Server Implementation

### Core MCP Server Structure

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from typing import Dict, Any, List, Optional
import json

class TechnicalDriftMCPServer:
    """
    MCP Server that registers all services as tools for the agent.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.2)
        self.tools = []
        self.agent = None
        self._register_all_tools()
        self._create_agent()
    
    def _register_all_tools(self):
        """Register all services as tools."""
        self.tools.extend([
            # MongoDB Tools
            self.mongodb_query_sessions,
            self.mongodb_upsert_insights,
            self.mongodb_upsert_trends,
            self.mongodb_upsert_baseline,
            self.mongodb_upsert_alert,
            self.mongodb_get_baseline,
            self.mongodb_get_drift_flag,
            
            # Redis Tools
            self.redis_send_to_queue,
            self.redis_listen_to_queue,
            self.redis_broadcast_message,
            
            # WebSocket Tools
            self.websocket_send_alert,
            self.websocket_send_followup,
            
            # Cloudflare Tools
            self.cloudflare_get_stream_url,
            self.cloudflare_upload_clip,
            
            # Retrieval Agent Tools
            self.retrieval_extract_insights,
            self.retrieval_track_trends,
            self.retrieval_establish_baseline,
            self.retrieval_detect_drift,
            
            # Video Agent Tools
            self.video_agent_get_session_status,
            self.video_agent_trigger_processing,
        ])
```

---

## 🛠️ Tool Definitions

### MongoDB Tools

#### 1. Query Sessions Tool

```python
@tool
def mongodb_query_sessions(
    athlete_id: Optional[str] = None,
    activity: Optional[str] = None,
    technique: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_confidence: float = 0.7,
    baseline_eligible: bool = True
) -> Dict[str, Any]:
    """
    Query sessions collection from MongoDB.
    
    Args:
        athlete_id: Filter by athlete ID
        activity: Filter by activity (e.g., "gymnastics")
        technique: Filter by technique (e.g., "back_handspring")
        date_from: Start date (ISO format)
        date_to: End date (ISO format)
        min_confidence: Minimum capture_confidence_score
        baseline_eligible: Filter by baseline_eligible flag
    
    Returns:
        Dictionary with sessions list and metadata
    """
    from videoAgent.mongodb_service import MongoDBService
    
    mongodb = MongoDBService()
    mongodb.connect()
    
    query = {}
    if athlete_id:
        query["athlete_id"] = athlete_id
    if activity:
        query["activity"] = activity
    if technique:
        query["technique"] = technique
    if min_confidence:
        query["capture_confidence_score"] = {"$gte": min_confidence}
    if baseline_eligible:
        query["baseline_eligible"] = baseline_eligible
    
    if date_from or date_to:
        query["timestamp"] = {}
        if date_from:
            query["timestamp"]["$gte"] = date_from
        if date_to:
            query["timestamp"]["$lte"] = date_to
    
    sessions = list(mongodb.get_sessions_collection().find(query))
    
    mongodb.close()
    
    return {
        "count": len(sessions),
        "sessions": [
            {
                "session_id": str(s.get("_id")),
                "athlete_id": s.get("athlete_id"),
                "metrics": s.get("metrics", {}),
                "timestamp": s.get("timestamp"),
                "confidence_score": s.get("capture_confidence_score")
            }
            for s in sessions
        ]
    }
```

#### 2. Upsert Insights Tool

```python
@tool
def mongodb_upsert_insights(
    session_id: str,
    athlete_id: str,
    insights: List[Dict[str, Any]],
    activity: str,
    technique: str
) -> Dict[str, Any]:
    """
    Upsert insights to MongoDB insights collection.
    
    Args:
        session_id: Session identifier
        athlete_id: Athlete identifier
        insights: List of insight objects with "insight" field
        activity: Activity type
        technique: Technique performed
    
    Returns:
        Dictionary with success status and document ID
    """
    from videoAgent.mongodb_service import MongoDBService
    
    mongodb = MongoDBService()
    mongodb.connect()
    
    result = mongodb.upsert_insights(
        session_id=session_id,
        athlete_id=athlete_id,
        insights=insights,
        activity=activity,
        technique=technique
    )
    
    mongodb.close()
    
    return {
        "success": True,
        "session_id": session_id,
        "insight_count": len(insights)
    }
```

#### 3. Upsert Baseline Tool

```python
@tool
def mongodb_upsert_baseline(
    athlete_id: str,
    baseline_vector: Dict[str, Dict[str, float]],
    baseline_window: Dict[str, Any],
    baseline_type: str = "pre_injury"
) -> Dict[str, Any]:
    """
    Upsert baseline to MongoDB baselines collection.
    
    Args:
        athlete_id: Athlete identifier
        baseline_vector: Dictionary of metric_key → {mean, sd, min, max}
        baseline_window: {start_date, end_date, session_count, session_ids}
        baseline_type: "pre_injury" | "pre_rehab" | "post_rehab"
    
    Returns:
        Dictionary with baseline_id and success status
    """
    from videoAgent.mongodb_service import MongoDBService
    
    mongodb = MongoDBService()
    mongodb.connect()
    
    baseline_doc = {
        "athlete_id": athlete_id,
        "baseline_type": baseline_type,
        "baseline_vector": baseline_vector,
        "baseline_window": baseline_window,
        "status": "active",
        "established_at": datetime.utcnow()
    }
    
    result = mongodb.upsert_baseline(baseline_doc)
    
    mongodb.close()
    
    return {
        "success": True,
        "baseline_id": str(result.get("_id")),
        "metric_count": len(baseline_vector)
    }
```

#### 4. Upsert Alert Tool

```python
@tool
def mongodb_upsert_alert(
    alert_id: str,
    athlete_id: str,
    session_id: str,
    baseline_id: str,
    drift_metrics: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Upsert drift alert to MongoDB alerts collection.
    
    Args:
        alert_id: Unique alert identifier
        athlete_id: Athlete identifier
        session_id: Session identifier
        baseline_id: Baseline identifier
        drift_metrics: Dictionary of metric_key → drift data
    
    Returns:
        Dictionary with alert_id and success status
    """
    from videoAgent.mongodb_service import MongoDBService
    
    mongodb = MongoDBService()
    mongodb.connect()
    
    alert_doc = {
        "alert_id": alert_id,
        "alert_type": "technical_drift",
        "athlete_id": athlete_id,
        "session_id": session_id,
        "baseline_id": baseline_id,
        "drift_metrics": drift_metrics,
        "status": "new",
        "created_at": datetime.utcnow()
    }
    
    result = mongodb.upsert_alert(alert_doc)
    
    mongodb.close()
    
    return {
        "success": True,
        "alert_id": alert_id,
        "metric_count": len(drift_metrics)
    }
```

---

### Redis Tools

#### 5. Send to Queue Tool

```python
@tool
def redis_send_to_queue(
    queue_name: str,
    message: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Send message to Redis queue.
    
    Args:
        queue_name: Queue name (e.g., "retrievalQueue", "drift_alerts_queue")
        message: Message dictionary to send
    
    Returns:
        Dictionary with success status
    """
    import redis
    
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    
    message_json = json.dumps(message)
    redis_client.lpush(queue_name, message_json)
    
    return {
        "success": True,
        "queue_name": queue_name,
        "message_sent": True
    }
```

#### 6. Listen to Queue Tool

```python
@tool
def redis_listen_to_queue(
    queue_name: str,
    timeout: int = 5,
    max_messages: int = 1
) -> Dict[str, Any]:
    """
    Listen to Redis queue and retrieve messages.
    
    Args:
        queue_name: Queue name to listen to
        timeout: Timeout in seconds
        max_messages: Maximum messages to retrieve
    
    Returns:
        Dictionary with messages list
    """
    import redis
    
    redis_client = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )
    
    messages = []
    for _ in range(max_messages):
        result = redis_client.brpop(queue_name, timeout=timeout)
        if result:
            queue, message_json = result
            messages.append(json.loads(message_json))
        else:
            break
    
    return {
        "count": len(messages),
        "messages": messages
    }
```

---

### WebSocket Tools

#### 7. Send Alert via WebSocket Tool

```python
@tool
def websocket_send_alert(
    alert_data: Dict[str, Any],
    websocket_port: int = 8765
) -> Dict[str, Any]:
    """
    Send drift alert via WebSocket to connected clients.
    
    Args:
        alert_data: Alert dictionary with athlete_id, session_id, drift_metrics
        websocket_port: WebSocket server port
    
    Returns:
        Dictionary with success status
    """
    import asyncio
    import websockets
    
    async def send_alert():
        uri = f"ws://localhost:{websocket_port}"
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps({
                "type": "drift_alert",
                "data": alert_data
            }))
            response = await websocket.recv()
            return json.loads(response)
    
    result = asyncio.run(send_alert())
    
    return {
        "success": True,
        "alert_sent": True
    }
```

---

### Cloudflare Stream Tools

#### 8. Get Stream URL Tool

```python
@tool
def cloudflare_get_stream_url(
    video_id: str
) -> Dict[str, Any]:
    """
    Get Cloudflare Stream URL for video.
    
    Args:
        video_id: Cloudflare Stream video ID
    
    Returns:
        Dictionary with stream URL and metadata
    """
    import requests
    
    cloudflare_account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID")
    cloudflare_api_token = os.getenv("CLOUDFLARE_API_TOKEN")
    
    url = f"https://api.cloudflare.com/client/v4/accounts/{cloudflare_account_id}/stream/{video_id}"
    headers = {
        "Authorization": f"Bearer {cloudflare_api_token}"
    }
    
    response = requests.get(url, headers=headers)
    data = response.json()
    
    return {
        "success": True,
        "video_id": video_id,
        "stream_url": data.get("result", {}).get("playback", {}).get("hls"),
        "thumbnail_url": data.get("result", {}).get("thumbnail")
    }
```

---

### Retrieval Agent Tools

#### 9. Extract Insights Tool

```python
@tool
def retrieval_extract_insights(
    activity: str,
    technique: str,
    athlete_id: Optional[str] = None,
    min_sessions_per_issue: int = 3
) -> Dict[str, Any]:
    """
    Extract form issues/insights from sessions.
    
    Args:
        activity: Activity type (e.g., "gymnastics")
        technique: Technique (e.g., "back_handspring")
        athlete_id: Optional athlete ID filter
        min_sessions_per_issue: Minimum sessions for issue to be saved
    
    Returns:
        Dictionary with sessions found and insights extracted
    """
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    
    agent = FormCorrectionRetrievalAgent()
    sessions = agent.find_sessions_with_form_issues(
        activity=activity,
        technique=technique,
        athlete_name=None,
        min_sessions_per_issue=min_sessions_per_issue
    )
    
    return {
        "success": True,
        "sessions_found": len(sessions),
        "sessions": [
            {
                "session_id": s.get("session_id"),
                "athlete_id": s.get("athlete_id"),
                "form_issues": s.get("form_issues", [])
            }
            for s in sessions
        ]
    }
```

#### 10. Track Trends Tool

```python
@tool
def retrieval_track_trends(
    sessions: List[Dict[str, Any]],
    activity: str,
    technique: str,
    athlete_name: Optional[str] = None
) -> Dict[str, Any]:
    """
    Track trends across sessions.
    
    Args:
        sessions: List of session dictionaries
        activity: Activity type
        technique: Technique
        athlete_name: Optional athlete name
    
    Returns:
        Dictionary with trends identified
    """
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    
    agent = FormCorrectionRetrievalAgent()
    trends = agent.track_form_issue_trends(
        sessions=sessions,
        activity=activity,
        technique=technique,
        athlete_name=athlete_name,
        min_sessions_per_issue=3
    )
    
    return {
        "success": True,
        "trends_identified": len(trends),
        "trends": trends
    }
```

#### 11. Establish Baseline Tool

```python
@tool
def retrieval_establish_baseline(
    athlete_id: str,
    baseline_type: str = "pre_injury",
    min_sessions: int = 8,
    min_confidence_score: float = 0.7
) -> Dict[str, Any]:
    """
    Establish baseline for athlete.
    
    Args:
        athlete_id: Athlete identifier
        baseline_type: "pre_injury" | "pre_rehab" | "post_rehab"
        min_sessions: Minimum sessions required
        min_confidence_score: Minimum capture confidence
    
    Returns:
        Dictionary with baseline_id and metrics
    """
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    
    agent = FormCorrectionRetrievalAgent()
    result = agent.establish_baseline(
        athlete_id=athlete_id,
        baseline_type=baseline_type,
        min_sessions=min_sessions,
        min_confidence_score=min_confidence_score
    )
    
    return {
        "success": True,
        "baseline_id": result.get("baseline_id"),
        "metric_count": result.get("metric_count"),
        "session_count": result.get("session_count")
    }
```

#### 12. Detect Drift Tool

```python
@tool
def retrieval_detect_drift(
    athlete_id: str,
    session_id: Optional[str] = None,
    drift_threshold: float = 2.0,
    analyze_multiple_sessions: bool = True,
    max_sessions: int = 10
) -> Dict[str, Any]:
    """
    Detect technical drift from baseline.
    
    Args:
        athlete_id: Athlete identifier
        session_id: Optional specific session ID
        drift_threshold: Z-score threshold (default: 2.0σ)
        analyze_multiple_sessions: Analyze multiple sessions if True
        max_sessions: Maximum sessions to analyze
    
    Returns:
        Dictionary with drift results and alerts
    """
    from form_correction_retrieval_agent import FormCorrectionRetrievalAgent
    
    agent = FormCorrectionRetrievalAgent()
    result = agent.detect_technical_drift(
        athlete_id=athlete_id,
        session_id=session_id,
        drift_threshold=drift_threshold,
        analyze_multiple_sessions=analyze_multiple_sessions,
        max_sessions=max_sessions
    )
    
    return {
        "success": result is not None,
        "drift_detected": result is not None,
        "drift_count": result.get("drift_count", 0) if result else 0,
        "alert_id": result.get("alert_id") if result else None,
        "insights": result.get("insights", []) if result else []
    }
```

---

## 🤖 Agent Orchestration

### Agent Prompt Template

```python
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Technical Drift Detection Agent that orchestrates the complete pipeline.

Your responsibilities:
1. Process video session messages from Redis queue
2. Extract insights from sessions
3. Track trends across sessions
4. Establish baselines when eligible
5. Detect technical drift from baselines
6. Generate alerts and send to appropriate queues

Available tools:
- MongoDB: Query sessions, upsert insights/trends/baselines/alerts
- Redis: Send/receive queue messages
- WebSocket: Broadcast alerts
- Cloudflare: Get stream URLs
- Retrieval Agent: Extract insights, track trends, establish baselines, detect drift

Processing order:
1. Query sessions from MongoDB
2. Extract insights (if 3+ sessions with same issue)
3. Track trends (if 3+ sessions)
4. Check baseline eligibility (8+ eligible sessions)
5. Detect drift (if baseline exists)

Always use tools to interact with services. Never make assumptions about data."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

### Agent Execution Logic

```python
class TechnicalDriftAgent:
    """
    Main agent that orchestrates the pipeline using tools.
    """
    
    def __init__(self):
        self.mcp_server = TechnicalDriftMCPServer()
        self.agent_executor = AgentExecutor(
            agent=self.mcp_server.agent,
            tools=self.mcp_server.tools,
            verbose=True
        )
    
    def process_video_session_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from video agent.
        
        Args:
            message: Message from video agent with session_id, athlete_id, etc.
        
        Returns:
            Dictionary with processing results
        """
        session_id = message.get("session_id")
        athlete_id = message.get("athlete_id")
        activity = message.get("activity", "gymnastics")
        technique = message.get("technique", "back_handspring")
        
        # Agent prompt
        prompt = f"""Process video session message:
- session_id: {session_id}
- athlete_id: {athlete_id}
- activity: {activity}
- technique: {technique}

Execute the complete pipeline:
1. Query sessions from MongoDB for this athlete/activity/technique
2. Extract insights from sessions (save to MongoDB)
3. Track trends across sessions (save to MongoDB)
4. Check if baseline exists or should be established
5. If baseline exists, detect technical drift
6. If drift detected, create alert and send to drift_alerts_queue

Use the available tools to complete each step."""
        
        result = self.agent_executor.invoke({"input": prompt})
        
        return {
            "success": True,
            "result": result,
            "session_id": session_id,
            "athlete_id": athlete_id
        }
    
    def listen_to_queue(self, queue_name: str = "retrievalQueue"):
        """
        Listen to Redis queue and process messages using agent.
        """
        import redis
        
        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            decode_responses=True
        )
        
        while True:
            result = redis_client.brpop(queue_name, timeout=5)
            if result:
                queue, message_json = result
                message = json.loads(message_json)
                
                print(f"📨 Processing message: {message.get('session_id')}")
                result = self.process_video_session_message(message)
                print(f"✅ Processed: {result.get('success')}")
```

---

## 🔄 Agent Decision Making

### Example Agent Workflow

```
1. Agent receives message: {session_id, athlete_id, activity, technique}

2. Agent decides: "I need to query sessions first"
   → Calls: mongodb_query_sessions(athlete_id, activity, technique)
   → Result: 10 sessions found

3. Agent decides: "I should extract insights from these sessions"
   → Calls: retrieval_extract_insights(activity, technique, athlete_id)
   → Result: 5 sessions with form issues, insights saved

4. Agent decides: "I should track trends across these sessions"
   → Calls: retrieval_track_trends(sessions, activity, technique)
   → Result: 3 trends identified, saved to MongoDB

5. Agent decides: "I should check if baseline exists"
   → Calls: mongodb_get_baseline(athlete_id)
   → Result: No baseline found

6. Agent decides: "I should check if athlete has 8+ eligible sessions"
   → Calls: mongodb_query_sessions(athlete_id, min_confidence=0.7, baseline_eligible=True)
   → Result: 8 eligible sessions found

7. Agent decides: "I should establish baseline"
   → Calls: retrieval_establish_baseline(athlete_id)
   → Result: Baseline established, baseline_id returned

8. Agent decides: "Now I should detect drift for the new session"
   → Calls: retrieval_detect_drift(athlete_id, session_id)
   → Result: Drift detected, 4 metrics with drift

9. Agent decides: "I should create alert and send to queue"
   → Calls: mongodb_upsert_alert(alert_id, athlete_id, session_id, baseline_id, drift_metrics)
   → Calls: redis_send_to_queue("drift_alerts_queue", alert_data)
   → Result: Alert created and queued
```

---

## 🎯 Benefits of Agentic Approach

### 1. **Intelligent Orchestration**
- Agent makes decisions based on context
- Can handle edge cases dynamically
- Adapts to different scenarios

### 2. **Unified Interface**
- All services accessible through tools
- Consistent API for all operations
- Easy to add new services

### 3. **Error Handling**
- Agent can retry failed operations
- Can handle partial failures gracefully
- Can make alternative decisions

### 4. **Observability**
- Agent reasoning is logged
- Tool calls are traceable
- Decision process is transparent

### 5. **Extensibility**
- Easy to add new tools
- Agent learns to use new capabilities
- No code changes needed for new services

---

## 📋 Implementation Plan

### Phase 1: MCP Server Setup
1. Create `mcp_server.py` with tool registry
2. Implement MongoDB tools
3. Implement Redis tools
4. Test tool registration

### Phase 2: Agent Integration
1. Create LangChain agent with tools
2. Implement agent prompt template
3. Test agent decision making
4. Add error handling

### Phase 3: Pipeline Migration
1. Replace `retrieval_queue_worker.py` with agent
2. Test full pipeline with agent
3. Compare results with current implementation
4. Optimize agent prompts

### Phase 4: Advanced Features
1. Add agent memory for context
2. Implement agent retry logic
3. Add agent monitoring/logging
4. Create agent dashboard

---

## 🔧 Tool Registration Pattern

### Example: MongoDB Tool Registration

```python
from langchain_core.tools import tool
from pydantic import BaseModel, Field

class QuerySessionsInput(BaseModel):
    athlete_id: Optional[str] = Field(None, description="Athlete ID filter")
    activity: Optional[str] = Field(None, description="Activity filter")
    technique: Optional[str] = Field(None, description="Technique filter")
    min_confidence: float = Field(0.7, description="Minimum confidence score")

@tool(args_schema=QuerySessionsInput)
def mongodb_query_sessions(
    athlete_id: Optional[str] = None,
    activity: Optional[str] = None,
    technique: Optional[str] = None,
    min_confidence: float = 0.7
) -> str:
    """
    Query sessions from MongoDB.
    
    Use this tool to find sessions for an athlete, activity, or technique.
    Returns JSON string with sessions list.
    """
    # Implementation...
    return json.dumps(result)
```

---

## 🧠 Agent Reasoning Examples

### Example 1: Baseline Establishment Decision

**Agent Reasoning**:
```
User: Process session for athlete_001

Agent thinks:
1. "I need to check if athlete_001 has a baseline"
   → Calls: mongodb_get_baseline("athlete_001")
   → Result: No baseline found

2. "I should check how many eligible sessions they have"
   → Calls: mongodb_query_sessions(athlete_id="athlete_001", min_confidence=0.7, baseline_eligible=True)
   → Result: 8 sessions found

3. "They have 8 sessions, which meets the threshold. I should establish a baseline"
   → Calls: retrieval_establish_baseline("athlete_001", min_sessions=8)
   → Result: Baseline established successfully

4. "Now I should create a drift detection flag"
   → Calls: mongodb_upsert_drift_flag(athlete_id="athlete_001", baseline_id=...)
   → Result: Flag created
```

### Example 2: Drift Detection Decision

**Agent Reasoning**:
```
User: Process session for athlete_001

Agent thinks:
1. "I need to check if athlete_001 has a baseline"
   → Calls: mongodb_get_baseline("athlete_001")
   → Result: Baseline found (baseline_id: "baseline_123")

2. "I should check if drift detection is enabled"
   → Calls: mongodb_get_drift_flag("athlete_001")
   → Result: Flag found, drift_detection_enabled=True

3. "I should detect drift for the new session"
   → Calls: retrieval_detect_drift("athlete_001", session_id="session_456")
   → Result: Drift detected, 4 metrics with drift

4. "I should create an alert and send it to the queue"
   → Calls: mongodb_upsert_alert(...)
   → Calls: redis_send_to_queue("drift_alerts_queue", alert_data)
   → Result: Alert created and queued
```

---

## 🔐 Security Considerations

### Tool Access Control

```python
class SecureMCPServer:
    """
    MCP Server with access control for tools.
    """
    
    def __init__(self, allowed_tools: List[str] = None):
        self.allowed_tools = allowed_tools or []
        self.tool_permissions = {
            "mongodb_query_sessions": ["read"],
            "mongodb_upsert_insights": ["write"],
            "mongodb_upsert_baseline": ["write"],
            "redis_send_to_queue": ["write"],
            # ... more tools
        }
    
    def _check_permission(self, tool_name: str, action: str) -> bool:
        """Check if tool has permission for action."""
        if tool_name not in self.allowed_tools:
            return False
        return action in self.tool_permissions.get(tool_name, [])
```

---

## 📊 Monitoring & Observability

### Agent Execution Logging

```python
class MonitoredAgent:
    """
    Agent with execution monitoring.
    """
    
    def __init__(self):
        self.execution_log = []
        self.tool_call_log = []
    
    def process_with_monitoring(self, message: Dict[str, Any]):
        """Process message with full monitoring."""
        start_time = time.time()
        
        # Log agent reasoning
        self.execution_log.append({
            "timestamp": datetime.utcnow(),
            "message": message,
            "tool_calls": []
        })
        
        result = self.agent_executor.invoke({"input": prompt})
        
        execution_time = time.time() - start_time
        
        # Log execution metrics
        self.execution_log[-1].update({
            "execution_time": execution_time,
            "tool_calls_count": len(self.tool_call_log),
            "result": result
        })
        
        return result
```

---

## 🚀 Deployment

### Docker Compose Setup

```yaml
version: '3.8'

services:
  mcp-server:
    build: .
    environment:
      - MONGODB_URI=${MONGODB_URI}
      - REDIS_HOST=redis
      - REDIS_PORT=6379
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - redis
      - mongodb
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
  
  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_DATABASE=gymnastics_analytics
```

---

## 📝 Next Steps

1. **Implement MCP Server**: Create `mcp_server.py` with all tool definitions
2. **Create Agent**: Integrate LangChain agent with MCP tools
3. **Test Pipeline**: Run full pipeline with agent
4. **Add Monitoring**: Implement execution logging
5. **Optimize Prompts**: Refine agent prompts for better decisions
6. **Deploy**: Set up production deployment

---

## 🔗 References

- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [MCP Protocol](https://modelcontextprotocol.io/)
- [LangChain Tools](https://python.langchain.com/docs/modules/tools/)


