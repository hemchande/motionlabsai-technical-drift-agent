# Supervisor Pattern Implementation

## 🎯 Overview

This implementation follows the **LangChain Supervisor Pattern** as described in the [official documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant).

Instead of directly exposing tools, we:
1. Create specialized **sub-agents** with domain-specific tools and prompts
2. **Wrap sub-agents as tools** for the supervisor
3. Create a **supervisor agent** that coordinates sub-agents

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│              Supervisor Agent                                │
│              - Coordinates sub-agents                         │
│              - Uses wrapped sub-agent tools                  │
│              - Makes high-level decisions                    │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Uses wrapped sub-agent tools
               ↓
┌─────────────────────────────────────────────────────────────┐
│         Wrapped Sub-Agent Tools                              │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ manage_mongodb│  │ manage_redis │  │manage_retrieval│    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Invokes sub-agents
               ↓
┌─────────────────────────────────────────────────────────────┐
│              Sub-Agents                                      │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ MongoDB Agent │  │  Redis Agent │  │Retrieval Agent│    │
│  │ - Tools: 4    │  │ - Tools: 2   │  │ - Tools: 4   │    │
│  │ - Prompt: DB  │  │ - Prompt: Q  │  │ - Prompt: DR │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────┬──────────────────────────────────────────────┘
               │
               │ Uses domain tools
               ↓
┌─────────────────────────────────────────────────────────────┐
│              Services                                         │
│  - MongoDB                                                    │
│  - Redis                                                      │
│  - Retrieval Agent                                            │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Components

### 1. Sub-Agents (`subagents/`)

Each sub-agent is a specialized agent with:
- **Domain-specific tools**: Only tools relevant to that domain
- **Focused prompt**: Instructions specific to that domain
- **Clear responsibility**: One domain per agent

#### MongoDB Sub-Agent (`subagents/mongodb_agent.py`)

**Tools**:
- `mongodb_query_sessions` - Query sessions collection
- `mongodb_upsert_insights` - Save insights
- `mongodb_get_baseline` - Get baseline
- `mongodb_get_drift_flag` - Get drift flag

**Prompt**: "You are a MongoDB database assistant for technical drift detection..."

**Responsibility**: All MongoDB operations

#### Redis Sub-Agent (`subagents/redis_agent.py`)

**Tools**:
- `redis_send_to_queue` - Send messages
- `redis_listen_to_queue` - Receive messages

**Prompt**: "You are a Redis queue management assistant..."

**Responsibility**: All Redis queue operations

#### Retrieval Sub-Agent (`subagents/retrieval_agent.py`)

**Tools**:
- `retrieval_extract_insights` - Extract form issues
- `retrieval_track_trends` - Track trends
- `retrieval_establish_baseline` - Establish baseline
- `retrieval_detect_drift` - Detect drift

**Prompt**: "You are a technical drift detection and analysis assistant..."

**Responsibility**: All retrieval/analysis operations

### 2. Wrapped Sub-Agent Tools

Sub-agents are wrapped as tools using the `@tool` decorator:

```python
@tool
def manage_mongodb(request: str) -> str:
    """Manage MongoDB database operations."""
    result = mongodb_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    return result["messages"][-1].content
```

**Key Points**:
- Takes a **natural language request** string
- Invokes the sub-agent with the request
- Returns the sub-agent's response
- Supervisor uses these as regular tools

### 3. Supervisor Agent (`supervisor_agent.py`)

The supervisor:
- Has access to **3 wrapped sub-agent tools**: `manage_mongodb`, `manage_redis`, `manage_retrieval`
- Makes **high-level decisions** about which sub-agent to use
- **Coordinates workflow** across domains
- Uses **natural language** to communicate with sub-agents

## 🔄 How It Works

### Example: Processing a Video Session

```
1. Supervisor receives: "Process session for athlete_001"

2. Supervisor decides: "I need to query MongoDB first"
   → Calls: manage_mongodb("Query sessions for athlete_001 with activity gymnastics and technique back_handspring")
   → MongoDB Agent executes: mongodb_query_sessions(athlete_id="athlete_001", ...)
   → Returns: "Found 10 sessions"

3. Supervisor decides: "I should extract insights"
   → Calls: manage_retrieval("Extract insights for activity gymnastics and technique back_handspring")
   → Retrieval Agent executes: retrieval_extract_insights(activity="gymnastics", ...)
   → Returns: "Extracted insights from 5 sessions"

4. Supervisor decides: "I should check baseline"
   → Calls: manage_mongodb("Get baseline for athlete_001")
   → MongoDB Agent executes: mongodb_get_baseline(athlete_id="athlete_001")
   → Returns: "No baseline found"

5. Supervisor decides: "I should check if athlete has 8+ sessions"
   → Calls: manage_mongodb("Query sessions for athlete_001 with min_confidence 0.7 and baseline_eligible true")
   → MongoDB Agent executes: mongodb_query_sessions(...)
   → Returns: "Found 8 eligible sessions"

6. Supervisor decides: "I should establish baseline"
   → Calls: manage_retrieval("Establish baseline for athlete_001 with min_sessions 8")
   → Retrieval Agent executes: retrieval_establish_baseline(athlete_id="athlete_001", ...)
   → Returns: "Baseline established successfully"

7. Supervisor decides: "I should detect drift"
   → Calls: manage_retrieval("Detect drift for athlete_001 with session_id session_123")
   → Retrieval Agent executes: retrieval_detect_drift(athlete_id="athlete_001", ...)
   → Returns: "Drift detected, 4 metrics with drift"

8. Supervisor decides: "I should send alert"
   → Calls: manage_redis("Send alert message to drift_alerts_queue with alert_id alert_123")
   → Redis Agent executes: redis_send_to_queue(queue_name="drift_alerts_queue", ...)
   → Returns: "Alert sent successfully"
```

## ✅ Benefits

### 1. **Clear Separation of Concerns**
- Each sub-agent has a focused responsibility
- Tools are grouped by domain
- Prompts are domain-specific

### 2. **Better Tool Selection**
- Supervisor chooses between 3 high-level tools (not 10+ low-level tools)
- Sub-agents handle tool selection within their domain
- Reduces complexity for the supervisor

### 3. **Natural Language Communication**
- Supervisor communicates with sub-agents in natural language
- Sub-agents parse natural language requests
- More flexible than structured tool calls

### 4. **Easier to Extend**
- Add new sub-agent → Wrap as tool → Supervisor can use it
- No need to modify supervisor prompt for new tools
- Each sub-agent can evolve independently

### 5. **Follows LangChain Best Practices**
- Based on official LangChain documentation
- Proven pattern for multi-agent systems
- Well-documented and supported

## 📖 Usage

### Basic Usage

```python
from supervisor_agent import TechnicalDriftSupervisor

supervisor = TechnicalDriftSupervisor()

result = supervisor.process_video_session_message({
    "session_id": "session_123",
    "athlete_id": "athlete_001",
    "activity": "gymnastics",
    "technique": "back_handspring"
})
```

### Queue Listener

```python
supervisor = TechnicalDriftSupervisor()
supervisor.listen_to_queue("retrievalQueue")
```

## 🔍 Comparison with Other Approaches

| Aspect | Supervisor Pattern | MCP Servers | Direct Tools |
|--------|-------------------|-------------|--------------|
| **Architecture** | Sub-agents wrapped as tools | MCP protocol servers | Direct tool definitions |
| **Tool Count** | 3 high-level tools | 10+ tools | 10+ tools |
| **Complexity** | Medium (sub-agents + supervisor) | High (MCP protocol) | Low (direct) |
| **Natural Language** | ✅ Yes (sub-agent requests) | ❌ No (structured) | ❌ No (structured) |
| **Separation** | ✅ Excellent | ✅ Good | ❌ Poor |
| **LangChain Pattern** | ✅ Official pattern | ⚠️ Custom | ⚠️ Custom |

## 📚 References

- [LangChain Supervisor Pattern Documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [Multi-Agent Overview](https://docs.langchain.com/oss/python/langchain/multi-agent/)


