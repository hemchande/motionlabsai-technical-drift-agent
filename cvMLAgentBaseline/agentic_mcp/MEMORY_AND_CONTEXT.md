# Memory and Context Passing in Supervisor Pattern

## 🎯 Overview

According to LangChain's supervisor pattern, **sub-agents are stateless** - they don't retain memory. All conversation memory is maintained by the **supervisor agent**. 

**Memory flows in both directions:**
1. **Supervisor → Sub-Agent**: Supervisor passes context to sub-agents via `ToolRuntime`
2. **Sub-Agent → Supervisor**: Sub-agent return values automatically become part of supervisor's memory

## 📚 LangChain Pattern

From the [LangChain documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant):

> Sub-agents are designed to be stateless, meaning they do not retain memory of past interactions. All conversation memory is maintained by the main agent (supervisor), which coordinates sub-agents by invoking them as tools.

### How Context is Passed

The supervisor can pass context to sub-agents in two ways:

1. **Via the request string** - Include context in the natural language request
2. **Via ToolRuntime** - Access supervisor's state and pass it to sub-agents

## 🔧 Implementation

### Using ToolRuntime

We use `ToolRuntime` to access the supervisor's conversation state and pass it to sub-agents:

```python
from langchain.tools import tool, ToolRuntime

@tool
def manage_mongodb(
    request: str,
    runtime: ToolRuntime = None
) -> str:
    """Manage MongoDB database operations."""
    
    # Access supervisor's conversation state
    if runtime and hasattr(runtime, 'state'):
        messages = runtime.state.get("messages", [])
        
        # Get original user message
        original_user_message = next(
            (msg for msg in messages if msg.type == "human"),
            None
        )
        
        # Get previous tool results
        previous_tool_results = [
            msg.content for msg in messages
            if msg.type == "tool"
        ]
        
        # Build enhanced prompt with context
        prompt = (
            "You are assisting with the following user inquiry:\n\n"
            f"{original_user_message.content}\n\n"
            "Previous results from other operations:\n"
            f"{previous_tool_results}\n\n"
            "You are tasked with the following sub-request:\n\n"
            f"{request}"
        )
    else:
        prompt = request
    
    # Invoke sub-agent with enhanced context
    result = mongodb_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    return result["messages"][-1].content
```

## 🔄 Context Flow

```
1. Supervisor receives: "Process session for athlete_001"

2. Supervisor → manage_mongodb("Query sessions...")
   ↓
   ToolRuntime provides:
   - Original user message: "Process session for athlete_001"
   - Previous tool results: []
   ↓
   MongoDB Agent receives:
   "You are assisting with: Process session for athlete_001
    You are tasked with: Query sessions..."
   ↓
   Returns: "Found 10 sessions"

3. Supervisor → manage_retrieval("Extract insights...")
   ↓
   ToolRuntime provides:
   - Original user message: "Process session for athlete_001"
   - Previous tool results: ["Found 10 sessions"]
   ↓
   Retrieval Agent receives:
   "You are assisting with: Process session for athlete_001
    Previous results: Found 10 sessions
    You are tasked with: Extract insights..."
   ↓
   Returns: "Extracted insights from 5 sessions"
```

## ✅ Benefits

### 1. **Sub-Agents See Full Context**
- Original user request
- Previous tool results
- Conversation history

### 2. **Resolve Ambiguities**
Sub-agents can understand references like:
- "schedule it for the same time tomorrow" (referencing previous conversation)
- "use the athlete from the previous query" (referencing previous results)

### 3. **Better Decision Making**
Sub-agents can make informed decisions based on:
- What the user originally asked for
- What has already been done
- What information is available

## 📋 What Gets Passed

### From Supervisor to Sub-Agent:

1. **Original User Message**
   - The initial request from the user
   - Provides full context of what the user wants

2. **Previous Tool Results**
   - Results from previous sub-agent invocations
   - Helps sub-agents understand what's already been done

3. **Current Sub-Request**
   - The specific task for this sub-agent
   - What the supervisor wants this sub-agent to do

## 🎯 Example: Resolving Ambiguity

**Without Context**:
```
Supervisor → manage_retrieval("Detect drift for the athlete")
Retrieval Agent: "Which athlete?" ❌
```

**With Context**:
```
Supervisor → manage_retrieval("Detect drift for the athlete")
ToolRuntime provides:
- Original: "Process session for athlete_001"
- Previous: "Found 10 sessions for athlete_001"

Retrieval Agent receives:
"You are assisting with: Process session for athlete_001
 Previous results: Found 10 sessions for athlete_001
 You are tasked with: Detect drift for the athlete"

Retrieval Agent: "Detecting drift for athlete_001" ✅
```

## 🔍 Implementation Details

### ToolRuntime Access

The `ToolRuntime` parameter is automatically provided by LangChain when the tool is called. It gives access to:

- `runtime.state` - The supervisor's conversation state
- `runtime.state["messages"]` - All messages in the conversation
- Message types: `"human"`, `"ai"`, `"tool"`

### Message Types

```python
messages = runtime.state.get("messages", [])

# Human messages (user input)
human_messages = [msg for msg in messages if msg.type == "human"]

# Tool messages (sub-agent results)
tool_messages = [msg for msg in messages if msg.type == "tool"]

# AI messages (supervisor responses)
ai_messages = [msg for msg in messages if msg.type == "ai"]
```

## 📖 Usage

The context passing happens automatically when using the supervisor:

```python
from supervisor_agent import TechnicalDriftSupervisor

supervisor = TechnicalDriftSupervisor()

# Supervisor automatically passes context to sub-agents
result = supervisor.process_video_session_message({
    "session_id": "session_123",
    "athlete_id": "athlete_001",
    "activity": "gymnastics",
    "technique": "back_handspring"
})
```

## 💾 How Sub-Agents Save Memory in Supervisor

### Automatic Memory Storage

When a sub-agent returns a value, it **automatically becomes part of the supervisor's memory**:

```python
@tool
def manage_mongodb(request: str, runtime: ToolRuntime = None) -> str:
    """Manage MongoDB database operations."""
    result = mongodb_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    
    # This return value automatically becomes a "tool" message
    # in the supervisor's conversation history
    return result["messages"][-1].content
```

**Flow:**
1. Sub-agent executes and returns a value
2. Tool wrapper returns the value to supervisor
3. Supervisor automatically stores it as a `"tool"` message
4. This tool message becomes part of supervisor's memory
5. Future sub-agents can access it via `ToolRuntime`

### Memory Storage Mechanism

```python
# Supervisor's conversation history after sub-agent call:
messages = [
    {"type": "human", "content": "Process session for athlete_001"},
    {"type": "ai", "content": "I'll help you process the session..."},
    {"type": "tool", "name": "manage_mongodb", "content": "Found 10 sessions"},  # ← Saved automatically
    {"type": "ai", "content": "Now I'll extract insights..."},
    {"type": "tool", "name": "manage_retrieval", "content": "Extracted 5 insights"},  # ← Saved automatically
]
```

### Best Practices for Sub-Agent Returns

Sub-agents should return **structured, informative responses** that:
1. **Summarize what was done**
2. **Include key data points**
3. **Indicate success/failure**
4. **Provide context for future operations**

#### ✅ Good Return Format

```python
# MongoDB Agent returns:
{
    "success": True,
    "count": 10,
    "sessions": [
        {"session_id": "sess_1", "athlete_id": "athlete_001", ...},
        ...
    ]
}

# Retrieval Agent returns:
{
    "success": True,
    "insights_extracted": 5,
    "insights": [
        {"insight": "Insufficient height", "severity": "moderate"},
        ...
    ],
    "session_id": "sess_123"
}
```

#### ❌ Poor Return Format

```python
# Too vague
"Done"

# Missing context
"10"

# No structure
"Found some sessions and extracted some insights"
```

### Structured Return Example

Here's how sub-agents should structure their returns:

```python
@tool
def mongodb_query_sessions(athlete_id: str, activity: str) -> str:
    """Query sessions from MongoDB."""
    # ... query logic ...
    
    sessions = list(mongodb.get_sessions_collection().find(query))
    
    # Return structured JSON that will be saved in supervisor's memory
    return json.dumps({
        "success": True,
        "count": len(sessions),
        "athlete_id": athlete_id,
        "activity": activity,
        "sessions": [
            {
                "session_id": str(s.get("_id")),
                "athlete_id": s.get("athlete_id"),
                "timestamp": s.get("timestamp"),
                "metrics": s.get("metrics", {})
            }
            for s in sessions
        ]
    })
```

### Accessing Saved Memory

Future sub-agents can access previously saved results:

```python
@tool
def manage_retrieval(request: str, runtime: ToolRuntime = None) -> str:
    """Manage retrieval operations."""
    
    if runtime and hasattr(runtime, 'state'):
        messages = runtime.state.get("messages", [])
        
        # Get previous tool results (saved memory from other sub-agents)
        previous_tool_results = [
            msg.content for msg in messages
            if msg.type == "tool"
        ]
        
        # Example: previous_tool_results might contain:
        # [
        #   '{"success": true, "count": 10, "sessions": [...]}',
        #   '{"success": true, "insights_extracted": 5, ...}'
        # ]
        
        # Use this saved memory to inform current operation
        if previous_tool_results:
            # Parse and use previous results
            last_result = json.loads(previous_tool_results[-1])
            if last_result.get("success"):
                # Use the data from previous sub-agent
                session_count = last_result.get("count", 0)
                # ... use in current operation ...
```

### Complete Memory Flow Example

```
1. User: "Process session for athlete_001"
   ↓ Supervisor stores as "human" message

2. Supervisor → manage_mongodb("Query sessions...")
   ↓
   MongoDB Agent executes
   ↓
   Returns: '{"success": true, "count": 10, "sessions": [...]}'
   ↓
   Supervisor automatically stores as "tool" message ✅
   ↓
   Supervisor memory now contains:
   - "human": "Process session for athlete_001"
   - "tool": '{"success": true, "count": 10, ...}'

3. Supervisor → manage_retrieval("Extract insights...")
   ↓
   ToolRuntime provides saved memory:
   - Previous tool result: '{"success": true, "count": 10, ...}'
   ↓
   Retrieval Agent receives:
   "Previous results: Found 10 sessions
    You are tasked with: Extract insights..."
   ↓
   Retrieval Agent executes
   ↓
   Returns: '{"success": true, "insights_extracted": 5, ...}'
   ↓
   Supervisor automatically stores as "tool" message ✅
   ↓
   Supervisor memory now contains:
   - "human": "Process session for athlete_001"
   - "tool": '{"success": true, "count": 10, ...}'
   - "tool": '{"success": true, "insights_extracted": 5, ...}'

4. Supervisor can now use all saved memory to:
   - Make decisions
   - Generate final response
   - Pass to future sub-agents
```

### Key Points

1. **Automatic Storage**: Sub-agent returns are automatically saved - no extra code needed
2. **Tool Messages**: Returns become `"tool"` type messages in supervisor's history
3. **Accessible via ToolRuntime**: Future sub-agents can access via `runtime.state["messages"]`
4. **Structured Returns**: Use JSON or structured strings for better parsing
5. **Informative Content**: Include key data points that future operations might need

### Implementation in Our System

Our wrapped tools automatically save sub-agent returns:

```python
@tool
def manage_mongodb(request: str, runtime: ToolRuntime = None) -> str:
    """Manage MongoDB database operations."""
    # ... get context from runtime ...
    
    result = mongodb_agent.invoke({
        "messages": [{"role": "user", "content": prompt}],
    })
    
    # This return is automatically saved in supervisor's memory
    final_message = result["messages"][-1]
    if hasattr(final_message, 'content'):
        return final_message.content  # ← Saved as "tool" message
    return str(final_message)  # ← Saved as "tool" message
```

The supervisor's `AgentExecutor` handles the storage automatically - we just need to return meaningful values from our tools.

## 🔗 References

- [LangChain Sub-Agents Documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [Pass Additional Conversational Context to Sub-Agents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant#pass-additional-conversational-context-to-sub-agents)

