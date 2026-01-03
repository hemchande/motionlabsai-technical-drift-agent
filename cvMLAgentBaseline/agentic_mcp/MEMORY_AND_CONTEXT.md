# Memory and Context Passing in Supervisor Pattern

## 🎯 Overview

According to LangChain's supervisor pattern, **sub-agents are stateless** - they don't retain memory. All conversation memory is maintained by the **supervisor agent**. However, the supervisor can pass context to sub-agents when invoking them.

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

## 🔗 References

- [LangChain Sub-Agents Documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)
- [Pass Additional Conversational Context to Sub-Agents](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant#pass-additional-conversational-context-to-sub-agents)

