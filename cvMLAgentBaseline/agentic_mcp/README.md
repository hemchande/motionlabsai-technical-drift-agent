# Agentic MCP Server Implementation

This directory contains the implementation of the agentic MCP (Model Context Protocol) server using LangChain for orchestrating the Technical Drift Detection pipeline.

## 🎯 Overview

This implementation replaces the procedural `retrieval_queue_worker.py` with an intelligent agent that:
- Uses LLM reasoning to make decisions
- Orchestrates all services through tools
- Handles edge cases dynamically
- Provides transparent decision-making

## 📁 Structure

```
agentic_mcp/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.py                    # Configuration management
│
├── agent.py                     # LangChain agent (direct tools approach)
├── agent_mcp.py                 # LangChain agent (MCP server approach) ⭐ RECOMMENDED
│
├── mcp_server.py               # Legacy: Direct tool registry
├── tools/                       # Legacy: Direct tool implementations
│   ├── __init__.py
│   ├── mongodb_tools.py
│   ├── redis_tools.py
│   ├── websocket_tools.py
│   ├── cloudflare_tools.py
│   └── retrieval_tools.py
│
├── mcp_servers/                 # MCP Protocol Servers ⭐ NEW
│   ├── __init__.py
│   ├── mongodb_server.py        # MongoDB MCP server
│   ├── redis_server.py          # Redis MCP server
│   └── retrieval_server.py      # Retrieval agent MCP server
│
└── examples/                    # Example usage
    ├── basic_usage.py           # Direct tools example
    ├── basic_usage_mcp.py       # MCP server example ⭐ RECOMMENDED
    └── full_pipeline.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd agentic_mcp
pip install -r requirements.txt
```

**Important**: Make sure `langchain-mcp-adapters` is installed for the MCP approach.

### 2. Configure Environment

```bash
cp ../env_template.txt .env
# Edit .env with your credentials
```

### 3. Run Supervisor Pattern Test ⭐⭐ RECOMMENDED

```bash
python examples/supervisor_usage.py
```

### 4. Run Basic Test (MCP Server Approach)

```bash
python examples/basic_usage_mcp.py
```

### 5. Run Basic Test (Direct Tools Approach)

```bash
python examples/basic_usage.py
```

### 6. Run Full Pipeline

```bash
python examples/full_pipeline.py --athlete-id test_athlete_001
```

## 🔄 Three Implementation Approaches

### Approach 1: Supervisor Pattern (Recommended) ⭐⭐

Uses LangChain's supervisor pattern with sub-agents wrapped as tools.

**Files**:
- `supervisor_agent.py` - Supervisor that coordinates sub-agents
- `subagents/mongodb_agent.py` - MongoDB sub-agent
- `subagents/redis_agent.py` - Redis sub-agent
- `subagents/retrieval_agent.py` - Retrieval sub-agent

**Usage**:
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

**Benefits**:
- ✅ Supervisor pattern from LangChain documentation
- ✅ Sub-agents have focused responsibilities
- ✅ Each sub-agent has its own tools and prompt
- ✅ Supervisor coordinates via wrapped sub-agent tools
- ✅ Clear separation of concerns
- ✅ Easy to add new sub-agents

**Based on**: [LangChain Supervisor Pattern Documentation](https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant)

### Approach 2: MCP Servers

Uses LangChain's `MultiServerMCPClient` to connect to proper MCP protocol servers.

**Files**:
- `agent_mcp.py` - Agent using MultiServerMCPClient
- `mcp_servers/mongodb_server.py` - MongoDB MCP server
- `mcp_servers/redis_server.py` - Redis MCP server
- `mcp_servers/retrieval_server.py` - Retrieval agent MCP server

**Usage**:
```python
from agent_mcp import TechnicalDriftAgentMCP
import asyncio

async def main():
    agent = TechnicalDriftAgentMCP()
    await agent.initialize()
    
    result = await agent.process_video_session_message({
        "session_id": "session_123",
        "athlete_id": "athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    })
    
    await agent.close()

asyncio.run(main())
```

**Benefits**:
- ✅ Proper MCP protocol implementation
- ✅ Servers can run independently
- ✅ Better separation of concerns
- ✅ Can use HTTP transport for remote servers

### Approach 3: Direct Tools (Legacy)

Uses LangChain tools directly without MCP protocol.

**Files**:
- `agent.py` - Agent with direct tools
- `tools/*.py` - Direct tool implementations

**Usage**:
```python
from agent import TechnicalDriftAgent

agent = TechnicalDriftAgent()
result = agent.process_video_session_message({
    "session_id": "session_123",
    "athlete_id": "athlete_001",
    "activity": "gymnastics",
    "technique": "back_handspring"
})
```

## 🔧 Components

### MCP Servers (`mcp_servers/`) ⭐ RECOMMENDED

Proper MCP protocol servers that expose tools:
- **MongoDB Server** (`mongodb_server.py`): Query sessions, upsert insights, get baseline/drift flags
- **Redis Server** (`redis_server.py`): Send/receive queue messages
- **Retrieval Server** (`retrieval_server.py`): Extract insights, track trends, establish baselines, detect drift

Each server follows the MCP protocol and can run as a subprocess or HTTP server.

**Automatic Initialization**: When the agent starts, it automatically:
1. ✅ Starts all MCP servers as subprocesses
2. ✅ Each server initializes its connections on startup:
   - MongoDB server connects to MongoDB using `Config.MONGODB_URI`
   - Redis server connects to Redis using `Config.REDIS_HOST` and `Config.REDIS_PORT`
   - Retrieval server initializes the retrieval agent (which uses MongoDB)
3. ✅ All configuration comes from environment variables (`.env` file)
4. ✅ No hardcoding - everything is configurable
5. ✅ Connection pooling - connections are reused, not recreated on each call

**Startup Logs**: Each server prints initialization status to stderr:
```
✅ MongoDB server initialized
   Database: gymnastics_analytics
   URI: mongodb+srv://...

✅ Redis server initialized
   Host: localhost:6379

✅ Retrieval server initialized
   Agent ready for: extract_insights, track_trends, establish_baseline, detect_drift
```

### Agent with MCP (`agent_mcp.py`) ⭐ RECOMMENDED

Uses `MultiServerMCPClient` to connect to MCP servers:
```python
client = MultiServerMCPClient({
    "mongodb": {
        "command": "python",
        "args": ["mcp_servers/mongodb_server.py"],
        "transport": "stdio",
    },
    "redis": {
        "command": "python",
        "args": ["mcp_servers/redis_server.py"],
        "transport": "stdio",
    },
    "retrieval": {
        "command": "python",
        "args": ["mcp_servers/retrieval_server.py"],
        "transport": "stdio",
    }
})
all_tools = await client.get_tools()
```

### Legacy Components

- **MCP Server** (`mcp_server.py`): Direct tool registry (legacy)
- **Tools** (`tools/`): Direct tool implementations (legacy)
- **Agent** (`agent.py`): Agent with direct tools (legacy)

## 📖 Usage

### Basic Usage

```python
from agent import TechnicalDriftAgent

agent = TechnicalDriftAgent()

# Process a message from video agent
message = {
    "session_id": "session_123",
    "athlete_id": "athlete_001",
    "activity": "gymnastics",
    "technique": "back_handspring"
}

result = agent.process_video_session_message(message)
print(result)
```

### Queue Listener

```python
from agent import TechnicalDriftAgent

agent = TechnicalDriftAgent()

# Listen to Redis queue
agent.listen_to_queue("retrievalQueue")
```

## 🧪 Testing

```bash
# Test MCP server
pytest tests/test_mcp_server.py

# Test individual tools
pytest tests/test_tools.py

# Test agent
pytest tests/test_agent.py
```

## 📚 Documentation

See `../AGENTIC_MCP_ARCHITECTURE.md` for complete architecture documentation.

