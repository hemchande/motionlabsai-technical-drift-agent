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
├── mcp_server.py               # Main MCP server with tool registry
├── tools/                       # Tool implementations
│   ├── __init__.py
│   ├── mongodb_tools.py         # MongoDB operations
│   ├── redis_tools.py           # Redis queue operations
│   ├── websocket_tools.py       # WebSocket operations
│   ├── cloudflare_tools.py      # Cloudflare Stream operations
│   └── retrieval_tools.py       # Retrieval agent operations
├── agent.py                     # LangChain agent orchestration
├── config.py                    # Configuration management
├── tests/                       # Test files
│   ├── test_mcp_server.py
│   ├── test_agent.py
│   └── test_tools.py
└── examples/                    # Example usage
    ├── basic_usage.py
    └── full_pipeline.py
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd agentic_mcp
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp ../env_template.txt .env
# Edit .env with your credentials
```

### 3. Run Basic Test

```bash
python examples/basic_usage.py
```

### 4. Run Full Pipeline

```bash
python examples/full_pipeline.py --athlete-id test_athlete_001
```

## 🔧 Components

### MCP Server (`mcp_server.py`)

Main server that registers all tools and provides the tool registry interface.

### Tools (`tools/`)

Individual tool implementations for each service:
- **MongoDB Tools**: Query sessions, upsert insights/trends/baselines/alerts
- **Redis Tools**: Send/receive queue messages
- **WebSocket Tools**: Broadcast alerts
- **Cloudflare Tools**: Get stream URLs
- **Retrieval Tools**: Extract insights, track trends, establish baselines, detect drift

### Agent (`agent.py`)

LangChain agent that orchestrates the pipeline using the tools.

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

