# MCP Server Initialization

## 🚀 Automatic Initialization Flow

When you start the agent, here's what happens automatically:

### 1. Agent Starts (`agent_mcp.py`)

```python
agent = TechnicalDriftAgentMCP()
await agent.initialize()
```

### 2. MultiServerMCPClient Starts All Servers

The `MultiServerMCPClient` automatically starts each MCP server as a subprocess:

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
```

### 3. Each Server Initializes on Startup

Each MCP server has an `@server.on_initialize()` hook that runs automatically:

#### MongoDB Server Initialization

```python
@server.on_initialize()
async def on_initialize() -> None:
    # 1. Validates configuration from Config class
    Config.validate()
    
    # 2. Creates MongoDB connection using Config.MONGODB_URI
    mongodb_service = MongoDBService()
    mongodb_service.connect()
    
    # 3. Tests connection
    mongodb_service.get_sessions_collection().find_one()
    
    # 4. Prints success message
    print("✅ MongoDB server initialized")
```

**Configuration Used**:
- `MONGODB_URI` from environment variables
- `MONGODB_DATABASE` from environment variables

#### Redis Server Initialization

```python
@server.on_initialize()
async def on_initialize() -> None:
    # 1. Validates configuration
    Config.validate()
    
    # 2. Creates Redis connection using Config values
    redis_client = redis.Redis(
        host=Config.REDIS_HOST,
        port=Config.REDIS_PORT,
        decode_responses=True,
        socket_connect_timeout=5,
        socket_keepalive=True
    )
    
    # 3. Tests connection
    redis_client.ping()
    
    # 4. Prints success message
    print("✅ Redis server initialized")
```

**Configuration Used**:
- `REDIS_HOST` from environment variables
- `REDIS_PORT` from environment variables

#### Retrieval Server Initialization

```python
@server.on_initialize()
async def on_initialize() -> None:
    # 1. Validates configuration
    Config.validate()
    
    # 2. Creates retrieval agent (uses MongoDB internally)
    retrieval_agent = FormCorrectionRetrievalAgent()
    
    # 3. Prints success message
    print("✅ Retrieval server initialized")
```

**Configuration Used**:
- Retrieval agent uses MongoDB (via `MongoDBService`)
- All MongoDB config is inherited

## 📋 Configuration Sources

All configuration comes from environment variables, loaded in this order:

1. **`.env` file** in `cvMLAgentBaseline/` directory
2. **System environment variables**
3. **Default values** (if not set)

### Required Environment Variables

```env
# MongoDB
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/
MONGODB_DATABASE=gymnastics_analytics

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# OpenAI (for agent)
OPENAI_API_KEY=sk-...
```

### Optional Environment Variables

```env
# OpenAI Model Settings
OPENAI_MODEL=gpt-4
OPENAI_TEMPERATURE=0.2

# WebSocket
WEBSOCKET_HOST=localhost
WEBSOCKET_PORT=8765

# Queue Names
RETRIEVAL_QUEUE=retrievalQueue
DRIFT_ALERTS_QUEUE=drift_alerts_queue
COACH_FOLLOWUP_QUEUE=coach_followup_queue

# Agent Settings
AGENT_MAX_ITERATIONS=50
AGENT_VERBOSE=true
```

## ✅ No Hardcoding

Everything is configurable:

- ❌ **No hardcoded connection strings**
- ❌ **No hardcoded host/port values**
- ❌ **No hardcoded database names**
- ❌ **No hardcoded queue names**

All values come from:
- ✅ Environment variables (`.env` file)
- ✅ `Config` class (reads from environment)
- ✅ Each server uses `Config` values

## 🔄 Connection Reuse

Connections are initialized once on startup and reused:

- **MongoDB**: Single connection pool, reused for all queries
- **Redis**: Single connection, reused for all operations
- **Retrieval Agent**: Single instance, reused for all operations

This is more efficient than creating new connections for each tool call.

## 🧪 Testing Initialization

You can test each server independently:

```bash
# Test MongoDB server
python mcp_servers/mongodb_server.py

# Test Redis server
python mcp_servers/redis_server.py

# Test Retrieval server
python mcp_servers/retrieval_server.py
```

Each server will:
1. Load configuration from `.env`
2. Initialize its connection
3. Print initialization status
4. Wait for MCP protocol messages (via stdio)

## 📊 Initialization Status

When the agent starts, you'll see:

```
✅ MongoDB server initialized
   Database: gymnastics_analytics
   URI: mongodb+srv://...

✅ Redis server initialized
   Host: localhost:6379

✅ Retrieval server initialized
   Agent ready for: extract_insights, track_trends, establish_baseline, detect_drift

✅ Loaded 10 tools from MCP servers
   Tools: mongodb_query_sessions, mongodb_upsert_insights, ...
```

If initialization fails, you'll see error messages indicating what went wrong.

