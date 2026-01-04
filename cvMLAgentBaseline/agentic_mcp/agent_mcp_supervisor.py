"""
MCP Agent with Supervisor Pattern - Combines MCP servers with sub-agents.

This implementation:
1. Uses MCP servers to expose tools (MongoDB, Redis, Retrieval)
2. Creates sub-agents that use MCP tools (domain-specific agents)
3. Wraps sub-agents as tools for the supervisor
4. Supervisor coordinates using 3 high-level tools instead of 10+ direct tools

Benefits:
- Natural language delegation to sub-agents
- Memory/context passing via ToolRuntime
- Clean separation of concerns
- MCP protocol for tool definitions
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool, ToolRuntime
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import Config

# Validate configuration
Config.validate()


# Initialize LLM for all agents
llm = ChatOpenAI(
    model=Config.OPENAI_MODEL,
    temperature=Config.OPENAI_TEMPERATURE,
    api_key=Config.OPENAI_API_KEY
)


class MCPSubAgent:
    """
    Base class for sub-agents that use MCP tools.
    Each sub-agent has access to a subset of MCP tools.
    """
    
    def __init__(self, mcp_tools: list, prompt: str):
        """
        Initialize sub-agent with MCP tools.
        
        Args:
            mcp_tools: List of tools from MCP servers for this domain
            prompt: System prompt for this sub-agent
        """
        self.tools = mcp_tools
        self.prompt = prompt
        self.agent = None
    
    async def initialize(self):
        """Initialize the sub-agent."""
        if self.agent is None:
            self.agent = create_agent(
                llm,
                tools=self.tools,
                system_prompt=self.prompt
            )
    
    async def invoke(self, request: str) -> str:
        """Invoke the sub-agent with a natural language request."""
        await self.initialize()
        result = await self.agent.ainvoke({"input": request})
        # Extract the final message content
        if isinstance(result, dict):
            if "messages" in result:
                final_message = result["messages"][-1]
                if hasattr(final_message, 'content'):
                    return final_message.content
                return str(final_message)
            elif "output" in result:
                return result["output"]
        return str(result)


class TechnicalDriftAgentMCPSupervisor:
    """
    Supervisor agent that uses MCP servers but wraps tools as sub-agents.
    
    Architecture:
    1. MCP Servers → Expose tools (MongoDB, Redis, Retrieval)
    2. Sub-Agents → Use domain-specific MCP tools
    3. Supervisor → Uses 3 wrapped sub-agent tools
    """
    
    def __init__(self):
        """Initialize supervisor with MCP client."""
        # Get absolute paths to MCP server scripts
        base_path = Path(__file__).parent
        mongodb_server_path = str(base_path / "mcp_servers" / "mongodb_server.py")
        redis_server_path = str(base_path / "mcp_servers" / "redis_server.py")
        retrieval_server_path = str(base_path / "mcp_servers" / "retrieval_server.py")
        
        # Initialize MCP client
        import sys
        python_executable = sys.executable
        
        self.mcp_client = MultiServerMCPClient(
            {
                "mongodb": {
                    "command": python_executable,
                    "args": [mongodb_server_path],
                    "transport": "stdio",
                },
                "redis": {
                    "command": python_executable,
                    "args": [redis_server_path],
                    "transport": "stdio",
                },
                "retrieval": {
                    "command": python_executable,
                    "args": [retrieval_server_path],
                    "transport": "stdio",
                }
            }
        )
        
        # Sub-agents will be initialized after tools are loaded
        self.mongodb_subagent = None
        self.redis_subagent = None
        self.retrieval_subagent = None
        self.supervisor_agent = None
    
    async def initialize(self):
        """Initialize MCP client, sub-agents, and supervisor."""
        # Get all tools from MCP servers
        all_tools = await self.mcp_client.get_tools()
        
        # Group tools by domain
        mongodb_tools = [t for t in all_tools if t.name.startswith("mongodb_")]
        redis_tools = [t for t in all_tools if t.name.startswith("redis_")]
        retrieval_tools = [t for t in all_tools if t.name.startswith("retrieval_")]
        
        print(f"✅ Loaded {len(all_tools)} tools from MCP servers")
        print(f"   MongoDB: {len(mongodb_tools)} tools")
        print(f"   Redis: {len(redis_tools)} tools")
        print(f"   Retrieval: {len(retrieval_tools)} tools")
        
        # Create sub-agents with domain-specific tools
        self.mongodb_subagent = MCPSubAgent(
            mcp_tools=mongodb_tools,
            prompt="""You are a MongoDB database assistant for technical drift detection.
            Your role is to query and manage data in MongoDB collections.
            You handle: sessions, insights, baselines, and drift detection flags.
            Always parse JSON responses from tools and provide clear summaries.
            When querying sessions, use appropriate filters (athlete_id, activity, technique).
            When checking baselines, verify if baseline_exists is true before proceeding.
            Always confirm what data was found or saved in your final response."""
        )
        
        self.redis_subagent = MCPSubAgent(
            mcp_tools=redis_tools,
            prompt="""You are a Redis queue management assistant for technical drift detection.
            Your role is to send and receive messages via Redis queues.
            You handle: retrievalQueue, drift_alerts_queue, coach_followup_queue.
            Always confirm queue operations (sent/received) in your responses.
            When sending messages, include all required fields (session_id, athlete_id, etc.).
            When listening to queues, specify timeout and max_messages appropriately."""
        )
        
        self.retrieval_subagent = MCPSubAgent(
            mcp_tools=retrieval_tools,
            prompt="""You are a technical drift detection and analysis assistant.
            Your role is to extract insights, track trends, establish baselines, and detect drift.
            You handle: form issue extraction, trend tracking, baseline establishment, drift detection.
            Always confirm what analysis was performed and what results were found.
            When extracting insights, ensure min_sessions_per_issue is met (default 3).
            When establishing baselines, verify min_sessions requirement (default 8).
            When detecting drift, provide detailed deviation information."""
        )
        
        # Wrap sub-agents as tools for supervisor
        @tool
        def manage_mongodb(
            request: str,
            runtime: ToolRuntime = None
        ) -> str:
            """
            Manage MongoDB database operations.
            
            Use this for:
            - Querying sessions from MongoDB
            - Upserting insights to MongoDB
            - Getting baseline information
            - Getting drift detection flags
            
            The request should be in natural language describing what MongoDB operation you need.
            """
            # Build enhanced prompt with supervisor context if available
            if runtime and hasattr(runtime, 'state'):
                messages = runtime.state.get("messages", [])
                original_user_message = next(
                    (msg for msg in messages if hasattr(msg, 'type') and msg.type == "human"),
                    None
                )
                previous_tool_results = [
                    msg.content if hasattr(msg, 'content') else str(msg)
                    for msg in messages
                    if hasattr(msg, 'type') and msg.type == "tool"
                ]
                
                if original_user_message:
                    context_parts = [
                        "You are assisting with the following user inquiry:\n\n",
                        f"{original_user_message.content if hasattr(original_user_message, 'content') else str(original_user_message)}\n\n"
                    ]
                    if previous_tool_results:
                        context_parts.append("Previous results from other operations:\n")
                        for i, result in enumerate(previous_tool_results[-3:], 1):
                            context_parts.append(f"{i}. {result}\n")
                        context_parts.append("\n")
                    context_parts.append("You are tasked with the following sub-request:\n\n")
                    context_parts.append(request)
                    enhanced_request = "".join(context_parts)
                else:
                    enhanced_request = request
            else:
                enhanced_request = request
            
            # Run async sub-agent invocation
            # Note: Tools are called from async context, but tool functions are sync
            # We'll create a new event loop for the sub-agent call
            try:
                # Check if we're in an async context
                loop = asyncio.get_running_loop()
                # If we are, we need to run in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._mongodb_subagent.invoke(enhanced_request))
                    return future.result(timeout=60)
            except RuntimeError:
                # No running loop, can use asyncio.run directly
                return asyncio.run(self._mongodb_subagent.invoke(enhanced_request))
        
        @tool
        def manage_redis(
            request: str,
            runtime: ToolRuntime = None
        ) -> str:
            """
            Manage Redis queue operations.
            
            Use this for:
            - Sending messages to Redis queues
            - Listening to Redis queues
            
            The request should be in natural language describing what Redis operation you need.
            """
            # Build enhanced prompt with supervisor context if available
            if runtime and hasattr(runtime, 'state'):
                messages = runtime.state.get("messages", [])
                original_user_message = next(
                    (msg for msg in messages if hasattr(msg, 'type') and msg.type == "human"),
                    None
                )
                if original_user_message:
                    enhanced_request = (
                        f"You are assisting with: {original_user_message.content if hasattr(original_user_message, 'content') else str(original_user_message)}\n\n"
                        f"You are tasked with: {request}"
                    )
                else:
                    enhanced_request = request
            else:
                enhanced_request = request
            
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._redis_subagent.invoke(enhanced_request))
                    return future.result(timeout=60)
            except RuntimeError:
                return asyncio.run(self._redis_subagent.invoke(enhanced_request))
        
        @tool
        def manage_retrieval(
            request: str,
            runtime: ToolRuntime = None
        ) -> str:
            """
            Manage retrieval agent operations for insights, trends, baselines, and drift detection.
            
            Use this for:
            - Extracting insights from sessions
            - Tracking trends across sessions
            - Establishing baselines for athletes
            - Detecting technical drift from baselines
            
            The request should be in natural language describing what retrieval operation you need.
            """
            # Build enhanced prompt with supervisor context if available
            if runtime and hasattr(runtime, 'state'):
                messages = runtime.state.get("messages", [])
                original_user_message = next(
                    (msg for msg in messages if hasattr(msg, 'type') and msg.type == "human"),
                    None
                )
                previous_tool_results = [
                    msg.content if hasattr(msg, 'content') else str(msg)
                    for msg in messages
                    if hasattr(msg, 'type') and msg.type == "tool"
                ]
                
                if original_user_message:
                    context_parts = [
                        "You are assisting with the following user inquiry:\n\n",
                        f"{original_user_message.content if hasattr(original_user_message, 'content') else str(original_user_message)}\n\n"
                    ]
                    if previous_tool_results:
                        context_parts.append("Previous results from other operations:\n")
                        for i, result in enumerate(previous_tool_results[-3:], 1):
                            context_parts.append(f"{i}. {result}\n")
                        context_parts.append("\n")
                    context_parts.append("You are tasked with the following sub-request:\n\n")
                    context_parts.append(request)
                    enhanced_request = "".join(context_parts)
                else:
                    enhanced_request = request
            else:
                enhanced_request = request
            
            try:
                loop = asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, self._retrieval_subagent.invoke(enhanced_request))
                    return future.result(timeout=60)
            except RuntimeError:
                return asyncio.run(self._retrieval_subagent.invoke(enhanced_request))
        
        # Create supervisor agent with 3 wrapped sub-agent tools
        supervisor_prompt = """You are a Technical Drift Detection Supervisor that orchestrates the complete pipeline.

Your responsibilities:
1. Process video session messages from Redis queue
2. Extract insights from sessions
3. Track trends across sessions
4. Establish baselines when eligible
5. Detect technical drift from baselines
6. Generate alerts and send to appropriate queues

Available sub-agents (use these tools):
- manage_mongodb: For all MongoDB operations (query sessions, upsert insights, get baseline/drift flags)
- manage_redis: For all Redis queue operations (send/receive messages)
- manage_retrieval: For all retrieval operations (extract insights, track trends, establish baselines, detect drift)

Processing order:
1. Use manage_mongodb to query sessions for the athlete/activity/technique
2. Use manage_retrieval to extract insights from sessions (saves to MongoDB automatically)
3. Use manage_retrieval to track trends across sessions (saves to MongoDB automatically)
4. Use manage_mongodb to check if baseline exists for this athlete
5. If no baseline exists, use manage_mongodb to check if athlete has 8+ eligible sessions, then use manage_retrieval to establish baseline if needed
6. If baseline exists, use manage_retrieval to detect technical drift for the new session
7. If drift detected, use manage_redis to send alert to drift_alerts_queue

Always use the sub-agent tools (manage_mongodb, manage_redis, manage_retrieval) to interact with services.
Never make assumptions about data. Parse responses from sub-agents and use the information to make decisions."""
        
        self.supervisor_agent = create_agent(
            llm,
            tools=[manage_mongodb, manage_redis, manage_retrieval],
            system_prompt=supervisor_prompt
        )
        
        print(f"✅ Supervisor agent initialized with 3 sub-agent tools")
    
    async def process_video_session_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a message from video agent."""
        if not self.supervisor_agent:
            await self.initialize()
        
        session_id = message.get("session_id")
        athlete_id = message.get("athlete_id")
        activity = message.get("activity", "gymnastics")
        technique = message.get("technique", "back_handspring")
        
        prompt = f"""Process video session message:
- session_id: {session_id}
- athlete_id: {athlete_id}
- activity: {activity}
- technique: {technique}

Execute the complete pipeline using the sub-agent tools (manage_mongodb, manage_redis, manage_retrieval)."""
        
        try:
            result = await self.supervisor_agent.ainvoke({"input": prompt})
            return {
                "success": True,
                "result": result,
                "session_id": session_id,
                "athlete_id": athlete_id
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "athlete_id": athlete_id
            }
    
    async def close(self):
        """Close MCP client connections."""
        if hasattr(self.mcp_client, 'close'):
            await self.mcp_client.close()


async def main():
    """Example usage."""
    agent = TechnicalDriftAgentMCPSupervisor()
    
    test_message = {
        "session_id": "test_session_001",
        "athlete_id": "test_athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print("🧪 Testing MCP Supervisor Agent...")
    result = await agent.process_video_session_message(test_message)
    print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())

