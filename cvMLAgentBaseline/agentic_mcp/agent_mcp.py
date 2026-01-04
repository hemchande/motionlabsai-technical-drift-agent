"""
LangChain Agent using MultiServerMCPClient for Technical Drift Detection.

This implementation uses LangChain's MCP adapters to connect to MCP servers.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import asyncio

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mcp_adapters.client import MultiServerMCPClient

from config import Config

# Validate configuration
Config.validate()


# Agent Prompt Template
AGENT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a Technical Drift Detection Agent that orchestrates the complete pipeline.

Your responsibilities:
1. Process video session messages from Redis queue
2. Extract insights from sessions
3. Track trends across sessions
4. Establish baselines when eligible
5. Detect technical drift from baselines
6. Generate alerts and send to appropriate queues

Available tools (from MCP servers):
- MongoDB Server: Query sessions, upsert insights, get baseline/drift flags
- Redis Server: Send/receive queue messages
- Retrieval Server: Extract insights, track trends, establish baselines, detect drift

Processing order:
1. Query sessions from MongoDB for the athlete/activity/technique
2. Extract insights (if 3+ sessions with same issue) - saves to MongoDB automatically
3. Track trends (if 3+ sessions) - saves to MongoDB automatically
4. Check baseline eligibility (8+ eligible sessions)
5. If baseline doesn't exist and athlete has 8+ sessions, establish baseline
6. If baseline exists, detect drift for new sessions
7. If drift detected, create alert and send to drift_alerts_queue

Always use tools to interact with services. Never make assumptions about data.
When you get results from tools, parse the JSON and use the information to make decisions.
If a tool returns an error, try to understand why and handle it appropriately."""),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])


class TechnicalDriftAgentMCP:
    """
    Main agent that orchestrates the pipeline using MCP servers.
    """
    
    def __init__(self):
        """Initialize agent with MCP client."""
        # Get absolute paths to MCP server scripts
        base_path = Path(__file__).parent
        mongodb_server_path = str(base_path / "mcp_servers" / "mongodb_server.py")
        redis_server_path = str(base_path / "mcp_servers" / "redis_server.py")
        retrieval_server_path = str(base_path / "mcp_servers" / "retrieval_server.py")
        
        # Initialize MCP client with multiple servers
        # Use python3 instead of python for compatibility
        import sys
        python_executable = sys.executable  # Use the same Python that's running this script
        
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
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Tools will be loaded asynchronously
        self.tools = None
        self.agent = None
    
    async def initialize(self):
        """Initialize agent by getting tools from MCP servers."""
        # Get all tools from MCP servers
        self.tools = await self.mcp_client.get_tools()
        
        print(f"✅ Loaded {len(self.tools)} tools from MCP servers")
        print(f"   Tools: {', '.join([t.name for t in self.tools[:10]])}...")
        
        # Create agent using create_agent (LangChain 1.2.0+)
        # create_agent returns a Runnable that can be invoked directly
        # Extract system prompt from the prompt template
        # AGENT_PROMPT is a ChatPromptTemplate with messages
        # The first message is the system message
        system_message = AGENT_PROMPT.messages[0]
        if hasattr(system_message, 'prompt') and hasattr(system_message.prompt, 'template'):
            system_prompt_text = system_message.prompt.template
        elif hasattr(system_message, 'content'):
            system_prompt_text = system_message.content
        else:
            # Fallback: use the string representation
            system_prompt_text = str(system_message)
        
        self.agent = create_agent(
            self.llm,
            tools=self.tools,
            system_prompt=system_prompt_text
        )
    
    async def process_video_session_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message from video agent.
        
        Args:
            message: Message from video agent with session_id, athlete_id, etc.
        
        Returns:
            Dictionary with processing results
        """
        if not self.agent:
            await self.initialize()
        
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
2. Extract insights from sessions (saves to MongoDB automatically)
3. Track trends across sessions (saves to MongoDB automatically)
4. Check if baseline exists for this athlete
5. If no baseline exists, check if athlete has 8+ eligible sessions and establish baseline if needed
6. If baseline exists, detect technical drift for the new session
7. If drift detected, create alert and send to drift_alerts_queue

Use the available tools to complete each step. Parse JSON responses from tools to make decisions."""
        
        try:
            # In LangChain 1.2.0+, create_agent returns a Runnable
            # We can invoke it directly with the input
            result = await self.agent.ainvoke({"input": prompt})
            
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
    
    async def listen_to_queue(self, queue_name: str = "retrievalQueue"):
        """
        Listen to Redis queue and process messages using agent.
        
        Args:
            queue_name: Redis queue name to listen to
        """
        import redis
        
        # Initialize agent first
        await self.initialize()
        
        redis_client = redis.Redis(
            host=Config.REDIS_HOST,
            port=Config.REDIS_PORT,
            decode_responses=True
        )
        
        print(f"🎧 Listening to queue: {queue_name}")
        print(f"   Redis: {Config.REDIS_HOST}:{Config.REDIS_PORT}")
        
        while True:
            try:
                result = redis_client.brpop(queue_name, timeout=5)
                if result:
                    queue, message_json = result
                    message = json.loads(message_json)
                    
                    print(f"\n📨 Processing message: {message.get('session_id')}")
                    print(f"   Athlete: {message.get('athlete_id')}")
                    print(f"   Activity: {message.get('activity')}")
                    print(f"   Technique: {message.get('technique')}")
                    
                    result = await self.process_video_session_message(message)
                    
                    if result.get("success"):
                        print(f"✅ Processed successfully")
                    else:
                        print(f"❌ Error: {result.get('error')}")
            except KeyboardInterrupt:
                print("\n🛑 Stopping queue listener...")
                break
            except Exception as e:
                print(f"❌ Error processing message: {e}")
                continue
    
    async def close(self):
        """Close MCP client connections."""
        # MultiServerMCPClient manages its own connections
        # No explicit close needed, but we can clean up if needed
        if hasattr(self.mcp_client, 'close'):
            await self.mcp_client.close()


async def main():
    """Example usage."""
    agent = TechnicalDriftAgentMCP()
    
    # Test with a sample message
    test_message = {
        "session_id": "test_session_001",
        "athlete_id": "test_athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print("🧪 Testing agent with sample message...")
    result = await agent.process_video_session_message(test_message)
    print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())


