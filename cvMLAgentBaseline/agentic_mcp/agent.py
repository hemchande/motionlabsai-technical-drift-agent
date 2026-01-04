"""
LangChain Agent for Technical Drift Detection Pipeline Orchestration.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from config import Config
from mcp_server import TechnicalDriftMCPServer

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

Available tools:
- MongoDB: Query sessions, upsert insights/trends/baselines/alerts, get baseline/drift flags
- Redis: Send/receive queue messages, broadcast messages
- WebSocket: Broadcast alerts
- Cloudflare: Get stream URLs
- Retrieval Agent: Extract insights, track trends, establish baselines, detect drift

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


class TechnicalDriftAgent:
    """
    Main agent that orchestrates the pipeline using tools.
    """
    
    def __init__(self):
        """Initialize agent with MCP server tools."""
        # Initialize MCP server
        self.mcp_server = TechnicalDriftMCPServer()
        self.tools = self.mcp_server.get_tools()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.OPENAI_MODEL,
            temperature=Config.OPENAI_TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=AGENT_PROMPT
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=Config.AGENT_VERBOSE,
            max_iterations=Config.AGENT_MAX_ITERATIONS,
            handle_parsing_errors=True
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
2. Extract insights from sessions (saves to MongoDB automatically)
3. Track trends across sessions (saves to MongoDB automatically)
4. Check if baseline exists for this athlete
5. If no baseline exists, check if athlete has 8+ eligible sessions and establish baseline if needed
6. If baseline exists, detect technical drift for the new session
7. If drift detected, create alert and send to drift_alerts_queue

Use the available tools to complete each step. Parse JSON responses from tools to make decisions."""
        
        try:
            result = self.agent_executor.invoke({"input": prompt})
            
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
    
    def listen_to_queue(self, queue_name: str = "retrievalQueue"):
        """
        Listen to Redis queue and process messages using agent.
        
        Args:
            queue_name: Redis queue name to listen to
        """
        import redis
        
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
                    
                    result = self.process_video_session_message(message)
                    
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


if __name__ == "__main__":
    # Example usage
    agent = TechnicalDriftAgent()
    
    # Test with a sample message
    test_message = {
        "session_id": "test_session_001",
        "athlete_id": "test_athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print("🧪 Testing agent with sample message...")
    result = agent.process_video_session_message(test_message)
    print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")


