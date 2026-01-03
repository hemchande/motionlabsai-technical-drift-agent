"""
Supervisor Agent for Technical Drift Detection Pipeline.

Uses the supervisor pattern: coordinates specialized sub-agents by wrapping them as tools.
Based on: https://docs.langchain.com/oss/python/langchain/multi-agent/subagents-personal-assistant
"""
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from langchain.agents import create_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from config import Config
from subagents import create_mongodb_agent, create_redis_agent, create_retrieval_agent

# Validate configuration
Config.validate()


# Initialize LLM for all agents
llm = ChatOpenAI(
    model=Config.OPENAI_MODEL,
    temperature=Config.OPENAI_TEMPERATURE,
    api_key=Config.OPENAI_API_KEY
)

# Create sub-agents
mongodb_agent = create_mongodb_agent(llm)
redis_agent = create_redis_agent(llm)
retrieval_agent = create_retrieval_agent(llm)


# Wrap sub-agents as tools for the supervisor
@tool
def manage_mongodb(request: str) -> str:
    """
    Manage MongoDB database operations.
    
    Use this for:
    - Querying sessions from MongoDB
    - Upserting insights to MongoDB
    - Getting baseline information
    - Getting drift detection flags
    
    The request should be in natural language describing what MongoDB operation you need.
    Examples:
    - "Query sessions for athlete_001 with activity gymnastics and technique back_handspring"
    - "Get baseline for athlete_001"
    - "Check if drift detection is enabled for athlete_001"
    """
    result = mongodb_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    # Return the final message from the sub-agent
    final_message = result["messages"][-1]
    if hasattr(final_message, 'content'):
        return final_message.content
    return str(final_message)


@tool
def manage_redis(request: str) -> str:
    """
    Manage Redis queue operations.
    
    Use this for:
    - Sending messages to Redis queues
    - Listening to Redis queues
    
    The request should be in natural language describing what Redis operation you need.
    Examples:
    - "Send alert message to drift_alerts_queue with athlete_id athlete_001"
    - "Listen to retrievalQueue for new messages"
    """
    result = redis_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    # Return the final message from the sub-agent
    final_message = result["messages"][-1]
    if hasattr(final_message, 'content'):
        return final_message.content
    return str(final_message)


@tool
def manage_retrieval(request: str) -> str:
    """
    Manage retrieval agent operations for insights, trends, baselines, and drift detection.
    
    Use this for:
    - Extracting insights from sessions
    - Tracking trends across sessions
    - Establishing baselines for athletes
    - Detecting technical drift from baselines
    
    The request should be in natural language describing what retrieval operation you need.
    Examples:
    - "Extract insights for activity gymnastics and technique back_handspring"
    - "Track trends for sessions with activity gymnastics"
    - "Establish baseline for athlete_001 with min_sessions 8"
    - "Detect drift for athlete_001 with session_id session_123"
    """
    result = retrieval_agent.invoke({
        "messages": [{"role": "user", "content": request}],
    })
    # Return the final message from the sub-agent
    final_message = result["messages"][-1]
    if hasattr(final_message, 'content'):
        return final_message.content
    return str(final_message)


# Supervisor Agent Prompt
SUPERVISOR_PROMPT = (
    "You are a Technical Drift Detection Supervisor that orchestrates the complete pipeline.\n\n"
    "Your responsibilities:\n"
    "1. Process video session messages from Redis queue\n"
    "2. Extract insights from sessions\n"
    "3. Track trends across sessions\n"
    "4. Establish baselines when eligible\n"
    "5. Detect technical drift from baselines\n"
    "6. Generate alerts and send to appropriate queues\n\n"
    "Available sub-agents (use these tools):\n"
    "- manage_mongodb: For all MongoDB operations (query sessions, upsert insights, get baseline/drift flags)\n"
    "- manage_redis: For all Redis queue operations (send/receive messages)\n"
    "- manage_retrieval: For all retrieval operations (extract insights, track trends, establish baselines, detect drift)\n\n"
    "Processing order:\n"
    "1. Use manage_mongodb to query sessions for the athlete/activity/technique\n"
    "2. Use manage_retrieval to extract insights from sessions (saves to MongoDB automatically)\n"
    "3. Use manage_retrieval to track trends across sessions (saves to MongoDB automatically)\n"
    "4. Use manage_mongodb to check if baseline exists for this athlete\n"
    "5. If no baseline exists, use manage_mongodb to check if athlete has 8+ eligible sessions, then use manage_retrieval to establish baseline if needed\n"
    "6. If baseline exists, use manage_retrieval to detect technical drift for the new session\n"
    "7. If drift detected, use manage_redis to send alert to drift_alerts_queue\n\n"
    "Always use the sub-agent tools (manage_mongodb, manage_redis, manage_retrieval) to interact with services.\n"
    "Never make assumptions about data. Parse responses from sub-agents and use the information to make decisions.\n"
    "If a sub-agent returns an error, try to understand why and handle it appropriately."
)


# Create supervisor agent
supervisor_agent = create_agent(
    llm,
    tools=[manage_mongodb, manage_redis, manage_retrieval],
    prompt=SUPERVISOR_PROMPT,
)

# Create supervisor executor
supervisor_executor = AgentExecutor(
    agent=supervisor_agent,
    tools=[manage_mongodb, manage_redis, manage_retrieval],
    verbose=Config.AGENT_VERBOSE,
    max_iterations=Config.AGENT_MAX_ITERATIONS,
    handle_parsing_errors=True
)


class TechnicalDriftSupervisor:
    """
    Supervisor agent that coordinates sub-agents for technical drift detection.
    """
    
    def __init__(self):
        """Initialize supervisor with sub-agents."""
        self.executor = supervisor_executor
        self.sub_agents = {
            "mongodb": mongodb_agent,
            "redis": redis_agent,
            "retrieval": retrieval_agent
        }
    
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
        
        # Supervisor prompt
        prompt = f"""Process video session message:
- session_id: {session_id}
- athlete_id: {athlete_id}
- activity: {activity}
- technique: {technique}

Execute the complete pipeline:
1. Use manage_mongodb to query sessions for this athlete/activity/technique
2. Use manage_retrieval to extract insights from sessions (saves to MongoDB automatically)
3. Use manage_retrieval to track trends across sessions (saves to MongoDB automatically)
4. Use manage_mongodb to check if baseline exists for this athlete
5. If no baseline exists, use manage_mongodb to check if athlete has 8+ eligible sessions, then use manage_retrieval to establish baseline if needed
6. If baseline exists, use manage_retrieval to detect technical drift for the new session
7. If drift detected, use manage_redis to send alert to drift_alerts_queue

Use the sub-agent tools (manage_mongodb, manage_redis, manage_retrieval) to complete each step."""
        
        try:
            result = self.executor.invoke({"input": prompt})
            
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
        Listen to Redis queue and process messages using supervisor.
        
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
    supervisor = TechnicalDriftSupervisor()
    
    # Test with a sample message
    test_message = {
        "session_id": "test_session_001",
        "athlete_id": "test_athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print("🧪 Testing supervisor with sample message...")
    result = supervisor.process_video_session_message(test_message)
    print(f"\n📊 Result: {json.dumps(result, indent=2, default=str)}")

