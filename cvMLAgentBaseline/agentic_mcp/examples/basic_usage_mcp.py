#!/usr/bin/env python3
"""
Basic usage example for Agentic MCP Agent using MultiServerMCPClient.
"""
import sys
import asyncio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_mcp import TechnicalDriftAgentMCP
import json


async def main():
    """Run basic agent test."""
    print("🚀 Initializing Technical Drift Agent with MCP...")
    agent = TechnicalDriftAgentMCP()
    
    await agent.initialize()
    
    print(f"✅ Agent initialized with {len(agent.tools)} tools")
    print(f"   Tools: {', '.join([t.name for t in agent.tools[:10]])}...")
    
    # Test with a sample message
    test_message = {
        "session_id": "test_session_001",
        "athlete_id": "test_athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print(f"\n🧪 Processing test message...")
    print(f"   Session ID: {test_message['session_id']}")
    print(f"   Athlete ID: {test_message['athlete_id']}")
    
    result = await agent.process_video_session_message(test_message)
    
    print(f"\n📊 Result:")
    print(json.dumps(result, indent=2, default=str))
    
    await agent.close()


if __name__ == "__main__":
    asyncio.run(main())

