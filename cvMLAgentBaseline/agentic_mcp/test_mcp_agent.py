#!/usr/bin/env python3
"""
Test script for MCP Agent.

Tests the agent logic with a sample message.
"""
import sys
import asyncio
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_mcp import TechnicalDriftAgentMCP


async def test_agent():
    """Test the MCP agent with a sample message."""
    print("=" * 60)
    print("🧪 Testing MCP Agent")
    print("=" * 60)
    
    try:
        print("\n1️⃣ Initializing agent...")
        agent = TechnicalDriftAgentMCP()
        
        print("   ✅ Agent created")
        print("   📡 Connecting to MCP servers...")
        
        await agent.initialize()
        
        print(f"   ✅ Agent initialized with {len(agent.tools)} tools")
        print(f"   📋 Available tools:")
        for i, tool in enumerate(agent.tools[:15], 1):
            print(f"      {i}. {tool.name}")
        if len(agent.tools) > 15:
            print(f"      ... and {len(agent.tools) - 15} more")
        
        print("\n2️⃣ Testing with sample message...")
        test_message = {
            "session_id": "test_session_001",
            "athlete_id": "test_athlete_001",
            "activity": "gymnastics",
            "technique": "back_handspring"
        }
        
        print(f"   📨 Message: {json.dumps(test_message, indent=6)}")
        print("\n   🔄 Processing...")
        
        result = await agent.process_video_session_message(test_message)
        
        print("\n3️⃣ Results:")
        print("=" * 60)
        print(json.dumps(result, indent=2, default=str))
        print("=" * 60)
        
        if result.get("success"):
            print("\n✅ Test completed successfully!")
        else:
            print(f"\n❌ Test failed: {result.get('error')}")
        
        print("\n4️⃣ Closing connections...")
        await agent.close()
        print("   ✅ Connections closed")
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(test_agent())
    sys.exit(0 if success else 1)

