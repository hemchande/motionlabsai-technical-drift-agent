#!/usr/bin/env python3
"""
Test script for MCP Supervisor Agent.

Tests the agent with a sample message.
"""
import sys
import asyncio
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from agent_mcp_supervisor import TechnicalDriftAgentMCPSupervisor


async def test_supervisor():
    """Test the MCP supervisor agent."""
    print("=" * 60)
    print("🧪 Testing MCP Supervisor Agent")
    print("=" * 60)
    
    try:
        print("\n1️⃣ Initializing supervisor agent...")
        agent = TechnicalDriftAgentMCPSupervisor()
        
        print("   ⏳ Connecting to MCP servers and creating sub-agents...")
        await agent.initialize()
        
        print("\n2️⃣ Testing with sample message...")
        test_message = {
            "session_id": "test_session_001",
            "athlete_id": "test_athlete_001",
            "activity": "gymnastics",
            "technique": "back_handspring"
        }
        
        print(f"   📨 Message: {json.dumps(test_message, indent=6)}")
        print("\n   🔄 Processing (this may take a moment)...")
        
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
        
        return result.get("success", False)
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_supervisor())
    sys.exit(0 if success else 1)

