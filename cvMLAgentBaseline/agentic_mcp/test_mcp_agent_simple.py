#!/usr/bin/env python3
"""
Simple test script for MCP Agent - tests initialization without database.

This test verifies:
1. MCP client can connect to servers
2. Tools are loaded correctly
3. Agent can be created
"""
import sys
import asyncio
import json
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Set minimal environment variables for testing
os.environ.setdefault("MONGODB_URI", "mongodb://localhost:27017/test")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("MONGODB_DATABASE", "test_db")

from agent_mcp import TechnicalDriftAgentMCP


async def test_agent_initialization():
    """Test agent initialization and tool loading."""
    print("=" * 60)
    print("🧪 Testing MCP Agent Initialization")
    print("=" * 60)
    
    try:
        print("\n1️⃣ Creating agent...")
        agent = TechnicalDriftAgentMCP()
        print("   ✅ Agent created")
        
        print("\n2️⃣ Initializing agent (connecting to MCP servers)...")
        print("   ⏳ This may take a moment...")
        
        try:
            await agent.initialize()
            
            print(f"\n   ✅ Agent initialized successfully!")
            print(f"   📊 Loaded {len(agent.tools)} tools from MCP servers")
            
            if agent.tools:
                print(f"\n   📋 Available tools:")
                for i, tool in enumerate(agent.tools[:20], 1):
                    print(f"      {i}. {tool.name}")
                    if hasattr(tool, 'description') and tool.description:
                        desc = tool.description[:60] + "..." if len(tool.description) > 60 else tool.description
                        print(f"         {desc}")
                if len(agent.tools) > 20:
                    print(f"      ... and {len(agent.tools) - 20} more tools")
            
            print(f"\n   ✅ Agent object: {type(agent.agent)}")
            
            print("\n3️⃣ Testing agent structure...")
            if agent.agent:
                print("   ✅ Agent Runnable created")
            else:
                print("   ❌ Agent Runnable not created")
                return False
            
            print("\n" + "=" * 60)
            print("✅ All initialization tests passed!")
            print("=" * 60)
            print("\n📝 Note: Full pipeline test requires:")
            print("   - Valid MongoDB connection")
            print("   - Valid OpenAI API key")
            print("   - Valid Redis connection (optional)")
            
            await agent.close()
            return True
            
        except Exception as init_error:
            print(f"\n   ❌ Initialization failed: {init_error}")
            print(f"   Error type: {type(init_error).__name__}")
            import traceback
            print("\n   Full traceback:")
            traceback.print_exc()
            return False
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_agent_initialization())
    sys.exit(0 if success else 1)

