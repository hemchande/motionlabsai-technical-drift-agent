#!/usr/bin/env python3
"""
Full pipeline example using Agentic MCP Agent.
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import TechnicalDriftAgent
import json


def main():
    """Run full pipeline test."""
    parser = argparse.ArgumentParser(description="Run full pipeline with agentic agent")
    parser.add_argument("--athlete-id", required=True, help="Athlete ID")
    parser.add_argument("--session-id", help="Session ID (optional)")
    parser.add_argument("--activity", default="gymnastics", help="Activity type")
    parser.add_argument("--technique", default="back_handspring", help="Technique")
    
    args = parser.parse_args()
    
    print("🚀 Initializing Technical Drift Agent...")
    agent = TechnicalDriftAgent()
    
    # Create message
    message = {
        "session_id": args.session_id or f"session_{args.athlete_id}_001",
        "athlete_id": args.athlete_id,
        "activity": args.activity,
        "technique": args.technique
    }
    
    print(f"\n📨 Processing message:")
    print(f"   Session ID: {message['session_id']}")
    print(f"   Athlete ID: {message['athlete_id']}")
    print(f"   Activity: {message['activity']}")
    print(f"   Technique: {message['technique']}")
    
    print(f"\n🤖 Agent processing...")
    result = agent.process_video_session_message(message)
    
    print(f"\n📊 Result:")
    print(json.dumps(result, indent=2, default=str))
    
    if result.get("success"):
        print(f"\n✅ Pipeline completed successfully!")
    else:
        print(f"\n❌ Pipeline failed: {result.get('error')}")


if __name__ == "__main__":
    main()

