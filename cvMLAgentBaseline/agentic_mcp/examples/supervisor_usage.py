#!/usr/bin/env python3
"""
Supervisor pattern example for Technical Drift Detection.

Demonstrates the supervisor coordinating sub-agents.
"""
import sys
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from supervisor_agent import TechnicalDriftSupervisor


def main():
    """Run supervisor example."""
    print("🚀 Initializing Technical Drift Supervisor...")
    supervisor = TechnicalDriftSupervisor()
    
    print(f"✅ Supervisor initialized with {len(supervisor.sub_agents)} sub-agents")
    print(f"   Sub-agents: {', '.join(supervisor.sub_agents.keys())}")
    
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
    
    result = supervisor.process_video_session_message(test_message)
    
    print(f"\n📊 Result:")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()


