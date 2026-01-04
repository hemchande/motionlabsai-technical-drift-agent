"""
Example demonstrating how sub-agents save memory in supervisor.

This shows the bidirectional memory flow:
1. Supervisor passes context to sub-agents via ToolRuntime
2. Sub-agents return values that automatically become supervisor's memory
"""
import json
from supervisor_agent import TechnicalDriftSupervisor

def demonstrate_memory_flow():
    """
    Demonstrates how memory flows between supervisor and sub-agents.
    """
    supervisor = TechnicalDriftSupervisor()
    
    # Example: Process a video session
    message = {
        "session_id": "session_123",
        "athlete_id": "athlete_001",
        "activity": "gymnastics",
        "technique": "back_handspring"
    }
    
    print("=" * 60)
    print("MEMORY FLOW DEMONSTRATION")
    print("=" * 60)
    
    print("\n1. User Request:")
    print(f"   {json.dumps(message, indent=2)}")
    print("\n   → Supervisor stores as 'human' message in memory")
    
    print("\n2. Supervisor → manage_mongodb('Query sessions...')")
    print("   → ToolRuntime provides context (if available)")
    print("   → MongoDB Agent executes")
    print("   → Returns: {'success': True, 'count': 10, 'sessions': [...]}")
    print("   → Supervisor automatically stores as 'tool' message ✅")
    
    print("\n3. Supervisor → manage_retrieval('Extract insights...')")
    print("   → ToolRuntime provides saved memory:")
    print("     - Previous tool result: {'success': True, 'count': 10, ...}")
    print("   → Retrieval Agent receives context + request")
    print("   → Returns: {'success': True, 'insights_extracted': 5, ...}")
    print("   → Supervisor automatically stores as 'tool' message ✅")
    
    print("\n4. Supervisor Memory Now Contains:")
    print("   - 'human': Original user request")
    print("   - 'tool': MongoDB query results")
    print("   - 'tool': Retrieval extraction results")
    print("   → All accessible to future sub-agents via ToolRuntime")
    
    print("\n" + "=" * 60)
    print("KEY POINTS:")
    print("=" * 60)
    print("✓ Sub-agent returns are AUTOMATICALLY saved in supervisor memory")
    print("✓ Returns become 'tool' type messages")
    print("✓ Future sub-agents can access via ToolRuntime")
    print("✓ Use structured JSON returns for better parsing")
    print("✓ Include key data points for future operations")
    
    # Uncomment to actually run:
    # result = supervisor.process_video_session_message(message)
    # print(f"\nResult: {result}")


def show_structured_returns():
    """
    Shows examples of good vs poor return formats from sub-agents.
    """
    print("\n" + "=" * 60)
    print("STRUCTURED RETURN EXAMPLES")
    print("=" * 60)
    
    print("\n✅ GOOD Return Format (MongoDB Agent):")
    good_return = {
        "success": True,
        "count": 10,
        "athlete_id": "athlete_001",
        "activity": "gymnastics",
        "sessions": [
            {
                "session_id": "sess_1",
                "timestamp": "2024-01-15T10:00:00Z",
                "metrics": {"height": 1.5, "knee_bend": 0.3}
            }
        ]
    }
    print(json.dumps(good_return, indent=2))
    print("\n   Benefits:")
    print("   - Clear success/failure indicator")
    print("   - Includes count for validation")
    print("   - Contains key identifiers (athlete_id, activity)")
    print("   - Structured data for future operations")
    
    print("\n❌ POOR Return Format:")
    poor_returns = [
        "Done",  # Too vague
        "10",    # Missing context
        "Found some sessions and extracted some insights"  # No structure
    ]
    for i, poor in enumerate(poor_returns, 1):
        print(f"   {i}. '{poor}'")
    print("\n   Problems:")
    print("   - No structure for parsing")
    print("   - Missing key information")
    print("   - Cannot be used by future sub-agents")
    
    print("\n✅ GOOD Return Format (Retrieval Agent):")
    good_retrieval_return = {
        "success": True,
        "insights_extracted": 5,
        "session_id": "sess_123",
        "athlete_id": "athlete_001",
        "insights": [
            {
                "insight": "Insufficient height",
                "severity": "moderate",
                "session_count": 4
            },
            {
                "insight": "Landing knee bend",
                "severity": "minor",
                "session_count": 3
            }
        ]
    }
    print(json.dumps(good_retrieval_return, indent=2))
    print("\n   Benefits:")
    print("   - Includes all relevant identifiers")
    print("   - Structured insights array")
    print("   - Severity and counts for decision-making")
    print("   - Can be parsed and used by supervisor")


if __name__ == "__main__":
    demonstrate_memory_flow()
    show_structured_returns()


