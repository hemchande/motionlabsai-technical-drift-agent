#!/usr/bin/env python3
"""
Run the original non-MCP agentic implementation with mock video agent.

This script:
1. Starts the retrieval queue worker in the background
2. Sends a test message from the mock video agent
3. Shows the processing results
"""
import subprocess
import sys
import time
import json
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from mock_video_agent import MockVideoAgent
from retrieval_queue_worker import RetrievalQueueWorker
import threading


def run_worker_in_background(worker, max_messages=1):
    """Run worker in a separate thread."""
    def worker_thread():
        worker.listen_to_queue(timeout=1, max_messages=max_messages)
    
    thread = threading.Thread(target=worker_thread, daemon=True)
    thread.start()
    return thread


def main():
    """Main entry point."""
    print("=" * 60)
    print("🚀 Running Original Non-MCP Agentic Implementation")
    print("=" * 60)
    
    # Initialize worker
    print("\n1️⃣ Initializing retrieval queue worker...")
    worker = RetrievalQueueWorker(queue_name="retrievalQueue")
    
    if not worker.redis_client:
        print("❌ Redis not available. Please install and start Redis.")
        return 1
    
    if not worker.retrieval_agent:
        print("❌ Retrieval agent not available. Check initialization.")
        return 1
    
    print("   ✅ Worker initialized")
    
    # Initialize mock video agent
    print("\n2️⃣ Initializing mock video agent...")
    video_agent = MockVideoAgent(queue_name="retrievalQueue")
    
    if not video_agent.redis_client:
        print("❌ Redis not available for video agent.")
        return 1
    
    print("   ✅ Video agent initialized")
    
    # Start worker in background
    print("\n3️⃣ Starting worker in background...")
    worker_thread = run_worker_in_background(worker, max_messages=1)
    time.sleep(2)  # Give worker time to start listening
    
    # Send test message
    print("\n4️⃣ Sending test message from mock video agent...")
    session_id = f"test_session_{int(time.time())}"
    athlete_id = "test_athlete_001"
    
    success = video_agent.simulate_video_call_ended(
        session_id=session_id,
        athlete_id=athlete_id,
        activity="gymnastics",
        technique="back_handspring",
        athlete_name="Test Athlete"
    )
    
    if not success:
        print("❌ Failed to send message")
        return 1
    
    print(f"   ✅ Message sent: {session_id}")
    
    # Wait for processing
    print("\n5️⃣ Waiting for worker to process message...")
    print("   (This may take a moment...)\n")
    
    # Wait for worker thread to complete
    worker_thread.join(timeout=60)
    
    if worker_thread.is_alive():
        print("⚠️  Worker still processing (timeout after 60s)")
    else:
        print("✅ Worker completed processing")
    
    # Cleanup
    print("\n6️⃣ Cleaning up...")
    worker.close()
    video_agent.close()
    
    print("\n" + "=" * 60)
    print("✅ Pipeline test completed!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

