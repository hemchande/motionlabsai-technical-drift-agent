#!/usr/bin/env python3
"""
Coach Follow-Up Worker - Redis Queue Worker for Follow-Up Actions

Listens to 'coach_followup_queue' and processes coach follow-up actions,
updating MongoDB and triggering monitoring if needed.
"""

import json
import logging
import os
import sys
import time
import signal
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CoachFollowUpWorker:
    """
    Redis queue worker that processes coach follow-up actions.
    """
    
    def __init__(self, queue_name: str = "coach_followup_queue"):
        """
        Initialize coach follow-up worker.
        
        Args:
            queue_name: Name of Redis queue to listen to
        """
        self.queue_name = queue_name
        self.redis_client = None
        self.running = False
        self.retrieval_agent = None
        
        if not REDIS_AVAILABLE:
            logger.error("❌ Redis not available. Install with: pip install redis")
            return
        
        try:
            redis_host = os.getenv("REDIS_HOST", "localhost")
            redis_port = int(os.getenv("REDIS_PORT", 6379))
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True
            )
            # Test connection
            self.redis_client.ping()
            logger.info(f"✅ Connected to Redis at {redis_host}:{redis_port}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            self.redis_client = None
        
        # Initialize retrieval agent
        try:
            logger.info("🔄 Initializing FormCorrectionRetrievalAgent...")
            self.retrieval_agent = FormCorrectionRetrievalAgent()
            logger.info("✅ FormCorrectionRetrievalAgent initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize retrieval agent: {e}", exc_info=True)
            self.retrieval_agent = None
    
    def process_followup(self, followup_data: Dict[str, Any]) -> bool:
        """
        Process a coach follow-up action from the queue.
        
        Args:
            followup_data: Follow-up data dictionary from queue
            
        Returns:
            True if processed successfully, False otherwise
        """
        if not self.retrieval_agent:
            logger.error("❌ Retrieval agent not available")
            return False
        
        try:
            alert_id = followup_data.get("alert_id")
            metric_key = followup_data.get("metric_key")
            coach_follow_up = followup_data.get("coach_follow_up")
            athlete_id = followup_data.get("athlete_id")
            
            logger.info(f"📨 Processing coach follow-up:")
            logger.info(f"   Alert ID: {alert_id}")
            logger.info(f"   Metric: {metric_key}")
            logger.info(f"   Action: {coach_follow_up}")
            logger.info(f"   Athlete: {athlete_id}")
            
            # Update alert in MongoDB (already done, but verify)
            success = self.retrieval_agent.update_drift_alert_coach_follow_up(
                alert_id=alert_id,
                metric_key=metric_key,
                coach_follow_up=coach_follow_up
            )
            
            if not success:
                logger.warning(f"⚠️  Failed to update alert in MongoDB")
                return False
            
            # If action is "Monitor", start tracking trend
            if coach_follow_up == "Monitor":
                logger.info(f"   📈 Starting trend tracking for {metric_key}...")
                try:
                    trend = self.retrieval_agent.track_monitored_drift_insights(
                        athlete_id=athlete_id,
                        metric_key=metric_key
                    )
                    
                    if trend:
                        logger.info(f"   ✅ Trend tracked: {trend.get('trend')} ({trend.get('change_percent'):.1f}% change)")
                    else:
                        logger.info(f"   ℹ️  Trend tracking returned None (may need more sessions)")
                except Exception as e:
                    logger.warning(f"   ⚠️  Trend tracking failed: {e}")
            
            logger.info(f"✅ Successfully processed follow-up: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing follow-up: {e}", exc_info=True)
            return False
    
    def listen_to_queue(self, timeout: int = 1, max_messages: Optional[int] = None) -> None:
        """
        Listen to Redis queue and process follow-up actions.
        
        Args:
            timeout: Blocking timeout in seconds (0 = no timeout, blocks forever)
            max_messages: Maximum number of messages to process (None = unlimited)
        """
        if not self.redis_client:
            logger.error("❌ Redis client not available")
            return
        
        if not self.retrieval_agent:
            logger.error("❌ Retrieval agent not available")
            return
        
        self.running = True
        
        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info("\n🛑 Received interrupt signal. Shutting down gracefully...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        logger.info(f"👂 Listening to queue: {self.queue_name}")
        logger.info(f"   Timeout: {timeout}s")
        logger.info(f"   Max messages: {max_messages or 'unlimited'}")
        logger.info("   Press Ctrl+C to stop\n")
        
        message_count = 0
        
        try:
            while self.running:
                try:
                    # Blocking pop from queue (BRPOP blocks until message available)
                    result = self.redis_client.brpop(self.queue_name, timeout=timeout)
                    
                    if result is None:
                        # Timeout - no message received
                        continue
                    
                    queue_name, message_json = result
                    
                    # Parse message
                    try:
                        followup_data = json.loads(message_json)
                        message_count += 1
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"📨 Follow-up #{message_count} received")
                        logger.info(f"{'='*60}")
                        
                        # Process follow-up
                        start_time = time.time()
                        success = self.process_followup(followup_data)
                        processing_time = time.time() - start_time
                        
                        if success:
                            logger.info(f"✅ Follow-up #{message_count} processed successfully ({processing_time:.2f}s)")
                        else:
                            logger.error(f"❌ Follow-up #{message_count} processing failed ({processing_time:.2f}s)")
                        
                        logger.info(f"{'='*60}\n")
                        
                        # Check if we've reached max messages
                        if max_messages and message_count >= max_messages:
                            logger.info(f"✅ Processed {message_count} follow-ups. Stopping.")
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse follow-up JSON: {e}")
                        logger.error(f"   Message: {message_json[:100]}")
                        continue
                
                except KeyboardInterrupt:
                    logger.info("\n🛑 Interrupted by user")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in queue listener: {e}", exc_info=True)
                    time.sleep(1)  # Wait before retrying
        
        finally:
            self.running = False
            logger.info("👋 Stopped listening to queue")
            self.cleanup()
    
    def stop(self):
        """Stop listening to queue."""
        self.running = False
        logger.info("🛑 Stopping follow-up worker...")
    
    def cleanup(self):
        """Clean up resources."""
        if self.retrieval_agent:
            try:
                self.retrieval_agent.close()
                logger.info("✅ Retrieval agent closed")
            except Exception as e:
                logger.warning(f"⚠️  Error closing retrieval agent: {e}")
        
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")
    
    def close(self):
        """Close all connections."""
        self.stop()
        self.cleanup()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Coach Follow-Up Worker - Process follow-up actions from Redis queue"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="coach_followup_queue",
        help="Redis queue name (default: coach_followup_queue)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1,
        help="Blocking timeout in seconds (0 = no timeout)"
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Maximum number of messages to process"
    )
    
    args = parser.parse_args()
    
    worker = CoachFollowUpWorker(queue_name=args.queue)
    
    if not worker.redis_client:
        print("❌ Redis not available. Please install and start Redis.")
        return 1
    
    if not worker.retrieval_agent:
        print("❌ Retrieval agent not available. Check initialization.")
        return 1
    
    try:
        worker.listen_to_queue(timeout=args.timeout, max_messages=args.max_messages)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    finally:
        worker.close()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

