#!/usr/bin/env python3
"""
Drift Alert Worker - Redis Queue Worker for Alert Delivery

Listens to 'drift_alerts_queue' and broadcasts alerts via WebSocket to PT/Instructor clients.
"""

import json
import logging
import os
import sys
import time
import signal
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Try to import WebSocket
try:
    import websockets
    from websockets.server import serve
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    websockets = None

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DriftAlertWorker:
    """
    Redis queue worker that processes drift alerts and broadcasts via WebSocket.
    """
    
    def __init__(self, queue_name: str = "drift_alerts_queue", websocket_port: int = 8766):
        """
        Initialize drift alert worker.
        
        Args:
            queue_name: Name of Redis queue to listen to
            websocket_port: Port for WebSocket server
        """
        self.queue_name = queue_name
        self.websocket_port = websocket_port
        self.redis_client = None
        self.running = False
        self.websocket_clients = set()
        
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
    
    async def websocket_handler(self, websocket, path):
        """
        Handle WebSocket connections for alert broadcasting.
        
        Args:
            websocket: WebSocket connection
            path: Connection path
        """
        logger.info(f"🔌 New WebSocket connection from {websocket.remote_address}")
        self.websocket_clients.add(websocket)
        
        try:
            # Send welcome message
            await websocket.send(json.dumps({
                "type": "connection",
                "status": "connected",
                "message": "Connected to drift alert server"
            }))
            
            # Keep connection alive
            async for message in websocket:
                # Echo back or handle client messages
                try:
                    data = json.loads(message)
                    if data.get("type") == "ping":
                        await websocket.send(json.dumps({"type": "pong"}))
                except:
                    pass
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"🔌 WebSocket connection closed: {websocket.remote_address}")
        except Exception as e:
            logger.error(f"❌ WebSocket error: {e}", exc_info=True)
        finally:
            self.websocket_clients.discard(websocket)
    
    async def broadcast_alert(self, alert_data: Dict[str, Any]) -> None:
        """
        Broadcast alert to all connected WebSocket clients.
        
        Args:
            alert_data: Alert data dictionary
        """
        if not self.websocket_clients:
            logger.debug("No WebSocket clients connected")
            return
        
        message = {
            "type": "drift_alert",
            "alert_id": alert_data.get("alert_id"),
            "athlete_id": alert_data.get("athlete_id"),
            "session_id": alert_data.get("session_id"),
            "severity": alert_data.get("severity", "moderate"),
            "drift_metrics": alert_data.get("drift_metrics", {}),
            "drift_count": alert_data.get("drift_count", 0),
            "alert_confidence": alert_data.get("alert_confidence", 0.92),
            "timestamp": alert_data.get("created_at", datetime.utcnow().isoformat())
        }
        
        message_json = json.dumps(message)
        disconnected = set()
        
        for client in self.websocket_clients:
            try:
                await client.send(message_json)
                logger.debug(f"✅ Broadcasted alert to client: {client.remote_address}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to send to client {client.remote_address}: {e}")
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.websocket_clients.discard(client)
        
        if self.websocket_clients:
            logger.info(f"✅ Broadcasted alert to {len(self.websocket_clients)} client(s)")
    
    def process_alert(self, alert_data: Dict[str, Any]) -> bool:
        """
        Process a drift alert from the queue.
        
        Args:
            alert_data: Alert data dictionary from queue
            
        Returns:
            True if processed successfully, False otherwise
        """
        try:
            alert_id = alert_data.get("alert_id", "unknown")
            athlete_id = alert_data.get("athlete_id", "unknown")
            
            logger.info(f"📨 Processing drift alert: {alert_id}")
            logger.info(f"   Athlete: {athlete_id}")
            logger.info(f"   Severity: {alert_data.get('severity', 'unknown')}")
            logger.info(f"   Drift metrics: {alert_data.get('drift_count', 0)}")
            
            # Broadcast via WebSocket (async)
            if WEBSOCKETS_AVAILABLE and self.websocket_clients:
                # Run async broadcast in event loop
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                loop.run_until_complete(self.broadcast_alert(alert_data))
            
            logger.info(f"✅ Successfully processed alert: {alert_id}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error processing alert: {e}", exc_info=True)
            return False
    
    async def start_websocket_server(self):
        """Start WebSocket server for alert broadcasting."""
        if not WEBSOCKETS_AVAILABLE:
            logger.warning("⚠️  WebSockets not available. Alerts will not be broadcast.")
            return
        
        logger.info(f"🚀 Starting WebSocket server on ws://localhost:{self.websocket_port}")
        
        async with serve(self.websocket_handler, "localhost", self.websocket_port):
            logger.info(f"✅ WebSocket server running on ws://localhost:{self.websocket_port}")
            await asyncio.Future()  # Run forever
    
    def listen_to_queue(self, timeout: int = 1, max_messages: Optional[int] = None) -> None:
        """
        Listen to Redis queue and process alerts.
        
        Args:
            timeout: Blocking timeout in seconds (0 = no timeout, blocks forever)
            max_messages: Maximum number of messages to process (None = unlimited)
        """
        if not self.redis_client:
            logger.error("❌ Redis client not available")
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
        logger.info(f"   WebSocket port: {self.websocket_port}")
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
                        alert_data = json.loads(message_json)
                        message_count += 1
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"📨 Alert #{message_count} received")
                        logger.info(f"{'='*60}")
                        
                        # Process alert
                        start_time = time.time()
                        success = self.process_alert(alert_data)
                        processing_time = time.time() - start_time
                        
                        if success:
                            logger.info(f"✅ Alert #{message_count} processed successfully ({processing_time:.2f}s)")
                        else:
                            logger.error(f"❌ Alert #{message_count} processing failed ({processing_time:.2f}s)")
                        
                        logger.info(f"{'='*60}\n")
                        
                        # Check if we've reached max messages
                        if max_messages and message_count >= max_messages:
                            logger.info(f"✅ Processed {message_count} alerts. Stopping.")
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse alert JSON: {e}")
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
        logger.info("🛑 Stopping alert worker...")
    
    def cleanup(self):
        """Clean up resources."""
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Drift Alert Worker - Process alerts from Redis queue and broadcast via WebSocket"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="drift_alerts_queue",
        help="Redis queue name (default: drift_alerts_queue)"
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
    parser.add_argument(
        "--websocket-port",
        type=int,
        default=8766,
        help="WebSocket server port (default: 8766)"
    )
    
    args = parser.parse_args()
    
    worker = DriftAlertWorker(queue_name=args.queue, websocket_port=args.websocket_port)
    
    if not worker.redis_client:
        print("❌ Redis not available. Please install and start Redis.")
        return 1
    
    try:
        worker.listen_to_queue(timeout=args.timeout, max_messages=args.max_messages)
    except KeyboardInterrupt:
        logger.info("\n🛑 Shutting down...")
    finally:
        worker.cleanup()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

