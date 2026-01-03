#!/usr/bin/env python3
"""
Retrieval Queue Worker - Redis Queue Worker for Form Correction Retrieval Agent

Listens to Redis queue 'retrievalQueue' and processes messages by running the
actual FormCorrectionRetrievalAgent to recalculate insights.
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

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Try to import BSON for ObjectId
try:
    from bson import ObjectId
    BSON_AVAILABLE = True
except ImportError:
    BSON_AVAILABLE = False
    ObjectId = None

# Import the actual retrieval agent
from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RetrievalQueueWorker:
    """
    Redis queue worker that processes messages using FormCorrectionRetrievalAgent.
    """
    
    def __init__(self, queue_name: str = "retrievalQueue"):
        """
        Initialize retrieval queue worker.
        
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
    
    def process_message(self, message: Dict[str, Any]) -> bool:
        """
        Process a message from the queue using the actual retrieval agent.
        
        Args:
            message: Message dictionary from queue
            
        Returns:
            True if processed successfully, False otherwise
        """
        if not self.retrieval_agent:
            logger.error("❌ Retrieval agent not available")
            return False
        
        try:
            event_type = message.get("event_type")
            session_id = message.get("session_id")
            
            if event_type != "video_call_ended":
                logger.warning(f"⚠️  Unknown event type: {event_type}")
                return False
            
            logger.info(f"📨 Processing message for session: {session_id}")
            logger.info(f"   Activity: {message.get('activity')}")
            logger.info(f"   Technique: {message.get('technique')}")
            logger.info(f"   Athlete: {message.get('athlete_name', 'N/A')}")
            
            # Extract filters from message
            activity = message.get("activity")
            technique = message.get("technique")
            athlete_name = message.get("athlete_name")
            
            # Recalculate insights using actual retrieval agent
            success = self.recalculate_insights(
                activity=activity,
                technique=technique,
                athlete_name=athlete_name,
                session_id=session_id,
                message=message  # Pass message for athlete_id
            )
            
            if success:
                logger.info(f"✅ Successfully processed session: {session_id}")
            else:
                logger.error(f"❌ Failed to process session: {session_id}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error processing message: {e}", exc_info=True)
            return False
    
    def recalculate_insights(
        self,
        activity: Optional[str] = None,
        technique: Optional[str] = None,
        athlete_name: Optional[str] = None,
        session_id: Optional[str] = None,
        message: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Recalculate insights using the actual FormCorrectionRetrievalAgent.
        
        Args:
            activity: Filter by activity
            technique: Filter by technique
            athlete_name: Filter by athlete name
            session_id: Session ID (for logging)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"🔄 Recalculating insights...")
            logger.info(f"   Filters: activity={activity}, technique={technique}, athlete={athlete_name}")
            
            # Step 1: Find sessions with form issues
            logger.info("   📊 Step 1: Finding sessions with form issues...")
            sessions = self.retrieval_agent.find_sessions_with_form_issues(
                activity=activity,
                technique=technique,
                min_sessions_per_issue=3
            )
            
            logger.info(f"   ✅ Found {len(sessions)} sessions with form issues")
            
            if len(sessions) == 0:
                logger.info("   ℹ️  No sessions with form issues found. Skipping trend tracking.")
                return True
            
            # Step 2: Track trends (if enough sessions)
            if len(sessions) >= 3:
                logger.info("   📈 Step 2: Tracking trends across sessions...")
                try:
                    trends_result = self.retrieval_agent.track_form_issue_trends(
                        sessions=sessions,
                        activity=activity,
                        technique=technique,
                        min_sessions=3
                    )
                    
                    if trends_result and trends_result.get("trends"):
                        trend_count = trends_result.get("trend_count", 0)
                        logger.info(f"   ✅ Identified {trend_count} trends")
                    else:
                        logger.info("   ℹ️  No trends identified")
                except Exception as e:
                    logger.warning(f"   ⚠️  Trend tracking failed: {e}")
            
            # Step 3: Analyze form patterns (optional)
            logger.info("   🔍 Step 3: Analyzing form patterns...")
            try:
                patterns = self.retrieval_agent.analyze_form_issues_across_sessions(
                    sessions=sessions,
                    activity=activity,
                    technique=technique,
                    min_sessions_per_issue=3
                )
                
                if patterns:
                    issue_types = patterns.get("issue_types", {})
                    logger.info(f"   ✅ Analyzed {len(issue_types)} issue types")
            except Exception as e:
                logger.warning(f"   ⚠️  Pattern analysis failed: {e}")
            
            # Step 4: Monitor flagged insights and track trends from flag timestamp
            logger.info("   📈 Step 4: Monitoring flagged insights...")
            try:
                from monitor_flagged_insights import FlaggedInsightsMonitor
                monitor = FlaggedInsightsMonitor()
                monitoring_results = monitor.monitor_flagged_insights(
                    activity=activity,
                    technique=technique,
                    min_new_sessions=3
                )
                
                if monitoring_results:
                    trends_identified = sum(1 for r in monitoring_results if r.get("trends"))
                    logger.info(f"   ✅ Monitored {len(monitoring_results)} flagged insights, identified {trends_identified} trends")
                else:
                    logger.info("   ℹ️  No flagged insights to monitor")
                
                monitor.close()
            except Exception as e:
                logger.warning(f"   ⚠️  Flagged insights monitoring failed: {e}")
            
            # Step 5: Check baseline eligibility and establish baseline if needed
            logger.info("   📊 Step 5: Checking baseline eligibility...")
            try:
                # Get athlete_id from message first, then try session
                athlete_id = message.get("athlete_id") if message else None
                
                if not athlete_id and session_id:
                    # Try to get athlete_id from session
                    collection = self.retrieval_agent.mongodb.get_sessions_collection()
                    session = collection.find_one({"_id": session_id}) if session_id else None
                    if not session and BSON_AVAILABLE:
                        try:
                            from bson import ObjectId
                            session = collection.find_one({"_id": ObjectId(session_id)})
                        except:
                            pass
                    if session:
                        athlete_id = session.get("athlete_id")
                
                if not athlete_id and athlete_name:
                    # Try to get athlete_id from athlete_name
                    logger.info(f"   🔍 Attempting to find athlete_id from name: {athlete_name}")
                    try:
                        collection = self.retrieval_agent.mongodb.get_sessions_collection()
                        sample_session = collection.find_one({"athlete_name": athlete_name})
                        if sample_session:
                            athlete_id = sample_session.get("athlete_id")
                            if athlete_id:
                                logger.info(f"   ✅ Found athlete_id: {athlete_id}")
                    except Exception as e2:
                        logger.warning(f"   ⚠️  Could not find athlete_id from name: {e2}")
                
                # Check all athletes from sessions found in Step 1 (even if we have athlete_id from message)
                # This ensures we check all athletes that have sessions, not just the message athlete
                if sessions:
                    logger.info(f"   🔍 Checking baseline eligibility for all athletes from Step 1 sessions...")
                    # Get unique athlete_ids from sessions found in Step 1
                    athlete_ids_from_sessions = set()
                    for s in sessions:
                        aid = s.get("athlete_id")
                        if aid:
                            athlete_ids_from_sessions.add(aid)
                    
                    # Also add athlete_id from message if we have one
                    if athlete_id:
                        athlete_ids_from_sessions.add(athlete_id)
                    
                    if athlete_ids_from_sessions:
                        # Check each athlete to see if they meet baseline threshold
                        for candidate_athlete_id in athlete_ids_from_sessions:
                            logger.info(f"   🔍 Checking athlete: {candidate_athlete_id}")
                            baseline = self.retrieval_agent._get_active_baseline(candidate_athlete_id)
                            
                            if not baseline:
                                collection = self.retrieval_agent.mongodb.get_sessions_collection()
                                eligible_count = collection.count_documents({
                                    "athlete_id": candidate_athlete_id,
                                    "capture_confidence_score": {"$gte": 0.7},
                                    "baseline_eligible": True
                                })
                                
                                min_sessions = 8
                                logger.info(f"   📊 Athlete {candidate_athlete_id}: {eligible_count}/{min_sessions} eligible sessions")
                                
                                if eligible_count >= min_sessions:
                                    logger.info(f"   ✅ Threshold met for {candidate_athlete_id}! Establishing baseline...")
                                    baseline_result = self.retrieval_agent.establish_baseline(
                                        athlete_id=candidate_athlete_id,
                                        baseline_type="pre_injury",
                                        min_sessions=min_sessions,
                                        min_confidence_score=0.7
                                    )
                                    
                                    if baseline_result:
                                        logger.info(f"   ✅ Baseline established successfully for {candidate_athlete_id}!")
                                        logger.info(f"      Baseline ID: {baseline_result.get('baseline_id')}")
                                        logger.info(f"      Metrics: {baseline_result.get('metric_count')}")
                                        logger.info(f"      ✅ Drift detection flag created automatically")
                                    else:
                                        logger.warning(f"   ⚠️  Baseline establishment failed for {candidate_athlete_id}")
                                else:
                                    logger.info(f"   ℹ️  Not enough sessions for {candidate_athlete_id} ({eligible_count} < {min_sessions})")
                            else:
                                logger.info(f"   ℹ️  Baseline already exists for {candidate_athlete_id}")
                                
                                # Step 6: Detect technical drift for this athlete (if baseline exists and session matches)
                                if candidate_athlete_id == athlete_id and session_id:
                                    logger.info(f"   🔍 Step 6: Detecting technical drift for {candidate_athlete_id}...")
                                    try:
                                        baseline = self.retrieval_agent._get_active_baseline(candidate_athlete_id)
                                        drift_flag = self.retrieval_agent._get_drift_detection_flag(candidate_athlete_id)
                                        
                                        if baseline and drift_flag and drift_flag.get("drift_detection_enabled"):
                                            # Check if drift detection should be active (past start date)
                                            start_date = drift_flag.get("drift_detection_start_date")
                                            if start_date:
                                                from datetime import datetime
                                                if isinstance(start_date, str):
                                                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                                                if datetime.utcnow() < start_date:
                                                    logger.info(f"   ℹ️  Drift detection not yet active (starts {start_date})")
                                                else:
                                                    # Run drift detection on the new session
                                                    logger.info(f"   🔍 Running drift detection on session: {session_id}")
                                                    drift_result = self.retrieval_agent.detect_technical_drift(
                                                        athlete_id=candidate_athlete_id,
                                                        session_id=session_id,
                                                        analyze_multiple_sessions=False,
                                                        drift_threshold=2.0
                                                    )
                                                    
                                                    if drift_result:
                                                        logger.info(f"   ✅ Drift detected for {candidate_athlete_id}!")
                                                        logger.info(f"      Metrics with drift: {drift_result.get('drift_count', 0)}")
                                                        logger.info(f"      Alert ID: {drift_result.get('alert_id', 'N/A')}")
                                                        # Print detailed drift insights
                                                        self._print_drift_insights(drift_result)
                                                    else:
                                                        logger.info(f"   ℹ️  No drift detected for {candidate_athlete_id} in this session")
                                            else:
                                                # No start date restriction - run immediately
                                                logger.info(f"   🔍 Running drift detection on session: {session_id}")
                                                drift_result = self.retrieval_agent.detect_technical_drift(
                                                    athlete_id=candidate_athlete_id,
                                                    session_id=session_id,
                                                    analyze_multiple_sessions=False,
                                                    drift_threshold=2.0
                                                )
                                                
                                                if drift_result:
                                                    logger.info(f"   ✅ Drift detected for {candidate_athlete_id}!")
                                                    logger.info(f"      Metrics with drift: {drift_result.get('drift_count', 0)}")
                                                    logger.info(f"      Alert ID: {drift_result.get('alert_id', 'N/A')}")
                                                    # Print detailed drift insights
                                                    self._print_drift_insights(drift_result)
                                                else:
                                                    logger.info(f"   ℹ️  No drift detected for {candidate_athlete_id} in this session")
                                        else:
                                            logger.info(f"   ℹ️  Baseline or drift detection flag not available for {candidate_athlete_id}")
                                    except Exception as e:
                                        logger.warning(f"   ⚠️  Drift detection failed for {candidate_athlete_id}: {e}")
                
                # Also check the athlete_id from message if we have one (fallback if no sessions found)
                elif athlete_id:
                    # Check if baseline exists
                    baseline = self.retrieval_agent._get_active_baseline(athlete_id)
                    
                    if not baseline:
                        # No baseline exists - check if we have enough sessions
                        logger.info(f"   🔍 No baseline found for {athlete_id}. Checking session count...")
                        collection = self.retrieval_agent.mongodb.get_sessions_collection()
                        
                        # Count eligible sessions (capture_confidence_score >= 0.7, baseline_eligible = true)
                        eligible_count = collection.count_documents({
                            "athlete_id": athlete_id,
                            "capture_confidence_score": {"$gte": 0.7},
                            "baseline_eligible": True
                        })
                        
                        min_sessions = 8  # Default threshold
                        logger.info(f"   📊 Eligible sessions for {athlete_id}: {eligible_count}/{min_sessions}")
                        
                        if eligible_count >= min_sessions:
                            logger.info(f"   ✅ Threshold met! Establishing baseline...")
                            baseline_result = self.retrieval_agent.establish_baseline(
                                athlete_id=athlete_id,
                                baseline_type="pre_injury",
                                min_sessions=min_sessions,
                                min_confidence_score=0.7
                            )
                            
                            if baseline_result:
                                logger.info(f"   ✅ Baseline established successfully!")
                                logger.info(f"      Baseline ID: {baseline_result.get('baseline_id')}")
                                logger.info(f"      Metrics: {baseline_result.get('metric_count')}")
                                # Drift detection flag is automatically created by establish_baseline
                                logger.info(f"      ✅ Drift detection flag created automatically")
                            else:
                                logger.warning(f"   ⚠️  Baseline establishment failed")
                        else:
                            logger.info(f"   ℹ️  Not enough sessions for baseline ({eligible_count} < {min_sessions})")
                    else:
                        # Baseline exists - check if drift detection flag exists
                        logger.info(f"   ✅ Baseline exists (ID: {str(baseline.get('_id'))[:8]}...)")
                        
                        drift_flag = self.retrieval_agent._get_drift_detection_flag(athlete_id)
                        
                        if not drift_flag:
                            logger.info(f"   ⚠️  Drift detection flag not found. Creating...")
                            # Create drift detection flag
                            baseline_window = baseline.get("baseline_window", {})
                            baseline_end = baseline_window.get("end_date")
                            
                            # Convert baseline_end to datetime if needed
                            if baseline_end:
                                from datetime import datetime
                                if isinstance(baseline_end, str):
                                    baseline_end = datetime.fromisoformat(baseline_end.replace('Z', '+00:00'))
                            
                            # Convert baseline_id to string if needed
                            baseline_id = baseline.get("_id")
                            if hasattr(baseline_id, '__str__') and not isinstance(baseline_id, str):
                                baseline_id = str(baseline_id)
                            elif not isinstance(baseline_id, str):
                                baseline_id = str(baseline_id)
                            
                            self.retrieval_agent._create_drift_detection_flag(
                                athlete_id=athlete_id,
                                baseline_id=baseline_id,
                                start_date=baseline_end
                            )
                            logger.info(f"   ✅ Drift detection flag created")
                        else:
                            drift_enabled = drift_flag.get("drift_detection_enabled", False)
                            if drift_enabled:
                                logger.info(f"   ✅ Drift detection is enabled")
                            else:
                                logger.info(f"   ℹ️  Drift detection flag exists but is disabled")
                    
                    # Step 6: Detect technical drift (if baseline and flag exist)
                    logger.info("   🔍 Step 6: Detecting technical drift...")
                    try:
                        # Re-check baseline and flag
                        baseline = self.retrieval_agent._get_active_baseline(athlete_id)
                        drift_flag = self.retrieval_agent._get_drift_detection_flag(athlete_id)
                        
                        if baseline and drift_flag and drift_flag.get("drift_detection_enabled"):
                            # Check if drift detection should be active (past start date)
                            start_date = drift_flag.get("drift_detection_start_date")
                            if start_date:
                                from datetime import datetime
                                if isinstance(start_date, str):
                                    start_date = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                                if datetime.utcnow() < start_date:
                                    logger.info(f"   ℹ️  Drift detection not yet active (starts {start_date})")
                                else:
                                    # Run drift detection on the new session
                                    logger.info(f"   🔍 Running drift detection on session: {session_id}")
                                    drift_result = self.retrieval_agent.detect_technical_drift(
                                        athlete_id=athlete_id,
                                        session_id=session_id,
                                        analyze_multiple_sessions=False,  # Single session for new session
                                        drift_threshold=2.0
                                    )
                                    
                                    if drift_result:
                                        logger.info(f"   ✅ Drift detected!")
                                        logger.info(f"      Metrics with drift: {drift_result.get('drift_count')}")
                                        logger.info(f"      Alert ID: {drift_result.get('alert_id')}")
                                    else:
                                        logger.info("   ℹ️  No drift detected in this session")
                            else:
                                # No start date restriction - run immediately
                                logger.info(f"   🔍 Running drift detection on session: {session_id}")
                                drift_result = self.retrieval_agent.detect_technical_drift(
                                    athlete_id=athlete_id,
                                    session_id=session_id,
                                    analyze_multiple_sessions=False,
                                    drift_threshold=2.0
                                )
                                
                                if drift_result:
                                    logger.info(f"   ✅ Drift detected!")
                                    logger.info(f"      Metrics with drift: {drift_result.get('drift_count')}")
                                    logger.info(f"      Alert ID: {drift_result.get('alert_id')}")
                                else:
                                    logger.info("   ℹ️  No drift detected in this session")
                        else:
                            logger.info("   ℹ️  Baseline or drift detection flag not available")
                    except Exception as e:
                        logger.warning(f"   ⚠️  Drift detection failed: {e}")
                else:
                    logger.info("   ℹ️  No athlete_id available for baseline/drift detection")
            except Exception as e:
                logger.warning(f"   ⚠️  Baseline check failed: {e}")
            
            logger.info("✅ Insights recalculation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error recalculating insights: {e}", exc_info=True)
            return False
    
    def listen_to_queue(self, timeout: int = 1, max_messages: Optional[int] = None) -> None:
        """
        Listen to Redis queue and process messages.
        
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
                        message = json.loads(message_json)
                        message_count += 1
                        
                        logger.info(f"\n{'='*60}")
                        logger.info(f"📨 Message #{message_count} received")
                        logger.info(f"{'='*60}")
                        
                        # Process message
                        start_time = time.time()
                        success = self.process_message(message)
                        processing_time = time.time() - start_time
                        
                        if success:
                            logger.info(f"✅ Message #{message_count} processed successfully ({processing_time:.2f}s)")
                        else:
                            logger.error(f"❌ Message #{message_count} processing failed ({processing_time:.2f}s)")
                        
                        logger.info(f"{'='*60}\n")
                        
                        # Check if we've reached max messages
                        if max_messages and message_count >= max_messages:
                            logger.info(f"✅ Processed {message_count} messages. Stopping.")
                            break
                    
                    except json.JSONDecodeError as e:
                        logger.error(f"❌ Failed to parse message JSON: {e}")
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
        logger.info("🛑 Stopping queue worker...")
    
    def _print_drift_insights(self, drift_result: Dict[str, Any]) -> None:
        """
        Print detected drift insights in a formatted way.
        
        Args:
            drift_result: Drift detection result dictionary
        """
        try:
            athlete_id = drift_result.get("athlete_id", "N/A")
            session_count = drift_result.get("session_count", 0)
            insights = drift_result.get("insights", [])
            summary = drift_result.get("summary", {})
            
            logger.info(f"\n{'='*70}")
            logger.info(f"🔍 DRIFT DETECTION RESULTS")
            logger.info(f"{'='*70}")
            logger.info(f"Athlete ID: {athlete_id}")
            logger.info(f"Sessions Analyzed: {session_count}")
            logger.info(f"Total Insights: {summary.get('total_insights', 0)}")
            logger.info(f"  - Worsening: {summary.get('worsening_insights', 0)}")
            logger.info(f"  - Improving: {summary.get('improving_insights', 0)}")
            logger.info(f"  - Unchanged: {summary.get('unchanged_insights', 0)}")
            logger.info(f"{'='*70}\n")
            
            if not insights:
                logger.info("   ℹ️  No drift insights found")
                return
            
            for idx, insight in enumerate(insights, 1):
                metric_key = insight.get("metric_key", "N/A")
                description = insight.get("insight_description", metric_key)
                trend = insight.get("trend", "N/A")
                overall_severity = insight.get("overall_severity", "N/A")
                baseline_value = insight.get("baseline_value", 0)
                deviations = insight.get("deviations", [])
                change_in_deviation = insight.get("change_in_deviation", 0)
                
                logger.info(f"\n📊 Insight #{idx}: {description}")
                logger.info(f"   Metric: {metric_key}")
                logger.info(f"   Trend: {trend.upper()}")
                logger.info(f"   Overall Severity: {overall_severity.upper()}")
                logger.info(f"   Baseline Value: {baseline_value:.3f}")
                logger.info(f"   Change in Deviation: {change_in_deviation:+.2f}σ")
                logger.info(f"   Sessions with Drift: {len(deviations)}")
                
                if deviations:
                    logger.info(f"\n   Deviations by Session:")
                    for dev_idx, deviation in enumerate(deviations, 1):
                        session_id_short = deviation.get("session_id", "N/A")[:8] + "..."
                        timestamp = deviation.get("session_timestamp", "N/A")
                        current_value = deviation.get("current_value", 0)
                        z_score = deviation.get("z_score", 0)
                        deviation_percent = deviation.get("deviation_percent", 0)
                        severity = deviation.get("severity", "N/A")
                        direction = deviation.get("direction", "N/A")
                        
                        logger.info(f"      Session {dev_idx}: {session_id_short}")
                        logger.info(f"         Timestamp: {timestamp}")
                        logger.info(f"         Value: {current_value:.3f} (baseline: {baseline_value:.3f})")
                        logger.info(f"         Z-score: {z_score:+.2f}σ")
                        logger.info(f"         Deviation: {deviation_percent:+.1f}%")
                        logger.info(f"         Severity: {severity.upper()}")
                        logger.info(f"         Direction: {direction.upper()}")
                
                logger.info(f"   {'-'*60}")
            
            logger.info(f"\n{'='*70}\n")
            
        except Exception as e:
            logger.error(f"❌ Error printing drift insights: {e}", exc_info=True)
    
    def cleanup(self):
        """Clean up resources."""
        if self.retrieval_agent:
            try:
                self.retrieval_agent.close()
                logger.info("✅ Retrieval agent closed")
            except Exception as e:
                logger.warning(f"⚠️  Error closing retrieval agent: {e}")
    
    def close(self):
        """Close all connections."""
        self.stop()
        self.cleanup()
        if self.redis_client:
            self.redis_client.close()
            logger.info("✅ Redis connection closed")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Retrieval Queue Worker - Process messages from Redis queue using FormCorrectionRetrievalAgent"
    )
    parser.add_argument(
        "--queue",
        type=str,
        default="retrievalQueue",
        help="Redis queue name (default: retrievalQueue)"
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
    
    worker = RetrievalQueueWorker(queue_name=args.queue)
    
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

