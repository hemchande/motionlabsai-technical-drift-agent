#!/usr/bin/env python3
"""
Test script to save JSON and text results to MongoDB without running the full agent.
This simulates what save_call_outputs does but can be run standalone.
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Add parent directory to path to import MongoDBService
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_session_document(
    session_id: str,
    metrics_data: Dict[str, Any],
    transcript_text: Optional[str] = None,
    metrics_file: Optional[Path] = None,
    transcript_file: Optional[Path] = None,
    call_id: Optional[str] = None,
    cloudflare_stream_url: Optional[str] = None,
    cloudflare_video_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Prepare MongoDB session document from metrics data and transcript.
    
    Args:
        session_id: Session identifier
        metrics_data: Metrics JSON data
        transcript_text: Transcript text content (optional)
        metrics_file: Path to metrics JSON file (optional)
        transcript_file: Path to transcript file (optional)
        call_id: Call ID (optional)
        cloudflare_stream_url: Cloudflare Stream URL (optional)
        cloudflare_video_info: Cloudflare video info (optional)
    
    Returns:
        Session document dictionary
    """
    # Extract ACL risk data
    acl_flagged = metrics_data.get("acl_flagged_timesteps", [])
    metrics_dict = metrics_data.get("metrics", {})
    
    # Extract ACL risk metrics
    acl_risk_score = metrics_dict.get("acl_tear_risk_score", 0.0)
    acl_risk_level = metrics_dict.get("acl_risk_level", "MINIMAL")
    acl_max_valgus = metrics_dict.get("acl_max_valgus_angle", 0.0)
    
    # Count risk moments by level
    high_risk_count = len([ts for ts in acl_flagged if ts.get("risk_score", 0.0) >= 0.7])
    moderate_risk_count = len([ts for ts in acl_flagged if 0.4 <= ts.get("risk_score", 0.0) < 0.7])
    total_risk_count = len(acl_flagged)
    
    # Determine risk flags
    has_risk_acl = (
        acl_risk_score >= 0.4 or
        total_risk_count > 0 or
        acl_risk_level in ["MODERATE", "HIGH"]
    )
    has_high_risk_acl = (
        acl_risk_score >= 0.7 or
        high_risk_count > 0 or
        acl_risk_level == "HIGH"
    )
    
    # Prepare session document
    session_doc = {
        "original_filename": session_id,
        "session_id": session_id,
        "activity": metrics_data.get("activity", "unknown"),
        "technique": metrics_data.get("technique", "unknown"),
        "timestamp": metrics_data.get("timestamp", datetime.utcnow().isoformat()),
        "call_id": call_id,
        
        # Metrics data
        "metrics": metrics_dict,
        "workflow_summary": metrics_data.get("workflow_summary", {}),
        "analysis": metrics_data.get("analysis", {}),
        "insights": metrics_data.get("insights", {}),
        
        # ACL risk data
        "acl_flagged_timesteps": acl_flagged,
        "acl_risk_score": acl_risk_score,
        "acl_risk_level": acl_risk_level,
        "acl_max_valgus_angle": acl_max_valgus,
        "acl_high_risk_count": high_risk_count,
        "acl_moderate_risk_count": moderate_risk_count,
        "acl_total_risk_count": total_risk_count,
        "has_high_risk_acl": has_high_risk_acl,
        "has_risk_acl": has_risk_acl,
        
        # Landing phases
        "landing_phases": metrics_data.get("landing_phases", []),
        
        # Frame metrics count
        "frame_metrics_count": metrics_data.get("frame_metrics_count", 0),
        
        # Transcript
        "transcript": transcript_text,
        "transcript_file": str(transcript_file) if transcript_file else None,
        
        # File paths
        "metrics_file": str(metrics_file) if metrics_file else None,
        
        # Cloudflare Stream info
        "cloudflare_stream_url": cloudflare_stream_url,
        "cloudflare_video_info": cloudflare_video_info,
        
        # Metadata
        "ingested_at": datetime.utcnow().isoformat(),
        "ingestion_source": "cvmlagent_test"
    }
    
    return session_doc

def test_save_to_mongodb(
    metrics_file: Optional[Path] = None,
    transcript_file: Optional[Path] = None,
    session_id: Optional[str] = None,
    call_id: Optional[str] = None
) -> bool:
    """
    Test saving a session to MongoDB.
    
    Args:
        metrics_file: Path to metrics JSON file (optional, will create test data if not provided)
        transcript_file: Path to transcript file (optional)
        session_id: Session ID (optional, will generate if not provided)
        call_id: Call ID (optional)
    
    Returns:
        True if successful, False otherwise
    """
    load_dotenv()
    
    try:
        # Connect to MongoDB
        mongodb = MongoDBService()
        if not mongodb.connect():
            logger.error("❌ Failed to connect to MongoDB")
            return False
        
        logger.info("✅ Connected to MongoDB")
        
        # Load or create test metrics data
        if metrics_file and metrics_file.exists():
            logger.info(f"📄 Loading metrics from: {metrics_file}")
            with open(metrics_file, 'r') as f:
                metrics_data = json.load(f)
            if not session_id:
                session_id = metrics_file.stem.replace("_metrics", "")
        else:
            logger.info("📝 Creating test metrics data...")
            # Create minimal test data
            session_id = session_id or f"test_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            metrics_data = {
                "activity": "gymnastics",
                "technique": "back_handspring",
                "timestamp": datetime.utcnow().isoformat(),
                "call_id": call_id,
                "metrics": {},
                "workflow_summary": {},
                "analysis": {},
                "insights": {"test": True},
                "acl_flagged_timesteps": [],
                "landing_phases": [],
                "frame_metrics_count": 0,
                "cloudflare_stream_url": None,
                "cloudflare_video_info": None
            }
        
        # Load transcript if provided
        transcript_text = None
        if transcript_file and transcript_file.exists():
            logger.info(f"📄 Loading transcript from: {transcript_file}")
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        
        # Prepare session document
        logger.info(f"📋 Preparing session document: {session_id}")
        session_doc = prepare_session_document(
            session_id=session_id,
            metrics_data=metrics_data,
            transcript_text=transcript_text,
            metrics_file=metrics_file,
            transcript_file=transcript_file,
            call_id=call_id,
            cloudflare_stream_url=metrics_data.get("cloudflare_stream_url"),
            cloudflare_video_info=metrics_data.get("cloudflare_video_info")
        )
        
        # Save to MongoDB
        logger.info("💾 Saving to MongoDB...")
        result_id = mongodb.upsert_session_by_filename(session_doc)
        
        if result_id:
            logger.info(f"✅ Successfully saved session to MongoDB: {result_id}")
            
            # Verify it was saved
            saved_session = mongodb.get_session(result_id)
            if saved_session:
                logger.info(f"✅ Verified: Session exists in MongoDB")
                logger.info(f"   Activity: {saved_session.get('activity')}")
                logger.info(f"   Technique: {saved_session.get('technique')}")
                logger.info(f"   Timestamp: {saved_session.get('timestamp')}")
                logger.info(f"   Has transcript: {bool(saved_session.get('transcript'))}")
            return True
        else:
            logger.error("❌ Failed to save session to MongoDB")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error: {e}", exc_info=True)
        return False

def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MongoDB save without running full agent")
    parser.add_argument("--metrics-file", type=str, help="Path to metrics JSON file")
    parser.add_argument("--transcript-file", type=str, help="Path to transcript file")
    parser.add_argument("--session-id", type=str, help="Session ID (auto-generated if not provided)")
    parser.add_argument("--call-id", type=str, help="Call ID")
    parser.add_argument("--use-latest", action="store_true", help="Use latest metrics file from stream_output/")
    
    args = parser.parse_args()
    
    # If --use-latest, find the latest metrics file
    if args.use_latest:
        stream_output_dir = Path(__file__).parent / "stream_output"
        if stream_output_dir.exists():
            metrics_files = list(stream_output_dir.glob("*_metrics.json"))
            if metrics_files:
                latest_file = max(metrics_files, key=lambda p: p.stat().st_mtime)
                args.metrics_file = str(latest_file)
                logger.info(f"📂 Using latest metrics file: {latest_file.name}")
            else:
                logger.error("❌ No metrics files found in stream_output/")
                return 1
        else:
            logger.error("❌ stream_output/ directory not found")
            return 1
    
    # Convert paths to Path objects
    metrics_file = Path(args.metrics_file) if args.metrics_file else None
    transcript_file = Path(args.transcript_file) if args.transcript_file else None
    
    # Run test
    success = test_save_to_mongodb(
        metrics_file=metrics_file,
        transcript_file=transcript_file,
        session_id=args.session_id,
        call_id=args.call_id
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())

