#!/usr/bin/env python3
"""
MongoDB Session Data Ingestion

Ingests session data from stream_output/ directory into MongoDB.
Stores metrics JSON files and transcript text files for retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import sys
import os

# Add parent directory to path to import MongoDBService
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SessionDataIngestion:
    """
    Ingests session data from stream_output/ into MongoDB.
    """
    
    def __init__(self, stream_output_dir: Optional[Path] = None):
        """
        Initialize the ingestion service.
        
        Args:
            stream_output_dir: Directory containing session files (default: cvMLAgent/stream_output/)
        """
        if stream_output_dir is None:
            # Default to cvMLAgent/stream_output/
            self.stream_output_dir = Path(__file__).parent / "stream_output"
        else:
            self.stream_output_dir = Path(stream_output_dir)
        
        self.mongodb = MongoDBService()
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        
        logger.info(f"✅ Initialized SessionDataIngestion (output dir: {self.stream_output_dir})")
    
    def ingest_all_sessions(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Ingest all session files from stream_output/ directory.
        
        Args:
            dry_run: If True, only report what would be ingested without actually saving
        
        Returns:
            Dictionary with ingestion statistics
        """
        stats = {
            "total_files": 0,
            "metrics_ingested": 0,
            "transcripts_ingested": 0,
            "sessions_created": 0,
            "sessions_updated": 0,
            "errors": []
        }
        
        # Find all metrics JSON files
        metrics_files = list(self.stream_output_dir.glob("*_metrics.json"))
        stats["total_files"] = len(metrics_files)
        
        logger.info(f"📊 Found {len(metrics_files)} metrics JSON files")
        
        for metrics_file in metrics_files:
            try:
                # Extract session identifier from filename
                # Format: {activity}_{technique}_{call_id}_metrics.json
                session_id = self._extract_session_id(metrics_file)
                
                # Load metrics JSON
                with open(metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                
                # Find corresponding transcript file
                transcript_file = self._find_transcript_file(metrics_file)
                transcript_text = None
                if transcript_file and transcript_file.exists():
                    with open(transcript_file, 'r') as f:
                        transcript_text = f.read()
                    stats["transcripts_ingested"] += 1
                
                # Prepare session document
                session_doc = self._prepare_session_document(
                    session_id=session_id,
                    metrics_file=metrics_file,
                    metrics_data=metrics_data,
                    transcript_file=transcript_file,
                    transcript_text=transcript_text
                )
                
                if not dry_run:
                    # Upsert session (create or update)
                    result_id = self.mongodb.upsert_session_by_filename(session_doc)
                    if result_id:
                        # Check if it was an update or create
                        existing = self.mongodb.get_session(result_id)
                        if existing and existing.get("created_at") == existing.get("updated_at"):
                            stats["sessions_created"] += 1
                        else:
                            stats["sessions_updated"] += 1
                        stats["metrics_ingested"] += 1
                        logger.info(f"✅ Ingested session: {session_id}")
                    else:
                        stats["errors"].append(f"Failed to ingest {session_id}")
                else:
                    stats["metrics_ingested"] += 1
                    logger.info(f"🔍 [DRY RUN] Would ingest: {session_id}")
                
            except Exception as e:
                error_msg = f"Error ingesting {metrics_file.name}: {e}"
                logger.error(f"❌ {error_msg}", exc_info=True)
                stats["errors"].append(error_msg)
        
        logger.info(f"📊 Ingestion complete: {stats['metrics_ingested']} metrics, {stats['transcripts_ingested']} transcripts")
        return stats
    
    def _extract_session_id(self, metrics_file: Path) -> str:
        """
        Extract session identifier from metrics filename.
        
        Args:
            metrics_file: Path to metrics JSON file
        
        Returns:
            Session identifier (call_id or filename stem)
        """
        # Format: {activity}_{technique}_{call_id}_metrics.json
        stem = metrics_file.stem  # Remove .json extension
        if stem.endswith("_metrics"):
            stem = stem[:-8]  # Remove "_metrics" suffix
        
        return stem
    
    def _find_transcript_file(self, metrics_file: Path) -> Optional[Path]:
        """
        Find corresponding transcript file for a metrics file.
        
        Args:
            metrics_file: Path to metrics JSON file
        
        Returns:
            Path to transcript file if found, None otherwise
        """
        # Format: {activity}_{technique}_{call_id}_transcript.txt
        stem = metrics_file.stem
        if stem.endswith("_metrics"):
            transcript_stem = stem[:-8] + "_transcript"  # Replace "_metrics" with "_transcript"
        else:
            transcript_stem = stem + "_transcript"
        
        transcript_file = metrics_file.parent / f"{transcript_stem}.txt"
        
        if transcript_file.exists():
            return transcript_file
        return None
    
    def _prepare_session_document(
        self,
        session_id: str,
        metrics_file: Path,
        metrics_data: Dict[str, Any],
        transcript_file: Optional[Path],
        transcript_text: Optional[str]
    ) -> Dict[str, Any]:
        """
        Prepare MongoDB session document from session data.
        
        Args:
            session_id: Session identifier
            metrics_file: Path to metrics JSON file
            metrics_data: Loaded metrics JSON data
            transcript_file: Path to transcript file (if exists)
            transcript_text: Transcript text content (if exists)
        
        Returns:
            Session document dictionary
        """
        # Extract ACL risk data
        acl_flagged = metrics_data.get("acl_flagged_timesteps", [])
        metrics = metrics_data.get("metrics", {})
        
        # Extract ACL risk metrics (check both metrics dict and top-level)
        acl_risk_score = metrics.get("acl_tear_risk_score") or metrics_data.get("metrics", {}).get("acl_tear_risk_score", 0.0)
        acl_risk_level = metrics.get("acl_risk_level") or metrics_data.get("metrics", {}).get("acl_risk_level", "MINIMAL")
        acl_max_valgus = metrics.get("acl_max_valgus_angle") or metrics_data.get("metrics", {}).get("acl_max_valgus_angle", 0.0)
        
        # Count risk moments by level from flagged timesteps
        high_risk_count = len([ts for ts in acl_flagged if ts.get("risk_score", 0.0) >= 0.7])
        moderate_risk_count = len([ts for ts in acl_flagged if 0.4 <= ts.get("risk_score", 0.0) < 0.7])
        total_risk_count = len(acl_flagged)
        
        # Determine if session has risk (moderate or high)
        # Check both flagged timesteps AND overall risk score
        has_risk_acl = (
            acl_risk_score >= 0.4 or  # Overall risk score is moderate or high
            total_risk_count > 0 or  # Has flagged timesteps
            acl_risk_level in ["MODERATE", "HIGH"]  # Risk level indicates risk
        )
        has_high_risk_acl = (
            acl_risk_score >= 0.7 or  # Overall risk score is high
            high_risk_count > 0 or  # Has high-risk flagged timesteps
            acl_risk_level == "HIGH"  # Risk level is high
        )
        
        # Prepare session document
        session_doc = {
            "original_filename": session_id,  # Used for upsert lookup
            "session_id": session_id,
            "activity": metrics_data.get("activity", "unknown"),
            "technique": metrics_data.get("technique", "unknown"),
            "timestamp": metrics_data.get("timestamp", datetime.utcnow().isoformat()),
            "call_id": self._extract_call_id(session_id),
            
            # Metrics data
            "metrics": metrics_data.get("metrics", {}),
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
            "has_risk_acl": has_risk_acl,  # Moderate or high risk
            
            # Landing phases
            "landing_phases": metrics_data.get("landing_phases", []),
            
            # Frame metrics (store count, not full data to avoid document size limits)
            "frame_metrics_count": metrics_data.get("frame_metrics_count", 0),
            
            # Transcript
            "transcript": transcript_text,
            "transcript_file": str(transcript_file) if transcript_file else None,
            
            # File paths
            "metrics_file": str(metrics_file),
            
            # Metadata
            "ingested_at": datetime.utcnow().isoformat(),
            "ingestion_source": "stream_output"
        }
        
        return session_doc
    
    def _extract_call_id(self, session_id: str) -> Optional[str]:
        """
        Extract call_id from session identifier.
        
        Args:
            session_id: Session identifier string
        
        Returns:
            Call ID if found (UUID format), None otherwise
        """
        # Session ID format: {activity}_{technique}_{call_id}
        parts = session_id.split("_")
        if len(parts) >= 3:
            # Call ID is typically a UUID (last part or combination of last parts)
            # Check if last part looks like a UUID
            last_part = parts[-1]
            if len(last_part) == 36 and last_part.count("-") == 4:
                return last_part
            # Otherwise, try to find UUID pattern in the string
            import re
            uuid_pattern = r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}'
            match = re.search(uuid_pattern, session_id, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongodb:
            self.mongodb.close()


def main():
    """Main function for CLI usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Ingest session data from stream_output/ into MongoDB")
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
        help="Directory containing session files (default: cvMLAgent/stream_output/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run mode - report what would be ingested without actually saving"
    )
    
    args = parser.parse_args()
    
    stream_output_dir = Path(args.dir) if args.dir else None
    
    try:
        ingestion = SessionDataIngestion(stream_output_dir=stream_output_dir)
        stats = ingestion.ingest_all_sessions(dry_run=args.dry_run)
        
        print("\n" + "="*60)
        print("📊 INGESTION SUMMARY")
        print("="*60)
        print(f"Total files found: {stats['total_files']}")
        print(f"Metrics ingested: {stats['metrics_ingested']}")
        print(f"Transcripts ingested: {stats['transcripts_ingested']}")
        print(f"Sessions created: {stats['sessions_created']}")
        print(f"Sessions updated: {stats['sessions_updated']}")
        if stats['errors']:
            print(f"\n⚠️  Errors ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
        print("="*60)
        
        ingestion.close()
        
    except Exception as e:
        logger.error(f"❌ Ingestion failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()


