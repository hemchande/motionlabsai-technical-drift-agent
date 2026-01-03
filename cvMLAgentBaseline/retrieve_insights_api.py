#!/usr/bin/env python3
"""
API wrapper for form correction retrieval agent.
Outputs JSON for API consumption.
Extracts actual form issues from session metrics.
"""

import sys
import json
import argparse
import logging
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from form_correction_retrieval_agent import FormCorrectionRetrievalAgent

logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Retrieve form correction insights as JSON")
    parser.add_argument("--activity", type=str, default=None, help="Filter by activity")
    parser.add_argument("--technique", type=str, default=None, help="Filter by technique")
    parser.add_argument("--issue-type", type=str, default=None, help="Filter by specific issue type")
    parser.add_argument("--min-severity", type=str, default="minor", choices=["minor", "moderate", "severe"], help="Minimum severity (default: minor)")
    parser.add_argument("--min-sessions-per-issue", type=int, default=3, help="Minimum sessions an issue must appear in (default: 3)")
    
    args = parser.parse_args()
    
    try:
        agent = FormCorrectionRetrievalAgent()
        
        # Parse issue types if provided
        issue_types = [args.issue_type] if args.issue_type else None
        
        # Find sessions with form issues (only issues appearing in 3+ sessions)
        sessions = agent.find_sessions_with_form_issues(
            activity=args.activity,
            technique=args.technique,
            issue_types=issue_types,
            min_severity=args.min_severity,
            min_sessions_per_issue=args.min_sessions_per_issue
        )
        
        # Get metadata for sessions
        metadata = agent.get_form_issue_metadata(
            sessions=sessions,
            activity=args.activity,
            technique=args.technique,
            min_sessions_per_issue=args.min_sessions_per_issue
        )
        
        # Analyze form issues across sessions
        analysis = agent.analyze_form_issues_across_sessions(
            sessions=sessions,
            activity=args.activity,
            technique=args.technique,
            min_sessions_per_issue=args.min_sessions_per_issue
        )
        
        # Track trends from sessions (if enough sessions available)
        trends_result = None
        if len(sessions) >= 3:  # Minimum sessions for trend tracking
            try:
                trends_result = agent.track_form_issue_trends(
                    sessions=sessions,
                    activity=args.activity,
                    technique=args.technique,
                    min_sessions=3
                )
            except Exception as e:
                logger.warning(f"⚠️  Trend tracking failed: {e}")
        
        # Output JSON
        output = {
            "success": True,
            "sessions": metadata,
            "count": len(metadata),
            "analysis": {
                "total_sessions": analysis.get("total_sessions", 0),
                "total_issues": analysis.get("total_issues", 0),
                "most_common_issues": analysis.get("most_common_issues", []),
                "summary": analysis.get("summary", "")
            },
            "trends": trends_result.get("trends", []) if trends_result else [],
            "trend_count": trends_result.get("trend_count", 0) if trends_result else 0
        }
        
        print(json.dumps(output, indent=2, default=str))
        
        agent.close()
        
    except Exception as e:
        error_output = {
            "success": False,
            "error": str(e),
            "sessions": [],
            "count": 0
        }
        print(json.dumps(error_output, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()

