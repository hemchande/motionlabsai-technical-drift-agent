#!/usr/bin/env python3
"""
Test ACL Risk Retrieval Agent

Tests the MongoDB ingestion and retrieval agent for high-risk ACL detection.
"""

import logging
import json
from pathlib import Path
from datetime import datetime
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import json for datetime serialization
from datetime import datetime

from mongodb_session_ingestion import SessionDataIngestion
from acl_risk_retrieval_agent import ACLRiskRetrievalAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_ingestion():
    """Test session data ingestion."""
    print("\n" + "="*60)
    print("🧪 TEST 1: Session Data Ingestion")
    print("="*60)
    
    try:
        ingestion = SessionDataIngestion()
        
        # First, do a dry run
        print("\n📋 Dry run (checking what would be ingested)...")
        stats = ingestion.ingest_all_sessions(dry_run=True)
        print(f"   Found {stats['total_files']} metrics files")
        print(f"   Would ingest {stats['metrics_ingested']} sessions")
        
        # Ask for confirmation
        response = input("\n❓ Proceed with actual ingestion? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("⏭️  Skipping ingestion")
            ingestion.close()
            return False
        
        # Actual ingestion
        print("\n💾 Ingesting sessions...")
        stats = ingestion.ingest_all_sessions(dry_run=False)
        
        print(f"\n✅ Ingestion complete!")
        print(f"   Metrics ingested: {stats['metrics_ingested']}")
        print(f"   Transcripts ingested: {stats['transcripts_ingested']}")
        print(f"   Sessions created: {stats['sessions_created']}")
        print(f"   Sessions updated: {stats['sessions_updated']}")
        
        if stats['errors']:
            print(f"\n⚠️  Errors: {len(stats['errors'])}")
            for error in stats['errors'][:5]:
                print(f"   - {error}")
        
        ingestion.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Ingestion test failed: {e}", exc_info=True)
        return False


def test_retrieval():
    """Test high-risk ACL retrieval."""
    print("\n" + "="*60)
    print("🧪 TEST 2: High-Risk ACL Retrieval")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Test 1: Find all risk sessions (moderate and high)
        print("\n📊 Finding all risk ACL sessions (moderate + high)...")
        all_sessions = agent.find_risk_sessions(min_risk_score=0.4, include_moderate=True, include_high=True)
        print(f"   Found {len(all_sessions)} sessions with risk ACL moments")
        
        if all_sessions:
            high_count = sum(1 for s in all_sessions if s.get("high_risk_count", 0) > 0)
            moderate_count = sum(1 for s in all_sessions if s.get("moderate_risk_count", 0) > 0)
            print(f"   - {high_count} sessions with HIGH risk")
            print(f"   - {moderate_count} sessions with MODERATE risk")
        
        if not all_sessions:
            print("   ⚠️  No high-risk sessions found. This is expected if:")
            print("      - No sessions have been ingested yet")
            print("      - No sessions have ACL risk scores >= 0.7")
            agent.close()
            return True
        
        # Test 2: Filter by activity
        print("\n📊 Finding risk gymnastics sessions...")
        gym_sessions = agent.find_risk_sessions(activity="gymnastics", include_moderate=True, include_high=True)
        print(f"   Found {len(gym_sessions)} gymnastics sessions with risk ACL moments")
        
        # Test 3: Get session metadata
        print("\n📋 Getting session metadata...")
        metadata = agent.get_risk_metadata(sessions=all_sessions[:5], include_moderate=True, include_high=True)  # First 5
        print(f"   Retrieved metadata for {len(metadata)} sessions")
        
        for i, meta in enumerate(metadata, 1):
            print(f"\n   Session {i}:")
            print(f"      ID: {meta['session_id']}")
            print(f"      Activity: {meta['activity']}, Technique: {meta['technique']}")
            print(f"      Risk level: {meta['acl_risk_level']}, Score: {meta['acl_risk_score']:.2f}")
            print(f"      HIGH-risk moments: {meta['high_risk_count']}")
            print(f"      MODERATE-risk moments: {meta['moderate_risk_count']}")
            print(f"      Total risk moments: {meta['total_risk_count']}")
            print(f"      Max valgus: {meta['acl_max_valgus_angle']:.1f}°")
        
        # Test 4: Pattern analysis
        print("\n🔍 Analyzing risk patterns...")
        analysis = agent.analyze_risk_patterns(sessions=all_sessions, include_moderate=True, include_high=True)
        
        print(f"\n   Analysis Results:")
        print(f"      Total sessions: {analysis['total_sessions']}")
        print(f"      Total risk moments: {analysis['total_risk_moments']}")
        print(f"      - HIGH risk: {analysis['total_high_risk_moments']}")
        print(f"      - MODERATE risk: {analysis['total_moderate_risk_moments']}")
        print(f"      Summary: {analysis['summary']}")
        
        if analysis['patterns']['by_risk_factor']:
            print(f"\n   Risk Factors:")
            for factor, count in analysis['patterns']['by_risk_factor'].items():
                print(f"      - {factor}: {count} occurrences")
        
        if analysis['patterns']['by_activity']:
            print(f"\n   By Activity:")
            for activity, count in analysis['patterns']['by_activity'].items():
                print(f"      - {activity}: {count} moments")
        
        if analysis['patterns']['risk_score_stats']:
            stats = analysis['patterns']['risk_score_stats']
            print(f"\n   Risk Score Statistics:")
            print(f"      Mean: {stats['mean']:.2f}")
            print(f"      Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        
        # Test 5: Get specific session details
        if all_sessions:
            print("\n📄 Getting detailed session information...")
            first_session = all_sessions[0]
            session_id = first_session.get("session_id")
            if session_id:
                details = agent.get_session_details(session_id)
                if details:
                    print(f"   Retrieved details for session: {session_id}")
                    print(f"      Activity: {details.get('activity')}")
                    print(f"      Technique: {details.get('technique')}")
                    print(f"      Has transcript: {details.get('transcript') is not None}")
                    print(f"      High-risk moments: {len(details.get('high_risk_moments', []))}")
        
        agent.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Retrieval test failed: {e}", exc_info=True)
        return False


def test_llm_reasoning():
    """Test LLM reasoning for ACL pattern analysis."""
    print("\n" + "="*60)
    print("🧪 TEST 3: LLM Reasoning Analysis")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Find risk sessions (moderate and high)
        sessions = agent.find_risk_sessions(min_risk_score=0.4, include_moderate=True, include_high=True)
        
        if not sessions:
            print("   ⚠️  No risk sessions found (score >= 0.4)")
            print("   LLM reasoning requires at least one risk session")
            agent.close()
            return True
        
        print(f"\n🧠 Running LLM reasoning on {len(sessions)} risk sessions...")
        
        # Run reasoning analysis
        reasoning_result = agent.reason_about_acl_patterns(
            sessions=sessions,
            include_moderate=True,
            include_high=True
        )
        
        llm_reasoning = reasoning_result.get("llm_reasoning", {})
        
        if not llm_reasoning or "executive_summary" not in llm_reasoning:
            print("   ⚠️  LLM reasoning returned empty or invalid response")
            print("   This may indicate:")
            print("      - No LLM API key configured (GEMINI_API_KEY or OPENAI_API_KEY)")
            print("      - LLM API call failed")
            agent.close()
            return False
        
        print(f"\n✅ LLM Reasoning Analysis Complete!")
        print(f"\n📊 Executive Summary:")
        print(f"   {llm_reasoning.get('executive_summary', 'N/A')}")
        
        key_insights = llm_reasoning.get("key_insights", [])
        if key_insights:
            print(f"\n🔍 Key Insights ({len(key_insights)}):")
            for i, insight in enumerate(key_insights[:5], 1):
                print(f"   {i}. {insight}")
        
        root_causes = llm_reasoning.get("root_cause_analysis", {})
        if root_causes.get("primary_causes"):
            print(f"\n🔬 Root Cause Analysis:")
            for i, cause in enumerate(root_causes["primary_causes"][:3], 1):
                print(f"   {i}. {cause.get('cause', 'N/A')}")
                print(f"      Severity: {cause.get('severity', 'N/A')}")
                print(f"      Evidence: {cause.get('evidence', 'N/A')[:100]}...")
        
        recommendations = llm_reasoning.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)}):")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"   {i}. [{rec.get('priority', 'N/A').upper()}] {rec.get('recommendation', 'N/A')}")
                print(f"      Category: {rec.get('category', 'N/A')}")
        
        coaching_priorities = llm_reasoning.get("coaching_priorities", [])
        if coaching_priorities:
            print(f"\n🎯 Coaching Priorities:")
            for i, priority in enumerate(coaching_priorities[:5], 1):
                print(f"   {i}. {priority}")
        
        # Save reasoning result to file
        output_file = Path("stream_output") / f"acl_reasoning_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.parent.mkdir(exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(reasoning_result, f, indent=2, default=str)
        print(f"\n💾 Reasoning analysis saved to: {output_file}")
        
        agent.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ LLM reasoning test failed: {e}", exc_info=True)
        return False


def test_pattern_detection():
    """Test pattern detection for high-risk ACL."""
    print("\n" + "="*60)
    print("🧪 TEST 4: High-Risk Pattern Detection")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Find risk sessions (moderate and high)
        sessions = agent.find_risk_sessions(min_risk_score=0.4, include_moderate=True, include_high=True)
        
        if not sessions:
            print("   ⚠️  No risk sessions found (score >= 0.4)")
            print("   This is expected if no sessions have MODERATE or HIGH risk ACL moments")
            agent.close()
            return True
        
        print(f"\n📊 Analyzing {len(sessions)} high-risk sessions...")
        
        # Pattern detection: Check for recurring patterns
        patterns = {
            "recurring_techniques": {},
            "recurring_risk_factors": {},
            "temporal_clustering": []
        }
        
        for session in sessions:
            technique = session.get("technique", "unknown")
            patterns["recurring_techniques"][technique] = patterns["recurring_techniques"].get(technique, 0) + 1
            
            risk_moments = session.get("risk_moments", [])
            for moment in risk_moments:
                risk_factors = moment.get("primary_risk_factors", [])
                for factor in risk_factors:
                    patterns["recurring_risk_factors"][factor] = patterns["recurring_risk_factors"].get(factor, 0) + 1
        
        print(f"\n   Recurring Techniques (with risk ACL):")
        for technique, count in sorted(patterns["recurring_techniques"].items(), key=lambda x: x[1], reverse=True):
            print(f"      - {technique}: {count} sessions")
        
        print(f"\n   Recurring Risk Factors:")
        for factor, count in sorted(patterns["recurring_risk_factors"].items(), key=lambda x: x[1], reverse=True):
            print(f"      - {factor}: {count} occurrences")
        
        # Check for sessions with multiple risk moments (pattern indicator)
        multi_risk_sessions = [s for s in sessions if s.get("total_risk_count", 0) > 1]
        if multi_risk_sessions:
            print(f"\n   ⚠️  {len(multi_risk_sessions)} sessions have multiple risk moments:")
            for session in multi_risk_sessions[:5]:
                print(f"      - {session.get('session_id')}: {session.get('total_risk_count')} moments ({session.get('high_risk_count', 0)} HIGH, {session.get('moderate_risk_count', 0)} MODERATE)")
        
        agent.close()
        return True
        
    except Exception as e:
        logger.error(f"❌ Pattern detection test failed: {e}", exc_info=True)
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 ACL RISK RETRIEVAL AGENT TESTS")
    print("="*60)
    
    results = {
        "ingestion": False,
        "retrieval": False,
        "llm_reasoning": False,
        "pattern_detection": False
    }
    
    # Test 1: Ingestion
    results["ingestion"] = test_ingestion()
    
    # Test 2: Retrieval (only if ingestion succeeded or was skipped)
    if results["ingestion"] is not False:
        results["retrieval"] = test_retrieval()
    
    # Test 3: LLM Reasoning (requires retrieval)
    if results["retrieval"]:
        results["llm_reasoning"] = test_llm_reasoning()
    
    # Test 4: Pattern Detection
    if results["retrieval"]:
        results["pattern_detection"] = test_pattern_detection()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED" if passed is False else "⏭️  SKIPPED"
        print(f"   {test_name}: {status}")
    print("="*60)
    
    # Exit code
    all_passed = all(results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()


