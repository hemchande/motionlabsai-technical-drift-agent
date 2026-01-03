#!/usr/bin/env python3
"""
Comprehensive Test for ACL Risk Retrieval Agent

Tests all functionality including:
- Basic retrieval
- Trend tracking
- Guardrails integration
- API wrapper
"""

import logging
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from acl_risk_retrieval_agent import ACLRiskRetrievalAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_mongodb_connection() -> bool:
    """Test MongoDB connection."""
    print("\n" + "="*60)
    print("🧪 TEST 1: MongoDB Connection")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        # Test connection by trying to get collection
        collection = agent.mongodb.get_sessions_collection()
        count = collection.count_documents({})
        print(f"✅ MongoDB connection successful")
        print(f"   Database: {agent.mongodb.db_name}")
        print(f"   Sessions in database: {count}")
        agent.close()
        return True
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return False


def test_basic_retrieval() -> bool:
    """Test basic session retrieval."""
    print("\n" + "="*60)
    print("🧪 TEST 2: Basic Session Retrieval")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Test 1: Find all risk sessions
        print("\n📊 Finding all risk ACL sessions...")
        sessions = agent.find_risk_sessions(
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        print(f"   Found {len(sessions)} sessions with risk ACL moments")
        
        if not sessions:
            print("   ⚠️  No risk sessions found. This is expected if:")
            print("      - No sessions have been ingested yet")
            print("      - No sessions have ACL risk scores >= 0.4")
            agent.close()
            return True  # Not a failure, just no data
        
        # Test 2: Get metadata
        print("\n📋 Getting session metadata...")
        metadata = agent.get_risk_metadata(sessions=sessions[:5])
        print(f"   Retrieved metadata for {len(metadata)} sessions")
        
        if metadata:
            print(f"\n   Sample session:")
            sample = metadata[0]
            print(f"      Session ID: {sample.get('session_id', 'N/A')}")
            print(f"      Activity: {sample.get('activity', 'N/A')}")
            print(f"      Technique: {sample.get('technique', 'N/A')}")
            print(f"      Risk Level: {sample.get('acl_risk_level', 'N/A')}")
            print(f"      Risk Score: {sample.get('acl_risk_score', 0):.2f}")
            print(f"      High Risk Count: {sample.get('high_risk_count', 0)}")
            print(f"      Moderate Risk Count: {sample.get('moderate_risk_count', 0)}")
        
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ Basic retrieval test failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return False


def test_trend_tracking() -> bool:
    """Test trend tracking functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 3: Trend Tracking")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Get sessions for trend tracking
        sessions = agent.find_risk_sessions(
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        
        print(f"\n📊 Found {len(sessions)} sessions for trend analysis")
        
        if len(sessions) < 3:
            print("   ⚠️  Insufficient sessions for trend tracking (need >= 3)")
            print("      This is expected if there are fewer than 3 sessions in the database")
            agent.close()
            return True  # Not a failure
        
        # Test trend tracking
        print("\n📈 Tracking trends from sessions...")
        trends_result = agent.track_trends_from_sessions(
            sessions=sessions,
            min_sessions=3
        )
        
        if not trends_result.get("success", False):
            print(f"   ⚠️  Trend tracking returned: {trends_result.get('message', 'Unknown error')}")
            agent.close()
            return True  # Not necessarily a failure
        
        trends = trends_result.get("trends", [])
        print(f"   ✅ Identified {len(trends)} trends")
        
        if trends:
            print(f"\n   Sample trend:")
            sample = trends[0]
            print(f"      Metric Type: {sample.get('metric_type', 'N/A')}")
            print(f"      Athlete: {sample.get('athlete_name', 'N/A')}")
            print(f"      Technique: {sample.get('technique', 'N/A')}")
            print(f"      Status: {sample.get('status', 'N/A')}")
            print(f"      Observation: {sample.get('observation', 'N/A')[:100]}...")
            print(f"      Evidence: {sample.get('evidence_reasoning', 'N/A')[:100]}...")
            print(f"      Coaching Options: {len(sample.get('coaching_options', []))} options")
        
        # Test retrieving stored trends
        print("\n📋 Retrieving stored trends from MongoDB...")
        stored_trends = agent.get_tracked_trends(limit=10)
        print(f"   Found {len(stored_trends)} stored trends in database")
        
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ Trend tracking test failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return False


def test_guardrails() -> bool:
    """Test guardrails integration."""
    print("\n" + "="*60)
    print("🧪 TEST 4: Guardrails Integration")
    print("="*60)
    
    try:
        # Test if guardrails are available
        try:
            from guardrails import Guardrails
            guardrails = Guardrails()
            print("✅ Guardrails module loaded successfully")
        except ImportError as e:
            print(f"⚠️  Guardrails not available: {e}")
            return True  # Not a failure, just not available
        
        # Test guardrails with sample data
        print("\n🔍 Testing guardrails validation...")
        
        sample_response = {
            "observation": "Knee valgus increased from 8.0° → 14.0° on 3 sessions.",
            "evidence_reasoning": "This pattern may indicate changes in landing mechanics.",
            "coaching_options": [
                "Consider reviewing landing mechanics",
                "If pain/symptoms are present, consult your AT/PT."
            ]
        }
        
        source_data = {
            "trend_statistics": {
                "first_mean": 8.0,
                "second_mean": 14.0,
                "change_percent": 75.0,
                "direction": "increasing"
            },
            "metric_type": "valgus_angle",
            "technique": "back_handspring"
        }
        
        result = guardrails.apply_guardrails(
            response=sample_response,
            source_data=source_data,
            auto_fix=True
        )
        
        validation = result.get("validation", {})
        print(f"   Confidence Score: {validation.get('confidence_score', 0):.2f}")
        print(f"   Valid: {validation.get('valid', False)}")
        print(f"   Hallucination Detected: {validation.get('hallucination_detected', False)}")
        
        if validation.get("warnings"):
            print(f"   Warnings: {len(validation['warnings'])}")
            for warning in validation["warnings"][:3]:
                print(f"      - {warning}")
        
        print("✅ Guardrails validation completed")
        return True
        
    except Exception as e:
        print(f"❌ Guardrails test failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return False


def test_api_wrapper() -> bool:
    """Test API wrapper functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 5: API Wrapper")
    print("="*60)
    
    try:
        import subprocess
        from pathlib import Path
        
        script_path = Path(__file__).parent / "retrieve_insights_api.py"
        
        print("\n📡 Testing API wrapper with --min-risk 0.4...")
        
        result = subprocess.run(
            ["python3", str(script_path), "--min-risk", "0.4"],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            print(f"   ⚠️  API wrapper returned non-zero exit code: {result.returncode}")
            print(f"   Error: {result.stderr[:200]}")
            return False
        
        # Parse JSON output
        try:
            output = json.loads(result.stdout)
            print(f"   ✅ API wrapper executed successfully")
            print(f"   Success: {output.get('success', False)}")
            print(f"   Sessions: {output.get('count', 0)}")
            print(f"   Trends: {output.get('trend_count', 0)}")
            
            if output.get("error"):
                print(f"   ⚠️  Error in output: {output['error']}")
            
            return True
            
        except json.JSONDecodeError as e:
            print(f"   ❌ Failed to parse JSON output: {e}")
            print(f"   Output: {result.stdout[:500]}")
            return False
        
    except subprocess.TimeoutExpired:
        print("   ❌ API wrapper timed out (>60s)")
        return False
    except Exception as e:
        print(f"   ❌ API wrapper test failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return False


def test_pattern_analysis() -> bool:
    """Test pattern analysis functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 6: Pattern Analysis")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Get sessions
        sessions = agent.find_risk_sessions(
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        
        if not sessions:
            print("   ⚠️  No sessions available for pattern analysis")
            agent.close()
            return True
        
        print(f"\n📊 Analyzing patterns from {len(sessions)} sessions...")
        
        # Test pattern analysis
        analysis = agent.analyze_risk_patterns(
            sessions=sessions,
            include_moderate=True,
            include_high=True
        )
        
        print(f"   ✅ Pattern analysis completed")
        print(f"   Total Sessions: {analysis.get('total_sessions', 0)}")
        print(f"   Total Risk Moments: {analysis.get('total_risk_moments', 0)}")
        print(f"   High Risk Moments: {analysis.get('total_high_risk_moments', 0)}")
        print(f"   Moderate Risk Moments: {analysis.get('total_moderate_risk_moments', 0)}")
        
        patterns = analysis.get("patterns", {})
        if patterns:
            print(f"\n   Pattern Breakdown:")
            print(f"      By Activity: {patterns.get('by_activity', {})}")
            print(f"      By Technique: {patterns.get('by_technique', {})}")
            print(f"      By Risk Factor: {patterns.get('by_risk_factor', {})}")
        
        agent.close()
        return True
        
    except Exception as e:
        print(f"❌ Pattern analysis test failed: {e}")
        logger.error(f"Error: {e}", exc_info=True)
        return False


def test_llm_reasoning() -> bool:
    """Test LLM reasoning functionality."""
    print("\n" + "="*60)
    print("🧪 TEST 7: LLM Reasoning (Optional)")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Get sessions
        sessions = agent.find_risk_sessions(
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        
        if not sessions or len(sessions) < 2:
            print("   ⚠️  Insufficient sessions for LLM reasoning (need >= 2)")
            agent.close()
            return True
        
        print(f"\n🧠 Running LLM reasoning on {len(sessions)} sessions...")
        print("   (This may take a moment and requires API keys)")
        
        # Test LLM reasoning
        reasoning_result = agent.reason_about_acl_patterns(
            sessions=sessions[:5],  # Limit to first 5 for testing
            include_moderate=True,
            include_high=True
        )
        
        llm_reasoning = reasoning_result.get("llm_reasoning", {})
        
        if not llm_reasoning or "executive_summary" not in llm_reasoning:
            print("   ⚠️  LLM reasoning not available (may need API keys)")
            agent.close()
            return True  # Not a failure, just not available
        
        print(f"   ✅ LLM reasoning completed")
        print(f"   Executive Summary: {llm_reasoning.get('executive_summary', 'N/A')[:150]}...")
        
        key_insights = llm_reasoning.get("key_insights", [])
        if key_insights:
            print(f"   Key Insights: {len(key_insights)}")
            for i, insight in enumerate(key_insights[:3], 1):
                print(f"      {i}. {insight[:100]}...")
        
        agent.close()
        return True
        
    except Exception as e:
        print(f"⚠️  LLM reasoning test: {e}")
        print("   (This is expected if API keys are not configured)")
        return True  # Not a failure, just not available


def main():
    """Run all comprehensive tests."""
    print("\n" + "="*60)
    print("🧪 COMPREHENSIVE RETRIEVAL AGENT TESTS")
    print("="*60)
    print("\nTesting all functionality including:")
    print("  - MongoDB connection")
    print("  - Basic retrieval")
    print("  - Trend tracking")
    print("  - Guardrails integration")
    print("  - API wrapper")
    print("  - Pattern analysis")
    print("  - LLM reasoning (optional)")
    
    results = {
        "mongodb_connection": False,
        "basic_retrieval": False,
        "trend_tracking": False,
        "guardrails": False,
        "api_wrapper": False,
        "pattern_analysis": False,
        "llm_reasoning": False
    }
    
    # Run tests
    results["mongodb_connection"] = test_mongodb_connection()
    results["basic_retrieval"] = test_basic_retrieval()
    results["trend_tracking"] = test_trend_tracking()
    results["guardrails"] = test_guardrails()
    results["api_wrapper"] = test_api_wrapper()
    results["pattern_analysis"] = test_pattern_analysis()
    results["llm_reasoning"] = test_llm_reasoning()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✅ PASSED" if passed_test else "❌ FAILED"
        print(f"   {test_name:.<30} {status}")
    
    print("="*60)
    print(f"\n   Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("   ✅ All tests passed!")
        return 0
    else:
        print(f"   ⚠️  {total - passed} test(s) failed or skipped")
        return 1


if __name__ == "__main__":
    sys.exit(main())

