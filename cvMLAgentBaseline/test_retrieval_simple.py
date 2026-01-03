#!/usr/bin/env python3
"""
Simple test for ACL Risk Retrieval Agent on MongoDB sessions.
Tests retrieval without requiring user input.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from acl_risk_retrieval_agent import ACLRiskRetrievalAgent

def main():
    print("\n" + "="*60)
    print("🧪 ACL RISK RETRIEVAL AGENT TEST")
    print("="*60)
    
    try:
        agent = ACLRiskRetrievalAgent()
        
        # Test 1: Find all risk sessions (moderate + high)
        print("\n📊 TEST 1: Finding risk sessions (threshold >= 0.4)...")
        all_sessions = agent.find_risk_sessions(
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        print(f"   ✅ Found {len(all_sessions)} sessions with risk ACL moments")
        
        if all_sessions:
            high_count = sum(1 for s in all_sessions if s.get("high_risk_count", 0) > 0)
            moderate_count = sum(1 for s in all_sessions if s.get("moderate_risk_count", 0) > 0)
            total_risk = sum(s.get("total_risk_count", 0) for s in all_sessions)
            print(f"   - {high_count} sessions with HIGH risk moments")
            print(f"   - {moderate_count} sessions with MODERATE risk moments")
            print(f"   - {total_risk} total risk moments across all sessions")
        
        # Test 2: Filter by activity
        print("\n📊 TEST 2: Filtering by activity (gymnastics)...")
        gym_sessions = agent.find_risk_sessions(
            activity="gymnastics",
            min_risk_score=0.4,
            include_moderate=True,
            include_high=True
        )
        print(f"   ✅ Found {len(gym_sessions)} gymnastics sessions with risk ACL moments")
        
        # Test 3: Pattern analysis
        print("\n🔍 TEST 3: Pattern analysis...")
        if all_sessions:
            analysis = agent.analyze_risk_patterns(
                sessions=all_sessions,
                include_moderate=True,
                include_high=True
            )
            print(f"   ✅ Analysis complete!")
            print(f"   - Total sessions: {analysis['total_sessions']}")
            print(f"   - Total risk moments: {analysis['total_risk_moments']}")
            print(f"   - HIGH risk moments: {analysis['total_high_risk_moments']}")
            print(f"   - MODERATE risk moments: {analysis['total_moderate_risk_moments']}")
            print(f"   - Summary: {analysis['summary']}")
            
            if analysis['patterns']['by_risk_factor']:
                print(f"\n   Risk Factors:")
                for factor, count in list(analysis['patterns']['by_risk_factor'].items())[:5]:
                    print(f"      - {factor}: {count} occurrences")
            
            if analysis['patterns']['by_risk_level']:
                print(f"\n   Risk Level Distribution:")
                for level, count in analysis['patterns']['by_risk_level'].items():
                    print(f"      - {level}: {count} sessions")
        else:
            print("   ⚠️  No risk sessions found for analysis")
        
        # Test 4: Get metadata
        print("\n📋 TEST 4: Getting session metadata...")
        if all_sessions:
            metadata = agent.get_risk_metadata(
                sessions=all_sessions[:5],  # First 5
                include_moderate=True,
                include_high=True
            )
            print(f"   ✅ Retrieved metadata for {len(metadata)} sessions")
            
            for i, meta in enumerate(metadata, 1):
                print(f"\n   Session {i}:")
                print(f"      ID: {meta['session_id']}")
                print(f"      Activity: {meta['activity']}, Technique: {meta['technique']}")
                print(f"      Risk Level: {meta['acl_risk_level']}, Score: {meta['acl_risk_score']:.2f}")
                print(f"      HIGH-risk: {meta['high_risk_count']}, MODERATE-risk: {meta['moderate_risk_count']}")
                print(f"      Total risk moments: {meta['total_risk_count']}")
        else:
            print("   ⚠️  No risk sessions found for metadata")
        
        # Test 5: LLM Reasoning (if API key available)
        print("\n🧠 TEST 5: LLM Reasoning Analysis...")
        if all_sessions:
            try:
                reasoning_result = agent.reason_about_acl_patterns(
                    sessions=all_sessions,
                    include_moderate=True,
                    include_high=True
                )
                
                llm_reasoning = reasoning_result.get("llm_reasoning", {})
                
                if llm_reasoning and "executive_summary" in llm_reasoning:
                    print(f"   ✅ LLM reasoning complete!")
                    print(f"   - Executive Summary: {llm_reasoning.get('executive_summary', 'N/A')[:100]}...")
                    
                    insights = llm_reasoning.get("key_insights", [])
                    if insights:
                        print(f"\n   Key Insights ({len(insights)}):")
                        for i, insight in enumerate(insights[:3], 1):
                            print(f"      {i}. {insight[:80]}...")
                    
                    recommendations = llm_reasoning.get("recommendations", [])
                    if recommendations:
                        print(f"\n   Recommendations ({len(recommendations)}):")
                        for i, rec in enumerate(recommendations[:3], 1):
                            print(f"      {i}. [{rec.get('priority', 'N/A')}] {rec.get('recommendation', 'N/A')[:60]}...")
                else:
                    print("   ⚠️  LLM reasoning returned empty response")
                    print("   - This may indicate no API key configured or API call failed")
                    print("   - Check GEMINI_API_KEY or OPENAI_API_KEY in .env")
            except Exception as e:
                print(f"   ⚠️  LLM reasoning test failed: {e}")
                print("   - This is expected if no API key is configured")
        else:
            print("   ⚠️  No risk sessions found for LLM reasoning")
        
        agent.close()
        
        print("\n" + "="*60)
        print("✅ RETRIEVAL AGENT TEST COMPLETE")
        print("="*60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)


