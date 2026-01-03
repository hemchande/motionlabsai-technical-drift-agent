#!/usr/bin/env python3
"""
Deep Search & Reasoning Module
Researches technique standards, reasons about data needs, and plans tool execution.
"""

import logging
import os
from typing import Dict, List, Any, Optional
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Try to import web search capabilities
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False
    logger.warning("⚠️  Tavily not available - web search will be limited")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("⚠️  Google Generative AI not available - LLM reasoning will be limited")


class DeepSearchReasoning:
    """
    Deep search and reasoning engine that:
    1. Researches technique standards via web search
    2. Uses LLM to reason about data requirements
    3. Plans CV tool execution based on research
    """
    
    def __init__(self, llm_instance=None):
        self.llm_instance = llm_instance
        self.tavily_client = None
        self._initialize_search()
    
    def _initialize_search(self):
        """Initialize web search client"""
        if TAVILY_AVAILABLE:
            tavily_api_key = os.getenv("TAVILY_API_KEY")
            if tavily_api_key:
                try:
                    self.tavily_client = TavilyClient(api_key=tavily_api_key)
                    logger.info("✅ Tavily web search initialized")
                except Exception as e:
                    logger.warning(f"⚠️  Failed to initialize Tavily: {e}")
            else:
                logger.warning("⚠️  TAVILY_API_KEY not set - web search disabled")
        else:
            logger.warning("⚠️  Tavily not installed - web search disabled")
    
    def research_technique(
        self,
        technique: str,
        sport: Optional[str] = None,
        user_requests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Research technique standards and best practices.
        
        Args:
            technique: Technique name (e.g., "back_handspring", "tennis_serve")
            sport: Sport name (e.g., "gymnastics", "tennis")
            user_requests: User's specific requests
        
        Returns:
            Research findings with standards, biomechanics, common errors
        """
        logger.info(f"🔍 Researching technique: {technique}")
        
        research_findings = {
            "technique": technique,
            "sport": sport or self._infer_sport(technique),
            "research_queries": [],
            "web_search_results": [],
            "standards": {},
            "biomechanics": {},
            "common_errors": [],
            "key_metrics": [],
            "recommended_analysis": []
        }
        
        # Generate research queries
        queries = self._generate_research_queries(technique, sport, user_requests)
        research_findings["research_queries"] = queries
        
        # Perform web search
        if self.tavily_client:
            for query in queries[:3]:  # Limit to 3 queries to avoid rate limits
                try:
                    results = self.tavily_client.search(
                        query=query,
                        search_depth="advanced",
                        max_results=5
                    )
                    research_findings["web_search_results"].extend(results.get("results", []))
                    logger.info(f"   ✅ Found {len(results.get('results', []))} results for: {query}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Search failed for '{query}': {e}")
        
        # Extract insights from search results
        if research_findings["web_search_results"]:
            insights = self._extract_insights(research_findings)
            research_findings.update(insights)
        
        # Use LLM to synthesize research (if available)
        if self.llm_instance or GEMINI_AVAILABLE:
            synthesized = self._synthesize_with_llm(research_findings)
            research_findings.update(synthesized)
        
        logger.info(f"✅ Research complete: {len(research_findings.get('key_metrics', []))} key metrics identified")
        
        return research_findings
    
    def _generate_research_queries(
        self,
        technique: str,
        sport: Optional[str],
        user_requests: Optional[List[str]]
    ) -> List[str]:
        """Generate web search queries for technique research"""
        queries = []
        
        sport_name = sport or self._infer_sport(technique)
        
        # Base queries
        queries.append(f"{sport_name} {technique} proper technique standards")
        queries.append(f"{sport_name} {technique} biomechanics key metrics")
        queries.append(f"{sport_name} {technique} common errors corrections")
        
        # ACL tear risk specific research (for landing/jumping techniques)
        technique_lower = technique.lower()
        landing_keywords = ["landing", "jump", "aerial", "handspring", "flip", "vault", "dismount"]
        has_landing = any(keyword in technique_lower for keyword in landing_keywords)
        
        if has_landing or "acl" in " ".join(user_requests or []).lower() or "injury" in " ".join(user_requests or []).lower():
            # Add ACL-specific research queries
            queries.append(f"{sport_name} {technique} ACL tear risk factors biomechanics")
            queries.append(f"{sport_name} {technique} landing technique ACL injury prevention")
            queries.append(f"{sport_name} {technique} knee valgus collapse ACL risk")
            queries.append(f"{sport_name} {technique} safe landing knee flexion angles")
        
        # Add user-specific queries
        if user_requests:
            for request in user_requests[:2]:  # Limit to avoid too many queries
                queries.append(f"{sport_name} {technique} {request} analysis")
        
        return queries
    
    def _infer_sport(self, technique: str) -> str:
        """Infer sport from technique name"""
        technique_lower = technique.lower()
        
        if any(word in technique_lower for word in ["handspring", "flip", "vault", "beam", "bars"]):
            return "gymnastics"
        elif any(word in technique_lower for word in ["serve", "forehand", "backhand", "volley"]):
            return "tennis"
        elif any(word in technique_lower for word in ["swing", "putt", "drive"]):
            return "golf"
        elif any(word in technique_lower for word in ["jump", "shot", "dunk"]):
            return "basketball"
        else:
            return "general"
    
    def _extract_insights(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Extract insights from web search results"""
        insights = {
            "standards": {},
            "biomechanics": {},
            "common_errors": [],
            "key_metrics": []
        }
        
        # Simple extraction from search results
        # In production, this would use NLP/LLM to extract structured data
        for result in research_findings.get("web_search_results", []):
            content = result.get("content", "").lower()
            title = result.get("title", "").lower()
            
            # Extract metrics mentioned
            metric_keywords = ["angle", "height", "velocity", "force", "rotation", "bend", "straightness", 
                             "valgus", "flexion", "impact", "landing", "knee", "acl", "injury"]
            for keyword in metric_keywords:
                if keyword in content or keyword in title:
                    if keyword not in insights["key_metrics"]:
                        insights["key_metrics"].append(keyword)
            
            # Extract ACL-specific information
            acl_keywords = ["acl", "anterior cruciate", "knee injury", "valgus", "knock-knee", "landing risk"]
            if any(keyword in content.lower() or keyword in title.lower() for keyword in acl_keywords):
                if "ACL risk metrics" not in insights["key_metrics"]:
                    insights["key_metrics"].append("ACL risk metrics")
                if "ACL tear risk" not in insights.get("common_errors", []):
                    if "common_errors" not in insights:
                        insights["common_errors"] = []
                    insights["common_errors"].append("ACL tear risk factors")
        
        return insights
    
    def _synthesize_with_llm(self, research_findings: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to synthesize research findings"""
        if not (self.llm_instance or GEMINI_AVAILABLE):
            return {}
        
        try:
            # Prepare prompt
            search_results_text = "\n\n".join([
                f"**{r.get('title', 'Result')}**\n{r.get('content', '')[:500]}"
                for r in research_findings.get("web_search_results", [])[:5]
            ])
            
            # Check if technique involves landing/jumping (ACL risk relevant)
            technique_lower = research_findings.get('technique', '').lower()
            has_landing = any(keyword in technique_lower for keyword in ["landing", "jump", "aerial", "handspring", "flip", "vault", "dismount"])
            
            acl_prompt_section = ""
            acl_json_section = ""
            if has_landing:
                acl_prompt_section = """

5. **ACL Tear Risk Factors** (CRITICAL for landing techniques):
   - Knee valgus collapse (knock-knee) angles and thresholds
   - Landing knee flexion angles (safe vs. risky)
   - Impact force thresholds
   - Asymmetric landing patterns
   - Technique-specific ACL risk factors
   - Recommended ACL risk metrics to track
"""
                acl_json_section = "\n- acl_risk_metrics: list of ACL-specific metrics to track (e.g., knee_valgus_angle, landing_knee_flexion, impact_force)"
            
            prompt = f"""Based on the following research about {research_findings['technique']}, extract:

1. Key metrics that should be measured (e.g., angles, heights, velocities)
2. Standard values or ranges for these metrics
3. Common errors and what to look for
4. Recommended analysis approach{acl_prompt_section}

Research Results:
{search_results_text}

Provide a structured JSON response with:
- key_metrics: list of metric names
- standards: dict of metric -> expected value/range
- common_errors: list of common mistakes
- recommended_analysis: list of analysis steps{acl_json_section}
"""
            
            # Use LLM (prefer instance, fallback to Gemini)
            if self.llm_instance:
                # If using vision-agents LLM
                response = self._call_llm_instance(prompt)
            elif GEMINI_AVAILABLE:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY"))
                model = genai.GenerativeModel('gemini-pro')
                response = model.generate_content(prompt)
                response = response.text
            
            # Parse response (simplified - in production use structured output)
            synthesized = self._parse_llm_response(response)
            return synthesized
            
        except Exception as e:
            logger.warning(f"⚠️  LLM synthesis failed: {e}")
            return {}
    
    def _call_llm_instance(self, prompt: str) -> str:
        """Call LLM instance if available"""
        # This would need to be adapted based on the LLM interface
        # For now, return empty
        return ""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured data"""
        # Simplified parsing - in production use structured output
        parsed = {
            "key_metrics": [],
            "standards": {},
            "common_errors": [],
            "recommended_analysis": [],
            "acl_risk_metrics": []  # ACL-specific metrics for techniques with landing
        }
        
        # Try to extract ACL-related metrics from response
        response_lower = response.lower()
        if "acl" in response_lower or "valgus" in response_lower or "knee injury" in response_lower:
            # Add ACL risk metrics if mentioned
            if "valgus" in response_lower:
                parsed["acl_risk_metrics"].append("knee_valgus_angle")
            if "flexion" in response_lower or "bend" in response_lower:
                parsed["acl_risk_metrics"].append("landing_knee_flexion")
            if "impact" in response_lower or "force" in response_lower:
                parsed["acl_risk_metrics"].append("impact_force")
            if "asymmetric" in response_lower or "asymmetry" in response_lower:
                parsed["acl_risk_metrics"].append("landing_asymmetry")
        
        return parsed
    
    def reason_about_data_needs(
        self,
        research_findings: Dict[str, Any],
        user_requests: List[str]
    ) -> Dict[str, Any]:
        """
        Reason about what data/metrics are needed based on research.
        
        Args:
            research_findings: Results from research_technique()
            user_requests: User's specific requests
        
        Returns:
            Data requirements with reasoning
        """
        logger.info("🧠 Reasoning about data requirements...")
        
        data_needs = {
            "required_metrics": [],
            "required_cv_tools": [],
            "reasoning": [],
            "priority": {}
        }
        
        # Extract key metrics from research
        key_metrics = research_findings.get("key_metrics", [])
        user_metric_keywords = " ".join(user_requests).lower()
        
        # Map metrics to CV tools
        metric_to_tool = {
            "angle": ["pose_estimation"],
            "height": ["pose_estimation"],
            "velocity": ["pose_estimation", "tracking"],
            "force": ["pose_estimation", "tracking"],
            "rotation": ["pose_estimation"],
            "bend": ["pose_estimation"],
            "straightness": ["pose_estimation"],
            "landing": ["pose_estimation", "person_detection"],
            "impact": ["pose_estimation", "tracking"]
        }
        
        # Determine required tools
        required_tools = set()
        for metric in key_metrics:
            if metric in metric_to_tool:
                required_tools.update(metric_to_tool[metric])
                data_needs["reasoning"].append(
                    f"Metric '{metric}' requires: {', '.join(metric_to_tool[metric])}"
                )
        
        # Add user-requested metrics
        for request in user_requests:
            request_lower = request.lower()
            for metric, tools in metric_to_tool.items():
                if metric in request_lower:
                    required_tools.update(tools)
                    if metric not in data_needs["required_metrics"]:
                        data_needs["required_metrics"].append(metric)
        
        data_needs["required_cv_tools"] = list(required_tools)
        data_needs["required_metrics"] = list(set(data_needs["required_metrics"] + key_metrics))
        
        logger.info(f"✅ Identified {len(data_needs['required_cv_tools'])} CV tools needed")
        
        return data_needs
    
    def plan_tool_execution(
        self,
        data_needs: Dict[str, Any],
        available_tools: List[str]
    ) -> Dict[str, Any]:
        """
        Plan CV tool execution sequence based on data needs.
        
        Args:
            data_needs: Results from reason_about_data_needs()
            available_tools: List of available CV tools
        
        Returns:
            Execution plan with sequence and dependencies
        """
        logger.info("📋 Planning tool execution sequence...")
        
        execution_plan = {
            "sequence": [],
            "dependencies": {},
            "reasoning": []
        }
        
        required_tools = data_needs.get("required_cv_tools", [])
        
        # Define tool dependencies
        tool_dependencies = {
            "pose_estimation": ["person_detection"],  # Pose needs person detection first
            "weight_estimation": ["pose_estimation"],  # Weight needs pose
            "metric_extraction": ["pose_estimation"],  # Metrics need pose
        }
        
        # Build execution sequence
        executed = set()
        step = 1
        
        # Step 1: Person detection (if needed)
        if "person_detection" in required_tools and "person_detection" in available_tools:
            execution_plan["sequence"].append({
                "step": step,
                "tool": "person_detection",
                "purpose": "Detect and locate athlete in frame",
                "output": "bounding_box"
            })
            executed.add("person_detection")
            step += 1
        
        # Step 2: Pose estimation (core)
        if "pose_estimation" in required_tools and "pose_estimation" in available_tools:
            execution_plan["sequence"].append({
                "step": step,
                "tool": "pose_estimation",
                "purpose": "Extract body keypoints for metric calculation",
                "output": "keypoints",
                "depends_on": [1] if "person_detection" in executed else []
            })
            executed.add("pose_estimation")
            step += 1
        
        # Step 3: Additional tools
        for tool in required_tools:
            if tool not in executed and tool in available_tools:
                deps = tool_dependencies.get(tool, [])
                dep_steps = [
                    s["step"] for s in execution_plan["sequence"]
                    if s["tool"] in deps
                ]
                
                execution_plan["sequence"].append({
                    "step": step,
                    "tool": tool,
                    "purpose": f"Execute {tool} analysis",
                    "output": f"{tool}_results",
                    "depends_on": dep_steps
                })
                executed.add(tool)
                step += 1
        
        # Step N: Metric extraction (always last)
        execution_plan["sequence"].append({
            "step": step,
            "tool": "metric_extraction",
            "purpose": "Calculate technique-specific metrics",
            "output": "metrics",
            "depends_on": [s["step"] for s in execution_plan["sequence"] if s["tool"] == "pose_estimation"]
        })
        
        logger.info(f"✅ Planned {len(execution_plan['sequence'])} execution steps")
        
        return execution_plan





















