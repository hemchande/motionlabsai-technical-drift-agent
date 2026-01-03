#!/usr/bin/env python3
"""
Memory Index Integration with Deep Research & LLM Reasoning
Connects to existing memory indexes for FIG standard reasoning.
Enhances with deep research using reasoning models and LLM-based insights.
"""

import logging
import sys
import os
from typing import Dict, List, Any, Optional
import json

logger = logging.getLogger(__name__)

# Add parent directory to path to import memory retrieval
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'videoAgent'))

# Try to import LLM capabilities
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import deep search reasoning
try:
    from deep_search_reasoning import DeepSearchReasoning
    DEEP_SEARCH_AVAILABLE = True
except ImportError:
    DEEP_SEARCH_AVAILABLE = False
    logger.warning("⚠️  Deep search reasoning not available")


class MemoryIndexIntegration:
    """
    Integrates with existing memory indexes for FIG standard reasoning.
    Enhances with:
    - Deep research using reasoning models
    - LLM-based insights from memory + web search
    - Reasoning-based analysis
    """
    
    def __init__(self, use_deep_research: bool = True, llm_instance=None, reasoning_model: str = "gemini-pro"):
        self.memory_retrieval = None
        self.use_deep_research = use_deep_research
        self.llm_instance = llm_instance
        self.reasoning_model = reasoning_model
        self.deep_search = None
        
        self._initialize_memory_retrieval()
        
        # Initialize deep research if enabled
        if use_deep_research and DEEP_SEARCH_AVAILABLE:
            self.deep_search = DeepSearchReasoning(llm_instance=llm_instance)
            logger.info("✅ Deep research enabled with reasoning model")
    
    def _initialize_memory_retrieval(self):
        """Initialize memory retrieval from videoAgent"""
        try:
            from memory_retrieval import (
                get_fig_standards,
                check_standards_met,
                get_technique_list,
                map_technique_to_metrics
            )
            
            self.get_fig_standards = get_fig_standards
            self.check_standards_met = check_standards_met
            self.get_technique_list = get_technique_list
            self.map_technique_to_metrics = map_technique_to_metrics
            
            logger.info("✅ Connected to memory indexes from videoAgent")
        except ImportError as e:
            logger.warning(f"⚠️  Could not import memory_retrieval: {e}")
            logger.warning("   FIG standard reasoning will be limited")
            self.get_fig_standards = self._fallback_get_fig_standards
            self.check_standards_met = self._fallback_check_standards_met
            self.get_technique_list = self._fallback_get_technique_list
            self.map_technique_to_metrics = self._fallback_map_technique_to_metrics
    
    def get_technique_mapping(self, technique: str) -> Optional[Dict[str, Any]]:
        """
        Get technique mapping from memory index.
        
        Args:
            technique: Technique name
        
        Returns:
            Technique mapping data or None
        """
        try:
            standards = self.get_fig_standards(technique)
            if standards and standards.get("found"):
                return {
                    "technique": technique,
                    "fig_standards": standards.get("fig_standards", {}),
                    "relevant_metrics": standards.get("relevant_metrics", []),
                    "deduction_thresholds": standards.get("deduction_thresholds", {}),
                    "critical_phases": standards.get("critical_phases", [])
                }
        except Exception as e:
            logger.error(f"❌ Error getting technique mapping: {e}")
        
        return None
    
    def compare_to_standards(
        self,
        technique: str,
        metrics: Dict[str, float],
        use_deep_research: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Compare metrics to FIG standards with optional deep research and LLM reasoning.
        
        Args:
            technique: Technique name
            metrics: Calculated metrics
            use_deep_research: Override instance setting for deep research
        
        Returns:
            Comparison results with compliance information and LLM-based insights
        """
        try:
            # Step 1: Get basic comparison from memory
            comparison = self.check_standards_met(technique, metrics)
            
            # Step 2: If deep research enabled, enhance with research and LLM reasoning
            if (use_deep_research if use_deep_research is not None else self.use_deep_research):
                enhanced_comparison = self._enhance_with_deep_research(
                    technique=technique,
                    metrics=metrics,
                    basic_comparison=comparison
                )
                return enhanced_comparison
            
            return comparison
        except Exception as e:
            logger.error(f"❌ Error comparing to standards: {e}")
            return {
                "all_met": False,
                "error": str(e),
                "gaps": [],
                "standards_checked": False
            }
    
    def _enhance_with_deep_research(
        self,
        technique: str,
        metrics: Dict[str, float],
        basic_comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance comparison with deep research and LLM reasoning.
        
        Args:
            technique: Technique name
            metrics: Calculated metrics
            basic_comparison: Basic comparison from memory retrieval
        
        Returns:
            Enhanced comparison with research findings and LLM insights
        """
        logger.info(f"🔬 Performing deep research for {technique}...")
        
        enhanced = basic_comparison.copy()
        enhanced["deep_research"] = {}
        enhanced["llm_insights"] = {}
        
        # Step 1: Get FIG standards from memory
        fig_standards = self.get_fig_standards(technique)
        enhanced["fig_standards_retrieved"] = fig_standards.get("found", False)
        
        # Step 2: Perform deep research (web search + reasoning)
        if self.deep_search:
            try:
                research_findings = self.deep_search.research_technique(
                    technique=technique,
                    sport="gymnastics",  # Can be inferred
                    user_requests=list(metrics.keys())
                )
                enhanced["deep_research"] = research_findings
                logger.info(f"✅ Deep research complete: {len(research_findings.get('web_search_results', []))} results")
            except Exception as e:
                logger.warning(f"⚠️  Deep research failed: {e}")
                enhanced["deep_research"]["error"] = str(e)
        
        # Step 3: Use LLM reasoning to generate insights
        if self.llm_instance or GEMINI_AVAILABLE or OPENAI_AVAILABLE:
            try:
                llm_insights = self._generate_llm_insights(
                    technique=technique,
                    metrics=metrics,
                    fig_standards=fig_standards,
                    basic_comparison=basic_comparison,
                    research_findings=enhanced.get("deep_research", {})
                )
                enhanced["llm_insights"] = llm_insights
                logger.info("✅ LLM insights generated")
            except Exception as e:
                logger.warning(f"⚠️  LLM insight generation failed: {e}")
                enhanced["llm_insights"]["error"] = str(e)
        
        return enhanced
    
    def _generate_llm_insights(
        self,
        technique: str,
        metrics: Dict[str, float],
        fig_standards: Dict[str, Any],
        basic_comparison: Dict[str, Any],
        research_findings: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use LLM reasoning model to generate insights from memory + research.
        
        Args:
            technique: Technique name
            metrics: Calculated metrics
            fig_standards: FIG standards from memory
            basic_comparison: Basic comparison results
            research_findings: Deep research findings
        
        Returns:
            LLM-generated insights with reasoning
        """
        logger.info("🧠 Using reasoning model to generate insights...")
        
        # Build comprehensive prompt
        prompt = self._build_insight_prompt(
            technique=technique,
            metrics=metrics,
            fig_standards=fig_standards,
            basic_comparison=basic_comparison,
            research_findings=research_findings
        )
        
        # Call LLM
        response = self._call_reasoning_model(prompt)
        
        # Parse response
        insights = self._parse_llm_insights(response)
        
        return insights
    
    def _build_insight_prompt(
        self,
        technique: str,
        metrics: Dict[str, float],
        fig_standards: Dict[str, Any],
        basic_comparison: Dict[str, Any],
        research_findings: Dict[str, Any]
    ) -> str:
        """Build prompt for LLM reasoning"""
        
        # Format metrics
        metrics_text = "\n".join([f"- {k}: {v:.2f}" for k, v in metrics.items()])
        
        # Format FIG standards
        fig_text = "Not found"
        if fig_standards.get("found"):
            fig_text = json.dumps(fig_standards.get("fig_standards", {}), indent=2)
        
        # Format gaps
        gaps_text = "None"
        gaps = basic_comparison.get("gaps", [])
        if gaps:
            gaps_text = "\n".join([
                f"- {g.get('metric', 'unknown')}: current={g.get('current_value', 0):.2f}, "
                f"target={g.get('target_value', 0):.2f}, gap={g.get('gap', 0):.2f}"
                for g in gaps[:5]
            ])
        
        # Format research findings
        research_text = "No research available"
        if research_findings.get("web_search_results"):
            research_text = "\n\n".join([
                f"**{r.get('title', 'Result')}**\n{r.get('content', '')[:300]}"
                for r in research_findings.get("web_search_results", [])[:3]
            ])
        
        prompt = f"""You are an expert gymnastics analyst with deep knowledge of FIG standards and biomechanics.

**Technique**: {technique}

**Calculated Metrics**:
{metrics_text}

**FIG Standards** (from memory index):
{fig_text}

**Gaps Identified**:
{gaps_text}

**Research Findings** (from web search):
{research_text}

**Your Task**: Provide deep insights with reasoning:

1. **Root Cause Analysis**: For each gap, explain the biomechanical root cause
2. **Technique Errors**: Identify specific technique errors causing the gaps
3. **Correlation**: How do the metrics correlate with each other?
4. **Research Integration**: How do the research findings relate to the FIG standards?
5. **Actionable Recommendations**: Specific coaching cues to address each gap

**Output Format** (JSON):
{{
  "root_causes": [
    {{
      "metric": "landing_knee_bend",
      "root_cause": "Insufficient knee extension during flight phase",
      "biomechanical_explanation": "...",
      "correlation_with_other_metrics": "..."
    }}
  ],
  "technique_errors": [
    "Early knee bend before landing",
    "Insufficient leg drive during takeoff"
  ],
  "research_insights": [
    "Research shows that landing with >20° knee bend increases injury risk",
    "Professional gymnasts maintain <10° knee bend at landing"
  ],
  "recommendations": [
    {{
      "priority": "high",
      "metric": "landing_knee_bend",
      "recommendation": "Focus on maintaining straight legs during flight phase",
      "coaching_cue": "Think 'stiff legs' during the entire flight"
    }}
  ],
  "reasoning_summary": "Overall analysis summary..."
}}
"""
        
        return prompt
    
    def _call_reasoning_model(self, prompt: str) -> str:
        """Call reasoning model (LLM)"""
        if self.llm_instance:
            # Use provided LLM instance (from vision-agents)
            return self._call_llm_instance(prompt)
        elif GEMINI_AVAILABLE:
            return self._call_gemini(prompt)
        elif OPENAI_AVAILABLE:
            return self._call_openai(prompt)
        else:
            logger.warning("⚠️  No reasoning model available")
            return "{}"
    
    def _call_llm_instance(self, prompt: str) -> str:
        """Call LLM instance if available"""
        # Would need to adapt based on vision-agents LLM interface
        return "{}"
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini reasoning model"""
        try:
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
            if not api_key:
                logger.warning("⚠️  GEMINI_API_KEY not set")
                return "{}"
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(self.reasoning_model)
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,  # Lower temperature for more focused reasoning
                    top_p=0.95,
                )
            )
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            return "{}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI reasoning model"""
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                logger.warning("⚠️  OPENAI_API_KEY not set")
                return "{}"
            
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert gymnastics analyst. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            return "{}"
    
    def _parse_llm_insights(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured insights"""
        try:
            # Try to extract JSON from response
            if "```json" in response:
                json_start = response.find("```json") + 7
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            elif "```" in response:
                json_start = response.find("```") + 3
                json_end = response.find("```", json_start)
                response = response[json_start:json_end].strip()
            
            insights = json.loads(response)
            return insights
        except Exception as e:
            logger.warning(f"⚠️  Failed to parse LLM insights: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return {
                "root_causes": [],
                "technique_errors": [],
                "research_insights": [],
                "recommendations": [],
                "reasoning_summary": "LLM parsing failed",
                "raw_response": response[:500]
            }
    
    def deep_research_technique(
        self,
        technique: str,
        user_requests: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Perform deep research on a technique using reasoning model.
        Combines memory retrieval with web search and LLM reasoning.
        
        Args:
            technique: Technique name
            user_requests: Optional user requests to guide research
        
        Returns:
            Comprehensive research findings with reasoning
        """
        logger.info(f"🔬 Performing deep research on {technique}...")
        
        research_result = {
            "technique": technique,
            "memory_retrieval": {},
            "web_research": {},
            "llm_synthesis": {}
        }
        
        # Step 1: Retrieve from memory index
        try:
            fig_standards = self.get_fig_standards(technique)
            relevant_metrics = self.get_relevant_metrics(technique)
            research_result["memory_retrieval"] = {
                "fig_standards": fig_standards,
                "relevant_metrics": relevant_metrics,
                "found": fig_standards.get("found", False)
            }
        except Exception as e:
            logger.warning(f"⚠️  Memory retrieval failed: {e}")
            research_result["memory_retrieval"]["error"] = str(e)
        
        # Step 2: Web research (if deep search available)
        if self.deep_search:
            try:
                web_research = self.deep_search.research_technique(
                    technique=technique,
                    sport="gymnastics",
                    user_requests=user_requests
                )
                research_result["web_research"] = web_research
            except Exception as e:
                logger.warning(f"⚠️  Web research failed: {e}")
                research_result["web_research"]["error"] = str(e)
        
        # Step 3: LLM synthesis of memory + web research
        if self.llm_instance or GEMINI_AVAILABLE or OPENAI_AVAILABLE:
            try:
                synthesis = self._synthesize_research_with_llm(research_result)
                research_result["llm_synthesis"] = synthesis
            except Exception as e:
                logger.warning(f"⚠️  LLM synthesis failed: {e}")
                research_result["llm_synthesis"]["error"] = str(e)
        
        return research_result
    
    def _synthesize_research_with_llm(self, research_result: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to synthesize memory retrieval + web research"""
        memory_data = research_result.get("memory_retrieval", {})
        web_data = research_result.get("web_research", {})
        
        prompt = f"""Synthesize the following research about {research_result['technique']}:

**Memory Retrieval (FIG Standards)**:
{json.dumps(memory_data.get('fig_standards', {}), indent=2)}

**Web Research**:
{json.dumps(web_data.get('web_search_results', [])[:3], indent=2)}

**Your Task**: Provide a comprehensive synthesis that:
1. Identifies key metrics to measure
2. Compares memory standards with web research findings
3. Highlights any discrepancies or additional insights
4. Recommends analysis approach

**Output Format** (JSON):
{{
  "key_metrics": ["metric1", "metric2"],
  "standards_comparison": {{"memory": "...", "web": "..."}},
  "insights": ["insight1", "insight2"],
  "recommended_analysis": ["step1", "step2"]
}}
"""
        
        response = self._call_reasoning_model(prompt)
        synthesis = self._parse_llm_insights(response)
        
        return synthesis
    
    def get_relevant_metrics(self, technique: str) -> List[str]:
        """Get relevant metrics for a technique"""
        try:
            return self.map_technique_to_metrics(technique)
        except Exception as e:
            logger.error(f"❌ Error getting relevant metrics: {e}")
            return []
    
    # Fallback methods if memory_retrieval is not available
    def _fallback_get_fig_standards(self, technique: str) -> Dict[str, Any]:
        """Fallback if memory retrieval not available"""
        return {
            "technique": technique,
            "found": False,
            "error": "Memory retrieval not available"
        }
    
    def _fallback_check_standards_met(self, technique: str, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Fallback if memory retrieval not available"""
        return {
            "all_met": False,
            "error": "Memory retrieval not available",
            "gaps": [],
            "standards_checked": False
        }
    
    def _fallback_get_technique_list(self) -> List[Dict[str, Any]]:
        """Fallback if memory retrieval not available"""
        return []
    
    def _fallback_map_technique_to_metrics(self, technique: str) -> List[str]:
        """Fallback if memory retrieval not available"""
        return []






















