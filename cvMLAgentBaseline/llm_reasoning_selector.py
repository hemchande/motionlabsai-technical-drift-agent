#!/usr/bin/env python3
"""
LLM-Based Model Selector with Reasoning
Uses LLM to reason about model selection and workflow sequence.
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
import json

logger = logging.getLogger(__name__)

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


class LLMReasoningSelector:
    """
    Uses LLM to reason about:
    1. Which models to select based on user requests
    2. Sequence of processing steps
    3. How to combine model outputs
    """
    
    def __init__(self, llm_instance=None, llm_provider="gemini"):
        self.llm_instance = llm_instance
        self.llm_provider = llm_provider
        self.available_models = self._get_available_models()
    
    def _get_available_models(self) -> Dict[str, List[str]]:
        """Get list of available models"""
        return {
            "pose_estimation": ["hrnet", "yolo_pose", "openpose"],
            "person_detection": ["hrnet_person_detector", "yolo_person"],
            "face_detection": ["opencv_dnn_face"],
            "weight_estimation": ["pose_based_weight", "body_shape_estimation"],
            "segmentation": ["yolo_seg", "deeplab"],
            "action_recognition": ["action_net", "slowfast"]
        }
    
    def reason_about_model_selection(
        self,
        user_requests: List[str],
        technique: str,
        research_findings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to reason about which models to select.
        
        Args:
            user_requests: User's requested metrics/analyses
            technique: Technique name
            research_findings: Optional research findings from deep search
        
        Returns:
            Reasoning result with selected models and justification
        """
        logger.info("🧠 Using LLM to reason about model selection...")
        
        # Build prompt
        prompt = self._build_selection_prompt(user_requests, technique, research_findings)
        
        # Call LLM
        response = self._call_llm(prompt)
        
        # Parse response
        reasoning = self._parse_llm_response(response)
        
        logger.info(f"✅ LLM reasoning complete: {len(reasoning.get('selected_models', {}))} models selected")
        
        return reasoning
    
    def _build_selection_prompt(
        self,
        user_requests: List[str],
        technique: str,
        research_findings: Optional[Dict[str, Any]]
    ) -> str:
        """Build prompt for LLM reasoning"""
        prompt = f"""You are an expert ML engineer selecting computer vision models for sports technique analysis.

**Task**: Select the best models and plan the execution sequence for analyzing: {technique}

**User Requests**:
{chr(10).join(f"- {req}" for req in user_requests)}

**Available Models**:
"""
        
        for model_type, variants in self.available_models.items():
            prompt += f"\n{model_type}:\n"
            for variant in variants:
                prompt += f"  - {variant}\n"
        
        if research_findings:
            prompt += f"\n**Research Findings**:\n"
            prompt += f"Key metrics to measure: {', '.join(research_findings.get('key_metrics', []))}\n"
            prompt += f"Standards: {json.dumps(research_findings.get('standards', {}), indent=2)}\n"
        
        prompt += """
**Your Task**:
1. Select the best models for each required task
2. Explain your reasoning for each selection
3. Plan the execution sequence (what runs first, dependencies)
4. Consider: accuracy, speed, resource requirements

**Output Format** (JSON):
{
  "selected_models": {
    "pose_estimation": "hrnet",
    "person_detection": "hrnet_person_detector",
    ...
  },
  "reasoning": {
    "pose_estimation": "Selected HRNet because it provides high accuracy for joint angles needed for landing bend analysis",
    ...
  },
  "execution_sequence": [
    {
      "step": 1,
      "tool": "person_detection",
      "model": "hrnet_person_detector",
      "reason": "Need to locate athlete first",
      "depends_on": []
    },
    ...
  ]
}
"""
        
        return prompt
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        if self.llm_instance:
            # Use provided LLM instance (from vision-agents)
            # This would need to be adapted based on the interface
            return self._call_llm_instance(prompt)
        elif GEMINI_AVAILABLE:
            return self._call_gemini(prompt)
        elif OPENAI_AVAILABLE:
            return self._call_openai(prompt)
        else:
            logger.warning("⚠️  No LLM available - returning empty reasoning")
            return "{}"
    
    def _call_llm_instance(self, prompt: str) -> str:
        """Call LLM instance if available"""
        # Would need to adapt based on vision-agents LLM interface
        return "{}"
    
    def _call_gemini(self, prompt: str) -> str:
        """Call Gemini API"""
        try:
            import os
            genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY"))
            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"❌ Gemini API call failed: {e}")
            return "{}"
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an expert ML engineer. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"❌ OpenAI API call failed: {e}")
            return "{}"
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response into structured reasoning"""
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
            
            reasoning = json.loads(response)
            return reasoning
        except Exception as e:
            logger.warning(f"⚠️  Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            return {
                "selected_models": {},
                "reasoning": {},
                "execution_sequence": []
            }





















