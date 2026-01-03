#!/usr/bin/env python3
"""
Guardrails Module for LLM Response Validation

Implements multiple guardrail strategies:
1. Post-Generation Validation (Chain-of-Verification)
2. Confidence Scoring
3. Self-Reflection
4. Data Fact-Checking
5. Hallucination Detection
"""

import logging
import os
import json
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import sys
from pathlib import Path
from dotenv import load_dotenv

# Try to import LLM libraries
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()


class Guardrails:
    """
    Comprehensive guardrails system for LLM response validation.
    Implements Chain-of-Verification, confidence scoring, and fact-checking.
    """
    
    def __init__(self):
        """Initialize guardrails system."""
        self.min_confidence_threshold = 0.6  # Minimum confidence score
        self.verification_enabled = True
        self.confidence_scoring_enabled = True
        self.fact_checking_enabled = True
        
        logger.info("✅ Initialized Guardrails system")
    
    def validate_response(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any],
        prompt_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of LLM response.
        
        Args:
            response: LLM-generated response dictionary
            source_data: Original source data used to generate response
            prompt_context: Optional context from the original prompt
            
        Returns:
            Validation result with confidence score, fact-check results, and recommendations
        """
        validation_result = {
            "valid": True,
            "confidence_score": 0.0,
            "fact_check": {},
            "hallucination_detected": False,
            "warnings": [],
            "recommendations": [],
            "verified": False
        }
        
        # 1. Confidence Scoring
        if self.confidence_scoring_enabled:
            confidence_result = self._calculate_confidence_score(response, source_data)
            validation_result["confidence_score"] = confidence_result["score"]
            validation_result["confidence_breakdown"] = confidence_result["breakdown"]
            
            if confidence_result["score"] < self.min_confidence_threshold:
                validation_result["valid"] = False
                validation_result["warnings"].append(
                    f"Low confidence score ({confidence_result['score']:.2f}) below threshold ({self.min_confidence_threshold})"
                )
                validation_result["recommendations"].append(
                    "Consider regenerating response with more source data or clarifying prompt"
                )
        
        # 2. Fact-Checking against source data
        if self.fact_checking_enabled:
            fact_check_result = self._fact_check_against_source(response, source_data)
            validation_result["fact_check"] = fact_check_result
            
            if fact_check_result.get("factual_errors", 0) > 0:
                validation_result["valid"] = False
                validation_result["hallucination_detected"] = True
                validation_result["warnings"].append(
                    f"Detected {fact_check_result['factual_errors']} factual errors in response"
                )
        
        # 3. Self-Reflection / Chain-of-Verification
        if self.verification_enabled and validation_result["confidence_score"] < 0.8:
            verification_result = self._chain_of_verification(
                response, source_data, prompt_context
            )
            validation_result["verification"] = verification_result
            validation_result["verified"] = verification_result.get("verified", False)
            
            if not verification_result.get("verified", False):
                validation_result["warnings"].append(
                    "Response failed Chain-of-Verification check"
                )
        
        # 4. Data Reference Validation
        data_ref_result = self._validate_data_references(response, source_data)
        validation_result["data_references"] = data_ref_result
        
        if not data_ref_result.get("has_sufficient_references", False):
            validation_result["warnings"].append(
                "Response may not reference sufficient source data"
            )
        
        # 5. Hallucination Detection
        hallucination_result = self._detect_hallucinations(response, source_data)
        validation_result["hallucination_check"] = hallucination_result
        
        if hallucination_result.get("hallucination_risk", "low") in ["high", "medium"]:
            validation_result["hallucination_detected"] = True
            validation_result["warnings"].append(
                f"Potential hallucination detected (risk: {hallucination_result['hallucination_risk']})"
            )
        
        return validation_result
    
    def _calculate_confidence_score(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate confidence score for the response.
        
        Args:
            response: LLM response
            source_data: Source data
            
        Returns:
            Confidence score and breakdown
        """
        score_components = {
            "data_coverage": 0.0,  # How much of source data is referenced
            "specificity": 0.0,    # How specific vs generic the response is
            "consistency": 0.0,     # Internal consistency
            "completeness": 0.0     # Completeness of response
        }
        
        # 1. Data Coverage: Check if response references source data
        response_text = json.dumps(response, default=str).lower()
        source_keys = list(source_data.keys())
        referenced_keys = sum(1 for key in source_keys if key.lower() in response_text)
        
        if source_keys:
            score_components["data_coverage"] = min(1.0, referenced_keys / len(source_keys))
        else:
            score_components["data_coverage"] = 0.5  # Neutral if no source keys
        
        # 2. Specificity: Check for specific values vs generic statements
        # Look for numbers, specific measurements, timestamps
        has_numbers = bool(re.search(r'\d+\.?\d*', response_text))
        has_specific_values = bool(re.search(r'\d+\.\d+', response_text))  # Decimal numbers
        
        score_components["specificity"] = 0.3 if has_numbers else 0.1
        score_components["specificity"] += 0.4 if has_specific_values else 0.0
        
        # 3. Consistency: Check for contradictions
        # Simple check: look for conflicting statements
        consistency_score = 1.0
        if "increased" in response_text and "decreased" in response_text:
            # Check if they're referring to different metrics (acceptable) or same (contradiction)
            consistency_score = 0.7
        
        score_components["consistency"] = consistency_score
        
        # 4. Completeness: Check if response has required fields
        required_fields = ["observation", "evidence_reasoning", "coaching_options"]
        if isinstance(response, dict):
            present_fields = sum(1 for field in required_fields if field in response)
            score_components["completeness"] = present_fields / len(required_fields)
        else:
            score_components["completeness"] = 0.5
        
        # Weighted average
        weights = {
            "data_coverage": 0.4,
            "specificity": 0.3,
            "consistency": 0.2,
            "completeness": 0.1
        }
        
        overall_score = sum(
            score_components[key] * weights[key]
            for key in score_components
        )
        
        return {
            "score": round(overall_score, 3),
            "breakdown": score_components,
            "weights": weights
        }
    
    def _fact_check_against_source(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Fact-check response against source data.
        
        Args:
            response: LLM response
            source_data: Source data to check against
            
        Returns:
            Fact-check results
        """
        factual_errors = 0
        verified_facts = 0
        unverifiable_facts = 0
        errors = []
        
        response_text = json.dumps(response, default=str)
        
        # Extract numeric claims from response
        numeric_claims = re.findall(r'(\d+\.?\d*)', response_text)
        
        # Check if numeric values match source data
        for claim in numeric_claims[:10]:  # Limit to first 10 claims
            try:
                value = float(claim)
                # Check if this value exists in source data (within tolerance)
                found = False
                for key, source_value in source_data.items():
                    if isinstance(source_value, (int, float)):
                        if abs(source_value - value) < 0.01:  # Tolerance for floating point
                            found = True
                            verified_facts += 1
                            break
                    elif isinstance(source_value, (list, dict)):
                        # Check nested structures
                        source_str = json.dumps(source_value, default=str)
                        if claim in source_str:
                            found = True
                            verified_facts += 1
                            break
                
                if not found and value > 0:  # Non-zero value not found
                    unverifiable_facts += 1
            except ValueError:
                pass
        
        # Check for claims about trends/directions
        if "evidence_reasoning" in response:
            evidence = response["evidence_reasoning"]
            
            # Extract trend claims
            if "increased" in evidence.lower() or "decreased" in evidence.lower():
                # Verify against source data trends
                if "trend_statistics" in source_data:
                    trend_stats = source_data["trend_statistics"]
                    direction = trend_stats.get("direction", "")
                    
                    if direction:
                        if "increased" in evidence.lower() and direction != "increasing":
                            factual_errors += 1
                            errors.append("Trend direction mismatch in evidence_reasoning")
                        elif "decreased" in evidence.lower() and direction != "decreasing":
                            factual_errors += 1
                            errors.append("Trend direction mismatch in evidence_reasoning")
        
        return {
            "factual_errors": factual_errors,
            "verified_facts": verified_facts,
            "unverifiable_facts": unverifiable_facts,
            "errors": errors,
            "status": "passed" if factual_errors == 0 else "failed"
        }
    
    def _chain_of_verification(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any],
        prompt_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Chain-of-Verification: Use separate LLM call to verify response.
        
        Args:
            response: LLM response to verify
            source_data: Source data
            prompt_context: Original prompt context
            
        Returns:
            Verification results
        """
        # Build verification prompt
        verification_prompt = f"""You are a fact-checker verifying an LLM-generated response against source data.

## Source Data
{json.dumps(source_data, indent=2, default=str)}

## Generated Response
{json.dumps(response, indent=2, default=str)}

## Your Task
Verify the accuracy of the generated response by checking:
1. Are all numeric claims supported by the source data?
2. Are all trend directions consistent with source data?
3. Are there any claims that cannot be verified from the source data?
4. Are there any contradictions between the response and source data?

Provide a JSON response:
{{
  "verified": true/false,
  "verified_claims": ["list of claims that are verified"],
  "unverified_claims": ["list of claims that cannot be verified"],
  "errors": ["list of factual errors found"],
  "confidence": 0.0-1.0,
  "reasoning": "explanation of verification results"
}}"""

        try:
            verification_response = self._call_llm_for_verification(verification_prompt)
            
            # Parse verification response
            verification_result = json.loads(verification_response)
            return verification_result
            
        except Exception as e:
            logger.warning(f"⚠️  Chain-of-Verification failed: {e}")
            return {
                "verified": False,
                "error": str(e),
                "reasoning": "Verification process failed"
            }
    
    def _validate_data_references(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate that response references source data.
        
        Args:
            response: LLM response
            source_data: Source data
            
        Returns:
            Data reference validation results
        """
        response_text = json.dumps(response, default=str).lower()
        
        # Extract key values from source data
        source_values = []
        for key, value in source_data.items():
            if isinstance(value, (int, float)):
                source_values.append(str(value))
            elif isinstance(value, str):
                source_values.append(value.lower())
        
        # Check how many source values are referenced
        referenced_count = sum(1 for val in source_values if val in response_text)
        
        has_sufficient_references = referenced_count >= min(2, len(source_values) * 0.3)
        
        return {
            "has_sufficient_references": has_sufficient_references,
            "referenced_count": referenced_count,
            "total_source_values": len(source_values),
            "reference_ratio": referenced_count / len(source_values) if source_values else 0.0
        }
    
    def _detect_hallucinations(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect potential hallucinations in response.
        
        Args:
            response: LLM response
            source_data: Source data
            
        Returns:
            Hallucination detection results
        """
        hallucination_indicators = []
        risk_level = "low"
        
        response_text = json.dumps(response, default=str).lower()
        
        # Check for generic template phrases that might indicate hallucination
        generic_phrases = [
            "this pattern often appears",
            "consider reducing",
            "consider adding",
            "may indicate changes",
            "further analysis recommended"
        ]
        
        generic_count = sum(1 for phrase in generic_phrases if phrase in response_text)
        
        # Check if response has specific data references
        has_specific_data = bool(re.search(r'\d+\.\d+', response_text))
        
        # Check for claims not in source data
        if "evidence_reasoning" in response:
            evidence = response["evidence_reasoning"]
            
            # Extract all numbers from evidence
            evidence_numbers = re.findall(r'\d+\.?\d*', evidence)
            
            # Check if numbers match source data
            source_numbers = []
            for value in source_data.values():
                if isinstance(value, (int, float)):
                    source_numbers.append(str(value))
            
            unmatched_numbers = [
                num for num in evidence_numbers
                if num not in source_numbers and float(num) > 0
            ]
            
            if unmatched_numbers:
                hallucination_indicators.append(
                    f"Found {len(unmatched_numbers)} numeric claims not in source data"
                )
        
        # Determine risk level
        if generic_count >= 3 and not has_specific_data:
            risk_level = "high"
        elif generic_count >= 2 or unmatched_numbers:
            risk_level = "medium"
        
        return {
            "hallucination_risk": risk_level,
            "indicators": hallucination_indicators,
            "generic_phrase_count": generic_count,
            "has_specific_data": has_specific_data
        }
    
    def _call_llm_for_verification(self, prompt: str) -> str:
        """
        Call LLM for verification (Chain-of-Verification).
        
        Args:
            prompt: Verification prompt
            
        Returns:
            LLM response
        """
        # Try Gemini first
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.1,  # Very low temperature for verification
                            top_p=0.9,
                        )
                    )
                    return response.text
            except Exception as e:
                logger.warning(f"⚠️  Gemini verification call failed: {e}")
        
        # Fallback to OpenAI
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a fact-checker. Analyze ONLY the provided data. Respond with valid JSON only."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        response_format={"type": "json_object"},
                        temperature=0.1  # Very low temperature for verification
                    )
                    return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"⚠️  OpenAI verification call failed: {e}")
        
        return '{"verified": false, "error": "No LLM available"}'
    
    def apply_guardrails(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any],
        prompt_context: Optional[str] = None,
        auto_fix: bool = False
    ) -> Dict[str, Any]:
        """
        Apply all guardrails and optionally auto-fix issues.
        
        Args:
            response: LLM response
            source_data: Source data
            prompt_context: Original prompt context
            auto_fix: Whether to attempt automatic fixes
            
        Returns:
            Guardrails result with validated/improved response
        """
        # Run validation
        validation = self.validate_response(response, source_data, prompt_context)
        
        result = {
            "original_response": response,
            "validation": validation,
            "final_response": response,
            "improved": False
        }
        
        # Auto-fix if enabled and issues detected
        if auto_fix and not validation["valid"]:
            improved_response = self._attempt_auto_fix(
                response, source_data, validation
            )
            
            if improved_response:
                # Re-validate improved response
                improved_validation = self.validate_response(
                    improved_response, source_data, prompt_context
                )
                
                if improved_validation["valid"] or improved_validation["confidence_score"] > validation["confidence_score"]:
                    result["final_response"] = improved_response
                    result["improved"] = True
                    result["improved_validation"] = improved_validation
        
        return result
    
    def _attempt_auto_fix(
        self,
        response: Dict[str, Any],
        source_data: Dict[str, Any],
        validation: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Attempt to automatically fix issues in response.
        
        Args:
            response: Original response
            source_data: Source data
            validation: Validation results
            
        Returns:
            Improved response or None
        """
        improved = response.copy()
        
        # Fix: Add missing data references
        if not validation.get("data_references", {}).get("has_sufficient_references", False):
            # Add actual values from source data to evidence_reasoning
            if "evidence_reasoning" in improved:
                # Extract key values from source data
                key_values = []
                if "trend_statistics" in source_data:
                    stats = source_data["trend_statistics"]
                    key_values.append(f"first mean: {stats.get('first_mean', 0):.2f}")
                    key_values.append(f"second mean: {stats.get('second_mean', 0):.2f}")
                    key_values.append(f"change: {stats.get('change_percent', 0):.1f}%")
                
                if key_values:
                    data_context = f"Data shows: {', '.join(key_values)}. "
                    improved["evidence_reasoning"] = data_context + improved["evidence_reasoning"]
        
        # Fix: Remove unverified claims
        fact_check = validation.get("fact_check", {})
        if fact_check.get("factual_errors", 0) > 0:
            # Remove or correct factual errors
            # This is a simple fix - in production, might need more sophisticated approach
            logger.info("⚠️  Factual errors detected - manual review recommended")
        
        return improved


def create_guardrails() -> Guardrails:
    """Factory function to create guardrails instance."""
    return Guardrails()

