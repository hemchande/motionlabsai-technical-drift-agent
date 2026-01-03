#!/usr/bin/env python3
"""
Trend Tracker for ACL Risk Patterns

Tracks trends across multiple sessions with three-layer output format:
1. Observation (measurement)
2. Evidence & reasoning (non-clinical)
3. Coaching options (not prescriptions)

Auto-monitors trends for N sessions and shows status:
- Improving / Unchanged / Worsening / Insufficient data
"""

import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to import MongoDBService
sys.path.insert(0, str(Path(__file__).parent.parent))
from videoAgent.mongodb_service import MongoDBService
from bson import ObjectId

# Import guardrails
try:
    from guardrails import Guardrails
    GUARDRAILS_AVAILABLE = True
except ImportError:
    GUARDRAILS_AVAILABLE = False
    Guardrails = None

# Load environment variables
load_dotenv()

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


class TrendTracker:
    """
    Tracks trends across multiple sessions with deep reasoning.
    Stores trends in MongoDB and monitors their status over time.
    """
    
    def __init__(self):
        """Initialize the trend tracker."""
        self.mongodb = MongoDBService()
        if not self.mongodb.connect():
            raise RuntimeError("Failed to connect to MongoDB")
        
        # Initialize guardrails if available
        if GUARDRAILS_AVAILABLE:
            try:
                self.guardrails = Guardrails()
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize Guardrails: {e}")
                self.guardrails = None
        else:
            self.guardrails = None
        
        logger.info("✅ Initialized TrendTracker")
    
    def identify_trends_from_sessions(
        self,
        sessions: List[Dict[str, Any]],
        min_sessions: int = 3,
        athlete_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Identify trends from multiple sessions with insights.
        
        Args:
            sessions: List of session documents with ACL risk data
            min_sessions: Minimum number of sessions required to identify a trend
            athlete_name: Optional athlete name filter
            
        Returns:
            List of identified trend documents
        """
        if len(sessions) < min_sessions:
            logger.info(f"⚠️  Insufficient sessions ({len(sessions)} < {min_sessions}) to identify trends")
            return []
        
        # Group sessions by athlete if not provided
        if not athlete_name:
            athlete_sessions = defaultdict(list)
            for session in sessions:
                athlete = session.get("athlete_name") or session.get("activity") or "unknown"
                athlete_sessions[athlete].append(session)
        else:
            athlete_sessions = {athlete_name: sessions}
        
        all_trends = []
        
        for athlete, athlete_sess in athlete_sessions.items():
            if len(athlete_sess) < min_sessions:
                continue
            
            # Sort sessions by timestamp
            athlete_sess.sort(key=lambda s: self._parse_timestamp(s.get("timestamp", "")))
            
            # Identify metric trends
            trends = self._identify_metric_trends(athlete_sess, athlete)
            all_trends.extend(trends)
        
        logger.info(f"📊 Identified {len(all_trends)} trends from {len(sessions)} sessions")
        return all_trends
    
    def _identify_metric_trends(
        self,
        sessions: List[Dict[str, Any]],
        athlete_name: str
    ) -> List[Dict[str, Any]]:
        """
        Identify metric trends from sessions.
        
        Args:
            sessions: List of sessions for one athlete
            athlete_name: Athlete name
            
        Returns:
            List of trend documents
        """
        trends = []
        
        # Track key metrics across sessions
        metric_data = defaultdict(list)
        
        for session in sessions:
            timestamp = self._parse_timestamp(session.get("timestamp", ""))
            technique = session.get("technique") or session.get("activity", "unknown")
            
            # Extract ACL risk metrics
            acl_risk_score = session.get("acl_risk_score") or session.get("metrics", {}).get("acl_tear_risk_score", 0.0)
            max_valgus = session.get("acl_max_valgus_angle") or session.get("metrics", {}).get("acl_max_valgus_angle", 0.0)
            risk_moments = session.get("risk_moments", [])
            high_risk_count = len([m for m in risk_moments if m.get("risk_score", 0.0) >= 0.7])
            moderate_risk_count = len([m for m in risk_moments if 0.4 <= m.get("risk_score", 0.0) < 0.7])
            
            # Track valgus angle trend
            if max_valgus > 0:
                metric_data[f"valgus_angle_{technique}"].append({
                    "timestamp": timestamp,
                    "value": max_valgus,
                    "session_id": session.get("session_id") or str(session.get("_id", "")),
                    "risk_score": acl_risk_score
                })
            
            # Track risk score trend
            if acl_risk_score > 0:
                metric_data[f"risk_score_{technique}"].append({
                    "timestamp": timestamp,
                    "value": acl_risk_score,
                    "session_id": session.get("session_id") or str(session.get("_id", "")),
                    "high_risk_count": high_risk_count,
                    "moderate_risk_count": moderate_risk_count
                })
            
            # Track high risk moment frequency
            if high_risk_count > 0:
                metric_data[f"high_risk_frequency_{technique}"].append({
                    "timestamp": timestamp,
                    "value": high_risk_count,
                    "session_id": session.get("session_id") or str(session.get("_id", "")),
                    "total_moments": len(risk_moments)
                })
        
        # Analyze each metric for trends
        for metric_key, data_points in metric_data.items():
            if len(data_points) < 3:  # Need at least 3 data points
                continue
            
            # Sort by timestamp
            data_points.sort(key=lambda x: x["timestamp"])
            
            # Extract technique from metric key
            technique = metric_key.split("_", -1)[-1] if "_" in metric_key else "unknown"
            metric_type = "_".join(metric_key.split("_")[:-1]) if "_" in metric_key else metric_key
            
            # Calculate trend
            trend = self._calculate_trend_statistics(data_points, metric_type)
            
            if trend:
                trend["athlete_name"] = athlete_name
                trend["technique"] = technique
                trend["metric_signature"] = f"{athlete_name}_{metric_type}_{technique}"
                trend["data_points"] = data_points
                trend["session_count"] = len(data_points)
                
                # Generate three-layer output with deep reasoning
                three_layer_output = self._generate_three_layer_output(trend, data_points, sessions)
                trend.update(three_layer_output)
                
                # Determine trend status
                trend["status"] = self._determine_trend_status(trend, data_points)
                
                trends.append(trend)
        
        return trends
    
    def _calculate_trend_statistics(
        self,
        data_points: List[Dict[str, Any]],
        metric_type: str
    ) -> Optional[Dict[str, Any]]:
        """
        Calculate trend statistics from data points.
        
        Args:
            data_points: List of data point dictionaries with timestamp and value
            metric_type: Type of metric being tracked
            
        Returns:
            Trend statistics dictionary or None
        """
        if len(data_points) < 3:
            return None
        
        values = [dp["value"] for dp in data_points]
        timestamps = [dp["timestamp"] for dp in data_points]
        
        # Calculate basic statistics
        first_half = values[:len(values)//2]
        second_half = values[len(values)//2:]
        
        first_mean = sum(first_half) / len(first_half) if first_half else 0
        second_mean = sum(second_half) / len(second_half) if second_half else 0
        
        overall_mean = sum(values) / len(values)
        overall_min = min(values)
        overall_max = max(values)
        
        # Calculate change
        change = second_mean - first_mean
        change_percent = (change / first_mean * 100) if first_mean > 0 else 0
        
        # Determine direction
        if abs(change_percent) < 5:  # Less than 5% change is considered unchanged
            direction = "stable"
        elif change > 0:
            direction = "increasing" if "risk" in metric_type.lower() or "valgus" in metric_type.lower() else "increasing"
        else:
            direction = "decreasing" if "risk" in metric_type.lower() or "valgus" in metric_type.lower() else "decreasing"
        
        return {
            "metric_type": metric_type,
            "first_mean": first_mean,
            "second_mean": second_mean,
            "overall_mean": overall_mean,
            "overall_min": overall_min,
            "overall_max": overall_max,
            "change": change,
            "change_percent": change_percent,
            "direction": direction,
            "data_point_count": len(data_points)
        }
    
    def _generate_three_layer_output(
        self,
        trend: Dict[str, Any],
        data_points: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate three-layer output with deep reasoning:
        1. Observation (measurement)
        2. Evidence & reasoning (non-clinical)
        3. Coaching options (not prescriptions)
        
        Args:
            trend: Trend statistics dictionary
            data_points: List of data points
            sessions: List of session documents
            
        Returns:
            Dictionary with three-layer output
        """
        metric_type = trend["metric_type"]
        first_mean = trend["first_mean"]
        second_mean = trend["second_mean"]
        change_percent = abs(trend["change_percent"])
        direction = trend["direction"]
        session_count = len(data_points)
        
        # Build observation
        if "valgus" in metric_type.lower():
            observation = f"Knee valgus {'increased' if direction == 'increasing' else 'decreased' if direction == 'decreasing' else 'remained stable'} from ~{first_mean:.1f}° → ~{second_mean:.1f}° on {session_count} sessions."
        elif "risk_score" in metric_type.lower():
            observation = f"ACL risk score {'increased' if direction == 'increasing' else 'decreased' if direction == 'decreasing' else 'remained stable'} from ~{first_mean:.2f} → ~{second_mean:.2f} across {session_count} sessions."
        elif "high_risk_frequency" in metric_type.lower():
            observation = f"High-risk moment frequency {'increased' if direction == 'increasing' else 'decreased' if direction == 'decreasing' else 'remained stable'} from ~{first_mean:.1f} → ~{second_mean:.1f} occurrences per session across {session_count} sessions."
        else:
            observation = f"{metric_type} {'increased' if direction == 'increasing' else 'decreased' if direction == 'decreasing' else 'remained stable'} from ~{first_mean:.2f} → ~{second_mean:.2f} across {session_count} sessions."
        
        # Use LLM for deep reasoning to generate evidence & reasoning and coaching options
        reasoning_output = self._generate_deep_reasoning(trend, data_points, sessions, observation)
        
        return {
            "observation": observation,
            "evidence_reasoning": reasoning_output.get("evidence_reasoning", ""),
            "coaching_options": reasoning_output.get("coaching_options", []),
            "generated_at": datetime.utcnow().isoformat()
        }
    
    def _generate_deep_reasoning(
        self,
        trend: Dict[str, Any],
        data_points: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]],
        observation: str
    ) -> Dict[str, Any]:
        """
        Use LLM for deep reasoning to generate evidence & reasoning and coaching options.
        
        Args:
            trend: Trend statistics dictionary
            data_points: List of data points
            sessions: List of session documents
            observation: Observation text
            
        Returns:
            Dictionary with evidence_reasoning and coaching_options
        """
        # Build context for LLM
        metric_type = trend["metric_type"]
        technique = trend.get("technique", "unknown")
        athlete_name = trend.get("athlete_name", "unknown")
        
        # Extract detailed session data with actual measurements
        session_data = []
        for session in sessions[:10]:  # Limit to first 10
            risk_moments = session.get("risk_moments", [])
            high_risk = [m for m in risk_moments if m.get("risk_score", 0.0) >= 0.7]
            moderate_risk = [m for m in risk_moments if 0.4 <= m.get("risk_score", 0.0) < 0.7]
            
            session_data.append({
                "session_id": session.get("session_id") or str(session.get("_id", "")),
                "timestamp": session.get("timestamp", ""),
                "technique": session.get("technique", "unknown"),
                "acl_risk_score": round(session.get("acl_risk_score", 0.0), 3),
                "max_valgus_angle": round(session.get("acl_max_valgus_angle", 0.0), 1),
                "total_risk_moments": len(risk_moments),
                "high_risk_count": len(high_risk),
                "moderate_risk_count": len(moderate_risk),
                "primary_risk_factors": list(set([
                    factor
                    for moment in risk_moments[:5]  # Limit to first 5 moments
                    for factor in moment.get("primary_risk_factors", [])
                ]))
            })
        
        # Extract actual data point values for the metric being tracked
        metric_values = []
        for dp in data_points:
            metric_values.append({
                "timestamp": dp.get("timestamp", ""),
                "value": round(dp.get("value", 0.0), 3),
                "session_id": dp.get("session_id", "")
            })
        
        # Calculate actual statistics from data
        values = [dp["value"] for dp in data_points]
        first_half_values = values[:len(values)//2]
        second_half_values = values[len(values)//2:]
        
        prompt = f"""You are an expert sports biomechanist analyzing ACL injury risk patterns. Analyze the ACTUAL data provided below and generate evidence-based reasoning and coaching options.

## Trend Observation
{observation}

## Actual Data Points (Use These Exact Values)
{json.dumps(metric_values, indent=2)}

## Calculated Statistics
- First half mean: {trend.get('first_mean', 0):.3f}
- Second half mean: {trend.get('second_mean', 0):.3f}
- Overall mean: {trend.get('overall_mean', 0):.3f}
- Minimum value: {trend.get('overall_min', 0):.3f}
- Maximum value: {trend.get('overall_max', 0):.3f}
- Change: {trend.get('change', 0):.3f} ({trend.get('change_percent', 0):.1f}%)
- Direction: {trend.get('direction', 'unknown')}

## Context
- Athlete: {athlete_name}
- Technique: {technique}
- Metric Type: {metric_type}
- Number of sessions: {len(data_points)}

## Session Details (Use These Actual Measurements)
{json.dumps(session_data, indent=2)}

## Your Task
Analyze ONLY the data provided above. Do NOT use generic examples or repeat template text. Base your analysis on the ACTUAL measurements and values shown.

Provide a JSON response with:

1. **evidence_reasoning**: A non-clinical explanation based on the ACTUAL data values shown above. Reference specific measurements when relevant. Focus on biomechanical factors and movement patterns. Use language like "The {metric_type} values show..." or "The data indicates..." based on the actual numbers provided. Do NOT make clinical diagnoses.

2. **coaching_options**: An array of 2-5 specific coaching considerations based on the ACTUAL trend pattern observed. Each should be tailored to the specific metric type ({metric_type}) and technique ({technique}). Always include: "If pain/symptoms are present, consult your AT/PT." Use language like "Consider..." not "Do..." or "Must...".

## Critical Instructions
- Use ONLY the actual data values provided above
- Reference specific measurements from the data points
- Do NOT repeat generic examples or template text
- Do NOT hallucinate data that isn't in the provided measurements
- Base reasoning on the actual trend direction and change percentage shown
- Tailor coaching options to the specific metric ({metric_type}) and technique ({technique})
- Use non-clinical, biomechanical language
- Never use language like "prevented injury" or "will cause injury"

Provide your response in valid JSON format only:"""

        # Call LLM
        llm_response = self._call_llm_for_reasoning(prompt)
        
        # Parse response
        try:
            response = llm_response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            response = response.strip()
            
            reasoning = json.loads(response)
            
            # Apply guardrails validation
            if self.guardrails:
                source_data_for_validation = {
                    "trend_statistics": {
                        "first_mean": trend.get("first_mean", 0),
                        "second_mean": trend.get("second_mean", 0),
                        "overall_mean": trend.get("overall_mean", 0),
                        "change": trend.get("change", 0),
                        "change_percent": trend.get("change_percent", 0),
                        "direction": trend.get("direction", "stable")
                    },
                    "metric_type": metric_type,
                    "technique": technique,
                    "athlete_name": athlete_name,
                    "data_points": metric_values
                }
                
                guardrails_result = self.guardrails.apply_guardrails(
                    reasoning,
                    source_data_for_validation,
                    prompt_context=observation,
                    auto_fix=True
                )
                
                # Use improved response if available
                if guardrails_result.get("improved", False):
                    reasoning = guardrails_result["final_response"]
                    logger.info(f"✅ Guardrails improved response (confidence: {guardrails_result['validation'].get('confidence_score', 0):.2f})")
                else:
                    validation = guardrails_result.get("validation", {})
                    confidence = validation.get("confidence_score", 0)
                    
                    if confidence < 0.6:
                        logger.warning(f"⚠️  Low confidence response ({confidence:.2f}) - consider manual review")
                    
                    if validation.get("hallucination_detected", False):
                        logger.warning("⚠️  Potential hallucination detected in response")
            
            # Validate that reasoning references actual data (not just generic text)
            evidence = reasoning.get("evidence_reasoning", "")
            # Check if evidence mentions actual values or is too generic
            has_data_reference = any(
                keyword in evidence.lower() 
                for keyword in ["value", "measurement", "data", "session", str(trend.get('first_mean', '')), str(trend.get('second_mean', ''))]
            )
            
            # If evidence seems too generic, enhance it with actual data
            if not has_data_reference and evidence:
                metric_type = trend.get("metric_type", "")
                first_mean = trend.get("first_mean", 0)
                second_mean = trend.get("second_mean", 0)
                change_percent = trend.get("change_percent", 0)
                
                # Prepend actual data context
                data_context = f"The {metric_type} measurements show a change from {first_mean:.2f} to {second_mean:.2f} ({change_percent:+.1f}%). "
                reasoning["evidence_reasoning"] = data_context + evidence
            
            # Ensure coaching_options is a list and includes AT/PT option
            if not isinstance(reasoning.get("coaching_options"), list):
                reasoning["coaching_options"] = []
            
            # Validate coaching options are specific to the metric/technique
            metric_type = trend.get("metric_type", "")
            technique = trend.get("technique", "unknown")
            
            # Filter out overly generic options if we have specific ones
            if len(reasoning.get("coaching_options", [])) > 0:
                # Keep all options but ensure they're relevant
                filtered_options = []
                for opt in reasoning.get("coaching_options", []):
                    # Keep if it mentions the technique, metric type, or is the AT/PT option
                    if (technique.lower() in opt.lower() or 
                        metric_type.lower() in opt.lower() or 
                        "at/pt" in opt.lower() or 
                        "consult" in opt.lower() or
                        len(reasoning.get("coaching_options", [])) <= 3):  # Keep if we have few options
                        filtered_options.append(opt)
                
                reasoning["coaching_options"] = filtered_options if filtered_options else reasoning.get("coaching_options", [])
            
            # Add AT/PT consultation if not present
            at_pt_present = any("AT/PT" in opt or "at/pt" in opt.lower() or "consult" in opt.lower() for opt in reasoning.get("coaching_options", []))
            if not at_pt_present:
                reasoning["coaching_options"].append("If pain/symptoms are present, consult your AT/PT.")
            
            return reasoning
            
        except json.JSONDecodeError as e:
            logger.warning(f"⚠️  Failed to parse LLM reasoning response: {e}")
            # Return data-driven fallback based on actual trend
            metric_type = trend.get("metric_type", "")
            technique = trend.get("technique", "unknown")
            first_mean = trend.get("first_mean", 0)
            second_mean = trend.get("second_mean", 0)
            change_percent = trend.get("change_percent", 0)
            direction = trend.get("direction", "stable")
            
            # Generate evidence based on actual data
            if "valgus" in metric_type.lower():
                evidence = f"Knee valgus angle measurements show {direction} from {first_mean:.1f}° to {second_mean:.1f}° ({change_percent:+.1f}%). This pattern may indicate changes in landing mechanics or lower limb control during {technique}."
            elif "risk" in metric_type.lower():
                evidence = f"ACL risk score measurements show {direction} from {first_mean:.2f} to {second_mean:.2f} ({change_percent:+.1f}%). This pattern may indicate changes in movement quality during {technique}."
            else:
                evidence = f"{metric_type} measurements show {direction} from {first_mean:.2f} to {second_mean:.2f} ({change_percent:+.1f}%). This pattern may indicate changes in movement mechanics during {technique}."
            
            # Generate coaching options based on metric type
            coaching_options = []
            if "valgus" in metric_type.lower():
                coaching_options.append(f"Consider reviewing knee alignment during {technique} landings")
                coaching_options.append("Consider cueing knee tracking over toes")
            elif "risk" in metric_type.lower():
                coaching_options.append(f"Consider reviewing landing mechanics for {technique}")
                coaching_options.append("Consider monitoring movement quality")
            else:
                coaching_options.append(f"Consider reviewing {technique} execution")
            
            coaching_options.append("If pain/symptoms are present, consult your AT/PT.")
            
            return {
                "evidence_reasoning": evidence,
                "coaching_options": coaching_options
            }
    
    def _call_llm_for_reasoning(self, prompt: str) -> str:
        """
        Call LLM (Gemini or OpenAI) for reasoning.
        
        Args:
            prompt: Analysis prompt
            
        Returns:
            LLM response text
        """
        # Try Gemini first
        if GEMINI_AVAILABLE:
            try:
                api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel('gemini-pro')
                    # Add system instruction to prevent hallucinations
                    system_instruction = "Analyze ONLY the actual data provided. Do NOT use generic examples or repeat template text. Base all reasoning on the specific measurements and values provided."
                    response = model.generate_content(
                        f"{system_instruction}\n\n{prompt}",
                        generation_config=genai.types.GenerationConfig(
                            temperature=0.2,  # Lower temperature for more focused, data-driven responses
                            top_p=0.9,
                        )
                    )
                    return response.text
            except Exception as e:
                logger.warning(f"⚠️  Gemini API call failed: {e}, trying OpenAI...")
        
        # Fallback to OpenAI
        if OPENAI_AVAILABLE:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    client = OpenAI(api_key=api_key)
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert sports biomechanist. Analyze ONLY the actual data provided in the user's message. Do NOT use generic examples or repeat template text. Base all reasoning on the specific measurements and values provided. Always respond with valid JSON only, no additional text."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            response_format={"type": "json_object"},
                            temperature=0.2  # Lower temperature to reduce hallucinations
                        )
                        return response.choices[0].message.content
                    except Exception as e:
                        logger.debug(f"JSON format not supported, trying without: {e}")
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert sports biomechanist. Analyze ONLY the actual data provided in the user's message. Do NOT use generic examples or repeat template text. Base all reasoning on the specific measurements and values provided. Always respond with valid JSON only, no additional text or markdown."
                                },
                                {"role": "user", "content": prompt}
                            ],
                            temperature=0.2  # Lower temperature to reduce hallucinations
                        )
                        content = response.choices[0].message.content
                        if content.strip().startswith("```"):
                            import re
                            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', content, re.DOTALL)
                            if json_match:
                                return json_match.group(1)
                        return content
            except Exception as e:
                logger.warning(f"⚠️  OpenAI API call failed: {e}")
        
        logger.error("❌ No LLM available for reasoning")
        return "{}"
    
    def _determine_trend_status(
        self,
        trend: Dict[str, Any],
        data_points: List[Dict[str, Any]],
        min_sessions: int = 3
    ) -> str:
        """
        Determine trend status: Improving / Unchanged / Worsening / Insufficient data
        
        Args:
            trend: Trend statistics dictionary
            data_points: List of data points
            min_sessions: Minimum sessions required
            
        Returns:
            Status string: "improving", "unchanged", "worsening", or "insufficient_data"
        """
        if len(data_points) < min_sessions:
            return "insufficient_data"
        
        metric_type = trend["metric_type"]
        direction = trend["direction"]
        change_percent = abs(trend["change_percent"])
        
        # For risk-related metrics, decreasing is improving
        # For performance metrics, increasing might be improving (context-dependent)
        is_risk_metric = "risk" in metric_type.lower() or "valgus" in metric_type.lower()
        
        if change_percent < 5:  # Less than 5% change
            return "unchanged"
        elif is_risk_metric:
            # For risk metrics: decreasing = improving, increasing = worsening
            if direction == "decreasing":
                return "improving"
            elif direction == "increasing":
                return "worsening"
            else:
                return "unchanged"
        else:
            # For other metrics, context-dependent (default: increasing = improving)
            if direction == "increasing":
                return "improving"
            elif direction == "decreasing":
                return "worsening"
            else:
                return "unchanged"
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """
        Parse timestamp from various formats.
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            datetime object
        """
        if isinstance(timestamp, datetime):
            return timestamp
        
        if isinstance(timestamp, str):
            try:
                # Try ISO format
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                try:
                    # Try common formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S', '%Y-%m-%d']:
                        try:
                            return datetime.strptime(timestamp, fmt)
                        except:
                            continue
                except:
                    pass
        
        # Default to now if parsing fails
        return datetime.utcnow()
    
    def upsert_trends(self, trends: List[Dict[str, Any]]) -> List[str]:
        """
        Upsert trends to MongoDB.
        
        Args:
            trends: List of trend documents
            
        Returns:
            List of trend IDs
        """
        trend_ids = []
        
        for trend in trends:
            # Prepare trend document
            trend_doc = {
                "athlete_name": trend.get("athlete_name"),
                "technique": trend.get("technique"),
                "metric_signature": trend.get("metric_signature"),
                "metric_type": trend.get("metric_type"),
                "observation": trend.get("observation"),
                "evidence_reasoning": trend.get("evidence_reasoning"),
                "coaching_options": trend.get("coaching_options", []),
                "status": trend.get("status", "insufficient_data"),
                "trend_statistics": {
                    "first_mean": trend.get("first_mean"),
                    "second_mean": trend.get("second_mean"),
                    "overall_mean": trend.get("overall_mean"),
                    "overall_min": trend.get("overall_min"),
                    "overall_max": trend.get("overall_max"),
                    "change": trend.get("change"),
                    "change_percent": trend.get("change_percent"),
                    "direction": trend.get("direction"),
                    "data_point_count": trend.get("data_point_count")
                },
                "data_points": trend.get("data_points", []),
                "session_count": trend.get("session_count", 0),
                "generated_at": trend.get("generated_at", datetime.utcnow().isoformat())
            }
            
            trend_id = self.mongodb.upsert_trend(trend_doc)
            if trend_id:
                trend_ids.append(trend_id)
                logger.info(f"✅ Upserted trend: {trend_id} ({trend.get('metric_signature', 'unknown')})")
        
        return trend_ids
    
    def get_trends(
        self,
        athlete_name: Optional[str] = None,
        technique: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        include_status_followup: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Get trends from MongoDB.
        
        Args:
            athlete_name: Filter by athlete name
            technique: Filter by technique
            status: Filter by status (improving/unchanged/worsening/insufficient_data)
            limit: Maximum number of results
            include_status_followup: Whether to include status follow-up language
            
        Returns:
            List of trend documents with status follow-up
        """
        trends = self.mongodb.find_trends(
            athlete_name=athlete_name,
            technique=technique,
            status=status,
            limit=limit
        )
        
        if include_status_followup:
            for trend in trends:
                trend_status = trend.get("status", "insufficient_data")
                status_followup = self._generate_status_followup(trend, trend_status)
                trend["status_followup"] = status_followup
        
        return trends
    
    def _generate_status_followup(
        self,
        trend: Dict[str, Any],
        status: str
    ) -> str:
        """
        Generate status follow-up language based on trend status.
        Uses language like "Pattern improved after coaching changes" - never "prevented injury"
        
        Args:
            trend: Trend document
            status: Trend status (improving/unchanged/worsening/insufficient_data)
            
        Returns:
            Status follow-up text
        """
        if status == "improving":
            return "Pattern improved after coaching changes"
        elif status == "worsening":
            return "Pattern requires continued monitoring and intervention"
        elif status == "unchanged":
            return "Pattern remains stable - continued monitoring recommended"
        elif status == "insufficient_data":
            return "Insufficient data to determine trend status - additional sessions needed"
        else:
            return "Trend status being monitored"
    
    def update_trend_status(
        self,
        trend_id: str,
        new_data_points: List[Dict[str, Any]],
        sessions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """
        Update trend status with new data points and regenerate three-layer output.
        
        Args:
            trend_id: Trend ID
            new_data_points: New data points to add
            sessions: List of session documents
            
        Returns:
            Updated trend document or None
        """
        trend = self.mongodb.get_trend(trend_id)
        if not trend:
            logger.warning(f"⚠️  Trend {trend_id} not found")
            return None
        
        # Merge new data points
        existing_points = trend.get("data_points", [])
        all_points = existing_points + new_data_points
        
        # Remove duplicates by session_id
        seen = set()
        unique_points = []
        for point in all_points:
            session_id = point.get("session_id")
            if session_id and session_id not in seen:
                seen.add(session_id)
                unique_points.append(point)
        
        # Sort by timestamp
        unique_points.sort(key=lambda x: self._parse_timestamp(x.get("timestamp", "")))
        
        if len(unique_points) < 3:
            # Update status to insufficient_data
            from bson import ObjectId
            self.mongodb.get_trends_collection().update_one(
                {"_id": ObjectId(trend_id)},
                {"$set": {
                    "status": "insufficient_data",
                    "data_points": unique_points,
                    "session_count": len(unique_points),
                    "updated_at": datetime.utcnow()
                }}
            )
            return self.mongodb.get_trend(trend_id)
        
        # Recalculate trend statistics
        metric_type = trend.get("metric_type", "")
        trend_stats = self._calculate_trend_statistics(unique_points, metric_type)
        
        if not trend_stats:
            return None
        
        # Update trend with new statistics
        trend.update(trend_stats)
        trend["data_points"] = unique_points
        trend["session_count"] = len(unique_points)
        
        # Regenerate three-layer output
        three_layer = self._generate_three_layer_output(trend, unique_points, sessions)
        trend.update(three_layer)
        
        # Determine new status
        trend["status"] = self._determine_trend_status(trend, unique_points)
        
        # Upsert updated trend
        trend_doc = {
            "athlete_name": trend.get("athlete_name"),
            "technique": trend.get("technique"),
            "metric_signature": trend.get("metric_signature"),
            "metric_type": trend.get("metric_type"),
            "observation": trend.get("observation"),
            "evidence_reasoning": trend.get("evidence_reasoning"),
            "coaching_options": trend.get("coaching_options", []),
            "status": trend.get("status"),
            "trend_statistics": {
                "first_mean": trend.get("first_mean"),
                "second_mean": trend.get("second_mean"),
                "overall_mean": trend.get("overall_mean"),
                "overall_min": trend.get("overall_min"),
                "overall_max": trend.get("overall_max"),
                "change": trend.get("change"),
                "change_percent": trend.get("change_percent"),
                "direction": trend.get("direction"),
                "data_point_count": trend.get("data_point_count")
            },
            "data_points": unique_points,
            "session_count": len(unique_points),
            "generated_at": trend.get("generated_at", datetime.utcnow().isoformat())
        }
        
        updated_id = self.mongodb.upsert_trend(trend_doc)
        if updated_id:
            logger.info(f"✅ Updated trend status: {updated_id} -> {trend.get('status')}")
            return self.mongodb.get_trend(updated_id)
        
        return None
    
    def close(self):
        """Close MongoDB connection."""
        if self.mongodb:
            self.mongodb.close()

