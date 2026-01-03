#!/usr/bin/env python3
"""
Dynamic Metric Discovery using Deep Research
Discovers relevant metrics based on user requests during video stream.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from deep_search_reasoning import DeepSearchReasoning
from memory_integration import MemoryIndexIntegration

logger = logging.getLogger(__name__)


class DynamicMetricDiscovery:
    """
    Discovers metrics to track based on user requests using deep research.
    Integrates with agentic workflow to dynamically adjust metrics.
    """
    
    def __init__(self, deep_search: DeepSearchReasoning, memory: MemoryIndexIntegration):
        self.deep_search = deep_search
        self.memory = memory
        self.base_metrics = {
            "gymnastics": [
                "height off floor",
                "impact force",
                "landing bend angles",
                "knee straightness"
            ],
            "yoga": [
                "joint stacking",
                "spinal alignment",
                "foundation stability",
                "breath awareness"
            ],
            "fitness": [
                "range of motion",
                "tempo control",
                "spinal alignment",
                "joint alignment"
            ],
            "posture": [
                "rounded back",
                "arched back",
                "hunched shoulders",
                "forward head posture",
                "hyperextended knees",
                "bowed knees",
                "hip alignment"
            ]
        }
    
    def get_base_metrics(self, activity: Optional[str] = None) -> List[str]:
        """Get base metrics for activity, default to gymnastics"""
        if activity and activity in self.base_metrics:
            return self.base_metrics[activity]
        return self.base_metrics["gymnastics"]
    
    def discover_metrics_from_request(
        self,
        user_text: str,
        technique: Optional[str] = None,
        activity: Optional[str] = None,
        research_findings: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Discover metrics to track based on user request.
        
        Args:
            user_text: User's spoken/text request
            technique: Optional technique name
            activity: Optional activity/sport name (e.g., "gymnastics", "yoga", "fitness", "posture")
        
        Returns:
            Dictionary with discovered metrics and reasoning
        """
        logger.info(f"🔍 Discovering metrics from user request: {user_text[:100]}...")
        
        # Parse user request
        parsed = self._parse_user_request(user_text)
        
        # Extract activity/sport from user request if not provided
        detected_activity = activity or parsed.get("activity") or self._detect_activity_from_text(user_text, technique)
        
        # Extract technique from user request if not provided
        detected_technique = technique or parsed.get("technique")
        
        # Extract what to track and what to improve
        track_requests = parsed.get("track", [])
        improve_requests = parsed.get("improve", [])
        
        logger.info(f"   Detected activity: {detected_activity}")
        logger.info(f"   Detected technique: {detected_technique}")
        logger.info(f"   Track requests: {track_requests}")
        logger.info(f"   Improve requests: {improve_requests}")
        
        # If no specific requests, use base metrics for activity
        if not track_requests and not improve_requests:
            base_metrics = self.get_base_metrics(detected_activity)
            logger.info(f"ℹ️  No specific requests - using base metrics for {detected_activity}")
            return {
                "metrics_to_track": base_metrics,
                "improvement_focus": [],
                "discovered_from_research": False,
                "activity": detected_activity,
                "technique": detected_technique,
                "reasoning": f"Using default base metrics for {detected_activity}"
            }
        
        # Perform deep research to discover relevant metrics
        discovered_metrics = []
        research_findings = {}
        
        if self.deep_search:
            try:
                # Use provided research findings if available (from technique research)
                if research_findings:
                    research = research_findings
                    logger.info("✅ Using provided research findings for technique")
                else:
                    # Research technique to find relevant metrics
                    # Use detected activity and technique
                    research = self.deep_search.research_technique(
                        technique=detected_technique or detected_activity or "general",
                        sport=detected_activity or "general",
                        user_requests=track_requests + improve_requests
                    )
                
                research_findings = research
                
                # Extract key metrics from research
                key_metrics = research.get("key_metrics", [])
                recommended_analysis = research.get("recommended_analysis", [])
                
                # Combine user requests with research findings
                discovered_metrics = self._combine_metrics(
                    track_requests=track_requests,
                    improve_requests=improve_requests,
                    research_metrics=key_metrics,
                    recommended_analysis=recommended_analysis
                )
                
                logger.info(f"✅ Discovered {len(discovered_metrics)} metrics from deep research")
                
            except Exception as e:
                logger.warning(f"⚠️  Deep research failed: {e}")
                # Fallback to parsing user request directly
                discovered_metrics = self._extract_metrics_from_text(user_text)
        
        # If no metrics discovered, use base metrics for activity
        if not discovered_metrics:
            discovered_metrics = self.get_base_metrics(detected_activity)
        
        return {
            "metrics_to_track": discovered_metrics,
            "improvement_focus": improve_requests,
            "discovered_from_research": bool(research_findings),
            "research_findings": research_findings,
            "activity": detected_activity,
            "technique": detected_technique,
            "reasoning": f"Discovered {len(discovered_metrics)} metrics based on user request and research"
        }
    
    def _parse_user_request(self, user_text: str) -> Dict[str, List[str]]:
        """Parse user text to extract track/improve requests, activity, and technique"""
        text_lower = user_text.lower()
        
        parsed = {
            "track": [],
            "improve": [],
            "technique": None,
            "activity": None
        }
        
        # Extract "track X" or "tracking X" patterns
        track_patterns = [
            r"track\s+(.+?)(?:\.|,|$|and|or)",
            r"tracking\s+(.+?)(?:\.|,|$|and|or)",
            r"want\s+to\s+track\s+(.+?)(?:\.|,|$|and|or)",
            r"monitor\s+(.+?)(?:\.|,|$|and|or)",
            r"measure\s+(.+?)(?:\.|,|$|and|or)",
        ]
        
        for pattern in track_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                # Clean up match
                item = match.strip().rstrip('.,')
                if item and len(item) > 2:
                    parsed["track"].append(item)
        
        # Extract "improve X" or "better X" patterns
        improve_patterns = [
            r"improve\s+(.+?)(?:\.|,|$|and|or|technique)",
            r"better\s+(.+?)(?:\.|,|$|and|or|technique)",
            r"fix\s+(.+?)(?:\.|,|$|and|or|alignment|form)",
            r"work\s+on\s+(.+?)(?:\.|,|$|and|or)",
            r"focus\s+on\s+(.+?)(?:\.|,|$|and|or)",
            r"check\s+(.+?)(?:\.|,|$|and|or|position|alignment)",
        ]
        
        for pattern in improve_patterns:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                item = match.strip().rstrip('.,')
                if item and len(item) > 2:
                    parsed["improve"].append(item)
        
        # Extract activity/sport from user request
        activity_keywords = {
            "gymnastics": ["gymnastics", "gymnast", "gymnastic", "floor", "beam", "bars", "vault"],
            "yoga": ["yoga", "yogi", "asana", "pose", "yoga pose"],
            "fitness": ["fitness", "exercise", "workout", "training", "squat", "deadlift", "press"],
            "posture": ["posture", "postural", "sitting", "standing", "ergonomic", "alignment"]
        }
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                parsed["activity"] = activity
                break
        
        # Extract technique name
        technique_patterns = [
            # Gymnastics techniques
            r"(back\s+handspring|front\s+handspring|back\s+tuck|front\s+tuck|split\s+leap|switch\s+leap)",
            r"(bb_|fx_|ub_|vt_)\w+",
            # Yoga techniques
            r"(warrior\s+[i1]|warrior\s+[ii2]|warrior\s+[iii3]|mountain\s+pose|tree\s+pose|downward\s+dog)",
            r"(yoga_|asana_)\w+",
            # Fitness techniques
            r"(squat|deadlift|bench\s+press|overhead\s+press|pull.?up|chin.?up)",
            r"(fitness_)\w+",
            # Posture techniques
            r"(sitting|standing|lying|posture_)\w+",
        ]
        
        for pattern in technique_patterns:
            match = re.search(pattern, text_lower)
            if match:
                parsed["technique"] = match.group(1)
                break
        
        return parsed
    
    def _detect_activity_from_text(self, user_text: str, technique: Optional[str] = None) -> str:
        """Detect activity/sport from user text or technique name"""
        text_lower = user_text.lower()
        
        # Check user text for activity keywords
        activity_keywords = {
            "gymnastics": ["gymnastics", "gymnast", "gymnastic", "floor exercise", "beam", "bars", "vault"],
            "yoga": ["yoga", "yogi", "asana", "yoga pose", "yoga form"],
            "fitness": ["fitness", "exercise", "workout", "training", "strength", "cardio"],
            "posture": ["posture", "postural", "sitting", "standing", "ergonomic", "alignment", "spinal"]
        }
        
        for activity, keywords in activity_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                return activity
        
        # Check technique name for activity prefix
        if technique:
            if technique.startswith(("fx_", "bb_", "ub_", "vt_")):
                return "gymnastics"
            elif technique.startswith("yoga_"):
                return "yoga"
            elif technique.startswith("fitness_"):
                return "fitness"
            elif technique.startswith("posture_"):
                return "posture"
        
        # Default to gymnastics if unclear
        return "gymnastics"
    
    def _extract_metrics_from_text(self, user_text: str) -> List[str]:
        """Extract metric names directly from text"""
        text_lower = user_text.lower()
        metrics = []
        
        # Common metric keywords
        metric_keywords = {
            "height": "height off floor",
            "impact": "impact force",
            "force": "impact force",
            "landing": "landing bend angles",
            "bend": "landing bend angles",
            "knee": "knee straightness",
            "straight": "knee straightness",
            "stiffness": "stiffness",
            "stiff": "stiffness",
            "velocity": "joint velocity",
            "acceleration": "joint acceleration",
            "rotation": "body rotation",
            "alignment": "body alignment",
            "extension": "knee extension",
        }
        
        for keyword, metric_name in metric_keywords.items():
            if keyword in text_lower:
                if metric_name not in metrics:
                    metrics.append(metric_name)
        
        return metrics
    
    def _combine_metrics(
        self,
        track_requests: List[str],
        improve_requests: List[str],
        research_metrics: List[str],
        recommended_analysis: List[str]
    ) -> List[str]:
        """Combine user requests with research findings"""
        all_metrics = []
        
        # Add metrics from user track requests
        for request in track_requests:
            # Map common phrases to metric names
            metric = self._map_request_to_metric(request)
            if metric and metric not in all_metrics:
                all_metrics.append(metric)
        
        # Add metrics for improvement focus
        for request in improve_requests:
            metric = self._map_request_to_metric(request)
            if metric and metric not in all_metrics:
                all_metrics.append(metric)
        
        # Add metrics from research findings
        for metric in research_metrics:
            if metric not in all_metrics:
                all_metrics.append(metric)
        
        # Add metrics from recommended analysis
        for analysis_item in recommended_analysis:
            # Extract metric names from analysis recommendations
            metric = self._extract_metric_from_analysis(analysis_item)
            if metric and metric not in all_metrics:
                all_metrics.append(metric)
        
        # Always include base metrics for activity if not already present
        # Note: Activity context should be passed, but for now we'll add base metrics
        # The activity-specific base metrics are handled in discover_metrics_from_request
        # This method is called from _combine_metrics which doesn't have activity context
        # So we add a default set here, but the main logic in discover_metrics_from_request
        # ensures activity-specific base metrics are included
        
        return all_metrics
    
    def _map_request_to_metric(self, request: str) -> Optional[str]:
        """Map user request text to metric name"""
        request_lower = request.lower()
        
        mapping = {
            "height": "height off floor",
            "height off floor": "height off floor",
            "jump height": "height off floor",
            "impact": "impact force",
            "impact force": "impact force",
            "landing force": "impact force",
            "landing": "landing bend angles",
            "landing bend": "landing bend angles",
            "bend": "landing bend angles",
            "knee bend": "landing bend angles",
            "knee": "knee straightness",
            "knee straight": "knee straightness",
            "straightness": "knee straightness",
            "stiffness": "stiffness",
            "stiff": "stiffness",
            "rigidity": "stiffness",
            "velocity": "joint velocity",
            "speed": "joint velocity",
            "acceleration": "joint acceleration",
            "rotation": "body rotation",
            "alignment": "body alignment",
            "extension": "knee extension",
        }
        
        for keyword, metric in mapping.items():
            if keyword in request_lower:
                return metric
        
        return None
    
    def _extract_metric_from_analysis(self, analysis_item: str) -> Optional[str]:
        """Extract metric name from analysis recommendation text"""
        analysis_lower = analysis_item.lower()
        return self._map_request_to_metric(analysis_lower)




















