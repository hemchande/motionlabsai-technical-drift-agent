#!/usr/bin/env python3
"""
User Request Handler for Video Stream
Parses user requests from LLM responses and triggers metric discovery.
"""

import logging
import re
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)


class UserRequestHandler:
    """
    Handles user requests during video stream.
    Parses LLM responses to extract tracking/improvement requests.
    """
    
    def __init__(self, metric_discovery, workflow, memory_integration):
        self.metric_discovery = metric_discovery
        self.workflow = workflow
        self.memory_integration = memory_integration
        self.current_technique: Optional[str] = None
        self.current_metrics: List[str] = []
        self.discovery_results: Optional[Dict[str, Any]] = None
    
    async def handle_user_message(
        self,
        agent,
        user_text: str
    ) -> bool:
        """
        Handle user message and discover metrics if needed.
        Handles follow-up queries, revisions, and unknown techniques.
        
        Args:
            agent: Agent instance for LLM communication
            user_text: User's spoken/text input
        
        Returns:
            True if request was handled, False otherwise
        """
        if not user_text:
            return False
        
        logger.info(f"📨 User message: {user_text[:100]}...")
        
        # ALWAYS add user message to transcript (capture ALL messages, not just metric requests)
        if self.workflow and hasattr(self.workflow, 'add_transcript_entry'):
            self.workflow.add_transcript_entry("user", user_text)
            logger.debug(f"📝 Added user message to transcript: {user_text[:50]}...")
        
        # Try to detect and update activity/technique from user message (even if not in request handler flow)
        if self.workflow:
            # Detect activity (update if not set, or if user explicitly mentions a different activity)
            detected_activity = self.metric_discovery._detect_activity_from_text(user_text, None) if self.metric_discovery else None
            if detected_activity:
                # Update if not set, or if user explicitly mentions activity
                if not self.workflow.current_activity or any(keyword in user_text.lower() for keyword in [detected_activity, f"for {detected_activity}", f"doing {detected_activity}"]):
                    self.workflow.current_activity = detected_activity
                    logger.info(f"✅ Detected/updated activity from user message: {detected_activity}")
            
            # Detect technique (update if not set, or if user explicitly mentions a different technique)
            technique = self._extract_technique(user_text)
            if technique:
                # Update if not set, or if user explicitly mentions technique
                if not self.workflow.current_technique or technique.lower() in user_text.lower():
                    self.workflow.current_technique = technique
                    logger.info(f"✅ Detected/updated technique from user message: {technique}")
        
        text_lower = user_text.lower()
        
        # Check for revision keywords (follow-up queries)
        revision_keywords = [
            "actually", "instead", "change", "update", "revise", "modify",
            "also", "add", "remove", "forget", "focus on", "switch to",
            "different", "new", "different technique", "change technique"
        ]
        
        # Check for metric requests
        request_keywords = [
            "track", "tracking", "monitor", "measure",
            "improve", "better", "fix", "work on", "focus on",
            "want to", "need to", "should", "can you", "please"
        ]
        
        # Check for technique mentions (including unknown techniques)
        technique_keywords = [
            "technique", "exercise", "move", "skill", "pose", "asana",
            "back handspring", "squat", "deadlift", "warrior", "downward dog"
        ]
        
        is_revision = any(keyword in text_lower for keyword in revision_keywords)
        is_metric_request = any(keyword in text_lower for keyword in request_keywords)
        has_technique_mention = any(keyword in text_lower for keyword in technique_keywords)
        
        # Handle if it's a revision, metric request, or technique mention
        if not (is_revision or is_metric_request or has_technique_mention):
            return False
        
        # Extract technique if mentioned (including unknown techniques)
        technique = self._extract_technique(user_text)
        detected_activity = self.metric_discovery._detect_activity_from_text(user_text, technique)
        
        # Check if technique is unknown or needs research
        technique_needs_research = False
        if technique:
            # Check if technique is in memory mapping
            if self.memory_integration:
                technique_data = self.memory_integration.get_technique_mapping(technique)
                if not technique_data or not technique_data.get("found", False):
                    technique_needs_research = True
                    logger.info(f"🔍 Technique '{technique}' not in mapping - will research")
            else:
                # If no memory integration, assume technique needs research if it's not a common one
                common_techniques = ["bb_back_handspring", "bb_front_handspring", "squat", "deadlift"]
                if technique not in common_techniques:
                    technique_needs_research = True
        
        # If technique needs research, trigger deep research
        if technique_needs_research and technique:
            logger.info(f"🔬 Triggering deep research for unknown technique: {technique}")
            try:
                # Research the technique using deep search
                research_result = await self._research_unknown_technique(
                    technique=technique,
                    activity=detected_activity,
                    user_text=user_text
                )
                
                # Update current technique
                self.current_technique = technique
                
                # Discover metrics based on research
                discovery = self.metric_discovery.discover_metrics_from_request(
                    user_text=user_text,
                    technique=technique,
                    activity=detected_activity,
                    research_findings=research_result
                )
                
                self.discovery_results = discovery
                self.current_metrics = discovery.get("metrics_to_track", [])
                
                # Update workflow with discovered technique and metrics
                if self.workflow:
                    await self.workflow.initialize_workflow(
                        technique=technique,
                        user_requests=self.current_metrics,
                        research_findings=research_result,
                        activity=detected_activity
                    )
                
                # Respond to user
                response = self._format_technique_research_response(technique, discovery, research_result)
                
                # Add agent response to transcript
                if self.workflow and hasattr(self.workflow, 'add_transcript_entry'):
                    self.workflow.add_transcript_entry("assistant", response)
                
                await agent.llm.simple_response(text=response)
                
                logger.info(f"✅ Researched technique '{technique}' and discovered {len(self.current_metrics)} metrics")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error researching technique: {e}", exc_info=True)
                # Fall through to regular discovery
        
        # Regular metric discovery (known technique or no technique change)
        if technique:
            self.current_technique = technique
            logger.info(f"✅ Detected technique: {technique}")
        
        if detected_activity:
            logger.info(f"✅ Detected activity: {detected_activity}")
        
        # Discover metrics from user request
        try:
            discovery = self.metric_discovery.discover_metrics_from_request(
                user_text=user_text,
                technique=self.current_technique,
                activity=detected_activity
            )
            
            self.discovery_results = discovery
            self.current_metrics = discovery.get("metrics_to_track", [])
            
            # Update workflow with discovered metrics (or revise if follow-up)
            if self.workflow:
                await self.workflow.initialize_workflow(
                    technique=self.current_technique or self.workflow.current_technique,
                    user_requests=self.current_metrics,
                    activity=detected_activity or self.workflow.current_activity
                )
            
            # Respond to user with discovered metrics
            if is_revision:
                response = self._format_revision_response(discovery)
            else:
                response = self._format_discovery_response(discovery)
            
            # Add agent response to transcript
            if self.workflow and hasattr(self.workflow, 'add_transcript_entry'):
                self.workflow.add_transcript_entry("assistant", response)
            
            await agent.llm.simple_response(text=response)
            
            logger.info(f"✅ Discovered {len(self.current_metrics)} metrics to track")
            logger.info(f"   Metrics: {', '.join(self.current_metrics[:5])}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error handling user request: {e}", exc_info=True)
            return False
    
    def _extract_technique(self, user_text: str) -> Optional[str]:
        """
        Extract technique name from user text.
        Handles both known and unknown techniques.
        """
        text_lower = user_text.lower()
        
        # Common technique patterns (known techniques)
        technique_patterns = [
            (r"back\s+handspring", "bb_back_handspring"),
            (r"front\s+handspring", "bb_front_handspring"),
            (r"back\s+tuck", "bb_back_tuck"),
            (r"front\s+tuck", "bb_front_tuck"),
            (r"split\s+leap", "fx_split_leap"),
            (r"switch\s+leap", "fx_switch_leap"),
            (r"squat", "squat"),
            (r"deadlift", "deadlift"),
            (r"push.?up", "push_up"),
            (r"pull.?up", "pull_up"),
            (r"warrior\s+i", "warrior_i"),
            (r"warrior\s+ii", "warrior_ii"),
            (r"downward\s+dog", "downward_dog"),
            (r"tree\s+pose", "tree_pose"),
        ]
        
        for pattern, technique in technique_patterns:
            if re.search(pattern, text_lower):
                return technique
        
        # Check for technique code format
        code_match = re.search(r"(bb_|fx_|ub_|vt_|fitness_|yoga_|posture_)\w+", text_lower)
        if code_match:
            return code_match.group(0)
        
        # Try to extract technique name from common phrases
        # "analyze my [technique]", "for [technique]", "doing [technique]"
        technique_phrases = [
            r"analyze\s+(?:my\s+)?([a-z\s]+?)(?:\s+technique|\s+form|\s+pose|$)",
            r"for\s+(?:my\s+)?([a-z\s]+?)(?:\s+technique|\s+form|\s+pose|$)",
            r"doing\s+([a-z\s]+?)(?:\s+technique|\s+form|\s+pose|$)",
            r"performing\s+([a-z\s]+?)(?:\s+technique|\s+form|\s+pose|$)",
            r"technique\s+(?:is\s+)?([a-z\s]+?)(?:$|\.|,|\s)",
        ]
        
        for pattern in technique_phrases:
            match = re.search(pattern, text_lower)
            if match:
                potential_technique = match.group(1).strip()
                # Filter out common words
                if potential_technique and len(potential_technique) > 2:
                    # Remove common stop words
                    stop_words = ["the", "my", "a", "an", "this", "that", "for", "with"]
                    words = [w for w in potential_technique.split() if w not in stop_words]
                    if words:
                        # Return as snake_case
                        technique_name = "_".join(words)
                        # Limit length to avoid capturing too much
                        if len(technique_name) < 50:
                            return technique_name
        
        return None
    
    def _format_discovery_response(self, discovery: Dict[str, Any]) -> str:
        """Format discovery results for user response"""
        metrics = discovery.get("metrics_to_track", [])
        improvement_focus = discovery.get("improvement_focus", [])
        from_research = discovery.get("discovered_from_research", False)
        
        response = "✅ I've discovered the following metrics to track:\n\n"
        
        if metrics:
            response += "**Metrics to Track:**\n"
            for i, metric in enumerate(metrics, 1):
                response += f"{i}. {metric}\n"
        
        if improvement_focus:
            response += "\n**Improvement Focus:**\n"
            for item in improvement_focus:
                response += f"- {item}\n"
        
        if from_research:
            response += "\n🔬 These metrics were discovered using deep research on technique standards and biomechanics."
        else:
            response += "\nℹ️  Using base metrics (deep research not available)."
        
        response += "\n\nI'll now track these metrics throughout your performance!"
        
        return response
    
    def get_current_metrics(self) -> List[str]:
        """Get currently tracked metrics"""
        return self.current_metrics.copy() if self.current_metrics else []
    
    def get_discovery_results(self) -> Optional[Dict[str, Any]]:
        """Get latest discovery results"""
        return self.discovery_results
    
    async def _research_unknown_technique(
        self,
        technique: str,
        activity: Optional[str],
        user_text: str
    ) -> Dict[str, Any]:
        """
        Research an unknown technique using deep search.
        
        Args:
            technique: Technique name to research
            activity: Activity/sport context
            user_text: Original user request
        
        Returns:
            Research findings with standards, metrics, biomechanics
        """
        logger.info(f"🔬 Researching unknown technique: {technique}")
        
        research_result = {
            "technique": technique,
            "activity": activity,
            "found": False,
            "standards": {},
            "key_metrics": [],
            "biomechanics": {},
            "common_errors": []
        }
        
        # Use deep search if available
        if self.metric_discovery and hasattr(self.metric_discovery, 'deep_search'):
            deep_search = self.metric_discovery.deep_search
            if deep_search:
                try:
                    # Research the technique
                    research = deep_search.research_technique(
                        technique=technique,
                        sport=activity or "general",
                        user_requests=[user_text] if user_text else None
                    )
                    
                    research_result.update(research)
                    research_result["found"] = True
                    logger.info(f"✅ Research complete: {len(research.get('key_metrics', []))} metrics found")
                    
                except Exception as e:
                    logger.warning(f"⚠️  Deep research failed: {e}")
                    research_result["error"] = str(e)
        
        return research_result
    
    def _format_technique_research_response(
        self,
        technique: str,
        discovery: Dict[str, Any],
        research_result: Dict[str, Any]
    ) -> str:
        """Format response for technique research"""
        metrics = discovery.get("metrics_to_track", [])
        from_research = discovery.get("discovered_from_research", False)
        
        response = f"🔬 I've researched the technique '{technique}' and discovered the following:\n\n"
        
        if research_result.get("found"):
            response += "**Research Findings:**\n"
            if research_result.get("standards"):
                response += f"- Technique standards identified\n"
            if research_result.get("biomechanics"):
                response += f"- Biomechanical analysis available\n"
            if research_result.get("common_errors"):
                response += f"- Common errors documented\n"
            response += "\n"
        
        if metrics:
            response += "**Metrics to Track:**\n"
            for i, metric in enumerate(metrics, 1):
                response += f"{i}. {metric}\n"
        
        if from_research:
            response += "\n🔬 These metrics were derived from deep research on technique standards and biomechanics."
        else:
            response += "\nℹ️  Using activity-specific base metrics."
        
        response += f"\n\nI'll now track these metrics for '{technique}' throughout your performance!"
        
        return response
    
    def _format_revision_response(self, discovery: Dict[str, Any]) -> str:
        """Format response for revision/follow-up queries"""
        metrics = discovery.get("metrics_to_track", [])
        improvement_focus = discovery.get("improvement_focus", [])
        
        response = "✅ Got it! I've updated the metrics based on your revision:\n\n"
        
        if metrics:
            response += "**Updated Metrics to Track:**\n"
            for i, metric in enumerate(metrics, 1):
                response += f"{i}. {metric}\n"
        
        if improvement_focus:
            response += "\n**Updated Improvement Focus:**\n"
            for item in improvement_focus:
                response += f"- {item}\n"
        
        response += "\n✅ Workflow updated! I'll now track these revised metrics."
        
        return response




















