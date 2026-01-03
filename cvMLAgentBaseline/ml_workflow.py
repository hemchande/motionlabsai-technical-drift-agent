#!/usr/bin/env python3
"""
Agentic ML Workflow
Orchestrates the ML workflow with reasoning on sequence of steps.
"""

import logging
import asyncio
import json
import cv2
import numpy as np
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class AgenticMLWorkflow:
    """
    Agentic ML workflow that:
    1. Reasons about sequence of steps
    2. Selects appropriate models
    3. Executes workflow steps
    4. Chains model outputs
    5. Provides comprehensive analysis
    """
    
    def __init__(
        self,
        model_selector,
        cv_tools: Dict[str, Any],
        technique_metrics,
        memory_integration
    ):
        self.model_selector = model_selector
        self.cv_tools = cv_tools
        self.technique_metrics = technique_metrics
        self.memory_integration = memory_integration
        self.session_history = None  # Will be set by main.py
        
        self.current_technique: Optional[str] = None
        self.current_activity: Optional[str] = None
        self.user_requests: List[str] = []
        self.selected_models: Dict = {}
        self.workflow_sequence: List[Dict[str, Any]] = []
        self.workflow_state: Dict[str, Any] = {
            "frame_capture_active": True  # Frame capture always active for ACL analysis
        }
        self.pending_tasks: List[Dict[str, Any]] = []
        self.completed_steps: List[Dict[str, Any]] = []
        self.call_id: Optional[str] = None
        self.transcript: List[Dict[str, Any]] = []  # List of {role, content, timestamp}
    
    async def initialize_workflow(
        self,
        technique: str,
        user_requests: List[str],
        research_findings: Optional[Dict[str, Any]] = None,
        activity: Optional[str] = None
    ):
        """
        Initialize workflow with technique and user requests.
        Can be called multiple times to update metrics dynamically.
        
        Args:
            technique: Technique name
            user_requests: List of requested metrics/analyses
            research_findings: Optional deep research findings to inform model selection
        """
        self.current_technique = technique
        self.current_activity = activity
        self.user_requests = user_requests
        
        # Reset rotation state for new sequence
        if self.technique_metrics and hasattr(self.technique_metrics, 'reset_rotation_state'):
            self.technique_metrics.reset_rotation_state()
            logger.debug("🔄 Reset rotation tracking state for new sequence")
        
        logger.info(f"🔧 Initializing workflow for technique: {technique}")
        logger.info(f"   User requests: {', '.join(user_requests)}")
        
        # Step 1: Select models with reasoning (can use research findings)
        selected_models, reasoning = self.model_selector.select_models(
            user_requests=user_requests,
            technique=technique,
            research_findings=research_findings
        )
        
        self.selected_models = selected_models
        self.workflow_sequence = reasoning.workflow_sequence
        
        logger.info(f"✅ Selected {len(selected_models)} models")
        logger.info(f"✅ Workflow sequence: {len(self.workflow_sequence)} steps")
        
        # Log reasoning
        reasoning_summary = self.model_selector.get_reasoning_summary()
        logger.info(f"\n{reasoning_summary}")
        
        # Initialize workflow state
        self.workflow_state = {
            "technique": technique,
            "user_requests": user_requests,
            "selected_models": {k.value: v.value for k, v in selected_models.items()},
            "workflow_sequence": self.workflow_sequence,
            "reasoning": reasoning.to_dict(),
            "start_time": datetime.utcnow().isoformat(),
            "current_step": 0,
            "frame_data": {},
            "frames": [],  # Store frames with overlays for video output
            "metrics": {},
            "analysis": {},
            "transcript": self.transcript,  # Store transcript in workflow state
            "frame_capture_active": True  # Frame capture always active for ACL analysis
        }
        
        # Create pending tasks from workflow sequence
        self.pending_tasks = [
            {
                "step": step["step"],
                "task": step,
                "status": "pending"
            }
            for step in self.workflow_sequence
        ]
    
    def has_pending_tasks(self) -> bool:
        """Check if there are pending tasks"""
        return len([t for t in self.pending_tasks if t["status"] == "pending"]) > 0
    
    async def execute_next_step(self, agent=None):
        """
        Execute the next step in the workflow.
        
        Args:
            agent: Optional agent instance for LLM communication
        """
        # Find next pending task
        next_task = None
        for task in self.pending_tasks:
            if task["status"] == "pending":
                # Check if dependencies are met
                step_info = task["task"]
                depends_on = step_info.get("depends_on", [])
                
                if not depends_on or all(
                    any(completed["step"] == dep for completed in self.completed_steps)
                    for dep in depends_on
                ):
                    next_task = task
                    break
        
        if not next_task:
            return
        
        step_info = next_task["task"]
        step_number = step_info["step"]
        model_type = step_info.get("model_type")
        
        logger.info(f"🔄 Executing step {step_number}: {step_info.get('purpose')}")
        
        try:
            # Execute based on step type
            if model_type == "person_detection":
                result = await self._execute_person_detection(step_info)
            elif model_type == "face_detection":
                result = await self._execute_face_detection(step_info)
            elif model_type == "pose_estimation":
                result = await self._execute_pose_estimation(step_info)
            elif model_type == "weight_estimation":
                result = await self._execute_weight_estimation(step_info)
            elif model_type == "metric_extraction":
                result = await self._execute_metric_extraction(step_info)
            elif model_type == "fig_standards_comparison":
                result = await self._execute_standards_comparison(step_info)
            else:
                result = {"status": "skipped", "reason": f"Unknown step type: {model_type}"}
            
            # Mark task as completed
            next_task["status"] = "completed"
            self.completed_steps.append({
                "step": step_number,
                "result": result,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Store result in workflow state
            self.workflow_state["frame_data"][f"step_{step_number}"] = result
            
            logger.info(f"✅ Step {step_number} completed")
            
            # If agent available, send update
            if agent and hasattr(agent, 'llm'):
                await self._send_step_update(agent, step_number, result)
                
        except Exception as e:
            logger.error(f"❌ Error executing step {step_number}: {e}", exc_info=True)
            next_task["status"] = "failed"
            next_task["error"] = str(e)
    
    async def _execute_person_detection(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute person detection step"""
        # This would process current frame
        # For now, return placeholder
        return {
            "status": "completed",
            "detections": [],
            "note": "Person detection would process current frame"
        }
    
    async def _execute_face_detection(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute face detection step"""
        face_tool = self.cv_tools.get("face_detection")
        if face_tool:
            # Would process current frame
            return {
                "status": "completed",
                "faces": [],
                "note": "Face detection would process current frame"
            }
        return {"status": "skipped", "reason": "Face detection tool not available"}
    
    async def _execute_pose_estimation(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pose estimation step using HRNet (preferred)"""
        pose_tool = self.cv_tools.get("pose_estimation")
        if pose_tool:
            # Use HRNet as primary method (more accurate for detailed analysis)
            model_variant = step_info.get("model_variant", "hrnet")
            
            # Try to get latest frame from workflow state if available
            latest_frame = self.workflow_state.get("latest_frame")
            if latest_frame is not None:
                try:
                    # Process frame with pose estimation
                    pose_result = pose_tool.estimate_pose(latest_frame, method="hrnet")
                    if "error" in pose_result:
                        # Fallback to YOLO
                        pose_result = pose_tool.estimate_pose(latest_frame, method="yolo")
                    
                    if "keypoints" in pose_result and pose_result["keypoints"]:
                        # Store keypoints for metric extraction
                        self.workflow_state["latest_pose_keypoints"] = pose_result["keypoints"]
                        return {
                            "status": "completed",
                            "keypoints": pose_result["keypoints"],
                            "model_variant": model_variant,
                            "method": "hrnet" if "error" not in pose_result else "yolo"
                        }
                except Exception as e:
                    logger.error(f"Error in pose estimation: {e}", exc_info=True)
            
            # If no frame available, return placeholder
            return {
                "status": "completed",
                "keypoints": {},
                "model_variant": model_variant,
                "method": "hrnet",
                "note": "Pose estimation ready (waiting for frame)"
            }
        return {"status": "skipped", "reason": "Pose estimation tool not available"}
    
    async def _execute_weight_estimation(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weight estimation step"""
        weight_tool = self.cv_tools.get("weight_detection")
        if weight_tool:
            # Would use pose keypoints from previous step
            return {
                "status": "completed",
                "weight_estimate": {},
                "note": "Weight estimation would use pose keypoints"
            }
        return {"status": "skipped", "reason": "Weight detection tool not available"}
    
    async def _execute_metric_extraction(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute metric extraction step"""
        # Get keypoints from pose estimation step or latest pose keypoints
        pose_result = None
        keypoints = None
        
        # First try to get from latest pose keypoints (from frame processing)
        if "latest_pose_keypoints" in self.workflow_state:
            keypoints = self.workflow_state["latest_pose_keypoints"]
            pose_result = {"keypoints": keypoints}
        else:
            # Fallback to completed steps
            for completed in self.completed_steps:
                if completed["step"] == 3:  # Pose estimation step
                    pose_result = completed.get("result", {})
                    break
        
        if pose_result and pose_result.get("keypoints"):
            keypoints = pose_result["keypoints"]
        elif keypoints is None:
            # Try to get from workflow state directly
            keypoints = self.workflow_state.get("latest_pose_keypoints")
        
        if keypoints:
            # Use discovered metrics if available, otherwise use user_requests
            metrics_to_calculate = self.user_requests
            if not metrics_to_calculate:
                # Default to base metrics based on activity
                if self.current_activity == "gymnastics":
                    metrics_to_calculate = [
                        "height off floor",
                        "impact force",
                        "landing bend angles",
                        "knee straightness",
                        "ACL tear risk"
                    ]
                elif self.current_activity == "posture":
                    metrics_to_calculate = [
                        "rounded back",
                        "arched back",
                        "hunched shoulders",
                        "forward head posture",
                        "hyperextended knees",
                        "bowed knees",
                        "hip alignment"
                    ]
                else:
                    metrics_to_calculate = [
                        "height off floor",
                        "impact force",
                        "landing bend angles",
                        "knee straightness"
                    ]
            
            # Get previous keypoints for velocity/acceleration calculations
            previous_keypoints = self.workflow_state.get("previous_pose_keypoints")
            frame_timestamp = self.workflow_state.get("latest_frame_timestamp")
            
            metrics = self.technique_metrics.calculate_metrics(
                keypoints=keypoints,
                technique=self.current_technique or "unknown",
                user_requests=metrics_to_calculate,
                frame_timestamp=frame_timestamp,
                previous_frame_keypoints=previous_keypoints
            )
            
            # Merge with existing metrics (accumulate across frames)
            existing_metrics = self.workflow_state.get("metrics", {})
            if existing_metrics:
                # For metrics that should be accumulated (like rotations), keep max/sum
                # For others, use latest value
                for key, value in metrics.items():
                    if "rotation" in key.lower() or "total" in key.lower():
                        # Keep maximum or sum for rotation metrics
                        if key in existing_metrics:
                            existing_metrics[key] = max(existing_metrics[key], value) if "peak" in key.lower() else value
                        else:
                            existing_metrics[key] = value
                    else:
                        # Use latest value
                        existing_metrics[key] = value
                self.workflow_state["metrics"] = existing_metrics
            else:
                self.workflow_state["metrics"] = metrics
            
            # Store previous keypoints for next calculation
            self.workflow_state["previous_pose_keypoints"] = keypoints.copy() if keypoints else None
            
            # Store metrics in session history for comparison
            if self.session_history and self.current_technique:
                self.session_history.add_metric_snapshot(
                    technique=self.current_technique,
                    activity=self.current_activity or "unknown",
                    metrics=self.workflow_state["metrics"]
                )
            
            return {
                "status": "completed",
                "metrics": self.workflow_state["metrics"],
                "metric_count": len(self.workflow_state["metrics"]),
                "metrics_tracked": metrics_to_calculate
            }
        else:
            return {
                "status": "skipped",
                "reason": "Pose keypoints not available"
            }
    
    async def _execute_standards_comparison(self, step_info: Dict[str, Any]) -> Dict[str, Any]:
        """Execute FIG standards comparison step"""
        # Get metrics from metric extraction step
        metrics = self.workflow_state.get("metrics", {})
        
        if metrics and self.current_technique:
            comparison = self.memory_integration.compare_to_standards(
                technique=self.current_technique,
                metrics=metrics
            )
            
            # Add historical comparison if session history available
            if self.session_history:
                historical_comparison = self.session_history.compare_metrics(
                    current_metrics=metrics,
                    technique=self.current_technique,
                    activity=self.current_activity
                )
                comparison["historical"] = historical_comparison
                
                # Generate feedback with history
                feedback_with_history = self.session_history.get_feedback_with_history(
                    current_metrics=metrics,
                    technique=self.current_technique,
                    activity=self.current_activity or "unknown"
                )
                comparison["feedback_with_history"] = feedback_with_history
            
            self.workflow_state["analysis"] = comparison
            
            return {
                "status": "completed",
                "comparison": comparison,
                "all_standards_met": comparison.get("all_met", False)
            }
        else:
            return {
                "status": "skipped",
                "reason": "Metrics not available for comparison"
            }
    
    async def _send_step_update(self, agent, step_number: int, result: Dict[str, Any]):
        """Send step update to agent LLM"""
        try:
            step_info = next(
                (s for s in self.workflow_sequence if s["step"] == step_number),
                {}
            )
            
            update_text = f"**Workflow Step {step_number} Complete:**\n"
            update_text += f"Purpose: {step_info.get('purpose', 'N/A')}\n"
            update_text += f"Status: {result.get('status', 'unknown')}\n"
            
            if result.get("metrics"):
                update_text += f"\n**Metrics Calculated:**\n"
                for key, value in list(result["metrics"].items())[:5]:
                    update_text += f"- {key}: {value:.2f}\n"
            
            if result.get("comparison"):
                comp = result["comparison"]
                update_text += f"\n**Standards Comparison:**\n"
                update_text += f"All standards met: {comp.get('all_met', False)}\n"
                if comp.get("gaps"):
                    update_text += f"Gaps found: {len(comp['gaps'])}\n"
                
                # Include historical feedback if available
                if comp.get("feedback_with_history"):
                    update_text += f"\n{comp['feedback_with_history']}\n"
            
            await agent.llm.simple_response(text=update_text)
        except Exception as e:
            logger.error(f"❌ Error sending step update: {e}")
    
    def add_transcript_entry(self, role: str, content: str):
        """
        Add an entry to the call transcript.
        
        Args:
            role: "user" or "assistant"
            content: Message content
        """
        entry = {
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        }
        self.transcript.append(entry)
        
        # Also update in workflow state
        if "transcript" not in self.workflow_state:
            self.workflow_state["transcript"] = []
        self.workflow_state["transcript"].append(entry)
        
        logger.debug(f"📝 Added {role} message to transcript: {content[:50]}...")
    
    def set_call_id(self, call_id: str):
        """Set the call ID for this workflow"""
        self.call_id = call_id
        if "call_id" not in self.workflow_state:
            self.workflow_state["call_id"] = call_id
    
    def process_frame_for_metrics(
        self,
        frame: np.ndarray,
        timestamp: Optional[float] = None
    ):
        """
        Process a frame to extract pose keypoints and calculate metrics.
        This is called during video stream processing.
        
        Args:
            frame: Video frame (BGR format, numpy array)
            timestamp: Optional timestamp for the frame
        """
        # Track frame count for exact frame number tracking
        # Frame capture is always active for ACL analysis
        frame_capture_active = self.workflow_state.get("frame_capture_active", True)
        if not hasattr(self, '_process_frame_count'):
            self._process_frame_count = 0
        elif not hasattr(self, '_frame_capture_was_active'):
            # Track if frame capture was active in previous call
            self._frame_capture_was_active = frame_capture_active
        elif not self._frame_capture_was_active and frame_capture_active:
            # Frame capture just activated - reset counter for clean numbering
            print(f"🟢 [Workflow] 🔄 Frame capture activated - resetting frame counter from {self._process_frame_count} to 1")
            self._process_frame_count = 0
        
        self._process_frame_count += 1
        self._frame_capture_was_active = frame_capture_active
        
        # Store frame number in workflow state for easy access
        self.workflow_state["current_frame_number"] = self._process_frame_count
        
        if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
            print(f"🟢 [Workflow] process_frame_for_metrics() called (call #{self._process_frame_count})")
        
        # Check for person in frame before processing
        person_tool = self.cv_tools.get("person_detection") if self.cv_tools else None
        if person_tool:
            try:
                person_detections = person_tool.detect_persons(frame, method="auto")
                if not person_detections or len(person_detections) == 0:
                    # No person detected - skip frame processing
                    if self._process_frame_count == 1 or self._process_frame_count % 100 == 0:
                        print(f"🟢 [Workflow] ⏸️  No person detected in frame #{self._process_frame_count} - skipping")
                    return
            except Exception as e:
                # If person detection fails, log but continue (don't block processing)
                if self._process_frame_count == 1 or self._process_frame_count % 100 == 0:
                    logger.debug(f"Person detection check failed: {e}")
        
        if not self.cv_tools:
            if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                print(f"🟢 [Workflow] ⚠️  No CV tools available")
            return
        
        pose_tool = self.cv_tools.get("pose_estimation")
        if not pose_tool:
            if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                print(f"🟢 [Workflow] ⚠️  No pose estimation tool available")
            return
        
        try:
            # Check if keypoints are already in workflow state (from MetricsProcessor)
            # This avoids duplicate pose estimation
            keypoints = self.workflow_state.get("latest_pose_keypoints")
            
            if keypoints is None:
                # Only do pose estimation if keypoints aren't already available
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    print(f"🟢 [Workflow] 🎯 Extracting pose keypoints (call #{self._process_frame_count}) - keypoints not in state")
                
                # Extract pose keypoints from frame
                pose_result = pose_tool.estimate_pose(frame, method="hrnet")
                if "error" in pose_result:
                    # Fallback to YOLO
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        print(f"🟢 [Workflow] ⚠️  HRNet failed, trying YOLO fallback")
                    pose_result = pose_tool.estimate_pose(frame, method="yolo")
                
                if "keypoints" in pose_result and pose_result["keypoints"]:
                    keypoints = pose_result["keypoints"]
                    # Store keypoints for metric extraction
                    self.workflow_state["latest_pose_keypoints"] = keypoints
                    self.workflow_state["latest_frame_timestamp"] = timestamp
                else:
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        print(f"🟢 [Workflow] ⚠️  No keypoints extracted from pose estimation")
                    return
            else:
                # Keypoints already available from MetricsProcessor
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    keypoint_count = len(keypoints) if isinstance(keypoints, (list, dict)) else 0
                    print(f"🟢 [Workflow] ✅ Using existing keypoints from MetricsProcessor! Count: {keypoint_count}")
            
            if keypoints:
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    keypoint_count = len(keypoints) if isinstance(keypoints, (list, dict)) else 0
                    validity = self._verify_keypoints_validity(keypoints)
                    print(f"🟢 [Workflow] ✅ Keypoints ready! Count: {keypoint_count}")
                    if not validity["valid"]:
                        print(f"🟢 [Workflow] ⚠️  Keypoint validity issues: {', '.join(validity.get('issues', []))}")
                    else:
                        print(f"🟢 [Workflow] ✅ Keypoints valid ({validity['unique_positions']} unique positions)")
                
                # Ensure timestamp is set
                if timestamp is None:
                    timestamp = self.workflow_state.get("latest_frame_timestamp") or time.time()
                self.workflow_state["latest_frame_timestamp"] = timestamp
                
                # Calculate metrics if we have user requests or default metrics
                metrics_to_calculate = self.user_requests
                if not metrics_to_calculate:
                    # Default to ACL-focused metrics for gymnastics
                    if self.current_activity == "gymnastics":
                        metrics_to_calculate = [
                            "ACL tear risk",  # Primary focus
                            "landing bend angles",
                            "impact force"
                        ]
                    elif self.current_activity == "posture":
                        metrics_to_calculate = [
                            "rounded back",
                            "arched back",
                            "hunched shoulders",
                            "forward head posture",
                            "hyperextended knees",
                            "bowed knees",
                            "hip alignment"
                        ]
                    else:
                        metrics_to_calculate = [
                            "height off floor",
                            "impact force",
                            "landing bend angles",
                            "knee straightness"
                        ]
                
                # Get previous keypoints for velocity/acceleration calculations
                previous_keypoints = self.workflow_state.get("previous_pose_keypoints")
                
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    print(f"🟢 [Workflow] 🧮 Calculating metrics (call #{self._process_frame_count})")
                    print(f"🟢 [Workflow]    Technique: {self.current_technique}")
                    print(f"🟢 [Workflow]    Metrics to calculate: {len(metrics_to_calculate)}")
                
                # Calculate basic metrics first (needed for landing detection)
                metrics = self.technique_metrics.calculate_metrics(
                    keypoints=keypoints,
                    technique=self.current_technique or "unknown",
                    user_requests=metrics_to_calculate,
                    frame_timestamp=timestamp,
                    previous_frame_keypoints=previous_keypoints
                )
                
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    print(f"🟢 [Workflow] ✅ Basic metrics calculated! Count: {len(metrics) if metrics else 0}")
                    if metrics:
                        sample_keys = list(metrics.keys())[:5]
                        print(f"🟢 [Workflow]    Sample metrics: {sample_keys}")
                
                # Landing detection (must happen after basic metrics are calculated)
                # This determines if we should calculate landing bend angles
                landing_detection = None
                in_landing_phase = False
                try:
                    from landing_detection import LandingDetectionTool
                    if not hasattr(self, 'landing_detector'):
                        self.landing_detector = LandingDetectionTool()
                    
                    landing_detection = self.landing_detector.detect_landing_phase(
                        keypoints=keypoints,
                        metrics=metrics,
                        frame_number=self._process_frame_count if hasattr(self, '_process_frame_count') else None,
                        timestamp=timestamp,
                        previous_keypoints=previous_keypoints
                    )
                    
                    in_landing_phase = landing_detection.get("in_landing_phase", False) if landing_detection else False
                    metrics["in_landing_phase"] = 1.0 if in_landing_phase else 0.0
                    
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        if in_landing_phase:
                            print(f"🟢 [Workflow] 🛬 Landing phase detected (confidence: {landing_detection.get('landing_confidence', 0.0):.2f})")
                    
                except Exception as e:
                    logger.debug(f"Landing detection error: {e}")
                    landing_detection = {"in_landing_phase": False}
                    in_landing_phase = False
                    metrics["in_landing_phase"] = 0.0
                
                # Recalculate metrics with landing phase context
                # This ensures landing bend angles are ONLY calculated during landing phases
                if in_landing_phase:
                    # Recalculate with landing context - landing bend angles will now be calculated
                    metrics = self.technique_metrics.calculate_metrics(
                        keypoints=keypoints,
                        technique=self.current_technique or "unknown",
                        user_requests=metrics_to_calculate,
                        frame_timestamp=timestamp,
                        previous_frame_keypoints=previous_keypoints,
                        existing_metrics=metrics  # Pass current metrics so landing phase flag is available
                    )
                    
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        if "landing_knee_bend_min" in metrics and metrics["landing_knee_bend_min"] is not None:
                            flexion = 180.0 - metrics["landing_knee_bend_min"]
                            print(f"🟢 [Workflow] ✅ Landing knee bend calculated: {flexion:.1f}° flexion (joint: {metrics['landing_knee_bend_min']:.1f}°)")
                else:
                    # Not in landing phase - ensure landing angles are cleared/not calculated
                    if "landing_knee_bend_min" in metrics:
                        metrics["landing_knee_bend_min"] = None
                        metrics["landing_knee_bend_left"] = None
                        metrics["landing_knee_bend_right"] = None
                        metrics["landing_knee_bend_avg"] = None
                
                # If in landing phase, enhance ACL risk assessment
                # Landing phases are critical for ACL risk - ensure ACL is calculated during landing
                if in_landing_phase:
                    user_requests_text = " ".join(self.user_requests).lower() if self.user_requests else ""
                    if "acl" in user_requests_text:
                        # Recalculate ACL risk with landing context if not already flagged
                        if metrics.get("acl_risk_flagged", 0.0) == 0.0:
                            # Force ACL calculation during landing (uses landing knee bend angles)
                            acl_metrics = self.technique_metrics._calculate_acl_tear_risk(
                                keypoints, previous_keypoints, timestamp, metrics
                            )
                            metrics.update(acl_metrics)
                
                # Track ACL risk flagged timesteps (ONLY HIGH risk)
                # For gymnastics: ACL risk is ALWAYS tracked (automatically included)
                # For other activities: Only if user explicitly requested ACL analysis
                if metrics.get("acl_risk_flagged", 0.0) == 1.0:
                    # For gymnastics, ACL risk is always tracked automatically
                    # For other activities, check if user requested it
                    acl_requested = False
                    
                    if self.current_activity == "gymnastics":
                        # Gymnastics always tracks ACL risk automatically
                        acl_requested = True
                    else:
                        # For other activities, check if user explicitly requested ACL analysis
                        user_requests_text = " ".join(self.user_requests).lower() if self.user_requests else ""
                        
                        # Also check transcript for ACL requests
                        transcript = self.workflow_state.get("transcript", [])
                        transcript_text = " ".join([entry.get("content", "") for entry in transcript if entry.get("role") == "user"]).lower()
                        
                        acl_requested = (
                            "acl" in user_requests_text or 
                            "acl tear" in user_requests_text or
                            "acl risk" in user_requests_text or
                            "acl injury" in user_requests_text or
                            "anterior cruciate" in user_requests_text or
                            "acl" in transcript_text or
                            "acl tear" in transcript_text or
                            "acl risk" in transcript_text or
                            "acot" in transcript_text  # Handle typo "ACOT" -> "ACL"
                        )
                    
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        if acl_requested:
                            if self.current_activity == "gymnastics":
                                print(f"🟢 [Workflow] ✅ ACL analysis automatically enabled for gymnastics - will flag HIGH risk moments")
                            else:
                                print(f"🟢 [Workflow] ✅ ACL analysis requested - will flag HIGH risk moments")
                        else:
                            print(f"🟢 [Workflow] ⚠️  ACL risk flagged but ACL analysis not requested - skipping timestep storage")
                            print(f"🟢 [Workflow]    user_requests: {self.user_requests}")
                    
                    if acl_requested:
                        # Initialize flagged timesteps list if not exists
                        if "acl_flagged_timesteps" not in self.workflow_state:
                            self.workflow_state["acl_flagged_timesteps"] = []
                        
                        # Get timestamp for this frame
                        frame_ts = timestamp or self.workflow_state.get("latest_frame_timestamp")
                        acl_risk_score = metrics.get("acl_tear_risk_score", 0.0)
                        acl_risk_level = metrics.get("acl_risk_level", "UNKNOWN")
                        
                        # Get detailed risk factors for explanation
                        valgus_angle = metrics.get("acl_max_valgus_angle", 0.0)
                        knee_flexion = metrics.get("acl_min_knee_flexion", 0.0)
                        impact_force = metrics.get("acl_impact_force_N", 0.0)
                        valgus_risk = metrics.get("acl_valgus_risk", 0.0)
                        flexion_risk = metrics.get("acl_insufficient_flexion_risk", 0.0)
                        impact_risk = metrics.get("acl_high_impact_risk", 0.0)
                        
                        # Determine primary risk factors contributing to HIGH risk
                        risk_factors = []
                        if valgus_risk >= 0.5:
                            risk_factors.append(f"Knee valgus collapse ({valgus_angle:.1f}°)")
                        if flexion_risk >= 0.5:
                            risk_factors.append(f"Insufficient knee flexion ({knee_flexion:.1f}°)")
                        if impact_risk >= 0.5:
                            risk_factors.append(f"High impact force ({impact_force:.0f}N)")
                        
                        # Add landing context if available
                        landing_context = None
                        if landing_detection and landing_detection.get("in_landing_phase", False):
                            landing_context = {
                                "in_landing_phase": True,
                                "landing_confidence": landing_detection.get("landing_confidence", 0.0),
                                "phase_type": landing_detection.get("phase_type", "unknown"),
                                "landing_indicators": landing_detection.get("landing_indicators", [])
                            }
                        
                        # Add flagged timestep (avoid duplicates)
                        flagged_entry = {
                            "timestamp": frame_ts,
                            "risk_score": acl_risk_score,
                            "risk_level": acl_risk_level,
                            "valgus_angle": valgus_angle,
                            "knee_flexion": knee_flexion,
                            "impact_force": impact_force,
                            "frame_number": self._process_frame_count if hasattr(self, '_process_frame_count') else None,
                            "primary_risk_factors": risk_factors,  # Why this frame is high risk
                            "valgus_risk_factor": valgus_risk,
                            "flexion_risk_factor": flexion_risk,
                            "impact_risk_factor": impact_risk,
                            "landing_context": landing_context  # Landing phase information
                        }
                        
                        # Only add if not duplicate (check last entry)
                        existing_timesteps = self.workflow_state["acl_flagged_timesteps"]
                        if not existing_timesteps or existing_timesteps[-1].get("timestamp") != frame_ts:
                            self.workflow_state["acl_flagged_timesteps"].append(flagged_entry)
                            if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                                factors_str = ", ".join(risk_factors) if risk_factors else "Multiple risk factors"
                                print(f"🟢 [Workflow] ⚠️  ACL HIGH RISK FLAGGED at timestamp {frame_ts:.2f}s (Score: {acl_risk_score:.2f}) - {factors_str}")
                
                # Merge with existing metrics (accumulate across frames)
                existing_metrics = self.workflow_state.get("metrics", {})
                if existing_metrics:
                    # For metrics that should be accumulated (like rotations), keep max/sum
                    # For others, use latest value
                    for key, value in metrics.items():
                        if "rotation" in key.lower() and "total" in key.lower():
                            # Accumulate rotation metrics
                            if key in existing_metrics:
                                existing_metrics[key] = max(existing_metrics[key], value)
                            else:
                                existing_metrics[key] = value
                        elif "peak" in key.lower():
                            # Keep peak values
                            if key in existing_metrics:
                                existing_metrics[key] = max(existing_metrics[key], value)
                            else:
                                existing_metrics[key] = value
                        else:
                            # Use latest value
                            existing_metrics[key] = value
                    self.workflow_state["metrics"] = existing_metrics
                else:
                    self.workflow_state["metrics"] = metrics
                
                # Store previous keypoints for next calculation
                self.workflow_state["previous_pose_keypoints"] = keypoints.copy() if keypoints else None
                
                # Store landing phases in workflow state
                if landing_detection and hasattr(self, 'landing_detector'):
                    landing_phases = self.landing_detector.get_landing_phases()
                    if landing_phases:
                        self.workflow_state["landing_phases"] = landing_phases
                
                # Store frame-by-frame metrics for analysis
                if "frame_metrics" not in self.workflow_state:
                    self.workflow_state["frame_metrics"] = []
                
                # Store frame for potential clip extraction (if ACL risk is requested)
                user_requests_text = " ".join(self.user_requests).lower() if self.user_requests else ""
                acl_requested = (
                    "acl" in user_requests_text or 
                    "acl tear" in user_requests_text or
                    "acl risk" in user_requests_text or
                    "acl injury" in user_requests_text or
                    "anterior cruciate" in user_requests_text
                )
                
                # Store frame for video segment saving (frame capture always active)
                # Store frame WITH pose overlay for video output
                frame_capture_active = self.workflow_state.get("frame_capture_active", True)
                if frame_capture_active:
                    if "video_clip_extractor" not in self.workflow_state:
                        from video_clip_extractor import VideoClipExtractor
                        self.workflow_state["video_clip_extractor"] = VideoClipExtractor()
                    
                    clip_extractor = self.workflow_state["video_clip_extractor"]
                    # Get annotated frame (with pose overlay) from workflow state if available
                    # Otherwise use latest_frame and draw overlay
                    annotated_frame = self.workflow_state.get("latest_annotated_frame")
                    if annotated_frame is not None:
                        # Use annotated frame from MetricsProcessor (already has pose overlay)
                        frame_to_store = annotated_frame
                    else:
                        # Fallback: get original frame and draw overlay
                        current_frame = self.workflow_state.get("latest_frame")
                        if current_frame is not None:
                            frame_to_store = current_frame.copy()
                            if keypoints:
                                # Draw pose skeleton overlay
                                frame_to_store = self._draw_pose_skeleton_on_frame(frame_to_store, keypoints)
                        else:
                            frame_to_store = None
                    
                    if frame_to_store is not None:
                        clip_extractor.store_frame(
                            frame=frame_to_store,  # Store frame WITH pose overlay
                            frame_number=self._process_frame_count if hasattr(self, '_process_frame_count') else 0,
                            timestamp=timestamp,
                            metrics=metrics.copy() if metrics else {}
                        )
                
                # Create frame metrics entry
                frame_entry = {
                    "frame_number": self._process_frame_count if hasattr(self, '_process_frame_count') else None,
                    "timestamp": timestamp,
                    "metrics": metrics.copy() if metrics else {},
                    "keypoints_valid": self._verify_keypoints_validity(keypoints),
                    "in_landing_phase": landing_detection.get("in_landing_phase", False) if landing_detection else False
                }
                
                # Only store every Nth frame to avoid excessive memory usage (store every 5th frame by default)
                # Store all frames if ACL risk is requested (for detailed analysis)
                store_frame = False
                user_requests_text = " ".join(self.user_requests).lower() if self.user_requests else ""
                acl_requested = (
                    "acl" in user_requests_text or 
                    "acl tear" in user_requests_text or
                    "acl risk" in user_requests_text or
                    "acl injury" in user_requests_text
                )
                
                if acl_requested:
                    # Store all frames when ACL analysis is requested
                    store_frame = True
                elif self._process_frame_count % 5 == 0:
                    # Store every 5th frame otherwise
                    store_frame = True
                
                if store_frame:
                    self.workflow_state["frame_metrics"].append(frame_entry)
                    if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                        print(f"🟢 [Workflow] 💾 Stored frame metrics (total: {len(self.workflow_state['frame_metrics'])})")
                
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    total_metrics = len(self.workflow_state.get("metrics", {}))
                    print(f"🟢 [Workflow] 📊 Total metrics in state: {total_metrics}")
                
                logger.debug(f"📊 Calculated {len(metrics)} metrics from frame")
            else:
                if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                    print(f"🟢 [Workflow] ⚠️  No keypoints available for processing (call #{self._process_frame_count})")
        
        except Exception as e:
            if self._process_frame_count == 1 or self._process_frame_count % 30 == 0:
                print(f"🟢 [Workflow] ❌ ERROR in process_frame_for_metrics (call #{self._process_frame_count}): {e}")
            logger.error(f"❌ Error in process_frame_for_metrics: {e}", exc_info=True)
    
    def _verify_keypoints_validity(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify pose detection is working correctly by checking keypoint validity.
        
        Returns:
            Dictionary with validity information
        """
        validity = {
            "valid": False,
            "keypoint_count": 0,
            "unique_positions": 0,
            "issues": []
        }
        
        if not keypoints:
            validity["issues"].append("No keypoints provided")
            return validity
        
        if isinstance(keypoints, dict):
            validity["keypoint_count"] = len(keypoints)
            
            # Check for duplicate/invalid coordinates (e.g., all at 353.0, 328.0)
            positions = []
            for key, value in keypoints.items():
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    pos = (float(value[0]), float(value[1]))
                    positions.append(pos)
            
            # Count unique positions
            unique_positions = len(set(positions))
            validity["unique_positions"] = unique_positions
            
            # Check if most keypoints are at the same position (pose detection failure)
            if len(positions) > 0:
                position_counts = {}
                for pos in positions:
                    position_counts[pos] = position_counts.get(pos, 0) + 1
                
                max_count = max(position_counts.values()) if position_counts else 0
                if max_count > len(positions) * 0.5:
                    # More than 50% of keypoints at same position = likely failure
                    validity["issues"].append(f"Most keypoints at same position ({max_count}/{len(positions)})")
                    validity["valid"] = False
                elif unique_positions >= len(positions) * 0.7:
                    # At least 70% unique positions = likely valid
                    validity["valid"] = True
                else:
                    validity["issues"].append(f"Low unique position count ({unique_positions}/{len(positions)})")
                    validity["valid"] = False
            else:
                validity["issues"].append("No valid positions found")
                validity["valid"] = False
        else:
            validity["issues"].append(f"Keypoints not in expected format (got {type(keypoints)})")
            validity["valid"] = False
        
        return validity
    
    def add_overlay_frame(
        self,
        frame: np.ndarray,
        fps: float = 30.0,
        timestamp: Optional[float] = None
    ):
        """
        Add a frame with overlays to the workflow state for video output.
        Also processes the frame to extract pose and calculate metrics.
        
        Args:
            frame: Frame with overlays (BGR format, numpy array)
            fps: Frames per second
            timestamp: Optional timestamp for the frame
        """
        if "frames" not in self.workflow_state:
            self.workflow_state["frames"] = []
        
        self.workflow_state["frames"].append({
            "frame": frame.copy() if isinstance(frame, np.ndarray) else frame,
            "fps": fps,
            "timestamp": timestamp or len(self.workflow_state["frames"]) / fps
        })
        
        # Process frame to extract pose and calculate metrics
        self.process_frame_for_metrics(frame, timestamp)
    
    def _draw_pose_skeleton_on_frame(self, frame: np.ndarray, keypoints: Dict[str, Any]) -> np.ndarray:
        """
        Draw pose skeleton overlay on frame.
        Similar to MetricsProcessor._draw_pose_skeleton but for workflow use.
        
        Args:
            frame: Input frame (BGR format)
            keypoints: Dictionary of keypoint name -> [x, y] coordinates
        
        Returns:
            Frame with pose skeleton drawn
        """
        overlay = frame.copy()
        
        # Handle different keypoint formats
        if isinstance(keypoints, dict):
            # Check if it's a dict of keypoint objects or coordinate lists
            if all(isinstance(v, (list, tuple)) and len(v) >= 2 for v in keypoints.values() if v is not None):
                # Direct coordinate format: {"left_shoulder": [x, y], ...}
                kp_dict = keypoints
            else:
                # Keypoint object format: extract coordinates
                kp_dict = {}
                for name, kp in keypoints.items():
                    if kp is not None:
                        if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                            kp_dict[name] = kp
                        elif isinstance(kp, dict) and "x" in kp and "y" in kp:
                            kp_dict[name] = [kp["x"], kp["y"]]
                        elif hasattr(kp, "x") and hasattr(kp, "y"):
                            kp_dict[name] = [kp.x, kp.y]
        else:
            return overlay  # Unknown format, return original
        
        # Define skeleton connections (COCO format)
        skeleton_connections = [
            # Head
            ("nose", "left_eye"),
            ("nose", "right_eye"),
            ("left_eye", "left_ear"),
            ("right_eye", "right_ear"),
            # Torso
            ("left_shoulder", "right_shoulder"),
            ("left_shoulder", "left_hip"),
            ("right_shoulder", "right_hip"),
            ("left_hip", "right_hip"),
            # Left arm
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            # Right arm
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            # Left leg
            ("left_hip", "left_knee"),
            ("left_knee", "left_ankle"),
            # Right leg
            ("right_hip", "right_knee"),
            ("right_knee", "right_ankle"),
        ]
        
        # Draw skeleton connections (green lines)
        for start_key, end_key in skeleton_connections:
            start_point = kp_dict.get(start_key)
            end_point = kp_dict.get(end_key)
            
            if start_point and end_point and len(start_point) >= 2 and len(end_point) >= 2:
                # Check if points are valid (not NaN)
                try:
                    start_x, start_y = float(start_point[0]), float(start_point[1])
                    end_x, end_y = float(end_point[0]), float(end_point[1])
                    
                    if not (np.isnan(start_x) or np.isnan(start_y) or np.isnan(end_x) or np.isnan(end_y)):
                        start = (int(start_x), int(start_y))
                        end = (int(end_x), int(end_y))
                        cv2.line(overlay, start, end, (0, 255, 0), 2)  # Green lines, thickness 2
                except (ValueError, TypeError, IndexError):
                    continue  # Skip invalid points
        
        # Draw keypoints (green circles)
        for keypoint_name, coords in kp_dict.items():
            if coords and len(coords) >= 2:
                try:
                    x, y = float(coords[0]), float(coords[1])
                    if not (np.isnan(x) or np.isnan(y)):
                        cv2.circle(overlay, (int(x), int(y)), 5, (0, 255, 0), -1)  # Green circles, radius 5
                except (ValueError, TypeError, IndexError):
                    continue  # Skip invalid points
        
        return overlay
    
    def get_workflow_summary(self) -> Dict[str, Any]:
        """Get summary of workflow execution"""
        return {
            "technique": self.current_technique,
            "user_requests": self.user_requests,
            "total_steps": len(self.workflow_sequence),
            "completed_steps": len(self.completed_steps),
            "pending_steps": len([t for t in self.pending_tasks if t["status"] == "pending"]),
            "metrics": self.workflow_state.get("metrics", {}),
            "analysis": self.workflow_state.get("analysis", {}),
            "workflow_state": self.workflow_state
        }






















