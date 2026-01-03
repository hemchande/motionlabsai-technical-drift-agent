#!/usr/bin/env python3
"""
Metrics Processor for CV ML Agent
Processes video frames to extract pose and calculate metrics for the workflow.
"""

import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)

# Try to import processor base classes if available
try:
    from vision_agents.core.processors.base_processor import VideoProcessor
    import aiortc
    from vision_agents.core.utils.video_forwarder import VideoForwarder
    PROCESSOR_AVAILABLE = True
except ImportError:
    PROCESSOR_AVAILABLE = False
    VideoProcessor = object
    VideoForwarder = None


class MetricsProcessor(VideoProcessor):
    """
    Processor that processes video frames to extract pose and calculate metrics.
    Integrates with the workflow to store metrics.
    """
    
    name = "metrics_processor"
    
    def __init__(self, workflow, cv_tools):
        """
        Initialize metrics processor.
        
        Args:
            workflow: AgenticMLWorkflow instance to store metrics
            cv_tools: Dictionary of CV tools (pose_estimation, etc.)
        """
        if PROCESSOR_AVAILABLE:
            super().__init__()
        
        self.workflow = workflow
        self.cv_tools = cv_tools
        self.frame_count = 0
        self.latest_frame = None
        self.latest_pose_data = None
        self.latest_keypoints = None
        self._video_forwarder: Optional[VideoForwarder] = None
        self._shutdown = False
        
        # Thread pool for CPU-intensive pose processing (prevents blocking event loop)
        self.executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="metrics_processor"
        )
        
        print("🔵 [MetricsProcessor] ✅ INITIALIZED")
        logger.info("✅ MetricsProcessor initialized")
        print(f"🔵 [MetricsProcessor] Workflow: {workflow is not None}, CV Tools: {len(cv_tools) if cv_tools else 0}")
        print(f"🔵 [MetricsProcessor] ThreadPoolExecutor created (max_workers=2) to prevent event loop blocking")
    
    def _process_pose_sync(self, frame_array: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Synchronous method to process pose estimation.
        This runs in a thread pool executor to avoid blocking the event loop.
        
        Args:
            frame_array: Video frame as numpy array (BGR format)
        
        Returns:
            Tuple of (frame_array, pose_data)
        """
        try:
            # Extract pose using pose estimation tool
            pose_tool = self.cv_tools.get("pose_estimation") if self.cv_tools else None
            if not pose_tool:
                return frame_array, None
            
            # Use HRNet as primary method
            pose_result = pose_tool.estimate_pose(frame_array, method="hrnet")
            if "error" in pose_result:
                # Fallback to YOLO
                pose_result = pose_tool.estimate_pose(frame_array, method="yolo")
            
            return frame_array, pose_result if ("keypoints" in pose_result and pose_result["keypoints"]) else None
            
        except Exception as e:
            logger.error(f"❌ Error in _process_pose_sync: {e}", exc_info=True)
            return frame_array, None
    
    async def add_pose_to_ndarray(self, frame_array: np.ndarray, **kwargs) -> tuple:
        """
        Process frame array to extract pose and calculate metrics.
        This method is called by vision-agents framework for each frame.
        Uses thread pool executor to prevent blocking the event loop.
        
        Args:
            frame_array: Video frame as numpy array (BGR format)
            **kwargs: Additional arguments
        
        Returns:
            Tuple of (annotated_frame, pose_data)
        """
        # Exit early if processor is shutting down
        if self._shutdown:
            return frame_array, None
        
        try:
            self.frame_count += 1
            
            # Frame capture is always active for ACL analysis - no check needed
            # Removed inactive frame check
            
            self.latest_frame = frame_array.copy()
            
            # Check for person in frame before processing
            person_tool = self.cv_tools.get("person_detection") if self.cv_tools else None
            if person_tool:
                try:
                    # Quick person detection check
                    person_detections = person_tool.detect_persons(frame_array, method="auto")
                    if not person_detections or len(person_detections) == 0:
                        # No person detected - skip frame processing
                        if self.frame_count == 1 or self.frame_count % 100 == 0:
                            print(f"🔵 [MetricsProcessor] ⏸️  No person detected in frame #{self.frame_count} - skipping")
                        return frame_array, None
                except Exception as e:
                    # If person detection fails, log but continue (don't block processing)
                    if self.frame_count == 1 or self.frame_count % 100 == 0:
                        logger.debug(f"Person detection check failed: {e}")
            
            # Print every 30 frames to avoid spam
            if self.frame_count % 30 == 0:
                print(f"🔵 [MetricsProcessor] 📊 Processing frame #{self.frame_count} (shape: {frame_array.shape})")
            
            # Run pose estimation in thread pool to avoid blocking event loop
            loop = asyncio.get_event_loop()
            try:
                frame_array_copy, pose_result = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, self._process_pose_sync, frame_array.copy()),
                    timeout=5.0  # 5 second timeout
                )
                
                # Verify pose detection quality
                if pose_result and "keypoints" in pose_result:
                    keypoints = pose_result["keypoints"]
                    validity = self._verify_pose_detection(keypoints)
                    if not validity["valid"] and (self.frame_count == 0 or (self.frame_count + 1) % 30 == 0):
                        print(f"🔵 [MetricsProcessor] ⚠️  Pose detection issues on frame {self.frame_count + 1}: {', '.join(validity.get('issues', []))}")
                    elif validity["valid"] and (self.frame_count == 0 or (self.frame_count + 1) % 30 == 0):
                        print(f"🔵 [MetricsProcessor] ✅ Pose detection valid on frame {self.frame_count + 1} ({validity['unique_positions']} unique positions)")
            except asyncio.TimeoutError:
                if self.frame_count % 30 == 0:
                    print(f"🔵 [MetricsProcessor] ⏰ Pose estimation timeout for frame #{self.frame_count}")
                logger.warning(f"⏰ Pose estimation timeout for frame {self.frame_count}")
                return frame_array, None
            
            if pose_result and "keypoints" in pose_result and pose_result["keypoints"]:
                self.latest_pose_data = pose_result
                self.latest_keypoints = pose_result["keypoints"]
                
                keypoint_count = len(pose_result["keypoints"]) if isinstance(pose_result["keypoints"], (list, dict)) else 0
                if self.frame_count == 1 or self.frame_count % 30 == 0:
                    print(f"🔵 [MetricsProcessor] ✅ Pose detected! Frame #{self.frame_count}, Keypoints: {keypoint_count}")
                
                # Store in workflow state for metric calculation
                if self.workflow:
                    timestamp = time.time()
                    # Store original frame for processing
                    self.workflow.workflow_state["latest_frame"] = frame_array
                    # Also store annotated frame (with pose overlay) for video output
                    self.workflow.workflow_state["latest_annotated_frame"] = annotated_frame
                    self.workflow.workflow_state["latest_pose_keypoints"] = pose_result["keypoints"]
                    self.workflow.workflow_state["latest_frame_timestamp"] = timestamp
                    # Sync frame number with workflow
                    if not hasattr(self.workflow, '_process_frame_count'):
                        self.workflow._process_frame_count = 0
                    self.workflow._process_frame_count = self.frame_count
                    self.workflow.workflow_state["current_frame_number"] = self.frame_count
                    
                    # Process frame for metrics (also in executor if it's blocking)
                    if hasattr(self.workflow, 'process_frame_for_metrics'):
                        if self.frame_count == 1 or self.frame_count % 30 == 0:
                            print(f"🔵 [MetricsProcessor] 🧮 Calculating metrics for frame #{self.frame_count}")
                        
                        # Run metric calculation in executor to avoid blocking
                        try:
                            await asyncio.wait_for(
                                loop.run_in_executor(
                                    self.executor,
                                    self.workflow.process_frame_for_metrics,
                                    frame_array,
                                    timestamp
                                ),
                                timeout=2.0  # 2 second timeout for metrics
                            )
                            
                            # Check if metrics were calculated
                            metrics = self.workflow.workflow_state.get("metrics", {})
                            if self.frame_count == 1 or self.frame_count % 30 == 0:
                                metric_count = len(metrics) if metrics else 0
                                print(f"🔵 [MetricsProcessor] 📈 Metrics calculated! Total metrics: {metric_count}")
                                if metric_count > 0:
                                    sample_keys = list(metrics.keys())[:5]
                                    print(f"🔵 [MetricsProcessor]    Sample metrics: {sample_keys}")
                        except asyncio.TimeoutError:
                            if self.frame_count % 30 == 0:
                                print(f"🔵 [MetricsProcessor] ⏰ Metrics calculation timeout for frame #{self.frame_count}")
                            logger.warning(f"⏰ Metrics calculation timeout for frame {self.frame_count}")
                    
                    if self.frame_count % 30 == 0:  # Log every 30 frames
                        logger.info(f"📊 Processed {self.frame_count} frames, calculated metrics")
            else:
                if self.frame_count == 1 or self.frame_count % 30 == 0:
                    print(f"🔵 [MetricsProcessor] ⚠️  No keypoints detected in frame #{self.frame_count}")
                logger.debug(f"⚠️  No keypoints detected in frame {self.frame_count}")
            
            # Draw pose overlay on frame if pose was detected
            # This overlay will appear in Stream.io recorded video
            annotated_frame = frame_array.copy()
            if pose_result and "keypoints" in pose_result and pose_result["keypoints"]:
                annotated_frame = self._draw_pose_skeleton(annotated_frame, pose_result["keypoints"])
            
            # Return annotated frame (with pose overlay) for Stream.io recording
            return annotated_frame, self.latest_pose_data
            
        except Exception as e:
            print(f"🔵 [MetricsProcessor] ❌ ERROR in add_pose_to_ndarray (frame #{self.frame_count}): {e}")
            logger.error(f"❌ Error in MetricsProcessor.add_pose_to_ndarray: {e}", exc_info=True)
            return frame_array, None
    
    def _draw_pose_skeleton(self, frame: np.ndarray, keypoints: Dict[str, Any]) -> np.ndarray:
        """
        Draw pose skeleton overlay on frame.
        This overlay will appear in Stream.io recorded video.
        
        Args:
            frame: Input frame (BGR format)
            keypoints: Dictionary of keypoint name -> [x, y] coordinates or keypoint dict
        
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
    
    async def add_pose_to_frame(self, frame, **kwargs):
        """
        Process VideoFrame object (if framework uses VideoFrame instead of ndarray).
        
        Args:
            frame: VideoFrame object
            **kwargs: Additional arguments
        
        Returns:
            Processed frame
        """
        try:
            if self.frame_count == 0 or (self.frame_count + 1) % 30 == 0:
                print(f"🔵 [MetricsProcessor] 🖼️  add_pose_to_frame() called (frame will be #{self.frame_count + 1})")
            
            # Convert VideoFrame to numpy array if needed
            if hasattr(frame, 'to_ndarray'):
                frame_array = frame.to_ndarray(format="bgr24")
            else:
                # Try to get array directly
                frame_array = np.array(frame) if hasattr(frame, '__array__') else None
            
            if frame_array is not None:
                # Process using add_pose_to_ndarray
                annotated_frame, pose_data = await self.add_pose_to_ndarray(frame_array, **kwargs)
                return annotated_frame
            else:
                if self.frame_count == 0 or (self.frame_count + 1) % 30 == 0:
                    print(f"🔵 [MetricsProcessor] ⚠️  Frame array is None in add_pose_to_frame")
                return frame
                
        except Exception as e:
            print(f"🔵 [MetricsProcessor] ❌ ERROR in add_pose_to_frame: {e}")
            logger.error(f"❌ Error in MetricsProcessor.add_pose_to_frame: {e}", exc_info=True)
            return frame
    
    async def process_video(
        self,
        track: "aiortc.VideoStreamTrack",
        participant_id: Optional[str] = None,
        shared_forwarder: Optional[VideoForwarder] = None,
    ) -> None:
        """
        Called by framework when video track is received.
        Sets up video processing and frame handlers.
        
        Args:
            track: Incoming video stream track
            participant_id: Optional participant ID
            shared_forwarder: Optional shared video forwarder
        """
        print("🔵 [MetricsProcessor] 🎬 process_video() CALLED by framework!")
        print(f"🔵 [MetricsProcessor]    Participant ID: {participant_id}")
        print(f"🔵 [MetricsProcessor]    Shared forwarder: {shared_forwarder is not None}")
        
        if not PROCESSOR_AVAILABLE:
            print("🔵 [MetricsProcessor] ⚠️  VideoProcessor not available - process_video may not work correctly")
            logger.warning("⚠️  VideoProcessor not available - process_video may not work correctly")
            return
        
        if self._video_forwarder is not None:
            print("🔵 [MetricsProcessor] 🎥 Stopping ongoing video processing for new track")
            logger.info("🎥 Stopping ongoing video processing for new track")
            await self._video_forwarder.remove_frame_handler(self._process_frame)
        
        print("🔵 [MetricsProcessor] 📹 Starting MetricsProcessor video processing")
        logger.info(f"📹 Starting MetricsProcessor video processing")
        self._video_forwarder = (
            shared_forwarder
            if shared_forwarder
            else VideoForwarder(
                track,
                max_buffer=30,  # 1 second at 30fps
                fps=30,
                name="metrics_forwarder",
            )
        )
        self._video_forwarder.add_frame_handler(
            self._process_frame, fps=30.0, name="metrics"
        )
        print("🔵 [MetricsProcessor] ✅ Video processing ACTIVE - frames will be processed!")
        logger.info("✅ MetricsProcessor video processing active")
    
    async def _process_frame(self, frame):
        """
        Internal frame handler called by VideoForwarder.
        Processes each frame to extract pose and calculate metrics.
        """
        try:
            # Print first frame and every 30th frame
            if self.frame_count == 0 or (self.frame_count + 1) % 30 == 0:
                print(f"🔵 [MetricsProcessor] 🎞️  _process_frame() called (frame will be #{self.frame_count + 1})")
            
            # Convert VideoFrame to numpy array
            if hasattr(frame, 'to_ndarray'):
                frame_array = frame.to_ndarray(format="bgr24")
            else:
                frame_array = np.array(frame)
            
            if frame_array is not None:
                # Process frame
                await self.add_pose_to_ndarray(frame_array)
            else:
                if self.frame_count == 0 or (self.frame_count + 1) % 30 == 0:
                    print(f"🔵 [MetricsProcessor] ⚠️  Frame array is None!")
        except Exception as e:
            print(f"🔵 [MetricsProcessor] ❌ ERROR in _process_frame: {e}")
            logger.error(f"❌ Error in _process_frame: {e}", exc_info=True)
    
    def _verify_pose_detection(self, keypoints: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify pose detection quality by checking keypoint validity.
        Similar to ml_workflow._verify_keypoints_validity but for processor.
        
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
            
            # Check for duplicate/invalid coordinates
            positions = []
            for key, value in keypoints.items():
                if isinstance(value, (list, tuple)) and len(value) >= 2:
                    try:
                        pos = (float(value[0]), float(value[1]))
                        positions.append(pos)
                    except (ValueError, TypeError):
                        continue
            
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
    
    async def close(self) -> None:
        """Close the processor and clean up resources"""
        print(f"🔵 [MetricsProcessor] 🔚 close() called - Total frames processed: {self.frame_count}")
        
        # Mark as shutting down
        self._shutdown = True
        
        if self._video_forwarder:
            try:
                await self._video_forwarder.remove_frame_handler(self._process_frame)
            except Exception as e:
                logger.debug(f"Error removing frame handler: {e}")
        
        # Shutdown thread pool executor
        if hasattr(self, 'executor') and self.executor:
            print(f"🔵 [MetricsProcessor] 🔄 Shutting down ThreadPoolExecutor...")
            try:
                # timeout parameter was added in Python 3.9+
                import sys
                if sys.version_info >= (3, 9):
                    self.executor.shutdown(wait=True, timeout=5.0)
                else:
                    self.executor.shutdown(wait=True)
            except TypeError:
                # Fallback for older Python versions
                self.executor.shutdown(wait=True)
            print(f"🔵 [MetricsProcessor] ✅ ThreadPoolExecutor shut down")
        
        # Print final metrics summary
        if self.workflow:
            final_metrics = self.workflow.workflow_state.get("metrics", {})
            metric_count = len(final_metrics) if final_metrics else 0
            print(f"🔵 [MetricsProcessor] 📊 FINAL SUMMARY:")
            print(f"🔵 [MetricsProcessor]    Total frames processed: {self.frame_count}")
            print(f"🔵 [MetricsProcessor]    Total metrics calculated: {metric_count}")
            if metric_count > 0:
                print(f"🔵 [MetricsProcessor]    Metrics: {list(final_metrics.keys())[:10]}")
        
        logger.info("✅ MetricsProcessor closed")
















