#!/usr/bin/env python3
"""
Video Inference Script for cvMLAgent
Analyzes video using HRNet pose estimation and other CV tools
"""

import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cv_tools import PoseEstimationTool, PersonDetectionTool, FaceDetectionTool
from technique_metrics import TechniqueSupervisedMetrics
from memory_integration import MemoryIndexIntegration
from model_selector import MLModelSelector
from deep_search_reasoning import DeepSearchReasoning
from segmentation_tool import SegmentationTool
from face_recognition_tool import FaceRecognitionTool
from segmentation_tool import SegmentationTool
from face_recognition_tool import FaceRecognitionTool

class VideoInference:
    """Inference engine for video analysis"""
    
    def __init__(
        self,
        use_deep_search: bool = True,
        use_llm_reasoning: bool = False,
        llm_instance=None,
        use_segmentation: bool = True,
        use_face_recognition: bool = True
    ):
        self.pose_tool = PoseEstimationTool()
        self.person_tool = PersonDetectionTool()
        self.face_tool = FaceDetectionTool()
        self.memory = MemoryIndexIntegration()
        self.technique_metrics = TechniqueSupervisedMetrics(self.memory)
        self.model_selector = MLModelSelector()
        self.use_deep_search = use_deep_search
        self.use_llm_reasoning = use_llm_reasoning
        self.llm_instance = llm_instance
        self.use_segmentation = use_segmentation
        self.use_face_recognition = use_face_recognition
        
        # Initialize segmentation tool
        if use_segmentation:
            self.segmentation_tool = SegmentationTool()
            logger.info("✅ Initialized Segmentation Tool")
        else:
            self.segmentation_tool = None
        
        # Initialize face recognition tool
        if use_face_recognition:
            self.face_recognition_tool = FaceRecognitionTool()
            logger.info("✅ Initialized Face Recognition Tool")
        else:
            self.face_recognition_tool = None
        
        # Initialize deep search if enabled
        if use_deep_search:
            self.deep_search = DeepSearchReasoning(llm_instance=llm_instance)
            logger.info("✅ Initialized Deep Search & Reasoning")
        else:
            self.deep_search = None
        
        logger.info("✅ Initialized CV tools")
    
    def analyze_video(
        self,
        video_path: str,
        technique: str = "bb_back_handspring",
        user_requests: List[str] = None,
        output_dir: str = "inference_output",
        save_overlay_video: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze video with full CV pipeline.
        
        Args:
            video_path: Path to video file
            technique: Technique name
            user_requests: List of requested metrics
            output_dir: Output directory for results
        
        Returns:
            Analysis results
        """
        if user_requests is None:
            user_requests = [
                "track height off floor",
                "impact force",
                "landing bend angles",
                "knee straightness"
            ]
        
        logger.info("=" * 60)
        logger.info(f"Analyzing video: {video_path}")
        logger.info(f"Technique: {technique}")
        logger.info(f"Requested metrics: {', '.join(user_requests)}")
        logger.info("=" * 60)
        
        # Select models based on requests
        logger.info("\n🔍 Selecting models...")
        selected_models, reasoning = self.model_selector.select_models(
            user_requests=user_requests,
            technique=technique
        )
        logger.info(f"✅ Selected {len(selected_models)} models")
        for model_type, variant in selected_models.items():
            logger.info(f"   {model_type.value}: {variant.value}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"\n📹 Video info:")
        logger.info(f"   FPS: {fps:.2f}")
        logger.info(f"   Frames: {total_frames}")
        logger.info(f"   Resolution: {width}x{height}")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Setup video writer for overlay video if requested
        video_writer = None
        overlay_video_path = None
        if save_overlay_video:
            overlay_video_path = output_path / f"{Path(video_path).stem}_overlay.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(overlay_video_path),
                fourcc,
                fps,
                (width, height)
            )
            logger.info(f"📹 Will save overlay video to: {overlay_video_path}")
        
        # Process video
        frame_results = []
        frame_count = 0
        previous_keypoints = None
        
        logger.info(f"\n🔄 Processing video frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            timestamp = frame_count / fps if fps > 0 else frame_count / 30.0
            
            if frame_count % 30 == 0:  # Log every second (at 30fps)
                logger.info(f"   Processing frame {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
            
            frame_result = {
                "frame_number": frame_count,
                "timestamp": timestamp,
                "keypoints": {},
                "metrics": {},
                "detections": {}
            }
            
            # Person detection
            try:
                persons = self.person_tool.detect_persons(frame, method="hrnet")
                if not persons:
                    persons = self.person_tool.detect_persons(frame, method="yolo")
                frame_result["detections"]["persons"] = len(persons)
            except Exception as e:
                logger.debug(f"Person detection error on frame {frame_count}: {e}")
                frame_result["detections"]["persons"] = 0
            
            # Pose estimation (primary analysis)
            pose_keypoints = None
            try:
                pose_result = self.pose_tool.estimate_pose(frame, method="hrnet")
                if "error" in pose_result:
                    # Fallback to YOLO
                    pose_result = self.pose_tool.estimate_pose(frame, method="yolo")
                
                if "keypoints" in pose_result and pose_result["keypoints"]:
                    keypoints = pose_result["keypoints"]
                    pose_keypoints = keypoints
                    frame_result["keypoints"] = keypoints
                    
                    # Calculate metrics (includes velocity and acceleration)
                    try:
                        metrics = self.technique_metrics.calculate_metrics(
                            keypoints=keypoints,
                            technique=technique,
                            user_requests=user_requests,
                            frame_timestamp=timestamp,
                            previous_frame_keypoints=previous_keypoints
                        )
                        frame_result["metrics"] = metrics
                    except Exception as e:
                        logger.debug(f"Metrics calculation error on frame {frame_count}: {e}")
                    
                    previous_keypoints = keypoints
            except Exception as e:
                logger.debug(f"Pose estimation error on frame {frame_count}: {e}")
            
            # Face detection (optional)
            try:
                faces = self.face_tool.detect_faces(frame)
                frame_result["detections"]["faces"] = len(faces)
            except Exception as e:
                logger.debug(f"Face detection error on frame {frame_count}: {e}")
                frame_result["detections"]["faces"] = 0
            
            # Create overlay frame starting with original frame
            overlay_frame = frame.copy()
            
            # Draw pose skeleton on overlay (if keypoints available)
            if pose_keypoints:
                overlay_frame = self._draw_pose_skeleton(overlay_frame, pose_keypoints)
            
            # Segmentation (if enabled)
            if self.use_segmentation and self.segmentation_tool:
                try:
                    seg_result = self.segmentation_tool.segment_frame(frame)
                    if seg_result.get("segments"):
                        frame_result["detections"]["segments"] = len(seg_result["segments"])
                        overlay_frame = self.segmentation_tool.draw_segmentation_overlay(
                            overlay_frame,
                            seg_result["segments"]
                        )
                except Exception as e:
                    logger.debug(f"Segmentation error on frame {frame_count}: {e}")
            
            # Face recognition (if enabled)
            if self.use_face_recognition and self.face_recognition_tool:
                try:
                    face_detections = self.face_recognition_tool.detect_and_identify(frame)
                    if face_detections:
                        frame_result["detections"]["recognized_faces"] = len([
                            f for f in face_detections if f.get("identified", False)
                        ])
                        frame_result["detections"]["unknown_faces"] = len([
                            f for f in face_detections if not f.get("identified", False)
                        ])
                        overlay_frame = self.face_recognition_tool.draw_face_overlay(
                            overlay_frame,
                            face_detections
                        )
                except Exception as e:
                    logger.debug(f"Face recognition error on frame {frame_count}: {e}")
            
            # Write overlay frame to video if enabled
            if video_writer:
                video_writer.write(overlay_frame)
            
            frame_results.append(frame_result)
        
        cap.release()
        
        # Release video writer
        if video_writer:
            video_writer.release()
            logger.info(f"✅ Overlay video saved to: {overlay_video_path}")
        
        logger.info(f"\n✅ Processed {len(frame_results)} frames")
        
        # Aggregate results
        logger.info("\n📊 Aggregating results...")
        aggregated = self._aggregate_results(frame_results, technique)
        
        # Compare to standards with deep research
        logger.info("\n📋 Comparing to standards with deep research...")
        standards_comparison = self.memory.compare_to_standards(
            technique=technique,
            metrics=aggregated.get("mean_metrics", {}),
            use_deep_research=self.use_deep_search  # Use deep research if enabled
        )
        
        # Save results
        output_file = output_path / f"{Path(video_path).stem}_analysis.json"
        results = {
            "video_path": video_path,
            "technique": technique,
            "user_requests": user_requests,
            "video_info": {
                "fps": fps,
                "total_frames": total_frames,
                "resolution": f"{width}x{height}",
                "duration": total_frames / fps if fps > 0 else 0
            },
            "selected_models": {k.value: v.value for k, v in selected_models.items()},
            "frame_results": frame_results,
            "aggregated_metrics": aggregated,
            "standards_comparison": standards_comparison,
            "overlay_video_path": str(overlay_video_path) if overlay_video_path else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"✅ Results saved to: {output_file}")
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _aggregate_results(
        self,
        frame_results: List[Dict[str, Any]],
        technique: str
    ) -> Dict[str, Any]:
        """Aggregate metrics across frames"""
        all_metrics = []
        
        for frame_result in frame_results:
            if frame_result.get("metrics"):
                all_metrics.append(frame_result["metrics"])
        
        if not all_metrics:
            return {"mean_metrics": {}, "min_metrics": {}, "max_metrics": {}}
        
        # Aggregate
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())
        
        aggregated = {
            "mean_metrics": {},
            "min_metrics": {},
            "max_metrics": {},
            "frame_count": len(all_metrics)
        }
        
        for metric_name in metric_names:
            values = [m.get(metric_name) for m in all_metrics if metric_name in m and m[metric_name] is not None]
            if values:
                aggregated["mean_metrics"][metric_name] = float(np.mean(values))
                aggregated["min_metrics"][metric_name] = float(np.min(values))
                aggregated["max_metrics"][metric_name] = float(np.max(values))
        
        return aggregated
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print analysis summary"""
        logger.info("\n" + "=" * 60)
        logger.info("ANALYSIS SUMMARY")
        logger.info("=" * 60)
        
        video_info = results.get("video_info", {})
        logger.info(f"\n📹 Video: {Path(results['video_path']).name}")
        logger.info(f"   Duration: {video_info.get('duration', 0):.2f}s")
        logger.info(f"   Frames: {video_info.get('total_frames', 0)}")
        
        aggregated = results.get("aggregated_metrics", {})
        mean_metrics = aggregated.get("mean_metrics", {})
        
        if mean_metrics:
            logger.info(f"\n📊 Key Metrics (averaged):")
            for metric_name, value in list(mean_metrics.items())[:10]:
                logger.info(f"   {metric_name}: {value:.2f}")
        
        comparison = results.get("standards_comparison", {})
        if comparison.get("standards_checked"):
            logger.info(f"\n📋 Standards Comparison:")
            logger.info(f"   All standards met: {comparison.get('all_met', False)}")
            gaps = comparison.get("gaps", [])
            if gaps:
                logger.info(f"   Gaps found: {len(gaps)}")
                for gap in gaps[:5]:
                    metric = gap.get("metric", "unknown")
                    current = gap.get("current_value", 0)
                    target = gap.get("target_value", 0)
                    gap_val = gap.get("gap", 0)
                    logger.info(f"     {metric}: {current:.2f} (target: {target:.2f}, gap: {gap_val:.2f})")
        
        logger.info("\n" + "=" * 60)
    
    def _draw_pose_skeleton(self, frame: np.ndarray, keypoints: Dict[str, List[float]]) -> np.ndarray:
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: Input frame
            keypoints: Dictionary of keypoint name -> [x, y] coordinates
        
        Returns:
            Frame with pose skeleton drawn
        """
        overlay = frame.copy()
        
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
        
        # Draw connections
        for start_key, end_key in skeleton_connections:
            start_point = keypoints.get(start_key)
            end_point = keypoints.get(end_key)
            
            if start_point and end_point and len(start_point) >= 2 and len(end_point) >= 2:
                # Check if points are valid (not NaN)
                if not (np.isnan(start_point[0]) or np.isnan(start_point[1]) or
                       np.isnan(end_point[0]) or np.isnan(end_point[1])):
                    start = (int(start_point[0]), int(start_point[1]))
                    end = (int(end_point[0]), int(end_point[1]))
                    cv2.line(overlay, start, end, (0, 255, 0), 2)  # Green lines
        
        # Draw keypoints
        for keypoint_name, coords in keypoints.items():
            if coords and len(coords) >= 2:
                if not (np.isnan(coords[0]) or np.isnan(coords[1])):
                    x, y = int(coords[0]), int(coords[1])
                    cv2.circle(overlay, (x, y), 5, (0, 255, 0), -1)  # Green circles
        
        return overlay


if __name__ == "__main__":
    import sys
    
    # Default video path
    video_path = "../Practice1/BB_Back_Handspring_1_overlay.mp4"
    
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    
    # Check if video exists
    if not Path(video_path).exists():
        # Try absolute path
        video_path = f"/Users/eishahemchand/meshTest/Practice1/BB_Back_Handspring_1_overlay.mp4"
        if not Path(video_path).exists():
            logger.error(f"❌ Video not found: {video_path}")
            logger.error("Please provide a valid video path")
            sys.exit(1)
    
    # Initialize inference
    inference = VideoInference()
    
    # Analyze video
    try:
        results = inference.analyze_video(
            video_path=video_path,
            technique="bb_back_handspring",
            user_requests=[
                "track height off floor",
                "impact force",
                "landing bend angles",
                "knee straightness",
                "stiffness"
            ]
        )
        
        logger.info("\n✅ Video analysis complete!")
        logger.info(f"Results saved to: inference_output/{Path(video_path).stem}_analysis.json")
        
    except Exception as e:
        logger.error(f"❌ Error analyzing video: {e}", exc_info=True)
        sys.exit(1)





















