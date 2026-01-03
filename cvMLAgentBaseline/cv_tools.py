#!/usr/bin/env python3
"""
CV Tools for Gymnastics Analytics
Face detection, weight detection, pose estimation, person detection
"""

import logging
import cv2
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import threading

# HRNet imports (from ONNX-HRNET repo, similar to hrnet-metrics-server)
HRNET_AVAILABLE = False
HRNET = None
PersonDetector = None
ModelType = None
filter_person_detections = None

# Try to find HRNet repo
HRNET_REPO_CANDIDATES = [
    os.getenv("HRNET_REPO_PATH"),
    "/Users/eishahemchand/ONNX-HRNET-Human-Pose-Estimation",
    "../ONNX-HRNET-Human-Pose-Estimation",
    "./ONNX-HRNET-Human-Pose-Estimation",
]

DEFAULT_HRNET_REPO = None
for candidate in HRNET_REPO_CANDIDATES:
    if candidate and Path(candidate).exists():
        DEFAULT_HRNET_REPO = Path(candidate)
        break

if DEFAULT_HRNET_REPO and DEFAULT_HRNET_REPO.exists():
    if str(DEFAULT_HRNET_REPO) not in sys.path:
        sys.path.append(str(DEFAULT_HRNET_REPO))
    try:
        from HRNET import HRNET, PersonDetector  # type: ignore
        from HRNET.utils import ModelType, filter_person_detections  # type: ignore
        HRNET_AVAILABLE = True
        logger = logging.getLogger(__name__)
        logger.info(f"✅ HRNET module loaded from {DEFAULT_HRNET_REPO}")
    except ImportError as e:
        logger = logging.getLogger(__name__)
        logger.warning(f"⚠️ HRNET module not found: {e}")
        HRNET_AVAILABLE = False
else:
    logger = logging.getLogger(__name__)
    if DEFAULT_HRNET_REPO:
        logger.warning(f"⚠️ HRNET repository not found at {DEFAULT_HRNET_REPO}")
    else:
        logger.warning("⚠️ HRNET_REPO_PATH not set - HRNet will not be available")
    HRNET_AVAILABLE = False

logger = logging.getLogger(__name__)


class FaceDetectionTool:
    """Face detection using OpenCV DNN (HRNet focus, face detection optional)"""
    
    def __init__(self):
        self.opencv_dnn_net = None
        self._initialize_opencv_dnn()
    
    def _initialize_opencv_dnn(self):
        """Initialize OpenCV DNN face detection"""
        try:
            # Try to load OpenCV DNN face detector
            # This requires model files (opencv_face_detector.pbtxt and .pb)
            # For now, we'll mark it as optional
            self.opencv_dnn_net = None  # Would load model here
            logger.info("ℹ️  OpenCV DNN face detection available (model files required)")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize OpenCV DNN: {e}")
    
    def detect_faces(self, image: np.ndarray, method: str = "opencv_dnn") -> List[Dict[str, Any]]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
            method: Detection method ("opencv_dnn")
        
        Returns:
            List of face detections with bounding boxes
        """
        if method == "opencv_dnn" and self.opencv_dnn_net:
            return self._detect_opencv_dnn(image)
        else:
            # Face detection is optional for gymnastics analytics
            logger.debug("Face detection not available - skipping")
            return []
    
    def _detect_opencv_dnn(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect faces using OpenCV DNN (if model available)"""
        # Implementation would go here if model files are available
        return []


class WeightDetectionTool:
    """Weight/body shape estimation from pose"""
    
    def __init__(self):
        self.pose_based_estimator = None
        self._initialize_pose_based()
    
    def _initialize_pose_based(self):
        """Initialize pose-based weight estimation"""
        # This would use pose keypoints to estimate body dimensions
        # and infer weight based on body proportions
        logger.info("✅ Initialized pose-based weight estimation")
    
    def estimate_weight(
        self,
        keypoints: Dict[str, Any],
        image_shape: Tuple[int, int],
        method: str = "pose_based"
    ) -> Dict[str, Any]:
        """
        Estimate weight from pose keypoints.
        
        Args:
            keypoints: Pose keypoints dictionary
            image_shape: (height, width) of image
            method: Estimation method
        
        Returns:
            Dictionary with weight estimate and confidence
        """
        if method == "pose_based":
            return self._estimate_from_pose(keypoints, image_shape)
        else:
            return {"weight_kg": None, "confidence": 0.0, "method": method}
    
    def _estimate_from_pose(
        self,
        keypoints: Dict[str, Any],
        image_shape: Tuple[int, int]
    ) -> Dict[str, Any]:
        """Estimate weight from pose keypoints using body proportions"""
        try:
            # Extract key body measurements from keypoints
            # This is a simplified estimation - real implementation would use
            # more sophisticated body shape models
            
            if "keypoints" not in keypoints and "persons" not in keypoints:
                return {"weight_kg": None, "confidence": 0.0, "error": "No keypoints available"}
            
            # Get keypoints in a usable format
            kpts = self._extract_keypoints(keypoints)
            if not kpts:
                return {"weight_kg": None, "confidence": 0.0, "error": "Could not extract keypoints"}
            
            # Estimate body dimensions from keypoints
            # Height estimation from head to ankle
            height_pixels = self._estimate_height(kpts, image_shape)
            
            # Shoulder width estimation
            shoulder_width_pixels = self._estimate_shoulder_width(kpts)
            
            # Hip width estimation
            hip_width_pixels = self._estimate_hip_width(kpts)
            
            # Simple weight estimation based on body proportions
            # This is a placeholder - real implementation would use
            # learned models or biomechanical formulas
            # For now, return relative measurements
            return {
                "weight_kg": None,  # Would need calibration for actual weight
                "relative_height": height_pixels / image_shape[0],
                "shoulder_width_pixels": shoulder_width_pixels,
                "hip_width_pixels": hip_width_pixels,
                "body_proportions": {
                    "shoulder_to_hip_ratio": shoulder_width_pixels / (hip_width_pixels + 1e-6),
                    "height_to_width_ratio": height_pixels / (shoulder_width_pixels + 1e-6)
                },
                "confidence": 0.7,  # Medium confidence for pose-based estimation
                "method": "pose_based"
            }
        except Exception as e:
            logger.error(f"❌ Error in weight estimation: {e}")
            return {"weight_kg": None, "confidence": 0.0, "error": str(e)}
    
    def _extract_keypoints(self, keypoints: Dict[str, Any]) -> Optional[Dict[str, Tuple[float, float]]]:
        """Extract keypoints in standardized format"""
        kpts = {}
        
        # Try different keypoint formats
        if "keypoints" in keypoints and isinstance(keypoints["keypoints"], dict):
            kpts = keypoints["keypoints"]
        elif "persons" in keypoints and keypoints["persons"]:
            # YOLO format
            person = keypoints["persons"][0]
            if "keypoints" in person:
                # Convert YOLO keypoints to dict format
                # This would depend on the actual format
                pass
        
        return kpts if kpts else None
    
    def _estimate_height(self, kpts: Dict, image_shape: Tuple[int, int]) -> float:
        """Estimate height in pixels"""
        # Get head (nose) and ankle positions
        head_y = None
        ankle_y = None
        
        for key in ["nose", "head", "top_head"]:
            if key in kpts:
                head_y = kpts[key][1]
                break
        
        for key in ["left_ankle", "right_ankle", "ankle"]:
            if key in kpts:
                ankle_y = max(ankle_y or 0, kpts[key][1])
        
        if head_y and ankle_y:
            return abs(ankle_y - head_y)
        return image_shape[0] * 0.8  # Fallback estimate
    
    def _estimate_shoulder_width(self, kpts: Dict) -> float:
        """Estimate shoulder width in pixels"""
        left_shoulder = kpts.get("left_shoulder") or kpts.get("shoulder_left")
        right_shoulder = kpts.get("right_shoulder") or kpts.get("shoulder_right")
        
        if left_shoulder and right_shoulder:
            return np.sqrt(
                (left_shoulder[0] - right_shoulder[0])**2 +
                (left_shoulder[1] - right_shoulder[1])**2
            )
        return 0.0
    
    def _estimate_hip_width(self, kpts: Dict) -> float:
        """Estimate hip width in pixels"""
        left_hip = kpts.get("left_hip") or kpts.get("hip_left")
        right_hip = kpts.get("right_hip") or kpts.get("hip_right")
        
        if left_hip and right_hip:
            return np.sqrt(
                (left_hip[0] - right_hip[0])**2 +
                (left_hip[1] - right_hip[1])**2
            )
        return 0.0


class PoseEstimationTool:
    """Pose estimation using HRNet (primary) and YOLO (fallback)"""
    
    def __init__(self):
        self.yolo_model = None
        self.hrnet_model = None
        self.person_detector = None
        self.hrnet_model_path = None
        self.person_detector_path = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize pose estimation models - HRNet primary, YOLO fallback"""
        # HRNet (primary method, similar to hrnet-metrics-server)
        if HRNET_AVAILABLE and HRNET is not None:
            try:
                # Find model paths (similar to hrnet-metrics-server)
                script_dir = Path(__file__).parent.resolve()
                model_dirs = [
                    script_dir / "models",
                    Path("/app/models"),
                    Path(os.getenv("HRNET_MODEL_DIR", "/app/models")),
                    DEFAULT_HRNET_REPO / "models" if DEFAULT_HRNET_REPO else None,
                ]
                
                hrnet_model_path = None
                person_detector_path = None
                
                for model_dir in model_dirs:
                    if model_dir and model_dir.exists():
                        hrnet_candidate = model_dir / "hrnet_coco_w32_256x192.onnx"
                        yolo_candidate = model_dir / "yolov5s6.onnx"
                        if hrnet_candidate.exists():
                            hrnet_model_path = hrnet_candidate
                        if yolo_candidate.exists():
                            person_detector_path = yolo_candidate
                
                # Allow override via environment variables
                hrnet_model_path = Path(
                    os.getenv("HRNET_MODEL_PATH", str(hrnet_model_path) if hrnet_model_path else "")
                ) if os.getenv("HRNET_MODEL_PATH") else hrnet_model_path
                
                person_detector_path = Path(
                    os.getenv("PERSON_DETECTOR_PATH", str(person_detector_path) if person_detector_path else "")
                ) if os.getenv("PERSON_DETECTOR_PATH") else person_detector_path
                
                if hrnet_model_path and hrnet_model_path.exists():
                    self.hrnet_model_path = hrnet_model_path
                    logger.info(f"✅ Found HRNet model: {hrnet_model_path}")
                    
                    # Initialize HRNet model (lazy load on first use)
                    # Similar to hrnet-metrics-server pattern
                    self.hrnet_model = None  # Will be loaded on first use
                    
                    if person_detector_path and person_detector_path.exists():
                        self.person_detector_path = person_detector_path
                        logger.info(f"✅ Found PersonDetector model: {person_detector_path}")
                    else:
                        logger.warning(f"⚠️  PersonDetector model not found, will use full image for HRNet")
                else:
                    logger.warning(f"⚠️  HRNet model not found. Expected at: {hrnet_model_path}")
                    logger.warning("   Set HRNET_MODEL_PATH environment variable or place model in models/")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize HRNet: {e}")
                self.hrnet_model = None
        else:
            logger.info("ℹ️  HRNet not available - will use YOLO fallback")
        
        # YOLO pose (fallback if HRNet not available)
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n-pose.pt")
            logger.info("✅ Initialized YOLO pose model (fallback)")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize YOLO: {e}")
    
    def _load_hrnet_if_needed(self):
        """Lazy-load HRNet model (similar to hrnet-metrics-server)"""
        if self.hrnet_model is None and HRNET_AVAILABLE and self.hrnet_model_path:
            try:
                logger.info(f"Loading HRNet model from {self.hrnet_model_path}")
                self.hrnet_model = HRNET(
                    str(self.hrnet_model_path),
                    ModelType.COCO,
                    conf_thres=float(os.getenv("HRNET_CONF_THRES", "0.25")),
                    search_region_ratio=float(os.getenv("HRNET_SEARCH_REGION_RATIO", "0.05")),
                )
                
                # Load person detector if available
                if self.person_detector_path and self.person_detector_path.exists():
                    self.person_detector = PersonDetector(
                        str(self.person_detector_path),
                        conf_thres=float(os.getenv("PERSON_DETECTOR_CONF", "0.5")),
                    )
                    logger.info("✅ PersonDetector loaded for HRNet")
                
                logger.info("✅ HRNet model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load HRNet model: {e}")
                self.hrnet_model = None
    
    def estimate_pose(
        self,
        image: np.ndarray,
        method: str = "hrnet"
    ) -> Dict[str, Any]:
        """
        Estimate pose from image.
        
        Args:
            image: Input image (BGR format)
            method: Estimation method ("hrnet" preferred, "yolo" fallback)
        
        Returns:
            Pose data with keypoints
        """
        # Try HRNet first (preferred for detailed analysis)
        if method == "hrnet" or method == "auto":
            if HRNET_AVAILABLE and self.hrnet_model_path:
                self._load_hrnet_if_needed()
                if self.hrnet_model:
                    return self._estimate_hrnet(image)
        
        # Fallback to YOLO
        if method == "yolo" or (method == "auto" and self.yolo_model):
            if self.yolo_model:
                return self._estimate_yolo(image)
        
        # No model available
        return {"keypoints": {}, "error": "No pose model available"}
    
    def _estimate_yolo(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using YOLO"""
        try:
            results = self.yolo_model(image, verbose=False)
            
            if results and len(results) > 0:
                result = results[0]
                keypoints = result.keypoints
                
                if keypoints is not None and len(keypoints.data) > 0:
                    # Extract keypoints
                    kpts_data = keypoints.data[0].cpu().numpy()
                    
                    # Convert to dictionary format
                    keypoint_names = [
                        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                        "left_wrist", "right_wrist", "left_hip", "right_hip",
                        "left_knee", "right_knee", "left_ankle", "right_ankle"
                    ]
                    
                    keypoints_dict = {}
                    for i, name in enumerate(keypoint_names):
                        if i < len(kpts_data):
                            keypoints_dict[name] = [float(kpts_data[i][0]), float(kpts_data[i][1])]
                    
                    return {
                        "keypoints": keypoints_dict,
                        "confidence": float(keypoints.conf[0]) if hasattr(keypoints, 'conf') else 0.8,
                        "method": "yolo"
                    }
            
            return {"keypoints": {}, "error": "No pose detected"}
        except Exception as e:
            logger.error(f"❌ Error in YOLO pose estimation: {e}")
            return {"keypoints": {}, "error": str(e)}
    
    
    def _estimate_hrnet(self, image: np.ndarray) -> Dict[str, Any]:
        """Estimate pose using HRNet (similar to hrnet-metrics-server)"""
        try:
            if not self.hrnet_model:
                return {"keypoints": {}, "error": "HRNet model not loaded"}
            
            # Person detection (if detector available) - REQUIRED before inference
            person_detections = None
            if self.person_detector:
                try:
                    detections = self.person_detector(image)
                    has_person, person_detections = filter_person_detections(detections)
                    if not has_person:
                        # No person detected - skip inference entirely
                        return {"keypoints": {}, "error": "No person detected in frame"}
                except Exception as e:
                    logger.warning(f"⚠️  Person detector error: {e}")
                    # If person detection fails, don't run inference
                    return {"keypoints": {}, "error": f"Person detection failed: {e}"}
            else:
                # No person detector available - can't verify person presence
                # Still run inference but this is not ideal
                logger.debug("No person detector available - running inference without person check")
            
            # HRNet pose inference (similar to hrnet-metrics-server)
            # Only run if person_detections is available (person was detected)
            if person_detections:
                total_heatmap, poses = self.hrnet_model(image, person_detections)
            else:
                # No person detections - don't run inference
                return {"keypoints": {}, "error": "No person detected - skipping inference"}
            
            if poses is None:
                return {"keypoints": {}, "error": "HRNet did not return any poses"}
            
            # Convert HRNet pose format to our keypoints dictionary
            # HRNet returns poses as list or single pose
            if isinstance(poses, list) and len(poses) > 0:
                pose_keypoints = poses[0]  # Use first person
            else:
                pose_keypoints = poses
            
            # HRNet keypoints format: COCO format (17 keypoints)
            # [x, y, confidence] for each keypoint
            coco_keypoint_names = [
                "nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"
            ]
            
            keypoints_dict = {}
            if hasattr(pose_keypoints, '__iter__'):
                for i, kp in enumerate(pose_keypoints):
                    if i < len(coco_keypoint_names):
                        if isinstance(kp, (list, tuple, np.ndarray)) and len(kp) >= 2:
                            keypoints_dict[coco_keypoint_names[i]] = [
                                float(kp[0]),
                                float(kp[1])
                            ]
            
            return {
                "keypoints": keypoints_dict,
                "confidence": 0.8,  # HRNet provides per-keypoint confidence
                "method": "hrnet",
                "heatmap": total_heatmap if total_heatmap is not None else None
            }
        except Exception as e:
            logger.error(f"❌ Error in HRNet pose estimation: {e}", exc_info=True)
            return {"keypoints": {}, "error": str(e)}


class PersonDetectionTool:
    """Person detection using HRNet PersonDetector (primary) or YOLO (fallback)"""
    
    def __init__(self):
        self.yolo_model = None
        self.person_detector = None
        self.person_detector_path = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize person detection models"""
        # HRNet PersonDetector (primary, same as hrnet-metrics-server)
        if HRNET_AVAILABLE and PersonDetector is not None:
            try:
                # Find model path (same logic as PoseEstimationTool)
                script_dir = Path(__file__).parent.resolve()
                model_dirs = [
                    script_dir / "models",
                    Path("/app/models"),
                    Path(os.getenv("HRNET_MODEL_DIR", "/app/models")),
                    DEFAULT_HRNET_REPO / "models" if DEFAULT_HRNET_REPO else None,
                ]
                
                for model_dir in model_dirs:
                    if model_dir and model_dir.exists():
                        yolo_candidate = model_dir / "yolov5s6.onnx"
                        if yolo_candidate.exists():
                            self.person_detector_path = yolo_candidate
                            break
                
                # Allow override via environment variable
                if os.getenv("PERSON_DETECTOR_PATH"):
                    self.person_detector_path = Path(os.getenv("PERSON_DETECTOR_PATH"))
                
                if self.person_detector_path and self.person_detector_path.exists():
                    logger.info(f"✅ Found PersonDetector model: {self.person_detector_path}")
                    # Lazy load on first use
                    self.person_detector = None
                else:
                    logger.warning(f"⚠️  PersonDetector model not found. Expected: {self.person_detector_path}")
            except Exception as e:
                logger.warning(f"⚠️  Could not initialize PersonDetector: {e}")
        
        # YOLO for person detection (fallback)
        try:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolo11n.pt")  # General YOLO model
            logger.info("✅ Initialized YOLO person detector (fallback)")
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize YOLO person detector: {e}")
    
    def _load_person_detector_if_needed(self):
        """Lazy-load PersonDetector"""
        if self.person_detector is None and HRNET_AVAILABLE and self.person_detector_path:
            try:
                self.person_detector = PersonDetector(
                    str(self.person_detector_path),
                    conf_thres=float(os.getenv("PERSON_DETECTOR_CONF", "0.5")),
                )
                logger.info("✅ PersonDetector loaded")
            except Exception as e:
                logger.error(f"❌ Failed to load PersonDetector: {e}")
                self.person_detector = None
    
    def detect_persons(
        self,
        image: np.ndarray,
        method: str = "hrnet"
    ) -> List[Dict[str, Any]]:
        """
        Detect persons in image.
        
        Args:
            image: Input image (BGR format)
            method: Detection method ("hrnet" preferred, "yolo" fallback)
        
        Returns:
            List of person detections with bounding boxes
        """
        # Try HRNet PersonDetector first
        if method == "hrnet" or method == "auto":
            if HRNET_AVAILABLE and self.person_detector_path:
                self._load_person_detector_if_needed()
                if self.person_detector:
                    return self._detect_hrnet(image)
        
        # Fallback to YOLO
        if method == "yolo" or (method == "auto" and self.yolo_model):
            if self.yolo_model:
                return self._detect_yolo(image)
        
        return []
    
    def _detect_hrnet(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons using HRNet PersonDetector"""
        try:
            if not self.person_detector:
                return []
            
            detections = self.person_detector(image)
            has_person, person_detections = filter_person_detections(detections)
            
            if not has_person:
                return []
            
            det_boxes, det_scores, det_class_ids = person_detections
            
            detections_list = []
            for i in range(len(det_boxes)):
                box = det_boxes[i]
                score = float(det_scores[i])
                
                detections_list.append({
                    "bbox": [int(box[0]), int(box[1]), int(box[2] - box[0]), int(box[3] - box[1])],
                    "confidence": score,
                    "class": "person",
                    "method": "hrnet"
                })
            
            return detections_list
        except Exception as e:
            logger.error(f"❌ Error in HRNet person detection: {e}")
            return []
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect persons using YOLO"""
        try:
            results = self.yolo_model(image, classes=[0], verbose=False)  # Class 0 is person
            
            detections = []
            if results and len(results) > 0:
                result = results[0]
                boxes = result.boxes
                
                if boxes is not None:
                    for box in boxes:
                        # Check if it's a person (class 0)
                        if int(box.cls[0]) == 0:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0])
                            
                            detections.append({
                                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                                "confidence": conf,
                                "class": "person"
                            })
            
            return detections
        except Exception as e:
            logger.error(f"❌ Error in YOLO person detection: {e}")
            return []
    






















