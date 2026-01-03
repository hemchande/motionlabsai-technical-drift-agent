#!/usr/bin/env python3
"""
Segmentation Tool using Roboflow Inference SDK
Detects athletes, beams, and person tumblings with segmentation overlays.
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Roboflow Inference SDK
try:
    from inference_sdk import InferenceHTTPClient
    ROBOFLOW_AVAILABLE = True
except ImportError:
    ROBOFLOW_AVAILABLE = False
    logger.warning("⚠️  inference_sdk not available - segmentation will be disabled")


class SegmentationTool:
    """
    Segmentation tool using Roboflow Inference SDK.
    Detects athletes, beams, and person tumblings with segmentation masks.
    """
    
    def __init__(self):
        self.client = None
        self.workspace_name = "mineralintelligence"
        self.workflow_id = "find-gymnasts-beams-and-a-person-tumblings"
        self.api_key = os.getenv("ROBOFLOW_API_KEY", "s0wETXLIwp979GIThfx2")
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize Roboflow Inference HTTP Client"""
        if not ROBOFLOW_AVAILABLE:
            logger.warning("⚠️  Roboflow Inference SDK not installed")
            logger.warning("   Install with: pip install inference-sdk")
            return
        
        try:
            self.client = InferenceHTTPClient(
                api_url="https://serverless.roboflow.com",
                api_key=self.api_key
            )
            logger.info("✅ Roboflow Inference Client initialized")
            logger.info(f"   Workspace: {self.workspace_name}")
            logger.info(f"   Workflow: {self.workflow_id}")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Roboflow client: {e}")
            self.client = None
    
    def segment_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Segment a frame to detect athletes, beams, and person tumblings.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            Dictionary with segmentation results
        """
        if not self.client:
            return {"error": "Roboflow client not initialized", "segments": []}
        
        try:
            # Convert BGR to RGB for Roboflow
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Save frame temporarily for API call
            temp_path = "/tmp/roboflow_temp.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Run workflow
            result = self.client.run_workflow(
                workspace_name=self.workspace_name,
                workflow_id=self.workflow_id,
                images={
                    "image": temp_path
                },
                use_cache=True  # Cache workflow definition for 15 minutes
            )
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Parse results
            segments = self._parse_workflow_result(result)
            
            return {
                "segments": segments,
                "count": len(segments),
                "raw_result": result
            }
            
        except Exception as e:
            logger.error(f"❌ Segmentation error: {e}")
            return {"error": str(e), "segments": []}
    
    def _parse_workflow_result(self, result: Any) -> List[Dict[str, Any]]:
        """Parse Roboflow workflow result into segments"""
        segments = []
        
        try:
            # Roboflow workflow can return either a dict or a list
            # Handle list first (common for workflow results)
            if isinstance(result, list):
                # Workflow returns list of results (one per workflow step/output)
                for item in result:
                    if isinstance(item, dict):
                        # Check for nested predictions structure: item['predictions']['predictions']
                        # This is the actual Roboflow workflow format!
                        predictions_data = item.get("predictions", {})
                        predictions = []
                        
                        if isinstance(predictions_data, dict):
                            # Nested structure: predictions.predictions (this is the actual format!)
                            predictions = predictions_data.get("predictions", [])
                            if not predictions:
                                predictions = predictions_data.get("detections", [])
                        elif isinstance(predictions_data, list):
                            # Direct predictions list
                            predictions = predictions_data
                        
                        # Also check top-level keys as fallback
                        if not predictions:
                            predictions = item.get("detections", [])
                        if not predictions:
                            predictions = item.get("results", [])
                        
                        # If item itself looks like a prediction, treat it as one
                        if not predictions and ("class" in item or "bbox" in item or "points" in item):
                            predictions = [item]
                        
                        # Process predictions from this item
                        for pred in predictions:
                            segment = self._extract_segment_from_prediction(pred)
                            if segment:
                                segments.append(segment)
                        
                        # Debug log
                        if isinstance(predictions_data, dict) and len(predictions) > 0:
                            logger.debug(f"Found {len(predictions)} predictions in nested structure")
            
            # Handle dict structure
            elif isinstance(result, dict):
                # Method 1: Direct predictions/detections
                predictions = result.get("predictions", [])
                if not predictions:
                    predictions = result.get("detections", [])
                if not predictions:
                    predictions = result.get("results", [])
                
                # Method 2: Check for workflow output keys (workflows can have multiple outputs)
                if not predictions:
                    # Workflow results might be in output keys
                    for key, value in result.items():
                        if isinstance(value, (list, dict)):
                            if isinstance(value, list):
                                predictions = value
                                break
                            elif isinstance(value, dict):
                                # Nested structure - check for predictions inside
                                nested_preds = value.get("predictions", [])
                                if nested_preds:
                                    predictions = nested_preds
                                    break
                
                # Method 3: Check for image-level results
                if not predictions and "image" in result:
                    image_result = result.get("image", {})
                    if isinstance(image_result, dict):
                        predictions = image_result.get("predictions", [])
                        if not predictions:
                            predictions = image_result.get("detections", [])
                
                # Process predictions
                for pred in predictions:
                    if isinstance(pred, dict):
                        # Extract class name (could be "class", "class_name", "label", etc.)
                        class_name = (
                            pred.get("class") or 
                            pred.get("class_name") or 
                            pred.get("label") or 
                            pred.get("name") or 
                            "unknown"
                        )
                        
                        # Extract confidence
                        confidence = (
                            pred.get("confidence") or 
                            pred.get("conf") or 
                            pred.get("score") or 
                            0.0
                        )
                        
                        # Extract bounding box (could be "bbox", "box", "bounding_box", or x/y/width/height)
                        bbox = {}
                        if "bbox" in pred:
                            bbox_data = pred["bbox"]
                            if isinstance(bbox_data, dict):
                                bbox = bbox_data
                            elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                                bbox = {"x": bbox_data[0], "y": bbox_data[1], "width": bbox_data[2], "height": bbox_data[3]}
                        elif all(k in pred for k in ["x", "y", "width", "height"]):
                            bbox = {k: pred[k] for k in ["x", "y", "width", "height"]}
                        elif all(k in pred for k in ["x_min", "y_min", "x_max", "y_max"]):
                            bbox = {
                                "x": pred["x_min"],
                                "y": pred["y_min"],
                                "width": pred["x_max"] - pred["x_min"],
                                "height": pred["y_max"] - pred["y_min"]
                            }
                        
                        # Extract segmentation points/mask
                        points = pred.get("points", [])
                        if not points:
                            points = pred.get("polygon", [])
                        if not points:
                            points = pred.get("segmentation", [])
                        
                        mask = pred.get("mask", None)
                        
                        segment = {
                            "class": class_name,
                            "confidence": float(confidence),
                            "bbox": bbox,
                            "points": points if isinstance(points, list) else [],
                            "mask": mask
                        }
                        segments.append(segment)
            
        except Exception as e:
            logger.warning(f"⚠️  Error parsing workflow result: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return segments
    
    def _extract_segment_from_prediction(self, pred: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract segment data from a single prediction dictionary"""
        if not isinstance(pred, dict):
            return None
        
        try:
            # Extract class name (could be "class", "class_name", "label", etc.)
            class_name = (
                pred.get("class") or 
                pred.get("class_name") or 
                pred.get("label") or 
                pred.get("name") or 
                "unknown"
            )
            
            # Extract confidence
            confidence = (
                pred.get("confidence") or 
                pred.get("conf") or 
                pred.get("score") or 
                0.0
            )
            
            # Extract bounding box (Roboflow format: x, y, width, height directly in pred)
            bbox = {}
            if all(k in pred for k in ["x", "y", "width", "height"]):
                # Direct format (Roboflow standard)
                bbox = {
                    "x": int(pred["x"]),
                    "y": int(pred["y"]),
                    "width": int(pred["width"]),
                    "height": int(pred["height"])
                }
            elif "bbox" in pred:
                bbox_data = pred["bbox"]
                if isinstance(bbox_data, dict):
                    bbox = bbox_data
                elif isinstance(bbox_data, (list, tuple)) and len(bbox_data) >= 4:
                    bbox = {"x": bbox_data[0], "y": bbox_data[1], "width": bbox_data[2], "height": bbox_data[3]}
            elif all(k in pred for k in ["x_min", "y_min", "x_max", "y_max"]):
                bbox = {
                    "x": int(pred["x_min"]),
                    "y": int(pred["y_min"]),
                    "width": int(pred["x_max"] - pred["x_min"]),
                    "height": int(pred["y_max"] - pred["y_min"])
                }
            elif all(k in pred for k in ["x1", "y1", "x2", "y2"]):
                bbox = {
                    "x": int(pred["x1"]),
                    "y": int(pred["y1"]),
                    "width": int(pred["x2"] - pred["x1"]),
                    "height": int(pred["y2"] - pred["y1"])
                }
            
            # Extract segmentation points/mask
            points = pred.get("points", [])
            if not points:
                points = pred.get("polygon", [])
            if not points:
                points = pred.get("segmentation", [])
            if not points and "point" in pred:
                points = [pred["point"]]
            
            mask = pred.get("mask", None)
            
            return {
                "class": class_name,
                "confidence": float(confidence),
                "bbox": bbox,
                "points": points if isinstance(points, list) else [],
                "mask": mask
            }
        except Exception as e:
            logger.debug(f"Error extracting segment from prediction: {e}")
            return None
    
    def draw_segmentation_overlay(
        self,
        frame: np.ndarray,
        segments: List[Dict[str, Any]],
        alpha: float = 0.5
    ) -> np.ndarray:
        """
        Draw segmentation overlays on frame.
        
        Args:
            frame: Input frame
            segments: List of segment dictionaries
            alpha: Transparency for overlay
        
        Returns:
            Frame with segmentation overlays
        """
        overlay = frame.copy()
        
        # Color mapping for different classes
        color_map = {
            "gymnast": (0, 255, 0),  # Green
            "athlete": (0, 255, 0),  # Green
            "person": (0, 255, 255),  # Yellow
            "beam": (255, 0, 0),  # Blue
            "tumbling": (255, 0, 255),  # Magenta
            "unknown": (128, 128, 128)  # Gray
        }
        
        for segment in segments:
            class_name = segment.get("class", "unknown").lower()
            color = color_map.get(class_name, (128, 128, 128))
            confidence = segment.get("confidence", 0.0)
            bbox = segment.get("bbox", {})
            points = segment.get("points", [])
            
            # Draw bounding box
            if bbox:
                # Handle bbox as dict or direct values
                if isinstance(bbox, dict):
                    x = int(float(bbox.get("x", 0)))
                    y = int(float(bbox.get("y", 0)))
                    w = int(float(bbox.get("width", 0)))
                    h = int(float(bbox.get("height", 0)))
                elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x, y, w, h = int(float(bbox[0])), int(float(bbox[1])), int(float(bbox[2])), int(float(bbox[3]))
                else:
                    continue  # Skip invalid bbox
                
                if w > 0 and h > 0:  # Only draw if valid dimensions
                    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                label = f"{class_name} {confidence:.2f}"
                cv2.putText(
                    overlay,
                    label,
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
            
            # Draw segmentation polygon if points available
            if points and len(points) > 2:
                try:
                    # Convert points to numpy array, handling various formats
                    if isinstance(points[0], (list, tuple)):
                        # Points are list of [x, y] pairs
                        pts = np.array([[int(float(p[0])), int(float(p[1]))] for p in points if len(p) >= 2], dtype=np.int32)
                    elif isinstance(points[0], dict):
                        # Points are list of dicts with x, y keys
                        pts = np.array([[int(float(p.get("x", 0))), int(float(p.get("y", 0)))] for p in points], dtype=np.int32)
                    else:
                        # Try to parse as flat list [x1, y1, x2, y2, ...]
                        if len(points) % 2 == 0:
                            pts = np.array([[int(float(points[i])), int(float(points[i+1]))] for i in range(0, len(points), 2)], dtype=np.int32)
                        else:
                            continue
                    
                    if len(pts) > 2:
                        cv2.fillPoly(overlay, [pts], color)
                except Exception as e:
                    logger.debug(f"Error drawing polygon: {e}")
                    continue
        
        # Blend overlay
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        return result
    
    def is_available(self) -> bool:
        """Check if segmentation is available"""
        return self.client is not None and ROBOFLOW_AVAILABLE
