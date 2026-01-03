#!/usr/bin/env python3
"""
Video Clip Extractor
Extracts video clips for high-risk ACL frames and landing phases.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class VideoClipExtractor:
    """
    Extracts video clips from frames for high-risk ACL moments.
    """
    
    def __init__(self):
        self.stored_frames: List[Dict[str, Any]] = []  # Store frames for clip extraction
    
    def store_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Store a frame for potential clip extraction.
        
        Args:
            frame: Video frame (BGR format)
            frame_number: Frame number
            timestamp: Frame timestamp
            metrics: Optional metrics for this frame
        """
        self.stored_frames.append({
            "frame": frame.copy(),
            "frame_number": frame_number,
            "timestamp": timestamp,
            "metrics": metrics or {}
        })
    
    def extract_clips_for_acl_risk(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]],
        output_dir: Path,
        base_name: str,
        fps: float = 30.0,
        clip_duration_seconds: float = 2.0,
        frames_before: int = 30,
        frames_after: int = 30
    ) -> List[Dict[str, Any]]:
        """
        Extract video clips for each high-risk ACL frame.
        
        Args:
            acl_flagged_timesteps: List of flagged timesteps with frame numbers
            output_dir: Directory to save clips
            base_name: Base filename for clips
            fps: Video frame rate
            clip_duration_seconds: Duration of clip in seconds
            frames_before: Number of frames before flagged frame
            frames_after: Number of frames after flagged frame
            
        Returns:
            List of clip information dictionaries
        """
        clips = []
        
        if not self.stored_frames:
            logger.warning("⚠️  No frames stored for clip extraction")
            return clips
        
        # Sort frames by frame number
        sorted_frames = sorted(self.stored_frames, key=lambda x: x["frame_number"])
        
        for i, flagged_entry in enumerate(acl_flagged_timesteps):
            frame_number = flagged_entry.get("frame_number")
            timestamp = flagged_entry.get("timestamp", 0.0)
            risk_score = flagged_entry.get("risk_score", 0.0)
            risk_level = flagged_entry.get("risk_level", "HIGH")
            
            if frame_number is None:
                logger.warning(f"⚠️  Flagged entry {i} missing frame_number, skipping")
                continue
            
            # Find the flagged frame
            flagged_frame_idx = None
            for idx, frame_data in enumerate(sorted_frames):
                if frame_data["frame_number"] == frame_number:
                    flagged_frame_idx = idx
                    break
            
            if flagged_frame_idx is None:
                logger.warning(f"⚠️  Frame {frame_number} not found in stored frames")
                continue
            
            # Extract clip (frames_before to frames_after around flagged frame)
            start_idx = max(0, flagged_frame_idx - frames_before)
            end_idx = min(len(sorted_frames), flagged_frame_idx + frames_after + 1)
            
            clip_frames = sorted_frames[start_idx:end_idx]
            
            if not clip_frames:
                logger.warning(f"⚠️  No frames for clip around frame {frame_number}")
                continue
            
            # Get video dimensions from first frame
            first_frame = clip_frames[0]["frame"]
            if first_frame is None or not isinstance(first_frame, np.ndarray):
                continue
            
            height, width = first_frame.shape[:2]
            
            # Create clip filename
            clip_filename = f"{base_name}_acl_risk_{i+1}_frame{frame_number}_score{risk_score:.2f}.mp4"
            clip_path = output_dir / clip_filename
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Write frames with annotations
            for frame_data in clip_frames:
                frame = frame_data["frame"].copy()
                
                # Add text overlay indicating this is a high-risk frame
                if frame_data["frame_number"] == frame_number:
                    # Highlight the flagged frame
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)  # Red border
                    cv2.putText(
                        frame,
                        f"ACL HIGH RISK - Frame {frame_number}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Risk Score: {risk_score:.2f} | Level: {risk_level}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    
                    # Add risk factors
                    risk_factors = flagged_entry.get("primary_risk_factors", [])
                    y_offset = 110
                    for factor in risk_factors[:3]:  # Show first 3 factors
                        cv2.putText(
                            frame,
                            f"- {factor}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )
                        y_offset += 30
                
                # Add frame number and timestamp
                cv2.putText(
                    frame,
                    f"Frame: {frame_data['frame_number']} | Time: {frame_data['timestamp']:.2f}s",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                video_writer.write(frame)
            
            video_writer.release()
            
            clip_info = {
                "clip_path": str(clip_path),
                "flagged_frame_number": frame_number,
                "flagged_timestamp": timestamp,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "clip_start_frame": clip_frames[0]["frame_number"],
                "clip_end_frame": clip_frames[-1]["frame_number"],
                "clip_duration": (clip_frames[-1]["timestamp"] - clip_frames[0]["timestamp"]) if len(clip_frames) > 1 else 0.0,
                "frames_in_clip": len(clip_frames)
            }
            
            clips.append(clip_info)
            logger.info(f"✅ Extracted ACL risk clip {i+1}: {clip_path} (Frame {frame_number}, Score: {risk_score:.2f})")
        
        return clips
    
    def extract_clips_from_video_file(
        self,
        video_path: str,
        acl_flagged_timesteps: List[Dict[str, Any]],
        output_dir: Path,
        base_name: str,
        clip_duration_seconds: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Extract clips directly from video file for flagged timesteps.
        Use this when frames weren't stored in memory.
        
        Args:
            video_path: Path to source video file
            acl_flagged_timesteps: List of flagged timesteps
            output_dir: Directory to save clips
            base_name: Base filename for clips
            clip_duration_seconds: Duration of clip in seconds (before + after flagged time)
            
        Returns:
            List of clip information dictionaries
        """
        clips = []
        
        if not Path(video_path).exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return clips
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"❌ Could not open video: {video_path}")
            return clips
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        logger.info(f"📹 Video info: {total_frames} frames, {fps} fps, {width}x{height}")
        
        for i, flagged_entry in enumerate(acl_flagged_timesteps):
            frame_number = flagged_entry.get("frame_number")
            timestamp = flagged_entry.get("timestamp", 0.0)
            risk_score = flagged_entry.get("risk_score", 0.0)
            risk_level = flagged_entry.get("risk_level", "HIGH")
            
            if frame_number is None:
                # Try to estimate frame number from timestamp
                if timestamp > 0 and fps > 0:
                    frame_number = int(timestamp * fps)
                else:
                    logger.warning(f"⚠️  Flagged entry {i} missing frame_number and timestamp, skipping")
                    continue
            
            # Calculate frame range for clip
            frames_before = int(clip_duration_seconds * fps / 2)
            frames_after = int(clip_duration_seconds * fps / 2)
            
            start_frame = max(0, frame_number - frames_before)
            end_frame = min(total_frames - 1, frame_number + frames_after)
            
            # Create clip filename
            clip_filename = f"{base_name}_acl_risk_{i+1}_frame{frame_number}_score{risk_score:.2f}.mp4"
            clip_path = output_dir / clip_filename
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(
                str(clip_path),
                fourcc,
                fps,
                (width, height)
            )
            
            # Seek to start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            # Read and write frames
            current_frame_num = start_frame
            while current_frame_num <= end_frame:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Add annotations for flagged frame
                if current_frame_num == frame_number:
                    # Highlight the flagged frame
                    cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), 5)  # Red border
                    cv2.putText(
                        frame,
                        f"ACL HIGH RISK - Frame {frame_number}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        2
                    )
                    cv2.putText(
                        frame,
                        f"Risk Score: {risk_score:.2f} | Level: {risk_level}",
                        (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )
                    
                    # Add risk factors
                    risk_factors = flagged_entry.get("primary_risk_factors", [])
                    y_offset = 110
                    for factor in risk_factors[:3]:  # Show first 3 factors
                        cv2.putText(
                            frame,
                            f"- {factor}",
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 165, 255),
                            2
                        )
                        y_offset += 30
                
                # Add frame number and timestamp
                current_timestamp = current_frame_num / fps if fps > 0 else 0.0
                cv2.putText(
                    frame,
                    f"Frame: {current_frame_num} | Time: {current_timestamp:.2f}s",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                
                video_writer.write(frame)
                current_frame_num += 1
            
            video_writer.release()
            
            clip_info = {
                "clip_path": str(clip_path),
                "flagged_frame_number": frame_number,
                "flagged_timestamp": timestamp,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "clip_start_frame": start_frame,
                "clip_end_frame": end_frame,
                "clip_duration": (end_frame - start_frame) / fps if fps > 0 else 0.0,
                "frames_in_clip": end_frame - start_frame + 1
            }
            
            clips.append(clip_info)
            logger.info(f"✅ Extracted ACL risk clip {i+1}: {clip_path} (Frame {frame_number}, Score: {risk_score:.2f})")
        
        cap.release()
        return clips
    
    def save_all_frames_as_video(
        self,
        output_path: Path,
        fps: float = 30.0,
        add_timestamps: bool = True,
        add_frame_numbers: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Save all stored frames as a single MP4 video.
        
        Args:
            output_path: Path to save the video file
            fps: Video frame rate
            add_timestamps: Whether to add timestamp overlay on frames
            add_frame_numbers: Whether to add frame number overlay on frames
            
        Returns:
            Dictionary with video info, or None if no frames stored
        """
        if not self.stored_frames:
            logger.warning("⚠️  No frames stored - cannot save video")
            return None
        
        # Sort frames by frame number to ensure correct order
        sorted_frames = sorted(self.stored_frames, key=lambda x: x["frame_number"])
        
        if not sorted_frames:
            logger.warning("⚠️  No valid frames to save")
            return None
        
        # Get video dimensions from first frame
        first_frame = sorted_frames[0]["frame"]
        if first_frame is None or not isinstance(first_frame, np.ndarray):
            logger.warning("⚠️  Invalid frame format")
            return None
        
        height, width = first_frame.shape[:2]
        
        # Create video writer
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height)
        )
        
        if not video_writer.isOpened():
            logger.error(f"❌ Could not open video writer for {output_path}")
            return None
        
        # Write all frames
        frame_count = 0
        start_timestamp = sorted_frames[0]["timestamp"] if sorted_frames else 0.0
        
        for frame_data in sorted_frames:
            frame = frame_data.get("frame")
            if frame is None or not isinstance(frame, np.ndarray):
                continue
            
            # Create a copy to avoid modifying original
            output_frame = frame.copy()
            
            # Add overlays if requested
            if add_timestamps or add_frame_numbers:
                timestamp = frame_data.get("timestamp", 0.0)
                frame_number = frame_data.get("frame_number", 0)
                relative_time = timestamp - start_timestamp
                
                # Add timestamp overlay
                if add_timestamps:
                    cv2.putText(
                        output_frame,
                        f"Time: {relative_time:.2f}s",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
                
                # Add frame number overlay
                if add_frame_numbers:
                    cv2.putText(
                        output_frame,
                        f"Frame: {frame_number}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
            
            video_writer.write(output_frame)
            frame_count += 1
        
        video_writer.release()
        
        # Calculate video duration
        if len(sorted_frames) > 1:
            duration = sorted_frames[-1]["timestamp"] - sorted_frames[0]["timestamp"]
        else:
            duration = 1.0 / fps if fps > 0 else 0.0
        
        video_info = {
            "video_path": str(output_path),
            "frame_count": frame_count,
            "duration_seconds": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "start_frame": sorted_frames[0]["frame_number"] if sorted_frames else 0,
            "end_frame": sorted_frames[-1]["frame_number"] if sorted_frames else 0,
            "start_timestamp": sorted_frames[0]["timestamp"] if sorted_frames else 0.0,
            "end_timestamp": sorted_frames[-1]["timestamp"] if sorted_frames else 0.0
        }
        
        logger.info(f"✅ Saved captured video segment: {output_path}")
        logger.info(f"   Frames: {frame_count}, Duration: {duration:.2f}s, FPS: {fps}")
        
        return video_info
    
    def clear_stored_frames(self):
        """Clear stored frames to free memory"""
        self.stored_frames = []








