#!/usr/bin/env python3
"""
Landing Detection Tool
Detects landing phases in video frames to help identify high ACL risk moments.
Landings are critical for ACL tear risk assessment.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import deque

logger = logging.getLogger(__name__)


class LandingDetectionTool:
    """
    Detects landing phases in video frames using:
    - Vertical velocity changes (deceleration)
    - Impact force spikes
    - Knee flexion changes
    - Height off floor (approaching ground)
    - Ankle position changes
    """
    
    def __init__(self):
        self.landing_state = {
            "in_landing_phase": False,
            "landing_start_frame": None,
            "landing_start_timestamp": None,
            "landing_end_frame": None,
            "landing_end_timestamp": None,
            "peak_impact_frame": None,
            "peak_impact_timestamp": None,
            "landing_phases": [],  # List of detected landing phases
            "previous_height": None,
            "previous_vertical_velocity": None,
            "height_history": deque(maxlen=10),  # Last 10 frames of height
            "velocity_history": deque(maxlen=10)  # Last 10 frames of velocity
        }
    
    def detect_landing_phase(
        self,
        keypoints: Dict[str, Any],
        metrics: Dict[str, float],
        frame_number: Optional[int] = None,
        timestamp: Optional[float] = None,
        previous_keypoints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Detect if current frame is in a landing phase.
        
        Landing indicators:
        1. Sudden deceleration (negative vertical acceleration)
        2. High impact force (>1000N)
        3. Decreasing height (approaching ground)
        4. Knee flexion increase (absorbing impact)
        5. Ankle vertical velocity change (contact with ground)
        
        Args:
            keypoints: Current frame keypoints
            metrics: Calculated metrics for current frame
            frame_number: Current frame number
            timestamp: Current timestamp
            previous_keypoints: Previous frame keypoints
            
        Returns:
            Dictionary with landing detection results
        """
        detection = {
            "in_landing_phase": False,
            "landing_confidence": 0.0,
            "landing_indicators": [],
            "phase_type": None,  # "pre_landing", "impact", "post_impact"
            "frame_number": frame_number,
            "timestamp": timestamp
        }
        
        try:
            # Get relevant metrics
            height = metrics.get("height_off_floor_meters", 0.0)
            impact_force = metrics.get("impact_force_N", 0.0)
            vertical_velocity = metrics.get("com_vertical_velocity_ms", 0.0)
            vertical_accel = metrics.get("com_vertical_acceleration_ms2", 0.0)
            knee_flexion = metrics.get("landing_knee_bend_min", None)
            if knee_flexion is None:
                knee_flexion = min(
                    metrics.get("landing_knee_bend_left", 180.0) or 180.0,
                    metrics.get("landing_knee_bend_right", 180.0) or 180.0
                )
            
            # Track height and velocity history
            self.landing_state["height_history"].append(height)
            self.landing_state["velocity_history"].append(vertical_velocity)
            
            # Landing indicators (each contributes to confidence)
            indicators = []
            confidence = 0.0
            
            # Indicator 1: High impact force (strongest indicator)
            if impact_force > 1000.0:
                indicators.append(f"High impact force ({impact_force:.0f}N)")
                confidence += 0.4
                if impact_force > 2000.0:
                    confidence += 0.2  # Very high impact
            
            # Indicator 2: Sudden deceleration (negative vertical acceleration)
            if vertical_accel < -20.0:  # Decelerating downward
                indicators.append(f"Sudden deceleration ({vertical_accel:.1f}m/s²)")
                confidence += 0.3
            
            # Indicator 3: Decreasing height (approaching ground)
            if len(self.landing_state["height_history"]) >= 3:
                recent_heights = list(self.landing_state["height_history"])[-3:]
                if len(recent_heights) == 3:
                    height_trend = recent_heights[-1] - recent_heights[0]
                    if height_trend < -0.05:  # Decreasing by at least 5cm
                        indicators.append(f"Decreasing height ({height_trend*100:.1f}cm)")
                        confidence += 0.2
            
            # Indicator 4: Negative vertical velocity (moving downward)
            if vertical_velocity < -0.5:  # Moving downward at >0.5 m/s
                indicators.append(f"Downward velocity ({vertical_velocity:.2f}m/s)")
                confidence += 0.15
            
            # Indicator 5: Knee flexion increase (absorbing impact)
            if knee_flexion < 60.0:  # Knees bent (flexed)
                indicators.append(f"Knee flexion ({knee_flexion:.1f}°)")
                confidence += 0.1
            
            # Indicator 6: Peak impact detection
            if metrics.get("peak_impact_detected", 0.0) == 1.0:
                indicators.append("Peak impact detected")
                confidence += 0.25
            
            # Determine if in landing phase
            # Threshold: 0.5 confidence = likely landing
            # Clamp confidence to 0-1 range
            confidence = min(confidence, 1.0)
            detection["landing_confidence"] = confidence
            detection["landing_indicators"] = indicators
            
            if confidence >= 0.5:
                detection["in_landing_phase"] = True
                
                # Determine phase type
                if impact_force > 2000.0 or metrics.get("peak_impact_detected", 0.0) == 1.0:
                    detection["phase_type"] = "impact"
                elif vertical_velocity < -1.0 and height > 0.1:
                    detection["phase_type"] = "pre_landing"
                else:
                    detection["phase_type"] = "post_impact"
                
                # Track landing phase
                if not self.landing_state["in_landing_phase"]:
                    # Landing phase just started
                    self.landing_state["in_landing_phase"] = True
                    self.landing_state["landing_start_frame"] = frame_number
                    self.landing_state["landing_start_timestamp"] = timestamp
                else:
                    # Update peak impact if this is the highest impact
                    if impact_force > metrics.get("peak_impact_force_N", 0.0):
                        self.landing_state["peak_impact_frame"] = frame_number
                        self.landing_state["peak_impact_timestamp"] = timestamp
            else:
                # Not in landing phase
                detection["in_landing_phase"] = False
                detection["phase_type"] = None
                
                if self.landing_state["in_landing_phase"]:
                    # Landing phase just ended
                    self.landing_state["in_landing_phase"] = False
                    self.landing_state["landing_end_frame"] = frame_number
                    self.landing_state["landing_end_timestamp"] = timestamp
                    
                    # Save completed landing phase
                    landing_phase = {
                        "start_frame": self.landing_state["landing_start_frame"],
                        "start_timestamp": self.landing_state["landing_start_timestamp"],
                        "end_frame": self.landing_state["landing_end_frame"],
                        "end_timestamp": self.landing_state["landing_end_timestamp"],
                        "peak_impact_frame": self.landing_state["peak_impact_frame"],
                        "peak_impact_timestamp": self.landing_state["peak_impact_timestamp"],
                        "duration": (
                            self.landing_state["landing_end_timestamp"] - 
                            self.landing_state["landing_start_timestamp"]
                        ) if (self.landing_state["landing_end_timestamp"] and 
                               self.landing_state["landing_start_timestamp"]) else None
                    }
                    self.landing_state["landing_phases"].append(landing_phase)
                    
                    # Reset for next landing
                    self.landing_state["landing_start_frame"] = None
                    self.landing_state["landing_start_timestamp"] = None
                    self.landing_state["peak_impact_frame"] = None
                    self.landing_state["peak_impact_timestamp"] = None
            
            detection["landing_indicators"] = indicators
            detection["landing_confidence"] = confidence
            
        except Exception as e:
            logger.error(f"❌ Error in landing detection: {e}", exc_info=True)
            detection["error"] = str(e)
        
        return detection
    
    def get_landing_phases(self) -> List[Dict[str, Any]]:
        """Get all detected landing phases"""
        return self.landing_state["landing_phases"].copy()
    
    def reset(self):
        """Reset landing detection state (call at start of new video)"""
        self.landing_state = {
            "in_landing_phase": False,
            "landing_start_frame": None,
            "landing_start_timestamp": None,
            "landing_end_frame": None,
            "landing_end_timestamp": None,
            "peak_impact_frame": None,
            "peak_impact_timestamp": None,
            "landing_phases": [],
            "previous_height": None,
            "previous_vertical_velocity": None,
            "height_history": deque(maxlen=10),
            "velocity_history": deque(maxlen=10)
        }








