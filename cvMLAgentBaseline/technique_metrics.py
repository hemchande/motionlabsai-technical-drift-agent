#!/usr/bin/env python3
"""
Technique Supervised Metrics
Calculates technique-specific metrics based on technique mapping and user requests.
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class TechniqueSupervisedMetrics:
    """
    Calculates technique-specific metrics based on:
    - Technique mapping (from memory indexes)
    - User requests (height, impact force, landing angles, etc.)
    - Pose keypoints
    - Joint velocities and accelerations for impact force calculation
    """
    
    def __init__(self, memory_integration):
        self.memory_integration = memory_integration
        self.technique_mapping = {}
        # Store previous frame data for velocity/acceleration calculation
        self.previous_keypoints: Optional[Dict[str, Any]] = None
        self.previous_velocities: Optional[Dict[str, np.ndarray]] = None
        self.previous_timestamp: Optional[float] = None
        # Rotation tracking state
        self.rotation_state = {
            "total_rotations": 0.0,  # Total rotation count (can be fractional)
            "current_rotation_angle": 0.0,  # Current body rotation angle (0-360)
            "previous_rotation_angle": 0.0,  # Previous frame rotation angle
            "rotation_direction": 0,  # 1 = clockwise, -1 = counterclockwise, 0 = unknown
            "in_rotation_phase": False,  # Whether currently in a rotation phase
            "rotation_start_frame": None,  # Frame when rotation started
            "peak_rotation_velocity": 0.0  # Peak angular velocity during rotation
        }
    
    def calculate_metrics(
        self,
        keypoints: Dict[str, Any],
        technique: str,
        user_requests: List[str],
        frame_timestamp: Optional[float] = None,
        previous_frame_keypoints: Optional[Dict[str, Any]] = None,
        existing_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate technique-specific metrics.
        
        Args:
            keypoints: Current frame keypoints
            technique: Technique name
            user_requests: List of requested metrics
            frame_timestamp: Optional timestamp for temporal metrics
            previous_frame_keypoints: Previous frame for velocity/acceleration
            existing_metrics: Optional existing metrics (used to pass landing phase state)
        
        Returns:
            Dictionary of metric_name -> value
        """
        metrics = existing_metrics.copy() if existing_metrics else {}
        
        # Get technique mapping from memory
        technique_data = self.memory_integration.get_technique_mapping(technique)
        if technique_data:
            self.technique_mapping = technique_data
        
        # Extract requested metrics
        request_text = " ".join(user_requests).lower()
        
        # Dynamic metric calculation - handle any discovered metric
        # This allows the system to calculate metrics discovered via deep research
        
        # Height off floor
        if "height" in request_text or "height off floor" in request_text:
            metrics.update(self._calculate_height_off_floor(keypoints))
        
        # Joint velocity and acceleration (for impact force calculation)
        if "velocity" in request_text or "acceleration" in request_text or "impact" in request_text or "force" in request_text:
            velocity_metrics, acceleration_metrics = self._calculate_joint_kinematics(
                keypoints, previous_frame_keypoints, frame_timestamp
            )
            metrics.update(velocity_metrics)
            metrics.update(acceleration_metrics)
        
        # Impact force (from acceleration)
        if "impact" in request_text or "force" in request_text:
            metrics.update(self._calculate_impact_force(
                keypoints, previous_frame_keypoints, frame_timestamp
            ))
        
        # Landing bend angles - ONLY calculate during landing phases
        # Check if we're in a landing phase (passed via existing_metrics)
        in_landing_phase = existing_metrics.get("in_landing_phase", False) if existing_metrics else False
        
        landing_calculated = False
        if ("landing" in request_text or "bend" in request_text) and in_landing_phase:
            # Only calculate landing bend angles if we're actually in a landing phase
            landing_metrics = self._calculate_landing_bend_angles(keypoints)
            metrics.update(landing_metrics)
            landing_calculated = True
        elif "landing" in request_text or "bend" in request_text:
            # Not in landing phase - clear any previous landing angles to avoid using stale data
            metrics["landing_knee_bend_left"] = None
            metrics["landing_knee_bend_right"] = None
            metrics["landing_knee_bend_avg"] = None
            metrics["landing_knee_bend_min"] = None
            landing_calculated = False
        
        # Stiffness (joint rigidity)
        if "stiffness" in request_text or "stiff" in request_text:
            metrics.update(self._calculate_stiffness(keypoints))
        
        # Knee straightness
        if "knee" in request_text or "straight" in request_text:
            metrics.update(self._calculate_knee_straightness(keypoints))
        
        # Hip angles and alignment
        if "hip" in request_text:
            metrics.update(self._calculate_hip_metrics(keypoints))
        
        # Shoulder angles and alignment
        if "shoulder" in request_text:
            metrics.update(self._calculate_shoulder_metrics(keypoints))
        
        # Extension angles (elbow, hip, knee)
        if "extension" in request_text:
            metrics.update(self._calculate_extension_angles(keypoints))
        
        # Flight time (time in air)
        if "flight" in request_text:
            metrics.update(self._calculate_flight_metrics(keypoints, frame_timestamp))
        
        # Generic angle requests
        if "angle" in request_text or "angles" in request_text:
            metrics.update(self._calculate_all_joint_angles(keypoints))
        
        # Rotation counting (for flips, spins, twists)
        if "rotation" in request_text or "rotations" in request_text or "spin" in request_text or "twist" in request_text:
            metrics.update(self._calculate_rotations(
                keypoints, previous_frame_keypoints, frame_timestamp
            ))
        
        # Explicit height requests (ensure it's calculated even if not in base metrics)
        if "height" in request_text or "height off" in request_text or "height off ground" in request_text:
            if "height_off_floor_pixels" not in metrics:
                metrics.update(self._calculate_height_off_floor(keypoints))
        
        # Knee valgus angles (inward collapse detection) - ALWAYS calculate for all frames
        # Valgus is important for ACL risk, posture analysis, and general biomechanical assessment
        # Calculate valgus angles from keypoints if not already calculated
        if "left_knee_valgus_angle" not in metrics and "right_knee_valgus_angle" not in metrics:
            valgus_metrics = self._calculate_knee_valgus_angles(keypoints)
            metrics.update(valgus_metrics)
        
        # ACL tear risk assessment
        # AUTOMATICALLY calculate for gymnastics techniques (standard requirement)
        # Also calculate if explicitly requested by user for other activities
        is_gymnastics = (
            technique and (
                technique.lower().startswith(("fx_", "bb_", "ub_", "vt_")) or
                "gymnastics" in technique.lower() or
                "gymnast" in technique.lower() or
                any(gym_word in request_text for gym_word in ["gymnastics", "gymnast", "floor", "beam", "bars", "vault", "back handspring", "back tuck", "flip"])
            )
        )
        
        should_calculate_acl_risk = (
            is_gymnastics or  # Always calculate for gymnastics
            "acl" in request_text or 
            "acl tear" in request_text or
            "acl risk" in request_text or
            "acl injury" in request_text or
            "anterior cruciate" in request_text
        )
        
        if should_calculate_acl_risk:
            # Calculate ACL risk (will use existing metrics including valgus angles)
            # Note: This is called AFTER landing metrics and valgus are calculated so it can use them
            acl_metrics = self._calculate_acl_tear_risk(
                keypoints, previous_frame_keypoints, frame_timestamp, metrics
            )
            metrics.update(acl_metrics)
        
        # Posture-specific base metrics (always calculated for posture activity)
        # Check if technique starts with "posture_" or if request mentions posture
        is_posture = technique.startswith("posture_") if technique else False
        posture_keywords = ["posture", "rounded", "arched", "hunched", "forward head", "hyperextended", "bowed", "kyphosis", "lordosis", "valgus", "varus"]
        has_posture_keyword = any(keyword in request_text.lower() for keyword in posture_keywords)
        
        if is_posture or has_posture_keyword:
            metrics.update(self._calculate_posture_base_metrics(keypoints))
        
        # Always calculate basic pose metrics
        metrics.update(self._calculate_basic_metrics(keypoints))
        
        # Update previous frame data for next calculation
        self.previous_keypoints = keypoints.copy() if keypoints else None
        self.previous_timestamp = frame_timestamp
        
        return metrics
    
    def _calculate_height_off_floor(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate height off floor (hip/center of mass height)"""
        metrics = {}
        
        try:
            # Get hip center
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            
            if left_hip and right_hip:
                hip_center_y = (left_hip[1] + right_hip[1]) / 2
                
                # Get ground level (lowest ankle point)
                left_ankle = keypoints.get("left_ankle") or keypoints.get("ankle_left")
                right_ankle = keypoints.get("right_ankle") or keypoints.get("ankle_right")
                
                ground_y = 0
                if left_ankle:
                    ground_y = max(ground_y, left_ankle[1])
                if right_ankle:
                    ground_y = max(ground_y, right_ankle[1])
                
                # Height in pixels (would need calibration for meters)
                height_pixels = abs(hip_center_y - ground_y) if ground_y > 0 else hip_center_y
                
                metrics["height_off_floor_pixels"] = height_pixels
                metrics["height_off_floor_normalized"] = height_pixels / 1000.0  # Normalized
                
                # Estimate in meters (rough calibration - would need actual calibration)
                # Assuming average person height ~1.7m, typical image height ~1000px
                metrics["height_off_floor_meters"] = height_pixels * (1.7 / 1000.0)
        except Exception as e:
            logger.error(f"❌ Error calculating height: {e}")
        
        return metrics
    
    def _calculate_joint_kinematics(
        self,
        keypoints: Dict[str, Any],
        previous_keypoints: Optional[Dict[str, Any]],
        timestamp: Optional[float]
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate joint velocities and accelerations.
        
        Args:
            keypoints: Current frame keypoints
            previous_keypoints: Previous frame keypoints
            timestamp: Current frame timestamp
        
        Returns:
            Tuple of (velocity_metrics, acceleration_metrics)
        """
        velocity_metrics = {}
        acceleration_metrics = {}
        
        try:
            if not previous_keypoints or timestamp is None:
                return velocity_metrics, acceleration_metrics
            
            # Calculate time delta
            if self.previous_timestamp is not None:
                dt = timestamp - self.previous_timestamp
            else:
                dt = 1.0 / 30.0  # Default 30fps
            
            if dt <= 0:
                return velocity_metrics, acceleration_metrics
            
            # Pixel to meter conversion (rough calibration)
            pixel_to_meter = 1.7 / 1000.0  # Assuming 1.7m person height = 1000px
            
            # Key joints to track
            joint_names = [
                "left_ankle", "right_ankle",
                "left_knee", "right_knee",
                "left_hip", "right_hip",
                "left_shoulder", "right_shoulder",
                "left_wrist", "right_wrist",
                "left_elbow", "right_elbow"
            ]
            
            current_velocities = {}
            
            # Calculate velocities for each joint
            for joint_name in joint_names:
                # Try different key name variations
                current_pos = (keypoints.get(joint_name) or 
                              keypoints.get(joint_name.replace("left_", "left_").replace("right_", "right_")))
                prev_pos = (previous_keypoints.get(joint_name) or 
                           previous_keypoints.get(joint_name.replace("left_", "left_").replace("right_", "right_")))
                
                if current_pos and prev_pos and len(current_pos) >= 2 and len(prev_pos) >= 2:
                    # Calculate velocity vector (pixels per second)
                    velocity_px = np.array([
                        (current_pos[0] - prev_pos[0]) / dt,
                        (current_pos[1] - prev_pos[1]) / dt
                    ])
                    
                    # Convert to m/s
                    velocity_ms = velocity_px * pixel_to_meter
                    
                    # Store velocity
                    current_velocities[joint_name] = velocity_ms
                    
                    # Calculate velocity magnitude
                    velocity_magnitude = np.linalg.norm(velocity_ms)
                    velocity_metrics[f"{joint_name}_velocity_ms"] = float(velocity_magnitude)
                    
                    # Vertical velocity (important for landing impact)
                    vertical_velocity = velocity_ms[1]  # Y component (positive = downward)
                    velocity_metrics[f"{joint_name}_vertical_velocity_ms"] = float(vertical_velocity)
                    
                    # Horizontal velocity
                    horizontal_velocity = velocity_ms[0]  # X component
                    velocity_metrics[f"{joint_name}_horizontal_velocity_ms"] = float(horizontal_velocity)
            
            # Calculate accelerations from velocity change
            if self.previous_velocities:
                for joint_name in joint_names:
                    if joint_name in current_velocities and joint_name in self.previous_velocities:
                        # Calculate acceleration vector
                        acceleration_ms2 = (current_velocities[joint_name] - self.previous_velocities[joint_name]) / dt
                        
                        # Acceleration magnitude
                        acceleration_magnitude = np.linalg.norm(acceleration_ms2)
                        acceleration_metrics[f"{joint_name}_acceleration_ms2"] = float(acceleration_magnitude)
                        
                        # Vertical acceleration (critical for impact force)
                        vertical_acceleration = acceleration_ms2[1]  # Y component
                        acceleration_metrics[f"{joint_name}_vertical_acceleration_ms2"] = float(vertical_acceleration)
            
            # Calculate center of mass (COM) velocity and acceleration
            com_velocity, com_acceleration = self._calculate_com_kinematics(
                keypoints, previous_keypoints, current_velocities, dt
            )
            if com_velocity is not None:
                velocity_metrics["com_velocity_ms"] = float(np.linalg.norm(com_velocity))
                velocity_metrics["com_vertical_velocity_ms"] = float(com_velocity[1])
            if com_acceleration is not None:
                acceleration_metrics["com_acceleration_ms2"] = float(np.linalg.norm(com_acceleration))
                acceleration_metrics["com_vertical_acceleration_ms2"] = float(com_acceleration[1])
            
            # Update previous velocities for next frame
            self.previous_velocities = current_velocities
            
        except Exception as e:
            logger.error(f"❌ Error calculating joint kinematics: {e}")
        
        return velocity_metrics, acceleration_metrics
    
    def _calculate_com_kinematics(
        self,
        keypoints: Dict[str, Any],
        previous_keypoints: Dict[str, Any],
        current_velocities: Dict[str, np.ndarray],
        dt: float
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Calculate center of mass velocity and acceleration"""
        try:
            # Estimate COM position from major joints
            com_joints = ["left_hip", "right_hip", "left_shoulder", "right_shoulder"]
            
            current_com = np.array([0.0, 0.0])
            prev_com = np.array([0.0, 0.0])
            count = 0
            
            for joint_name in com_joints:
                current_pos = keypoints.get(joint_name)
                prev_pos = previous_keypoints.get(joint_name)
                
                if current_pos and prev_pos and len(current_pos) >= 2 and len(prev_pos) >= 2:
                    current_com += np.array([float(current_pos[0]), float(current_pos[1])])
                    prev_com += np.array([float(prev_pos[0]), float(prev_pos[1])])
                    count += 1
            
            if count == 0:
                return None, None
            
            current_com /= count
            prev_com /= count
            
            # Calculate COM velocity
            pixel_to_meter = 1.7 / 1000.0
            com_velocity = (current_com - prev_com) / dt * pixel_to_meter
            
            # Calculate COM acceleration
            com_acceleration = None
            if self.previous_velocities:
                # Average velocities of COM joints
                com_prev_velocity = np.array([0.0, 0.0])
                com_curr_velocity = np.array([0.0, 0.0])
                count = 0
                
                for joint_name in com_joints:
                    if joint_name in current_velocities and joint_name in self.previous_velocities:
                        com_curr_velocity += current_velocities[joint_name]
                        com_prev_velocity += self.previous_velocities[joint_name]
                        count += 1
                
                if count > 0:
                    com_curr_velocity /= count
                    com_prev_velocity /= count
                    com_acceleration = (com_curr_velocity - com_prev_velocity) / dt
            
            return com_velocity, com_acceleration
            
        except Exception as e:
            logger.debug(f"Error calculating COM kinematics: {e}")
            return None, None
    
    def _calculate_impact_force(
        self,
        keypoints: Dict[str, Any],
        previous_keypoints: Optional[Dict[str, Any]],
        timestamp: Optional[float]
    ) -> Dict[str, float]:
        """
        Calculate impact force from joint acceleration (F = ma).
        More accurate than velocity-based estimation.
        Uses acceleration from joint kinematics calculation.
        """
        metrics = {}
        
        try:
            if not previous_keypoints or timestamp is None:
                return {"impact_force_N": 0.0, "landing_velocity_ms": 0.0}
            
            # Calculate velocity and acceleration (will use stored previous_velocities if available)
            velocity_metrics, accel_metrics = self._calculate_joint_kinematics(
                keypoints, previous_keypoints, timestamp
            )
            
            # Get ankle vertical acceleration for impact force (most relevant for landing)
            left_ankle_accel = accel_metrics.get("left_ankle_vertical_acceleration_ms2", 0.0)
            right_ankle_accel = accel_metrics.get("right_ankle_vertical_acceleration_ms2", 0.0)
            
            # Use maximum acceleration (worst case impact)
            max_ankle_accel = max(abs(left_ankle_accel), abs(right_ankle_accel))
            
            # Use COM acceleration if available (more accurate for whole-body impact)
            com_accel = accel_metrics.get("com_vertical_acceleration_ms2")
            if com_accel is not None and abs(com_accel) > 0:
                impact_acceleration = abs(com_accel)
            elif max_ankle_accel > 0:
                impact_acceleration = max_ankle_accel
            else:
                # Fallback: calculate from velocity if acceleration not available yet
                landing_velocity = max(
                    abs(velocity_metrics.get("left_ankle_vertical_velocity_ms", 0.0)),
                    abs(velocity_metrics.get("right_ankle_vertical_velocity_ms", 0.0))
                )
                if landing_velocity > 0:
                    # Estimate acceleration from velocity change (a = Δv / Δt)
                    # This is a fallback when we don't have 2 frames of velocity data yet
                    dt = timestamp - self.previous_timestamp if self.previous_timestamp else 1.0/30.0
                    if dt > 0:
                        # Rough estimate: assume deceleration during landing
                        impact_acceleration = landing_velocity / dt
                    else:
                        impact_acceleration = 0.0
                else:
                    impact_acceleration = 0.0
            
            # Calculate impact force using F = ma
            # Mass estimate (would need actual weight - configurable)
            mass_kg = 50.0  # Default for gymnast, should be configurable
            
            # Impact force (positive acceleration = downward impact)
            impact_force_N = mass_kg * impact_acceleration
            
            # Get landing velocity for reporting
            landing_velocity = max(
                abs(velocity_metrics.get("left_ankle_vertical_velocity_ms", 0.0)),
                abs(velocity_metrics.get("right_ankle_vertical_velocity_ms", 0.0)),
                abs(velocity_metrics.get("com_vertical_velocity_ms", 0.0))
            )
            
            # Alternative force calculation from velocity (F = mv^2 / (2d)) for comparison
            # d = compression distance during landing
            compression_distance_m = 0.1  # Typical landing compression (10cm)
            if landing_velocity > 0:
                force_from_velocity = mass_kg * (landing_velocity ** 2) / (2 * compression_distance_m)
            else:
                force_from_velocity = 0.0
            
            metrics["impact_force_N"] = impact_force_N
            metrics["impact_force_from_velocity_N"] = force_from_velocity
            metrics["landing_velocity_ms"] = landing_velocity
            metrics["impact_acceleration_ms2"] = impact_acceleration
            metrics["deceleration_ms2"] = impact_acceleration  # Same as acceleration for landing
            
            # Joint-specific impact forces (per leg)
            if left_ankle_accel != 0:
                metrics["left_ankle_impact_force_N"] = mass_kg * abs(left_ankle_accel) * 0.5  # Half body weight per leg
            if right_ankle_accel != 0:
                metrics["right_ankle_impact_force_N"] = mass_kg * abs(right_ankle_accel) * 0.5
            
            # Peak impact detection (sudden acceleration spike indicating landing)
            # Threshold: > 50 m/s² (approximately 5g) indicates significant impact
            if impact_acceleration > 50.0:
                metrics["peak_impact_detected"] = 1.0
                metrics["peak_impact_force_N"] = impact_force_N
                metrics["peak_impact_acceleration_ms2"] = impact_acceleration
            else:
                metrics["peak_impact_detected"] = 0.0
            
            # Additional metrics for analysis
            if com_accel is not None:
                metrics["com_impact_force_N"] = mass_kg * abs(com_accel)
            
        except Exception as e:
            logger.error(f"❌ Error calculating impact force: {e}", exc_info=True)
            metrics = {"impact_force_N": 0.0, "landing_velocity_ms": 0.0}
        
        return metrics
    
    def _calculate_landing_bend_angles(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate landing bend angles (knee, hip, ankle)"""
        metrics = {}
        
        try:
            # Knee angles
            left_knee_angle = self._calculate_joint_angle(
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left")
            )
            right_knee_angle = self._calculate_joint_angle(
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right")
            )
            
            if left_knee_angle is not None:
                metrics["landing_knee_bend_left"] = left_knee_angle
            if right_knee_angle is not None:
                metrics["landing_knee_bend_right"] = right_knee_angle
            
            if left_knee_angle is not None and right_knee_angle is not None:
                metrics["landing_knee_bend_avg"] = (left_knee_angle + right_knee_angle) / 2
                metrics["landing_knee_bend_min"] = min(left_knee_angle, right_knee_angle)
            elif left_knee_angle is not None:
                metrics["landing_knee_bend_avg"] = left_knee_angle
                metrics["landing_knee_bend_min"] = left_knee_angle
            elif right_knee_angle is not None:
                metrics["landing_knee_bend_avg"] = right_knee_angle
                metrics["landing_knee_bend_min"] = right_knee_angle
            
            # Hip angles
            left_hip_angle = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left")
            )
            right_hip_angle = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right")
            )
            
            if left_hip_angle is not None:
                metrics["landing_hip_bend_left"] = left_hip_angle
            if right_hip_angle is not None:
                metrics["landing_hip_bend_right"] = right_hip_angle
            
            if left_hip_angle is not None and right_hip_angle is not None:
                metrics["landing_hip_bend_avg"] = (left_hip_angle + right_hip_angle) / 2
            elif left_hip_angle is not None:
                metrics["landing_hip_bend_avg"] = left_hip_angle
            elif right_hip_angle is not None:
                metrics["landing_hip_bend_avg"] = right_hip_angle
            
            # Ankle angles
            left_ankle_angle = self._calculate_joint_angle(
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left"),
                [keypoints.get("left_ankle", [0, 0])[0], keypoints.get("left_ankle", [0, 0])[1] + 10]  # Ground reference
            )
            right_ankle_angle = self._calculate_joint_angle(
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right"),
                [keypoints.get("right_ankle", [0, 0])[0], keypoints.get("right_ankle", [0, 0])[1] + 10]
            )
            
            metrics["landing_ankle_bend_left"] = left_ankle_angle
            metrics["landing_ankle_bend_right"] = right_ankle_angle
            
        except Exception as e:
            logger.error(f"❌ Error calculating landing bend angles: {e}")
        
        return metrics
    
    def _calculate_stiffness(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate joint stiffness (rigidity) from joint angle consistency"""
        metrics = {}
        
        try:
            # Stiffness is measured as consistency of joint angles
            # More consistent = stiffer (less variation)
            # This is a simplified measure - real stiffness requires force measurements
            
            # Calculate joint angles
            joint_angles = []
            
            # Knee angles
            left_knee = self._calculate_joint_angle(
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left")
            )
            right_knee = self._calculate_joint_angle(
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right")
            )
            if left_knee is not None:
                joint_angles.append(left_knee)
            if right_knee is not None:
                joint_angles.append(right_knee)
            
            # Elbow angles
            left_elbow = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_elbow") or keypoints.get("elbow_left"),
                keypoints.get("left_wrist") or keypoints.get("wrist_left")
            )
            right_elbow = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_elbow") or keypoints.get("elbow_right"),
                keypoints.get("right_wrist") or keypoints.get("wrist_right")
            )
            if left_elbow is not None:
                joint_angles.append(left_elbow)
            if right_elbow is not None:
                joint_angles.append(right_elbow)
            
            # Stiffness = inverse of variance (higher = stiffer)
            if len(joint_angles) > 1:
                variance = np.var(joint_angles)
                stiffness = 1.0 / (variance + 1e-6)  # Add small epsilon to avoid division by zero
                metrics["joint_stiffness"] = stiffness
                metrics["joint_angle_variance"] = variance
            else:
                metrics["joint_stiffness"] = 0.0
                metrics["joint_angle_variance"] = 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating stiffness: {e}")
        
        return metrics
    
    def _calculate_knee_straightness(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate knee straightness (extension angle)"""
        metrics = {}
        
        try:
            # Knee extension = 180 - knee bend angle
            # Higher extension = straighter knee
            
            left_knee_angle = self._calculate_joint_angle(
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left")
            )
            right_knee_angle = self._calculate_joint_angle(
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right")
            )
            
            if left_knee_angle:
                left_extension = 180.0 - left_knee_angle
                metrics["knee_extension_left"] = left_extension
                metrics["knee_straightness_left"] = left_extension / 180.0  # Normalized 0-1
            
            if right_knee_angle:
                right_extension = 180.0 - right_knee_angle
                metrics["knee_extension_right"] = right_extension
                metrics["knee_straightness_right"] = right_extension / 180.0
            
            if left_knee_angle and right_knee_angle:
                avg_extension = ((180.0 - left_knee_angle) + (180.0 - right_knee_angle)) / 2
                metrics["knee_extension_avg"] = avg_extension
                metrics["knee_straightness_avg"] = avg_extension / 180.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating knee straightness: {e}")
        
        return metrics
    
    def _calculate_basic_metrics(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate basic pose metrics"""
        metrics = {}
        
        try:
            # Body alignment
            nose = keypoints.get("nose")
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            
            if nose and left_hip and right_hip:
                hip_center = [
                    (left_hip[0] + right_hip[0]) / 2,
                    (left_hip[1] + right_hip[1]) / 2
                ]
                
                # Calculate alignment (deviation from vertical)
                body_vector = np.array([nose[0] - hip_center[0], nose[1] - hip_center[1]])
                vertical = np.array([0, 1])
                
                if np.linalg.norm(body_vector) > 0:
                    cos_angle = np.dot(body_vector, vertical) / np.linalg.norm(body_vector)
                    alignment_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    metrics["body_alignment_degrees"] = alignment_angle
            
            # Split angle (for leaps/jumps)
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            left_knee = keypoints.get("left_knee") or keypoints.get("knee_left")
            right_knee = keypoints.get("right_knee") or keypoints.get("knee_right")
            
            if left_hip and right_hip and left_knee and right_knee:
                left_leg = np.array(left_knee) - np.array(left_hip)
                right_leg = np.array(right_knee) - np.array(right_hip)
                
                if np.linalg.norm(left_leg) > 0 and np.linalg.norm(right_leg) > 0:
                    cos_angle = np.dot(left_leg, right_leg) / (np.linalg.norm(left_leg) * np.linalg.norm(right_leg))
                    split_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    metrics["split_angle_degrees"] = split_angle
                    
        except Exception as e:
            logger.error(f"❌ Error calculating basic metrics: {e}")
        
        return metrics
    
    def _calculate_hip_metrics(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive hip angles and alignment metrics"""
        metrics = {}
        
        try:
            # Hip flexion/extension angles
            left_hip_angle = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left")
            )
            right_hip_angle = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right")
            )
            
            if left_hip_angle is not None:
                metrics["hip_angle_left"] = left_hip_angle
                metrics["hip_extension_left"] = 180.0 - left_hip_angle  # Extension = 180 - flexion
            if right_hip_angle is not None:
                metrics["hip_angle_right"] = right_hip_angle
                metrics["hip_extension_right"] = 180.0 - right_hip_angle
            
            if left_hip_angle is not None and right_hip_angle is not None:
                metrics["hip_angle_avg"] = (left_hip_angle + right_hip_angle) / 2
                metrics["hip_extension_avg"] = (metrics["hip_extension_left"] + metrics["hip_extension_right"]) / 2
            
            # Hip alignment (horizontal alignment of hips)
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            
            if left_hip and right_hip and len(left_hip) >= 2 and len(right_hip) >= 2:
                # Vertical difference indicates hip tilt
                hip_tilt = abs(float(left_hip[1]) - float(right_hip[1]))
                metrics["hip_tilt_pixels"] = hip_tilt
                
                # Horizontal alignment (should be symmetric)
                hip_width = abs(float(left_hip[0]) - float(right_hip[0]))
                metrics["hip_width_pixels"] = hip_width
                
                # Hip center position
                hip_center_x = (float(left_hip[0]) + float(right_hip[0])) / 2
                hip_center_y = (float(left_hip[1]) + float(right_hip[1])) / 2
                metrics["hip_center_x"] = hip_center_x
                metrics["hip_center_y"] = hip_center_y
            
            # Hip-shoulder alignment (body alignment)
            left_shoulder = keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            right_shoulder = keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            
            if left_hip and right_hip and left_shoulder and right_shoulder:
                # Calculate angle between shoulder-hip line and vertical
                shoulder_center_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2
                shoulder_center_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                hip_center_x = (float(left_hip[0]) + float(right_hip[0])) / 2
                hip_center_y = (float(left_hip[1]) + float(right_hip[1])) / 2
                
                # Body alignment angle (deviation from vertical)
                dx = shoulder_center_x - hip_center_x
                dy = shoulder_center_y - hip_center_y
                if dx != 0:
                    body_alignment_angle = np.arctan2(dx, abs(dy)) * 180.0 / np.pi
                    metrics["body_alignment_angle"] = abs(body_alignment_angle)
                
        except Exception as e:
            logger.error(f"❌ Error calculating hip metrics: {e}")
        
        return metrics
    
    def _calculate_shoulder_metrics(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive shoulder angles and alignment metrics"""
        metrics = {}
        
        try:
            # Shoulder angles (elbow-shoulder-shoulder line)
            left_shoulder_angle = self._calculate_joint_angle(
                keypoints.get("left_elbow") or keypoints.get("elbow_left"),
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            )
            right_shoulder_angle = self._calculate_joint_angle(
                keypoints.get("right_elbow") or keypoints.get("elbow_right"),
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            )
            
            if left_shoulder_angle is not None:
                metrics["shoulder_angle_left"] = left_shoulder_angle
            if right_shoulder_angle is not None:
                metrics["shoulder_angle_right"] = right_shoulder_angle
            
            if left_shoulder_angle is not None and right_shoulder_angle is not None:
                metrics["shoulder_angle_avg"] = (left_shoulder_angle + right_shoulder_angle) / 2
            
            # Shoulder alignment (horizontal alignment)
            left_shoulder = keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            right_shoulder = keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            
            if left_shoulder and right_shoulder and len(left_shoulder) >= 2 and len(right_shoulder) >= 2:
                # Vertical difference indicates shoulder tilt
                shoulder_tilt = abs(float(left_shoulder[1]) - float(right_shoulder[1]))
                metrics["shoulder_tilt_pixels"] = shoulder_tilt
                
                # Horizontal alignment (shoulder width)
                shoulder_width = abs(float(left_shoulder[0]) - float(right_shoulder[0]))
                metrics["shoulder_width_pixels"] = shoulder_width
                
                # Shoulder center position
                shoulder_center_x = (float(left_shoulder[0]) + float(right_shoulder[0])) / 2
                shoulder_center_y = (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                metrics["shoulder_center_x"] = shoulder_center_x
                metrics["shoulder_center_y"] = shoulder_center_y
            
            # Shoulder elevation (arm position relative to body)
            left_elbow = keypoints.get("left_elbow") or keypoints.get("elbow_left")
            right_elbow = keypoints.get("right_elbow") or keypoints.get("elbow_right")
            
            if left_shoulder and left_elbow:
                shoulder_elevation_left = float(left_shoulder[1]) - float(left_elbow[1])  # Negative = arm up
                metrics["shoulder_elevation_left"] = shoulder_elevation_left
            
            if right_shoulder and right_elbow:
                shoulder_elevation_right = float(right_shoulder[1]) - float(right_elbow[1])
                metrics["shoulder_elevation_right"] = shoulder_elevation_right
            
        except Exception as e:
            logger.error(f"❌ Error calculating shoulder metrics: {e}")
        
        return metrics
    
    def _calculate_extension_angles(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate extension angles for all major joints"""
        metrics = {}
        
        try:
            # Knee extension (already calculated in knee_straightness, but add here for completeness)
            left_knee_angle = self._calculate_joint_angle(
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left")
            )
            right_knee_angle = self._calculate_joint_angle(
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right")
            )
            
            if left_knee_angle is not None:
                metrics["knee_extension_left"] = 180.0 - left_knee_angle
            if right_knee_angle is not None:
                metrics["knee_extension_right"] = 180.0 - right_knee_angle
            
            # Elbow extension
            left_elbow_angle = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_elbow") or keypoints.get("elbow_left"),
                keypoints.get("left_wrist") or keypoints.get("wrist_left")
            )
            right_elbow_angle = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_elbow") or keypoints.get("elbow_right"),
                keypoints.get("right_wrist") or keypoints.get("wrist_right")
            )
            
            if left_elbow_angle is not None:
                metrics["elbow_extension_left"] = 180.0 - left_elbow_angle
            if right_elbow_angle is not None:
                metrics["elbow_extension_right"] = 180.0 - right_elbow_angle
            
            # Hip extension (calculated in hip_metrics, but add here)
            left_hip_angle = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left")
            )
            right_hip_angle = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right")
            )
            
            if left_hip_angle is not None:
                metrics["hip_extension_left"] = 180.0 - left_hip_angle
            if right_hip_angle is not None:
                metrics["hip_extension_right"] = 180.0 - right_hip_angle
            
            # Ankle extension (dorsiflexion/plantarflexion)
            left_ankle_angle = self._calculate_joint_angle(
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left"),
                [keypoints.get("left_ankle", [0, 0])[0], keypoints.get("left_ankle", [0, 0])[1] + 10]  # Ground reference
            )
            right_ankle_angle = self._calculate_joint_angle(
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right"),
                [keypoints.get("right_ankle", [0, 0])[0], keypoints.get("right_ankle", [0, 0])[1] + 10]
            )
            
            if left_ankle_angle is not None:
                metrics["ankle_extension_left"] = left_ankle_angle
            if right_ankle_angle is not None:
                metrics["ankle_extension_right"] = right_ankle_angle
            
        except Exception as e:
            logger.error(f"❌ Error calculating extension angles: {e}")
        
        return metrics
    
    def _calculate_flight_metrics(
        self,
        keypoints: Dict[str, Any],
        frame_timestamp: Optional[float]
    ) -> Dict[str, float]:
        """Calculate flight time and phase metrics"""
        metrics = {}
        
        try:
            # Flight detection: when feet are off ground (ankles above a threshold)
            # This is simplified - real flight detection needs ground plane estimation
            
            left_ankle = keypoints.get("left_ankle") or keypoints.get("ankle_left")
            right_ankle = keypoints.get("right_ankle") or keypoints.get("ankle_right")
            
            if left_ankle and right_ankle and len(left_ankle) >= 2 and len(right_ankle) >= 2:
                # Estimate ground level from lowest ankle position in previous frames
                # For now, use a simple heuristic: if ankles are moving upward, likely in flight
                
                # Check if in flight phase (ankles moving up or high position)
                ankle_y_avg = (float(left_ankle[1]) + float(right_ankle[1])) / 2
                
                if self.previous_keypoints:
                    prev_left_ankle = self.previous_keypoints.get("left_ankle") or self.previous_keypoints.get("ankle_left")
                    prev_right_ankle = self.previous_keypoints.get("right_ankle") or self.previous_keypoints.get("ankle_right")
                    
                    if prev_left_ankle and prev_right_ankle:
                        prev_ankle_y_avg = (float(prev_left_ankle[1]) + float(prev_right_ankle[1])) / 2
                        
                        # If ankles moved up, likely in flight
                        if ankle_y_avg < prev_ankle_y_avg:
                            metrics["in_flight"] = 1.0
                            metrics["flight_phase"] = 1.0
                        else:
                            metrics["in_flight"] = 0.0
                            metrics["flight_phase"] = 0.0
                        
                        # Vertical displacement during flight
                        vertical_displacement = prev_ankle_y_avg - ankle_y_avg  # Positive = upward
                        metrics["vertical_displacement_pixels"] = vertical_displacement
                else:
                    metrics["in_flight"] = 0.0
                    metrics["flight_phase"] = 0.0
                
                # Flight height (vertical position relative to estimated ground)
                # This is a simplified measure
                metrics["flight_height_pixels"] = ankle_y_avg
            
            # Flight time calculation would require tracking across multiple frames
            # This is a placeholder - actual flight time needs frame-by-frame tracking
            if frame_timestamp and self.previous_timestamp:
                time_delta = frame_timestamp - self.previous_timestamp
                if metrics.get("in_flight", 0) > 0:
                    # Accumulate flight time (would need persistent state)
                    metrics["flight_time_delta_s"] = time_delta
                else:
                    metrics["flight_time_delta_s"] = 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating flight metrics: {e}")
        
        return metrics
    
    def _calculate_all_joint_angles(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """Calculate all major joint angles"""
        metrics = {}
        
        try:
            # Knee angles
            left_knee = self._calculate_joint_angle(
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left")
            )
            right_knee = self._calculate_joint_angle(
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right")
            )
            if left_knee is not None:
                metrics["knee_angle_left"] = left_knee
            if right_knee is not None:
                metrics["knee_angle_right"] = right_knee
            
            # Hip angles
            left_hip = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_hip") or keypoints.get("hip_left"),
                keypoints.get("left_knee") or keypoints.get("knee_left")
            )
            right_hip = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_hip") or keypoints.get("hip_right"),
                keypoints.get("right_knee") or keypoints.get("knee_right")
            )
            if left_hip is not None:
                metrics["hip_angle_left"] = left_hip
            if right_hip is not None:
                metrics["hip_angle_right"] = right_hip
            
            # Elbow angles
            left_elbow = self._calculate_joint_angle(
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("left_elbow") or keypoints.get("elbow_left"),
                keypoints.get("left_wrist") or keypoints.get("wrist_left")
            )
            right_elbow = self._calculate_joint_angle(
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("right_elbow") or keypoints.get("elbow_right"),
                keypoints.get("right_wrist") or keypoints.get("wrist_right")
            )
            if left_elbow is not None:
                metrics["elbow_angle_left"] = left_elbow
            if right_elbow is not None:
                metrics["elbow_angle_right"] = right_elbow
            
            # Shoulder angles (simplified)
            left_shoulder = self._calculate_joint_angle(
                keypoints.get("left_elbow") or keypoints.get("elbow_left"),
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left"),
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            )
            right_shoulder = self._calculate_joint_angle(
                keypoints.get("right_elbow") or keypoints.get("elbow_right"),
                keypoints.get("right_shoulder") or keypoints.get("shoulder_right"),
                keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            )
            if left_shoulder is not None:
                metrics["shoulder_angle_left"] = left_shoulder
            if right_shoulder is not None:
                metrics["shoulder_angle_right"] = right_shoulder
            
            # Ankle angles
            left_ankle = self._calculate_joint_angle(
                keypoints.get("left_knee") or keypoints.get("knee_left"),
                keypoints.get("left_ankle") or keypoints.get("ankle_left"),
                [keypoints.get("left_ankle", [0, 0])[0], keypoints.get("left_ankle", [0, 0])[1] + 10]
            )
            right_ankle = self._calculate_joint_angle(
                keypoints.get("right_knee") or keypoints.get("knee_right"),
                keypoints.get("right_ankle") or keypoints.get("ankle_right"),
                [keypoints.get("right_ankle", [0, 0])[0], keypoints.get("right_ankle", [0, 0])[1] + 10]
            )
            if left_ankle is not None:
                metrics["ankle_angle_left"] = left_ankle
            if right_ankle is not None:
                metrics["ankle_angle_right"] = right_ankle
            
        except Exception as e:
            logger.error(f"❌ Error calculating all joint angles: {e}")
        
        return metrics
    
    def _calculate_posture_base_metrics(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate base posture metrics:
        - Rounded/arched back
        - Hunched shoulders
        - Forward head posture (distended neck)
        - Hyperextended knees
        - Bowed knees (varus/valgus)
        - Hip alignment
        """
        metrics = {}
        
        try:
            # Get key points
            nose = keypoints.get("nose")
            left_shoulder = keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            right_shoulder = keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            left_knee = keypoints.get("left_knee") or keypoints.get("knee_left")
            right_knee = keypoints.get("right_knee") or keypoints.get("knee_right")
            left_ankle = keypoints.get("left_ankle") or keypoints.get("ankle_left")
            right_ankle = keypoints.get("right_ankle") or keypoints.get("ankle_right")
            
            # 1. Rounded Back (Kyphosis)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_center = np.array([
                    (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
                    (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                ])
                hip_center = np.array([
                    (float(left_hip[0]) + float(right_hip[0])) / 2,
                    (float(left_hip[1]) + float(right_hip[1])) / 2
                ])
                
                # Calculate forward shoulder position (rounded back indicator)
                # If shoulders are forward relative to hips, back is rounded
                shoulder_forward = float(shoulder_center[0]) - float(hip_center[0])
                
                # Calculate kyphosis angle (rounded back)
                spine_vector = hip_center - shoulder_center
                vertical = np.array([0, 1])
                
                if np.linalg.norm(spine_vector) > 0:
                    cos_angle = np.dot(spine_vector, vertical) / np.linalg.norm(spine_vector)
                    spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # Rounded back: spine angle < 90° (forward lean)
                    rounded_back_angle = 90.0 - spine_angle if spine_angle < 90 else 0.0
                    metrics["rounded_back_angle"] = rounded_back_angle
                    metrics["rounded_back_detected"] = 1.0 if rounded_back_angle > 10.0 else 0.0
                    metrics["shoulder_forward_position_cm"] = shoulder_forward * 0.01  # Approximate cm
            
            # 2. Arched Back (Hyperlordosis)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_center = np.array([
                    (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
                    (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                ])
                hip_center = np.array([
                    (float(left_hip[0]) + float(right_hip[0])) / 2,
                    (float(left_hip[1]) + float(right_hip[1])) / 2
                ])
                
                # Calculate lumbar arch (arched back indicator)
                # If hips are pushed forward relative to shoulders, back is arched
                hip_forward = float(hip_center[0]) - float(shoulder_center[0])
                
                # Calculate lordosis angle
                spine_vector = hip_center - shoulder_center
                vertical = np.array([0, 1])
                
                if np.linalg.norm(spine_vector) > 0:
                    cos_angle = np.dot(spine_vector, vertical) / np.linalg.norm(spine_vector)
                    spine_angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    # Arched back: excessive lumbar curve (hip forward)
                    arched_back_angle = abs(hip_forward) * 0.1  # Approximate angle
                    metrics["arched_back_angle"] = arched_back_angle
                    metrics["arched_back_detected"] = 1.0 if arched_back_angle > 10.0 else 0.0
                    metrics["lumbar_arch_cm"] = abs(hip_forward) * 0.01
            
            # 3. Hunched Shoulders (Forward shoulder position)
            if left_shoulder and right_shoulder and left_hip and right_hip:
                shoulder_center = np.array([
                    (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
                    (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                ])
                hip_center = np.array([
                    (float(left_hip[0]) + float(right_hip[0])) / 2,
                    (float(left_hip[1]) + float(right_hip[1])) / 2
                ])
                
                # Shoulder forward position relative to hip
                shoulder_forward = float(shoulder_center[0]) - float(hip_center[0])
                
                # Hunched: shoulders significantly forward
                metrics["shoulder_forward_position_cm"] = shoulder_forward * 0.01
                metrics["hunched_shoulders_detected"] = 1.0 if shoulder_forward > 20.0 else 0.0
                metrics["hunched_shoulders_angle"] = abs(shoulder_forward) * 0.1
            
            # 4. Forward Head Posture (Distended Neck)
            if nose and left_shoulder and right_shoulder:
                shoulder_center = np.array([
                    (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
                    (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
                ])
                nose_pos = np.array([float(nose[0]), float(nose[1])])
                
                # Head forward position relative to shoulders
                head_forward = float(nose_pos[0]) - float(shoulder_center[0])
                
                # Forward head: >2.5cm forward
                metrics["forward_head_posture_cm"] = head_forward * 0.01
                metrics["forward_head_detected"] = 1.0 if abs(head_forward) > 25.0 else 0.0
                metrics["forward_head_angle"] = abs(head_forward) * 0.1
            
            # 5. Hyperextended Knees
            if left_knee and left_ankle and left_hip:
                # Calculate knee extension angle
                knee_angle = self._calculate_joint_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )
                
                if knee_angle is not None:
                    # Hyperextended: knee angle > 180° (straight or beyond)
                    extension = 180.0 - knee_angle
                    hyperextension = extension if extension < 0 else 0.0
                    metrics["left_knee_hyperextension_angle"] = abs(hyperextension)
                    metrics["left_knee_hyperextended"] = 1.0 if abs(hyperextension) > 5.0 else 0.0
            
            if right_knee and right_ankle and right_hip:
                knee_angle = self._calculate_joint_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )
                
                if knee_angle is not None:
                    extension = 180.0 - knee_angle
                    hyperextension = extension if extension < 0 else 0.0
                    metrics["right_knee_hyperextension_angle"] = abs(hyperextension)
                    metrics["right_knee_hyperextended"] = 1.0 if abs(hyperextension) > 5.0 else 0.0
            
            if "left_knee_hyperextended" in metrics and "right_knee_hyperextended" in metrics:
                metrics["knee_hyperextended_avg"] = (
                    metrics["left_knee_hyperextended"] + metrics["right_knee_hyperextended"]
                ) / 2.0
            
            # 6. Bowed Knees (Varus/Valgus) - use shared function
            valgus_metrics = self._calculate_knee_valgus_angles(keypoints)
            metrics.update(valgus_metrics)
            
            if "left_knee_bowed_detected" in metrics and "right_knee_bowed_detected" in metrics:
                metrics["knee_bowed_avg"] = (
                    metrics["left_knee_bowed_detected"] + metrics["right_knee_bowed_detected"]
                ) / 2.0
            
            if "left_knee_bowed_detected" in metrics and "right_knee_bowed_detected" in metrics:
                metrics["knee_bowed_avg"] = (
                    metrics["left_knee_bowed_detected"] + metrics["right_knee_bowed_detected"]
                ) / 2.0
            
            # 7. Hip Alignment
            if left_hip and right_hip:
                # Hip levelness (vertical alignment)
                hip_tilt = abs(float(left_hip[1]) - float(right_hip[1]))
                metrics["hip_tilt_pixels"] = hip_tilt
                metrics["hip_tilt_cm"] = hip_tilt * 0.01
                metrics["hip_misaligned"] = 1.0 if hip_tilt > 10.0 else 0.0
                
                # Hip width (horizontal alignment)
                hip_width = abs(float(left_hip[0]) - float(right_hip[0]))
                metrics["hip_width_pixels"] = hip_width
                
                # Hip center position
                hip_center_x = (float(left_hip[0]) + float(right_hip[0])) / 2
                hip_center_y = (float(left_hip[1]) + float(right_hip[1])) / 2
                metrics["hip_center_x"] = hip_center_x
                metrics["hip_center_y"] = hip_center_y
                
                # Hip alignment angle (relative to horizontal)
                if hip_width > 0:
                    hip_alignment_angle = np.degrees(np.arctan(hip_tilt / hip_width))
                    metrics["hip_alignment_angle"] = hip_alignment_angle
            
            # Summary posture score (0-1, higher = better posture)
            posture_issues = sum([
                metrics.get("rounded_back_detected", 0),
                metrics.get("arched_back_detected", 0),
                metrics.get("hunched_shoulders_detected", 0),
                metrics.get("forward_head_detected", 0),
                metrics.get("left_knee_hyperextended", 0),
                metrics.get("right_knee_hyperextended", 0),
                metrics.get("left_knee_bowed_detected", 0),
                metrics.get("right_knee_bowed_detected", 0),
                metrics.get("hip_misaligned", 0)
            ])
            
            metrics["posture_issues_count"] = float(posture_issues)
            metrics["posture_score"] = max(0.0, 1.0 - (posture_issues * 0.1))  # Deduct 0.1 per issue
            
        except Exception as e:
            logger.error(f"❌ Error calculating posture base metrics: {e}", exc_info=True)
        
        return metrics
    
    def _calculate_knee_valgus_angles(self, keypoints: Dict[str, Any]) -> Dict[str, float]:
        """
        Calculate knee valgus (knock-knee) and varus (bow-leg) angles from keypoints.
        This is used for both posture analysis and ACL risk assessment.
        
        Args:
            keypoints: Pose keypoints dictionary
            
        Returns:
            Dictionary with valgus/varus angles for left and right knees
        """
        metrics = {}
        
        try:
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            left_knee = keypoints.get("left_knee") or keypoints.get("knee_left")
            right_knee = keypoints.get("right_knee") or keypoints.get("knee_right")
            left_ankle = keypoints.get("left_ankle") or keypoints.get("ankle_left")
            right_ankle = keypoints.get("right_ankle") or keypoints.get("ankle_right")
            
            # Calculate left knee valgus/varus
            if left_knee and left_ankle and left_hip:
                hip_pos = np.array([float(left_hip[0]), float(left_hip[1])])
                knee_pos = np.array([float(left_knee[0]), float(left_knee[1])])
                ankle_pos = np.array([float(left_ankle[0]), float(left_ankle[1])])
                
                # Calculate joint angle at knee (hip-knee-ankle)
                # Use vectors FROM knee TO hip and FROM knee TO ankle
                # This gives us the joint angle at the knee
                knee_hip = hip_pos - knee_pos  # Vector from knee to hip
                knee_ankle = ankle_pos - knee_pos  # Vector from knee to ankle
                
                if np.linalg.norm(knee_hip) > 0 and np.linalg.norm(knee_ankle) > 0:
                    # Calculate joint angle at knee using _calculate_joint_angle method
                    joint_angle = self._calculate_joint_angle(
                        left_hip, left_knee, left_ankle
                    )
                    
                    if joint_angle is not None:
                        # Joint angle: 180° = straight leg, <180° = valgus, >180° = varus
                        # Valgus angle = deviation from 180° (straight line)
                        # For straight leg: joint_angle = 180°, valgus = 0°
                        # For valgus: joint_angle < 180°, valgus = 180° - joint_angle
                        # For varus: joint_angle > 180°, varus = joint_angle - 180°
                        
                        deviation_from_straight = 180.0 - joint_angle
                        
                        if deviation_from_straight > 0:
                            # Valgus (knee collapses inward, joint angle < 180°)
                            metrics["left_knee_valgus_angle"] = deviation_from_straight
                            metrics["left_knee_varus_angle"] = 0.0
                        else:
                            # Varus (knee bows outward, joint angle > 180°)
                            metrics["left_knee_valgus_angle"] = 0.0
                            metrics["left_knee_varus_angle"] = abs(deviation_from_straight)
                        
                        # Detect bowed knee if deviation > 5°
                        metrics["left_knee_bowed_detected"] = 1.0 if abs(deviation_from_straight) > 5.0 else 0.0
                    else:
                        # Fallback: use horizontal deviation
                        hip_knee_horizontal = (knee_pos - hip_pos)[0]
                        knee_ankle_horizontal = (ankle_pos - knee_pos)[0]
                        deviation = hip_knee_horizontal - knee_ankle_horizontal
                        
                        # Rough pixel-to-angle conversion (approximate)
                        # This is a fallback and less accurate
                        if abs(deviation) > 0:
                            # Estimate angle from horizontal pixel deviation
                            # Rough approximation: 1 pixel ≈ 0.1° for typical video resolution
                            estimated_angle = abs(deviation) * 0.1
                            if deviation > 0:
                                metrics["left_knee_valgus_angle"] = estimated_angle
                                metrics["left_knee_varus_angle"] = 0.0
                            else:
                                metrics["left_knee_valgus_angle"] = 0.0
                                metrics["left_knee_varus_angle"] = estimated_angle
                            metrics["left_knee_bowed_detected"] = 1.0 if estimated_angle > 5.0 else 0.0
                        else:
                            metrics["left_knee_valgus_angle"] = 0.0
                            metrics["left_knee_varus_angle"] = 0.0
                            metrics["left_knee_bowed_detected"] = 0.0
                else:
                    metrics["left_knee_valgus_angle"] = 0.0
                    metrics["left_knee_varus_angle"] = 0.0
                    metrics["left_knee_bowed_detected"] = 0.0
            else:
                metrics["left_knee_valgus_angle"] = 0.0
                metrics["left_knee_varus_angle"] = 0.0
                metrics["left_knee_bowed_detected"] = 0.0
            
            # Calculate right knee valgus/varus
            if right_knee and right_ankle and right_hip:
                hip_pos = np.array([float(right_hip[0]), float(right_hip[1])])
                knee_pos = np.array([float(right_knee[0]), float(right_knee[1])])
                ankle_pos = np.array([float(right_ankle[0]), float(right_ankle[1])])
                
                # Calculate joint angle at knee (hip-knee-ankle)
                # Use vectors FROM knee TO hip and FROM knee TO ankle
                knee_hip = hip_pos - knee_pos  # Vector from knee to hip
                knee_ankle = ankle_pos - knee_pos  # Vector from knee to ankle
                
                if np.linalg.norm(knee_hip) > 0 and np.linalg.norm(knee_ankle) > 0:
                    # Calculate joint angle at knee using _calculate_joint_angle method
                    joint_angle = self._calculate_joint_angle(
                        right_hip, right_knee, right_ankle
                    )
                    
                    if joint_angle is not None:
                        # Joint angle: 180° = straight leg, <180° = valgus, >180° = varus
                        # Valgus angle = deviation from 180° (straight line)
                        # For straight leg: joint_angle = 180°, valgus = 0°
                        # For valgus: joint_angle < 180°, valgus = 180° - joint_angle
                        # For varus: joint_angle > 180°, varus = joint_angle - 180°
                        
                        deviation_from_straight = 180.0 - joint_angle
                        
                        if deviation_from_straight > 0:
                            # Valgus (knee collapses inward, joint angle < 180°)
                            metrics["right_knee_valgus_angle"] = deviation_from_straight
                            metrics["right_knee_varus_angle"] = 0.0
                        else:
                            # Varus (knee bows outward, joint angle > 180°)
                            metrics["right_knee_valgus_angle"] = 0.0
                            metrics["right_knee_varus_angle"] = abs(deviation_from_straight)
                        
                        # Detect bowed knee if deviation > 5°
                        metrics["right_knee_bowed_detected"] = 1.0 if abs(deviation_from_straight) > 5.0 else 0.0
                    else:
                        # Fallback: use horizontal deviation
                        hip_knee_horizontal = (knee_pos - hip_pos)[0]
                        knee_ankle_horizontal = (ankle_pos - knee_pos)[0]
                        deviation = hip_knee_horizontal - knee_ankle_horizontal
                        
                        # Rough pixel-to-angle conversion (approximate)
                        if abs(deviation) > 0:
                            estimated_angle = abs(deviation) * 0.1
                            if deviation > 0:
                                metrics["right_knee_valgus_angle"] = estimated_angle
                                metrics["right_knee_varus_angle"] = 0.0
                            else:
                                metrics["right_knee_valgus_angle"] = 0.0
                                metrics["right_knee_varus_angle"] = estimated_angle
                            metrics["right_knee_bowed_detected"] = 1.0 if estimated_angle > 5.0 else 0.0
                        else:
                            metrics["right_knee_valgus_angle"] = 0.0
                            metrics["right_knee_varus_angle"] = 0.0
                            metrics["right_knee_bowed_detected"] = 0.0
                else:
                    metrics["right_knee_valgus_angle"] = 0.0
                    metrics["right_knee_varus_angle"] = 0.0
                    metrics["right_knee_bowed_detected"] = 0.0
            else:
                metrics["right_knee_valgus_angle"] = 0.0
                metrics["right_knee_varus_angle"] = 0.0
                metrics["right_knee_bowed_detected"] = 0.0
            
            # Average values
            if "left_knee_bowed_detected" in metrics and "right_knee_bowed_detected" in metrics:
                metrics["knee_bowed_avg"] = (
                    metrics["left_knee_bowed_detected"] + metrics["right_knee_bowed_detected"]
                ) / 2.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating knee valgus angles: {e}", exc_info=True)
            # Return zeros on error
            metrics["left_knee_valgus_angle"] = 0.0
            metrics["right_knee_valgus_angle"] = 0.0
            metrics["left_knee_varus_angle"] = 0.0
            metrics["right_knee_varus_angle"] = 0.0
        
        return metrics
    
    def _calculate_rotations(
        self,
        keypoints: Dict[str, Any],
        previous_keypoints: Optional[Dict[str, Any]],
        frame_timestamp: Optional[float]
    ) -> Dict[str, float]:
        """
        Calculate number of rotations (flips, spins, twists) from body orientation changes.
        
        Tracks body rotation by monitoring the angle of the shoulder-hip line relative to a reference.
        Counts full 360° rotations.
        """
        metrics = {}
        
        try:
            # Get shoulder and hip positions for body orientation
            left_shoulder = keypoints.get("left_shoulder") or keypoints.get("shoulder_left")
            right_shoulder = keypoints.get("right_shoulder") or keypoints.get("shoulder_right")
            left_hip = keypoints.get("left_hip") or keypoints.get("hip_left")
            right_hip = keypoints.get("right_hip") or keypoints.get("hip_right")
            
            if not all([left_shoulder, right_shoulder, left_hip, right_hip]):
                return {"rotation_count": 0.0, "current_rotation_angle": 0.0}
            
            # Calculate body orientation vector (shoulder-hip line)
            shoulder_center = np.array([
                (float(left_shoulder[0]) + float(right_shoulder[0])) / 2,
                (float(left_shoulder[1]) + float(right_shoulder[1])) / 2
            ])
            hip_center = np.array([
                (float(left_hip[0]) + float(right_hip[0])) / 2,
                (float(left_hip[1]) + float(right_hip[1])) / 2
            ])
            
            # Body orientation vector (from hip to shoulder)
            body_vector = shoulder_center - hip_center
            
            # Calculate rotation angle (0-360 degrees)
            # Angle from vertical (0° = upright, 90° = horizontal, 180° = upside down, 270° = horizontal other way)
            if np.linalg.norm(body_vector) > 0:
                # Normalize
                body_vector_norm = body_vector / np.linalg.norm(body_vector)
                
                # Reference vector (vertical, pointing up)
                vertical_ref = np.array([0, -1])  # Negative Y = up in image coordinates
                
                # Calculate angle using atan2
                angle_rad = np.arctan2(body_vector_norm[1], body_vector_norm[0])
                angle_deg = np.degrees(angle_rad)
                
                # Convert to 0-360 range
                current_rotation_angle = (angle_deg + 90) % 360  # Adjust so 0° = upright
                
                # Track rotation if we have previous frame
                if previous_keypoints is not None and self.rotation_state["previous_rotation_angle"] is not None:
                    prev_angle = self.rotation_state["previous_rotation_angle"]
                    
                    # Calculate angular change
                    angle_diff = current_rotation_angle - prev_angle
                    
                    # Handle wrap-around (crossing 0°/360° boundary)
                    if angle_diff > 180:
                        angle_diff -= 360  # Going backwards through 0
                    elif angle_diff < -180:
                        angle_diff += 360  # Going forwards through 0
                    
                    # Accumulate rotation
                    accumulated_rotation = self.rotation_state.get("accumulated_rotation", 0.0) + angle_diff
                    
                    # Count full rotations (360° = 1 rotation)
                    new_total_rotations = accumulated_rotation / 360.0
                    
                    # Update rotation count if we've completed a full rotation
                    if abs(new_total_rotations) >= 1.0:
                        self.rotation_state["total_rotations"] += int(new_total_rotations)
                        accumulated_rotation = accumulated_rotation % 360.0
                    
                    self.rotation_state["accumulated_rotation"] = accumulated_rotation
                    
                    # Detect rotation direction
                    if abs(angle_diff) > 5.0:  # Minimum threshold to avoid noise
                        if angle_diff > 0:
                            self.rotation_state["rotation_direction"] = 1  # Clockwise
                        else:
                            self.rotation_state["rotation_direction"] = -1  # Counterclockwise
                        
                        self.rotation_state["in_rotation_phase"] = True
                        
                        # Calculate angular velocity (degrees per second)
                        if frame_timestamp and self.previous_timestamp:
                            dt = frame_timestamp - self.previous_timestamp
                            if dt > 0:
                                angular_velocity = abs(angle_diff) / dt
                                metrics["angular_velocity_deg_per_sec"] = angular_velocity
                                
                                # Track peak velocity
                                if angular_velocity > self.rotation_state["peak_rotation_velocity"]:
                                    self.rotation_state["peak_rotation_velocity"] = angular_velocity
                    else:
                        # Check if rotation phase ended
                        if self.rotation_state["in_rotation_phase"] and abs(angle_diff) < 2.0:
                            # Rotation has slowed/stopped
                            self.rotation_state["in_rotation_phase"] = False
                
                # Update state
                self.rotation_state["current_rotation_angle"] = current_rotation_angle
                self.rotation_state["previous_rotation_angle"] = current_rotation_angle
                
                # Return metrics
                metrics["rotation_count"] = self.rotation_state["total_rotations"]
                metrics["current_rotation_angle"] = current_rotation_angle
                metrics["rotation_direction"] = float(self.rotation_state["rotation_direction"])
                metrics["in_rotation_phase"] = 1.0 if self.rotation_state["in_rotation_phase"] else 0.0
                metrics["peak_rotation_velocity_deg_per_sec"] = self.rotation_state["peak_rotation_velocity"]
                
                # Partial rotation (0-1, where 1.0 = full 360°)
                partial_rotation = abs(self.rotation_state.get("accumulated_rotation", 0.0)) / 360.0
                metrics["partial_rotation"] = partial_rotation
                
            else:
                # No valid body vector
                metrics["rotation_count"] = self.rotation_state["total_rotations"]
                metrics["current_rotation_angle"] = 0.0
                
        except Exception as e:
            logger.error(f"❌ Error calculating rotations: {e}")
            metrics = {"rotation_count": self.rotation_state.get("total_rotations", 0.0)}
        
        return metrics
    
    def reset_rotation_state(self):
        """Reset rotation tracking state (call at start of new sequence)"""
        self.rotation_state = {
            "total_rotations": 0.0,
            "current_rotation_angle": 0.0,
            "previous_rotation_angle": 0.0,
            "rotation_direction": 0,
            "in_rotation_phase": False,
            "rotation_start_frame": None,
            "peak_rotation_velocity": 0.0,
            "accumulated_rotation": 0.0
        }
    
    def _calculate_acl_tear_risk(
        self,
        keypoints: Dict[str, Any],
        previous_frame_keypoints: Optional[Dict[str, Any]] = None,
        frame_timestamp: Optional[float] = None,
        existing_metrics: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Calculate ACL tear risk based on multiple biomechanical factors.
        
        ACL injury risk factors:
        1. Knee valgus collapse (knock-knee) - MAJOR risk factor
        2. Insufficient knee flexion on landing (<20-30°)
        3. High impact forces
        4. Landing with straight/stiff knees
        5. Asymmetric landing
        6. Hip internal rotation
        7. Trunk lean
        
        Returns:
            Dictionary with ACL risk metrics and overall risk score (0-1, higher = higher risk)
        """
        metrics = {}
        existing = existing_metrics or {}
        
        try:
            # Get or calculate required metrics
            # 1. Knee valgus angles (knock-knee collapse) - PRIMARY ACL risk factor
            left_valgus = existing.get("left_knee_valgus_angle", 0.0)
            right_valgus = existing.get("right_knee_valgus_angle", 0.0)
            max_valgus = max(left_valgus, right_valgus)
            avg_valgus = (left_valgus + right_valgus) / 2.0 if (left_valgus > 0 or right_valgus > 0) else 0.0
            
            # 2. Landing knee bend angles (insufficient flexion = higher risk)
            # IMPORTANT: Only use landing knee bend angles if we're actually in a landing phase
            # Landing angles should only be calculated during landing, not for every frame
            in_landing_phase = existing.get("in_landing_phase", False)
            
            left_knee_bend = None
            right_knee_bend = None
            min_knee_bend = None
            
            # Only use landing knee bend angles if we're in a landing phase
            if in_landing_phase:
                left_knee_bend = existing.get("landing_knee_bend_left")
                right_knee_bend = existing.get("landing_knee_bend_right")
                min_knee_bend = existing.get("landing_knee_bend_min")
                
                # If landing angles not available but we're in landing phase, calculate them
                if left_knee_bend is None or right_knee_bend is None:
                    landing_metrics = self._calculate_landing_bend_angles(keypoints)
                    left_knee_bend = landing_metrics.get("landing_knee_bend_left")
                    right_knee_bend = landing_metrics.get("landing_knee_bend_right")
                    min_knee_bend = landing_metrics.get("landing_knee_bend_min")
            else:
                # Not in landing phase - don't use landing knee bend angles for ACL risk
                # Use general knee angles instead if needed, or skip flexion risk factor
                pass
            
            # 3. Impact force
            impact_force = existing.get("impact_force_N", 0.0)
            if impact_force == 0.0 and previous_frame_keypoints:
                impact_metrics = self._calculate_impact_force(
                    keypoints, previous_frame_keypoints, frame_timestamp
                )
                impact_force = impact_metrics.get("impact_force_N", 0.0)
            
            # 4. Landing velocity
            landing_velocity = existing.get("landing_velocity_ms", 0.0)
            
            # Calculate ACL risk factors
            risk_factors = {}
            
            # Factor 1: Valgus collapse (0-1, higher = more risk)
            # Valgus >10° = high risk, >5° = moderate risk
            if max_valgus > 10.0:
                valgus_risk = 1.0  # Maximum risk
            elif max_valgus > 5.0:
                valgus_risk = 0.5 + (max_valgus - 5.0) / 10.0  # 0.5 to 1.0
            elif max_valgus > 0.0:
                valgus_risk = max_valgus / 10.0  # 0.0 to 0.5
            else:
                valgus_risk = 0.0
            risk_factors["valgus_collapse_risk"] = valgus_risk
            metrics["acl_valgus_risk"] = valgus_risk
            
            # Factor 2: Insufficient knee flexion (0-1, higher = more risk)
            # NOTE: Only assess knee flexion risk during landing phases
            # Landing knee bend angles should only be calculated/used during actual landings
            # NOTE: knee_bend angles are joint angles where 180° = straight, lower = more bent
            # To get flexion degrees: flexion = 180 - knee_bend_angle
            # FIG standards for gymnastics landings: 30-45° flexion is ideal, 20-30° is acceptable, <20° is risky
            # Convert joint angle to flexion degrees for ACL risk assessment
            if min_knee_bend is not None and in_landing_phase:
                # Convert joint angle to flexion degrees (180° = straight = 0° flexion)
                flexion_degrees = 180.0 - min_knee_bend
                
                # ACL risk based on flexion degrees (aligned with FIG standards and biomechanical research):
                # Ideal: 30-45° flexion (135-150° joint angle) - excellent shock absorption, zero ACL risk
                # Acceptable: 20-30° flexion (150-160° joint angle) - adequate shock absorption, zero ACL risk per FIG
                # Risky: 10-20° flexion (160-170° joint angle) - insufficient shock absorption, moderate risk
                # High risk: <10° flexion (>170° joint angle) - very stiff landing, high ACL risk
                if flexion_degrees < 10.0:
                    flexion_risk = 1.0  # Very high risk (stiff landing, <10° flexion)
                elif flexion_degrees < 20.0:
                    flexion_risk = 0.5 + (20.0 - flexion_degrees) / 20.0  # 0.5 to 1.0 (10-20° flexion, risky)
                elif flexion_degrees <= 45.0:
                    # 20-45° flexion per FIG standards is acceptable/ideal - zero ACL risk
                    flexion_risk = 0.0  # Acceptable to ideal range (20-45° flexion per FIG standards)
                else:
                    # >45° flexion (excessive bend) - may indicate other issues but not ACL risk
                    flexion_risk = 0.0  # Not a risk factor for ACL (though may indicate other concerns)
                
                risk_factors["insufficient_flexion_risk"] = flexion_risk
                metrics["acl_insufficient_flexion_risk"] = flexion_risk
                # Store flexion degrees for reference
                metrics["acl_knee_flexion_degrees"] = flexion_degrees
            else:
                risk_factors["insufficient_flexion_risk"] = 0.0
                metrics["acl_insufficient_flexion_risk"] = 0.0
            
            # Factor 3: High impact force (0-1, higher = more risk)
            # Normal landing: ~1000-2000N, High risk: >3000N, Critical: >4000N
            if impact_force > 4000.0:
                impact_risk = 1.0  # Critical
            elif impact_force > 3000.0:
                impact_risk = 0.7 + (impact_force - 3000.0) / 2000.0  # 0.7 to 1.0
            elif impact_force > 2000.0:
                impact_risk = 0.3 + (impact_force - 2000.0) / 2000.0  # 0.3 to 0.7
            elif impact_force > 1000.0:
                impact_risk = (impact_force - 1000.0) / 2000.0  # 0.0 to 0.3
            else:
                impact_risk = 0.0
            risk_factors["high_impact_risk"] = impact_risk
            metrics["acl_high_impact_risk"] = impact_risk
            
            # Factor 4: Asymmetric landing (0-1, higher = more risk)
            if left_knee_bend is not None and right_knee_bend is not None:
                asymmetry = abs(left_knee_bend - right_knee_bend)
                if asymmetry > 15.0:
                    asymmetry_risk = 1.0  # High asymmetry
                elif asymmetry > 10.0:
                    asymmetry_risk = 0.5 + (asymmetry - 10.0) / 10.0  # 0.5 to 1.0
                elif asymmetry > 5.0:
                    asymmetry_risk = asymmetry / 10.0  # 0.0 to 0.5
                else:
                    asymmetry_risk = 0.0
                risk_factors["asymmetric_landing_risk"] = asymmetry_risk
                metrics["acl_asymmetric_landing_risk"] = asymmetry_risk
                metrics["acl_landing_asymmetry_degrees"] = asymmetry
            else:
                risk_factors["asymmetric_landing_risk"] = 0.0
                metrics["acl_asymmetric_landing_risk"] = 0.0
            
            # Calculate overall ACL risk score (weighted combination)
            # Valgus collapse is the PRIMARY risk factor (weight: 0.4)
            # Insufficient flexion (weight: 0.3)
            # High impact (weight: 0.2)
            # Asymmetric landing (weight: 0.1)
            
            acl_risk_score = (
                valgus_risk * 0.4 +
                risk_factors.get("insufficient_flexion_risk", 0.0) * 0.3 +
                impact_risk * 0.2 +
                risk_factors.get("asymmetric_landing_risk", 0.0) * 0.1
            )
            
            # Clamp to 0-1
            acl_risk_score = max(0.0, min(1.0, acl_risk_score))
            metrics["acl_tear_risk_score"] = acl_risk_score
            
            # Risk level classification
            if acl_risk_score >= 0.7:
                metrics["acl_risk_level"] = "HIGH"
            elif acl_risk_score >= 0.4:
                metrics["acl_risk_level"] = "MODERATE"
            elif acl_risk_score >= 0.2:
                metrics["acl_risk_level"] = "LOW"
            else:
                metrics["acl_risk_level"] = "MINIMAL"
            
            # Store individual risk factors
            metrics["acl_max_valgus_angle"] = max_valgus
            metrics["acl_avg_valgus_angle"] = avg_valgus
            metrics["acl_min_knee_flexion"] = min_knee_bend if min_knee_bend is not None else 0.0
            metrics["acl_impact_force_N"] = impact_force
            
            # Risk flags
            metrics["acl_high_valgus_detected"] = 1.0 if max_valgus > 10.0 else 0.0
            metrics["acl_stiff_landing_detected"] = 1.0 if (min_knee_bend is not None and min_knee_bend < 20.0) else 0.0
            metrics["acl_high_impact_detected"] = 1.0 if impact_force > 3000.0 else 0.0
            
            # Flag ONLY if ACL risk is HIGH (>= 0.7) - do not flag MODERATE or LOW
            # Only flag HIGH risk for timestep tracking and reporting
            if acl_risk_score >= 0.7:  # HIGH risk only
                metrics["acl_risk_flagged"] = 1.0
                # Store timestamp for this flagged frame
                if frame_timestamp is not None:
                    metrics["acl_risk_timestamp"] = frame_timestamp
            else:
                metrics["acl_risk_flagged"] = 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating ACL tear risk: {e}", exc_info=True)
            metrics["acl_tear_risk_score"] = 0.0
            metrics["acl_risk_level"] = "UNKNOWN"
            metrics["acl_risk_flagged"] = 0.0
        
        return metrics
    
    def _calculate_joint_angle(
        self,
        point1: Optional[List[float]],
        point2: Optional[List[float]],
        point3: Optional[List[float]]
    ) -> Optional[float]:
        """Calculate angle at point2 formed by point1-point2-point3"""
        if not all([point1, point2, point3]):
            return None
        
        try:
            vec1 = np.array(point1) - np.array(point2)
            vec2 = np.array(point3) - np.array(point2)
            
            if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
                return None
            
            cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            
            return angle
        except Exception as e:
            logger.debug(f"Error calculating joint angle: {e}")
            return None






















