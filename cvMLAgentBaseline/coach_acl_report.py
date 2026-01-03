#!/usr/bin/env python3
"""
Coach-Friendly ACL Risk Report Generator

Generates comprehensive, coach-friendly reports for ACL injury risk analysis.
Focuses on:
- Alerting coaches to high-risk biomechanics
- Pinpointing exactly where and why injuries can occur
- Recommending evidence-backed corrections
- Validating whether risk is actually reduced over time
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class CoachACLReportGenerator:
    """
    Generates coach-friendly ACL risk reports with actionable insights.
    """
    
    # Evidence-backed correction recommendations
    CORRECTION_GUIDE = {
        "knee_valgus": {
            "name": "Knee Valgus Collapse",
            "description": "Knee collapsing inward (knock-knee pattern) during landing",
            "why_dangerous": "Valgus collapse places excessive stress on the ACL, particularly the anteromedial bundle. Research shows valgus angles >10° during landing increase ACL injury risk by 2-3x.",
            "evidence": "Hewett et al. (2005) found that athletes with valgus collapse had 2.5x higher ACL injury rates. Biomechanical studies show valgus moments >0.5 Nm/kg increase ACL strain.",
            "corrections": [
                {
                    "priority": 1,
                    "intervention": "Hip strengthening (gluteus medius focus)",
                    "rationale": "Weak hip abductors allow femoral adduction, causing knee valgus",
                    "exercises": [
                        "Side-lying clamshells (3 sets × 15 reps)",
                        "Lateral band walks (3 sets × 20 steps)",
                        "Single-leg glute bridges (3 sets × 12 reps each leg)",
                        "Monster walks with resistance band (3 sets × 15 steps)"
                    ],
                    "frequency": "Daily for 4-6 weeks, then 3x/week maintenance",
                    "expected_improvement": "Reduce valgus angle by 5-8° within 6-8 weeks"
                },
                {
                    "priority": 2,
                    "intervention": "Landing technique retraining",
                    "rationale": "Proper landing mechanics reduce valgus moments",
                    "exercises": [
                        "Drop landing drills (focus on soft, controlled landings)",
                        "Single-leg landing practice (emphasize knee over toe alignment)",
                        "Plyometric landing progression (start low, increase height gradually)",
                        "Video feedback sessions (compare before/after landing mechanics)"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Improve landing alignment and reduce valgus by 3-5°"
                },
                {
                    "priority": 3,
                    "intervention": "Ankle mobility and proprioception",
                    "rationale": "Limited ankle dorsiflexion can contribute to compensatory knee valgus",
                    "exercises": [
                        "Ankle dorsiflexion stretches (3 sets × 30 seconds)",
                        "Single-leg balance on unstable surface (3 sets × 30 seconds)",
                        "Ankle mobility drills (calf raises, ankle circles)",
                        "Proprioceptive training (balance board, BOSU ball)"
                    ],
                    "frequency": "Daily for 4 weeks, then 3x/week",
                    "expected_improvement": "Improve ankle mobility and reduce compensatory valgus"
                }
            ]
        },
        "insufficient_flexion": {
            "name": "Insufficient Knee Flexion (Stiff Landing)",
            "description": "Landing with knees too straight (<20° flexion)",
            "why_dangerous": "Stiff landings increase ground reaction forces and reduce shock absorption. Research shows knee flexion <20° during landing increases ACL strain by 40-60%.",
            "evidence": "Yu et al. (2006) demonstrated that landing with <20° knee flexion increases ACL loading. Biomechanical models show optimal landing requires 30-45° knee flexion for shock absorption.",
            "corrections": [
                {
                    "priority": 1,
                    "intervention": "Eccentric quadriceps and hamstring strengthening",
                    "rationale": "Strong eccentric control allows controlled knee flexion during landing",
                    "exercises": [
                        "Eccentric squats (slow descent, 3 seconds down) (3 sets × 12 reps)",
                        "Nordic hamstring curls (3 sets × 8 reps)",
                        "Single-leg Romanian deadlifts (3 sets × 10 reps each leg)",
                        "Eccentric step-downs (3 sets × 12 reps each leg)"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Increase landing knee flexion by 10-15° within 6-8 weeks"
                },
                {
                    "priority": 2,
                    "intervention": "Landing mechanics retraining",
                    "rationale": "Teaching proper landing technique with increased knee flexion",
                    "exercises": [
                        "Soft landing drills (focus on 'sitting back' into landing)",
                        "Progressive height drops (start at 6 inches, progress to 12-18 inches)",
                        "Landing with visual cues ('land soft like a cat', 'bend your knees')",
                        "Video feedback comparing stiff vs. soft landings"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Achieve 30-45° knee flexion during landing within 8 weeks"
                },
                {
                    "priority": 3,
                    "intervention": "Ankle and hip mobility",
                    "rationale": "Limited mobility at other joints can force stiff knee landing",
                    "exercises": [
                        "Hip flexor stretches (3 sets × 30 seconds)",
                        "Ankle dorsiflexion mobility (3 sets × 30 seconds)",
                        "Dynamic warm-up with full range of motion",
                        "Foam rolling for quadriceps and hip flexors"
                    ],
                    "frequency": "Daily before training",
                    "expected_improvement": "Improve overall landing mechanics and reduce stiffness"
                }
            ]
        },
        "high_impact": {
            "name": "High Impact Force",
            "description": "Excessive ground reaction forces during landing (>3000N)",
            "why_dangerous": "High impact forces increase ACL loading. Forces >3000N during landing significantly increase ACL injury risk, especially when combined with poor landing mechanics.",
            "evidence": "Griffin et al. (2000) found that landing forces >3x body weight increase ACL injury risk. Biomechanical studies show impact forces correlate with ACL strain.",
            "corrections": [
                {
                    "priority": 1,
                    "intervention": "Landing technique optimization",
                    "rationale": "Proper landing mechanics reduce impact forces through better shock absorption",
                    "exercises": [
                        "Soft landing progression (focus on quiet, controlled landings)",
                        "Landing with increased knee and hip flexion (30-45° knee, 30° hip)",
                        "Progressive height drops with focus on force reduction",
                        "Video feedback showing impact force reduction"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Reduce impact forces by 20-30% within 6-8 weeks"
                },
                {
                    "priority": 2,
                    "intervention": "Eccentric strength training",
                    "rationale": "Strong eccentric control allows gradual force absorption",
                    "exercises": [
                        "Eccentric squats (3 sets × 12 reps)",
                        "Depth jumps with focus on soft landing (3 sets × 8 reps)",
                        "Single-leg landing drills (3 sets × 10 reps each leg)",
                        "Plyometric progression (start low intensity, increase gradually)"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Improve force absorption capacity and reduce peak impact"
                },
                {
                    "priority": 3,
                    "intervention": "Core stability and proprioception",
                    "rationale": "Strong core and good proprioception improve landing control",
                    "exercises": [
                        "Plank variations (3 sets × 30-60 seconds)",
                        "Single-leg balance on unstable surface (3 sets × 30 seconds)",
                        "Landing with eyes closed (advanced, supervised)",
                        "Reactive balance training"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Improve landing control and reduce impact variability"
                }
            ]
        },
        "asymmetric_landing": {
            "name": "Asymmetric Landing",
            "description": "Significant difference between left and right leg landing mechanics",
            "why_dangerous": "Asymmetric landing patterns increase injury risk on the weaker side and indicate muscle imbalances or movement compensations.",
            "evidence": "Paterno et al. (2010) found that athletes with asymmetric landing patterns had 2.5x higher ACL injury risk. Bilateral differences >10° in knee flexion or >5° in valgus indicate increased risk.",
            "corrections": [
                {
                    "priority": 1,
                    "intervention": "Unilateral strength training (focus on weaker side)",
                    "rationale": "Address muscle imbalances that cause asymmetric movement",
                    "exercises": [
                        "Single-leg squats (3 sets × 12 reps each leg, focus on weaker side)",
                        "Single-leg Romanian deadlifts (3 sets × 10 reps each leg)",
                        "Single-leg glute bridges (3 sets × 15 reps each leg)",
                        "Unilateral plyometric training (equal volume both sides)"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Reduce asymmetry by 50% within 6-8 weeks"
                },
                {
                    "priority": 2,
                    "intervention": "Bilateral landing retraining",
                    "rationale": "Teach symmetric landing mechanics",
                    "exercises": [
                        "Bilateral landing drills with focus on equal weight distribution",
                        "Mirror feedback during landing practice",
                        "Video analysis comparing left vs. right landing",
                        "Progressive drills emphasizing symmetry"
                    ],
                    "frequency": "3x/week for 6-8 weeks",
                    "expected_improvement": "Achieve <5° difference between sides within 8 weeks"
                }
            ]
        }
    }
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    def generate_report(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]],
        landing_phases: List[Dict[str, Any]],
        metrics: Dict[str, Any],
        activity: str,
        technique: str,
        athlete_name: Optional[str] = None,
        session_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive coach-friendly ACL risk report.
        
        Args:
            acl_flagged_timesteps: List of HIGH risk flagged timesteps
            landing_phases: List of detected landing phases
            metrics: Overall metrics dictionary
            activity: Activity type (e.g., "gymnastics")
            technique: Technique name (e.g., "back handspring")
            athlete_name: Optional athlete name
            session_date: Optional session date
        
        Returns:
            Dictionary containing the full report
        """
        report = {
            "report_type": "ACL Injury Risk Assessment",
            "generated_at": datetime.utcnow().isoformat(),
            "athlete_name": athlete_name or "Athlete",
            "session_date": session_date or datetime.utcnow().strftime("%Y-%m-%d"),
            "activity": activity,
            "technique": technique,
            "executive_summary": self._generate_executive_summary(acl_flagged_timesteps, metrics),
            "high_risk_alerts": self._generate_high_risk_alerts(acl_flagged_timesteps, landing_phases),
            "biomechanical_analysis": self._generate_biomechanical_analysis(acl_flagged_timesteps),
            "injury_location_analysis": self._generate_injury_location_analysis(acl_flagged_timesteps),
            "evidence_backed_corrections": self._generate_corrections(acl_flagged_timesteps),
            "risk_reduction_plan": self._generate_risk_reduction_plan(acl_flagged_timesteps),
            "tracking_metrics": self._generate_tracking_metrics(acl_flagged_timesteps, metrics),
            "video_evidence": self._generate_video_evidence_summary(acl_flagged_timesteps)
        }
        
        return report
    
    def _generate_executive_summary(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate executive summary of ACL risk."""
        total_high_risk = len(acl_flagged_timesteps)
        
        if total_high_risk == 0:
            return {
                "overall_risk": "LOW",
                "summary": "No HIGH risk ACL injury moments detected during this session.",
                "recommendation": "Continue current training with focus on maintaining proper landing mechanics."
            }
        
        # Calculate average risk score
        avg_risk_score = sum(ts.get("risk_score", 0.0) for ts in acl_flagged_timesteps) / total_high_risk
        
        # Identify most common risk factors
        risk_factors = defaultdict(int)
        for ts in acl_flagged_timesteps:
            for factor in ts.get("primary_risk_factors", []):
                if "valgus" in factor.lower():
                    risk_factors["knee_valgus"] += 1
                elif "flexion" in factor.lower():
                    risk_factors["insufficient_flexion"] += 1
                elif "impact" in factor.lower():
                    risk_factors["high_impact"] += 1
        
        most_common = max(risk_factors.items(), key=lambda x: x[1]) if risk_factors else ("unknown", 0)
        
        return {
            "overall_risk": "HIGH",
            "total_high_risk_moments": total_high_risk,
            "average_risk_score": round(avg_risk_score, 2),
            "most_common_risk_factor": most_common[0],
            "summary": f"⚠️ **HIGH RISK ALERT**: {total_high_risk} high-risk ACL injury moments detected during this session. Average risk score: {avg_risk_score:.2f}. Immediate intervention recommended.",
            "urgency": "IMMEDIATE ACTION REQUIRED" if total_high_risk >= 3 else "ACTION RECOMMENDED",
            "recommendation": f"Focus on correcting {self.CORRECTION_GUIDE.get(most_common[0], {}).get('name', 'identified risk factors')} through targeted interventions."
        }
    
    def _generate_high_risk_alerts(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]],
        landing_phases: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate detailed alerts for each high-risk moment."""
        alerts = []
        
        for i, ts in enumerate(acl_flagged_timesteps, 1):
            alert = {
                "alert_number": i,
                "timestamp": ts.get("timestamp", 0.0),
                "frame_number": ts.get("frame_number", 0),
                "risk_score": ts.get("risk_score", 0.0),
                "risk_level": ts.get("risk_level", "HIGH"),
                "location": f"Frame {ts.get('frame_number', 0)} at {ts.get('timestamp', 0.0):.2f}s",
                "primary_risk_factors": ts.get("primary_risk_factors", []),
                "biomechanical_values": {
                    "valgus_angle": ts.get("valgus_angle", 0.0),
                    "knee_flexion": ts.get("knee_flexion", 0.0),
                    "impact_force_N": ts.get("impact_force", 0.0)
                },
                "landing_context": ts.get("landing_context", {}),
                "why_dangerous": self._explain_why_dangerous(ts),
                "immediate_recommendation": self._get_immediate_recommendation(ts)
            }
            alerts.append(alert)
        
        return alerts
    
    def _explain_why_dangerous(self, timestep: Dict[str, Any]) -> str:
        """Explain why this specific moment is dangerous."""
        factors = timestep.get("primary_risk_factors", [])
        explanations = []
        
        for factor in factors:
            if "valgus" in factor.lower():
                explanations.append(
                    "Knee valgus collapse places excessive stress on the ACL's anteromedial bundle. "
                    "Research shows valgus angles >10° increase ACL injury risk by 2-3x."
                )
            elif "flexion" in factor.lower():
                explanations.append(
                    "Insufficient knee flexion (<20°) creates a 'stiff landing' that increases ground reaction forces. "
                    "This reduces shock absorption and increases ACL strain by 40-60%."
                )
            elif "impact" in factor.lower():
                explanations.append(
                    "High impact forces (>3000N) significantly increase ACL loading. "
                    "Forces >3x body weight during landing are associated with increased injury risk."
                )
        
        if not explanations:
            return "Multiple risk factors combine to create a dangerous landing pattern."
        
        return " ".join(explanations)
    
    def _get_immediate_recommendation(self, timestep: Dict[str, Any]) -> str:
        """Get immediate recommendation for this specific moment."""
        factors = timestep.get("primary_risk_factors", [])
        
        if any("valgus" in f.lower() for f in factors):
            return "Immediate focus: Hip strengthening (gluteus medius) and landing technique retraining to reduce knee valgus."
        elif any("flexion" in f.lower() for f in factors):
            return "Immediate focus: Eccentric strength training and landing mechanics to increase knee flexion during landing."
        elif any("impact" in f.lower() for f in factors):
            return "Immediate focus: Landing technique optimization to reduce impact forces through better shock absorption."
        else:
            return "Immediate focus: Comprehensive landing mechanics retraining addressing all identified risk factors."
    
    def _generate_biomechanical_analysis(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate detailed biomechanical analysis."""
        if not acl_flagged_timesteps:
            return {"message": "No high-risk moments to analyze."}
        
        # Aggregate biomechanical data
        valgus_angles = [ts.get("valgus_angle", 0.0) for ts in acl_flagged_timesteps]
        knee_flexions = [ts.get("knee_flexion", 0.0) for ts in acl_flagged_timesteps]
        impact_forces = [ts.get("impact_force", 0.0) for ts in acl_flagged_timesteps]
        
        analysis = {
            "knee_valgus_analysis": {
                "average_valgus": round(sum(valgus_angles) / len(valgus_angles), 1) if valgus_angles else 0.0,
                "max_valgus": round(max(valgus_angles), 1) if valgus_angles else 0.0,
                "threshold": 10.0,
                "interpretation": "Valgus angles >10° indicate dangerous knee collapse pattern" if any(v > 10.0 for v in valgus_angles) else "Valgus angles within acceptable range",
                "biomechanical_explanation": "Knee valgus occurs when the femur adducts and internally rotates, causing the knee to collapse inward. This places excessive stress on the ACL, particularly during landing when forces are highest."
            },
            "knee_flexion_analysis": {
                "average_flexion": round(sum(knee_flexions) / len(knee_flexions), 1) if knee_flexions else 0.0,
                "min_flexion": round(min(knee_flexions), 1) if knee_flexions else 0.0,
                "optimal_range": "30-45°",
                "interpretation": "Knee flexion <20° indicates stiff landing with poor shock absorption" if any(f < 20.0 for f in knee_flexions) else "Knee flexion within acceptable range",
                "biomechanical_explanation": "Adequate knee flexion (30-45°) during landing allows the quadriceps and hamstrings to eccentrically absorb impact forces. Insufficient flexion increases ground reaction forces and ACL strain."
            },
            "impact_force_analysis": {
                "average_impact": round(sum(impact_forces) / len(impact_forces), 0) if impact_forces else 0.0,
                "max_impact": round(max(impact_forces), 0) if impact_forces else 0.0,
                "threshold": 3000.0,
                "interpretation": "Impact forces >3000N indicate excessive landing forces" if any(f > 3000.0 for f in impact_forces) else "Impact forces within acceptable range",
                "biomechanical_explanation": "High impact forces during landing increase ACL loading. Proper landing mechanics with increased knee and hip flexion can reduce impact forces by 20-30%."
            }
        }
        
        return analysis
    
    def _generate_injury_location_analysis(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Pinpoint exactly where and why injuries can occur."""
        if not acl_flagged_timesteps:
            return {"message": "No high-risk moments detected."}
        
        locations = []
        for ts in acl_flagged_timesteps:
            location = {
                "exact_location": f"Frame {ts.get('frame_number', 0)} at {ts.get('timestamp', 0.0):.2f}s",
                "phase": ts.get("landing_context", {}).get("phase_type", "unknown"),
                "risk_factors": ts.get("primary_risk_factors", []),
                "why_injury_occurs_here": self._explain_injury_mechanism(ts),
                "video_clip_available": ts.get("video_clip_path") is not None
            }
            locations.append(location)
        
        return {
            "total_high_risk_locations": len(locations),
            "locations": locations,
            "pattern_analysis": self._analyze_patterns(acl_flagged_timesteps)
        }
    
    def _explain_injury_mechanism(self, timestep: Dict[str, Any]) -> str:
        """Explain the specific injury mechanism at this moment."""
        factors = timestep.get("primary_risk_factors", [])
        phase = timestep.get("landing_context", {}).get("phase_type", "unknown")
        
        explanation = f"During the {phase} phase of landing, "
        
        if any("valgus" in f.lower() for f in factors):
            explanation += "knee valgus collapse creates a combined loading pattern (valgus + internal rotation + anterior translation) that places maximum stress on the ACL. "
        
        if any("flexion" in f.lower() for f in factors):
            explanation += "Insufficient knee flexion prevents proper shock absorption, increasing ground reaction forces. "
        
        if any("impact" in f.lower() for f in factors):
            explanation += "High impact forces combined with poor landing mechanics create a 'perfect storm' for ACL injury. "
        
        explanation += "This combination of factors at this exact moment creates the highest risk for ACL injury."
        
        return explanation
    
    def _analyze_patterns(self, timestep_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across high-risk moments."""
        if len(timestep_list) < 2:
            return {"message": "Insufficient data for pattern analysis."}
        
        # Check if risks occur in clusters
        timestamps = sorted([ts.get("timestamp", 0.0) for ts in timestep_list])
        clusters = []
        current_cluster = [timestamps[0]]
        
        for i in range(1, len(timestamps)):
            if timestamps[i] - timestamps[i-1] < 1.0:  # Within 1 second
                current_cluster.append(timestamps[i])
            else:
                if len(current_cluster) > 1:
                    clusters.append(current_cluster)
                current_cluster = [timestamps[i]]
        
        if len(current_cluster) > 1:
            clusters.append(current_cluster)
        
        return {
            "risk_clustering": len(clusters) > 0,
            "number_of_clusters": len(clusters),
            "interpretation": "High-risk moments occur in clusters, indicating consistent landing pattern issues" if clusters else "High-risk moments are scattered, indicating intermittent technique breakdown",
            "recommendation": "Focus on consistent landing mechanics training" if clusters else "Focus on identifying triggers for technique breakdown"
        }
    
    def _generate_corrections(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate evidence-backed correction recommendations."""
        if not acl_flagged_timesteps:
            return {"message": "No corrections needed - no high-risk moments detected."}
        
        # Identify which risk factors are present
        risk_factors_present = set()
        for ts in acl_flagged_timesteps:
            for factor in ts.get("primary_risk_factors", []):
                if "valgus" in factor.lower():
                    risk_factors_present.add("knee_valgus")
                elif "flexion" in factor.lower():
                    risk_factors_present.add("insufficient_flexion")
                elif "impact" in factor.lower():
                    risk_factors_present.add("high_impact")
                elif "asymmetric" in factor.lower() or "asymmetry" in factor.lower():
                    risk_factors_present.add("asymmetric_landing")
        
        corrections = {}
        for factor_key in risk_factors_present:
            if factor_key in self.CORRECTION_GUIDE:
                guide = self.CORRECTION_GUIDE[factor_key]
                corrections[factor_key] = {
                    "risk_factor_name": guide["name"],
                    "description": guide["description"],
                    "why_dangerous": guide["why_dangerous"],
                    "evidence": guide["evidence"],
                    "interventions": guide["corrections"]
                }
        
        return {
            "identified_risk_factors": list(risk_factors_present),
            "corrections": corrections,
            "implementation_priority": "Address all identified risk factors, starting with Priority 1 interventions"
        }
    
    def _generate_risk_reduction_plan(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate a structured risk reduction plan."""
        if not acl_flagged_timesteps:
            return {"message": "No risk reduction plan needed."}
        
        # Count occurrences of each risk factor
        risk_factor_counts = defaultdict(int)
        for ts in acl_flagged_timesteps:
            for factor in ts.get("primary_risk_factors", []):
                if "valgus" in factor.lower():
                    risk_factor_counts["knee_valgus"] += 1
                elif "flexion" in factor.lower():
                    risk_factor_counts["insufficient_flexion"] += 1
                elif "impact" in factor.lower():
                    risk_factor_counts["high_impact"] += 1
        
        # Prioritize based on frequency
        prioritized_factors = sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)
        
        plan = {
            "timeline": "6-8 weeks for initial improvements, 12-16 weeks for significant risk reduction",
            "phases": [
                {
                    "phase": "Phase 1: Immediate Intervention (Weeks 1-2)",
                    "focus": f"Address most common risk factor: {prioritized_factors[0][0] if prioritized_factors else 'identified risk factors'}",
                    "actions": [
                        "Begin Priority 1 interventions for most common risk factor",
                        "Reduce training intensity/volume if risk is very high",
                        "Implement daily corrective exercises",
                        "Video analysis and feedback sessions"
                    ]
                },
                {
                    "phase": "Phase 2: Comprehensive Training (Weeks 3-6)",
                    "focus": "Address all identified risk factors",
                    "actions": [
                        "Continue Priority 1 interventions",
                        "Add Priority 2 interventions",
                        "Progressive landing mechanics retraining",
                        "Weekly progress assessments"
                    ]
                },
                {
                    "phase": "Phase 3: Integration & Maintenance (Weeks 7-12)",
                    "focus": "Integrate corrections into sport-specific movements",
                    "actions": [
                        "Sport-specific landing practice",
                        "Gradual return to full training intensity",
                        "Maintenance exercises 3x/week",
                        "Monthly reassessment"
                    ]
                }
            ],
            "success_metrics": {
                "valgus_angle": "Reduce to <10° (target: <5°)",
                "knee_flexion": "Increase to 30-45° during landing",
                "impact_force": "Reduce by 20-30%",
                "high_risk_moments": "Reduce to 0 high-risk moments per session"
            },
            "reassessment_schedule": "Reassess ACL risk every 4 weeks to track progress"
        }
        
        return plan
    
    def _generate_tracking_metrics(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]],
        metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate metrics for tracking risk reduction over time."""
        tracking = {
            "baseline_metrics": {
                "total_high_risk_moments": len(acl_flagged_timesteps),
                "average_risk_score": round(
                    sum(ts.get("risk_score", 0.0) for ts in acl_flagged_timesteps) / len(acl_flagged_timesteps),
                    2
                ) if acl_flagged_timesteps else 0.0,
                "max_risk_score": round(
                    max(ts.get("risk_score", 0.0) for ts in acl_flagged_timesteps),
                    2
                ) if acl_flagged_timesteps else 0.0
            },
            "biomechanical_baselines": {
                "average_valgus_angle": round(
                    sum(ts.get("valgus_angle", 0.0) for ts in acl_flagged_timesteps) / len(acl_flagged_timesteps),
                    1
                ) if acl_flagged_timesteps else 0.0,
                "average_knee_flexion": round(
                    sum(ts.get("knee_flexion", 0.0) for ts in acl_flagged_timesteps) / len(acl_flagged_timesteps),
                    1
                ) if acl_flagged_timesteps else 0.0,
                "average_impact_force": round(
                    sum(ts.get("impact_force", 0.0) for ts in acl_flagged_timesteps) / len(acl_flagged_timesteps),
                    0
                ) if acl_flagged_timesteps else 0.0
            },
            "target_metrics": {
                "target_high_risk_moments": 0,
                "target_risk_score": "<0.4 (MODERATE or lower)",
                "target_valgus_angle": "<10° (ideally <5°)",
                "target_knee_flexion": "30-45°",
                "target_impact_force": "20-30% reduction from baseline"
            },
            "tracking_recommendations": [
                "Reassess ACL risk every 4 weeks using the same analysis",
                "Compare baseline metrics to reassessment metrics",
                "Track progress on each biomechanical parameter",
                "Document any changes in training or technique",
                "Use video clips to visually compare before/after landing mechanics"
            ]
        }
        
        return tracking
    
    def _generate_video_evidence_summary(
        self,
        acl_flagged_timesteps: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate summary of available video evidence."""
        video_clips = []
        for ts in acl_flagged_timesteps:
            if ts.get("video_clip_path"):
                video_clips.append({
                    "frame_number": ts.get("frame_number", 0),
                    "timestamp": ts.get("timestamp", 0.0),
                    "clip_path": ts.get("video_clip_path"),
                    "risk_score": ts.get("risk_score", 0.0)
                })
        
        return {
            "total_video_clips": len(video_clips),
            "clips_available": len(video_clips) > 0,
            "clips": video_clips,
            "usage": "Video clips can be used for: visual feedback, technique comparison, progress tracking, and educational purposes"
        }
    
    def save_report(
        self,
        report: Dict[str, Any],
        output_dir: Path,
        base_name: str
    ) -> Path:
        """
        Save the report to a JSON file.
        
        Args:
            report: The report dictionary
            output_dir: Directory to save the report
            base_name: Base name for the file
        
        Returns:
            Path to the saved report file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{base_name}_acl_risk_report.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Coach-friendly ACL risk report saved to: {report_file}")
        return report_file
    
    def generate_text_summary(self, report: Dict[str, Any]) -> str:
        """
        Generate a human-readable text summary of the report.
        
        Args:
            report: The report dictionary
        
        Returns:
            Formatted text summary
        """
        lines = []
        lines.append("=" * 80)
        lines.append("ACL INJURY RISK ASSESSMENT REPORT")
        lines.append("=" * 80)
        lines.append(f"Athlete: {report.get('athlete_name', 'N/A')}")
        lines.append(f"Session Date: {report.get('session_date', 'N/A')}")
        lines.append(f"Activity: {report.get('activity', 'N/A')}")
        lines.append(f"Technique: {report.get('technique', 'N/A')}")
        lines.append(f"Generated: {report.get('generated_at', 'N/A')}")
        lines.append("")
        
        # Executive Summary
        exec_summary = report.get("executive_summary", {})
        lines.append("EXECUTIVE SUMMARY")
        lines.append("-" * 80)
        lines.append(f"Overall Risk: {exec_summary.get('overall_risk', 'N/A')}")
        if exec_summary.get("total_high_risk_moments", 0) > 0:
            lines.append(f"High-Risk Moments: {exec_summary.get('total_high_risk_moments', 0)}")
            lines.append(f"Average Risk Score: {exec_summary.get('average_risk_score', 0.0)}")
            lines.append(f"Urgency: {exec_summary.get('urgency', 'N/A')}")
        lines.append(f"Summary: {exec_summary.get('summary', 'N/A')}")
        lines.append(f"Recommendation: {exec_summary.get('recommendation', 'N/A')}")
        lines.append("")
        
        # High-Risk Alerts
        alerts = report.get("high_risk_alerts", [])
        if alerts:
            lines.append("HIGH-RISK ALERTS")
            lines.append("-" * 80)
            for alert in alerts:
                lines.append(f"Alert #{alert.get('alert_number', 0)}: {alert.get('location', 'N/A')}")
                lines.append(f"  Risk Score: {alert.get('risk_score', 0.0)}")
                lines.append(f"  Risk Factors: {', '.join(alert.get('primary_risk_factors', []))}")
                lines.append(f"  Why Dangerous: {alert.get('why_dangerous', 'N/A')}")
                lines.append(f"  Immediate Recommendation: {alert.get('immediate_recommendation', 'N/A')}")
                lines.append("")
        
        # Corrections
        corrections = report.get("evidence_backed_corrections", {})
        if corrections.get("corrections"):
            lines.append("EVIDENCE-BACKED CORRECTIONS")
            lines.append("-" * 80)
            for factor_key, correction_data in corrections.get("corrections", {}).items():
                lines.append(f"\n{correction_data.get('risk_factor_name', 'N/A')}")
                lines.append(f"Description: {correction_data.get('description', 'N/A')}")
                lines.append(f"Why Dangerous: {correction_data.get('why_dangerous', 'N/A')}")
                lines.append(f"Evidence: {correction_data.get('evidence', 'N/A')}")
                lines.append("\nInterventions:")
                for intervention in correction_data.get("interventions", []):
                    lines.append(f"  Priority {intervention.get('priority', 0)}: {intervention.get('intervention', 'N/A')}")
                    lines.append(f"    Rationale: {intervention.get('rationale', 'N/A')}")
                    lines.append(f"    Exercises: {', '.join(intervention.get('exercises', []))}")
                    lines.append(f"    Frequency: {intervention.get('frequency', 'N/A')}")
                    lines.append(f"    Expected Improvement: {intervention.get('expected_improvement', 'N/A')}")
                    lines.append("")
        
        # Risk Reduction Plan
        plan = report.get("risk_reduction_plan", {})
        if plan.get("phases"):
            lines.append("RISK REDUCTION PLAN")
            lines.append("-" * 80)
            lines.append(f"Timeline: {plan.get('timeline', 'N/A')}")
            for phase in plan.get("phases", []):
                lines.append(f"\n{phase.get('phase', 'N/A')}")
                lines.append(f"Focus: {phase.get('focus', 'N/A')}")
                lines.append("Actions:")
                for action in phase.get("actions", []):
                    lines.append(f"  - {action}")
            lines.append("")
        
        # Tracking Metrics
        tracking = report.get("tracking_metrics", {})
        if tracking:
            lines.append("TRACKING METRICS")
            lines.append("-" * 80)
            baseline = tracking.get("baseline_metrics", {})
            lines.append("Baseline Metrics:")
            for key, value in baseline.items():
                lines.append(f"  {key}: {value}")
            lines.append("\nTarget Metrics:")
            targets = tracking.get("target_metrics", {})
            for key, value in targets.items():
                lines.append(f"  {key}: {value}")
            lines.append("")
        
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def save_text_report(
        self,
        report: Dict[str, Any],
        output_dir: Path,
        base_name: str
    ) -> Path:
        """
        Save a human-readable text version of the report.
        
        Args:
            report: The report dictionary
            output_dir: Directory to save the report
            base_name: Base name for the file
        
        Returns:
            Path to the saved text report file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        text_file = output_dir / f"{base_name}_acl_risk_report.txt"
        
        text_summary = self.generate_text_summary(report)
        
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_summary)
        
        logger.info(f"✅ Coach-friendly ACL risk text report saved to: {text_file}")
        return text_file







