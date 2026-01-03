#!/usr/bin/env python3
"""
ACL Trajectory Analyzer Agent

A separate agent that analyzes ACL risk patterns across multiple sessions.
Identifies trajectories, trends, and problematic patterns in ACL injury risk over time.

This agent is separate from the main cvMLAgent and processes saved session results.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict
import statistics

logger = logging.getLogger(__name__)


class ACLTrajectoryAnalyzer:
    """
    Analyzes ACL risk trajectories across multiple sessions.
    Identifies patterns, trends, and problematic biomechanical trajectories.
    """
    
    def __init__(self, results_dir: Optional[Path] = None):
        """
        Initialize the trajectory analyzer.
        
        Args:
            results_dir: Directory containing session JSON files (default: stream_output/)
        """
        self.results_dir = results_dir or Path("stream_output")
        self.sessions: List[Dict[str, Any]] = []
    
    def load_sessions(self, pattern: str = "*_metrics.json") -> int:
        """
        Load all session JSON files from the results directory.
        
        Args:
            pattern: Glob pattern to match session files
        
        Returns:
            Number of sessions loaded
        """
        session_files = list(self.results_dir.glob(pattern))
        self.sessions = []
        
        for session_file in session_files:
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    
                    # Extract ACL risk data
                    acl_data = self._extract_acl_data(session_data, session_file)
                    if acl_data:
                        self.sessions.append(acl_data)
                        logger.info(f"✅ Loaded session: {session_file.name}")
            except Exception as e:
                logger.warning(f"⚠️  Could not load session {session_file.name}: {e}")
        
        logger.info(f"📊 Loaded {len(self.sessions)} sessions with ACL risk data")
        return len(self.sessions)
    
    def _extract_acl_data(self, session_data: Dict[str, Any], session_file: Path) -> Optional[Dict[str, Any]]:
        """
        Extract ACL risk data from a session JSON file.
        
        Args:
            session_data: Full session JSON data
            session_file: Path to the session file
        
        Returns:
            Extracted ACL data dictionary or None
        """
        acl_flagged = session_data.get("acl_flagged_timesteps", [])
        
        # Only include sessions with ACL risk data
        if not acl_flagged:
            return None
        
        # Extract session metadata
        session_info = {
            "session_file": str(session_file),
            "session_name": session_file.stem,
            "timestamp": session_data.get("timestamp", ""),
            "activity": session_data.get("activity", "unknown"),
            "technique": session_data.get("technique", "unknown"),
            "date": self._extract_date(session_data.get("timestamp", ""), session_file),
            "acl_flagged_timesteps": acl_flagged,
            "total_high_risk_moments": len(acl_flagged),
            "metrics": session_data.get("metrics", {}),
            "landing_phases": session_data.get("landing_phases", [])
        }
        
        # Calculate session-level ACL risk metrics
        if acl_flagged:
            risk_scores = [ts.get("risk_score", 0.0) for ts in acl_flagged]
            valgus_angles = [ts.get("valgus_angle", 0.0) for ts in acl_flagged]
            knee_flexions = [ts.get("knee_flexion", 0.0) for ts in acl_flagged]
            impact_forces = [ts.get("impact_force", 0.0) for ts in acl_flagged]
            
            session_info["acl_metrics"] = {
                "avg_risk_score": statistics.mean(risk_scores) if risk_scores else 0.0,
                "max_risk_score": max(risk_scores) if risk_scores else 0.0,
                "min_risk_score": min(risk_scores) if risk_scores else 0.0,
                "avg_valgus_angle": statistics.mean(valgus_angles) if valgus_angles else 0.0,
                "max_valgus_angle": max(valgus_angles) if valgus_angles else 0.0,
                "avg_knee_flexion": statistics.mean(knee_flexions) if knee_flexions else 0.0,
                "min_knee_flexion": min(knee_flexions) if knee_flexions else 0.0,
                "avg_impact_force": statistics.mean(impact_forces) if impact_forces else 0.0,
                "max_impact_force": max(impact_forces) if impact_forces else 0.0
            }
        
        return session_info
    
    def _extract_date(self, timestamp: str, session_file: Path) -> str:
        """Extract date from timestamp or filename."""
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                return dt.strftime("%Y-%m-%d")
            except:
                pass
        
        # Try to extract from filename
        parts = session_file.stem.split('_')
        for part in parts:
            if len(part) == 10 and part.count('-') == 2:  # YYYY-MM-DD format
                return part
        
        return datetime.now().strftime("%Y-%m-%d")
    
    def analyze_trajectory(self) -> Dict[str, Any]:
        """
        Analyze ACL risk trajectory across all loaded sessions.
        
        Returns:
            Comprehensive trajectory analysis
        """
        if not self.sessions:
            return {"error": "No sessions loaded. Call load_sessions() first."}
        
        # Sort sessions by date
        sorted_sessions = sorted(self.sessions, key=lambda x: x.get("date", ""))
        
        analysis = {
            "total_sessions": len(sorted_sessions),
            "date_range": {
                "earliest": sorted_sessions[0].get("date", "") if sorted_sessions else "",
                "latest": sorted_sessions[-1].get("date", "") if sorted_sessions else ""
            },
            "trajectory_analysis": self._analyze_risk_trajectory(sorted_sessions),
            "pattern_analysis": self._analyze_patterns(sorted_sessions),
            "trend_analysis": self._analyze_trends(sorted_sessions),
            "problematic_patterns": self._identify_problematic_patterns(sorted_sessions),
            "recommendations": self._generate_trajectory_recommendations(sorted_sessions)
        }
        
        return analysis
    
    def _analyze_risk_trajectory(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how ACL risk changes over time."""
        if len(sessions) < 2:
            return {"message": "Insufficient sessions for trajectory analysis (need at least 2)"}
        
        # Extract metrics over time
        risk_scores = []
        valgus_angles = []
        knee_flexions = []
        impact_forces = []
        high_risk_counts = []
        dates = []
        
        for session in sessions:
            metrics = session.get("acl_metrics", {})
            risk_scores.append(metrics.get("avg_risk_score", 0.0))
            valgus_angles.append(metrics.get("avg_valgus_angle", 0.0))
            knee_flexions.append(metrics.get("avg_knee_flexion", 0.0))
            impact_forces.append(metrics.get("avg_impact_force", 0.0))
            high_risk_counts.append(session.get("total_high_risk_moments", 0))
            dates.append(session.get("date", ""))
        
        # Calculate trends
        def calculate_trend(values):
            if len(values) < 2:
                return "insufficient_data"
            first_half = statistics.mean(values[:len(values)//2])
            second_half = statistics.mean(values[len(values)//2:])
            if second_half < first_half * 0.9:
                return "improving"
            elif second_half > first_half * 1.1:
                return "worsening"
            else:
                return "stable"
        
        trajectory = {
            "risk_score_trajectory": {
                "values": risk_scores,
                "dates": dates,
                "trend": calculate_trend(risk_scores),
                "change": risk_scores[-1] - risk_scores[0] if len(risk_scores) > 1 else 0.0,
                "percent_change": ((risk_scores[-1] - risk_scores[0]) / risk_scores[0] * 100) if len(risk_scores) > 1 and risk_scores[0] > 0 else 0.0
            },
            "valgus_angle_trajectory": {
                "values": valgus_angles,
                "dates": dates,
                "trend": calculate_trend(valgus_angles),
                "change": valgus_angles[-1] - valgus_angles[0] if len(valgus_angles) > 1 else 0.0
            },
            "knee_flexion_trajectory": {
                "values": knee_flexions,
                "dates": dates,
                "trend": calculate_trend(knee_flexions),
                "change": knee_flexions[-1] - knee_flexions[0] if len(knee_flexions) > 1 else 0.0
            },
            "impact_force_trajectory": {
                "values": impact_forces,
                "dates": dates,
                "trend": calculate_trend(impact_forces),
                "change": impact_forces[-1] - impact_forces[0] if len(impact_forces) > 1 else 0.0
            },
            "high_risk_count_trajectory": {
                "values": high_risk_counts,
                "dates": dates,
                "trend": calculate_trend(high_risk_counts),
                "change": high_risk_counts[-1] - high_risk_counts[0] if len(high_risk_counts) > 1 else 0
            }
        }
        
        return trajectory
    
    def _analyze_patterns(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns across sessions."""
        # Identify most common risk factors
        risk_factor_counts = defaultdict(int)
        technique_risks = defaultdict(list)
        
        for session in sessions:
            technique = session.get("technique", "unknown")
            flagged = session.get("acl_flagged_timesteps", [])
            
            for ts in flagged:
                factors = ts.get("primary_risk_factors", [])
                for factor in factors:
                    if "valgus" in factor.lower():
                        risk_factor_counts["knee_valgus"] += 1
                    elif "flexion" in factor.lower():
                        risk_factor_counts["insufficient_flexion"] += 1
                    elif "impact" in factor.lower():
                        risk_factor_counts["high_impact"] += 1
                
                technique_risks[technique].append(ts.get("risk_score", 0.0))
        
        # Calculate technique-specific risk
        technique_avg_risks = {
            tech: statistics.mean(risks) if risks else 0.0
            for tech, risks in technique_risks.items()
        }
        
        return {
            "most_common_risk_factors": dict(sorted(risk_factor_counts.items(), key=lambda x: x[1], reverse=True)),
            "technique_specific_risks": technique_avg_risks,
            "highest_risk_technique": max(technique_avg_risks.items(), key=lambda x: x[1])[0] if technique_avg_risks else "unknown"
        }
    
    def _analyze_trends(self, sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends in ACL risk metrics."""
        if len(sessions) < 3:
            return {"message": "Insufficient sessions for trend analysis (need at least 3)"}
        
        # Calculate linear regression-like trends
        def calculate_slope(values):
            if len(values) < 2:
                return 0.0
            n = len(values)
            x = list(range(n))
            x_mean = statistics.mean(x)
            y_mean = statistics.mean(values)
            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
            return numerator / denominator if denominator != 0 else 0.0
        
        risk_scores = [s.get("acl_metrics", {}).get("avg_risk_score", 0.0) for s in sessions]
        valgus_angles = [s.get("acl_metrics", {}).get("avg_valgus_angle", 0.0) for s in sessions]
        knee_flexions = [s.get("acl_metrics", {}).get("avg_knee_flexion", 0.0) for s in sessions]
        high_risk_counts = [s.get("total_high_risk_moments", 0) for s in sessions]
        
        return {
            "risk_score_trend": {
                "slope": calculate_slope(risk_scores),
                "direction": "improving" if calculate_slope(risk_scores) < -0.01 else ("worsening" if calculate_slope(risk_scores) > 0.01 else "stable"),
                "interpretation": "Risk is decreasing over time" if calculate_slope(risk_scores) < -0.01 else ("Risk is increasing over time" if calculate_slope(risk_scores) > 0.01 else "Risk is stable")
            },
            "valgus_angle_trend": {
                "slope": calculate_slope(valgus_angles),
                "direction": "improving" if calculate_slope(valgus_angles) < -0.1 else ("worsening" if calculate_slope(valgus_angles) > 0.1 else "stable"),
                "interpretation": "Valgus angles are decreasing" if calculate_slope(valgus_angles) < -0.1 else ("Valgus angles are increasing" if calculate_slope(valgus_angles) > 0.1 else "Valgus angles are stable")
            },
            "knee_flexion_trend": {
                "slope": calculate_slope(knee_flexions),
                "direction": "improving" if calculate_slope(knee_flexions) > 0.5 else ("worsening" if calculate_slope(knee_flexions) < -0.5 else "stable"),
                "interpretation": "Knee flexion is improving" if calculate_slope(knee_flexions) > 0.5 else ("Knee flexion is worsening" if calculate_slope(knee_flexions) < -0.5 else "Knee flexion is stable")
            },
            "high_risk_count_trend": {
                "slope": calculate_slope(high_risk_counts),
                "direction": "improving" if calculate_slope(high_risk_counts) < -0.1 else ("worsening" if calculate_slope(high_risk_counts) > 0.1 else "stable"),
                "interpretation": "High-risk moments are decreasing" if calculate_slope(high_risk_counts) < -0.1 else ("High-risk moments are increasing" if calculate_slope(high_risk_counts) > 0.1 else "High-risk moments are stable")
            }
        }
    
    def _identify_problematic_patterns(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify problematic ACL risk patterns across sessions."""
        problematic = []
        
        # Pattern 1: Consistently high risk across sessions
        high_risk_sessions = [s for s in sessions if s.get("acl_metrics", {}).get("avg_risk_score", 0.0) >= 0.7]
        if len(high_risk_sessions) >= len(sessions) * 0.5:  # 50% or more sessions
            problematic.append({
                "pattern": "Consistently High Risk",
                "severity": "HIGH",
                "description": f"{len(high_risk_sessions)} out of {len(sessions)} sessions show high ACL risk (>=0.7)",
                "sessions_affected": len(high_risk_sessions),
                "recommendation": "Immediate intervention required. Focus on fundamental landing mechanics and strength training."
            })
        
        # Pattern 2: Worsening trajectory
        if len(sessions) >= 3:
            risk_scores = [s.get("acl_metrics", {}).get("avg_risk_score", 0.0) for s in sessions]
            if risk_scores[-1] > risk_scores[0] * 1.15:  # 15% increase
                problematic.append({
                    "pattern": "Worsening Risk Trajectory",
                    "severity": "HIGH",
                    "description": f"ACL risk has increased by {((risk_scores[-1] - risk_scores[0]) / risk_scores[0] * 100):.1f}% over {len(sessions)} sessions",
                    "sessions_affected": len(sessions),
                    "recommendation": "Risk is increasing over time. Review training load, technique, and recovery protocols."
                })
        
        # Pattern 3: High valgus angles consistently
        high_valgus_sessions = [s for s in sessions if s.get("acl_metrics", {}).get("avg_valgus_angle", 0.0) > 10.0]
        if len(high_valgus_sessions) >= len(sessions) * 0.5:
            problematic.append({
                "pattern": "Consistent Knee Valgus Collapse",
                "severity": "MODERATE",
                "description": f"{len(high_valgus_sessions)} sessions show valgus angles >10°",
                "sessions_affected": len(high_valgus_sessions),
                "recommendation": "Focus on hip strengthening (gluteus medius) and landing technique to reduce valgus collapse."
            })
        
        # Pattern 4: Stiff landings consistently
        stiff_landing_sessions = [s for s in sessions if s.get("acl_metrics", {}).get("avg_knee_flexion", 0.0) < 20.0]
        if len(stiff_landing_sessions) >= len(sessions) * 0.5:
            problematic.append({
                "pattern": "Consistent Stiff Landings",
                "severity": "MODERATE",
                "description": f"{len(stiff_landing_sessions)} sessions show knee flexion <20° during landing",
                "sessions_affected": len(stiff_landing_sessions),
                "recommendation": "Focus on eccentric strength training and landing mechanics to increase knee flexion."
            })
        
        # Pattern 5: Increasing high-risk moment frequency
        if len(sessions) >= 3:
            high_risk_counts = [s.get("total_high_risk_moments", 0) for s in sessions]
            if high_risk_counts[-1] > high_risk_counts[0] * 1.5:  # 50% increase
                problematic.append({
                    "pattern": "Increasing High-Risk Moment Frequency",
                    "severity": "MODERATE",
                    "description": f"Number of high-risk moments per session has increased from {high_risk_counts[0]} to {high_risk_counts[-1]}",
                    "sessions_affected": len(sessions),
                    "recommendation": "Review training intensity and technique. Consider reducing load until mechanics improve."
                })
        
        return problematic
    
    def _generate_trajectory_recommendations(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate recommendations based on trajectory analysis."""
        recommendations = []
        
        if len(sessions) < 2:
            return [{"message": "Need more sessions for trajectory-based recommendations"}]
        
        # Analyze trends
        risk_scores = [s.get("acl_metrics", {}).get("avg_risk_score", 0.0) for s in sessions]
        valgus_angles = [s.get("acl_metrics", {}).get("avg_valgus_angle", 0.0) for s in sessions]
        knee_flexions = [s.get("acl_metrics", {}).get("avg_knee_flexion", 0.0) for s in sessions]
        
        # Recommendation 1: Overall trajectory
        if risk_scores[-1] < risk_scores[0] * 0.9:
            recommendations.append({
                "priority": 1,
                "category": "Positive Progress",
                "recommendation": "ACL risk is decreasing over time. Continue current interventions and maintain focus on proper landing mechanics.",
                "action_items": [
                    "Maintain current training protocols",
                    "Continue strength and technique work",
                    "Monitor for any regression"
                ]
            })
        elif risk_scores[-1] > risk_scores[0] * 1.1:
            recommendations.append({
                "priority": 1,
                "category": "Urgent Intervention",
                "recommendation": "ACL risk is increasing over time. Immediate intervention required to reverse this trend.",
                "action_items": [
                    "Reduce training intensity/volume",
                    "Focus on fundamental landing mechanics",
                    "Implement comprehensive strength program",
                    "Consider professional biomechanical assessment"
                ]
            })
        
        # Recommendation 2: Valgus trajectory
        if valgus_angles[-1] > valgus_angles[0] * 1.1:
            recommendations.append({
                "priority": 2,
                "category": "Valgus Correction",
                "recommendation": "Knee valgus angles are increasing. Focus on hip strengthening and landing alignment.",
                "action_items": [
                    "Daily hip strengthening (gluteus medius focus)",
                    "Landing technique retraining with video feedback",
                    "Progressive plyometric training emphasizing proper alignment"
                ]
            })
        
        # Recommendation 3: Knee flexion trajectory
        if knee_flexions[-1] < knee_flexions[0] * 0.9:
            recommendations.append({
                "priority": 2,
                "category": "Landing Mechanics",
                "recommendation": "Knee flexion during landing is decreasing. Focus on eccentric strength and landing technique.",
                "action_items": [
                    "Eccentric quadriceps and hamstring strengthening",
                    "Soft landing drills with increased flexion",
                    "Progressive height drops with focus on shock absorption"
                ]
            })
        
        return recommendations
    
    def generate_report(self, output_dir: Optional[Path] = None) -> Path:
        """
        Generate a comprehensive trajectory analysis report.
        
        Args:
            output_dir: Directory to save the report (default: results_dir)
        
        Returns:
            Path to the saved report file
        """
        output_dir = output_dir or self.results_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        analysis = self.analyze_trajectory()
        
        # Save JSON report
        report_file = output_dir / f"acl_trajectory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate text summary
        text_file = output_dir / f"acl_trajectory_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self._generate_text_report(analysis, text_file)
        
        logger.info(f"✅ Trajectory analysis report saved to: {report_file}")
        logger.info(f"✅ Trajectory analysis text report saved to: {text_file}")
        
        return report_file
    
    def _generate_text_report(self, analysis: Dict[str, Any], output_file: Path):
        """Generate a human-readable text report."""
        lines = []
        lines.append("=" * 80)
        lines.append("ACL RISK TRAJECTORY ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total Sessions Analyzed: {analysis.get('total_sessions', 0)}")
        lines.append(f"Date Range: {analysis.get('date_range', {}).get('earliest', 'N/A')} to {analysis.get('date_range', {}).get('latest', 'N/A')}")
        lines.append("")
        
        # Trajectory Analysis
        trajectory = analysis.get("trajectory_analysis", {})
        if trajectory and "risk_score_trajectory" in trajectory:
            lines.append("RISK TRAJECTORY")
            lines.append("-" * 80)
            risk_traj = trajectory["risk_score_trajectory"]
            lines.append(f"Trend: {risk_traj.get('trend', 'N/A')}")
            lines.append(f"Change: {risk_traj.get('change', 0.0):.3f} ({risk_traj.get('percent_change', 0.0):.1f}%)")
            lines.append("")
        
        # Problematic Patterns
        problematic = analysis.get("problematic_patterns", [])
        if problematic:
            lines.append("PROBLEMATIC PATTERNS IDENTIFIED")
            lines.append("-" * 80)
            for i, pattern in enumerate(problematic, 1):
                lines.append(f"{i}. {pattern.get('pattern', 'N/A')} - {pattern.get('severity', 'N/A')} SEVERITY")
                lines.append(f"   Description: {pattern.get('description', 'N/A')}")
                lines.append(f"   Recommendation: {pattern.get('recommendation', 'N/A')}")
                lines.append("")
        
        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            lines.append("RECOMMENDATIONS")
            lines.append("-" * 80)
            for rec in recommendations:
                lines.append(f"Priority {rec.get('priority', 'N/A')}: {rec.get('category', 'N/A')}")
                lines.append(f"  {rec.get('recommendation', 'N/A')}")
                lines.append("  Action Items:")
                for item in rec.get("action_items", []):
                    lines.append(f"    - {item}")
                lines.append("")
        
        lines.append("=" * 80)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))


def main():
    """Main function for standalone trajectory analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze ACL risk trajectories across multiple sessions")
    parser.add_argument("--results-dir", type=str, default="stream_output", help="Directory containing session JSON files")
    parser.add_argument("--pattern", type=str, default="*_metrics.json", help="File pattern to match session files")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for reports (default: results-dir)")
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ACLTrajectoryAnalyzer(results_dir=Path(args.results_dir))
    
    # Load sessions
    print(f"📊 Loading sessions from: {args.results_dir}")
    session_count = analyzer.load_sessions(pattern=args.pattern)
    
    if session_count == 0:
        print("❌ No sessions with ACL risk data found")
        return
    
    print(f"✅ Loaded {session_count} sessions")
    
    # Generate analysis
    print("🔍 Analyzing ACL risk trajectories...")
    output_dir = Path(args.output_dir) if args.output_dir else None
    report_file = analyzer.generate_report(output_dir=output_dir)
    
    print(f"✅ Trajectory analysis complete!")
    print(f"📄 Report saved to: {report_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()






