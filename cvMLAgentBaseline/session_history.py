#!/usr/bin/env python3
"""
Session Chat History Manager
Stores chat history and metrics in vector indexes for retrieval and comparison.
"""

import logging
import os
import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Try to import FAISS for vector index
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️  faiss-cpu not available - will use simple list-based storage")

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("⚠️  sentence-transformers not available - will use simple keyword matching")


@dataclass
class ChatMessage:
    """Represents a chat message in the session"""
    role: str  # "user" or "assistant"
    content: str
    timestamp: str
    technique: Optional[str] = None
    activity: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    feedback: Optional[str] = None


@dataclass
class MetricSnapshot:
    """Represents a snapshot of metrics at a point in time"""
    timestamp: str
    technique: str
    activity: str
    metrics: Dict[str, float]
    feedback: Optional[str] = None
    improvement_suggestions: Optional[List[str]] = None


class SessionHistoryManager:
    """
    Manages session chat history and metrics using vector indexes.
    Stores and retrieves past conversations and metrics for comparison.
    """
    
    def __init__(self, session_id: str, storage_dir: str = "session_history"):
        self.session_id = session_id
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Session storage
        self.chat_messages: List[ChatMessage] = []
        self.metric_snapshots: List[MetricSnapshot] = []
        
        # Vector indexes
        self.chat_index = None  # FAISS index for chat messages
        self.metrics_index = None  # FAISS index for metrics
        self.embedding_model = None
        
        # Initialize embedding model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Sentence transformer model loaded for embeddings")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load embedding model: {e}")
        
        # Initialize vector indexes
        self._initialize_indexes()
        
        # Load existing session data
        self._load_session()
    
    def _initialize_indexes(self):
        """Initialize FAISS indexes for chat and metrics"""
        if FAISS_AVAILABLE and self.embedding_model:
            try:
                # Chat index: 384-dimensional vectors (all-MiniLM-L6-v2)
                dimension = 384
                self.chat_index = faiss.IndexFlatL2(dimension)
                self.metrics_index = faiss.IndexFlatL2(dimension)
                logger.info("✅ FAISS vector indexes initialized for session history")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize FAISS indexes: {e}")
                self.chat_index = None
                self.metrics_index = None
        else:
            logger.info("ℹ️  Using list-based storage (FAISS or embeddings not available)")
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text"""
        if self.embedding_model:
            try:
                embedding = self.embedding_model.encode(text, convert_to_numpy=True)
                return embedding
            except Exception as e:
                logger.warning(f"⚠️  Failed to generate embedding: {e}")
        return None
    
    def add_chat_message(
        self,
        role: str,
        content: str,
        technique: Optional[str] = None,
        activity: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        feedback: Optional[str] = None
    ):
        """
        Add a chat message to the session history.
        
        Args:
            role: "user" or "assistant"
            content: Message content
            technique: Optional technique name
            activity: Optional activity name
            metrics: Optional metrics associated with this message
            feedback: Optional feedback provided
        """
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat(),
            technique=technique,
            activity=activity,
            metrics=metrics,
            feedback=feedback
        )
        
        self.chat_messages.append(message)
        
        # Add to vector index if available
        if self.chat_index and self.embedding_model:
            embedding = self._get_embedding(content)
            if embedding is not None:
                try:
                    self.chat_index.add(np.array([embedding]))
                except Exception as e:
                    logger.warning(f"⚠️  Failed to add to chat index: {e}")
        
        # Save session
        self._save_session()
        
        logger.info(f"✅ Added {role} message to session history")
    
    def add_metric_snapshot(
        self,
        technique: str,
        activity: str,
        metrics: Dict[str, float],
        feedback: Optional[str] = None,
        improvement_suggestions: Optional[List[str]] = None
    ):
        """
        Add a metric snapshot for comparison.
        
        Args:
            technique: Technique name
            activity: Activity name
            metrics: Dictionary of metric_name -> value
            feedback: Optional feedback
            improvement_suggestions: Optional improvement suggestions
        """
        snapshot = MetricSnapshot(
            timestamp=datetime.utcnow().isoformat(),
            technique=technique,
            activity=activity,
            metrics=metrics,
            feedback=feedback,
            improvement_suggestions=improvement_suggestions
        )
        
        self.metric_snapshots.append(snapshot)
        
        # Add to vector index if available
        if self.metrics_index and self.embedding_model:
            # Create text representation of metrics for embedding
            metrics_text = self._format_metrics_for_embedding(technique, activity, metrics)
            embedding = self._get_embedding(metrics_text)
            if embedding is not None:
                try:
                    self.metrics_index.add(np.array([embedding]))
                except Exception as e:
                    logger.warning(f"⚠️  Failed to add to metrics index: {e}")
        
        # Save session
        self._save_session()
        
        logger.info(f"✅ Added metric snapshot for {technique} ({activity})")
    
    def _format_metrics_for_embedding(self, technique: str, activity: str, metrics: Dict[str, float]) -> str:
        """Format metrics as text for embedding"""
        parts = [f"technique: {technique}", f"activity: {activity}"]
        for key, value in sorted(metrics.items()):
            parts.append(f"{key}: {value:.2f}")
        return " ".join(parts)
    
    def search_chat_history(
        self,
        query: str,
        limit: int = 5,
        technique: Optional[str] = None,
        activity: Optional[str] = None
    ) -> List[ChatMessage]:
        """
        Search chat history for relevant messages.
        
        Args:
            query: Search query
            limit: Maximum number of results
            technique: Optional filter by technique
            activity: Optional filter by activity
        
        Returns:
            List of relevant chat messages
        """
        if not self.chat_messages:
            return []
        
        # If vector index available, use semantic search
        if self.chat_index and self.embedding_model and len(self.chat_messages) > 0:
            try:
                query_embedding = self._get_embedding(query)
                if query_embedding is not None:
                    # Search in index
                    k = min(limit, len(self.chat_messages))
                    distances, indices = self.chat_index.search(
                        np.array([query_embedding]), k
                    )
                    
                    # Get messages
                    results = []
                    for idx in indices[0]:
                        if idx < len(self.chat_messages):
                            msg = self.chat_messages[idx]
                            # Apply filters
                            if technique and msg.technique != technique:
                                continue
                            if activity and msg.activity != activity:
                                continue
                            results.append(msg)
                    
                    return results[:limit]
            except Exception as e:
                logger.warning(f"⚠️  Vector search failed, using keyword search: {e}")
        
        # Fallback to keyword search
        query_lower = query.lower()
        results = []
        for msg in self.chat_messages:
            if query_lower in msg.content.lower():
                # Apply filters
                if technique and msg.technique != technique:
                    continue
                if activity and msg.activity != activity:
                    continue
                results.append(msg)
                if len(results) >= limit:
                    break
        
        return results
    
    def get_past_metrics(
        self,
        technique: str,
        activity: Optional[str] = None,
        limit: int = 5
    ) -> List[MetricSnapshot]:
        """
        Get past metric snapshots for comparison.
        
        Args:
            technique: Technique name
            activity: Optional activity filter
            limit: Maximum number of snapshots
        
        Returns:
            List of metric snapshots
        """
        results = []
        for snapshot in reversed(self.metric_snapshots):  # Most recent first
            if snapshot.technique == technique:
                if activity and snapshot.activity != activity:
                    continue
                results.append(snapshot)
                if len(results) >= limit:
                    break
        
        return results
    
    def compare_metrics(
        self,
        current_metrics: Dict[str, float],
        technique: str,
        activity: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compare current metrics with past metrics.
        
        Args:
            current_metrics: Current metric values
            technique: Technique name
            activity: Optional activity filter
        
        Returns:
            Dictionary with comparison results
        """
        past_snapshots = self.get_past_metrics(technique, activity, limit=1)
        
        if not past_snapshots:
            return {
                "has_comparison": False,
                "message": "No past metrics found for comparison"
            }
        
        latest_snapshot = past_snapshots[0]
        past_metrics = latest_snapshot.metrics
        
        # Calculate differences
        improvements = []
        regressions = []
        unchanged = []
        
        for metric_name, current_value in current_metrics.items():
            if metric_name in past_metrics:
                past_value = past_metrics[metric_name]
                difference = current_value - past_value
                percent_change = (difference / past_value * 100) if past_value != 0 else 0
                
                # Determine if improvement or regression based on metric type
                # Higher is better metrics
                higher_is_better = [
                    "height_off_floor", "height_off_floor_pixels", "height_off_floor_normalized",
                    "knee_straightness", "knee_straightness_avg", "knee_straightness_left", "knee_straightness_right",
                    "posture_score", "knee_extension", "knee_extension_avg", "knee_extension_left", "knee_extension_right",
                    "hip_extension", "hip_extension_avg", "hip_extension_left", "hip_extension_right",
                    "elbow_extension", "elbow_extension_left", "elbow_extension_right",
                    "range_of_motion", "flight_height", "vertical_displacement"
                ]
                
                # Lower is better metrics
                lower_is_better = [
                    "impact_force", "impact_force_estimated", "impact_force_normalized",
                    "rounded_back_angle", "arched_back_angle", "rounded_back_detected", "arched_back_detected",
                    "forward_head_posture_cm", "forward_head_detected", "forward_head_angle",
                    "hunched_shoulders_detected", "shoulder_forward_position_cm", "hunched_shoulders_angle",
                    "landing_bend_angle", "landing_bend_angles", "knee_hyperextension_angle",
                    "knee_valgus_angle", "knee_varus_angle", "knee_bowed_detected",
                    "hip_tilt_cm", "hip_misaligned", "hip_alignment_angle",
                    "posture_issues_count"
                ]
                
                # Check if metric name matches (with or without suffix)
                metric_base = metric_name.split("_")[0] if "_" in metric_name else metric_name
                is_higher_better = any(metric_name.startswith(m) or m in metric_name for m in higher_is_better)
                is_lower_better = any(metric_name.startswith(m) or m in metric_name for m in lower_is_better)
                
                # Determine improvement
                if is_higher_better:
                    is_improvement = difference > 0.01
                elif is_lower_better:
                    is_improvement = difference < -0.01
                else:
                    # Default: assume higher is better for unknown metrics
                    is_improvement = abs(difference) > 0.01 and difference > 0
                
                if is_improvement:
                    improvements.append({
                        "metric": metric_name,
                        "past": past_value,
                        "current": current_value,
                        "change": difference,
                        "percent_change": percent_change
                    })
                elif abs(difference) > 0.01:
                    regressions.append({
                        "metric": metric_name,
                        "past": past_value,
                        "current": current_value,
                        "change": difference,
                        "percent_change": percent_change
                    })
                else:
                    unchanged.append(metric_name)
        
        return {
            "has_comparison": True,
            "past_timestamp": latest_snapshot.timestamp,
            "improvements": improvements,
            "regressions": regressions,
            "unchanged": unchanged,
            "summary": f"Compared with metrics from {latest_snapshot.timestamp}"
        }
    
    def get_feedback_with_history(
        self,
        current_metrics: Dict[str, float],
        technique: str,
        activity: str
    ) -> str:
        """
        Generate feedback that includes comparison with past metrics.
        
        Args:
            current_metrics: Current metric values
            technique: Technique name
            activity: Activity name
        
        Returns:
            Feedback text with historical context
        """
        comparison = self.compare_metrics(current_metrics, technique, activity)
        
        feedback_parts = []
        
        if comparison["has_comparison"]:
            feedback_parts.append("📊 **Progress Comparison:**")
            feedback_parts.append(f"Comparing with your previous session from {comparison['past_timestamp']}")
            feedback_parts.append("")
            
            if comparison["improvements"]:
                feedback_parts.append("✅ **Improvements:**")
                for imp in comparison["improvements"]:
                    feedback_parts.append(
                        f"  • {imp['metric']}: {imp['past']:.2f} → {imp['current']:.2f} "
                        f"({imp['percent_change']:+.1f}%)"
                    )
                feedback_parts.append("")
            
            if comparison["regressions"]:
                feedback_parts.append("⚠️  **Areas to Watch:**")
                for reg in comparison["regressions"]:
                    feedback_parts.append(
                        f"  • {reg['metric']}: {reg['past']:.2f} → {reg['current']:.2f} "
                        f"({reg['percent_change']:+.1f}%)"
                    )
                feedback_parts.append("")
        else:
            feedback_parts.append("📊 **First Session:**")
            feedback_parts.append("This is your first session with this technique. Keep practicing to see progress!")
            feedback_parts.append("")
        
        return "\n".join(feedback_parts)
    
    def _save_session(self):
        """Save session data to disk"""
        session_file = self.storage_dir / f"{self.session_id}.json"
        
        try:
            data = {
                "session_id": self.session_id,
                "chat_messages": [asdict(msg) for msg in self.chat_messages],
                "metric_snapshots": [asdict(snap) for snap in self.metric_snapshots],
                "last_updated": datetime.utcnow().isoformat()
            }
            
            with open(session_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"✅ Saved session data: {len(self.chat_messages)} messages, {len(self.metric_snapshots)} snapshots")
        except Exception as e:
            logger.error(f"❌ Failed to save session: {e}")
    
    def _load_session(self):
        """Load session data from disk"""
        session_file = self.storage_dir / f"{self.session_id}.json"
        
        if not session_file.exists():
            logger.info(f"ℹ️  No existing session data for {self.session_id}")
            return
        
        try:
            with open(session_file, 'r') as f:
                data = json.load(f)
            
            # Load chat messages
            self.chat_messages = [
                ChatMessage(**msg) for msg in data.get("chat_messages", [])
            ]
            
            # Load metric snapshots
            self.metric_snapshots = [
                MetricSnapshot(**snap) for snap in data.get("metric_snapshots", [])
            ]
            
            # Rebuild vector indexes
            if self.chat_index and self.embedding_model and self.chat_messages:
                try:
                    embeddings = []
                    for msg in self.chat_messages:
                        embedding = self._get_embedding(msg.content)
                        if embedding is not None:
                            embeddings.append(embedding)
                    
                    if embeddings:
                        self.chat_index.add(np.array(embeddings))
                except Exception as e:
                    logger.warning(f"⚠️  Failed to rebuild chat index: {e}")
            
            if self.metrics_index and self.embedding_model and self.metric_snapshots:
                try:
                    embeddings = []
                    for snap in self.metric_snapshots:
                        metrics_text = self._format_metrics_for_embedding(
                            snap.technique, snap.activity, snap.metrics
                        )
                        embedding = self._get_embedding(metrics_text)
                        if embedding is not None:
                            embeddings.append(embedding)
                    
                    if embeddings:
                        self.metrics_index.add(np.array(embeddings))
                except Exception as e:
                    logger.warning(f"⚠️  Failed to rebuild metrics index: {e}")
            
            logger.info(f"✅ Loaded session: {len(self.chat_messages)} messages, {len(self.metric_snapshots)} snapshots")
        except Exception as e:
            logger.error(f"❌ Failed to load session: {e}")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get summary of session history"""
        return {
            "session_id": self.session_id,
            "total_messages": len(self.chat_messages),
            "total_snapshots": len(self.metric_snapshots),
            "techniques_analyzed": list(set(
                msg.technique for msg in self.chat_messages if msg.technique
            )),
            "activities_analyzed": list(set(
                msg.activity for msg in self.chat_messages if msg.activity
            )),
            "first_message": self.chat_messages[0].timestamp if self.chat_messages else None,
            "last_message": self.chat_messages[-1].timestamp if self.chat_messages else None
        }
