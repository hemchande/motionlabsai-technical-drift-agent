#!/usr/bin/env python3
"""
Face Recognition Tool
Detects and recognizes faces, stores embeddings in vector index for identity matching.
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import pickle
import json

logger = logging.getLogger(__name__)

# Try to import face_recognition
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    logger.warning("⚠️  face_recognition not available - install with: pip install face-recognition")

# Try to import FAISS for vector index
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("⚠️  faiss-cpu not available - will use simple list-based matching")


class FaceRecognitionTool:
    """
    Face recognition tool with vector index for identity matching.
    Stores face embeddings and names for cross-referencing.
    """
    
    def __init__(self, embeddings_dir: str = "face_embeddings"):
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        
        # Storage for face embeddings
        self.known_face_encodings: List[np.ndarray] = []
        self.known_face_names: List[str] = []
        self.face_index = None  # FAISS index if available
        
        self._initialize_face_index()
        self._load_saved_embeddings()
    
    def _initialize_face_index(self):
        """Initialize FAISS index for fast face matching"""
        if FAISS_AVAILABLE:
            try:
                # Face encodings are 128-dimensional vectors
                dimension = 128
                self.face_index = faiss.IndexFlatL2(dimension)
                logger.info("✅ FAISS vector index initialized for face embeddings")
            except Exception as e:
                logger.warning(f"⚠️  Failed to initialize FAISS index: {e}")
                self.face_index = None
        else:
            logger.info("ℹ️  Using list-based face matching (FAISS not available)")
    
    def _load_saved_embeddings(self):
        """Load saved face embeddings from disk"""
        embeddings_file = self.embeddings_dir / "embeddings.pkl"
        names_file = self.embeddings_dir / "names.json"
        
        if embeddings_file.exists() and names_file.exists():
            try:
                with open(embeddings_file, 'rb') as f:
                    self.known_face_encodings = pickle.load(f)
                
                with open(names_file, 'r') as f:
                    self.known_face_names = json.load(f)
                
                # Rebuild FAISS index if available
                if self.face_index and len(self.known_face_encodings) > 0:
                    encodings_array = np.array(self.known_face_encodings)
                    self.face_index.add(encodings_array)
                
                logger.info(f"✅ Loaded {len(self.known_face_names)} saved face embeddings")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load saved embeddings: {e}")
    
    def _save_embeddings(self):
        """Save face embeddings to disk"""
        embeddings_file = self.embeddings_dir / "embeddings.pkl"
        names_file = self.embeddings_dir / "names.json"
        
        try:
            with open(embeddings_file, 'wb') as f:
                pickle.dump(self.known_face_encodings, f)
            
            with open(names_file, 'w') as f:
                json.dump(self.known_face_names, f)
            
            logger.info(f"✅ Saved {len(self.known_face_names)} face embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to save embeddings: {e}")
    
    def add_face_encoding(self, name: str, image_path: str) -> bool:
        """
        Add a face encoding from an image file.
        
        Args:
            name: Name to associate with this face
            image_path: Path to image file
        
        Returns:
            True if successful, False otherwise
        """
        if not FACE_RECOGNITION_AVAILABLE:
            logger.error("❌ face_recognition library not available")
            return False
        
        try:
            # Load image
            image = face_recognition.load_image_file(image_path)
            
            # Get face encodings
            encodings = face_recognition.face_encodings(image)
            
            if not encodings:
                logger.warning(f"⚠️  No face found in image: {image_path}")
                return False
            
            # Use first face encoding
            encoding = encodings[0]
            
            # Add to known faces
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            
            # Add to FAISS index if available
            if self.face_index:
                encoding_array = np.array([encoding])
                self.face_index.add(encoding_array)
            
            # Save to disk
            self._save_embeddings()
            
            logger.info(f"✅ Added face encoding for: {name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add face encoding: {e}")
            return False
    
    def detect_and_identify(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame and identify them using stored embeddings.
        
        Args:
            frame: Input frame (BGR format from OpenCV)
        
        Returns:
            List of face detection dictionaries with identification info
        """
        if not FACE_RECOGNITION_AVAILABLE:
            return []
        
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find face locations
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            detections = []
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # Try to identify face
                name = "Unknown"
                confidence = 0.0
                identified = False
                
                if len(self.known_face_encodings) > 0:
                    if self.face_index:
                        # Use FAISS for fast matching
                        encoding_array = np.array([face_encoding])
                        distances, indices = self.face_index.search(encoding_array, 1)
                        
                        if len(indices[0]) > 0 and distances[0][0] < 0.6:  # Threshold for match
                            idx = indices[0][0]
                            name = self.known_face_names[idx]
                            confidence = 1.0 - min(distances[0][0] / 0.6, 1.0)
                            identified = True
                    else:
                        # Use list-based matching
                        matches = face_recognition.compare_faces(
                            self.known_face_encodings,
                            face_encoding,
                            tolerance=0.6
                        )
                        
                        face_distances = face_recognition.face_distance(
                            self.known_face_encodings,
                            face_encoding
                        )
                        
                        if True in matches:
                            best_match_index = np.argmin(face_distances)
                            if face_distances[best_match_index] < 0.6:
                                name = self.known_face_names[best_match_index]
                                confidence = 1.0 - face_distances[best_match_index]
                                identified = True
                
                detections.append({
                    "location": (top, right, bottom, left),
                    "encoding": face_encoding,
                    "name": name,
                    "confidence": confidence,
                    "identified": identified
                })
            
            return detections
            
        except Exception as e:
            logger.error(f"❌ Face detection error: {e}")
            return []
    
    def draw_face_overlay(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Draw face detection and identification overlays on frame.
        
        Args:
            frame: Input frame
            detections: List of face detection dictionaries
        
        Returns:
            Frame with face overlays
        """
        overlay = frame.copy()
        
        for detection in detections:
            top, right, bottom, left = detection["location"]
            name = detection["name"]
            confidence = detection["confidence"]
            identified = detection["identified"]
            
            # Choose color based on identification
            if identified:
                color = (0, 255, 0)  # Green for identified
            else:
                color = (0, 0, 255)  # Red for unknown
            
            # Draw rectangle
            cv2.rectangle(overlay, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"{name} ({confidence:.2f})" if identified else "Unknown"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            # Draw label background
            cv2.rectangle(
                overlay,
                (left, top - label_size[1] - 10),
                (left + label_size[0], top),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                overlay,
                label,
                (left, top - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
        
        return overlay
    
    def list_known_faces(self) -> List[str]:
        """Get list of known face names"""
        return self.known_face_names.copy()
    
    def remove_face(self, name: str) -> bool:
        """Remove a face encoding by name"""
        if name not in self.known_face_names:
            return False
        
        # Find all indices with this name
        indices = [i for i, n in enumerate(self.known_face_names) if n == name]
        
        # Remove in reverse order to maintain indices
        for idx in reversed(indices):
            del self.known_face_encodings[idx]
            del self.known_face_names[idx]
        
        # Rebuild FAISS index
        if self.face_index and len(self.known_face_encodings) > 0:
            self.face_index.reset()
            encodings_array = np.array(self.known_face_encodings)
            self.face_index.add(encodings_array)
        elif self.face_index:
            self.face_index.reset()
        
        # Save to disk
        self._save_embeddings()
        
        logger.info(f"✅ Removed face encoding for: {name}")
        return True
    
    def is_available(self) -> bool:
        """Check if face recognition is available"""
        return FACE_RECOGNITION_AVAILABLE
