#!/usr/bin/env python3
"""
Register Face Script
Register a face with a name in the face recognition vector index.
"""

import cv2
import sys
import argparse
from pathlib import Path
from face_recognition_tool import FaceRecognitionTool

def main():
    parser = argparse.ArgumentParser(description="Register a face with a name")
    parser.add_argument("image_path", help="Path to image containing the face")
    parser.add_argument("name", help="Name to associate with the face")
    parser.add_argument("--athlete-id", help="Optional athlete ID")
    parser.add_argument("--team", help="Optional team name")
    parser.add_argument("--metadata", help="Optional JSON metadata")
    
    args = parser.parse_args()
    
    # Load image
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"❌ Image not found: {image_path}")
        sys.exit(1)
    
    frame = cv2.imread(str(image_path))
    if frame is None:
        print(f"❌ Could not load image: {image_path}")
        sys.exit(1)
    
    # Prepare metadata
    metadata = {}
    if args.athlete_id:
        metadata["athlete_id"] = args.athlete_id
    if args.team:
        metadata["team"] = args.team
    if args.metadata:
        import json
        try:
            metadata.update(json.loads(args.metadata))
        except Exception as e:
            print(f"⚠️  Failed to parse metadata JSON: {e}")
    
    # Register face
    face_tool = FaceRecognitionTool()
    
    print(f"📸 Registering face: {args.name}")
    success = face_tool.register_face(frame, args.name, metadata)
    
    if success:
        print(f"✅ Successfully registered: {args.name}")
        print(f"   Total registered faces: {len(face_tool.face_names)}")
    else:
        print(f"❌ Failed to register face")
        sys.exit(1)

if __name__ == "__main__":
    main()
