#!/usr/bin/env python3
"""
List Registered Faces
List all faces registered in the face recognition vector index.
"""

from face_recognition_tool import FaceRecognitionTool
import json

def main():
    face_tool = FaceRecognitionTool()
    faces = face_tool.list_registered_faces()
    
    print("=" * 60)
    print("REGISTERED FACES")
    print("=" * 60)
    print(f"\nTotal: {len(faces)} faces\n")
    
    for i, face in enumerate(faces, 1):
        print(f"{i}. {face['name']}")
        if face.get('metadata'):
            print(f"   Metadata: {json.dumps(face['metadata'], indent=2)}")
        print()
    
    print("=" * 60)

if __name__ == "__main__":
    main()
