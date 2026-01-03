#!/usr/bin/env python3
"""
Face Management Script
Add, list, and remove face encodings for the face recognition system.
"""

import sys
import argparse
from pathlib import Path
from face_recognition_tool import FaceRecognitionTool

def add_face(name: str, image_path: str):
    """Add a face encoding"""
    tool = FaceRecognitionTool()
    
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    success = tool.add_face_encoding(name, image_path)
    if success:
        print(f"✅ Added face encoding for: {name}")
        print(f"   Image: {image_path}")
    else:
        print(f"❌ Failed to add face encoding for: {name}")
    
    return success

def list_faces():
    """List all known faces"""
    tool = FaceRecognitionTool()
    faces = tool.list_known_faces()
    
    if not faces:
        print("No faces registered yet.")
        print("\nTo add a face:")
        print("  python manage_faces.py add --name 'John Doe' --image path/to/image.jpg")
    else:
        print(f"Registered faces ({len(faces)}):")
        for i, name in enumerate(faces, 1):
            print(f"  {i}. {name}")

def remove_face(name: str):
    """Remove a face encoding"""
    tool = FaceRecognitionTool()
    
    success = tool.remove_face(name)
    if success:
        print(f"✅ Removed face encoding for: {name}")
    else:
        print(f"❌ Face not found: {name}")
        print("\nAvailable faces:")
        faces = tool.list_known_faces()
        for face in faces:
            print(f"  - {face}")
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manage face encodings for recognition")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Add face command
    add_parser = subparsers.add_parser("add", help="Add a face encoding")
    add_parser.add_argument("--name", required=True, help="Name for the face")
    add_parser.add_argument("--image", required=True, help="Path to image file")
    
    # List faces command
    subparsers.add_parser("list", help="List all registered faces")
    
    # Remove face command
    remove_parser = subparsers.add_parser("remove", help="Remove a face encoding")
    remove_parser.add_argument("--name", required=True, help="Name of face to remove")
    
    args = parser.parse_args()
    
    if args.command == "add":
        add_face(args.name, args.image)
    elif args.command == "list":
        list_faces()
    elif args.command == "remove":
        remove_face(args.name)
    else:
        parser.print_help()




















