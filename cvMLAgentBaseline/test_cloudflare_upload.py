#!/usr/bin/env python3
"""
Test script to upload a video from Practice1 to Cloudflare Stream.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from cloudflare_stream import CloudflareStreamUploader

def main():
    load_dotenv()
    
    # Path to Practice1 directory (one level up from cvMLAgent)
    practice1_dir = Path(__file__).parent.parent / "Practice1"
    
    # Find the first mp4 file
    mp4_files = list(practice1_dir.glob("*.mp4"))
    
    if not mp4_files:
        print(f"❌ No mp4 files found in {practice1_dir}")
        return
    
    # Use the first mp4 file
    video_path = mp4_files[0]
    print(f"📹 Testing upload with: {video_path.name}")
    print(f"   Full path: {video_path}")
    print(f"   Size: {video_path.stat().st_size / (1024*1024):.2f} MB")
    print()
    
    # Initialize uploader
    uploader = CloudflareStreamUploader()
    
    if not uploader.enabled:
        print("❌ Cloudflare Stream uploader is not enabled (missing credentials)")
        print("   Please check your .env file has:")
        print("   CLOUDFLARE_STREAM_API_TOKEN=...")
        print("   CLOUDFLARE_STREAM_ACCOUNT_ID=...")
        return
    
    # Upload the video
    print("📤 Uploading to Cloudflare Stream...")
    metadata = {
        "test": True,
        "source": "Practice1",
        "filename": video_path.name
    }
    
    result = uploader.upload_video(str(video_path), metadata)
    
    if result:
        print()
        print("✅ Upload successful!")
        print()
        print("📋 Video Information:")
        print(f"   Video ID: {result.get('uid', 'N/A')}")
        print(f"   Stream URL: {result.get('stream_url', 'N/A')}")
        print(f"   Ready to Stream: {result.get('ready_to_stream', False)}")
        print(f"   Duration: {result.get('duration', 'N/A')} seconds")
        print(f"   Size: {result.get('size', 'N/A')} bytes")
        print()
        print("🔗 Use this stream URL in your frontend:")
        print(f"   {result.get('stream_url', '')}")
        print()
        
        # Save result to a JSON file for the frontend to read
        import json
        output_file = Path(__file__).parent / "test_upload_result.json"
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"💾 Saved result to: {output_file}")
    else:
        print()
        print("❌ Upload failed")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

