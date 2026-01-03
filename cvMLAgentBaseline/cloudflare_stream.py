#!/usr/bin/env python3
"""
Cloudflare Stream API integration for uploading videos.

Uploads video files to Cloudflare Stream and returns streaming URLs.
"""

import os
import json
import logging
import requests
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class CloudflareStreamUploader:
    """Upload videos to Cloudflare Stream API."""
    
    def __init__(self):
        """Initialize with credentials from environment variables."""
        self.api_token = os.getenv("CLOUDFLARE_STREAM_API_TOKEN") or os.getenv("CLOUDFLARE_API_TOKEN")
        self.account_id = os.getenv("CLOUDFLARE_STREAM_ACCOUNT_ID") or os.getenv("CLOUDFLARE_ACCOUNT_ID")
        
        if not self.api_token or not self.account_id:
            logger.warning("⚠️  Cloudflare Stream credentials not found in environment variables")
            logger.warning("   Set CLOUDFLARE_STREAM_API_TOKEN and CLOUDFLARE_STREAM_ACCOUNT_ID")
            self.enabled = False
        else:
            self.enabled = True
            logger.info("✅ Cloudflare Stream uploader initialized")
    
    def upload_video(
        self,
        video_path: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Upload a video file to Cloudflare Stream.
        
        Args:
            video_path: Path to the video file to upload
            metadata: Optional metadata dictionary to attach to the video
        
        Returns:
            Dictionary with video info including:
            - id: Stream video ID
            - uid: Stream video UID
            - stream_url: Streaming URL
            - playback: Playback URLs
            Or None if upload failed or disabled
        """
        if not self.enabled:
            logger.warning("⚠️  Cloudflare Stream uploader is disabled (missing credentials)")
            return None
        
        video_path = Path(video_path)
        if not video_path.exists():
            logger.error(f"❌ Video file not found: {video_path}")
            return None
        
        try:
            logger.info(f"📤 Uploading video to Cloudflare Stream: {video_path.name}")
            
            # Prepare the upload URL
            url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/stream"
            
            # Prepare headers
            headers = {
                "Authorization": f"Bearer {self.api_token}"
            }
            
            # Prepare the file
            with open(video_path, 'rb') as video_file:
                files = {
                    'file': (video_path.name, video_file, 'video/mp4')
                }
                
                # Prepare form data with metadata if provided
                data = {}
                if metadata:
                    data['meta'] = json.dumps(metadata)
                
                # Upload the video
                response = requests.post(
                    url,
                    headers=headers,
                    files=files,
                    data=data,
                    timeout=300  # 5 minute timeout for large videos
                )
            
            if response.status_code != 200:
                logger.error(f"❌ Cloudflare Stream upload failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return None
            
            result = response.json()
            
            if not result.get("success"):
                errors = result.get("errors", [])
                logger.error(f"❌ Cloudflare Stream upload failed: {errors}")
                return None
            
            video_info = result.get("result", {})
            video_id = video_info.get("uid") or video_info.get("id")
            
            # Get playback URLs
            playback = video_info.get("playback", {})
            stream_url = playback.get("hls") or playback.get("dash") or ""
            
            # If no playback URL yet, construct from video ID
            if not stream_url and video_id:
                stream_url = f"https://customer-{self.account_id}.cloudflarestream.com/{video_id}/manifest/video.m3u8"
            
            upload_result = {
                "id": video_info.get("id"),
                "uid": video_id,
                "stream_url": stream_url,
                "playback": playback,
                "thumbnail": video_info.get("thumbnail"),
                "ready_to_stream": video_info.get("readyToStream", False),
                "duration": video_info.get("duration"),
                "size": video_info.get("size"),
                "meta": video_info.get("meta", {}),
                "created": video_info.get("created"),
            }
            
            logger.info(f"✅ Video uploaded to Cloudflare Stream: {video_id}")
            logger.info(f"   Stream URL: {stream_url}")
            
            return upload_result
            
        except Exception as e:
            logger.error(f"❌ Error uploading to Cloudflare Stream: {e}", exc_info=True)
            return None
    
    def get_video_stream_url(self, video_uid: str) -> str:
        """
        Get the streaming URL for a video.
        
        Args:
            video_uid: Cloudflare Stream video UID
        
        Returns:
            HLS streaming URL
        """
        if not self.account_id:
            return ""
        
        return f"https://customer-{self.account_id}.cloudflarestream.com/{video_uid}/manifest/video.m3u8"


def upload_video_to_cloudflare(
    video_path: str,
    metadata: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Convenience function to upload a video to Cloudflare Stream.
    
    Args:
        video_path: Path to video file
        metadata: Optional metadata
    
    Returns:
        Video info dict or None if failed
    """
    uploader = CloudflareStreamUploader()
    return uploader.upload_video(video_path, metadata)

