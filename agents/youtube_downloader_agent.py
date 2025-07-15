# agents/youtube_downloader_agent.py
import os
import yt_dlp
import asyncio
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import logging

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager, Validators

logger = logging.getLogger(__name__)

class YouTubeDownloaderAgent(BaseChatAgent):
    """Agent responsible for downloading audio from YouTube videos"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Downloads audio from YouTube videos")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages and download YouTube audio if requested"""
        
        # Get the latest message
        latest_message = messages[-1] if messages else None
        if not latest_message:
            return Response(
                chat_message=TextMessage(
                    content="No message provided.",
                    source=self.name
                )
            )
        
        content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # Check if this is a YouTube download request
        if not self._should_process(content):
            return Response(
                chat_message=TextMessage(
                    content="This agent only processes YouTube audio download requests.",
                    source=self.name
                )
            )
        
        try:
            # Extract YouTube URL from the message
            youtube_url = self._extract_youtube_url(content)
            if not youtube_url:
                return Response(
                    chat_message=TextMessage(
                        content="No valid YouTube URL found in the message.",
                        source=self.name
                    )
                )
            
            # Download the audio
            result = await self._download_audio(youtube_url)
            
            if result["success"]:
                return Response(
                    chat_message=TextMessage(
                        content=f"Successfully downloaded audio from {youtube_url}. "
                               f"File: {result['file_path']}. "
                               f"Title: {result['title']}. "
                               f"Duration: {result['duration']} seconds.",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"Failed to download audio: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in YouTubeDownloaderAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred while processing the request: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        download_keywords = ["download", "audio", "youtube", "extract"]
        content_lower = content.lower()
        
        # Check for YouTube URL and download request
        has_youtube_url = any(url_part in content_lower for url_part in ["youtube.com", "youtu.be"])
        has_download_request = any(keyword in content_lower for keyword in download_keywords)
        
        return has_youtube_url and has_download_request
    
    def _extract_youtube_url(self, content: str) -> Optional[str]:
        """Extract YouTube URL from message content"""
        import re
        
        # Common YouTube URL patterns
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtu\.be/[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[A-Za-z0-9_-]+',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                url = match.group()
                if self.validators.is_youtube_url(url):
                    return url
        
        return None
    
    async def _download_audio(self, youtube_url: str) -> Dict[str, Any]:
        """Download audio from YouTube URL"""
        try:
            video_id = self.validators.extract_video_id(youtube_url)
            if not video_id:
                return {"success": False, "error": "Could not extract video ID"}
            
            output_path = self.file_manager.get_temp_path("audio")
            output_template = str(output_path / f"{video_id}.%(ext)s")
            
            ydl_opts = {
                'format': 'bestaudio/best',
                'outtmpl': output_template,
                'extractaudio': True,
                'audioformat': 'wav',
                'audioquality': 0,  # Best quality
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    'preferredquality': '192',
                }],
                'postprocessor_args': [
                    '-ar', '16000'  # 16kHz sample rate for Whisper
                ],
                'prefer_ffmpeg': True,
                'keepvideo': False,
            }
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._download_with_ydl, 
                youtube_url, 
                ydl_opts
            )
            
            if result["success"]:
                # Find the downloaded file
                audio_file = output_path / f"{video_id}.wav"
                if audio_file.exists():
                    file_size = self.file_manager.get_file_size_mb(audio_file)
                    
                    # Check file size
                    if not self.validators.validate_file_size(file_size, self.config.max_file_size_mb):
                        return {
                            "success": False, 
                            "error": f"File too large: {file_size:.2f}MB > {self.config.max_file_size_mb}MB"
                        }
                    
                    result.update({
                        "file_path": str(audio_file),
                        "file_size_mb": file_size
                    })
                    
                    logger.info(f"Successfully downloaded audio: {audio_file}")
                    return result
                else:
                    return {"success": False, "error": "Downloaded file not found"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return {"success": False, "error": str(e)}
    
    def _download_with_ydl(self, url: str, ydl_opts: dict) -> Dict[str, Any]:
        """Download using yt-dlp (runs in thread pool)"""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown')
                duration = info.get('duration', 0)
                
                # Check duration (optional limit)
                max_duration = 3600  # 1 hour
                if duration > max_duration:
                    return {
                        "success": False, 
                        "error": f"Video too long: {duration}s > {max_duration}s"
                    }
                
                # Download
                ydl.download([url])
                
                return {
                    "success": True,
                    "title": title,
                    "duration": duration,
                    "video_id": self.validators.extract_video_id(url)
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")
        # Clean up any temporary files if needed
        pass