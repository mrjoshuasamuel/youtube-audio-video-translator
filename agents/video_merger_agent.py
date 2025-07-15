# agents/video_merger_agent.py
import asyncio
import yt_dlp
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import logging
import subprocess
import json

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager, Validators
from moviepy.editor import VideoFileClip, AudioFileClip

logger = logging.getLogger(__name__)

class VideoMergerAgent(BaseChatAgent):
    """Agent responsible for merging translated audio with original video"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Merges translated audio with original video")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages for video merging requests"""
        
        latest_message = messages[-1] if messages else None
        if not latest_message:
            return Response(
                chat_message=TextMessage(
                    content="No message provided.",
                    source=self.name
                )
            )
        
        content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
        # Check if this agent should process the message
        if not self._should_process(content):
            return Response(
                chat_message=TextMessage(
                    content="This agent merges translated audio with original video.",
                    source=self.name
                )
            )
        
        try:
            # Extract information needed for merging
            merge_info = self._extract_merge_info(content)
            
            if not merge_info["audio_file"] or not merge_info["youtube_url"]:
                return Response(
                    chat_message=TextMessage(
                        content="Could not find both translated audio file and YouTube URL for merging.",
                        source=self.name
                    )
                )
            
            # Perform video merging
            result = await self._merge_video_audio(merge_info)
            
            if result["success"]:
                return Response(
                    chat_message=TextMessage(
                        content=result["message"],
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"Video merging error: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in VideoMergerAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        keywords = [
            "merge video", "combine audio", "video with audio", "replace audio",
            "merge translated audio", "final video", "video output"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _extract_merge_info(self, content: str) -> Dict[str, Any]:
        """Extract information needed for video merging"""
        merge_info = {
            "audio_file": None,
            "youtube_url": None,
            "language": None
        }
        
        # Find translated audio file
        audio_file = self._find_translated_audio_file()
        if audio_file:
            merge_info["audio_file"] = audio_file
            # Extract language from filename
            merge_info["language"] = self._extract_language_from_filename(audio_file.name)
        
        # Extract YouTube URL from conversation history or content
        youtube_url = self._extract_youtube_url(content)
        if youtube_url:
            merge_info["youtube_url"] = youtube_url
        
        return merge_info
    
    def _find_translated_audio_file(self) -> Optional[Path]:
        """Find the most recent translated audio file"""
        audio_folder = self.file_manager.get_temp_path("", "audio")
        if audio_folder.exists():
            # Look for TTS generated files
            tts_files = list(audio_folder.glob("tts_*.wav"))
            if tts_files:
                return max(tts_files, key=lambda p: p.stat().st_mtime)
        return None
    
    def _extract_language_from_filename(self, filename: str) -> Optional[str]:
        """Extract language code from audio filename"""
        for lang_code in ['en', 'es', 'fr', 'de', 'it', 'pt', 'ru', 'ja', 'ko', 'zh']:
            if lang_code in filename:
                return lang_code
        return None
    
    def _extract_youtube_url(self, content: str) -> Optional[str]:
        """Extract YouTube URL from content or find from conversation history"""
        import re
        
        # Look for YouTube URL in current content
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtu\.be/[A-Za-z0-9_-]+',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group()
        
        # If not found, check for video ID and construct URL
        video_id_match = re.search(r'video[_\s]?id[:\s]*([A-Za-z0-9_-]{11})', content, re.IGNORECASE)
        if video_id_match:
            return f"https://www.youtube.com/watch?v={video_id_match.group(1)}"
        
        return None
    
    async def _merge_video_audio(self, merge_info: Dict[str, Any]) -> Dict[str, Any]:
        """Merge translated audio with original video"""
        try:
            # Download original video
            video_path = await self._download_video(merge_info["youtube_url"])
            if not video_path:
                return {"success": False, "error": "Failed to download original video"}
            
            # Merge audio and video
            merged_video_path = await self._perform_merge(
                video_path, 
                merge_info["audio_file"], 
                merge_info["language"]
            )
            
            if merged_video_path:
                # Move to output folder
                output_filename = f"translated_video_{merge_info['language']}.mp4"
                final_output_path = self.file_manager.get_output_path(output_filename)
                
                # Copy to final output location
                import shutil
                shutil.copy2(merged_video_path, final_output_path)
                
                return {
                    "success": True,
                    "message": f"Successfully merged translated audio with video. "
                             f"Final video saved to: {final_output_path}",
                    "output_path": str(final_output_path),
                    "language": merge_info["language"]
                }
            else:
                return {"success": False, "error": "Failed to merge audio and video"}
                
        except Exception as e:
            logger.error(f"Error in video merging: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_video(self, youtube_url: str) -> Optional[str]:
        """Download original video from YouTube"""
        try:
            video_id = self.validators.extract_video_id(youtube_url)
            if not video_id:
                logger.error("Could not extract video ID")
                return None
            
            output_path = self.file_manager.get_temp_path("video")
            output_template = str(output_path / f"{video_id}.%(ext)s")
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best',
                'outtmpl': output_template,
                'writeinfojson': True,  # Save video info
            }
            
            # Run in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, 
                self._download_video_with_ydl, 
                youtube_url, 
                ydl_opts
            )
            
            if result["success"]:
                video_file = output_path / f"{video_id}.mp4"
                if video_file.exists():
                    return str(video_file)
            
            return None
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return None
    
    def _download_video_with_ydl(self, url: str, ydl_opts: dict) -> Dict[str, Any]:
        """Download video using yt-dlp (runs in thread pool)"""
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
                return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _perform_merge(self, video_path: str, audio_path: Path, language: str) -> Optional[str]:
        """Perform the actual merging of video and audio"""
        try:
            # Use moviepy for merging
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._merge_with_moviepy,
                video_path,
                str(audio_path),
                language
            )
            
            return result if result else None
            
        except Exception as e:
            logger.error(f"Error in merge operation: {e}")
            return None
    
    def _merge_with_moviepy(self, video_path: str, audio_path: str, language: str) -> Optional[str]:
        """Merge video and audio using moviepy (runs in thread pool)"""
        try:
            # Load video and audio
            video_clip = VideoFileClip(video_path)
            audio_clip = AudioFileClip(audio_path)
            
            # Adjust audio duration to match video if needed
            video_duration = video_clip.duration
            audio_duration = audio_clip.duration
            
            if audio_duration < video_duration:
                # If audio is shorter, we might need to adjust or extend
                logger.warning(f"Audio ({audio_duration}s) is shorter than video ({video_duration}s)")
                # Option 1: Cut video to match audio
                video_clip = video_clip.subclip(0, audio_duration)
            elif audio_duration > video_duration:
                # If audio is longer, cut audio to match video
                logger.warning(f"Audio ({audio_duration}s) is longer than video ({video_duration}s)")
                audio_clip = audio_clip.subclip(0, video_duration)
            
            # Replace video audio with new audio
            final_clip = video_clip.set_audio(audio_clip)
            
            # Generate output filename
            output_filename = f"merged_video_{language}_{Path(video_path).stem}.mp4"
            output_path = self.file_manager.get_temp_path(output_filename, "video")
            
            # Export the final video
            final_clip.write_videofile(
                str(output_path),
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                verbose=False,
                logger=None
            )
            
            # Clean up clips
            video_clip.close()
            audio_clip.close()
            final_clip.close()
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error in moviepy merge: {e}")
            return None
    
    async def _merge_with_ffmpeg(self, video_path: str, audio_path: str, language: str) -> Optional[str]:
        """Alternative merge method using FFmpeg directly"""
        try:
            output_filename = f"merged_video_{language}_{Path(video_path).stem}.mp4"
            output_path = self.file_manager.get_temp_path(output_filename, "video")
            
            # FFmpeg command to replace audio
            cmd = [
                'ffmpeg',
                '-i', video_path,           # Input video
                '-i', audio_path,           # Input audio
                '-c:v', 'copy',             # Copy video stream
                '-c:a', 'aac',              # Encode audio to AAC
                '-map', '0:v:0',            # Map video from first input
                '-map', '1:a:0',            # Map audio from second input
                '-shortest',                # Finish when shortest stream ends
                '-y',                       # Overwrite output file
                str(output_path)
            ]
            
            # Run FFmpeg
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return str(output_path)
            else:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                return None
                
        except Exception as e:
            logger.error(f"Error in FFmpeg merge: {e}")
            return None
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")