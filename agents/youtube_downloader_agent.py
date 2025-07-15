# agents/youtube_downloader_agent.py - COMPLETE SMART VERSION
import os
import yt_dlp
import asyncio
import re
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
    """Smart YouTube downloader agent with workflow awareness"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Downloads audio/video from YouTube with workflow intelligence")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
        # Workflow intelligence mappings
        self.language_codes = {
            'german': 'de', 'french': 'fr', 'spanish': 'es', 'italian': 'it',
            'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
            'chinese': 'zh', 'arabic': 'ar', 'hindi': 'hi', 'dutch': 'nl'
        }
        
        self.workflow_indicators = {
            'translation': ['translate', 'translation', 'german', 'french', 'spanish', 'italian'],
            'transcription': ['transcribe', 'transcript', 'text', 'speech to text'],
            'tts': ['voice', 'speech', 'audio generation', 'text to speech'],
            'summary': ['summarize', 'summary', 'brief', 'overview']
        }
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """🔧 SMART: Process with full conversation context awareness"""
        
        try:
            # Analyze complete conversation context
            context = self._analyze_conversation_context(messages)
            
            logger.info(f"🧠 Context Analysis: {context}")
            
            # Determine if we should process
            should_process, reason = self._should_process_smart(context)
            
            if not should_process:
                logger.info(f"⏭️  Skipping: {reason}")
                return Response(
                    chat_message=TextMessage(
                        content=f"YouTubeDownloader: {reason}",
                        source=self.name
                    )
                )
            
            # Determine download strategy
            download_strategy = self._determine_download_strategy(context)
            logger.info(f"📋 Download Strategy: {download_strategy}")
            
            if download_strategy == "skip":
                return Response(
                    chat_message=TextMessage(
                        content="✅ All required files already downloaded. WORKFLOW_COMPLETE",
                        source=self.name
                    )
                )
            
            # Execute download
            result = await self._download_content(context['youtube_url'], download_strategy, context)
            
            if result["success"]:
                files_msg = f"Files: {', '.join(result['files'])}" if result['files'] else "Processing complete"
                return Response(
                    chat_message=TextMessage(
                        content=f"✅ Downloaded {download_strategy} from YouTube. "
                               f"{files_msg}. "
                               f"Title: {result.get('title', 'Unknown')}. "
                               f"Duration: {result.get('duration', 0)} seconds.",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ Download failed: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in YouTubeDownloaderAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"Error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _analyze_conversation_context(self, messages) -> Dict[str, Any]:
        """🧠 CORE: Analyze complete conversation for workflow intelligence"""
        
        context = {
            'youtube_url': None,
            'video_id': None,
            'workflow_type': None,
            'target_language': None,
            'language_code': None,
            'audio_downloaded': False,
            'video_downloaded': False,
            'latest_message': "",
            'full_conversation': "",
            'user_request': "",
            'workflow_stage': 'initial'
        }
        
        if not messages:
            return context
        
        # Extract latest message
        latest_msg = messages[-1]
        context['latest_message'] = str(latest_msg.content) if hasattr(latest_msg, 'content') else ""
        
        # Build full conversation
        conversation_parts = []
        for msg in messages:
            if hasattr(msg, 'content') and hasattr(msg, 'source'):
                content = str(msg.content)
                source = str(msg.source)
                conversation_parts.append(f"{source}: {content}")
                
                # Capture original user request
                if source == 'user' and not context['user_request']:
                    context['user_request'] = content
        
        context['full_conversation'] = "\n".join(conversation_parts)
        conversation_lower = context['full_conversation'].lower()
        
        # Extract YouTube URL and video ID
        youtube_patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=([A-Za-z0-9_-]+)',
            r'https?://(?:www\.)?youtu\.be/([A-Za-z0-9_-]+)',
            r'https?://(?:www\.)?youtube\.com/embed/([A-Za-z0-9_-]+)'
        ]
        
        for pattern in youtube_patterns:
            match = re.search(pattern, context['full_conversation'])
            if match:
                context['youtube_url'] = match.group(0)
                context['video_id'] = match.group(1)
                break
        
        # Detect workflow type
        context['workflow_type'] = self._detect_workflow_type(conversation_lower)
        
        # Extract target language
        context['target_language'], context['language_code'] = self._extract_target_language(conversation_lower)
        
        # Check download status
        context['audio_downloaded'] = self._check_audio_downloaded(conversation_lower)
        context['video_downloaded'] = self._check_video_downloaded(conversation_lower)
        
        # Determine workflow stage
        context['workflow_stage'] = self._determine_workflow_stage(conversation_lower)
        
        return context
    
    def _detect_workflow_type(self, conversation_lower: str) -> str:
        """🔍 Detect the type of workflow from conversation"""
        
        for workflow_type, indicators in self.workflow_indicators.items():
            if any(indicator in conversation_lower for indicator in indicators):
                return workflow_type
        
        # Check for explicit download requests
        if any(word in conversation_lower for word in ['download', 'get', 'fetch', 'save']):
            if 'video' in conversation_lower:
                return 'video_download'
            elif 'audio' in conversation_lower:
                return 'audio_download'
        
        return 'simple_download'
    
    def _extract_target_language(self, conversation_lower: str) -> tuple[Optional[str], Optional[str]]:
        """🌍 Extract target language from conversation"""
        
        for language, code in self.language_codes.items():
            if language in conversation_lower:
                return language, code
        
        # Check for language codes
        for language, code in self.language_codes.items():
            if code in conversation_lower:
                return language, code
        
        return None, None
    
    def _check_audio_downloaded(self, conversation_lower: str) -> bool:
        """🎵 Check if audio was already downloaded"""
        
        audio_indicators = [
            'downloaded audio', 'audio downloaded', 'audio.wav', 
            'audio file', '_audio.wav', 'extracted audio',
            'successfully downloaded audio'
        ]
        
        return any(indicator in conversation_lower for indicator in audio_indicators)
    
    def _check_video_downloaded(self, conversation_lower: str) -> bool:
        """🎥 Check if video was already downloaded"""
        
        video_indicators = [
            'downloaded video', 'video downloaded', 'video.mp4',
            'video file', '_video.mp4', 'extracted video',
            'successfully downloaded video'
        ]
        
        return any(indicator in conversation_lower for indicator in video_indicators)
    
    def _determine_workflow_stage(self, conversation_lower: str) -> str:
        """📊 Determine current stage of workflow"""
        
        if 'merged' in conversation_lower or 'final video' in conversation_lower:
            return 'complete'
        elif 'generated' in conversation_lower and 'audio' in conversation_lower:
            return 'tts_complete'
        elif 'translated' in conversation_lower:
            return 'translation_complete'
        elif 'transcribed' in conversation_lower or 'transcript' in conversation_lower:
            return 'transcription_complete'
        elif 'downloaded' in conversation_lower:
            return 'download_complete'
        else:
            return 'initial'
    
    def _should_process_smart(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """🤖 SMART: Intelligent decision making for processing"""
        
        # No YouTube URL found
        if not context['youtube_url']:
            return False, "No YouTube URL found in conversation"
        
        # Direct YouTube URL in latest message (always process)
        if context['youtube_url'] in context['latest_message']:
            return True, "Direct YouTube URL request detected"
        
        # Workflow-based decisions
        workflow_type = context['workflow_type']
        
        if workflow_type == 'translation':
            return self._should_process_translation_workflow(context)
        elif workflow_type == 'transcription':
            return self._should_process_transcription_workflow(context)
        elif workflow_type in ['tts', 'summary']:
            return self._should_process_audio_workflow(context)
        elif workflow_type in ['video_download', 'audio_download']:
            return self._should_process_simple_download(context)
        
        # Check if other agents need files
        latest_lower = context['latest_message'].lower()
        if any(phrase in latest_lower for phrase in [
            'need video', 'require video', 'video not found', 'missing video',
            'need audio', 'require audio', 'audio not found', 'missing audio'
        ]):
            return True, "Other agent requesting missing files"
        
        return False, "No processing triggers found"
    
    def _should_process_translation_workflow(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """🌍 Translation workflow processing logic"""
        
        audio_downloaded = context['audio_downloaded']
        video_downloaded = context['video_downloaded']
        workflow_stage = context['workflow_stage']
        
        if workflow_stage == 'initial':
            if not audio_downloaded and not video_downloaded:
                return True, "Translation workflow: downloading both audio and video"
        elif workflow_stage == 'transcription_complete':
            if audio_downloaded and not video_downloaded:
                return True, "Translation workflow: need video for final merge"
        elif workflow_stage == 'tts_complete':
            if not video_downloaded:
                return True, "Translation workflow: need video for merging with new audio"
        elif workflow_stage == 'complete':
            return False, "Translation workflow already complete"
        
        return False, "Translation workflow: no download needed at this stage"
    
    def _should_process_transcription_workflow(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """📝 Transcription workflow processing logic"""
        
        if not context['audio_downloaded']:
            return True, "Transcription workflow: need audio file"
        
        return False, "Transcription workflow: audio already available"
    
    def _should_process_audio_workflow(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """🎵 Audio-focused workflow processing logic"""
        
        if not context['audio_downloaded']:
            return True, "Audio workflow: need audio file"
        
        return False, "Audio workflow: audio already available"
    
    def _should_process_simple_download(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """📥 Simple download processing logic"""
        
        workflow_type = context['workflow_type']
        
        if workflow_type == 'video_download' and not context['video_downloaded']:
            return True, "Simple video download requested"
        elif workflow_type == 'audio_download' and not context['audio_downloaded']:
            return True, "Simple audio download requested"
        
        return False, "Simple download: files already available"
    
    def _determine_download_strategy(self, context: Dict[str, Any]) -> str:
        """📋 STRATEGY: Determine what to download based on context"""
        
        workflow_type = context['workflow_type']
        audio_downloaded = context['audio_downloaded']
        video_downloaded = context['video_downloaded']
        workflow_stage = context['workflow_stage']
        
        if workflow_type == 'translation':
            if workflow_stage == 'initial':
                if not audio_downloaded and not video_downloaded:
                    return "both"  # Smart: get everything upfront
            elif workflow_stage in ['transcription_complete', 'translation_complete', 'tts_complete']:
                if audio_downloaded and not video_downloaded:
                    return "video"  # Need video for merging
                elif not audio_downloaded and video_downloaded:
                    return "audio"  # Need audio for transcription
            elif workflow_stage == 'complete':
                return "skip"  # All done
        
        elif workflow_type == 'transcription':
            if not audio_downloaded:
                return "audio"
            else:
                return "skip"
        
        elif workflow_type in ['tts', 'summary']:
            if not audio_downloaded:
                return "audio"
            else:
                return "skip"
        
        elif workflow_type == 'video_download':
            if not video_downloaded:
                return "video"
            else:
                return "skip"
        
        elif workflow_type == 'audio_download':
            if not audio_downloaded:
                return "audio"
            else:
                return "skip"
        
        # Default strategy based on latest message
        latest_lower = context['latest_message'].lower()
        if 'video' in latest_lower and not video_downloaded:
            return "video"
        elif 'audio' in latest_lower and not audio_downloaded:
            return "audio"
        elif not audio_downloaded:
            return "audio"  # Default to audio
        
        return "skip"
    
    async def _download_content(self, youtube_url: str, download_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """🚀 Execute the download based on strategy"""
        
        try:
            video_id = context.get('video_id') or self.validators.extract_video_id(youtube_url)
            if not video_id:
                return {"success": False, "error": "Could not extract video ID"}
            
            logger.info(f"🎯 Downloading {download_type} for video {video_id}")
            
            results = {"success": True, "files": [], "title": "", "duration": 0}
            
            # Execute downloads based on strategy
            if download_type in ["audio", "both"]:
                audio_result = await self._download_audio(youtube_url, video_id)
                if audio_result["success"]:
                    results["files"].append(audio_result["file_path"])
                    results["title"] = audio_result["title"]
                    results["duration"] = audio_result["duration"]
                    logger.info(f"✅ Audio download successful: {audio_result['file_path']}")
                else:
                    logger.error(f"❌ Audio download failed: {audio_result['error']}")
                    return audio_result
            
            if download_type in ["video", "both"]:
                video_result = await self._download_video(youtube_url, video_id)
                if video_result["success"]:
                    results["files"].append(video_result["file_path"])
                    if not results["title"]:  # Set title if not already set by audio
                        results["title"] = video_result["title"]
                        results["duration"] = video_result["duration"]
                    logger.info(f"✅ Video download successful: {video_result['file_path']}")
                else:
                    logger.error(f"❌ Video download failed: {video_result['error']}")
                    return video_result
            
            return results
            
        except Exception as e:
            logger.error(f"Error in download execution: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_audio(self, youtube_url: str, video_id: str) -> Dict[str, Any]:
        """🎵 Download audio with optimized settings"""
        
        try:
            output_path = self.file_manager.get_temp_path("audio")
            output_template = str(output_path / f"{video_id}_audio.%(ext)s")
            
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
                'postprocessor_args': ['-ar', '16000'],  # 16kHz for Whisper
                'prefer_ffmpeg': True,
                'keepvideo': False,
            }
            
            # Execute download in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._download_with_ydl, youtube_url, ydl_opts)
            
            if result["success"]:
                # Find the downloaded audio file
                audio_file = output_path / f"{video_id}_audio.wav"
                if audio_file.exists():
                    file_size = self.file_manager.get_file_size_mb(audio_file)
                    
                    # Validate file size
                    if not self.validators.validate_file_size(file_size, self.config.max_file_size_mb):
                        return {
                            "success": False,
                            "error": f"Audio file too large: {file_size:.2f}MB > {self.config.max_file_size_mb}MB"
                        }
                    
                    result["file_path"] = str(audio_file)
                    result["file_size_mb"] = file_size
                    return result
                else:
                    return {"success": False, "error": "Audio file not found after download"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading audio: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_video(self, youtube_url: str, video_id: str) -> Dict[str, Any]:
        """🎥 Download video with optimized settings"""
        
        try:
            output_path = self.file_manager.get_temp_path("video")
            output_template = str(output_path / f"{video_id}_video.%(ext)s")
            
            ydl_opts = {
                'format': 'best[ext=mp4]/best[ext=webm]/best',  # Prefer MP4, fallback to WebM
                'outtmpl': output_template,
                'writeinfojson': True,  # Save video metadata
                'writethumbnail': False,  # Skip thumbnail
            }
            
            # Execute download in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, self._download_with_ydl, youtube_url, ydl_opts)
            
            if result["success"]:
                # Find the downloaded video file
                video_extensions = ['.mp4', '.webm', '.mkv', '.avi']
                video_files = []
                
                for ext in video_extensions:
                    video_file = output_path / f"{video_id}_video{ext}"
                    if video_file.exists():
                        video_files.append(video_file)
                
                if video_files:
                    # Use the first found video file
                    video_file = video_files[0]
                    file_size = self.file_manager.get_file_size_mb(video_file)
                    
                    # Validate file size
                    max_video_size = self.config.max_file_size_mb * 5  # Allow larger videos
                    if file_size > max_video_size:
                        return {
                            "success": False,
                            "error": f"Video file too large: {file_size:.2f}MB > {max_video_size}MB"
                        }
                    
                    result["file_path"] = str(video_file)
                    result["file_size_mb"] = file_size
                    return result
                else:
                    return {"success": False, "error": "Video file not found after download"}
            
            return result
            
        except Exception as e:
            logger.error(f"Error downloading video: {e}")
            return {"success": False, "error": str(e)}
    
    def _download_with_ydl(self, url: str, ydl_opts: dict) -> Dict[str, Any]:
        """⚙️ Execute download using yt-dlp (runs in thread pool)"""
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract video information first
                info = ydl.extract_info(url, download=False)
                title = info.get('title', 'Unknown Video')
                duration = info.get('duration', 0)
                uploader = info.get('uploader', 'Unknown')
                
                logger.info(f"📹 Video Info: {title} by {uploader} ({duration}s)")
                
                # Validate duration
                max_duration = 3600  # 1 hour limit
                if duration > max_duration:
                    return {
                        "success": False,
                        "error": f"Video too long: {duration}s > {max_duration}s"
                    }
                
                # Download the content
                ydl.download([url])
                
                return {
                    "success": True,
                    "title": title,
                    "duration": duration,
                    "uploader": uploader,
                    "video_id": self.validators.extract_video_id(url)
                }
                
        except Exception as e:
            logger.error(f"yt-dlp download error: {e}")
            return {"success": False, "error": str(e)}
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """🔄 Reset agent state"""
        logger.info(f"{self.name} reset - clearing workflow context")
        # Could add workflow state cleanup here if needed