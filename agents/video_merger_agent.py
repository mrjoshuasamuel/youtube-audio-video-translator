# agents/video_merger_agent.py - COMPLETE SMART VERSION
import os
import asyncio
import json
import logging
import subprocess
import shutil
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
from datetime import datetime
import re

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager, Validators

logger = logging.getLogger(__name__)

class VideoMergerAgent(BaseChatAgent):
    """Smart video merger agent with workflow awareness and FFmpeg integration"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Merges translated audio with original video using FFmpeg with workflow intelligence")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
        # Video merger settings
        self.supported_video_formats = ['.mp4', '.webm', '.mkv', '.avi', '.mov']
        self.supported_audio_formats = ['.wav', '.mp3', '.aac', '.m4a']
        self.output_format = '.mp4'  # Standard output format
        self.video_codec = 'libx264'  # H.264 for compatibility
        self.audio_codec = 'aac'  # AAC for compatibility
        self.max_file_size_gb = 2.0  # 2GB limit for safety
        
        # Quality settings
        self.video_quality_crf = 23  # Constant Rate Factor (18-28 range, 23 is default)
        self.audio_bitrate = '128k'  # Audio bitrate
        self.video_preset = 'medium'  # Encoding speed vs compression
        
        # Workflow stage indicators
        self.workflow_stages = {
            'download_complete': ['downloaded', 'download complete', 'files:', 'audio.wav', 'video.mp4'],
            'transcription_complete': ['transcribed', 'transcript saved', 'transcription complete'],
            'translation_complete': ['translated', 'translation complete', 'translation saved'],
            'tts_complete': ['generated tts', 'tts complete', 'audio generated', 'tts audio'],
            'merge_complete': ['merged', 'final video', 'workflow complete', 'merged video']
        }
        
        # Check FFmpeg availability
        self.ffmpeg_available = self._check_ffmpeg()
        
        logger.info(f"✅ VideoMergerAgent '{name}' initialized (FFmpeg available: {self.ffmpeg_available})")
    
    def _check_ffmpeg(self) -> bool:
        """🔧 Check if FFmpeg is available"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, 
                                  text=True, 
                                  timeout=10)
            if result.returncode == 0:
                logger.info("✅ FFmpeg found and available")
                return True
            else:
                logger.error("❌ FFmpeg not working properly")
                return False
        except FileNotFoundError:
            logger.error("❌ FFmpeg not found. Please install FFmpeg")
            return False
        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg check timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Error checking FFmpeg: {e}")
            return False
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """🔧 SMART: Process with full conversation context awareness"""
        
        try:
            # Check FFmpeg availability first
            if not self.ffmpeg_available:
                return Response(
                    chat_message=TextMessage(
                        content="❌ FFmpeg not available. Please install FFmpeg to merge video and audio.",
                        source=self.name
                    )
                )
            
            # Analyze complete conversation context
            context = self._analyze_conversation_context(messages)
            
            logger.info(f"🧠 VideoMergerAgent Context Analysis: {context}")
            
            # Determine if we should process
            should_process, reason = self._should_process_smart(context)
            
            if not should_process:
                logger.info(f"⏭️  VideoMergerAgent Skipping: {reason}")
                return Response(
                    chat_message=TextMessage(
                        content=f"VideoMergerAgent: {reason}",
                        source=self.name
                    )
                )
            
            # Find required files for merging
            files_result = self._find_merge_files(context)
            
            if not files_result["success"]:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ {files_result['error']}",
                        source=self.name
                    )
                )
            
            # Execute video merging
            result = await self._merge_video_audio(files_result["files"], context)
            
            if result["success"]:
                # Workflow always completes after video merging
                completion_message = " WORKFLOW_COMPLETE"
                
                return Response(
                    chat_message=TextMessage(
                        content=f"✅ Merged translated audio with original video. "
                               f"Final video saved to {result['output_path']}. "
                               f"Duration: {result['duration']:.1f}s. "
                               f"Size: {result['file_size_mb']:.1f}MB. "
                               f"Quality: {result['quality_score']:.2f}/10.{completion_message}",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ Video merging failed: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in VideoMergerAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"❌ VideoMergerAgent error: {str(e)}",
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
            'target_language_code': None,
            'audio_downloaded': False,
            'video_downloaded': False,
            'transcription_complete': False,
            'translation_complete': False,
            'tts_complete': False,
            'merge_complete': False,
            'latest_message': "",
            'full_conversation': "",
            'user_request': "",
            'workflow_stage': 'initial',
            'video_file_mentioned': None,
            'tts_audio_file_mentioned': None,
            'merge_requested': False,
            'output_preference': None
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
        context['target_language'], context['target_language_code'] = self._extract_target_language(conversation_lower)
        
        # Check completion status
        context['audio_downloaded'] = self._check_audio_downloaded(conversation_lower)
        context['video_downloaded'] = self._check_video_downloaded(conversation_lower)
        context['transcription_complete'] = self._check_transcription_complete(conversation_lower)
        context['translation_complete'] = self._check_translation_complete(conversation_lower)
        context['tts_complete'] = self._check_tts_complete(conversation_lower)
        context['merge_complete'] = self._check_merge_complete(conversation_lower)
        
        # Determine workflow stage
        context['workflow_stage'] = self._determine_workflow_stage(conversation_lower)
        
        # Check for explicit merge request
        context['merge_requested'] = any(keyword in conversation_lower for keyword in [
            'merge', 'combine', 'join', 'mix', 'overlay', 'final video'
        ])
        
        # Look for mentioned video files
        video_file_match = re.search(r'([^\s]+_video\.(mp4|webm|mkv|avi))', context['full_conversation'])
        if video_file_match:
            context['video_file_mentioned'] = video_file_match.group(1)
        
        # Look for mentioned TTS audio files
        tts_audio_match = re.search(r'([^\s]+_tts_[^\s]*\.(wav|mp3|aac))', context['full_conversation'])
        if tts_audio_match:
            context['tts_audio_file_mentioned'] = tts_audio_match.group(1)
        
        # Extract output preference if mentioned
        context['output_preference'] = self._extract_output_preference(conversation_lower)
        
        return context
    
    def _detect_workflow_type(self, conversation_lower: str) -> str:
        """🔍 Detect the type of workflow from conversation"""
        
        if any(word in conversation_lower for word in ['translate', 'translation', 'german', 'french', 'spanish']):
            return 'translation'
        elif any(word in conversation_lower for word in ['merge', 'combine', 'join', 'mix']):
            return 'merge'
        elif any(word in conversation_lower for word in ['tts', 'text to speech', 'voice', 'speech']):
            return 'tts'
        elif any(word in conversation_lower for word in ['transcribe', 'transcript', 'speech to text']):
            return 'transcription'
        
        return 'simple_merge'
    
    def _extract_target_language(self, conversation_lower: str) -> tuple[Optional[str], Optional[str]]:
        """🌍 Extract target language from conversation"""
        
        # Language detection mappings
        language_codes = {
            'german': 'de', 'french': 'fr', 'spanish': 'es', 'italian': 'it',
            'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
            'chinese': 'zh', 'arabic': 'ar', 'hindi': 'hi', 'dutch': 'nl'
        }
        
        # Look for "to [language]" patterns
        to_language_patterns = [
            r'to\s+(\w+)',
            r'in\s+(\w+)',
            r'convert\s+to\s+(\w+)',
            r'translate.*to\s+(\w+)'
        ]
        
        for pattern in to_language_patterns:
            match = re.search(pattern, conversation_lower)
            if match:
                potential_lang = match.group(1).lower()
                if potential_lang in language_codes:
                    return potential_lang, language_codes[potential_lang]
        
        # Look for language detection from previous agents
        language_patterns = [
            r'generated tts.*in\s+(\w+)',
            r'translated.*to\s+(\w+)',
            r'target language:\s*(\w+)',
            r'language:\s*(\w+)'
        ]
        
        for pattern in language_patterns:
            match = re.search(pattern, conversation_lower)
            if match:
                potential_lang = match.group(1).lower()
                if potential_lang in language_codes:
                    return potential_lang, language_codes[potential_lang]
        
        # Direct language detection
        for language, code in language_codes.items():
            if language in conversation_lower:
                return language, code
        
        return None, None
    
    def _extract_output_preference(self, conversation_lower: str) -> Optional[str]:
        """📁 Extract output file preference from conversation"""
        
        output_patterns = [
            r'save.*to\s+([^\s]+)',
            r'output.*to\s+([^\s]+)',
            r'final.*file\s+([^\s]+)',
            r'save as\s+([^\s]+)'
        ]
        
        for pattern in output_patterns:
            match = re.search(pattern, conversation_lower)
            if match:
                return match.group(1)
        
        return None
    
    def _check_audio_downloaded(self, conversation_lower: str) -> bool:
        """🎵 Check if audio was downloaded"""
        
        audio_indicators = [
            'downloaded audio', 'audio downloaded', 'audio.wav', 
            'audio file', '_audio.wav', 'extracted audio'
        ]
        
        return any(indicator in conversation_lower for indicator in audio_indicators)
    
    def _check_video_downloaded(self, conversation_lower: str) -> bool:
        """🎥 Check if video was downloaded"""
        
        video_indicators = [
            'downloaded video', 'video downloaded', 'video.mp4',
            'video file', '_video.mp4', 'extracted video', 'downloaded both'
        ]
        
        return any(indicator in conversation_lower for indicator in video_indicators)
    
    def _check_transcription_complete(self, conversation_lower: str) -> bool:
        """📝 Check if transcription is complete"""
        
        transcription_indicators = [
            'transcribed', 'transcript saved', 'transcription complete',
            'transcript.txt', 'text transcribed', 'transcribed audio to text'
        ]
        
        return any(indicator in conversation_lower for indicator in transcription_indicators)
    
    def _check_translation_complete(self, conversation_lower: str) -> bool:
        """🌍 Check if translation is complete"""
        
        translation_indicators = [
            '✅ translated', 'translation saved to', 'translated transcript to',
            'translation complete.', 'translation finished.'
        ]
        
        return any(indicator in conversation_lower for indicator in translation_indicators)
    
    def _check_tts_complete(self, conversation_lower: str) -> bool:
        """🎙️ Check if TTS is complete"""
        
        # Only check for SUCCESS indicators
        tts_success_indicators = [
            '✅ generated tts',
            'tts audio saved to',
            'audio saved to',
            'generated tts audio in'
        ]
        
        # Must find success indicator AND exclude error messages
        has_success = any(indicator in conversation_lower for indicator in tts_success_indicators)
        has_error = any(error in conversation_lower for error in [
            'tts not complete',
            'cannot merge without tts',
            'tts failed'
        ])
        
        return has_success and not has_error
    
    def _check_merge_complete(self, conversation_lower: str) -> bool:
        """🎬 Check if merge is already complete"""
        
        merge_indicators = [
            'merged', 'final video', 'workflow complete', 'video merging complete',
            'merged video', 'combine complete'
        ]
        
        return any(indicator in conversation_lower for indicator in merge_indicators)
    
    def _determine_workflow_stage(self, conversation_lower: str) -> str:
        """📊 Determine current stage of workflow"""
        
        if any(indicator in conversation_lower for indicator in self.workflow_stages['merge_complete']):
            return 'merge_complete'
        elif any(indicator in conversation_lower for indicator in self.workflow_stages['tts_complete']):
            return 'tts_complete'
        elif any(indicator in conversation_lower for indicator in self.workflow_stages['translation_complete']):
            return 'translation_complete'
        elif any(indicator in conversation_lower for indicator in self.workflow_stages['transcription_complete']):
            return 'transcription_complete'
        elif any(indicator in conversation_lower for indicator in self.workflow_stages['download_complete']):
            return 'download_complete'
        else:
            return 'initial'
    
    def _should_process_smart(self, context: Dict[str, Any]) -> tuple[bool, str]:
        """🤖 SMART: Intelligent decision making for video merging"""
        
        # Check if merge is already complete
        if context['merge_complete']:
            return False, "Video merging already completed"
        
        # Check if TTS is complete (prerequisite for merging)
        if not context['tts_complete']:
            return False, "TTS not complete - cannot merge without translated audio"
        
        # Check if video is available (prerequisite for merging)
        if not context['video_downloaded']:
            return False, "Original video not available - cannot merge"
        
        # Workflow-based decisions
        workflow_type = context['workflow_type']
        workflow_stage = context['workflow_stage']
        
        if workflow_type == 'translation':
            # Translation workflow with video merging
            if workflow_stage == 'tts_complete':
                return True, "Translation workflow: TTS audio ready for video merging"
        elif workflow_type == 'merge':
            # Direct merge workflow
            if workflow_stage == 'tts_complete':
                return True, "Merge workflow: ready to combine video and TTS audio"
        
        # Check for explicit merge request
        if context['merge_requested']:
            return True, "Explicit video merge request detected"
        
        # Check if latest message mentions TTS completion
        latest_lower = context['latest_message'].lower()
        if any(phrase in latest_lower for phrase in [
            'generated tts', 'tts complete', 'audio generated', 'tts audio'
        ]):
            return True, "TTS just completed - ready for video merging"
        
        return False, "No video merge triggers found"
    
    def _find_merge_files(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """🔍 Find the video and TTS audio files for merging"""
        
        files = {
            "video_file": None,
            "audio_file": None
        }
        
        # Find original video file
        video_file = self._find_video_file(context)
        if not video_file:
            return {
                "success": False,
                "error": "Original video file not found. Please download video first."
            }
        
        files["video_file"] = video_file
        
        # Find TTS audio file
        audio_file = self._find_tts_audio_file(context)
        if not audio_file:
            return {
                "success": False,
                "error": "TTS audio file not found. Please generate TTS audio first."
            }
        
        files["audio_file"] = audio_file
        
        # Validate files exist and are readable
        video_path = Path(video_file)
        audio_path = Path(audio_file)
        
        if not video_path.exists():
            return {
                "success": False,
                "error": f"Video file not found: {video_file}"
            }
        
        if not audio_path.exists():
            return {
                "success": False,
                "error": f"Audio file not found: {audio_file}"
            }
        
        # Check file sizes
        video_size_mb = self.file_manager.get_file_size_mb(video_path)
        audio_size_mb = self.file_manager.get_file_size_mb(audio_path)
        
        if video_size_mb > self.max_file_size_gb * 1024:
            return {
                "success": False,
                "error": f"Video file too large: {video_size_mb:.1f}MB > {self.max_file_size_gb * 1024}MB"
            }
        
        logger.info(f"📁 Found merge files - Video: {video_file} ({video_size_mb:.1f}MB), Audio: {audio_file} ({audio_size_mb:.1f}MB)")
        
        return {
            "success": True,
            "files": files,
            "video_size_mb": video_size_mb,
            "audio_size_mb": audio_size_mb
        }
    
    def _find_video_file(self, context: Dict[str, Any]) -> Optional[str]:
        """🎥 Find the original video file"""
        
        # Try to find video file based on video ID
        if context['video_id']:
            video_path = self.file_manager.get_temp_path("video")
            
            for ext in self.supported_video_formats:
                video_file = video_path / f"{context['video_id']}_video{ext}"
                if video_file.exists():
                    return str(video_file)
        
        # Try to find video file mentioned in conversation
        if context['video_file_mentioned']:
            video_path = self.file_manager.get_temp_path("video")
            video_file = video_path / context['video_file_mentioned']
            
            if video_file.exists():
                return str(video_file)
        
        # Search for any video files in video directory
        video_path = self.file_manager.get_temp_path("video")
        if video_path.exists():
            for ext in self.supported_video_formats:
                for video_file in video_path.glob(f"*{ext}"):
                    return str(video_file)
        
        return None
    
    def _find_tts_audio_file(self, context: Dict[str, Any]) -> Optional[str]:
        """🎙️ Find the TTS audio file"""
        
        # Try to find TTS audio file based on video ID and target language
        if context['video_id'] and context['target_language_code']:
            audio_path = self.file_manager.get_temp_path("audio")
            
            lang_code = context['target_language_code'].split('-')[0]
            tts_patterns = [
                f"{context['video_id']}_tts_{lang_code}.wav",
                f"{context['video_id']}_tts_{context['target_language'].lower()}.wav",
                f"tts_{lang_code}.wav",
                f"tts_{context['target_language'].lower()}.wav"
            ]
            
            for pattern in tts_patterns:
                audio_file = audio_path / pattern
                if audio_file.exists():
                    return str(audio_file)
        
        # Try to find TTS audio file mentioned in conversation
        if context['tts_audio_file_mentioned']:
            audio_path = self.file_manager.get_temp_path("audio")
            audio_file = audio_path / context['tts_audio_file_mentioned']
            
            if audio_file.exists():
                return str(audio_file)
        
        # Search for any TTS audio files in audio directory
        audio_path = self.file_manager.get_temp_path("audio")
        if audio_path.exists():
            # Look for files with TTS indicators
            for audio_file in audio_path.glob("*tts*.wav"):
                return str(audio_file)
            
            for audio_file in audio_path.glob("*tts*.mp3"):
                return str(audio_file)
        
        return None
    
    async def _merge_video_audio(self, files: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """🎬 Merge video and TTS audio using FFmpeg"""
        
        try:
            video_file = files["video_file"]
            audio_file = files["audio_file"]
            
            logger.info(f"🎬 Merging video: {video_file} with TTS audio: {audio_file}")
            
            # Prepare output file
            output_result = self._prepare_output_file(context)
            if not output_result["success"]:
                return output_result
            
            output_file = output_result["output_file"]
            
            # Build FFmpeg command
            ffmpeg_cmd = self._build_ffmpeg_command(video_file, audio_file, output_file)
            
            # Execute FFmpeg merge
            merge_result = await self._execute_ffmpeg_merge(ffmpeg_cmd, output_file)
            
            if merge_result["success"]:
                # Get output file information
                file_info = self._get_output_file_info(output_file)
                
                # Save merge metadata
                metadata_result = await self._save_merge_metadata(output_file, context, {
                    "video_file": video_file,
                    "audio_file": audio_file,
                    "ffmpeg_command": ' '.join(ffmpeg_cmd),
                    **file_info
                })
                
                # Move to final output folder
                final_output = await self._move_to_output_folder(output_file, context)
                
                logger.info(f"✅ Video merge completed: {final_output}")
                
                return {
                    "success": True,
                    "output_path": final_output,
                    "metadata_path": metadata_result.get("metadata_path"),
                    "duration": file_info.get("duration", 0),
                    "file_size_mb": file_info.get("file_size_mb", 0),
                    "quality_score": self._calculate_merge_quality_score(file_info),
                    "ffmpeg_command": ' '.join(ffmpeg_cmd)
                }
            else:
                return merge_result
            
        except Exception as e:
            logger.error(f"Error in video merge: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_output_file(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """📁 Prepare output file path"""
        
        try:
            # Create output directory in temp first
            output_path = self.file_manager.get_temp_path("merged")
            output_path.mkdir(exist_ok=True)
            
            # Determine output filename
            if context.get('output_preference'):
                output_filename = context['output_preference']
                if not output_filename.endswith(self.output_format):
                    output_filename += self.output_format
            elif context.get('video_id') and context.get('target_language'):
                target_lang = context['target_language'].lower()
                output_filename = f"{context['video_id']}_translated_{target_lang}{self.output_format}"
            else:
                output_filename = f"merged_video_{datetime.now().strftime('%Y%m%d_%H%M%S')}{self.output_format}"
            
            output_file = output_path / output_filename
            
            # Ensure unique filename
            counter = 1
            while output_file.exists():
                name_part = output_filename.replace(self.output_format, '')
                output_filename = f"{name_part}_{counter}{self.output_format}"
                output_file = output_path / output_filename
                counter += 1
            
            return {
                "success": True,
                "output_file": str(output_file)
            }
            
        except Exception as e:
            logger.error(f"Error preparing output file: {e}")
            return {"success": False, "error": str(e)}
    
    def _build_ffmpeg_command(self, video_file: str, audio_file: str, output_file: str) -> list:
        """🔧 Build FFmpeg command for merging"""
        
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_file,  # Input video
            '-i', audio_file,  # Input audio
            '-c:v', self.video_codec,  # Video codec
            '-c:a', self.audio_codec,  # Audio codec
            '-crf', str(self.video_quality_crf),  # Video quality
            '-b:a', self.audio_bitrate,  # Audio bitrate
            '-preset', self.video_preset,  # Encoding preset
            '-map', '0:v:0',  # Map first video stream from first input
            '-map', '1:a:0',  # Map first audio stream from second input
            '-shortest',  # Finish when shortest stream ends
            '-movflags', '+faststart',  # Optimize for web streaming
            output_file
        ]
        
        return cmd
    
    async def _execute_ffmpeg_merge(self, ffmpeg_cmd: list, output_file: str) -> Dict[str, Any]:
        """⚙️ Execute FFmpeg merge command"""
        
        try:
            logger.info(f"🔧 Executing FFmpeg: {' '.join(ffmpeg_cmd)}")
            
            # Execute FFmpeg in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _run_ffmpeg():
                process = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=300  # 5 minute timeout
                )
                return process
            
            process = await loop.run_in_executor(None, _run_ffmpeg)
            
            if process.returncode == 0:
                # Check if output file was created
                output_path = Path(output_file)
                if output_path.exists() and output_path.stat().st_size > 0:
                    logger.info(f"✅ FFmpeg merge successful: {output_file}")
                    return {
                        "success": True,
                        "output_file": output_file,
                        "ffmpeg_stdout": process.stdout,
                        "ffmpeg_stderr": process.stderr
                    }
                else:
                    return {
                        "success": False,
                        "error": "FFmpeg completed but output file is empty or missing"
                    }
            else:
                logger.error(f"❌ FFmpeg failed with return code {process.returncode}")
                logger.error(f"FFmpeg stderr: {process.stderr}")
                return {
                    "success": False,
                    "error": f"FFmpeg failed: {process.stderr}"
                }
                
        except subprocess.TimeoutExpired:
            logger.error("❌ FFmpeg merge timed out")
            return {"success": False, "error": "FFmpeg merge timed out (5 minutes)"}
        except Exception as e:
            logger.error(f"❌ FFmpeg execution error: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_output_file_info(self, output_file: str) -> Dict[str, Any]:
        """📊 Get information about the output file"""
        
        try:
            output_path = Path(output_file)
            
            # Get file size
            file_size_mb = self.file_manager.get_file_size_mb(output_path)
            
            # Get duration using ffprobe (if available)
            duration = self._get_video_duration(output_file)
            
            # Get video resolution using ffprobe (if available)
            resolution = self._get_video_resolution(output_file)
            
            return {
                "file_size_mb": file_size_mb,
                "duration": duration,
                "resolution": resolution,
                "format": self.output_format,
                "created_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting output file info: {e}")
            return {
                "file_size_mb": 0,
                "duration": 0,
                "resolution": "unknown",
                "format": self.output_format,
                "created_at": datetime.now().isoformat()
            }
    
    def _get_video_duration(self, video_file: str) -> float:
        """⏱️ Get video duration using ffprobe"""
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', video_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                duration = float(data.get('format', {}).get('duration', 0))
                return duration
                
        except Exception as e:
            logger.error(f"Error getting video duration: {e}")
        
        return 0.0
    
    def _get_video_resolution(self, video_file: str) -> str:
        """📐 Get video resolution using ffprobe"""
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-select_streams', 'v:0', video_file
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                streams = data.get('streams', [])
                if streams:
                    stream = streams[0]
                    width = stream.get('width', 0)
                    height = stream.get('height', 0)
                    if width and height:
                        return f"{width}x{height}"
                        
        except Exception as e:
            logger.error(f"Error getting video resolution: {e}")
        
        return "unknown"
    
    def _calculate_merge_quality_score(self, file_info: Dict[str, Any]) -> float:
        """📊 Calculate merge quality score"""
        
        try:
            score = 8.0  # Base score
            
            # File size factor
            file_size_mb = file_info.get("file_size_mb", 0)
            if 10 <= file_size_mb <= 500:  # Reasonable size range
                score += 1.0
            elif file_size_mb < 1:  # Too small, likely problem
                score -= 2.0
            elif file_size_mb > 1000:  # Very large
                score -= 0.5
            
            # Duration factor
            duration = file_info.get("duration", 0)
            if duration > 10:  # Has meaningful duration
                score += 1.0
            elif duration < 5:  # Too short
                score -= 1.0
            
            # Resolution factor
            resolution = file_info.get("resolution", "unknown")
            if "1920x1080" in resolution or "1280x720" in resolution:
                score += 0.5  # Good resolution
            elif "unknown" in resolution:
                score -= 0.5  # Unknown resolution
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating merge quality score: {e}")
            return 7.0  # Default score
    
    async def _save_merge_metadata(self, output_file: str, context: Dict[str, Any], merge_info: Dict[str, Any]) -> Dict[str, Any]:
        """💾 Save merge metadata"""
        
        try:
            # Create metadata filename
            output_path = Path(output_file)
            metadata_file = output_path.with_suffix('.json')
            
            # Prepare metadata
            metadata = {
                "video_id": context.get('video_id'),
                "youtube_url": context.get('youtube_url'),
                "target_language": context.get('target_language'),
                "target_language_code": context.get('target_language_code'),
                "original_video_file": merge_info.get("video_file"),
                "tts_audio_file": merge_info.get("audio_file"),
                "output_file": output_file,
                "file_size_mb": merge_info.get("file_size_mb"),
                "duration": merge_info.get("duration"),
                "resolution": merge_info.get("resolution"),
                "format": self.output_format,
                "video_codec": self.video_codec,
                "audio_codec": self.audio_codec,
                "quality_crf": self.video_quality_crf,
                "audio_bitrate": self.audio_bitrate,
                "ffmpeg_command": merge_info.get("ffmpeg_command"),
                "merge_timestamp": datetime.now().isoformat(),
                "workflow_type": context.get('workflow_type'),
                "workflow_stage": "merge_complete"
            }
            
            # Save metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "metadata_path": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Error saving merge metadata: {e}")
            return {"success": False, "error": str(e)}
    
    async def _move_to_output_folder(self, temp_output_file: str, context: Dict[str, Any]) -> str:
        """📁 Move merged video to final output folder"""
        
        try:
            # Get final output folder
            final_output_path = self.file_manager.get_output_path()
            final_output_path.mkdir(exist_ok=True)
            
            # Determine final filename
            temp_file = Path(temp_output_file)
            final_filename = temp_file.name
            
            final_output_file = final_output_path / final_filename
            
            # Ensure unique filename in output folder
            counter = 1
            while final_output_file.exists():
                name_part = final_filename.replace(self.output_format, '')
                final_filename = f"{name_part}_{counter}{self.output_format}"
                final_output_file = final_output_path / final_filename
                counter += 1
            
            # Move file
            shutil.move(temp_output_file, final_output_file)
            
            # Also move metadata if it exists
            temp_metadata = temp_file.with_suffix('.json')
            if temp_metadata.exists():
                final_metadata = final_output_file.with_suffix('.json')
                shutil.move(temp_metadata, final_metadata)
            
            logger.info(f"📁 Moved merged video to: {final_output_file}")
            
            return str(final_output_file)
            
        except Exception as e:
            logger.error(f"Error moving file to output folder: {e}")
            # Return original path if move fails
            return temp_output_file
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """🔄 Reset agent state"""
        logger.info(f"{self.name} reset - clearing video merge context")
        # Could add merge state cleanup here if needed