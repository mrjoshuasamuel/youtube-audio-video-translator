# agents/transcription_agent.py - COMPLETE SMART VERSION
import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import whisper
import openai
from datetime import datetime

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager, Validators

logger = logging.getLogger(__name__)

class TranscriptionAgent(BaseChatAgent):
    """Smart transcription agent with workflow awareness"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Transcribes audio to text with workflow intelligence")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
        # Initialize OpenAI client for Whisper API
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Transcription settings
        self.whisper_model = "whisper-1"  # OpenAI API model
        self.supported_formats = ['.wav', '.mp3', '.m4a', '.flac', '.ogg']
        self.max_file_size_mb = 25  # OpenAI Whisper API limit
        
        # Language detection mappings
        self.language_codes = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja',
            'korean': 'ko', 'chinese': 'zh', 'arabic': 'ar', 'hindi': 'hi',
            'dutch': 'nl', 'swedish': 'sv', 'norwegian': 'no', 'danish': 'da'
        }
        
        # Workflow stage indicators
        self.workflow_stages = {
            'download_complete': ['downloaded', 'download complete', 'files:', 'audio.wav'],
            'transcription_complete': ['transcribed', 'transcript saved', 'transcription complete'],
            'translation_complete': ['translated', 'translation complete'],
            'tts_complete': ['generated', 'tts complete', 'audio generated'],
            'merge_complete': ['merged', 'final video', 'workflow complete']
        }
        
        logger.info(f"✅ TranscriptionAgent '{name}' initialized")
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """🔧 SMART: Process with full conversation context awareness"""
        
        try:
            # Analyze complete conversation context
            context = self._analyze_conversation_context(messages)
            
            logger.info(f"🧠 TranscriptionAgent Context Analysis: {context}")
            
            # Determine if we should process
            should_process, reason = self._should_process_smart(context)
            
            if not should_process:
                logger.info(f"⏭️  TranscriptionAgent Skipping: {reason}")
                return Response(
                    chat_message=TextMessage(
                        content=f"TranscriptionAgent: {reason}",
                        source=self.name
                    )
                )
            
            # Find audio file to transcribe
            audio_file_path = self._find_audio_file(context)
            
            if not audio_file_path:
                return Response(
                    chat_message=TextMessage(
                        content="❌ No audio file found for transcription. Please download audio first.",
                        source=self.name
                    )
                )
            
            # Execute transcription
            result = await self._transcribe_audio(audio_file_path, context)
            
            if result["success"]:
                # Determine if workflow should continue or complete
                workflow_status = self._determine_workflow_status(context)
                
                completion_message = ""
                if workflow_status == "complete":
                    completion_message = " WORKFLOW_COMPLETE"
                
                return Response(
                    chat_message=TextMessage(
                        content=f"✅ Transcribed audio to text. "
                               f"Transcript saved to {result['transcript_path']}. "
                               f"Language: {result['detected_language']}. "
                               f"Duration: {result['duration']:.1f}s. "
                               f"Word count: {result['word_count']}.{completion_message}",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ Transcription failed: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TranscriptionAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"❌ TranscriptionAgent error: {str(e)}",
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
            'audio_downloaded': False,
            'video_downloaded': False,
            'transcription_complete': False,
            'translation_complete': False,
            'latest_message': "",
            'full_conversation': "",
            'user_request': "",
            'workflow_stage': 'initial',
            'audio_file_mentioned': None,
            'transcript_requested': False
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
        import re
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
        context['target_language'] = self._extract_target_language(conversation_lower)
        
        # Check completion status
        context['audio_downloaded'] = self._check_audio_downloaded(conversation_lower)
        context['video_downloaded'] = self._check_video_downloaded(conversation_lower)
        context['transcription_complete'] = self._check_transcription_complete(conversation_lower)
        context['translation_complete'] = self._check_translation_complete(conversation_lower)
        
        # Determine workflow stage
        context['workflow_stage'] = self._determine_workflow_stage(conversation_lower)
        
        # Check for explicit transcript request
        context['transcript_requested'] = any(keyword in conversation_lower for keyword in [
            'transcribe', 'transcript', 'text', 'speech to text', 'convert to text'
        ])
        
        # Look for mentioned audio files
        audio_file_match = re.search(r'([^\s]+_audio\.wav)', context['full_conversation'])
        if audio_file_match:
            context['audio_file_mentioned'] = audio_file_match.group(1)
        
        return context
    
    def _detect_workflow_type(self, conversation_lower: str) -> str:
        """🔍 Detect the type of workflow from conversation"""
        
        if any(word in conversation_lower for word in ['translate', 'translation', 'german', 'french', 'spanish']):
            return 'translation'
        elif any(word in conversation_lower for word in ['transcribe', 'transcript', 'speech to text']):
            return 'transcription'
        elif any(word in conversation_lower for word in ['summarize', 'summary', 'brief']):
            return 'summary'
        elif any(word in conversation_lower for word in ['voice', 'speech', 'tts']):
            return 'tts'
        
        return 'simple_transcription'
    
    def _extract_target_language(self, conversation_lower: str) -> Optional[str]:
        """🌍 Extract target language from conversation"""
        
        for language, code in self.language_codes.items():
            if language in conversation_lower:
                return language
        
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
            'video file', '_video.mp4', 'extracted video'
        ]
        
        return any(indicator in conversation_lower for indicator in video_indicators)
    
    def _check_transcription_complete(self, conversation_lower: str) -> bool:
        """📝 Check if transcription is already complete"""
        
        transcription_indicators = [
            'transcribed', 'transcript saved', 'transcription complete',
            'transcript.txt', 'text transcribed'
        ]
        
        return any(indicator in conversation_lower for indicator in transcription_indicators)
    
    def _check_translation_complete(self, conversation_lower: str) -> bool:
        """🌍 Check if translation is already complete"""
        
        translation_indicators = [
            'translated', 'translation complete', 'translation saved',
            'translated text', 'translation finished'
        ]
        
        return any(indicator in conversation_lower for indicator in translation_indicators)
    
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
        """🤖 SMART: Intelligent decision making for transcription processing"""
        
        # Check if transcription is already complete
        if context['transcription_complete']:
            return False, "Transcription already completed"
        
        # Check if audio is available
        if not context['audio_downloaded']:
            return False, "No audio file available for transcription"
        
        # Workflow-based decisions
        workflow_type = context['workflow_type']
        workflow_stage = context['workflow_stage']
        
        if workflow_type == 'transcription':
            # Simple transcription workflow
            if workflow_stage == 'download_complete':
                return True, "Transcription workflow: audio ready for transcription"
        elif workflow_type == 'translation':
            # Translation workflow needs transcription first
            if workflow_stage == 'download_complete':
                return True, "Translation workflow: need transcript for translation"
        elif workflow_type == 'summary':
            # Summary workflow needs transcription
            if workflow_stage == 'download_complete':
                return True, "Summary workflow: need transcript for summarization"
        
        # Check for explicit transcript request
        if context['transcript_requested']:
            return True, "Explicit transcription request detected"
        
        # Check if latest message mentions downloaded audio
        latest_lower = context['latest_message'].lower()
        if any(phrase in latest_lower for phrase in [
            'downloaded audio', 'audio downloaded', 'audio.wav', 'files:'
        ]):
            return True, "Audio download just completed"
        
        return False, "No transcription triggers found"
    
    def _find_audio_file(self, context: Dict[str, Any]) -> Optional[str]:
        """🔍 Find the audio file to transcribe"""
        
        # Try to find audio file based on video ID
        if context['video_id']:
            audio_path = self.file_manager.get_temp_path("audio")
            audio_file = audio_path / f"{context['video_id']}_audio.wav"
            
            if audio_file.exists():
                return str(audio_file)
        
        # Try to find any audio file mentioned in conversation
        if context['audio_file_mentioned']:
            audio_path = self.file_manager.get_temp_path("audio")
            audio_file = audio_path / context['audio_file_mentioned']
            
            if audio_file.exists():
                return str(audio_file)
        
        # Search for any .wav files in audio directory
        audio_path = self.file_manager.get_temp_path("audio")
        if audio_path.exists():
            for audio_file in audio_path.glob("*.wav"):
                return str(audio_file)
        
        return None
    
    async def _transcribe_audio(self, audio_file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """🎙️ Transcribe audio file using OpenAI Whisper"""
        
        try:
            logger.info(f"🎙️ Transcribing audio file: {audio_file_path}")
            
            # Validate audio file
            audio_path = Path(audio_file_path)
            if not audio_path.exists():
                return {"success": False, "error": "Audio file not found"}
            
            if audio_path.suffix.lower() not in self.supported_formats:
                return {"success": False, "error": f"Unsupported audio format: {audio_path.suffix}"}
            
            # Check file size
            file_size_mb = self.file_manager.get_file_size_mb(audio_path)
            if file_size_mb > self.max_file_size_mb:
                return {
                    "success": False, 
                    "error": f"Audio file too large: {file_size_mb:.2f}MB > {self.max_file_size_mb}MB"
                }
            
            # Prepare transcription parameters
            transcription_params = {
                "model": self.whisper_model,
                "response_format": "verbose_json",
                "timestamp_granularities": ["word"]
            }
            
            # Add language hint if available
            if context.get('target_language'):
                lang_code = self.language_codes.get(context['target_language'])
                if lang_code:
                    transcription_params["language"] = lang_code
            
            # Execute transcription
            result = await self._call_whisper_api(audio_file_path, transcription_params)
            
            if result["success"]:
                # Save transcript
                transcript_result = await self._save_transcript(result["transcript_data"], context)
                
                if transcript_result["success"]:
                    return {
                        "success": True,
                        "transcript_path": transcript_result["transcript_path"],
                        "detected_language": result["transcript_data"].get("language", "unknown"),
                        "duration": result["transcript_data"].get("duration", 0),
                        "word_count": len(result["transcript_data"].get("text", "").split()),
                        "confidence": self._calculate_confidence(result["transcript_data"])
                    }
                else:
                    return transcript_result
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"success": False, "error": str(e)}
    
    async def _call_whisper_api(self, audio_file_path: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """🌐 Call OpenAI Whisper API"""
        
        try:
            # Execute API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _transcribe():
                with open(audio_file_path, "rb") as audio_file:
                    transcript = self.openai_client.audio.transcriptions.create(
                        file=audio_file,
                        **params
                    )
                return transcript
            
            transcript = await loop.run_in_executor(None, _transcribe)
            
            # Parse response
            if hasattr(transcript, 'model_dump'):
                transcript_data = transcript.model_dump()
            else:
                transcript_data = dict(transcript)
            
            return {
                "success": True,
                "transcript_data": transcript_data
            }
            
        except Exception as e:
            logger.error(f"Whisper API error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _save_transcript(self, transcript_data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """💾 Save transcript to file"""
        
        try:
            # Create transcripts directory
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            transcripts_path.mkdir(exist_ok=True)
            
            # Determine transcript filename
            if context.get('video_id'):
                transcript_filename = f"{context['video_id']}_transcript.txt"
            else:
                transcript_filename = "transcript.txt"
            
            transcript_file = transcripts_path / transcript_filename
            
            # Prepare transcript content
            transcript_content = self._format_transcript(transcript_data, context)
            
            # Save transcript
            with open(transcript_file, 'w', encoding='utf-8') as f:
                f.write(transcript_content)
            
            logger.info(f"✅ Transcript saved to: {transcript_file}")
            
            # Also save metadata
            metadata_file = transcripts_path / f"{transcript_filename}.json"
            metadata = {
                "video_id": context.get('video_id'),
                "youtube_url": context.get('youtube_url'),
                "detected_language": transcript_data.get('language', 'unknown'),
                "duration": transcript_data.get('duration', 0),
                "word_count": len(transcript_data.get('text', '').split()),
                "transcription_timestamp": datetime.now().isoformat(),
                "model_used": self.whisper_model
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "transcript_path": str(transcript_file),
                "metadata_path": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Error saving transcript: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_transcript(self, transcript_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """📝 Format transcript for output"""
        
        lines = []
        
        # Add header
        lines.append("=" * 80)
        lines.append("TRANSCRIPT")
        lines.append("=" * 80)
        
        if context.get('youtube_url'):
            lines.append(f"Source: {context['youtube_url']}")
        
        lines.append(f"Language: {transcript_data.get('language', 'unknown')}")
        lines.append(f"Duration: {transcript_data.get('duration', 0):.1f} seconds")
        lines.append(f"Model: {self.whisper_model}")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Add main transcript text
        lines.append("TRANSCRIPT TEXT:")
        lines.append("-" * 40)
        
        transcript_text = transcript_data.get('text', '')
        if transcript_text:
            # Add paragraphs for better readability
            sentences = transcript_text.split('. ')
            paragraph = ""
            for sentence in sentences:
                if len(paragraph) + len(sentence) > 80:
                    if paragraph:
                        lines.append(paragraph.strip())
                        paragraph = ""
                paragraph += sentence + ". "
            if paragraph:
                lines.append(paragraph.strip())
        else:
            lines.append("No transcript text available.")
        
        # Add word-level timestamps if available
        if transcript_data.get('words'):
            lines.append("")
            lines.append("WORD-LEVEL TIMESTAMPS:")
            lines.append("-" * 40)
            
            for word_info in transcript_data['words'][:10]:  # Show first 10 words
                word = word_info.get('word', '')
                start = word_info.get('start', 0)
                end = word_info.get('end', 0)
                lines.append(f"{start:.2f}s - {end:.2f}s: {word}")
            
            if len(transcript_data['words']) > 10:
                lines.append(f"... and {len(transcript_data['words']) - 10} more words")
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _calculate_confidence(self, transcript_data: Dict[str, Any]) -> float:
        """📊 Calculate average confidence score"""
        
        try:
            if 'words' in transcript_data:
                confidences = []
                for word_info in transcript_data['words']:
                    if 'confidence' in word_info:
                        confidences.append(word_info['confidence'])
                
                if confidences:
                    return sum(confidences) / len(confidences)
            
            return 0.0  # No confidence data available
            
        except Exception as e:
            logger.error(f"Error calculating confidence: {e}")
            return 0.0
    
    def _determine_workflow_status(self, context: Dict[str, Any]) -> str:
        """🎯 Determine if workflow should complete after transcription"""
        
        workflow_type = context['workflow_type']
        
        if workflow_type == 'transcription':
            # Simple transcription workflow completes here
            return "complete"
        elif workflow_type in ['translation', 'summary', 'tts']:
            # These workflows continue after transcription
            return "continue"
        
        return "continue"
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """🔄 Reset agent state"""
        logger.info(f"{self.name} reset - clearing transcription context")
        # Could add transcription state cleanup here if needed


















# # agents/transcription_agent.py
# import os
# import asyncio
# import whisper
# import json
# from typing import Dict, Any, Optional, Sequence
# from pathlib import Path
# import logging

# from autogen_agentchat.agents import BaseChatAgent
# from autogen_agentchat.base import Response
# from autogen_agentchat.messages import BaseChatMessage, TextMessage
# from autogen_core import CancellationToken
# from autogen_ext.models.openai import OpenAIChatCompletionClient

# from utils import Config, FileManager

# logger = logging.getLogger(__name__)

# class TranscriptionAgent(BaseChatAgent):
#     """Agent responsible for transcribing audio and generating summaries"""
    
#     def __init__(self, name: str, config: Config, file_manager: FileManager):
#         super().__init__(name, description="Transcribes audio and generates summaries")
#         self.config = config
#         self.file_manager = file_manager
#         self.whisper_model = None
#         self.openai_client = OpenAIChatCompletionClient(
#             model=config.openai_model,
#             api_key=config.openai_api_key
#         )
        
#     @property
#     def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
#         return (TextMessage,)
    
#     async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
#         """Process messages for transcription or summarization requests"""
        
#         latest_message = messages[-1] if messages else None
#         if not latest_message:
#             return Response(
#                 chat_message=TextMessage(
#                     content="No message provided.",
#                     source=self.name
#                 )
#             )
        
#         content = latest_message.content if hasattr(latest_message, 'content') else str(latest_message)
        
#         # Check if this agent should process the message
#         if not self._should_process(content):
#             return Response(
#                 chat_message=TextMessage(
#                     content="This agent handles transcription and summarization requests.",
#                     source=self.name
#                 )
#             )
        
#         try:
#             # Determine the task type
#             task_type = self._determine_task_type(content)
            
#             if task_type == "transcribe":
#                 result = await self._transcribe_audio(content)
#             elif task_type == "summarize":
#                 result = await self._summarize_content(content)
#             else:
#                 return Response(
#                     chat_message=TextMessage(
#                         content="Please specify whether you want transcription or summarization.",
#                         source=self.name
#                     )
#                 )
            
#             if result["success"]:
#                 return Response(
#                     chat_message=TextMessage(
#                         content=result["message"],
#                         source=self.name
#                     )
#                 )
#             else:
#                 return Response(
#                     chat_message=TextMessage(
#                         content=f"Error: {result['error']}",
#                         source=self.name
#                     )
#                 )
                
#         except Exception as e:
#             logger.error(f"Error in TranscriptionAgent: {e}")
#             return Response(
#                 chat_message=TextMessage(
#                     content=f"An error occurred: {str(e)}",
#                     source=self.name
#                 )
#             )
    
#     def _should_process(self, content: str) -> bool:
#         """Check if this agent should process the message"""
#         keywords = ["transcribe", "transcript", "summarize", "summary", "convert to text"]
#         content_lower = content.lower()
#         return any(keyword in content_lower for keyword in keywords)
    
#     def _determine_task_type(self, content: str) -> str:
#         """Determine whether user wants transcription or summarization"""
#         content_lower = content.lower()
        
#         if any(word in content_lower for word in ["summarize", "summary", "summarise"]):
#             return "summarize"
#         elif any(word in content_lower for word in ["transcribe", "transcript", "text"]):
#             return "transcribe"
        
#         return "unknown"
    
#     async def _transcribe_audio(self, content: str) -> Dict[str, Any]:
#         """Transcribe audio file to text"""
#         try:
#             # Find audio file from previous agent or specified path
#             audio_file_path = self._extract_audio_file_path(content)
            
#             if not audio_file_path or not Path(audio_file_path).exists():
#                 return {"success": False, "error": "Audio file not found"}
            
#             # Load Whisper model if not already loaded
#             if self.whisper_model is None:
#                 loop = asyncio.get_event_loop()
#                 self.whisper_model = await loop.run_in_executor(
#                     None, whisper.load_model, self.config.whisper_model
#                 )
            
#             # Transcribe audio
#             loop = asyncio.get_event_loop()
#             result = await loop.run_in_executor(
#                 None, self._transcribe_with_whisper, audio_file_path
#             )
            
#             if result["success"]:
#                 # Save transcript to file
#                 transcript_file = self.file_manager.get_temp_path(
#                     f"transcript_{Path(audio_file_path).stem}.txt",
#                     "transcripts"
#                 )
                
#                 async with self.file_manager.save_file(
#                     result["transcript"].encode('utf-8'),
#                     transcript_file
#                 ):
#                     pass
                
#                 # Also save detailed result as JSON
#                 detailed_file = self.file_manager.get_temp_path(
#                     f"transcript_detailed_{Path(audio_file_path).stem}.json",
#                     "transcripts"
#                 )
                
#                 detailed_data = {
#                     "transcript": result["transcript"],
#                     "language": result["language"],
#                     "segments": result.get("segments", []),
#                     "audio_file": audio_file_path
#                 }
                
#                 async with self.file_manager.save_file(
#                     json.dumps(detailed_data, indent=2).encode('utf-8'),
#                     detailed_file
#                 ):
#                     pass
                
#                 return {
#                     "success": True,
#                     "message": f"Successfully transcribed audio. "
#                              f"Language: {result['language']}. "
#                              f"Transcript saved to: {transcript_file}. "
#                              f"Text: {result['transcript'][:200]}..."
#                 }
#             else:
#                 return result
            
#         except Exception as e:
#             logger.error(f"Error in transcription: {e}")
#             return {"success": False, "error": str(e)}
    
#     def _transcribe_with_whisper(self, audio_file_path: str) -> Dict[str, Any]:
#         """Transcribe using Whisper model (runs in thread pool)"""
#         try:
#             result = self.whisper_model.transcribe(audio_file_path)
#             return {
#                 "success": True,
#                 "transcript": result["text"],
#                 "language": result["language"],
#                 "segments": result.get("segments", [])
#             }
#         except Exception as e:
#             return {"success": False, "error": str(e)}
    
#     async def _summarize_content(self, content: str) -> Dict[str, Any]:
#         """Summarize transcript content"""
#         try:
#             # Extract transcript text or file
#             transcript_text = self._extract_transcript_content(content)
            
#             if not transcript_text:
#                 return {"success": False, "error": "No transcript content found to summarize"}
            
#             # Create summarization prompt
#             prompt = self._create_summary_prompt(transcript_text)
            
#             # Use OpenAI to generate summary
#             from autogen_core.models import UserMessage
            
#             result = await self.openai_client.create([
#                 UserMessage(content=prompt, source="user")
#             ])
            
#             summary = result.content
            
#             # Save summary to file
#             summary_file = self.file_manager.get_temp_path(
#                 "summary.txt",
#                 "transcripts"
#             )
            
#             await self.file_manager.save_file(
#                 summary.encode('utf-8'),
#                 summary_file
#             )
            
#             return {
#                 "success": True,
#                 "message": f"Successfully generated summary. "
#                          f"Summary saved to: {summary_file}. "
#                          f"Summary: {summary[:300]}..."
#             }
            
#         except Exception as e:
#             logger.error(f"Error in summarization: {e}")
#             return {"success": False, "error": str(e)}
    
#     def _extract_audio_file_path(self, content: str) -> Optional[str]:
#         """Extract audio file path from message content"""
#         import re
        
#         # Look for file paths in the content
#         path_patterns = [
#             r'File: ([^\s]+\.wav)',
#             r'file_path["\']?\s*[:=]\s*["\']?([^"\']+\.wav)',
#             r'([^\s]+\.wav)'
#         ]
        
#         for pattern in path_patterns:
#             match = re.search(pattern, content)
#             if match:
#                 return match.group(1)
        
#         # Check for recent audio files in temp folder
#         audio_folder = self.file_manager.get_temp_path("", "audio")
#         if audio_folder.exists():
#             audio_files = list(audio_folder.glob("*.wav"))
#             if audio_files:
#                 # Return the most recent file
#                 return str(max(audio_files, key=lambda p: p.stat().st_mtime))
        
#         return None
    
#     def _extract_transcript_content(self, content: str) -> Optional[str]:
#         """Extract transcript content for summarization"""
#         # First, try to find transcript file
#         transcript_file = self._find_transcript_file()
#         if transcript_file:
#             try:
#                 with open(transcript_file, 'r', encoding='utf-8') as f:
#                     return f.read()
#             except Exception as e:
#                 logger.error(f"Error reading transcript file: {e}")
        
#         # If no file found, check if transcript is in the message
#         if len(content) > 100:  # Assume long content might be transcript
#             return content
        
#         return None
    
#     def _find_transcript_file(self) -> Optional[Path]:
#         """Find the most recent transcript file"""
#         transcript_folder = self.file_manager.get_temp_path("", "transcripts")
#         if transcript_folder.exists():
#             transcript_files = list(transcript_folder.glob("transcript_*.txt"))
#             if transcript_files:
#                 return max(transcript_files, key=lambda p: p.stat().st_mtime)
#         return None
    
#     def _create_summary_prompt(self, transcript: str) -> str:
#         """Create prompt for summarization"""
#         return f"""
# Please provide a comprehensive summary of the following transcript. 
# Include the main topics, key points, and important details.
# Structure the summary with clear sections if applicable.

# Transcript:
# {transcript}

# Summary:
# """
    
#     async def on_reset(self, cancellation_token: CancellationToken) -> None:
#         """Reset agent state"""
#         logger.info(f"{self.name} reset")
#         await self.openai_client.close()