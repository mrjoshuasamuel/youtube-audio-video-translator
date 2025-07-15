# agents/tts_agent.py - COMPLETE SMART VERSION
import os
import asyncio
import json
import logging
import requests
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

class MurfTTSClient:
    """Murf TTS client wrapper"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        try:
            from murf import Murf
            self.client = Murf(api_key=api_key)
        except ImportError:
            logger.error("Murf package not installed. Run: pip install murf")
            self.client = None
    
    def generate_audio(self, text: str, voice_id: str, locale: str) -> dict:
        """Generate TTS audio using Murf API"""
        if not self.client:
            raise RuntimeError("Murf client not initialized")
        
        logger.info(f"Generating TTS audio with voice_id={voice_id}, locale={locale}")
        
        try:
            res = self.client.text_to_speech.generate(
                text=text,
                voice_id=voice_id,
                multi_native_locale=locale
            )
            
            audio_url = res.audio_file
            logger.info(f"TTS audio generated successfully: {audio_url}")
            
            return {
                "success": True,
                "audio_url": audio_url,
                "voice_id": voice_id,
                "locale": locale
            }
            
        except Exception as e:
            logger.error(f"Murf TTS generation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }

class TTSAgent(BaseChatAgent):
    """Smart TTS agent with workflow awareness and Murf integration"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Converts text to speech using Murf TTS with workflow intelligence")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
        # Initialize Murf TTS client
        self.murf_client = MurfTTSClient(config.murf_api_key)
        
        # TTS settings
        self.max_text_length = 5000  # Murf API limit
        self.supported_formats = ['.wav', '.mp3']
        
        # Language mapping with voice_id and multi_native_locale
        self.language_mapping = {
            # German
            "German": ("en-UK-ruby", "de-DE"),
            "Deutsch": ("en-UK-ruby", "de-DE"),
            "german": ("en-UK-ruby", "de-DE"),
            "de": ("en-UK-ruby", "de-DE"),
            
            # Greek
            "Greek": ("en-UK-ruby", "el-GR"),
            "Ελληνικά": ("en-UK-ruby", "el-GR"),
            "greek": ("en-UK-ruby", "el-GR"),
            "el": ("en-UK-ruby", "el-GR"),
            
            # English variants
            "English": ("en-UK-ruby", "en-US"),  # Default to US English
            "English (UK)": ("en-UK-ruby", "en-UK"),
            "English (US)": ("en-UK-ruby", "en-US"),
            "British English": ("en-UK-ruby", "en-UK"),
            "American English": ("en-UK-ruby", "en-US"),
            "english": ("en-UK-ruby", "en-US"),
            "en": ("en-UK-ruby", "en-US"),
            
            # Spanish variants
            "Spanish": ("en-UK-ruby", "es-ES"),  # Default to Spain Spanish
            "Spanish (Spain)": ("en-UK-ruby", "es-ES"),
            "Spanish (Mexico)": ("en-UK-ruby", "es-MX"),
            "Español": ("en-UK-ruby", "es-ES"),
            "Castilian": ("en-UK-ruby", "es-ES"),
            "spanish": ("en-UK-ruby", "es-ES"),
            "es": ("en-UK-ruby", "es-ES"),
            
            # French
            "French": ("en-UK-ruby", "fr-FR"),
            "Français": ("en-UK-ruby", "fr-FR"),
            "french": ("en-UK-ruby", "fr-FR"),
            "fr": ("en-UK-ruby", "fr-FR"),
            
            # Hindi
            "Hindi": ("en-UK-ruby", "hi-IN"),
            "हिन्दी": ("en-UK-ruby", "hi-IN"),
            "hindi": ("en-UK-ruby", "hi-IN"),
            "hi": ("en-UK-ruby", "hi-IN"),
            
            # Croatian
            "Croatian": ("en-UK-ruby", "hr-HR"),
            "Hrvatski": ("en-UK-ruby", "hr-HR"),
            "croatian": ("en-UK-ruby", "hr-HR"),
            "hr": ("en-UK-ruby", "hr-HR"),
            
            # Indonesian
            "Indonesian": ("en-UK-ruby", "id-ID"),
            "Bahasa Indonesia": ("en-UK-ruby", "id-ID"),
            "indonesian": ("en-UK-ruby", "id-ID"),
            "id": ("en-UK-ruby", "id-ID"),
            
            # Turkish
            "Turkish": ("en-UK-ruby", "tr-TR"),
            "Türkçe": ("en-UK-ruby", "tr-TR"),
            "turkish": ("en-UK-ruby", "tr-TR"),
            "tr": ("en-UK-ruby", "tr-TR"),
            
            # Italian
            "Italian": ("it-IT-lorenzo", "it-IT"),
            "Italiano": ("it-IT-lorenzo", "it-IT"),
            "italian": ("it-IT-lorenzo", "it-IT"),
            "it": ("it-IT-lorenzo", "it-IT"),
            
            # Chinese
            "Chinese": ("zh-CN-tao", "zh-CN"),
            "Chinese (Simplified)": ("zh-CN-tao", "zh-CN"),
            "Mandarin": ("zh-CN-tao", "zh-CN"),
            "中文": ("zh-CN-tao", "zh-CN"),
            "chinese": ("zh-CN-tao", "zh-CN"),
            "zh": ("zh-CN-tao", "zh-CN"),
            
            # Korean
            "Korean": ("ko-KR-gyeong", "ko-KR"),
            "한국어": ("ko-KR-gyeong", "ko-KR"),
            "korean": ("ko-KR-gyeong", "ko-KR"),
            "ko": ("ko-KR-gyeong", "ko-KR"),
            
            # Japanese
            "Japanese": ("ja-JP-kenji", "ja-JP"),
            "日本語": ("ja-JP-kenji", "ja-JP"),
            "japanese": ("ja-JP-kenji", "ja-JP"),
            "ja": ("ja-JP-kenji", "ja-JP"),
            
            # Slovak
            "Slovak": ("sk-SK-nina", "sk-SK"),
            "Slovenčina": ("sk-SK-nina", "sk-SK"),
            "slovak": ("sk-SK-nina", "sk-SK"),
            "sk": ("sk-SK-nina", "sk-SK"),
            
            # Portuguese
            "Portuguese": ("en-UK-ruby", "pt-PT"),
            "Português": ("en-UK-ruby", "pt-PT"),
            "portuguese": ("en-UK-ruby", "pt-PT"),
            "pt": ("en-UK-ruby", "pt-PT"),
            
            # Russian
            "Russian": ("en-UK-ruby", "ru-RU"),
            "Русский": ("en-UK-ruby", "ru-RU"),
            "russian": ("en-UK-ruby", "ru-RU"),
            "ru": ("en-UK-ruby", "ru-RU"),
            
            # Dutch
            "Dutch": ("en-UK-ruby", "nl-NL"),
            "Nederlands": ("en-UK-ruby", "nl-NL"),
            "dutch": ("en-UK-ruby", "nl-NL"),
            "nl": ("en-UK-ruby", "nl-NL"),
            
            # Arabic
            "Arabic": ("en-UK-ruby", "ar-SA"),
            "العربية": ("en-UK-ruby", "ar-SA"),
            "arabic": ("en-UK-ruby", "ar-SA"),
            "ar": ("en-UK-ruby", "ar-SA"),
        }
        
        # Workflow stage indicators
        self.workflow_stages = {
            'download_complete': ['downloaded', 'download complete', 'files:', 'audio.wav'],
            'transcription_complete': ['transcribed', 'transcript saved', 'transcription complete'],
            'translation_complete': ['translated', 'translation complete', 'translation saved'],
            'tts_complete': ['generated', 'tts complete', 'audio generated', 'tts audio'],
            'merge_complete': ['merged', 'final video', 'workflow complete']
        }
        
        logger.info(f"✅ TTSAgent '{name}' initialized with {len(self.language_mapping)} language mappings")
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """🔧 SMART: Process with full conversation context awareness"""
        
        try:
            # Analyze complete conversation context
            context = self._analyze_conversation_context(messages)
            
            logger.info(f"🧠 TTSAgent Context Analysis: {context}")
            
            # Determine if we should process
            should_process, reason = self._should_process_smart(context)
            
            if not should_process:
                logger.info(f"⏭️  TTSAgent Skipping: {reason}")
                return Response(
                    chat_message=TextMessage(
                        content=f"TTSAgent: {reason}",
                        source=self.name
                    )
                )
            
            # Find translation file to convert to speech
            translation_file_path = self._find_translation_file(context)
            
            if not translation_file_path:
                return Response(
                    chat_message=TextMessage(
                        content="❌ No translation file found for TTS conversion. Please translate text first.",
                        source=self.name
                    )
                )
            
            # Execute TTS generation
            result = await self._generate_tts_audio(translation_file_path, context)
            
            if result["success"]:
                # Determine if workflow should continue or complete
                workflow_status = self._determine_workflow_status(context)
                
                completion_message = ""
                if workflow_status == "complete":
                    completion_message = " WORKFLOW_COMPLETE"
                elif workflow_status == "continue_merge":
                    completion_message = " Ready for video merging."
                
                return Response(
                    chat_message=TextMessage(
                        content=f"✅ Generated TTS audio in {result['target_language']}. "
                               f"Audio saved to {result['audio_path']}. "
                               f"Voice: {result['voice_id']}. "
                               f"Duration: {result['duration']:.1f}s. "
                               f"Quality: {result['quality_score']:.2f}/10.{completion_message}",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ TTS generation failed: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TTSAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"❌ TTSAgent error: {str(e)}",
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
            'latest_message': "",
            'full_conversation': "",
            'user_request': "",
            'workflow_stage': 'initial',
            'translation_file_mentioned': None,
            'tts_requested': False,
            'voice_preference': None
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
        
        # Determine workflow stage
        context['workflow_stage'] = self._determine_workflow_stage(conversation_lower)
        
        # Check for explicit TTS request
        context['tts_requested'] = any(keyword in conversation_lower for keyword in [
            'tts', 'text to speech', 'voice', 'speech', 'audio generation', 'generate audio'
        ])
        
        # Look for mentioned translation files
        translation_file_match = re.search(r'([^\s]+translation[^\s]*\.txt)', context['full_conversation'])
        if translation_file_match:
            context['translation_file_mentioned'] = translation_file_match.group(1)
        
        # Extract voice preference if mentioned
        context['voice_preference'] = self._extract_voice_preference(conversation_lower)
        
        return context
    
    def _detect_workflow_type(self, conversation_lower: str) -> str:
        """🔍 Detect the type of workflow from conversation"""
        
        if any(word in conversation_lower for word in ['translate', 'translation', 'german', 'french', 'spanish']):
            return 'translation'
        elif any(word in conversation_lower for word in ['tts', 'text to speech', 'voice', 'speech']):
            return 'tts'
        elif any(word in conversation_lower for word in ['transcribe', 'transcript', 'speech to text']):
            return 'transcription'
        elif any(word in conversation_lower for word in ['summarize', 'summary', 'brief']):
            return 'summary'
        
        return 'simple_tts'
    
    def _extract_target_language(self, conversation_lower: str) -> tuple[Optional[str], Optional[str]]:
        """🌍 Extract target language from conversation"""
        
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
                if potential_lang in self.language_mapping:
                    return potential_lang, self.language_mapping[potential_lang][1]
        
        # Look for language detection from translation
        language_patterns = [
            r'translated.*to\s+(\w+)',
            r'translation.*to\s+(\w+)',
            r'target language:\s*(\w+)',
            r'language:\s*(\w+)'
        ]
        
        for pattern in language_patterns:
            match = re.search(pattern, conversation_lower)
            if match:
                potential_lang = match.group(1).lower()
                if potential_lang in self.language_mapping:
                    return potential_lang, self.language_mapping[potential_lang][1]
        
        # Direct language detection
        for language in self.language_mapping.keys():
            if language.lower() in conversation_lower:
                return language.lower(), self.language_mapping[language][1]
        
        return None, None
    
    def _extract_voice_preference(self, conversation_lower: str) -> Optional[str]:
        """🎙️ Extract voice preference from conversation"""
        
        voice_patterns = [
            r'voice:\s*([^\s]+)',
            r'use voice\s+([^\s]+)',
            r'with voice\s+([^\s]+)',
            r'voice_id:\s*([^\s]+)'
        ]
        
        for pattern in voice_patterns:
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
            'video file', '_video.mp4', 'extracted video'
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
        
        # Only check for SUCCESS indicators, not just keywords
        success_indicators = [
            '✅ translated',
            'translation saved to',
            'translated transcript to', 
            'translation complete.',
            'translation finished.'
        ]
        
        # Must find success indicator AND exclude error messages
        has_success = any(indicator in conversation_lower for indicator in success_indicators)
        has_error = any(error in conversation_lower for error in [
            'translation not complete',
            'cannot generate tts without translated',
            'translation failed'
        ])
        
        return has_success and not has_error
    
    def _check_tts_complete(self, conversation_lower: str) -> bool:
        """🎙️ Check if TTS is already complete"""
        
        tts_indicators = [
            'generated tts', 'tts complete', 'audio generated',
            'tts audio', 'speech generated', 'voice generated'
        ]
        
        return any(indicator in conversation_lower for indicator in tts_indicators)
    
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
        """🤖 SMART: Intelligent decision making for TTS processing"""
        
        # Check if TTS is already complete
        if context['tts_complete']:
            return False, "TTS generation already completed"
        
        # Check if translation is complete (prerequisite for TTS)
        if not context['translation_complete']:
            return False, "Translation not complete - cannot generate TTS without translated text"
        
        # Check if target language is specified
        if not context['target_language']:
            return False, "No target language specified for TTS generation"
        
        # Check if target language is supported
        if context['target_language'] not in self.language_mapping:
            return False, f"Target language '{context['target_language']}' not supported for TTS"
        
        # Workflow-based decisions
        workflow_type = context['workflow_type']
        workflow_stage = context['workflow_stage']
        
        if workflow_type == 'translation':
            # Translation workflow with TTS
            if workflow_stage == 'translation_complete':
                return True, "Translation workflow: translated text ready for TTS"
        elif workflow_type == 'tts':
            # Direct TTS workflow
            if workflow_stage == 'translation_complete':
                return True, "TTS workflow: translated text ready for speech generation"
        
        # Check for explicit TTS request
        if context['tts_requested']:
            return True, "Explicit TTS request detected"
        
        # Check if latest message mentions translation completion
        latest_lower = context['latest_message'].lower()
        if any(phrase in latest_lower for phrase in [
            'translated', 'translation complete', 'translation saved'
        ]):
            return True, "Translation just completed - ready for TTS"
        
        return False, "No TTS triggers found"
    
    def _find_translation_file(self, context: Dict[str, Any]) -> Optional[str]:
        """🔍 Find the translation file to convert to speech"""
        
        # Try to find translation file based on video ID and target language
        if context['video_id'] and context['target_language_code']:
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            
            # Look for language-specific translation files
            lang_code = context['target_language_code'].split('-')[0]  # Extract language code
            translation_patterns = [
                f"{context['video_id']}_transcript_{lang_code}.txt",
                f"{context['video_id']}_transcript_{context['target_language'].lower()}.txt",
                f"transcript_{lang_code}.txt",
                f"transcript_{context['target_language'].lower()}.txt"
            ]
            
            for pattern in translation_patterns:
                translation_file = transcripts_path / pattern
                if translation_file.exists():
                    return str(translation_file)
        
        # Try to find translation file mentioned in conversation
        if context['translation_file_mentioned']:
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            translation_file = transcripts_path / context['translation_file_mentioned']
            
            if translation_file.exists():
                return str(translation_file)
        
        # Search for any translation files in transcripts directory
        transcripts_path = self.file_manager.get_temp_path("transcripts")
        if transcripts_path.exists():
            # Look for files with language indicators
            for translation_file in transcripts_path.glob("*translation*.txt"):
                return str(translation_file)
            
            # Look for files with language codes
            if context['target_language_code']:
                lang_code = context['target_language_code'].split('-')[0]
                for translation_file in transcripts_path.glob(f"*{lang_code}*.txt"):
                    return str(translation_file)
        
        return None
    
    async def _generate_tts_audio(self, translation_file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """🎙️ Generate TTS audio from translation file using Murf"""
        
        try:
            logger.info(f"🎙️ Generating TTS audio from: {translation_file_path}")
            
            # Validate translation file
            translation_path = Path(translation_file_path)
            if not translation_path.exists():
                return {"success": False, "error": "Translation file not found"}
            
            # Read translation content
            translation_content = await self._read_translation_file(translation_path)
            
            if not translation_content:
                return {"success": False, "error": "Empty translation file"}
            
            # Validate text length
            if len(translation_content) > self.max_text_length:
                return {
                    "success": False, 
                    "error": f"Translation text too long: {len(translation_content)} > {self.max_text_length} characters"
                }
            
            # Get voice settings for target language
            voice_settings = self._get_voice_settings(context)
            
            if not voice_settings:
                return {"success": False, "error": f"No voice settings found for language: {context['target_language']}"}
            
            # Generate TTS audio
            tts_result = await self._call_murf_tts(translation_content, voice_settings, context)
            
            if tts_result["success"]:
                # Download and save audio
                audio_result = await self._download_and_save_audio(tts_result["audio_url"], context)
                
                if audio_result["success"]:
                    return {
                        "success": True,
                        "audio_path": audio_result["audio_path"],
                        "target_language": context['target_language'].title(),
                        "voice_id": voice_settings["voice_id"],
                        "duration": audio_result.get("duration", 0),
                        "quality_score": tts_result.get("quality_score", 8.0),
                        "file_size_mb": audio_result.get("file_size_mb", 0)
                    }
                else:
                    return audio_result
            else:
                return tts_result
            
        except Exception as e:
            logger.error(f"Error in TTS generation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _read_translation_file(self, translation_path: Path) -> str:
        """📖 Read and extract translation content"""
        
        try:
            with open(translation_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract just the translated text (remove headers and metadata)
            lines = content.split('\n')
            translation_lines = []
            in_translation_section = False
            
            for line in lines:
                if 'TRANSLATED TEXT:' in line:
                    in_translation_section = True
                    continue
                elif line.startswith('ORIGINAL TEXT') or line.startswith('====='):
                    in_translation_section = False
                    continue
                elif in_translation_section and line.strip() and not line.startswith('-'):
                    translation_lines.append(line.strip())
            
            # If no structured translation found, use entire content
            if not translation_lines:
                # Filter out obvious metadata lines
                for line in lines:
                    if (line.strip() and 
                        not line.startswith('=') and 
                        not line.startswith('-') and 
                        not line.startswith('Source:') and 
                        not line.startswith('Original Language:') and 
                        not line.startswith('Target Language:') and 
                        not line.startswith('Quality Score:') and 
                        not line.startswith('Model:') and 
                        not line.startswith('Timestamp:') and 
                        not line.startswith('Tokens Used:') and 
                        'TRANSLATED TRANSCRIPT' not in line):
                        translation_lines.append(line.strip())
            
            return '\n'.join(translation_lines)
            
        except Exception as e:
            logger.error(f"Error reading translation file: {e}")
            return ""
    
    def _get_voice_settings(self, context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """🎚️ Get voice settings for target language"""
        
        target_language = context['target_language']
        voice_preference = context.get('voice_preference')
        
        # Check if target language is supported
        if target_language not in self.language_mapping:
            logger.error(f"Target language '{target_language}' not supported")
            return None
        
        voice_id, locale = self.language_mapping[target_language]
        
        # Override voice if user specified preference
        if voice_preference:
            voice_id = voice_preference
        
        return {
            "voice_id": voice_id,
            "locale": locale,
            "language": target_language
        }
    
    async def _call_murf_tts(self, text: str, voice_settings: Dict[str, str], context: Dict[str, Any]) -> Dict[str, Any]:
        """🌐 Call Murf TTS API"""
        
        try:
            # Execute API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _generate_tts():
                return self.murf_client.generate_audio(
                    text=text,
                    voice_id=voice_settings["voice_id"],
                    locale=voice_settings["locale"]
                )
            
            result = await loop.run_in_executor(None, _generate_tts)
            
            if result["success"]:
                # Calculate quality score (simple heuristic)
                quality_score = self._calculate_tts_quality_score(text, voice_settings)
                
                return {
                    "success": True,
                    "audio_url": result["audio_url"],
                    "voice_id": result["voice_id"],
                    "locale": result["locale"],
                    "quality_score": quality_score,
                    "text_length": len(text),
                    "voice_settings": voice_settings
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Murf TTS API error: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_tts_quality_score(self, text: str, voice_settings: Dict[str, str]) -> float:
        """📊 Calculate TTS quality score (simple heuristic)"""
        
        try:
            score = 8.0  # Base score
            
            # Text length factor
            text_length = len(text)
            if 100 <= text_length <= 2000:  # Optimal range
                score += 1.0
            elif text_length < 50:  # Too short
                score -= 1.0
            elif text_length > 4000:  # Too long
                score -= 0.5
            
            # Voice quality factor (based on voice selection)
            voice_id = voice_settings["voice_id"]
            if any(premium_voice in voice_id for premium_voice in ["lorenzo", "kenji", "gyeong", "tao", "nina"]):
                score += 1.0  # Premium voices
            
            # Language match factor
            locale = voice_settings["locale"]
            if locale and "-" in locale:
                score += 0.5  # Proper locale specified
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating TTS quality score: {e}")
            return 8.0  # Default score
    
    async def _download_and_save_audio(self, audio_url: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """💾 Download audio from URL and save to file"""
        
        try:
            logger.info(f"💾 Downloading TTS audio from: {audio_url}")
            
            # Create audio directory
            audio_path = self.file_manager.get_temp_path("audio")
            audio_path.mkdir(exist_ok=True)
            
            # Determine audio filename
            target_lang = context.get('target_language_code', 'unknown').split('-')[0]
            if context.get('video_id'):
                audio_filename = f"{context['video_id']}_tts_{target_lang}.wav"
            else:
                audio_filename = f"tts_{target_lang}.wav"
            
            audio_file = audio_path / audio_filename
            
            # Download audio file
            download_result = await self._download_file(audio_url, audio_file)
            
            if download_result["success"]:
                # Get file information
                file_size_mb = self.file_manager.get_file_size_mb(audio_file)
                duration = self._estimate_audio_duration(audio_file)
                
                # Save metadata
                metadata_result = await self._save_tts_metadata(audio_file, context, {
                    "audio_url": audio_url,
                    "file_size_mb": file_size_mb,
                    "duration": duration
                })
                
                logger.info(f"✅ TTS audio saved to: {audio_file}")
                
                return {
                    "success": True,
                    "audio_path": str(audio_file),
                    "metadata_path": metadata_result.get("metadata_path"),
                    "file_size_mb": file_size_mb,
                    "duration": duration
                }
            else:
                return download_result
            
        except Exception as e:
            logger.error(f"Error downloading and saving audio: {e}")
            return {"success": False, "error": str(e)}
    
    async def _download_file(self, url: str, file_path: Path) -> Dict[str, Any]:
        """📥 Download file from URL"""
        
        try:
            # Execute download in thread pool
            loop = asyncio.get_event_loop()
            
            def _download():
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                return response.headers
            
            headers = await loop.run_in_executor(None, _download)
            
            return {
                "success": True,
                "file_path": str(file_path),
                "content_type": headers.get("content-type", "audio/wav")
            }
            
        except Exception as e:
            logger.error(f"File download error: {e}")
            return {"success": False, "error": str(e)}
    
    def _estimate_audio_duration(self, audio_file: Path) -> float:
        """⏱️ Estimate audio duration (simple heuristic)"""
        
        try:
            # Simple estimation based on file size
            # Average: 1MB ≈ 60 seconds for compressed audio
            file_size_mb = self.file_manager.get_file_size_mb(audio_file)
            estimated_duration = file_size_mb * 60  # Rough estimate
            
            return max(1.0, estimated_duration)  # Minimum 1 second
            
        except Exception as e:
            logger.error(f"Error estimating audio duration: {e}")
            return 30.0  # Default duration
    
    async def _save_tts_metadata(self, audio_file: Path, context: Dict[str, Any], audio_info: Dict[str, Any]) -> Dict[str, Any]:
        """💾 Save TTS metadata"""
        
        try:
            # Create metadata filename
            metadata_file = audio_file.with_suffix('.json')
            
            # Prepare metadata
            metadata = {
                "video_id": context.get('video_id'),
                "youtube_url": context.get('youtube_url'),
                "target_language": context.get('target_language'),
                "target_language_code": context.get('target_language_code'),
                "voice_id": context.get('voice_preference'),
                "audio_url": audio_info.get("audio_url"),
                "file_size_mb": audio_info.get("file_size_mb"),
                "duration": audio_info.get("duration"),
                "tts_timestamp": datetime.now().isoformat(),
                "murf_api_used": True,
                "quality_score": audio_info.get("quality_score", 8.0),
                "audio_file": str(audio_file)
            }
            
            # Save metadata
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "metadata_path": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Error saving TTS metadata: {e}")
            return {"success": False, "error": str(e)}
    
    def _determine_workflow_status(self, context: Dict[str, Any]) -> str:
        """🎯 Determine if workflow should complete or continue after TTS"""
        
        workflow_type = context['workflow_type']
        
        if workflow_type == 'translation':
            # Check if video is available for merging
            if context['video_downloaded']:
                return "continue_merge"  # Continue to video merger
            else:
                return "complete"  # Complete if no video to merge
        elif workflow_type == 'tts':
            # Direct TTS workflow completes here
            return "complete"
        
        return "continue"
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """🔄 Reset agent state"""
        logger.info(f"{self.name} reset - clearing TTS context")
        # Could add TTS state cleanup here if needed