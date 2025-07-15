# agents/translation_agent.py - COMPLETE SMART VERSION
import os
import asyncio
import json
import logging
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import openai
from datetime import datetime

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager, Validators

logger = logging.getLogger(__name__)

class TranslationAgent(BaseChatAgent):
    """Smart translation agent with workflow awareness"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Translates text to target languages with workflow intelligence")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        
        # Initialize OpenAI client for translation
        self.openai_client = openai.OpenAI(api_key=config.openai_api_key)
        
        # Translation settings
        self.openai_model = config.openai_model
        self.max_tokens = 4000
        self.temperature = 0.3  # Lower temperature for more consistent translations
        
        # Language detection and mapping
        self.language_codes = {
            'german': 'de', 'french': 'fr', 'spanish': 'es', 'italian': 'it',
            'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja', 'korean': 'ko',
            'chinese': 'zh', 'arabic': 'ar', 'hindi': 'hi', 'dutch': 'nl',
            'swedish': 'sv', 'norwegian': 'no', 'danish': 'da', 'polish': 'pl',
            'finnish': 'fi', 'greek': 'el', 'hebrew': 'he', 'turkish': 'tr',
            'thai': 'th', 'vietnamese': 'vi', 'czech': 'cs', 'hungarian': 'hu'
        }
        
        self.language_names = {
            'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian',
            'pt': 'Portuguese', 'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean',
            'zh': 'Chinese', 'ar': 'Arabic', 'hi': 'Hindi', 'nl': 'Dutch',
            'sv': 'Swedish', 'no': 'Norwegian', 'da': 'Danish', 'pl': 'Polish',
            'fi': 'Finnish', 'el': 'Greek', 'he': 'Hebrew', 'tr': 'Turkish',
            'th': 'Thai', 'vi': 'Vietnamese', 'cs': 'Czech', 'hu': 'Hungarian'
        }
        
        # Workflow stage indicators
        self.workflow_stages = {
            'download_complete': ['downloaded', 'download complete', 'files:', 'audio.wav'],
            'transcription_complete': ['transcribed', 'transcript saved', 'transcription complete'],
            'translation_complete': ['translated', 'translation complete', 'translation saved'],
            'tts_complete': ['generated', 'tts complete', 'audio generated'],
            'merge_complete': ['merged', 'final video', 'workflow complete']
        }
        
        logger.info(f"✅ TranslationAgent '{name}' initialized")
    
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """🔧 SMART: Process with full conversation context awareness"""
        
        try:
            # Analyze complete conversation context
            context = self._analyze_conversation_context(messages)
            
            logger.info(f"🧠 TranslationAgent Context Analysis: {context}")
            
            # Determine if we should process
            should_process, reason = self._should_process_smart(context)
            
            if not should_process:
                logger.info(f"⏭️  TranslationAgent Skipping: {reason}")
                return Response(
                    chat_message=TextMessage(
                        content=f"TranslationAgent: {reason}",
                        source=self.name
                    )
                )
            
            # Find transcript file to translate
            transcript_file_path = self._find_transcript_file(context)
            
            if not transcript_file_path:
                return Response(
                    chat_message=TextMessage(
                        content="❌ No transcript file found for translation. Please transcribe audio first.",
                        source=self.name
                    )
                )
            
            # Execute translation
            result = await self._translate_transcript(transcript_file_path, context)
            
            if result["success"]:
                # Determine if workflow should continue or complete
                workflow_status = self._determine_workflow_status(context)
                
                completion_message = ""
                if workflow_status == "complete":
                    completion_message = " WORKFLOW_COMPLETE"
                elif workflow_status == "continue_tts":
                    completion_message = " Ready for TTS generation."
                
                return Response(
                    chat_message=TextMessage(
                        content=f"✅ Translated transcript to {result['target_language']}. "
                               f"Translation saved to {result['translation_path']}. "
                               f"Source language: {result['source_language']}. "
                               f"Word count: {result['word_count']}. "
                               f"Quality: {result['quality_score']:.2f}/10.{completion_message}",
                        source=self.name
                    )
                )
            else:
                return Response(
                    chat_message=TextMessage(
                        content=f"❌ Translation failed: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TranslationAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"❌ TranslationAgent error: {str(e)}",
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
            'source_language': None,
            'audio_downloaded': False,
            'video_downloaded': False,
            'transcription_complete': False,
            'translation_complete': False,
            'latest_message': "",
            'full_conversation': "",
            'user_request': "",
            'workflow_stage': 'initial',
            'transcript_file_mentioned': None,
            'translation_requested': False
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
        context['target_language'], context['target_language_code'] = self._extract_target_language(conversation_lower)
        
        # Extract source language
        context['source_language'] = self._extract_source_language(conversation_lower)
        
        # Check completion status
        context['audio_downloaded'] = self._check_audio_downloaded(conversation_lower)
        context['video_downloaded'] = self._check_video_downloaded(conversation_lower)
        context['transcription_complete'] = self._check_transcription_complete(conversation_lower)
        context['translation_complete'] = self._check_translation_complete(conversation_lower)
        
        # Determine workflow stage
        context['workflow_stage'] = self._determine_workflow_stage(conversation_lower)
        
        # Check for explicit translation request
        context['translation_requested'] = any(keyword in conversation_lower for keyword in [
            'translate', 'translation', 'convert to', 'change to', 'in german', 'to french'
        ])
        
        # Look for mentioned transcript files
        transcript_file_match = re.search(r'([^\s]+transcript[^\s]*\.txt)', context['full_conversation'])
        if transcript_file_match:
            context['transcript_file_mentioned'] = transcript_file_match.group(1)
        
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
        
        return 'simple_translation'
    
    def _extract_target_language(self, conversation_lower: str) -> tuple[Optional[str], Optional[str]]:
        """🌍 Extract target language from conversation"""
        
        # Look for "to [language]" patterns
        import re
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
                if potential_lang in self.language_codes:
                    return potential_lang, self.language_codes[potential_lang]
        
        # Direct language detection
        for language, code in self.language_codes.items():
            if language in conversation_lower:
                return language, code
        
        # Check for language codes
        for language, code in self.language_codes.items():
            if code in conversation_lower:
                return language, code
        
        return None, None
    
    def _extract_source_language(self, conversation_lower: str) -> Optional[str]:
        """📝 Extract source language from conversation or detect from transcript"""
        
        # Look for language detection from transcription
        import re
        language_patterns = [
            r'language:\s*(\w+)',
            r'detected language:\s*(\w+)',
            r'source language:\s*(\w+)'
        ]
        
        for pattern in language_patterns:
            match = re.search(pattern, conversation_lower)
            if match:
                return match.group(1).lower()
        
        return 'english'  # Default assumption
    
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
        """🤖 SMART: Intelligent decision making for translation processing"""
        
        # Check if translation is already complete
        if context['translation_complete']:
            return False, "Translation already completed"
        
        # Check if transcription is complete (prerequisite for translation)
        if not context['transcription_complete']:
            return False, "Transcription not complete - cannot translate without transcript"
        
        # Check if target language is specified
        if not context['target_language']:
            return False, "No target language specified for translation"
        
        # Workflow-based decisions
        workflow_type = context['workflow_type']
        workflow_stage = context['workflow_stage']
        
        if workflow_type == 'translation':
            # Translation workflow
            if workflow_stage == 'transcription_complete':
                return True, "Translation workflow: transcript ready for translation"
        
        # Check for explicit translation request
        if context['translation_requested']:
            return True, "Explicit translation request detected"
        
        # Check if latest message mentions transcription completion
        latest_lower = context['latest_message'].lower()
        if any(phrase in latest_lower for phrase in [
            'transcribed', 'transcript saved', 'transcription complete'
        ]):
            return True, "Transcription just completed - ready for translation"
        
        return False, "No translation triggers found"
    
    def _find_transcript_file(self, context: Dict[str, Any]) -> Optional[str]:
        """🔍 Find the transcript file to translate"""
        
        # Try to find transcript file based on video ID
        if context['video_id']:
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            transcript_file = transcripts_path / f"{context['video_id']}_transcript.txt"
            
            if transcript_file.exists():
                return str(transcript_file)
        
        # Try to find transcript file mentioned in conversation
        if context['transcript_file_mentioned']:
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            transcript_file = transcripts_path / context['transcript_file_mentioned']
            
            if transcript_file.exists():
                return str(transcript_file)
        
        # Search for any transcript files in transcripts directory
        transcripts_path = self.file_manager.get_temp_path("transcripts")
        if transcripts_path.exists():
            for transcript_file in transcripts_path.glob("*transcript*.txt"):
                if not transcript_file.name.endswith('_translated.txt'):  # Skip already translated files
                    return str(transcript_file)
        
        return None
    
    async def _translate_transcript(self, transcript_file_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """🌍 Translate transcript file using OpenAI"""
        
        try:
            logger.info(f"🌍 Translating transcript file: {transcript_file_path}")
            
            # Validate transcript file
            transcript_path = Path(transcript_file_path)
            if not transcript_path.exists():
                return {"success": False, "error": "Transcript file not found"}
            
            # Read transcript content
            transcript_content = await self._read_transcript_file(transcript_path)
            
            if not transcript_content:
                return {"success": False, "error": "Empty transcript file"}
            
            # Prepare translation parameters
            translation_params = {
                "source_language": context.get('source_language', 'english'),
                "target_language": context['target_language'],
                "target_language_code": context['target_language_code'],
                "transcript_content": transcript_content
            }
            
            # Execute translation
            result = await self._call_openai_translation(translation_params)
            
            if result["success"]:
                # Save translation
                translation_result = await self._save_translation(result["translation_data"], context, transcript_path)
                
                if translation_result["success"]:
                    return {
                        "success": True,
                        "translation_path": translation_result["translation_path"],
                        "target_language": context['target_language'].title(),
                        "source_language": context.get('source_language', 'english').title(),
                        "word_count": len(result["translation_data"]["translated_text"].split()),
                        "quality_score": result["translation_data"].get("quality_score", 8.0)
                    }
                else:
                    return translation_result
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _read_transcript_file(self, transcript_path: Path) -> str:
        """📖 Read and extract transcript content"""
        
        try:
            with open(transcript_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract just the transcript text (remove headers and metadata)
            lines = content.split('\n')
            transcript_lines = []
            in_transcript_section = False
            
            for line in lines:
                if 'TRANSCRIPT TEXT:' in line:
                    in_transcript_section = True
                    continue
                elif line.startswith('WORD-LEVEL TIMESTAMPS:') or line.startswith('====='):
                    in_transcript_section = False
                    continue
                elif in_transcript_section and line.strip() and not line.startswith('-'):
                    transcript_lines.append(line.strip())
            
            # If no structured transcript found, use entire content
            if not transcript_lines:
                # Filter out obvious metadata lines
                for line in lines:
                    if (line.strip() and 
                        not line.startswith('=') and 
                        not line.startswith('-') and 
                        not line.startswith('Source:') and 
                        not line.startswith('Language:') and 
                        not line.startswith('Duration:') and 
                        not line.startswith('Model:') and 
                        not line.startswith('Timestamp:') and 
                        'TRANSCRIPT' not in line):
                        transcript_lines.append(line.strip())
            
            return '\n'.join(transcript_lines)
            
        except Exception as e:
            logger.error(f"Error reading transcript file: {e}")
            return ""
    
    async def _call_openai_translation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """🌐 Call OpenAI for translation"""
        
        try:
            source_lang = params["source_language"].title()
            target_lang = params["target_language"].title()
            transcript_text = params["transcript_content"]
            
            # Create translation prompt
            prompt = f"""
You are a professional translator. Translate the following transcript from {source_lang} to {target_lang}.

Important instructions:
1. Maintain the original meaning and context
2. Use natural, fluent {target_lang} 
3. Preserve any technical terms appropriately
4. Keep the same paragraph structure
5. Ensure cultural appropriateness
6. Maintain the same tone and style

Source text ({source_lang}):
{transcript_text}

Please provide only the translation in {target_lang}:
"""
            
            # Execute API call in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            
            def _translate():
                response = self.openai_client.chat.completions.create(
                    model=self.openai_model,
                    messages=[
                        {"role": "system", "content": f"You are a professional translator specializing in {source_lang} to {target_lang} translation."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                return response
            
            response = await loop.run_in_executor(None, _translate)
            
            # Extract translation
            translated_text = response.choices[0].message.content.strip()
            
            # Calculate quality score (simple heuristic)
            quality_score = self._calculate_quality_score(
                params["transcript_content"], 
                translated_text, 
                params["target_language"]
            )
            
            return {
                "success": True,
                "translation_data": {
                    "original_text": transcript_text,
                    "translated_text": translated_text,
                    "source_language": source_lang,
                    "target_language": target_lang,
                    "quality_score": quality_score,
                    "model_used": self.openai_model,
                    "tokens_used": response.usage.total_tokens if hasattr(response, 'usage') else 0
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_quality_score(self, original_text: str, translated_text: str, target_language: str) -> float:
        """📊 Calculate translation quality score (simple heuristic)"""
        
        try:
            # Basic quality indicators
            score = 8.0  # Base score
            
            # Length similarity (should be relatively similar)
            original_len = len(original_text.split())
            translated_len = len(translated_text.split())
            
            if original_len > 0:
                length_ratio = translated_len / original_len
                if 0.5 <= length_ratio <= 2.0:  # Reasonable length range
                    score += 1.0
                else:
                    score -= 1.0
            
            # Check for untranslated text (should be minimal)
            if original_text.lower() == translated_text.lower():
                score -= 3.0  # Likely not translated
            
            # Check for completeness
            if len(translated_text.strip()) < 10:
                score -= 2.0  # Too short, likely incomplete
            
            # Bonus for reasonable length
            if 50 <= translated_len <= 5000:
                score += 0.5
            
            return max(0.0, min(10.0, score))
            
        except Exception as e:
            logger.error(f"Error calculating quality score: {e}")
            return 7.0  # Default score
    
    async def _save_translation(self, translation_data: Dict[str, Any], context: Dict[str, Any], original_transcript_path: Path) -> Dict[str, Any]:
        """💾 Save translation to file"""
        
        try:
            # Create transcripts directory if it doesn't exist
            transcripts_path = self.file_manager.get_temp_path("transcripts")
            transcripts_path.mkdir(exist_ok=True)
            
            # Determine translation filename
            target_lang = context['target_language_code']
            if context.get('video_id'):
                translation_filename = f"{context['video_id']}_transcript_{target_lang}.txt"
            else:
                translation_filename = f"transcript_{target_lang}.txt"
            
            translation_file = transcripts_path / translation_filename
            
            # Prepare translation content
            translation_content = self._format_translation(translation_data, context)
            
            # Save translation
            with open(translation_file, 'w', encoding='utf-8') as f:
                f.write(translation_content)
            
            logger.info(f"✅ Translation saved to: {translation_file}")
            
            # Also save metadata
            metadata_file = transcripts_path / f"{translation_filename}.json"
            metadata = {
                "video_id": context.get('video_id'),
                "youtube_url": context.get('youtube_url'),
                "source_language": translation_data["source_language"],
                "target_language": translation_data["target_language"],
                "target_language_code": context['target_language_code'],
                "quality_score": translation_data["quality_score"],
                "word_count": len(translation_data["translated_text"].split()),
                "original_transcript": str(original_transcript_path),
                "translation_timestamp": datetime.now().isoformat(),
                "model_used": translation_data["model_used"],
                "tokens_used": translation_data.get("tokens_used", 0)
            }
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return {
                "success": True,
                "translation_path": str(translation_file),
                "metadata_path": str(metadata_file)
            }
            
        except Exception as e:
            logger.error(f"Error saving translation: {e}")
            return {"success": False, "error": str(e)}
    
    def _format_translation(self, translation_data: Dict[str, Any], context: Dict[str, Any]) -> str:
        """📝 Format translation for output"""
        
        lines = []
        
        # Add header
        lines.append("=" * 80)
        lines.append("TRANSLATED TRANSCRIPT")
        lines.append("=" * 80)
        
        if context.get('youtube_url'):
            lines.append(f"Source: {context['youtube_url']}")
        
        lines.append(f"Original Language: {translation_data['source_language']}")
        lines.append(f"Target Language: {translation_data['target_language']}")
        lines.append(f"Quality Score: {translation_data['quality_score']:.1f}/10")
        lines.append(f"Model: {translation_data['model_used']}")
        lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if translation_data.get('tokens_used'):
            lines.append(f"Tokens Used: {translation_data['tokens_used']}")
        
        lines.append("")
        
        # Add translation text
        lines.append("TRANSLATED TEXT:")
        lines.append("-" * 40)
        
        translated_text = translation_data["translated_text"]
        if translated_text:
            # Add paragraphs for better readability
            paragraphs = translated_text.split('\n\n')
            for paragraph in paragraphs:
                if paragraph.strip():
                    # Break long paragraphs
                    if len(paragraph) > 300:
                        sentences = paragraph.split('. ')
                        current_para = ""
                        for sentence in sentences:
                            if len(current_para) + len(sentence) > 300:
                                if current_para:
                                    lines.append(current_para.strip())
                                    current_para = ""
                            current_para += sentence + ". "
                        if current_para:
                            lines.append(current_para.strip())
                    else:
                        lines.append(paragraph.strip())
                    lines.append("")  # Empty line between paragraphs
        else:
            lines.append("No translation text available.")
        
        # Add original text for reference
        lines.append("")
        lines.append("ORIGINAL TEXT (for reference):")
        lines.append("-" * 40)
        
        original_text = translation_data["original_text"]
        if original_text:
            # Show first 200 characters
            preview = original_text[:200] + "..." if len(original_text) > 200 else original_text
            lines.append(preview)
        
        lines.append("")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def _determine_workflow_status(self, context: Dict[str, Any]) -> str:
        """🎯 Determine if workflow should complete or continue after translation"""
        
        workflow_type = context['workflow_type']
        
        if workflow_type == 'translation':
            # Check if video is available for merging
            if context['video_downloaded']:
                return "continue_tts"  # Continue to TTS for full translation workflow
            else:
                return "complete"  # Complete if only text translation needed
        
        return "continue"
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """🔄 Reset agent state"""
        logger.info(f"{self.name} reset - clearing translation context")
        # Could add translation state cleanup here if needed