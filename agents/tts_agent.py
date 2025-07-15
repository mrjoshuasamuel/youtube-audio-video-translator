# agents/tts_agent.py
import asyncio
import aiohttp
import json
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import logging
import base64

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from utils import Config, FileManager

logger = logging.getLogger(__name__)

class TTSAgent(BaseChatAgent):
    """Agent responsible for generating audio from text using Murf API"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Generates audio from translated text using Murf API")
        self.config = config
        self.file_manager = file_manager
        
        # Murf API configuration
        self.murf_api_url = "https://api.murf.ai/v1/speech/generate"
        self.murf_voices = {
            'en': {'voice_id': 'en-US-davis', 'name': 'Davis'},
            'es': {'voice_id': 'es-ES-alvaro', 'name': 'Alvaro'},
            'fr': {'voice_id': 'fr-FR-antoine', 'name': 'Antoine'},
            'de': {'voice_id': 'de-DE-bernd', 'name': 'Bernd'},
            'it': {'voice_id': 'it-IT-alberto', 'name': 'Alberto'},
            'pt': {'voice_id': 'pt-BR-antonio', 'name': 'Antonio'},
            'ru': {'voice_id': 'ru-RU-dmitri', 'name': 'Dmitri'},
            'ja': {'voice_id': 'ja-JP-akira', 'name': 'Akira'},
            'ko': {'voice_id': 'ko-KR-hyunwoo', 'name': 'Hyunwoo'},
            'zh': {'voice_id': 'zh-CN-yunxi', 'name': 'Yunxi'}
        }
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages for text-to-speech requests"""
        
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
                    content="This agent generates audio from translated text.",
                    source=self.name
                )
            )
        
        try:
            # Extract language and determine source text
            language = self._extract_language(content)
            if not language:
                return Response(
                    chat_message=TextMessage(
                        content="Could not determine target language for TTS generation.",
                        source=self.name
                    )
                )
            
            # Generate TTS audio
            result = await self._generate_tts_audio(content, language)
            
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
                        content=f"TTS generation error: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TTSAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        keywords = [
            "generate audio", "text to speech", "tts", "voice over",
            "create audio", "audio from text", "speech synthesis"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _extract_language(self, content: str) -> Optional[str]:
        """Extract target language from content or find from translated files"""
        content_lower = content.lower()
        
        # Check for explicit language mention
        for lang_code in self.murf_voices.keys():
            if lang_code in content_lower:
                return lang_code
        
        # Check for translated files in the temp folder
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            translated_files = list(transcript_folder.glob("*_translated_*.txt"))
            if translated_files:
                # Extract language from filename
                latest_file = max(translated_files, key=lambda p: p.stat().st_mtime)
                filename = latest_file.name
                for lang_code in self.murf_voices.keys():
                    if f"_{lang_code}." in filename:
                        return lang_code
        
        # Default to English
        return 'en'
    
    async def _generate_tts_audio(self, content: str, language: str) -> Dict[str, Any]:
        """Generate TTS audio using Murf API"""
        try:
            # Get the text to convert to speech
            text_content = await self._get_text_content(content, language)
            
            if not text_content:
                return {"success": False, "error": "No text content found for TTS generation"}
            
            # Get voice configuration for the language
            voice_config = self.murf_voices.get(language, self.murf_voices['en'])
            
            # Split text into chunks if too long (Murf has character limits)
            text_chunks = self._split_text_for_tts(text_content)
            
            # Generate audio for each chunk
            audio_files = []
            for i, chunk in enumerate(text_chunks):
                chunk_result = await self._generate_audio_chunk(chunk, voice_config, i)
                if chunk_result["success"]:
                    audio_files.append(chunk_result["file_path"])
                else:
                    return {"success": False, "error": f"Failed to generate audio for chunk {i}: {chunk_result['error']}"}
            
            # Merge audio files if multiple chunks
            if len(audio_files) > 1:
                final_audio_path = await self._merge_audio_files(audio_files, language)
            else:
                final_audio_path = audio_files[0]
            
            return {
                "success": True,
                "message": f"Successfully generated TTS audio in {language}. "
                         f"Audio saved to: {final_audio_path}. "
                         f"Voice: {voice_config['name']}",
                "file_path": final_audio_path,
                "language": language,
                "voice": voice_config['name']
            }
            
        except Exception as e:
            logger.error(f"Error in TTS generation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_text_content(self, content: str, language: str) -> Optional[str]:
        """Get text content for TTS generation"""
        # First, try to find translated file for the language
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            # Look for translated transcript first, then translated summary
            patterns = [
                f"transcript_translated_{language}.txt",
                f"summary_translated_{language}.txt",
                f"*_translated_{language}.txt"
            ]
            
            for pattern in patterns:
                files = list(transcript_folder.glob(pattern))
                if files:
                    latest_file = max(files, key=lambda p: p.stat().st_mtime)
                    try:
                        with open(latest_file, 'r', encoding='utf-8') as f:
                            return f.read()
                    except Exception as e:
                        logger.error(f"Error reading file {latest_file}: {e}")
        
        # If no translated file found, check if content contains the text directly
        if len(content) > 100:  # Assume long content might be the text itself
            return content
        
        return None
    
    async def _generate_audio_chunk(self, text: str, voice_config: Dict[str, str], chunk_index: int) -> Dict[str, Any]:
        """Generate audio for a single text chunk using Murf API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.config.murf_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "voiceId": voice_config["voice_id"],
                "text": text,
                "format": "WAV",
                "sampleRate": 22050,
                "bitRate": 320000,
                "speed": 1.0,
                "pitch": 1.0,
                "emphasis": 1.0
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.murf_api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=300)  # 5 minutes timeout
                ) as response:
                    
                    if response.status == 200:
                        response_data = await response.json()
                        
                        # Extract audio data (base64 encoded)
                        audio_data = response_data.get("audioContent")
                        if audio_data:
                            # Decode base64 audio
                            audio_bytes = base64.b64decode(audio_data)
                            
                            # Save audio file
                            filename = f"tts_chunk_{chunk_index}_{voice_config['voice_id']}.wav"
                            audio_file_path = self.file_manager.get_temp_path(filename, "audio")
                            
                            await self.file_manager.save_file(audio_bytes, audio_file_path)
                            
                            return {
                                "success": True,
                                "file_path": str(audio_file_path)
                            }
                        else:
                            return {"success": False, "error": "No audio content in response"}
                    else:
                        error_text = await response.text()
                        return {"success": False, "error": f"Murf API error {response.status}: {error_text}"}
            
        except asyncio.TimeoutError:
            return {"success": False, "error": "TTS generation timed out"}
        except Exception as e:
            logger.error(f"Error calling Murf API: {e}")
            return {"success": False, "error": str(e)}
    
    def _split_text_for_tts(self, text: str, max_length: int = 3000) -> list:
        """Split text into chunks suitable for TTS API"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
        # Split by sentences first
        sentences = text.split('. ')
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 2 <= max_length:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    async def _merge_audio_files(self, audio_files: list, language: str) -> str:
        """Merge multiple audio files into one"""
        try:
            from pydub import AudioSegment
            
            # Load all audio files
            audio_segments = []
            for file_path in audio_files:
                try:
                    audio = AudioSegment.from_wav(file_path)
                    audio_segments.append(audio)
                except Exception as e:
                    logger.error(f"Error loading audio file {file_path}: {e}")
                    continue
            
            if not audio_segments:
                raise Exception("No valid audio segments to merge")
            
            # Merge all segments
            merged_audio = audio_segments[0]
            for segment in audio_segments[1:]:
                merged_audio += segment
            
            # Save merged audio
            merged_filename = f"tts_merged_{language}.wav"
            merged_path = self.file_manager.get_temp_path(merged_filename, "audio")
            
            merged_audio.export(str(merged_path), format="wav")
            
            # Clean up chunk files
            for file_path in audio_files:
                try:
                    Path(file_path).unlink()
                except Exception:
                    pass
            
            return str(merged_path)
            
        except Exception as e:
            logger.error(f"Error merging audio files: {e}")
            # Return the first file if merging fails
            return audio_files[0] if audio_files else ""
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")