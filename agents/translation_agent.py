# agents/translation_agent.py
import asyncio
import json
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import logging

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from deep_translator import GoogleTranslator
from utils import Config, FileManager, Validators

logger = logging.getLogger(__name__)

class TranslationAgent(BaseChatAgent):
    """Agent responsible for translating transcripts to different languages"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Translates transcripts and summaries to different languages")
        self.config = config
        self.file_manager = file_manager
        self.validators = Validators()
        self.openai_client = OpenAIChatCompletionClient(
            model=config.openai_model,
            api_key=config.openai_api_key
        )
        
        # Language code mapping for better user experience
        self.language_codes = {
            'english': 'en', 'spanish': 'es', 'french': 'fr', 'german': 'de',
            'italian': 'it', 'portuguese': 'pt', 'russian': 'ru', 'japanese': 'ja',
            'korean': 'ko', 'chinese': 'zh', 'arabic': 'ar', 'hindi': 'hi',
            'dutch': 'nl', 'swedish': 'sv', 'norwegian': 'no', 'danish': 'da',
            'finnish': 'fi', 'polish': 'pl', 'czech': 'cs', 'hungarian': 'hu'
        }
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages for translation requests"""
        
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
                    content="This agent handles translation requests for transcripts and summaries.",
                    source=self.name
                )
            )
        
        try:
            # Extract target language
            target_language = self._extract_target_language(content)
            if not target_language:
                return Response(
                    chat_message=TextMessage(
                        content="Please specify the target language for translation. "
                               f"Supported languages: {', '.join(self.config.supported_languages)}",
                        source=self.name
                    )
                )
            
            # Determine content type (transcript or summary)
            content_type = self._determine_content_type(content)
            
            # Perform translation
            result = await self._translate_content(content, target_language, content_type)
            
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
                        content=f"Translation error: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TranslationAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        keywords = ["translate", "translation", "convert to", "in language", "to spanish", "to french"]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _extract_target_language(self, content: str) -> Optional[str]:
        """Extract target language from message content"""
        content_lower = content.lower()
        
        # Check for language names
        for lang_name, lang_code in self.language_codes.items():
            if lang_name in content_lower or lang_code in content_lower:
                if self.validators.is_supported_language(lang_code, self.config.supported_languages):
                    return lang_code
        
        # Check for patterns like "to Spanish", "in French"
        import re
        patterns = [
            r'(?:to|in|into)\s+(\w+)',
            r'translate.*?(?:to|into)\s+(\w+)',
            r'(\w+)\s+(?:language|translation)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content_lower)
            for match in matches:
                if match in self.language_codes:
                    lang_code = self.language_codes[match]
                    if self.validators.is_supported_language(lang_code, self.config.supported_languages):
                        return lang_code
        
        return None
    
    def _determine_content_type(self, content: str) -> str:
        """Determine if user wants to translate transcript or summary"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["summary", "summarize"]):
            return "summary"
        elif any(word in content_lower for word in ["transcript", "transcription"]):
            return "transcript"
        
        # Default to transcript
        return "transcript"
    
    async def _translate_content(self, content: str, target_language: str, content_type: str) -> Dict[str, Any]:
        """Translate content to target language"""
        try:
            # Get source content
            source_text = await self._get_source_content(content_type)
            
            if not source_text:
                return {"success": False, "error": f"No {content_type} content found to translate"}
            
            # Choose translation method based on content length
            if len(source_text) > 5000:  # Use OpenAI for long texts
                translated_text = await self._translate_with_openai(source_text, target_language)
            else:  # Use Google Translator for shorter texts
                translated_text = await self._translate_with_google(source_text, target_language)
            
            if not translated_text:
                return {"success": False, "error": "Translation failed"}
            
            # Save translated content
            filename = f"{content_type}_translated_{target_language}.txt"
            translated_file = self.file_manager.get_temp_path(filename, "transcripts")
            
            await self.file_manager.save_file(
                translated_text.encode('utf-8'),
                translated_file
            )
            
            # Also save as JSON with metadata
            metadata = {
                "original_content": source_text[:500] + "..." if len(source_text) > 500 else source_text,
                "translated_content": translated_text,
                "source_language": "auto-detected",
                "target_language": target_language,
                "content_type": content_type,
                "translation_method": "openai" if len(source_text) > 5000 else "google"
            }
            
            json_filename = f"{content_type}_translated_{target_language}.json"
            json_file = self.file_manager.get_temp_path(json_filename, "transcripts")
            
            await self.file_manager.save_file(
                json.dumps(metadata, indent=2, ensure_ascii=False).encode('utf-8'),
                json_file
            )
            
            return {
                "success": True,
                "message": f"Successfully translated {content_type} to {target_language}. "
                         f"Translation saved to: {translated_file}. "
                         f"Preview: {translated_text[:200]}..."
            }
            
        except Exception as e:
            logger.error(f"Error in translation: {e}")
            return {"success": False, "error": str(e)}
    
    async def _get_source_content(self, content_type: str) -> Optional[str]:
        """Get source content for translation"""
        if content_type == "summary":
            file_pattern = "summary*.txt"
        else:
            file_pattern = "transcript*.txt"
        
        # Find the most recent file
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            files = list(transcript_folder.glob(file_pattern))
            if files:
                latest_file = max(files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading file {latest_file}: {e}")
        
        return None
    
    async def _translate_with_google(self, text: str, target_language: str) -> Optional[str]:
        """Translate using Google Translator"""
        try:
            loop = asyncio.get_event_loop()
            
            # Split text into chunks if too long
            chunks = self._split_text(text, max_length=4500)  # Google has 5000 char limit
            translated_chunks = []
            
            for chunk in chunks:
                translator = GoogleTranslator(source='auto', target=target_language)
                translated_chunk = await loop.run_in_executor(
                    None, translator.translate, chunk
                )
                translated_chunks.append(translated_chunk)
            
            return " ".join(translated_chunks)
            
        except Exception as e:
            logger.error(f"Google translation error: {e}")
            return None
    
    async def _translate_with_openai(self, text: str, target_language: str) -> Optional[str]:
        """Translate using OpenAI"""
        try:
            # Map language code to language name
            language_names = {v: k for k, v in self.language_codes.items()}
            target_lang_name = language_names.get(target_language, target_language)
            
            prompt = f"""
Please translate the following text to {target_lang_name}. 
Maintain the original structure and formatting as much as possible.
Provide only the translation without additional comments.

Text to translate:
{text}

Translation:
"""
            
            from autogen_core.models import UserMessage
            
            result = await self.openai_client.create([
                UserMessage(content=prompt, source="user")
            ])
            
            return result.content
            
        except Exception as e:
            logger.error(f"OpenAI translation error: {e}")
            return None
    
    def _split_text(self, text: str, max_length: int) -> list:
        """Split text into chunks while preserving sentence boundaries"""
        if len(text) <= max_length:
            return [text]
        
        chunks = []
        current_chunk = ""
        
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
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")
        await self.openai_client.close()