# agents/transcription_agent.py
import os
import asyncio
import whisper
import json
from typing import Dict, Any, Optional, Sequence
from pathlib import Path
import logging

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import Config, FileManager

logger = logging.getLogger(__name__)

class TranscriptionAgent(BaseChatAgent):
    """Agent responsible for transcribing audio and generating summaries"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Transcribes audio and generates summaries")
        self.config = config
        self.file_manager = file_manager
        self.whisper_model = None
        self.openai_client = OpenAIChatCompletionClient(
            model=config.openai_model,
            api_key=config.openai_api_key
        )
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages for transcription or summarization requests"""
        
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
                    content="This agent handles transcription and summarization requests.",
                    source=self.name
                )
            )
        
        try:
            # Determine the task type
            task_type = self._determine_task_type(content)
            
            if task_type == "transcribe":
                result = await self._transcribe_audio(content)
            elif task_type == "summarize":
                result = await self._summarize_content(content)
            else:
                return Response(
                    chat_message=TextMessage(
                        content="Please specify whether you want transcription or summarization.",
                        source=self.name
                    )
                )
            
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
                        content=f"Error: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in TranscriptionAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        keywords = ["transcribe", "transcript", "summarize", "summary", "convert to text"]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    def _determine_task_type(self, content: str) -> str:
        """Determine whether user wants transcription or summarization"""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["summarize", "summary", "summarise"]):
            return "summarize"
        elif any(word in content_lower for word in ["transcribe", "transcript", "text"]):
            return "transcribe"
        
        return "unknown"
    
    async def _transcribe_audio(self, content: str) -> Dict[str, Any]:
        """Transcribe audio file to text"""
        try:
            # Find audio file from previous agent or specified path
            audio_file_path = self._extract_audio_file_path(content)
            
            if not audio_file_path or not Path(audio_file_path).exists():
                return {"success": False, "error": "Audio file not found"}
            
            # Load Whisper model if not already loaded
            if self.whisper_model is None:
                loop = asyncio.get_event_loop()
                self.whisper_model = await loop.run_in_executor(
                    None, whisper.load_model, self.config.whisper_model
                )
            
            # Transcribe audio
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None, self._transcribe_with_whisper, audio_file_path
            )
            
            if result["success"]:
                # Save transcript to file
                transcript_file = self.file_manager.get_temp_path(
                    f"transcript_{Path(audio_file_path).stem}.txt",
                    "transcripts"
                )
                
                async with self.file_manager.save_file(
                    result["transcript"].encode('utf-8'),
                    transcript_file
                ):
                    pass
                
                # Also save detailed result as JSON
                detailed_file = self.file_manager.get_temp_path(
                    f"transcript_detailed_{Path(audio_file_path).stem}.json",
                    "transcripts"
                )
                
                detailed_data = {
                    "transcript": result["transcript"],
                    "language": result["language"],
                    "segments": result.get("segments", []),
                    "audio_file": audio_file_path
                }
                
                async with self.file_manager.save_file(
                    json.dumps(detailed_data, indent=2).encode('utf-8'),
                    detailed_file
                ):
                    pass
                
                return {
                    "success": True,
                    "message": f"Successfully transcribed audio. "
                             f"Language: {result['language']}. "
                             f"Transcript saved to: {transcript_file}. "
                             f"Text: {result['transcript'][:200]}..."
                }
            else:
                return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {"success": False, "error": str(e)}
    
    def _transcribe_with_whisper(self, audio_file_path: str) -> Dict[str, Any]:
        """Transcribe using Whisper model (runs in thread pool)"""
        try:
            result = self.whisper_model.transcribe(audio_file_path)
            return {
                "success": True,
                "transcript": result["text"],
                "language": result["language"],
                "segments": result.get("segments", [])
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _summarize_content(self, content: str) -> Dict[str, Any]:
        """Summarize transcript content"""
        try:
            # Extract transcript text or file
            transcript_text = self._extract_transcript_content(content)
            
            if not transcript_text:
                return {"success": False, "error": "No transcript content found to summarize"}
            
            # Create summarization prompt
            prompt = self._create_summary_prompt(transcript_text)
            
            # Use OpenAI to generate summary
            from autogen_core.models import UserMessage
            
            result = await self.openai_client.create([
                UserMessage(content=prompt, source="user")
            ])
            
            summary = result.content
            
            # Save summary to file
            summary_file = self.file_manager.get_temp_path(
                "summary.txt",
                "transcripts"
            )
            
            await self.file_manager.save_file(
                summary.encode('utf-8'),
                summary_file
            )
            
            return {
                "success": True,
                "message": f"Successfully generated summary. "
                         f"Summary saved to: {summary_file}. "
                         f"Summary: {summary[:300]}..."
            }
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return {"success": False, "error": str(e)}
    
    def _extract_audio_file_path(self, content: str) -> Optional[str]:
        """Extract audio file path from message content"""
        import re
        
        # Look for file paths in the content
        path_patterns = [
            r'File: ([^\s]+\.wav)',
            r'file_path["\']?\s*[:=]\s*["\']?([^"\']+\.wav)',
            r'([^\s]+\.wav)'
        ]
        
        for pattern in path_patterns:
            match = re.search(pattern, content)
            if match:
                return match.group(1)
        
        # Check for recent audio files in temp folder
        audio_folder = self.file_manager.get_temp_path("", "audio")
        if audio_folder.exists():
            audio_files = list(audio_folder.glob("*.wav"))
            if audio_files:
                # Return the most recent file
                return str(max(audio_files, key=lambda p: p.stat().st_mtime))
        
        return None
    
    def _extract_transcript_content(self, content: str) -> Optional[str]:
        """Extract transcript content for summarization"""
        # First, try to find transcript file
        transcript_file = self._find_transcript_file()
        if transcript_file:
            try:
                with open(transcript_file, 'r', encoding='utf-8') as f:
                    return f.read()
            except Exception as e:
                logger.error(f"Error reading transcript file: {e}")
        
        # If no file found, check if transcript is in the message
        if len(content) > 100:  # Assume long content might be transcript
            return content
        
        return None
    
    def _find_transcript_file(self) -> Optional[Path]:
        """Find the most recent transcript file"""
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            transcript_files = list(transcript_folder.glob("transcript_*.txt"))
            if transcript_files:
                return max(transcript_files, key=lambda p: p.stat().st_mtime)
        return None
    
    def _create_summary_prompt(self, transcript: str) -> str:
        """Create prompt for summarization"""
        return f"""
Please provide a comprehensive summary of the following transcript. 
Include the main topics, key points, and important details.
Structure the summary with clear sections if applicable.

Transcript:
{transcript}

Summary:
"""
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")
        await self.openai_client.close()