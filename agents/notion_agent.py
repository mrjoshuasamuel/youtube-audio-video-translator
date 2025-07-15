# agents/notion_agent.py
import asyncio
from typing import Dict, Any, Optional, Sequence, List
from pathlib import Path
import logging
import json
from datetime import datetime

from autogen_agentchat.agents import BaseChatAgent
from autogen_agentchat.base import Response
from autogen_agentchat.messages import BaseChatMessage, TextMessage
from autogen_core import CancellationToken

from notion_client import Client as NotionClient
from utils import Config, FileManager

logger = logging.getLogger(__name__)

class NotionAgent(BaseChatAgent):
    """Agent responsible for posting content to Notion database"""
    
    def __init__(self, name: str, config: Config, file_manager: FileManager):
        super().__init__(name, description="Posts summaries, transcripts, and media to Notion database")
        self.config = config
        self.file_manager = file_manager
        self.notion_client = NotionClient(auth=config.notion_api_key)
        
    @property
    def produced_message_types(self) -> Sequence[type[BaseChatMessage]]:
        return (TextMessage,)
    
    async def on_messages(self, messages: Sequence[BaseChatMessage], cancellation_token: CancellationToken) -> Response:
        """Process messages for Notion posting requests"""
        
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
                    content="This agent posts content to Notion database.",
                    source=self.name
                )
            )
        
        try:
            # Gather content to post
            content_data = await self._gather_content_data()
            
            if not content_data:
                return Response(
                    chat_message=TextMessage(
                        content="No content found to post to Notion.",
                        source=self.name
                    )
                )
            
            # Post to Notion
            result = await self._post_to_notion(content_data)
            
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
                        content=f"Notion posting error: {result['error']}",
                        source=self.name
                    )
                )
                
        except Exception as e:
            logger.error(f"Error in NotionAgent: {e}")
            return Response(
                chat_message=TextMessage(
                    content=f"An error occurred: {str(e)}",
                    source=self.name
                )
            )
    
    def _should_process(self, content: str) -> bool:
        """Check if this agent should process the message"""
        keywords = [
            "post to notion", "save to notion", "notion database", 
            "upload to notion", "store in notion", "add to notion"
        ]
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in keywords)
    
    async def _gather_content_data(self) -> Optional[Dict[str, Any]]:
        """Gather all available content for posting to Notion"""
        try:
            content_data = {
                "title": "YouTube Video Processing Result",
                "youtube_url": None,
                "video_title": None,
                "original_transcript": None,
                "summary": None,
                "translations": {},
                "audio_files": [],
                "video_files": [],
                "processing_date": datetime.now().isoformat(),
                "languages_processed": []
            }
            
            # Extract YouTube URL and title from conversation or files
            youtube_info = await self._extract_youtube_info()
            if youtube_info:
                content_data.update(youtube_info)
            
            # Get original transcript
            original_transcript = await self._get_original_transcript()
            if original_transcript:
                content_data["original_transcript"] = original_transcript
            
            # Get summary
            summary = await self._get_summary()
            if summary:
                content_data["summary"] = summary
            
            # Get translations
            translations = await self._get_translations()
            content_data["translations"] = translations
            content_data["languages_processed"] = list(translations.keys())
            
            # Get generated files
            audio_files = await self._get_audio_files()
            content_data["audio_files"] = audio_files
            
            video_files = await self._get_video_files()
            content_data["video_files"] = video_files
            
            return content_data if any([
                content_data["original_transcript"],
                content_data["summary"],
                content_data["translations"],
                content_data["audio_files"],
                content_data["video_files"]
            ]) else None
            
        except Exception as e:
            logger.error(f"Error gathering content data: {e}")
            return None
    
    async def _extract_youtube_info(self) -> Optional[Dict[str, str]]:
        """Extract YouTube URL and video title"""
        # Check for video info JSON files
        video_folder = self.file_manager.get_temp_path("", "video")
        if video_folder.exists():
            info_files = list(video_folder.glob("*.info.json"))
            if info_files:
                latest_info = max(info_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_info, 'r', encoding='utf-8') as f:
                        info_data = json.load(f)
                        return {
                            "youtube_url": info_data.get("webpage_url", ""),
                            "video_title": info_data.get("title", "Unknown Video"),
                            "video_id": info_data.get("id", ""),
                            "duration": info_data.get("duration", 0),
                            "uploader": info_data.get("uploader", ""),
                            "upload_date": info_data.get("upload_date", "")
                        }
                except Exception as e:
                    logger.error(f"Error reading video info: {e}")
        
        return None
    
    async def _get_original_transcript(self) -> Optional[str]:
        """Get original transcript text"""
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            # Look for original transcript files (not translated)
            transcript_files = [
                f for f in transcript_folder.glob("transcript_*.txt")
                if "translated" not in f.name
            ]
            if transcript_files:
                latest_file = max(transcript_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading transcript: {e}")
        return None
    
    async def _get_summary(self) -> Optional[str]:
        """Get summary text"""
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        if transcript_folder.exists():
            summary_files = [
                f for f in transcript_folder.glob("summary*.txt")
                if "translated" not in f.name
            ]
            if summary_files:
                latest_file = max(summary_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"Error reading summary: {e}")
        return None
    
    async def _get_translations(self) -> Dict[str, Dict[str, str]]:
        """Get all translation files"""
        translations = {}
        transcript_folder = self.file_manager.get_temp_path("", "transcripts")
        
        if transcript_folder.exists():
            # Get translated transcripts
            translated_files = list(transcript_folder.glob("*_translated_*.txt"))
            
            for file_path in translated_files:
                try:
                    # Extract language and content type from filename
                    filename = file_path.name
                    parts = filename.split("_")
                    
                    if len(parts) >= 3:
                        content_type = parts[0]  # transcript or summary
                        language = parts[2].replace(".txt", "")
                        
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        if language not in translations:
                            translations[language] = {}
                        
                        translations[language][content_type] = content
                        
                except Exception as e:
                    logger.error(f"Error reading translation file {file_path}: {e}")
        
        return translations
    
    async def _get_audio_files(self) -> List[Dict[str, str]]:
        """Get information about generated audio files"""
        audio_files = []
        audio_folder = self.file_manager.get_temp_path("", "audio")
        
        if audio_folder.exists():
            tts_files = list(audio_folder.glob("tts_*.wav"))
            
            for file_path in tts_files:
                try:
                    file_size = self.file_manager.get_file_size_mb(file_path)
                    audio_files.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size_mb": round(file_size, 2),
                        "type": "TTS Generated Audio"
                    })
                except Exception as e:
                    logger.error(f"Error processing audio file {file_path}: {e}")
        
        return audio_files
    
    async def _get_video_files(self) -> List[Dict[str, str]]:
        """Get information about generated video files"""
        video_files = []
        
        # Check output folder for final videos
        output_folder = self.file_manager.output_folder / "final"
        if output_folder.exists():
            video_files_paths = list(output_folder.glob("*.mp4"))
            
            for file_path in video_files_paths:
                try:
                    file_size = self.file_manager.get_file_size_mb(file_path)
                    video_files.append({
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size_mb": round(file_size, 2),
                        "type": "Translated Video"
                    })
                except Exception as e:
                    logger.error(f"Error processing video file {file_path}: {e}")
        
        return video_files
    
    async def _post_to_notion(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Post content to Notion database"""
        try:
            # Prepare page properties
            properties = self._prepare_notion_properties(content_data)
            
            # Prepare page content (blocks)
            blocks = self._prepare_notion_blocks(content_data)
            
            # Create page in Notion
            loop = asyncio.get_event_loop()
            page = await loop.run_in_executor(
                None,
                self._create_notion_page,
                properties,
                blocks
            )
            
            if page:
                page_url = page.get("url", "")
                return {
                    "success": True,
                    "message": f"Successfully posted content to Notion. "
                             f"Page URL: {page_url}",
                    "page_id": page.get("id", ""),
                    "page_url": page_url
                }
            else:
                return {"success": False, "error": "Failed to create Notion page"}
                
        except Exception as e:
            logger.error(f"Error posting to Notion: {e}")
            return {"success": False, "error": str(e)}
    
    def _prepare_notion_properties(self, content_data: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare properties for Notion page"""
        properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": content_data.get("video_title", "YouTube Video Processing Result")
                        }
                    }
                ]
            },
            "Processing Date": {
                "date": {
                    "start": content_data["processing_date"][:10]  # Just the date part
                }
            },
            "YouTube URL": {
                "url": content_data.get("youtube_url", "")
            },
            "Languages": {
                "multi_select": [
                    {"name": lang} for lang in content_data.get("languages_processed", [])
                ]
            },
            "Status": {
                "select": {
                    "name": "Completed"
                }
            }
        }
        
        # Add optional properties if they exist
        if content_data.get("video_id"):
            properties["Video ID"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": content_data["video_id"]
                        }
                    }
                ]
            }
        
        return properties
    
    def _prepare_notion_blocks(self, content_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prepare content blocks for Notion page"""
        blocks = []
        
        # Add overview
        blocks.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {
                "rich_text": [{"type": "text", "text": {"content": "Processing Overview"}}]
            }
        })
        
        overview_text = f"Video processed on {content_data['processing_date'][:10]}\n"
        if content_data.get("youtube_url"):
            overview_text += f"Source: {content_data['youtube_url']}\n"
        if content_data.get("languages_processed"):
            overview_text += f"Languages: {', '.join(content_data['languages_processed'])}"
        
        blocks.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": overview_text}}]
            }
        })
        
        # Add original transcript if available
        if content_data.get("original_transcript"):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Original Transcript"}}]
                }
            })
            
            # Split long transcript into chunks
            transcript_chunks = self._split_text_for_notion(content_data["original_transcript"])
            for chunk in transcript_chunks:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    }
                })
        
        # Add summary if available
        if content_data.get("summary"):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Summary"}}]
                }
            })
            
            summary_chunks = self._split_text_for_notion(content_data["summary"])
            for chunk in summary_chunks:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    }
                })
        
        # Add translations
        if content_data.get("translations"):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Translations"}}]
                }
            })
            
            for language, translations in content_data["translations"].items():
                blocks.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": f"{language.upper()}"}}]
                    }
                })
                
                for content_type, content in translations.items():
                    blocks.append({
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": [
                                {"type": "text", "text": {"content": f"{content_type.title()}: "}},
                                {"type": "text", "text": {"content": content[:500] + "..." if len(content) > 500 else content}}
                            ]
                        }
                    })
        
        # Add file information
        if content_data.get("audio_files") or content_data.get("video_files"):
            blocks.append({
                "object": "block",
                "type": "heading_3",
                "heading_3": {
                    "rich_text": [{"type": "text", "text": {"content": "Generated Files"}}]
                }
            })
            
            all_files = content_data.get("audio_files", []) + content_data.get("video_files", [])
            for file_info in all_files:
                file_text = f"• {file_info['filename']} ({file_info['size_mb']} MB) - {file_info['type']}"
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": file_text}}]
                    }
                })
        
        return blocks
    
    def _split_text_for_notion(self, text: str, max_length: int = 2000) -> List[str]:
        """Split text into chunks suitable for Notion blocks"""
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
    
    def _create_notion_page(self, properties: Dict[str, Any], blocks: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create Notion page (runs in thread pool)"""
        try:
            page = self.notion_client.pages.create(
                parent={"database_id": self.config.notion_database_id},
                properties=properties,
                children=blocks
            )
            return page
        except Exception as e:
            logger.error(f"Error creating Notion page: {e}")
            return None
    
    async def on_reset(self, cancellation_token: CancellationToken) -> None:
        """Reset agent state"""
        logger.info(f"{self.name} reset")