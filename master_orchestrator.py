# master_orchestrator.py
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from autogen_agentchat.teams import DiGraphBuilder, GraphFlow
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import Config, FileManager, Validators
from agents import (
    YouTubeDownloaderAgent,
    TranscriptionAgent,
    TranslationAgent,
    TTSAgent,
    VideoMergerAgent,
    NotionAgent
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterOrchestrator:
    """Master orchestrator for the YouTube video processing workflow"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = Config(_env_file=config_path) if config_path else Config()
        
        # Initialize file manager
        self.file_manager = FileManager(
            temp_folder=self.config.temp_folder,
            output_folder=self.config.output_folder
        )
        
        # Initialize validators
        self.validators = Validators()
        
        # Initialize agents
        self._initialize_agents()
        
        # Build workflow graph
        self._build_workflow_graph()
        
        logger.info("Master Orchestrator initialized successfully")
    
    def _initialize_agents(self):
        """Initialize all specialized agents"""
        self.downloader_agent = YouTubeDownloaderAgent(
            "downloader", self.config, self.file_manager
        )
        
        self.transcription_agent = TranscriptionAgent(
            "transcription", self.config, self.file_manager
        )
        
        self.translation_agent = TranslationAgent(
            "translation", self.config, self.file_manager
        )
        
        self.tts_agent = TTSAgent(
            "tts", self.config, self.file_manager
        )
        
        self.video_merger_agent = VideoMergerAgent(
            "video_merger", self.config, self.file_manager
        )
        
        self.notion_agent = NotionAgent(
            "notion", self.config, self.file_manager
        )
        
        logger.info("All agents initialized")
    
    def _build_workflow_graph(self):
        """Build the workflow graph using DiGraphBuilder"""
        builder = DiGraphBuilder()
        
        # Add all agents as nodes
        builder.add_node(self.downloader_agent)
        builder.add_node(self.transcription_agent)
        builder.add_node(self.translation_agent)
        builder.add_node(self.tts_agent)
        builder.add_node(self.video_merger_agent)
        builder.add_node(self.notion_agent)
        
        # Define workflow edges with conditions
        
        # Step 1: Download audio from YouTube (if requested)
        # This is the entry point for download requests
        
        # Step 2: Transcription follows download (if transcript requested)
        builder.add_edge(
            self.downloader_agent, 
            self.transcription_agent,
            condition=lambda msg: self._requires_transcription(msg)
        )
        
        # Step 3: Translation follows transcription (if translation requested)
        builder.add_edge(
            self.transcription_agent,
            self.translation_agent,
            condition=lambda msg: self._requires_translation(msg)
        )
        
        # Step 4: TTS follows translation (if audio generation requested)
        builder.add_edge(
            self.translation_agent,
            self.tts_agent,
            condition=lambda msg: self._requires_tts(msg)
        )
        
        # Step 5: Video merging follows TTS (if video output requested)
        builder.add_edge(
            self.tts_agent,
            self.video_merger_agent,
            condition=lambda msg: self._requires_video_merge(msg)
        )
        
        # Step 6: Notion posting can happen after various stages
        # After transcription (for transcript-only workflows)
        builder.add_edge(
            self.transcription_agent,
            self.notion_agent,
            condition=lambda msg: self._requires_notion_post(msg) and not self._requires_translation(msg)
        )
        
        # After translation (for translation-only workflows)
        builder.add_edge(
            self.translation_agent,
            self.notion_agent,
            condition=lambda msg: self._requires_notion_post(msg) and not self._requires_tts(msg)
        )
        
        # After video merging (for complete workflows)
        builder.add_edge(
            self.video_merger_agent,
            self.notion_agent,
            condition=lambda msg: self._requires_notion_post(msg)
        )
        
        # Alternative paths for direct requests
        # Direct transcription without download (if audio file provided)
        builder.add_edge(
            self.transcription_agent,
            self.transcription_agent,  # Self-loop for summarization after transcription
            condition=lambda msg: self._is_summarization_request(msg)
        )
        
        # Build the graph
        self.graph = builder.build()
        
        # Create the GraphFlow team
        self.workflow = GraphFlow(
            participants=[
                self.downloader_agent,
                self.transcription_agent,
                self.translation_agent,
                self.tts_agent,
                self.video_merger_agent,
                self.notion_agent
            ],
            graph=self.graph
        )
        
        logger.info("Workflow graph built successfully")
    
    def _requires_transcription(self, message) -> bool:
        """Check if transcription is required"""
        content = self._get_message_content(message)
        keywords = ["transcript", "transcribe", "text", "summarize", "summary"]
        return any(keyword in content.lower() for keyword in keywords)
    
    def _requires_translation(self, message) -> bool:
        """Check if translation is required"""
        content = self._get_message_content(message)
        keywords = ["translate", "translation", "language", "spanish", "french", "german"]
        return any(keyword in content.lower() for keyword in keywords)
    
    def _requires_tts(self, message) -> bool:
        """Check if TTS is required"""
        content = self._get_message_content(message)
        keywords = ["audio", "voice", "speech", "tts", "generate audio", "voice over"]
        return any(keyword in content.lower() for keyword in keywords)
    
    def _requires_video_merge(self, message) -> bool:
        """Check if video merging is required"""
        content = self._get_message_content(message)
        keywords = ["video", "merge", "combine", "final video", "translated video"]
        return any(keyword in content.lower() for keyword in keywords)
    
    def _requires_notion_post(self, message) -> bool:
        """Check if Notion posting is required"""
        content = self._get_message_content(message)
        keywords = ["notion", "save", "store", "database", "post"]
        return any(keyword in content.lower() for keyword in keywords)
    
    def _is_summarization_request(self, message) -> bool:
        """Check if this is a summarization request after transcription"""
        content = self._get_message_content(message)
        return "summarize" in content.lower() and "transcript" in content.lower()
    
    def _get_message_content(self, message) -> str:
        """Extract content from message object"""
        if hasattr(message, 'content'):
            return str(message.content)
        elif hasattr(message, 'to_model_text'):
            return message.to_model_text()
        else:
            return str(message)
    
    async def process_request(self, user_request: str) -> Dict[str, Any]:
        """Process a user request through the workflow"""
        try:
            logger.info(f"Processing request: {user_request}")
            
            # Reset file manager for new request
            self.file_manager.cleanup_temp_files()
            
            # Validate the request
            validation_result = self._validate_request(user_request)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "suggestions": validation_result.get("suggestions", [])
                }
            
            # Run the workflow
            result = await self.workflow.run(task=user_request)
            
            # Process the result
            return self._process_workflow_result(result)
            
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_request_stream(self, user_request: str):
        """Process a user request with streaming output"""
        try:
            logger.info(f"Processing request with streaming: {user_request}")
            
            # Reset file manager for new request
            self.file_manager.cleanup_temp_files()
            
            # Validate the request
            validation_result = self._validate_request(user_request)
            if not validation_result["valid"]:
                yield {
                    "type": "error",
                    "message": validation_result["error"],
                    "suggestions": validation_result.get("suggestions", [])
                }
                return
            
            # Stream the workflow execution
            async for message in self.workflow.run_stream(task=user_request):
                if hasattr(message, 'source') and hasattr(message, 'content'):
                    yield {
                        "type": "agent_message",
                        "agent": message.source,
                        "content": message.content,
                        "timestamp": getattr(message, 'created_at', None)
                    }
                elif hasattr(message, 'messages'):  # TaskResult
                    yield {
                        "type": "workflow_complete",
                        "result": self._process_workflow_result(message)
                    }
                
        except Exception as e:
            logger.error(f"Error in streaming workflow: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _validate_request(self, request: str) -> Dict[str, Any]:
        """Validate user request"""
        request_lower = request.lower()
        
        # Check for basic requirements
        has_youtube_url = any(
            domain in request_lower 
            for domain in ["youtube.com", "youtu.be"]
        )
        
        has_action = any(
            action in request_lower
            for action in [
                "download", "transcribe", "translate", "summary", 
                "audio", "video", "notion"
            ]
        )
        
        if not has_youtube_url and not has_action:
            return {
                "valid": False,
                "error": "Please provide a YouTube URL and specify what you want to do.",
                "suggestions": [
                    "Download audio from YouTube video",
                    "Transcribe YouTube video",
                    "Translate video to another language",
                    "Generate summary of video content"
                ]
            }
        
        # Check for YouTube URL validity if present
        if has_youtube_url:
            youtube_url = self._extract_youtube_url(request)
            if youtube_url and not self.validators.is_youtube_url(youtube_url):
                return {
                    "valid": False,
                    "error": "Invalid YouTube URL provided."
                }
        
        return {"valid": True}
    
    def _extract_youtube_url(self, content: str) -> Optional[str]:
        """Extract YouTube URL from content"""
        import re
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtu\.be/[A-Za-z0-9_-]+',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group()
        return None
    
    def _process_workflow_result(self, result) -> Dict[str, Any]:
        """Process workflow result and generate summary"""
        try:
            messages = getattr(result, 'messages', [])
            stop_reason = getattr(result, 'stop_reason', 'completed')
            
            # Analyze what was accomplished
            accomplishments = []
            files_generated = []
            errors = []
            
            for message in messages:
                content = self._get_message_content(message)
                source = getattr(message, 'source', 'unknown')
                
                if 'successfully' in content.lower():
                    accomplishments.append(f"{source}: {content}")
                
                if 'error' in content.lower():
                    errors.append(f"{source}: {content}")
                
                # Extract file paths
                if 'saved to:' in content.lower():
                    import re
                    file_match = re.search(r'saved to: ([^\s]+)', content)
                    if file_match:
                        files_generated.append(file_match.group(1))
            
            return {
                "success": len(errors) == 0,
                "stop_reason": stop_reason,
                "accomplishments": accomplishments,
                "files_generated": files_generated,
                "errors": errors,
                "message_count": len(messages)
            }
            
        except Exception as e:
            logger.error(f"Error processing workflow result: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current status of the orchestrator"""
        return {
            "status": "ready",
            "temp_folder": str(self.file_manager.temp_folder),
            "output_folder": str(self.file_manager.output_folder),
            "supported_languages": self.config.supported_languages,
            "agents_count": 6
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        try:
            self.file_manager.cleanup_temp_files()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    async def reset(self):
        """Reset the orchestrator state"""
        try:
            await self.workflow.reset()
            self.file_manager.cleanup_temp_files()
            logger.info("Orchestrator reset completed")
        except Exception as e:
            logger.error(f"Error during reset: {e}")

# Example usage and testing
async def main():
    """Example usage of the MasterOrchestrator"""
    
    # Initialize orchestrator
    orchestrator = MasterOrchestrator()
    
    # Example requests
    test_requests = [
        "Download audio from https://www.youtube.com/watch?v=dQw4w9WgXcQ and transcribe it",
        "Transcribe the video and translate to Spanish",
        "Generate Spanish audio and create final video",
        "Post everything to Notion database"
    ]
    
    try:
        for request in test_requests:
            print(f"\n{'='*50}")
            print(f"Processing: {request}")
            print('='*50)
            
            # Process with streaming
            async for update in orchestrator.process_request_stream(request):
                if update["type"] == "agent_message":
                    print(f"[{update['agent']}]: {update['content']}")
                elif update["type"] == "workflow_complete":
                    print(f"Workflow completed: {update['result']}")
                elif update["type"] == "error":
                    print(f"Error: {update['message']}")
            
            await asyncio.sleep(1)  # Brief pause between requests
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        await orchestrator.cleanup()

if __name__ == "__main__":
    asyncio.run(main())