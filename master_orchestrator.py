import asyncio
import logging
from typing import Dict, Any, Optional

from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from utils import Config, FileManager, Validators
from agents import YouTubeDownloaderAgent, TranscriptionAgent, TranslationAgent, TTSAgent, VideoMergerAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MasterOrchestrator:
    """Master orchestrator with complete YouTube processing pipeline - Phase 4"""
    
    def __init__(self, config_path: Optional[str] = None, auto_cleanup: bool = False):
        # Load configuration
        self.config = Config(_env_file=config_path) if config_path else Config()
        
        # Store cleanup preference
        self.auto_cleanup = auto_cleanup
        
        # Initialize file manager
        self.file_manager = FileManager(
            temp_folder=self.config.temp_folder,
            output_folder=self.config.output_folder
        )
        
        # Initialize validators
        self.validators = Validators()
        
        # Initialize agents
        self._initialize_agents()
        
        # Build workflow
        self._build_workflow()
        
        logger.info(f"🚀 Master Orchestrator Phase 4 initialized (auto_cleanup={auto_cleanup})")
    
    def _initialize_agents(self):
        """Initialize smart agents - Phase 4: Complete Pipeline"""
        
        # Initialize YouTube Downloader Agent
        self.downloader_agent = YouTubeDownloaderAgent(
            "downloader", self.config, self.file_manager
        )
        
        # Initialize Transcription Agent
        self.transcription_agent = TranscriptionAgent(
            "transcriber", self.config, self.file_manager
        )
        
        # Initialize Translation Agent
        self.translation_agent = TranslationAgent(
            "translator", self.config, self.file_manager
        )

        # Initialize TTS Agent
        self.tts_agent = TTSAgent(
            "tts_generator", self.config, self.file_manager
        )
        
        # Initialize Video Merger Agent
        self.video_merger_agent = VideoMergerAgent(
            "video_merger", self.config, self.file_manager
        )
        
        logger.info("✅ Smart YouTube Downloader Agent initialized")
        logger.info("✅ Smart Transcription Agent initialized")
        logger.info("✅ Smart Translation Agent initialized")
        logger.info("✅ Smart TTS Agent initialized")
        logger.info("✅ Smart Video Merger Agent initialized")
        logger.info("📋 Phase 4 agents ready: Download → Transcribe → Translate → TTS → Merge → Complete")
    
    def _build_workflow(self):
        """Build intelligent workflow with all Phase 4 agents"""
        
        # Create smart termination conditions
        text_termination = TextMentionTermination("WORKFLOW_COMPLETE")
        max_message_termination = MaxMessageTermination(max_messages=20)  # Increased for 5 agents
        termination_condition = text_termination | max_message_termination
        
        # Create workflow with smart agents - Phase 4 Setup (Complete Pipeline)
        self.workflow = RoundRobinGroupChat(
            participants=[
                self.downloader_agent,
                self.transcription_agent,
                self.translation_agent,
                self.tts_agent,
                self.video_merger_agent
            ],
            termination_condition=termination_condition,
            max_turns=25  # Increased for 5-agent workflow
        )
        
        logger.info("✅ Phase 4 workflow built: Complete translation pipeline")
        logger.info("📋 Full workflow: YouTubeDownloader → Transcription → Translation → TTS → VideoMerger")
        logger.info("🎬 Supports complete video translation with merged output")
    
    async def process_request(self, user_request: str) -> Dict[str, Any]:
        """Process user request with complete 5-agent translation pipeline"""
        try:
            logger.info(f"📥 Processing request: {user_request}")
            
            # Cleanup if enabled
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("🧹 Cleaned up temp files")
            
            # Enhanced validation
            validation_result = self._validate_request_smart(user_request)
            if not validation_result["valid"]:
                return {
                    "success": False,
                    "error": validation_result["error"],
                    "suggestions": validation_result.get("suggestions", [])
                }
            
            # Run complete workflow
            result = await self.workflow.run(task=user_request)
            
            # Process results
            return self._process_workflow_result(result)
            
        except Exception as e:
            logger.error(f"❌ Error processing request: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def process_request_stream(self, user_request: str):
        """Process with streaming and complete pipeline feedback"""
        try:
            logger.info(f"🎯 Streaming request: {user_request}")
            
            # Cleanup if enabled
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("🧹 Cleaned up temp files")
            
            # Smart validation
            validation_result = self._validate_request_smart(user_request)
            if not validation_result["valid"]:
                yield {
                    "type": "error",
                    "message": validation_result["error"],
                    "suggestions": validation_result.get("suggestions", [])
                }
                return
            
            # Yield validation success
            yield {
                "type": "validation_success",
                "message": f"✅ Request validated: {validation_result['workflow_type']} workflow detected"
            }
            
            # Extract and display detected info
            detected_info = self._extract_request_info(user_request)
            if detected_info:
                yield {
                    "type": "request_analysis",
                    "message": f"🎯 Analysis: {detected_info}"
                }
            
            # Show complete workflow plan
            workflow_plan = self._generate_workflow_plan(validation_result['workflow_type'])
            yield {
                "type": "workflow_plan",
                "message": f"📋 Complete Pipeline: {workflow_plan}"
            }
            
            # Stream workflow execution
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
            logger.error(f"❌ Error in streaming workflow: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _validate_request_smart(self, request: str) -> Dict[str, Any]:
        """🧠 Smart validation with complete workflow detection"""
        request_lower = request.lower()
        
        # Check for YouTube URL
        youtube_url = self._extract_youtube_url(request)
        if not youtube_url:
            return {
                "valid": False,
                "error": "No YouTube URL found in request.",
                "suggestions": [
                    "Include a valid YouTube URL (youtube.com or youtu.be)",
                    "Example: Download from https://www.youtube.com/watch?v=VIDEO_ID",
                    "Example: Transcribe https://youtu.be/VIDEO_ID",
                    "Example: Translate https://youtu.be/VIDEO_ID to German",
                    "Example: Create German version of https://youtu.be/VIDEO_ID"
                ]
            }
        
        # Validate URL format
        if not self.validators.is_youtube_url(youtube_url):
            return {
                "valid": False,
                "error": f"Invalid YouTube URL format: {youtube_url}"
            }
        
        # Detect workflow type
        workflow_type = self._detect_workflow_type(request_lower)
        
        # Validate workflow requirements
        if workflow_type == 'unknown':
            return {
                "valid": False,
                "error": "Please specify what you want to do with the video.",
                "suggestions": [
                    "Download audio/video from YouTube",
                    "Transcribe YouTube video to text",
                    "Translate YouTube video to another language (complete pipeline)",
                    "Create dubbed version of YouTube video",
                    "Generate summary of YouTube video"
                ]
            }
        
        # Phase 4 validation: Check for translation requirements
        if workflow_type == 'translation':
            target_language = self._extract_target_language(request_lower)
            if not target_language:
                return {
                    "valid": False,
                    "error": "Please specify the target language for translation.",
                    "suggestions": [
                        "Example: Translate to German",
                        "Example: Convert to French",
                        "Example: Create Spanish version",
                        "Example: Dub to Italian",
                        "Supported languages: German, French, Spanish, Italian, Portuguese, Russian, Japanese, Korean, Chinese, Arabic, Hindi, Dutch, Swedish, Norwegian, Danish, Polish, Finnish, Greek, Hebrew, Turkish, Thai, Vietnamese, Czech, Hungarian"
                    ]
                }
        
        return {
            "valid": True,
            "workflow_type": workflow_type,
            "youtube_url": youtube_url
        }
    
    def _detect_workflow_type(self, request_lower: str) -> str:
        """🔍 Detect workflow type from request - Phase 4 enhanced"""
        
        # Complete translation workflow (Phase 4 priority)
        if any(word in request_lower for word in [
            'translate', 'translation', 'dub', 'dubbed', 'convert to',
            'german', 'french', 'spanish', 'italian', 'portuguese', 'russian', 
            'japanese', 'korean', 'chinese', 'arabic', 'hindi', 'dutch',
            'create', 'make', 'generate'
        ]):
            return 'translation'
        
        # Video merging workflow
        if any(word in request_lower for word in ['merge', 'combine', 'join', 'mix', 'overlay']):
            return 'merge'
        
        # TTS workflow
        if any(word in request_lower for word in ['tts', 'text to speech', 'voice', 'speech', 'audio generation']):
            return 'tts'
        
        # Transcription workflow
        if any(word in request_lower for word in ['transcribe', 'transcript', 'text', 'speech to text']):
            return 'transcription'
        
        # Summary workflow
        if any(word in request_lower for word in ['summarize', 'summary', 'brief', 'overview']):
            return 'summary'
        
        # Simple download
        if any(word in request_lower for word in ['download', 'get', 'fetch', 'save']):
            if 'video' in request_lower:
                return 'video_download'
            elif 'audio' in request_lower:
                return 'audio_download'
            else:
                return 'simple_download'
        
        return 'unknown'
    
    def _extract_target_language(self, request_lower: str) -> Optional[str]:
        """🌍 Extract target language from request - Phase 4 enhanced"""
        
        # Enhanced language detection mappings
        language_indicators = {
            'german': ['german', 'deutsch', 'de', 'germany'],
            'french': ['french', 'français', 'francais', 'fr', 'france'],
            'spanish': ['spanish', 'español', 'espanol', 'es', 'spain'],
            'italian': ['italian', 'italiano', 'it', 'italy'],
            'portuguese': ['portuguese', 'português', 'portugues', 'pt', 'portugal', 'brazil'],
            'russian': ['russian', 'русский', 'ru', 'russia'],
            'japanese': ['japanese', '日本語', 'ja', 'japan'],
            'korean': ['korean', '한국어', 'ko', 'korea'],
            'chinese': ['chinese', 'mandarin', '中文', 'zh', 'china'],
            'arabic': ['arabic', 'العربية', 'ar'],
            'hindi': ['hindi', 'हिन्दी', 'hi', 'india'],
            'dutch': ['dutch', 'nederlands', 'nl', 'netherlands'],
            'swedish': ['swedish', 'svenska', 'sv', 'sweden'],
            'norwegian': ['norwegian', 'norsk', 'no', 'norway'],
            'danish': ['danish', 'dansk', 'da', 'denmark'],
            'polish': ['polish', 'polski', 'pl', 'poland'],
            'finnish': ['finnish', 'suomi', 'fi', 'finland'],
            'greek': ['greek', 'ελληνικά', 'el', 'greece'],
            'hebrew': ['hebrew', 'עברית', 'he', 'israel'],
            'turkish': ['turkish', 'türkçe', 'tr', 'turkey'],
            'thai': ['thai', 'ไทย', 'th', 'thailand'],
            'vietnamese': ['vietnamese', 'tiếng việt', 'vi', 'vietnam'],
            'czech': ['czech', 'čeština', 'cs', 'czech republic'],
            'hungarian': ['hungarian', 'magyar', 'hu', 'hungary']
        }
        
        # Look for "to [language]" patterns
        import re
        to_language_patterns = [
            r'to\s+(\w+)',
            r'in\s+(\w+)',
            r'convert\s+to\s+(\w+)',
            r'translate.*to\s+(\w+)',
            r'dub.*to\s+(\w+)',
            r'create.*(\w+)\s+version',
            r'make.*(\w+)\s+version'
        ]
        
        for pattern in to_language_patterns:
            match = re.search(pattern, request_lower)
            if match:
                potential_lang = match.group(1).lower()
                for lang, indicators in language_indicators.items():
                    if potential_lang in indicators:
                        return lang
        
        # Direct language detection
        for lang, indicators in language_indicators.items():
            if any(indicator in request_lower for indicator in indicators):
                return lang
        
        return None
    
    def _generate_workflow_plan(self, workflow_type: str) -> str:
        """📋 Generate complete workflow execution plan - Phase 4"""
        
        plans = {
            'translation': "Download Audio+Video → Transcribe → Translate → Generate TTS → Merge Video → Complete",
            'transcription': "Download Audio → Transcribe → Complete",
            'summary': "Download Audio → Transcribe → [Summary (Future)] → Complete",
            'tts': "Download Audio → Transcribe → Translate → Generate TTS → Complete",
            'merge': "Download Video → Generate/Load TTS → Merge Video → Complete",
            'video_download': "Download Video → Complete",
            'audio_download': "Download Audio → Complete",
            'simple_download': "Download Audio → Complete"
        }
        
        return plans.get(workflow_type, "Download → Process → Complete")
    
    def _extract_request_info(self, request: str) -> str:
        """📋 Extract and format request information - Phase 4 enhanced"""
        info_parts = []
        
        # YouTube URL
        youtube_url = self._extract_youtube_url(request)
        if youtube_url:
            video_id = self.validators.extract_video_id(youtube_url)
            info_parts.append(f"Video ID: {video_id}")
        
        # Workflow type
        workflow_type = self._detect_workflow_type(request.lower())
        info_parts.append(f"Workflow: {workflow_type}")
        
        # Target language (for translation)
        if workflow_type == 'translation':
            target_language = self._extract_target_language(request.lower())
            if target_language:
                info_parts.append(f"Target: {target_language.title()}")
        
        # Expected outputs
        expected_outputs = self._get_expected_outputs(workflow_type)
        if expected_outputs:
            info_parts.append(f"Outputs: {expected_outputs}")
        
        # Pipeline stage
        info_parts.append("Pipeline: Complete (5 agents)")
        
        return " | ".join(info_parts)
    
    def _get_expected_outputs(self, workflow_type: str) -> str:
        """📁 Get expected output files for workflow type - Phase 4"""
        
        outputs = {
            'translation': "transcript.txt, translation.txt, tts_audio.wav, final_video.mp4",
            'transcription': "transcript.txt",
            'summary': "transcript.txt, summary.txt (future)",
            'tts': "transcript.txt, translation.txt, tts_audio.wav",
            'merge': "final_video.mp4",
            'video_download': "video.mp4",
            'audio_download': "audio.wav",
            'simple_download': "audio.wav"
        }
        
        return outputs.get(workflow_type, "processed_files")
    
    def _extract_youtube_url(self, content: str) -> Optional[str]:
        """🔗 Extract YouTube URL from content"""
        import re
        patterns = [
            r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtu\.be/[A-Za-z0-9_-]+',
            r'https?://(?:www\.)?youtube\.com/embed/[A-Za-z0-9_-]+'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                return match.group()
        return None
    
    def _process_workflow_result(self, result) -> Dict[str, Any]:
        """📊 Process and analyze complete workflow results - Phase 4"""
        try:
            messages = getattr(result, 'messages', [])
            stop_reason = getattr(result, 'stop_reason', 'completed')
            
            # Analyze accomplishments and errors
            accomplishments = []
            files_generated = []
            errors = []
            agent_summary = {}
            
            for message in messages:
                content = str(getattr(message, 'content', ''))
                source = getattr(message, 'source', 'unknown')
                
                # Track per-agent accomplishments
                if source not in agent_summary:
                    agent_summary[source] = {"messages": 0, "successes": 0, "errors": 0}
                
                agent_summary[source]["messages"] += 1
                
                # Track successes
                if any(indicator in content.lower() for indicator in ['successfully', '✅', 'completed']):
                    accomplishments.append(f"{source}: {content}")
                    agent_summary[source]["successes"] += 1
                
                # Track errors
                if any(indicator in content.lower() for indicator in ['error', 'failed', '❌']):
                    errors.append(f"{source}: {content}")
                    agent_summary[source]["errors"] += 1
                
                # Extract file paths (enhanced for Phase 4)
                import re
                file_patterns = [
                    r'Files?: ([^.]+\.(?:wav|mp4|txt|json))',
                    r'saved to: ([^\s]+)',
                    r'File: ([^\s]+)',
                    r'Transcript saved to ([^\s]+)',
                    r'Translation saved to ([^\s]+)',
                    r'Audio saved to ([^\s]+)',
                    r'Final video saved to ([^\s]+)',
                    r'Merged.*saved to ([^\s]+)',
                    r'output_path[:\s]+([^\s]+)',
                ]
                
                for pattern in file_patterns:
                    matches = re.findall(pattern, content)
                    files_generated.extend(matches)
            
            # Determine overall success
            success = len(errors) == 0 and len(accomplishments) > 0
            
            # Analyze complete workflow completion
            workflow_completed = any("WORKFLOW_COMPLETE" in str(getattr(msg, 'content', '')) 
                                   for msg in messages)
            
            # Check if complete translation pipeline was executed
            complete_pipeline = self._check_complete_pipeline_execution(messages)
            
            return {
                "success": success,
                "workflow_completed": workflow_completed,
                "complete_pipeline_executed": complete_pipeline,
                "stop_reason": stop_reason,
                "accomplishments": accomplishments,
                "files_generated": list(set(files_generated)),  # Remove duplicates
                "errors": errors,
                "message_count": len(messages),
                "agent_summary": agent_summary,
                "workflow_summary": self._generate_workflow_summary(messages),
                "phase": "Phase 4: Complete Translation Pipeline",
                "pipeline_stages": self._analyze_pipeline_stages(messages)
            }
            
        except Exception as e:
            logger.error(f"❌ Error processing workflow result: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _check_complete_pipeline_execution(self, messages) -> bool:
        """🔍 Check if complete translation pipeline was executed"""
        full_conversation = " ".join([str(getattr(msg, 'content', '')) for msg in messages])
        conversation_lower = full_conversation.lower()
        
        # Check for all pipeline stages
        stages = {
            'download': any(indicator in conversation_lower for indicator in [
                'downloaded', 'download complete', 'files:'
            ]),
            'transcription': any(indicator in conversation_lower for indicator in [
                'transcribed', 'transcript saved'
            ]),
            'translation': any(indicator in conversation_lower for indicator in [
                'translated', 'translation saved'
            ]),
            'tts': any(indicator in conversation_lower for indicator in [
                'generated tts', 'tts audio', 'audio generated'
            ]),
            'merge': any(indicator in conversation_lower for indicator in [
                'merged', 'final video', 'video saved'
            ])
        }
        
        # Complete pipeline requires all stages
        return all(stages.values())
    
    def _analyze_pipeline_stages(self, messages) -> Dict[str, bool]:
        """📊 Analyze which pipeline stages were completed"""
        full_conversation = " ".join([str(getattr(msg, 'content', '')) for msg in messages])
        conversation_lower = full_conversation.lower()
        
        return {
            'download_completed': any(indicator in conversation_lower for indicator in [
                'downloaded', 'download complete', 'files:'
            ]),
            'transcription_completed': any(indicator in conversation_lower for indicator in [
                'transcribed', 'transcript saved'
            ]),
            'translation_completed': any(indicator in conversation_lower for indicator in [
                'translated', 'translation saved'
            ]),
            'tts_completed': any(indicator in conversation_lower for indicator in [
                'generated tts', 'tts audio', 'audio generated'
            ]),
            'merge_completed': any(indicator in conversation_lower for indicator in [
                'merged', 'final video', 'video saved'
            ])
        }
    
    def _generate_workflow_summary(self, messages) -> str:
        """📝 Generate human-readable workflow summary - Phase 4"""
        if not messages:
            return "No workflow executed"
        
        # Extract key information
        youtube_urls = []
        file_types = []
        languages = []
        pipeline_stages = []
        
        full_conversation = " ".join([str(getattr(msg, 'content', '')) for msg in messages])
        
        # Find YouTube URLs
        import re
        youtube_matches = re.findall(r'https?://(?:www\.)?youtube\.com/watch\?v=[A-Za-z0-9_-]+', full_conversation)
        youtube_urls.extend(youtube_matches)
        
        # Analyze pipeline progression
        stages = self._analyze_pipeline_stages(messages)
        if stages['download_completed']:
            pipeline_stages.append('Downloaded')
        if stages['transcription_completed']:
            pipeline_stages.append('Transcribed')
        if stages['translation_completed']:
            pipeline_stages.append('Translated')
        if stages['tts_completed']:
            pipeline_stages.append('TTS Generated')
        if stages['merge_completed']:
            pipeline_stages.append('Video Merged')
        
        # Find file types
        if 'audio' in full_conversation.lower():
            file_types.append('audio')
        if 'video' in full_conversation.lower():
            file_types.append('video')
        if 'transcript' in full_conversation.lower():
            file_types.append('transcript')
        if 'translation' in full_conversation.lower():
            file_types.append('translation')
        if 'final video' in full_conversation.lower() or 'merged' in full_conversation.lower():
            file_types.append('final_video')
        
        # Find languages
        languages_found = []
        for lang in ['german', 'french', 'spanish', 'italian', 'portuguese', 'russian', 
                    'japanese', 'korean', 'chinese', 'arabic', 'hindi', 'dutch']:
            if lang in full_conversation.lower():
                languages_found.append(lang.title())
        
        # Build summary
        summary_parts = []
        if youtube_urls:
            summary_parts.append(f"Processed YouTube video")
        if pipeline_stages:
            summary_parts.append(f"Pipeline: {' → '.join(pipeline_stages)}")
        if file_types:
            summary_parts.append(f"Generated: {', '.join(file_types)}")
        if languages_found:
            summary_parts.append(f"Languages: {', '.join(languages_found)}")
        
        # Add completion status
        if len(pipeline_stages) >= 4:  # Most stages completed
            summary_parts.append("Complete Pipeline Executed")
        elif len(pipeline_stages) >= 2:
            summary_parts.append("Partial Pipeline Executed")
        
        return " | ".join(summary_parts) if summary_parts else "Phase 4 workflow completed"
    
    async def get_status(self) -> Dict[str, Any]:
        """📊 Get orchestrator status - Phase 4"""
        return {
            "status": "ready",
            "temp_folder": str(self.file_manager.temp_folder),
            "output_folder": str(self.file_manager.output_folder),
            "agents_count": 5,
            "auto_cleanup": self.auto_cleanup,
            "available_agents": [
                "smart_downloader", "smart_transcriber", "smart_translator", 
                "smart_tts_generator", "smart_video_merger"
            ],
            "supported_workflows": [
                "complete_translation", "transcription", "summary", "tts", 
                "video_merge", "video_download", "audio_download"
            ],
            "workflow_phase": "Phase 4: Complete Translation Pipeline",
            "next_phase": "Phase 5: Add NotionAgent (Future)",
            "pipeline_capabilities": [
                "YouTube video download (audio + video)",
                "Audio transcription with OpenAI Whisper",
                "Text translation with OpenAI API",
                "TTS generation with Murf API",
                "Video merging with FFmpeg",
                "Complete dubbed video output"
            ],
            "agent_capabilities": {
                "downloader": "YouTube audio/video download with workflow intelligence",
                "transcriber": "Audio transcription with OpenAI Whisper API",
                "translator": "Text translation with OpenAI API and quality scoring",
                "tts_generator": "Text-to-speech with Murf API and voice mapping",
                "video_merger": "Video+audio merging with FFmpeg and quality control"
            },
            "supported_languages": [
                "German", "French", "Spanish", "Italian", "Portuguese", "Russian",
                "Japanese", "Korean", "Chinese", "Arabic", "Hindi", "Dutch",
                "Swedish", "Norwegian", "Danish", "Polish", "Finnish", "Greek",
                "Hebrew", "Turkish", "Thai", "Vietnamese", "Czech", "Hungarian"
            ],
            "output_formats": {
                "transcripts": ".txt",
                "translations": ".txt",
                "tts_audio": ".wav",
                "final_video": ".mp4"
            },
            "quality_features": [
                "Smart context awareness across agents",
                "File validation and error handling",
                "Quality scoring for translations and TTS",
                "Automatic workflow stage detection",
                "FFmpeg video processing with optimization"
            ]
        }
    
    async def manual_cleanup(self):
        """🧹 Manual cleanup"""
        try:
            self.file_manager.cleanup_temp_files()
            logger.info("✅ Manual cleanup completed")
        except Exception as e:
            logger.error(f"❌ Error during manual cleanup: {e}")
    
    async def cleanup(self):
        """🔄 Smart cleanup"""
        try:
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("✅ Auto-cleanup completed")
            else:
                logger.info("📁 Files preserved (auto_cleanup=False)")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")
    
    async def reset(self):
        """🔄 Reset orchestrator"""
        try:
            await self.workflow.reset()
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("✅ Orchestrator reset with cleanup")
            else:
                logger.info("🔄 Orchestrator reset (files preserved)")
        except Exception as e:
            logger.error(f"❌ Error during reset: {e}")


class SimpleSequentialProcessor:
    """Simple sequential processor with complete 5-agent pipeline"""
    
    def __init__(self, config_path: Optional[str] = None, auto_cleanup: bool = False):
        self.config = Config(_env_file=config_path) if config_path else Config()
        self.auto_cleanup = auto_cleanup
        
        self.file_manager = FileManager(
            temp_folder=self.config.temp_folder,
            output_folder=self.config.output_folder
        )
        
        self.validators = Validators()
        self._initialize_agents()
        
        logger.info(f"✅ Simple Sequential Processor Phase 4 initialized (auto_cleanup={auto_cleanup})")
    
    def _initialize_agents(self):
        """Initialize complete agent pipeline"""
        self.agents = {
            "downloader": YouTubeDownloaderAgent("downloader", self.config, self.file_manager),
            "transcriber": TranscriptionAgent("transcriber", self.config, self.file_manager),
            "translator": TranslationAgent("translator", self.config, self.file_manager),
            "tts_generator": TTSAgent("tts_generator", self.config, self.file_manager),
            "video_merger": VideoMergerAgent("video_merger", self.config, self.file_manager)
        }
        logger.info("✅ Complete agent pipeline initialized - Phase 4: Full Translation Pipeline")
    
    async def process_request_stream(self, user_request: str):
        """Process with complete sequential execution"""
        try:
            logger.info(f"🎯 Sequential processing: {user_request}")
            
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("🧹 Cleaned up temp files")
            
            # Smart agent sequence for complete pipeline
            agent_sequence = self._determine_complete_sequence(user_request)
            
            current_context = user_request
            results = []
            
            for agent_name in agent_sequence:
                agent = self.agents[agent_name]
                logger.info(f"🤖 Running agent: {agent_name}")
                
                try:
                    from autogen_agentchat.messages import TextMessage
                    from autogen_core import CancellationToken
                    
                    messages = [TextMessage(content=current_context, source="user")]
                    response = await agent.on_messages(messages, CancellationToken())
                    result_content = response.chat_message.content
                    
                    results.append({
                        "agent": agent_name,
                        "content": result_content,
                        "success": True
                    })
                    
                    yield {
                        "type": "agent_message",
                        "agent": agent_name,
                        "content": result_content
                    }
                    
                    # Update context with previous agent results
                    current_context = f"{current_context}\n{result_content}"
                    
                except Exception as e:
                    error_msg = f"Error in {agent_name}: {str(e)}"
                    logger.error(error_msg)
                    results.append({
                        "agent": agent_name,
                        "content": error_msg,
                        "success": False
                    })
                    
                    yield {
                        "type": "error",
                        "agent": agent_name,
                        "message": error_msg
                    }
                    break
            
            # Final result
            success = all(r["success"] for r in results)
            yield {
                "type": "workflow_complete",
                "result": {
                    "success": success,
                    "accomplishments": [f"{r['agent']}: {r['content']}" for r in results if r["success"]],
                    "errors": [f"{r['agent']}: {r['content']}" for r in results if not r["success"]],
                    "message_count": len(results),
                    "phase": "Phase 4: Complete Translation Pipeline",
                    "agents_executed": len(results)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Error in sequential processing: {e}")
            yield {
                "type": "error",
                "message": str(e)
            }
    
    def _determine_complete_sequence(self, request: str) -> list:
        """🧠 Smart agent sequence determination for complete pipeline"""
        request_lower = request.lower()
        
        # Check for YouTube URL
        if any(domain in request_lower for domain in ["youtube.com", "youtu.be"]):
            # Check for complete translation workflow
            if any(word in request_lower for word in [
                "translate", "translation", "dub", "dubbed", "convert to",
                "german", "french", "spanish", "italian", "portuguese", "russian",
                "japanese", "korean", "chinese", "arabic", "hindi", "dutch"
            ]):
                return ["downloader", "transcriber", "translator", "tts_generator", "video_merger"]
            # Check for video merge workflow
            elif any(word in request_lower for word in ["merge", "combine", "join"]):
                return ["downloader", "tts_generator", "video_merger"]
            # Check for TTS workflow
            elif any(word in request_lower for word in ["tts", "text to speech", "voice"]):
                return ["downloader", "transcriber", "translator", "tts_generator"]
            # Check for transcription workflow
            elif any(word in request_lower for word in ["transcribe", "transcript", "text"]):
                return ["downloader", "transcriber"]
            else:
                return ["downloader"]
        
        return []
    
    async def cleanup(self):
        """🔄 Cleanup resources"""
        try:
            if self.auto_cleanup:
                self.file_manager.cleanup_temp_files()
                logger.info("✅ Cleanup completed")
            else:
                logger.info("📁 Files preserved")
        except Exception as e:
            logger.error(f"❌ Error during cleanup: {e}")


# 🚀 EXAMPLE USAGE AND TESTING
async def main():
    """Complete example with Phase 4 orchestrator including complete translation pipeline"""
    
    try:
        # Initialize complete orchestrator
        logger.info("🚀 Initializing Complete Orchestrator Phase 4 with VideoMergerAgent...")
        orchestrator = MasterOrchestrator(auto_cleanup=False)  # Preserve files
        processor_type = "MasterOrchestrator"
    except Exception as e:
        logger.warning(f"⚠️  MasterOrchestrator failed: {e}")
        logger.info("🔄 Falling back to SimpleSequentialProcessor...")
        orchestrator = SimpleSequentialProcessor(auto_cleanup=False)
        processor_type = "SimpleSequentialProcessor"
    
    # Test cases for Phase 4 - Complete Translation Pipeline
    test_requests = [
        "from https://www.youtube.com/watch?v=PWzRFGXIR10 and translate the audio and video to German",
        "Create a French version of https://www.youtube.com/watch?v=PWzRFGXIR10",
        "Dub https://www.youtube.com/watch?v=PWzRFGXIR10 to Spanish",
        "Make an Italian dubbed version of https://www.youtube.com/watch?v=PWzRFGXIR10"
    ]
    
    for i, test_request in enumerate(test_requests, 1):
        try:
            print(f"\n{'='*80}")
            print(f"🧪 TEST {i}: {processor_type} - Phase 4 Complete Pipeline")
            print(f"📝 Request: {test_request}")
            print('='*80)
            
            # Process with streaming
            async for update in orchestrator.process_request_stream(test_request):
                if update["type"] == "validation_success":
                    print(f"\n✅ {update['message']}")
                elif update["type"] == "request_analysis":
                    print(f"🔍 {update['message']}")
                elif update["type"] == "workflow_plan":
                    print(f"📋 {update['message']}")
                elif update["type"] == "agent_message":
                    print(f"\n[{update['agent']}]:")
                    print(f"{update['content']}")
                elif update["type"] == "workflow_complete":
                    result = update["result"]
                    print(f"\n{'='*60}")
                    print(f"🎉 Phase 4 Complete Pipeline completed!")
                    print(f"✅ Success: {result['success']}")
                    if result.get('workflow_summary'):
                        print(f"📋 Summary: {result['workflow_summary']}")
                    if result.get('files_generated'):
                        print(f"📁 Files: {result['files_generated']}")
                    if result.get('complete_pipeline_executed'):
                        print(f"🔄 Complete Pipeline: {result['complete_pipeline_executed']}")
                    if result.get('pipeline_stages'):
                        print(f"📊 Stages: {result['pipeline_stages']}")
                    if result.get('agent_summary'):
                        print(f"🤖 Agent Performance: {result['agent_summary']}")
                    if result.get('errors'):
                        print("❌ Errors:")
                        for err in result['errors']:
                            print(f"  • {err}")
                elif update["type"] == "error":
                    print(f"\n❌ Error: {update['message']}")
            
            # Wait between tests
            if i < len(test_requests):
                print("\n⏳ Waiting 3 seconds before next test...")
                await asyncio.sleep(3)
                
        except KeyboardInterrupt:
            print("\n⏹️  Test interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Test error: {e}")
    
    # Show final status
    try:
        if hasattr(orchestrator, 'get_status'):
            status = await orchestrator.get_status()
            print(f"\n📊 Final Status:")
            print(f"   Phase: {status.get('workflow_phase', 'Unknown')}")
            print(f"   Next Phase: {status.get('next_phase', 'Unknown')}")
            print(f"   Agents: {status['available_agents']}")
            print(f"   Pipeline Capabilities: {len(status.get('pipeline_capabilities', []))} features")
            print(f"   Supported Languages: {len(status.get('supported_languages', []))} languages")
            print(f"   Output Formats: {status.get('output_formats', {})}")
            print(f"   Temp folder: {status['temp_folder']}")
            print(f"   Output folder: {status['output_folder']}")
            print(f"   Workflows: {status['supported_workflows']}")
        
        # Ask user about cleanup
        cleanup_choice = input("\n🗑️  Cleanup temp files? (y/N): ").lower().strip()
        if cleanup_choice == 'y':
            if hasattr(orchestrator, 'manual_cleanup'):
                await orchestrator.manual_cleanup()
            print("✅ Files cleaned up")
        else:
            print("📁 Files preserved!")
            
    except Exception as e:
        print(f"❌ Status error: {e}")
    finally:
        print("\n🏁 Phase 4 complete translation pipeline testing complete!")
        print("🎬 Final output: Complete dubbed videos with translated audio!")
        print("🔮 Next: Phase 5 - Add NotionAgent for result upload")

if __name__ == "__main__":
    asyncio.run(main())