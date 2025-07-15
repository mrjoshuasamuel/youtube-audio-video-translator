# agents/__init__.py - Module initialization for smart agents
"""
Smart YouTube Video Processing Agents

This module contains intelligent agents for processing YouTube videos with workflow awareness:
- YouTubeDownloaderAgent: Downloads audio/video with smart context analysis
- TranscriptionAgent: Transcribes audio with OpenAI Whisper API
- TranslationAgent: Translates text with OpenAI API and quality scoring
- TTSAgent: Generates speech with Murf API and voice mapping
- VideoMergerAgent: Merges video+audio with FFmpeg and quality control

Each agent has smart context awareness and can intelligently determine when to process
based on the complete conversation history and workflow stage.
"""

# Import all agent classes for easy access
from .youtube_downloader_agent import YouTubeDownloaderAgent
from .transcription_agent import TranscriptionAgent
from .translation_agent import TranslationAgent
from .tts_agent import TTSAgent
from .video_merger_agent import VideoMergerAgent

# Define what gets exported when using "from agents import *"
__all__ = [
    'YouTubeDownloaderAgent',
    'TranscriptionAgent', 
    'TranslationAgent',
    'TTSAgent',
    'VideoMergerAgent'
]

# Module metadata
__version__ = '1.0.0'
__author__ = 'YouTube Processing Pipeline'
__description__ = 'Smart agents for complete YouTube video translation pipeline'

# Pipeline capabilities
SUPPORTED_WORKFLOWS = [
    'complete_translation',  # Full pipeline: Download → Transcribe → Translate → TTS → Merge
    'transcription',         # Download → Transcribe
    'translation_only',      # Transcribe → Translate (requires pre-downloaded audio)
    'tts_generation',        # Translate → TTS (requires translation)
    'video_merge',           # TTS → Merge (requires TTS audio and video)
    'simple_download'        # Download only
]

SUPPORTED_LANGUAGES = [
    'German', 'French', 'Spanish', 'Italian', 'Portuguese', 'Russian',
    'Japanese', 'Korean', 'Chinese', 'Arabic', 'Hindi', 'Dutch',
    'Swedish', 'Norwegian', 'Danish', 'Polish', 'Finnish', 'Greek',
    'Hebrew', 'Turkish', 'Thai', 'Vietnamese', 'Czech', 'Hungarian'
]

# Agent capabilities summary
AGENT_CAPABILITIES = {
    'YouTubeDownloaderAgent': {
        'description': 'Smart YouTube content downloader with workflow intelligence',
        'inputs': ['YouTube URLs'],
        'outputs': ['audio.wav', 'video.mp4', 'metadata.json'],
        'features': [
            'Smart workflow prediction',
            'Context-aware file selection',
            'Quality validation',
            'Format optimization'
        ]
    },
    'TranscriptionAgent': {
        'description': 'Audio transcription with OpenAI Whisper API',
        'inputs': ['audio.wav'],
        'outputs': ['transcript.txt', 'metadata.json'],
        'features': [
            'Multi-language detection',
            'Word-level timestamps',
            'Quality scoring',
            'Error handling'
        ]
    },
    'TranslationAgent': {
        'description': 'Text translation with OpenAI API and quality scoring',
        'inputs': ['transcript.txt'],
        'outputs': ['translation.txt', 'metadata.json'],
        'features': [
            'Multi-language support',
            'Quality assessment',
            'Context preservation',
            'Cultural adaptation'
        ]
    },
    'TTSAgent': {
        'description': 'Text-to-speech with Murf API and voice mapping',
        'inputs': ['translation.txt'],
        'outputs': ['tts_audio.wav', 'metadata.json'],
        'features': [
            'Voice selection by language',
            'Quality optimization',
            'Natural speech generation',
            'Audio format conversion'
        ]
    },
    'VideoMergerAgent': {
        'description': 'Video+audio merging with FFmpeg and quality control',
        'inputs': ['video.mp4', 'tts_audio.wav'],
        'outputs': ['final_video.mp4', 'metadata.json'],
        'features': [
            'FFmpeg integration',
            'Quality preservation',
            'Format standardization',
            'Sync optimization'
        ]
    }
}

# Workflow stage definitions
WORKFLOW_STAGES = {
    'initial': 'Starting workflow',
    'download_complete': 'Files downloaded from YouTube',
    'transcription_complete': 'Audio transcribed to text',
    'translation_complete': 'Text translated to target language',
    'tts_complete': 'TTS audio generated',
    'merge_complete': 'Final video with translated audio ready'
}

def get_agent_info(agent_name: str = None):
    """Get information about available agents"""
    if agent_name:
        return AGENT_CAPABILITIES.get(agent_name, f"Agent '{agent_name}' not found")
    return AGENT_CAPABILITIES

def get_supported_workflows():
    """Get list of supported workflow types"""
    return SUPPORTED_WORKFLOWS

def get_supported_languages():
    """Get list of supported languages for translation/TTS"""
    return SUPPORTED_LANGUAGES

def get_workflow_stages():
    """Get workflow stage definitions"""
    return WORKFLOW_STAGES

# Version information
def get_version():
    """Get module version"""
    return __version__

# Usage example and quick start
QUICK_START_EXAMPLE = """
# Quick Start Example:

from agents import YouTubeDownloaderAgent, TranscriptionAgent, TranslationAgent, TTSAgent, VideoMergerAgent
from utils import Config, FileManager

# Initialize configuration and file manager
config = Config()
file_manager = FileManager()

# Create agents
downloader = YouTubeDownloaderAgent("downloader", config, file_manager)
transcriber = TranscriptionAgent("transcriber", config, file_manager)
translator = TranslationAgent("translator", config, file_manager)
tts_generator = TTSAgent("tts_generator", config, file_manager)
video_merger = VideoMergerAgent("video_merger", config, file_manager)

# Use with MasterOrchestrator for complete workflow automation
from master_orchestrator import MasterOrchestrator

orchestrator = MasterOrchestrator()
result = await orchestrator.process_request(
    "Translate https://youtube.com/watch?v=VIDEO_ID to German"
)
"""

def print_quick_start():
    """Print quick start example"""
    print(QUICK_START_EXAMPLE)