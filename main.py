# main.py
import asyncio
import sys
import argparse
from pathlib import Path
import logging
from typing import Optional

from master_orchestrator import MasterOrchestrator
from autogen_agentchat.ui import Console

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class YouTubeWorkflowApp:
    """Main application for YouTube video processing workflow"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.orchestrator = MasterOrchestrator(config_path)
        self.running = True
        
    async def run_interactive(self):
        """Run interactive mode"""
        print("=" * 60)
        print("🎥 YouTube Video Processing Workflow")
        print("=" * 60)
        print("Available operations:")
        print("1. Download audio from YouTube")
        print("2. Transcribe video content")
        print("3. Summarize content")
        print("4. Translate to other languages")
        print("5. Generate translated audio (TTS)")
        print("6. Create translated video")
        print("7. Post to Notion database")
        print("\nType 'help' for examples, 'status' for system status, 'quit' to exit")
        print("=" * 60)
        
        while self.running:
            try:
                user_input = input("\n🎬 Enter your request: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                elif user_input.lower() == 'help':
                    self._show_help()
                    continue
                elif user_input.lower() == 'status':
                    await self._show_status()
                    continue
                elif user_input.lower() == 'reset':
                    await self.orchestrator.reset()
                    print("✅ System reset completed")
                    continue
                
                print(f"\n🔄 Processing your request...")
                
                # Process the request with streaming
                async for update in self.orchestrator.process_request_stream(user_input):
                    if update["type"] == "agent_message":
                        print(f"[{update['agent']}]: {update['content'][:100]}...")
                    elif update["type"] == "workflow_complete":
                        result = update['result']
                        if result['success']:
                            print(f"\n✅ Workflow completed successfully!")
                            if result['accomplishments']:
                                print("📋 Accomplishments:")
                                for acc in result['accomplishments']:
                                    print(f"  • {acc}")
                            if result['files_generated']:
                                print("📁 Files generated:")
                                for file in result['files_generated']:
                                    print(f"  • {file}")
                        else:
                            print(f"\n❌ Workflow failed: {result.get('error', 'Unknown error')}")
                    elif update["type"] == "error":
                        print(f"\n❌ Error: {update['message']}")
                        if 'suggestions' in update:
                            print("💡 Suggestions:")
                            for suggestion in update['suggestions']:
                                print(f"  • {suggestion}")
                                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {e}")
                print(f"❌ An error occurred: {e}")
        
        await self.orchestrator.cleanup()
    
    async def run_single_command(self, command: str):
        """Run a single command"""
        try:
            print(f"🔄 Processing: {command}")
            
            result = await self.orchestrator.process_request(command)
            
            if result['success']:
                print("✅ Processing completed successfully!")
                if result.get('accomplishments'):
                    for acc in result['accomplishments']:
                        print(f"  • {acc}")
            else:
                print(f"❌ Processing failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"Error processing command: {e}")
            print(f"❌ Error: {e}")
        finally:
            await self.orchestrator.cleanup()
    
    def _show_help(self):
        """Show help information"""
        print("\n📖 Example requests:")
        print("1. 'Download audio from https://www.youtube.com/watch?v=VIDEO_ID'")
        print("2. 'Transcribe the YouTube video https://youtu.be/VIDEO_ID'")
        print("3. 'Translate the transcript to Spanish and French'")
        print("4. 'Generate Spanish audio from the translated transcript'")
        print("5. 'Create final video with Spanish audio'")
        print("6. 'Post the summary and transcript to Notion'")
        print("\n🔗 Complex workflow:")
        print("'Download audio from https://www.youtube.com/watch?v=VIDEO_ID, transcribe it, translate to Spanish, generate Spanish audio, create final video, and post to Notion'")
        print("\n📝 Supported languages: en, es, fr, de, it, pt, ru, ja, ko, zh")
    
    async def _show_status(self):
        """Show system status"""
        try:
            status = await self.orchestrator.get_status()
            print("\n📊 System Status:")
            print(f"  Status: {status['status']}")
            print(f"  Temp folder: {status['temp_folder']}")
            print(f"  Output folder: {status['output_folder']}")
            print(f"  Agents: {status['agents_count']}")
            print(f"  Supported languages: {', '.join(status['supported_languages'])}")
        except Exception as e:
            print(f"❌ Error getting status: {e}")

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YouTube Video Processing Workflow")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--command", help="Single command to execute")
    parser.add_argument("--interactive", "-i", action="store_true", 
                       help="Run in interactive mode (default)")
    
    args = parser.parse_args()
    
    # Create app instance
    app = YouTubeWorkflowApp(config_path=args.config)
    
    try:
        if args.command:
            await app.run_single_command(args.command)
        else:
            await app.run_interactive()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"❌ Application error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())