# YouTube Video Processing Agentic Workflow

A comprehensive multi-agent system built with Microsoft AutoGen for processing YouTube videos through various stages including audio download, transcription, translation, text-to-speech generation, video merging, and Notion integration.

## 🎯 Features

### Core Agents
1. **YouTube Downloader Agent** - Downloads audio from YouTube videos
2. **Transcription Agent** - Converts audio to text and generates summaries
3. **Translation Agent** - Translates content to multiple languages
4. **Text-to-Speech Agent** - Generates audio using Murf API
5. **Video Merger Agent** - Combines translated audio with original video
6. **Notion Agent** - Posts results to Notion database

### Workflow Capabilities
- ✅ Download high-quality audio from YouTube videos
- ✅ Transcribe audio to text using OpenAI Whisper
- ✅ Generate intelligent summaries of content
- ✅ Translate to 10+ languages (EN, ES, FR, DE, IT, PT, RU, JA, KO, ZH)
- ✅ Generate natural-sounding voice overs using Murf API
- ✅ Create final videos with translated audio
- ✅ Automatically post results to Notion database
- ✅ Stream processing with real-time updates
- ✅ Web interface for easy interaction

## 🚀 Quick Start

### Prerequisites
- Python 3.11+
- Docker and Docker Compose (optional)
- FFmpeg (for video processing)
- API Keys for:
  - OpenAI (for transcription and AI features)
  - Murf API (for text-to-speech)
  - Notion (for database integration)

### Installation

#### Option 1: Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd youtube-video-processing

# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys

# Run the application
python main.py --interactive
```

#### Option 2: Docker Setup
```bash
# Clone the repository
git clone <repository-url>
cd youtube-video-processing

# Copy environment template and configure
cp .env.example .env
# Edit .env with your API keys

# Build and run with Docker Compose
docker-compose up --build

# For interactive mode
docker-compose exec youtube-workflow python main.py --interactive
```

### Environment Configuration

Create a `.env` file with the following variables:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o

# Murf API Configuration
MURF_API_KEY=your_murf_api_key_here

# Notion Configuration
NOTION_API_KEY=your_notion_api_key_here
NOTION_DATABASE_ID=your_notion_database_id_here

# File Paths
TEMP_FOLDER=./temp
OUTPUT_FOLDER=./output

# Processing Settings
MAX_FILE_SIZE_MB=500
WHISPER_MODEL=base
```

## 📖 Usage Examples

### Command Line Interface
```bash
# Interactive mode
python main.py --interactive

# Single command
python main.py --command "Download audio from https://youtube.com/watch?v=VIDEO_ID and transcribe it"
```

### Web Interface
```bash
# Start web server
python web_interface.py

# Open browser to http://localhost:8000
```

### Example Workflows

#### Simple Transcription
```
Download audio from https://www.youtube.com/watch?v=dQw4w9WgXcQ and transcribe it
```

#### Translation Workflow
```
Transcribe the video and translate to Spanish and French
```

#### Complete Video Processing
```
Download audio from https://youtube.com/watch?v=VIDEO_ID, transcribe it, translate to Spanish, generate Spanish audio, create final video, and post to Notion
```

#### Summary Only
```
Transcribe https://youtube.com/watch?v=VIDEO_ID and create a summary
```

## 🏗️ Architecture

### Agent Flow
```
User Request → Master Orchestrator → Agent Selection → Processing → Results
```

### Agent Dependencies
```
YouTube Downloader → Transcription → Translation → TTS → Video Merger → Notion
                          ↓
                       Summary ────────────────────────────────────→ Notion
```

### Technology Stack
- **Framework**: Microsoft AutoGen (AgentChat)
- **AI Models**: OpenAI GPT-4, Whisper
- **TTS**: Murf AI API
- **Video Processing**: MoviePy, FFmpeg
- **Translation**: Google Translate, Deep Translator
- **Database**: Notion API
- **Web Interface**: FastAPI, WebSocket
- **Containerization**: Docker

## 🔧 Configuration

### Supported Languages
- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Italian (it)
- Portuguese (pt)
- Russian (ru)
- Japanese (ja)
- Korean (ko)
- Chinese (zh)

### File Structure
```
├── agents/
│   ├── youtube_downloader_agent.py
│   ├── transcription_agent.py
│   ├── translation_agent.py
│   ├── tts_agent.py
│   ├── video_merger_agent.py
│   └── notion_agent.py
├── utils/
│   ├── config.py
│   ├── file_manager.py
│   └── validators.py
├── main.py
├── master_orchestrator.py
├── web_interface.py
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🐳 Docker Deployment

### Building the Image
```bash
docker build -t youtube-workflow .
```

### Running with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Environment Variables in Docker
All configuration is handled through environment variables defined in `.env` file and mounted to the container.

## 🔍 Monitoring and Logging

### Log Files
- Application logs: `logs/youtube_workflow.log`
- Agent-specific logs: Individual agent logging
- Docker logs: `docker-compose logs`

### Status Monitoring
- Web interface status page: `http://localhost:8000/status`
- CLI status command: Type `status` in interactive mode

## 🛠️ Troubleshooting

### Common Issues

#### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Docker: Already included in Dockerfile
```

#### API Key Issues
- Verify all API keys are correctly set in `.env`
- Check API quotas and billing
- Ensure Notion database permissions are correct

#### File Permissions
```bash
# Fix file permissions
chmod -R 755 temp output logs
```

#### Memory Issues
- Increase Docker memory allocation
- Use smaller Whisper models (`tiny`, `base` instead of `large`)
- Process shorter videos

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Microsoft AutoGen team for the excellent framework
- OpenAI for Whisper and GPT models
- Murf AI for text-to-speech capabilities
- Notion team for the database API

## 📞 Support

For support, please:
1. Check the troubleshooting section
2. Search existing issues
3. Create a new issue with detailed information
4. Include logs and error messages

---

**Built with ❤️ using Microsoft AutoGen**