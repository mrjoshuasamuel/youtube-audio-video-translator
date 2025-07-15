# web_interface.py
import asyncio
import json
import logging
from typing import Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

from master_orchestrator import MasterOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="YouTube Video Processing Workflow", version="1.0.0")

# Initialize orchestrator
orchestrator = MasterOrchestrator()

class ProcessRequest(BaseModel):
    request: str
    stream: bool = True

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

manager = ConnectionManager()

# Static HTML interface
HTML_INTERFACE = """
<!DOCTYPE html>
<html>
<head>
    <title>YouTube Video Processing Workflow</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .header { text-align: center; color: #333; margin-bottom: 30px; }
        .input-section { margin-bottom: 20px; }
        .input-section input { width: 70%; padding: 12px; border: 1px solid #ddd; border-radius: 4px; }
        .input-section button { width: 25%; padding: 12px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; margin-left: 10px; }
        .input-section button:hover { background: #0056b3; }
        .output { height: 400px; border: 1px solid #ddd; padding: 15px; overflow-y: auto; background: #f8f9fa; border-radius: 4px; font-family: monospace; }
        .message { margin: 5px 0; padding: 8px; border-radius: 4px; }
        .user-message { background: #e3f2fd; border-left: 4px solid #2196f3; }
        .agent-message { background: #f3e5f5; border-left: 4px solid #9c27b0; }
        .success-message { background: #e8f5e8; border-left: 4px solid #4caf50; }
        .error-message { background: #ffebee; border-left: 4px solid #f44336; }
        .examples { margin-top: 20px; }
        .examples h3 { color: #555; }
        .examples ul { list-style-type: none; padding: 0; }
        .examples li { margin: 8px 0; padding: 10px; background: #f0f0f0; border-radius: 4px; cursor: pointer; }
        .examples li:hover { background: #e0e0e0; }
        .status { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 4px; }
    </style>
</head>
<body>
    <div class="container">
        <h1 class="header">🎥 YouTube Video Processing Workflow</h1>
        
        <div class="input-section">
            <input type="text" id="userInput" placeholder="Enter your request (e.g., 'Download audio from https://youtube.com/watch?v=...')" />
            <button onclick="sendMessage()">Process</button>
        </div>
        
        <div id="output" class="output"></div>
        
        <div class="examples">
            <h3>📖 Example Requests:</h3>
            <ul>
                <li onclick="fillInput(this.textContent)">Download audio from https://www.youtube.com/watch?v=dQw4w9WgXcQ</li>
                <li onclick="fillInput(this.textContent)">Transcribe the YouTube video and create a summary</li>
                <li onclick="fillInput(this.textContent)">Translate the transcript to Spanish and French</li>
                <li onclick="fillInput(this.textContent)">Generate Spanish audio from translated text</li>
                <li onclick="fillInput(this.textContent)">Create final video with Spanish audio</li>
                <li onclick="fillInput(this.textContent)">Post everything to Notion database</li>
            </ul>
        </div>
        
        <div class="status">
            <h3>🔧 System Status</h3>
            <p id="statusInfo">Loading...</p>
            <button onclick="getStatus()">Refresh Status</button>
        </div>
    </div>

    <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        const output = document.getElementById('output');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            displayMessage(data);
        };
        
        function displayMessage(data) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message';
            
            const timestamp = new Date().toLocaleTimeString();
            
            if (data.type === 'user_message') {
                messageDiv.className += ' user-message';
                messageDiv.innerHTML = `<strong>[${timestamp}] User:</strong> ${data.content}`;
            } else if (data.type === 'agent_message') {
                messageDiv.className += ' agent-message';
                messageDiv.innerHTML = `<strong>[${timestamp}] ${data.agent}:</strong> ${data.content}`;
            } else if (data.type === 'workflow_complete') {
                messageDiv.className += ' success-message';
                const result = data.result;
                let content = `<strong>[${timestamp}] Workflow Complete:</strong><br>`;
                if (result.success) {
                    content += '✅ Success!<br>';
                    if (result.accomplishments) {
                        content += '<strong>Accomplishments:</strong><br>';
                        result.accomplishments.forEach(acc => content += `• ${acc}<br>`);
                    }
                    if (result.files_generated) {
                        content += '<strong>Files Generated:</strong><br>';
                        result.files_generated.forEach(file => content += `• ${file}<br>`);
                    }
                } else {
                    content += `❌ Failed: ${result.error}`;
                }
                messageDiv.innerHTML = content;
            } else if (data.type === 'error') {
                messageDiv.className += ' error-message';
                messageDiv.innerHTML = `<strong>[${timestamp}] Error:</strong> ${data.message}`;
            }
            
            output.appendChild(messageDiv);
            output.scrollTop = output.scrollHeight;
        }
        
        function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (message) {
                displayMessage({type: 'user_message', content: message});
                ws.send(JSON.stringify({request: message}));
                input.value = '';
            }
        }
        
        function fillInput(text) {
            document.getElementById('userInput').value = text;
        }
        
        function getStatus() {
            fetch('/status')
                .then(response => response.json())
                .then(data => {
                    const statusInfo = document.getElementById('statusInfo');
                    statusInfo.innerHTML = `
                        Status: ${data.status}<br>
                        Agents: ${data.agents_count}<br>
                        Supported Languages: ${data.supported_languages.join(', ')}
                    `;
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        // Allow Enter key to send message
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Load status on page load
        getStatus();
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_interface():
    """Serve the web interface"""
    return HTML_INTERFACE

@app.get("/status")
async def get_status():
    """Get system status"""
    try:
        status = await orchestrator.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process")
async def process_request(request: ProcessRequest):
    """Process a request without streaming"""
    try:
        result = await orchestrator.process_request(request.request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for streaming communication"""
    await manager.connect(websocket)
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            request_data = json.loads(data)
            user_request = request_data.get("request", "")
            
            if user_request:
                # Process the request with streaming
                async for update in orchestrator.process_request_stream(user_request):
                    await manager.send_personal_message(
                        json.dumps(update), 
                        websocket
                    )
                    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}),
            websocket
        )

@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("YouTube Workflow API starting up...")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("YouTube Workflow API shutting down...")
    await orchestrator.cleanup()

if __name__ == "__main__":
    uvicorn.run(
        "web_interface:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )