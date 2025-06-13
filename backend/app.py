from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import json
import asyncio
from pathlib import Path
import shutil
from excel_processor import ExcelProcessor
from llm_client import LLMClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Excel MCP Tool")
processor = ExcelProcessor()
llm_client = LLMClient()

# Create uploads directory
Path("uploads").mkdir(exist_ok=True)

# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")

@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process Excel/CSV file"""
    try:
        file_path = f"uploads/{file.filename}"
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        result = processor.load_file(file_path)
        return result
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return {"success": False, "error": str(e)}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time analysis"""
    await websocket.accept()
    
    try:
        while True:
            # Receive command from frontend
            data = await websocket.receive_text()
            command_data = json.loads(data)
            
            command = command_data.get("command")
            
            if not command:
                await websocket.send_text(json.dumps({"error": "No command provided"}))
                continue
            
            # Get data info for LLM context
            data_info = {
                "columns": processor.data.columns.tolist() if processor.data is not None else [],
                "shape": processor.data.shape if processor.data is not None else None
            }
            
            # Send processing status
            await websocket.send_text(json.dumps({"status": "processing", "message": "Analyzing command..."}))
            
            # Get analysis instruction from LLM
            instruction = llm_client.analyze_command(command, data_info)
            
            # Execute analysis
            result = processor.execute_analysis(instruction)
            
            # Send result
            await websocket.send_text(json.dumps({
                "status": "complete",
                "command": command,
                "instruction": instruction,
                "result": result
            }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_text(json.dumps({"error": str(e)}))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
