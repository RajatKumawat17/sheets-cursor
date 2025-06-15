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

@app.get("/data/info")
async def get_data_info():
    """Get current data information"""
    if processor.data is None:
        return {"error": "No data loaded"}
    
    return {
        "success": True,
        "columns": processor.data.columns.tolist(),
        "shape": processor.data.shape,
        "dtypes": {col: str(dtype) for col, dtype in processor.data.dtypes.items()},
        "memory_usage": f"{processor.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        "null_counts": processor.data.isnull().sum().to_dict()
    }

@app.get("/data/preview")
async def get_data_preview(rows: int = 10):
    """Get data preview"""
    if processor.data is None:
        return {"error": "No data loaded"}
    
    try:
        preview_data = processor.data.head(rows)
        return {
            "success": True,
            "data": processor._safe_json_convert(preview_data.to_dict('records')),
            "columns": preview_data.columns.tolist(),
            "total_rows": len(processor.data)
        }
    except Exception as e:
        logger.error(f"Preview error: {e}")
        return {"error": str(e)}

@app.get("/suggestions")
async def get_suggestions():
    """Get smart suggestions based on current data"""
    if processor.data is None:
        return {"error": "No data loaded"}
    
    try:
        data_info = {
            "columns": processor.data.columns.tolist(),
            "shape": processor.data.shape,
            "dtypes": {col: str(dtype) for col, dtype in processor.data.dtypes.items()}
        }
        
        suggestions = llm_client.generate_smart_suggestions(data_info)
        return {
            "success": True,
            "suggestions": suggestions
        }
    except Exception as e:
        logger.error(f"Suggestions error: {e}")
        return {"error": str(e)}

@app.get("/history")
async def get_history():
    """Get operation history"""
    return {
        "success": True,
        "history": processor.get_history()
    }

@app.post("/undo")
async def undo_operation():
    """Undo last operation"""
    result = processor.undo_last_operation()
    return result

@app.post("/save")
async def save_data(file_path: str):
    """Save current data to file"""
    if processor.data is None:
        return {"error": "No data loaded"}
    
    try:
        # Ensure the file is saved in uploads directory for security
        safe_path = f"uploads/{Path(file_path).name}"
        result = processor.save_data(safe_path)
        return result
    except Exception as e:
        logger.error(f"Save error: {e}")
        return {"error": str(e)}

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
            
            # Check if data is loaded
            if processor.data is None:
                await websocket.send_text(json.dumps({
                    "error": "No data loaded. Please upload a file first."
                }))
                continue
            
            # Get data info for LLM context
            data_info = {
                "columns": processor.data.columns.tolist(),
                "shape": processor.data.shape,
                "dtypes": {col: str(dtype) for col, dtype in processor.data.dtypes.items()}
            }
            
            # Send processing status
            await websocket.send_text(json.dumps({
                "status": "processing", 
                "message": "Analyzing command..."
            }))
            
            try:
                # Get analysis instruction from LLM
                instruction = llm_client.analyze_command(command, data_info)
                
                # Send analysis status
                await websocket.send_text(json.dumps({
                    "status": "executing",
                    "message": f"Executing {instruction.get('action', 'operation')}...",
                    "instruction": instruction
                }))
                
                # Execute smart operation (updated method name)
                result = processor.execute_smart_operation(instruction)
                
                # Enhanced response with data preview if operation was successful
                if result.get("success"):
                    # Include updated data preview for successful operations
                    preview = processor._safe_json_convert(
                        processor.data.head(5).to_dict('records')
                    ) if processor.data is not None else []
                    
                    result["data_preview"] = preview
                    result["data_shape"] = processor.data.shape if processor.data is not None else None
                
                # Send final result
                await websocket.send_text(json.dumps({
                    "status": "complete",
                    "command": command,
                    "instruction": instruction,
                    "result": result,
                    "timestamp": command_data.get("timestamp")
                }))
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "error": f"Invalid JSON in LLM response: {str(e)}"
                }))
                
            except Exception as e:
                logger.error(f"Command processing error: {e}")
                await websocket.send_text(json.dumps({
                    "status": "error",
                    "error": f"Processing failed: {str(e)}",
                    "command": command
                }))
    
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.send_text(json.dumps({
                "status": "error",
                "error": str(e)
            }))
        except:
            # Connection might be closed
            pass

@app.on_event("startup")
async def startup_event():
    """Startup event handler"""
    logger.info("Excel MCP Tool starting up...")
    logger.info("Available endpoints:")
    logger.info("  GET / - Serve frontend")
    logger.info("  POST /upload - Upload Excel/CSV files")
    logger.info("  GET /data/info - Get data information")
    logger.info("  GET /data/preview - Get data preview")
    logger.info("  GET /suggestions - Get smart suggestions")
    logger.info("  GET /history - Get operation history")
    logger.info("  POST /undo - Undo last operation")
    logger.info("  POST /save - Save current data")
    logger.info("  WebSocket /ws - Real-time analysis")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event handler"""
    logger.info("Excel MCP Tool shutting down...")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Excel MCP Tool",
        "data_loaded": processor.data is not None,
        "data_shape": processor.data.shape if processor.data is not None else None
    }

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Global exception: {exc}")
    return {
        "error": "Internal server error",
        "detail": str(exc) if app.debug else "An unexpected error occurred"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="localhost", 
        port=8000,
        log_level="info",
        reload=True  # Enable auto-reload for development
    )