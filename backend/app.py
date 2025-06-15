from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Dict, Any, List, Optional, Union
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

# Pydantic models for request validation
class OperationRequest(BaseModel):
    action: str
    parameters: Dict[str, Any]

class FilterRequest(BaseModel):
    column: str
    operator: str = "=="
    value: Union[str, int, float]

class GroupAggregateRequest(BaseModel):
    group_by: str
    agg_functions: Dict[str, str]

class CreateColumnRequest(BaseModel):
    column_name: str
    source_columns: Optional[List[str]] = []
    operation: str = "concat"
    formula: Optional[str] = None

class CleanDataRequest(BaseModel):
    operations: List[str] = ["remove_nulls", "remove_duplicates"]
    case_type: Optional[str] = "lower"

class TransformDataRequest(BaseModel):
    transformation: str
    columns: List[str]

class StatisticalAnalysisRequest(BaseModel):
    analysis_type: str = "descriptive"
    columns: Optional[List[str]] = []

class SmartFillRequest(BaseModel):
    column: str
    method: str = "forward"

class BatchEditRequest(BaseModel):
    edits: List[Dict[str, Any]]

class DerivedColumnRequest(BaseModel):
    column_name: str
    derivation_type: str
    source_columns: Optional[List[str]] = []
    custom_logic: Optional[str] = None
    condition: Optional[str] = None
    true_value: Optional[Any] = None
    false_value: Optional[Any] = None
    bins: Optional[int] = 5
    labels: Optional[List[str]] = None
    date_feature: Optional[str] = "year"

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

# New dedicated endpoints for specific operations
@app.post("/operations/filter")
async def filter_data(request: FilterRequest):
    """Filter data based on conditions"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'filter_data',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/group-aggregate")
async def group_aggregate(request: GroupAggregateRequest):
    """Group data and perform aggregations"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'group_aggregate',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/create-column")
async def create_column(request: CreateColumnRequest):
    """Create new column"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'create_column',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/clean-data")
async def clean_data(request: CleanDataRequest):
    """Clean data with various operations"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'clean_data',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/transform-data")
async def transform_data(request: TransformDataRequest):
    """Transform data"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'transform_data',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/statistical-analysis")
async def statistical_analysis(request: StatisticalAnalysisRequest):
    """Perform statistical analysis"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'statistical_analysis',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/smart-fill")
async def smart_fill(request: SmartFillRequest):
    """Smart fill missing values"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'smart_fill',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/batch-edit")
async def batch_edit(request: BatchEditRequest):
    """Batch edit multiple cells"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'batch_edit',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/derived-column")
async def create_derived_column(request: DerivedColumnRequest):
    """Create derived column with advanced logic"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'create_derived_column',
        'parameters': request.dict()
    })
    return result

@app.post("/operations/pivot-table")
async def create_pivot_table(values: str, index: str, columns: Optional[str] = None, 
                           aggfunc: str = "sum", fill_value: Union[int, float] = 0):
    """Create pivot table"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'pivot_table',
        'parameters': {
            'values': values,
            'index': index,
            'columns': columns,
            'aggfunc': aggfunc,
            'fill_value': fill_value
        }
    })
    return result

@app.post("/operations/derive-features")
async def derive_features(feature_type: str = "basic", columns: Optional[List[str]] = None):
    """Create multiple derived features"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation({
        'action': 'derive_features',
        'parameters': {
            'feature_type': feature_type,
            'columns': columns or []
        }
    })
    return result

@app.post("/operations/execute")
async def execute_operation(request: OperationRequest):
    """Generic operation executor"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    result = processor.execute_smart_operation(request.dict())
    return result

@app.get("/operations/available")
async def get_available_operations():
    """Get list of available operations with their parameters"""
    return {
        "operations": {
            "filter_data": {
                "description": "Filter data based on conditions",
                "parameters": ["column", "operator", "value"],
                "operators": ["==", "!=", ">", "<", ">=", "<=", "contains", "startswith", "endswith"]
            },
            "group_aggregate": {
                "description": "Group data and perform aggregations",
                "parameters": ["group_by", "agg_functions"],
                "agg_functions": ["sum", "mean", "count", "max", "min", "std"]
            },
            "create_column": {
                "description": "Create new column",
                "parameters": ["column_name", "source_columns", "operation", "formula"],
                "operations": ["concat", "sum", "mean", "max", "min"]
            },
            "clean_data": {
                "description": "Clean data with various operations",
                "parameters": ["operations", "case_type"],
                "operations": ["remove_nulls", "remove_duplicates", "strip_whitespace", "standardize_case"]
            },
            "transform_data": {
                "description": "Transform data",
                "parameters": ["transformation", "columns"],
                "transformations": ["normalize", "standardize", "log_transform", "one_hot_encode"]
            },
            "statistical_analysis": {
                "description": "Perform statistical analysis",
                "parameters": ["analysis_type", "columns"],
                "analysis_types": ["descriptive", "correlation", "missing_values", "data_types"]
            },
            "smart_fill": {
                "description": "Smart fill missing values",
                "parameters": ["column", "method"],
                "methods": ["forward", "backward", "mean", "median", "mode"]
            },
            "derive_features": {
                "description": "Create multiple derived features",
                "parameters": ["feature_type", "columns"],
                "feature_types": ["basic", "rolling", "lag", "all"]
            },
            "pivot_table": {
                "description": "Create pivot table",
                "parameters": ["values", "index", "columns", "aggfunc", "fill_value"]
            },
            "create_derived_column": {
                "description": "Create derived column with advanced logic",
                "parameters": ["column_name", "derivation_type", "source_columns"],
                "derivation_types": ["conditional", "binning", "date_features", "custom"]
            }
        }
    }

@app.get("/data/summary")
async def get_data_summary():
    """Get comprehensive data summary"""
    if processor.data is None:
        raise HTTPException(status_code=400, detail="No data loaded")
    
    try:
        # Basic info
        numeric_columns = processor.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_columns = processor.data.select_dtypes(include=['object']).columns.tolist()
        
        summary = {
            "basic_info": {
                "shape": processor.data.shape,
                "columns": processor.data.columns.tolist(),
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "memory_usage": f"{processor.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            "missing_values": processor.data.isnull().sum().to_dict(),
            "data_types": {col: str(dtype) for col, dtype in processor.data.dtypes.items()}
        }
        
        # Add descriptive statistics for numeric columns
        if numeric_columns:
            desc_stats = processor.data[numeric_columns].describe()
            summary["descriptive_stats"] = processor._safe_json_convert(desc_stats.to_dict())
        
        # Add value counts for categorical columns (top 5 values each)
        if categorical_columns:
            categorical_info = {}
            for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
                value_counts = processor.data[col].value_counts().head().to_dict()
                categorical_info[col] = {
                    "unique_count": int(processor.data[col].nunique()),
                    "top_values": processor._safe_json_convert(value_counts)
                }
            summary["categorical_info"] = categorical_info
        
        return {
            "success": True,
            "summary": summary
        }
    except Exception as e:
        logger.error(f"Summary error: {e}")
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
                
                # Execute smart operation
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
    logger.info("  GET /data/summary - Get comprehensive data summary")
    logger.info("  GET /suggestions - Get smart suggestions")
    logger.info("  GET /history - Get operation history")
    logger.info("  POST /undo - Undo last operation")
    logger.info("  POST /save - Save current data")
    logger.info("  GET /operations/available - Get available operations")
    logger.info("  POST /operations/* - Execute specific operations")
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
        # reload=True  # Enable auto-reload for development
    )