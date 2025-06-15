import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent, ImageContent, EmbeddedResource
from excel_processor import ExcelProcessor
from llm_client import LLMClient
import logging
from typing import Dict, Any, List
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPServer:
    def __init__(self):
        self.app = Server("excel-mcp-tool")
        self.processor = ExcelProcessor()
        self.llm_client = LLMClient()
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup all MCP handlers"""
        
        @self.app.list_resources()
        async def list_resources() -> List[Resource]:
            """List available resources"""
            resources = [
                Resource(
                    uri="excel://current-data",
                    name="Current Excel Data",
                    mimeType="application/json",
                    description="Currently loaded Excel/CSV data with metadata"
                ),
                Resource(
                    uri="excel://history",
                    name="Operation History",
                    mimeType="application/json",
                    description="History of all operations performed on the data"
                ),
                Resource(
                    uri="excel://suggestions",
                    name="Smart Suggestions",
                    mimeType="application/json",
                    description="AI-generated suggestions for data operations"
                )
            ]
            
            # Add schema resource if data is loaded
            if self.processor.data is not None:
                resources.append(
                    Resource(
                        uri="excel://schema",
                        name="Data Schema",
                        mimeType="application/json",
                        description="Detailed schema information of the current dataset"
                    )
                )
            
            return resources
        
        @self.app.read_resource()
        async def read_resource(uri: str) -> str:
            """Read resource content"""
            try:
                if uri == "excel://current-data":
                    if self.processor.data is None:
                        return json.dumps({"error": "No data loaded"})
                    
                    return json.dumps({
                        "filename": self.processor.filename,
                        "shape": self.processor.data.shape,
                        "columns": self.processor.data.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in self.processor.data.dtypes.items()},
                        "preview": self.processor._safe_json_convert(
                            self.processor.data.head(10).to_dict('records')
                        ),
                        "memory_usage": f"{self.processor.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                        "null_counts": self.processor.data.isnull().sum().to_dict()
                    }, indent=2)
                
                elif uri == "excel://history":
                    return json.dumps(self.processor.get_history(), indent=2)
                
                elif uri == "excel://suggestions":
                    if self.processor.data is None:
                        return json.dumps({"suggestions": []})
                    
                    data_info = {
                        "columns": self.processor.data.columns.tolist(),
                        "dtypes": {col: str(dtype) for col, dtype in self.processor.data.dtypes.items()},
                        "shape": self.processor.data.shape
                    }
                    
                    suggestions = self.llm_client.generate_smart_suggestions(data_info)
                    return json.dumps({"suggestions": suggestions}, indent=2)
                
                elif uri == "excel://schema":
                    if self.processor.data is None:
                        return json.dumps({"error": "No data loaded"})
                    
                    schema_info = {
                        "columns": [],
                        "data_quality": {},
                        "statistics": {}
                    }
                    
                    for col in self.processor.data.columns:
                        col_info = {
                            "name": col,
                            "dtype": str(self.processor.data[col].dtype),
                            "null_count": int(self.processor.data[col].isnull().sum()),
                            "unique_count": int(self.processor.data[col].nunique()),
                            "sample_values": self.processor.data[col].dropna().head(3).tolist()
                        }
                        
                        # Add statistics for numeric columns
                        if self.processor.data[col].dtype in ['int64', 'float64']:
                            col_info["statistics"] = {
                                "mean": float(self.processor.data[col].mean()),
                                "std": float(self.processor.data[col].std()),
                                "min": float(self.processor.data[col].min()),
                                "max": float(self.processor.data[col].max())
                            }
                        
                        schema_info["columns"].append(col_info)
                    
                    return json.dumps(schema_info, indent=2)
                
                else:
                    return json.dumps({"error": f"Unknown resource: {uri}"})
            
            except Exception as e:
                logger.error(f"Resource read error: {e}")
                return json.dumps({"error": str(e)})
        
        @self.app.list_tools()
        async def list_tools() -> List[Tool]:
            """List available tools"""
            return [
                Tool(
                    name="load_excel_file",
                    description="Load an Excel or CSV file for analysis",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string", 
                                "description": "Path to the Excel/CSV file to load"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="smart_data_analysis",
                    description="Perform intelligent data analysis using natural language commands",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string", 
                                "description": "Natural language command describing the analysis to perform"
                            }
                        },
                        "required": ["command"]
                    }
                ),
                Tool(
                    name="edit_cell_value",
                    description="Edit a specific cell in the dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "row": {"type": "integer", "description": "Row index (0-based)"},
                            "column": {"type": "string", "description": "Column name"},
                            "value": {"description": "New value for the cell"}
                        },
                        "required": ["row", "column", "value"]
                    }
                ),
                Tool(
                    name="create_new_column",
                    description="Create a new column with calculated values",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "column_name": {"type": "string", "description": "Name for the new column"},
                            "source_columns": {
                                "type": "array", 
                                "items": {"type": "string"}, 
                                "description": "Source columns to use"
                            },
                            "operation": {
                                "type": "string", 
                                "enum": ["sum", "mean", "concat", "max", "min"],
                                "description": "Operation to perform"
                            },
                            "formula": {
                                "type": "string", 
                                "description": "Custom pandas formula (optional)"
                            }
                        },
                        "required": ["column_name"]
                    }
                ),
                Tool(
                    name="derive_features",
                    description="Create derived features for machine learning",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "feature_type": {
                                "type": "string",
                                "enum": ["basic", "rolling", "lag", "all"],
                                "description": "Type of features to derive"
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Target columns (optional, defaults to numeric columns)"
                            }
                        }
                    }
                ),
                Tool(
                    name="apply_excel_formula",
                    description="Apply Excel-like formulas to create new columns",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "formula": {"type": "string", "description": "Excel-like formula"},
                            "target_column": {"type": "string", "description": "Column to store results"}
                        },
                        "required": ["formula", "target_column"]
                    }
                ),
                Tool(
                    name="batch_edit_cells",
                    description="Edit multiple cells in a single operation",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "edits": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "row": {"type": "integer"},
                                        "column": {"type": "string"},
                                        "value": {}
                                    },
                                    "required": ["row", "column", "value"]
                                },
                                "description": "List of edits to perform"
                            }
                        },
                        "required": ["edits"]
                    }
                ),
                Tool(
                    name="smart_fill_missing",
                    description="Intelligently fill missing values in a column",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "column": {"type": "string", "description": "Column to fill"},
                            "method": {
                                "type": "string",
                                "enum": ["forward", "backward", "mean", "median", "mode"],
                                "description": "Fill method"
                            }
                        },
                        "required": ["column", "method"]
                    }
                ),
                Tool(
                    name="generate_pandas_code",
                    description="Generate pandas code for complex operations",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "operation_type": {
                                "type": "string",
                                "enum": ["data_cleaning", "feature_engineering", "aggregation", "pivot"],
                                "description": "Type of operation"
                            },
                            "description": {"type": "string", "description": "Description of desired operation"}
                        },
                        "required": ["operation_type"]
                    }
                ),
                Tool(
                    name="execute_custom_code",
                    description="Execute custom pandas code on the dataset",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "code": {"type": "string", "description": "Pandas code to execute"}
                        },
                        "required": ["code"]
                    }
                ),
                Tool(
                    name="save_data",
                    description="Save the current dataset to a file",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string", "description": "Path to save the file"},
                            "format": {
                                "type": "string",
                                "enum": ["csv", "excel"],
                                "description": "File format"
                            }
                        },
                        "required": ["file_path"]
                    }
                ),
                Tool(
                    name="undo_operation",
                    description="Undo the last operation performed on the dataset",
                    inputSchema={"type": "object", "properties": {}}
                ),
                Tool(
                    name="get_data_summary",
                    description="Get a comprehensive summary of the current dataset",
                    inputSchema={"type": "object", "properties": {}}
                )
            ]
        
        @self.app.call_tool()
        async def call_tool(name: str, arguments: dict) -> List[TextContent]:
            """Handle tool calls"""
            try:
                result = None
                
                if name == "load_excel_file":
                    file_path = arguments["file_path"]
                    result = self.processor.load_file(file_path)
                
                elif name == "smart_data_analysis":
                    if self.processor.data is None:
                        result = {"error": "No data loaded. Please load a file first."}
                    else:
                        command = arguments["command"]
                        data_info = {
                            "columns": self.processor.data.columns.tolist(),
                            "shape": self.processor.data.shape,
                            "dtypes": {col: str(dtype) for col, dtype in self.processor.data.dtypes.items()}
                        }
                        
                        instruction = self.llm_client.analyze_command(command, data_info)
                        result = self.processor.execute_smart_operation(instruction)
                
                elif name == "edit_cell_value":
                    result = self.processor.execute_smart_operation({
                        "action": "edit_cell",
                        "parameters": arguments
                    })
                
                elif name == "create_new_column":
                    result = self.processor.execute_smart_operation({
                        "action": "create_column",
                        "parameters": arguments
                    })
                
                elif name == "derive_features":
                    result = self.processor.execute_smart_operation({
                        "action": "derive_features",
                        "parameters": arguments
                    })
                
                elif name == "apply_excel_formula":
                    result = self.processor.execute_smart_operation({
                        "action": "apply_formula",
                        "parameters": arguments
                    })
                
                elif name == "batch_edit_cells":
                    result = self.processor.execute_smart_operation({
                        "action": "batch_edit",
                        "parameters": arguments
                    })
                
                elif name == "smart_fill_missing":
                    result = self.processor.execute_smart_operation({
                        "action": "smart_fill",
                        "parameters": arguments
                    })
                
                elif name == "generate_pandas_code":
                    result = self.processor.execute_smart_operation({
                        "action": "generate_code",
                        "parameters": arguments
                    })
                
                elif name == "execute_custom_code":
                    result = self.processor.execute_smart_operation({
                        "action": "execute_code",
                        "parameters": arguments
                    })
                
                elif name == "save_data":
                    file_path = arguments["file_path"]
                    file_format = arguments.get("format", "csv")
                    
                    if not file_path.endswith(('.csv', '.xlsx')):
                        if file_format == "excel":
                            file_path += ".xlsx"
                        else:
                            file_path += ".csv"
                    
                    result = self.processor.save_data(file_path)
                
                elif name == "undo_operation":
                    result = self.processor.undo_last_operation()
                
                elif name == "get_data_summary":
                    if self.processor.data is None:
                        result = {"error": "No data loaded"}
                    else:
                        result = {
                            "filename": self.processor.filename,
                            "shape": self.processor.data.shape,
                            "columns": len(self.processor.data.columns),
                            "memory_usage": f"{self.processor.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                            "data_types": self.processor.data.dtypes.value_counts().to_dict(),
                            "missing_values": self.processor.data.isnull().sum().sum(),
                            "operations_performed": len(self.processor.history)
                        }
                
                else:
                    result = {"error": f"Unknown tool: {name}"}
                
                return [TextContent(type="text", text=json.dumps(result, indent=2))]
            
            except Exception as e:
                logger.error(f"Tool execution error for {name}: {e}")
                return [TextContent(type="text", text=json.dumps({"error": str(e)}, indent=2))]
    
    async def run(self):
        """Run the MCP server"""
        try:
            async with stdio_server() as (read_stream, write_stream):
                await self.app.run(read_stream, write_stream, self.app.create_initialization_options())
        except Exception as e:
            logger.error(f"Server error: {e}")
            raise

# Create and run server
async def main():
    """Main entry point"""
    server = MCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())