import asyncio
import json
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, Tool, TextContent
from excel_processor import ExcelProcessor
from llm_client import LLMClient
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Server("excel-mcp-tool")
processor = ExcelProcessor()
llm_client = LLMClient()

@app.list_resources()
async def list_resources() -> list[Resource]:
    """List available resources"""
    return [
        Resource(
            uri="excel://data",
            name="Excel Data",
            mimeType="application/json",
            description="Current loaded Excel/CSV data"
        )
    ]

@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools"""
    return [
        Tool(
            name="load_excel",
            description="Load an Excel or CSV file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path to the Excel/CSV file"}
                },
                "required": ["file_path"]
            }
        ),
        Tool(
            name="analyze_data",
            description="Analyze data using natural language commands",
            inputSchema={
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Natural language command for data analysis"}
                },
                "required": ["command"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls"""
    try:
        if name == "load_excel":
            file_path = arguments["file_path"]
            result = processor.load_file(file_path)
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        elif name == "analyze_data":
            command = arguments["command"]
            
            # Get data info for LLM context
            data_info = {
                "columns": processor.data.columns.tolist() if processor.data is not None else [],
                "shape": processor.data.shape if processor.data is not None else None
            }
            
            # Get analysis instruction from LLM
            instruction = llm_client.analyze_command(command, data_info)
            
            # Execute analysis
            result = processor.execute_analysis(instruction)
            
            return [TextContent(type="text", text=json.dumps(result, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool execution error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

async def main():
    """Run the MCP server"""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
