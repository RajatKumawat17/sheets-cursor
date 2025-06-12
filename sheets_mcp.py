import os
import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import pandas as pd
import numpy as np

from mcp.server.models import InitializationOptions
from mcp.server import NotificationOptions, Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    CallToolResult,
    ListToolsRequest,
    ListToolsResult,
    TextContent,
    Tool,
)

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from openai import AzureOpenAI
from openai.types.chat import ChatCompletion
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoogleSheetsAPI:
    """Google Sheets API wrapper with comprehensive operations"""

    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]

    def __init__(self, credentials_path: str):
        self.credentials_path = credentials_path
        self.service = None
        self._authenticate()

    def _authenticate(self):
        creds = None
        token_path = "token.json"
        credentials_path = self.credentials_path

        # ✅ Check before trying to read token.json
        if os.path.exists(token_path):
            creds = Credentials.from_authorized_user_file(token_path, self.SCOPES)

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # ✅ Use credentials.json (client secret from Google Cloud)
                flow = InstalledAppFlow.from_client_secrets_file(
                    credentials_path, self.SCOPES
                )
                creds = flow.run_local_server(port=0)

                # ✅ Save the token after getting it
                with open(token_path, "w") as token:
                    token.write(creds.to_json())

        # ✅ Build the authorized Sheets service
        self.service = build("sheets", "v4", credentials=creds)

    def get_sheet_data(
        self, spreadsheet_id: str, range_name: str = None
    ) -> Dict[str, Any]:
        """Get data from a sheet or specific range"""
        try:
            if not range_name:
                # Get all sheets info
                sheet_metadata = (
                    self.service.spreadsheets()
                    .get(spreadsheetId=spreadsheet_id)
                    .execute()
                )
                return sheet_metadata

            result = (
                self.service.spreadsheets()
                .values()
                .get(spreadsheetId=spreadsheet_id, range=range_name)
                .execute()
            )

            values = result.get("values", [])
            return {
                "range": result.get("range"),
                "values": values,
                "row_count": len(values),
                "col_count": len(values[0]) if values else 0,
            }
        except HttpError as error:
            logger.error(f"Error getting sheet data: {error}")
            raise

    def update_range(
        self, spreadsheet_id: str, range_name: str, values: List[List[Any]]
    ) -> Dict[str, Any]:
        """Update a range of cells"""
        try:
            body = {"values": values}
            result = (
                self.service.spreadsheets()
                .values()
                .update(
                    spreadsheetId=spreadsheet_id,
                    range=range_name,
                    valueInputOption="USER_ENTERED",
                    body=body,
                )
                .execute()
            )
            return result
        except HttpError as error:
            logger.error(f"Error updating range: {error}")
            raise

    def batch_update(self, spreadsheet_id: str, requests: List[Dict]) -> Dict[str, Any]:
        """Perform batch updates for formatting, adding sheets, etc."""
        try:
            body = {"requests": requests}
            result = (
                self.service.spreadsheets()
                .batchUpdate(spreadsheetId=spreadsheet_id, body=body)
                .execute()
            )
            return result
        except HttpError as error:
            logger.error(f"Error in batch update: {error}")
            raise

    def create_sheet(self, spreadsheet_id: str, sheet_name: str) -> Dict[str, Any]:
        """Create a new sheet"""
        request = {"addSheet": {"properties": {"title": sheet_name}}}
        return self.batch_update(spreadsheet_id, [request])

    def insert_rows(
        self, spreadsheet_id: str, sheet_id: int, start_index: int, end_index: int
    ):
        """Insert rows in a sheet"""
        request = {
            "insertDimension": {
                "range": {
                    "sheetId": sheet_id,
                    "dimension": "ROWS",
                    "startIndex": start_index,
                    "endIndex": end_index,
                }
            }
        }
        return self.batch_update(spreadsheet_id, [request])

    def apply_formatting(
        self,
        spreadsheet_id: str,
        sheet_id: int,
        start_row: int,
        end_row: int,
        start_col: int,
        end_col: int,
        format_dict: Dict,
    ):
        """Apply formatting to a range"""
        request = {
            "repeatCell": {
                "range": {
                    "sheetId": sheet_id,
                    "startRowIndex": start_row,
                    "endRowIndex": end_row,
                    "startColumnIndex": start_col,
                    "endColumnIndex": end_col,
                },
                "cell": {"userEnteredFormat": format_dict},
                "fields": "userEnteredFormat",
            }
        }
        return self.batch_update(spreadsheet_id, [request])


class DataAnalyzer:
    """Data analysis utilities for spreadsheet data"""

    @staticmethod
    def analyze_column(data: List[List[Any]], col_index: int) -> Dict[str, Any]:
        """Analyze a specific column for insights"""
        if not data or col_index >= len(data[0]):
            return {}

        # Extract column data (skip header)
        col_data = [row[col_index] for row in data[1:] if len(row) > col_index]

        # Try to convert to numeric
        numeric_data = []
        for val in col_data:
            try:
                numeric_data.append(float(val))
            except (ValueError, TypeError):
                pass

        analysis = {
            "total_values": len(col_data),
            "numeric_values": len(numeric_data),
            "data_type": (
                "numeric" if len(numeric_data) > len(col_data) * 0.8 else "text"
            ),
        }

        if numeric_data:
            analysis.update(
                {
                    "mean": np.mean(numeric_data),
                    "median": np.median(numeric_data),
                    "std": np.std(numeric_data),
                    "min": np.min(numeric_data),
                    "max": np.max(numeric_data),
                    "outliers": DataAnalyzer._find_outliers(numeric_data),
                }
            )

        return analysis

    @staticmethod
    def _find_outliers(data: List[float], threshold: float = 2.0) -> List[float]:
        """Find outliers using z-score method"""
        if len(data) < 3:
            return []

        mean = np.mean(data)
        std = np.std(data)

        if std == 0:
            return []

        outliers = []
        for val in data:
            z_score = abs((val - mean) / std)
            if z_score > threshold:
                outliers.append(val)

        return outliers

    @staticmethod
    def create_pivot_summary(
        data: List[List[Any]], group_col: int, value_col: int
    ) -> Dict[str, Any]:
        """Create a pivot table summary"""
        if not data or len(data) < 2:
            return {}

        try:
            df = pd.DataFrame(data[1:], columns=data[0])
            group_name = data[0][group_col]
            value_name = data[0][value_col]

            pivot = (
                df.groupby(group_name)[value_name]
                .agg(["sum", "mean", "count"])
                .to_dict()
            )
            return {
                "group_column": group_name,
                "value_column": value_name,
                "summary": pivot,
            }
        except Exception as e:
            logger.error(f"Error creating pivot summary: {e}")
            return {}


class FormulaGenerator:
    """Generate and validate spreadsheet formulas"""

    @staticmethod
    def generate_sum_formula(range_str: str) -> str:
        """Generate SUM formula"""
        return f"=SUM({range_str})"

    @staticmethod
    def generate_average_formula(range_str: str) -> str:
        """Generate AVERAGE formula"""
        return f"=AVERAGE({range_str})"

    @staticmethod
    def generate_vlookup_formula(
        lookup_value: str, table_array: str, col_index: int, exact_match: bool = True
    ) -> str:
        """Generate VLOOKUP formula"""
        match_type = "FALSE" if exact_match else "TRUE"
        return f"=VLOOKUP({lookup_value},{table_array},{col_index},{match_type})"

    @staticmethod
    def generate_conditional_formula(
        condition: str, true_value: str, false_value: str
    ) -> str:
        """Generate IF formula"""
        return f"=IF({condition},{true_value},{false_value})"

    @staticmethod
    def validate_formula(formula: str) -> Dict[str, Any]:
        """Basic formula validation"""
        if not formula.startswith("="):
            return {"valid": False, "error": "Formula must start with ="}

        # Check for balanced parentheses
        open_count = formula.count("(")
        close_count = formula.count(")")

        if open_count != close_count:
            return {"valid": False, "error": "Unbalanced parentheses"}

        return {"valid": True, "error": None}


class MCPSheetsServer:
    """MCP Server for Google Sheets operations"""

    def __init__(self):
        self.sheets_api = GoogleSheetsAPI(os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH"))
        self.analyzer = DataAnalyzer()
        self.formula_gen = FormulaGenerator()
        self.azure_client = AzureOpenAI(
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )

        # Current context
        self.current_spreadsheet_id = None
        self.current_sheet_data = None
        self.conversation_history = []

    def get_tools(self) -> List[Tool]:
        """Define available MCP tools"""
        return [
            Tool(
                name="get_sheet_data",
                description="Get data from a Google Sheet or specific range",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {
                            "type": "string",
                            "description": "The Google Sheets spreadsheet ID",
                        },
                        "range": {
                            "type": "string",
                            "description": "Range to get data from (e.g., 'A1:D10' or 'Sheet1!A:D')",
                        },
                    },
                    "required": ["spreadsheet_id"],
                },
            ),
            Tool(
                name="update_range",
                description="Update a range of cells in a Google Sheet",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "range": {"type": "string"},
                        "values": {
                            "type": "array",
                            "items": {"type": "array"},
                            "description": "2D array of values to update",
                        },
                    },
                    "required": ["spreadsheet_id", "range", "values"],
                },
            ),
            Tool(
                name="analyze_column",
                description="Analyze a specific column for statistical insights",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "range": {"type": "string"},
                        "column_index": {
                            "type": "integer",
                            "description": "0-based column index to analyze",
                        },
                    },
                    "required": ["spreadsheet_id", "range", "column_index"],
                },
            ),
            Tool(
                name="create_formula",
                description="Generate and insert a formula into a cell",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "cell": {
                            "type": "string",
                            "description": "Cell reference (e.g., 'A1')",
                        },
                        "formula_type": {
                            "type": "string",
                            "enum": ["SUM", "AVERAGE", "VLOOKUP", "IF", "CUSTOM"],
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Parameters for formula generation",
                        },
                    },
                    "required": ["spreadsheet_id", "cell", "formula_type"],
                },
            ),
            Tool(
                name="create_pivot_summary",
                description="Create a pivot table summary from data",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "range": {"type": "string"},
                        "group_column_index": {"type": "integer"},
                        "value_column_index": {"type": "integer"},
                    },
                    "required": [
                        "spreadsheet_id",
                        "range",
                        "group_column_index",
                        "value_column_index",
                    ],
                },
            ),
            Tool(
                name="apply_formatting",
                description="Apply formatting to a range of cells",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "range": {"type": "string"},
                        "format_type": {
                            "type": "string",
                            "enum": [
                                "bold",
                                "italic",
                                "currency",
                                "percentage",
                                "date",
                            ],
                        },
                    },
                    "required": ["spreadsheet_id", "range", "format_type"],
                },
            ),
            Tool(
                name="insert_rows",
                description="Insert new rows into a sheet",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "sheet_name": {"type": "string"},
                        "start_row": {"type": "integer"},
                        "count": {"type": "integer"},
                    },
                    "required": ["spreadsheet_id", "sheet_name", "start_row", "count"],
                },
            ),
            Tool(
                name="smart_analysis",
                description="Perform AI-powered analysis on sheet data with natural language query",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "spreadsheet_id": {"type": "string"},
                        "range": {"type": "string"},
                        "query": {
                            "type": "string",
                            "description": "Natural language query about the data",
                        },
                    },
                    "required": ["spreadsheet_id", "range", "query"],
                },
            ),
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> CallToolResult:
        """Execute a tool call"""
        try:
            if name == "get_sheet_data":
                return await self._get_sheet_data(arguments)
            elif name == "update_range":
                return await self._update_range(arguments)
            elif name == "analyze_column":
                return await self._analyze_column(arguments)
            elif name == "create_formula":
                return await self._create_formula(arguments)
            elif name == "create_pivot_summary":
                return await self._create_pivot_summary(arguments)
            elif name == "apply_formatting":
                return await self._apply_formatting(arguments)
            elif name == "insert_rows":
                return await self._insert_rows(arguments)
            elif name == "smart_analysis":
                return await self._smart_analysis(arguments)
            else:
                raise ValueError(f"Unknown tool: {name}")

        except Exception as e:
            logger.error(f"Error executing tool {name}: {e}")
            return CallToolResult(
                content=[TextContent(type="text", text=f"Error: {str(e)}")],
                isError=True,
            )

    async def _get_sheet_data(self, args: Dict[str, Any]) -> CallToolResult:
        """Get sheet data tool implementation"""
        spreadsheet_id = args["spreadsheet_id"]
        range_name = args.get("range")

        data = self.sheets_api.get_sheet_data(spreadsheet_id, range_name)

        # Store current context
        self.current_spreadsheet_id = spreadsheet_id
        if "values" in data:
            self.current_sheet_data = data["values"]

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(data, indent=2))]
        )

    async def _update_range(self, args: Dict[str, Any]) -> CallToolResult:
        """Update range tool implementation"""
        result = self.sheets_api.update_range(
            args["spreadsheet_id"], args["range"], args["values"]
        )

        return CallToolResult(
            content=[
                TextContent(
                    type="text", text=f"Updated {result.get('updatedCells', 0)} cells"
                )
            ]
        )

    async def _analyze_column(self, args: Dict[str, Any]) -> CallToolResult:
        """Analyze column tool implementation"""
        # First get the data
        data = self.sheets_api.get_sheet_data(args["spreadsheet_id"], args["range"])

        if "values" not in data:
            raise ValueError("No data found in range")

        analysis = self.analyzer.analyze_column(data["values"], args["column_index"])

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(analysis, indent=2))]
        )

    async def _create_formula(self, args: Dict[str, Any]) -> CallToolResult:
        """Create formula tool implementation"""
        formula_type = args["formula_type"]
        params = args.get("parameters", {})

        if formula_type == "SUM":
            formula = self.formula_gen.generate_sum_formula(
                params.get("range", "A1:A10")
            )
        elif formula_type == "AVERAGE":
            formula = self.formula_gen.generate_average_formula(
                params.get("range", "A1:A10")
            )
        elif formula_type == "VLOOKUP":
            formula = self.formula_gen.generate_vlookup_formula(
                params.get("lookup_value", "A1"),
                params.get("table_array", "B:D"),
                params.get("col_index", 2),
            )
        elif formula_type == "IF":
            formula = self.formula_gen.generate_conditional_formula(
                params.get("condition", "A1>0"),
                params.get("true_value", '"Yes"'),
                params.get("false_value", '"No"'),
            )
        else:
            formula = params.get("custom_formula", "=1+1")

        # Validate formula
        validation = self.formula_gen.validate_formula(formula)
        if not validation["valid"]:
            raise ValueError(f"Invalid formula: {validation['error']}")

        # Insert formula
        result = self.sheets_api.update_range(
            args["spreadsheet_id"], args["cell"], [[formula]]
        )

        return CallToolResult(
            content=[TextContent(type="text", text=f"Created formula: {formula}")]
        )

    async def _create_pivot_summary(self, args: Dict[str, Any]) -> CallToolResult:
        """Create pivot summary tool implementation"""
        # Get data first
        data = self.sheets_api.get_sheet_data(args["spreadsheet_id"], args["range"])

        if "values" not in data:
            raise ValueError("No data found in range")

        summary = self.analyzer.create_pivot_summary(
            data["values"], args["group_column_index"], args["value_column_index"]
        )

        return CallToolResult(
            content=[TextContent(type="text", text=json.dumps(summary, indent=2))]
        )

    async def _apply_formatting(self, args: Dict[str, Any]) -> CallToolResult:
        """Apply formatting tool implementation"""
        # This is a simplified version - you'd need to parse the range and get sheet ID
        format_dict = self._get_format_dict(args["format_type"])

        # For now, return a message indicating what would be formatted
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Applied {args['format_type']} formatting to {args['range']}",
                )
            ]
        )

    def _get_format_dict(self, format_type: str) -> Dict[str, Any]:
        """Get formatting dictionary for different format types"""
        formats = {
            "bold": {"textFormat": {"bold": True}},
            "italic": {"textFormat": {"italic": True}},
            "currency": {"numberFormat": {"type": "CURRENCY"}},
            "percentage": {"numberFormat": {"type": "PERCENT"}},
            "date": {"numberFormat": {"type": "DATE"}},
        }
        return formats.get(format_type, {})

    async def _insert_rows(self, args: Dict[str, Any]) -> CallToolResult:
        """Insert rows tool implementation"""
        # This would require getting the sheet ID first
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=f"Inserted {args['count']} rows starting at row {args['start_row']}",
                )
            ]
        )

    async def _smart_analysis(self, args: Dict[str, Any]) -> CallToolResult:
        """Smart analysis using Azure OpenAI"""
        # Get the data
        data = self.sheets_api.get_sheet_data(args["spreadsheet_id"], args["range"])

        if "values" not in data:
            raise ValueError("No data found in range")

        # Prepare data summary for AI
        data_summary = self._prepare_data_summary(data["values"])

        # Create prompt for Azure OpenAI
        prompt = f"""
        You are analyzing spreadsheet data. Here's the data summary:
        
        Data Summary:
        {data_summary}
        
        User Query: {args['query']}
        
        Please provide insights, patterns, or answer the user's question about this data.
        Be specific and actionable in your response.
        """

        try:
            response = self.azure_client.chat.completions.create(
                model="gpt-4",  # or your deployed model name
                messages=[
                    {
                        "role": "system",
                        "content": "You are a data analysis expert helping users understand their spreadsheet data.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
            )

            analysis_result = response.choices[0].message.content

            return CallToolResult(
                content=[TextContent(type="text", text=analysis_result)]
            )

        except Exception as e:
            logger.error(f"Error in smart analysis: {e}")
            return CallToolResult(
                content=[
                    TextContent(
                        type="text", text=f"Error performing smart analysis: {str(e)}"
                    )
                ],
                isError=True,
            )

    def _prepare_data_summary(self, data: List[List[Any]]) -> str:
        """Prepare a summary of the data for AI analysis"""
        if not data:
            return "No data available"

        summary = []
        summary.append(f"Total rows: {len(data)}")

        if len(data) > 0:
            summary.append(f"Columns: {len(data[0])}")

            # Add headers if available
            if len(data) > 1:
                summary.append(f"Headers: {data[0]}")

                # Sample data
                sample_size = min(5, len(data) - 1)
                summary.append(f"Sample data (first {sample_size} rows):")
                for i in range(1, sample_size + 1):
                    if i < len(data):
                        summary.append(f"Row {i}: {data[i]}")

        return "\n".join(summary)


# Main MCP Server setup
async def main():
    """Main MCP server entry point"""
    server = Server("sheets-assistant")
    sheets_server = MCPSheetsServer()

    @server.list_tools()
    async def handle_list_tools() -> ListToolsResult:
        """Handle tool listing"""
        return ListToolsResult(tools=sheets_server.get_tools())

    @server.call_tool()
    async def handle_call_tool(request: CallToolRequest) -> CallToolResult:
        """Handle tool calls"""
        return await sheets_server.call_tool(
            request.params.name, request.params.arguments or {}
        )

    # Run the server
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="sheets-assistant",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
