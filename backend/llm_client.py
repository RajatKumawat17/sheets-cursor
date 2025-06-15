from openai import AzureOpenAI
import json
import logging
import re
from typing import Dict, List, Any
from dotenv import load_dotenv
import os

load_dotenv()

class LLMClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY")
        )
        
        # Enhanced command patterns
        self.command_patterns = {
            'edit_cell': [r'edit cell', r'change cell', r'update cell', r'modify cell'],
            'create_column': [r'create column', r'add column', r'new column'],
            'derive_features': [r'derive features', r'create features', r'feature engineering'],
            'apply_formula': [r'apply formula', r'formula', r'calculate'],
            'batch_edit': [r'batch edit', r'multiple edits', r'bulk edit'],
            'smart_fill': [r'fill missing', r'fill nulls', r'fill na'],
            'generate_code': [r'generate code', r'create code', r'write code'],
            'execute_code': [r'execute code', r'run code']
        }
    
    def analyze_command(self, command: str, data_info: Dict) -> Dict:
        """Enhanced command analysis with smart pattern matching"""
        
        # First try pattern matching for quick responses
        detected_action = self._detect_action_pattern(command)
        
        if detected_action:
            params = self._extract_parameters(command, detected_action, data_info)
            return {
                "action": detected_action,
                "parameters": params,
                "confidence": "high",
                "method": "pattern_matching"
            }
        
        # Fallback to LLM analysis for complex commands
        return self._llm_analyze_command(command, data_info)
    
    def _detect_action_pattern(self, command: str) -> str:
        """Detect action using pattern matching"""
        command_lower = command.lower()
        
        for action, patterns in self.command_patterns.items():
            if any(re.search(pattern, command_lower) for pattern in patterns):
                return action
        
        return None
    
    def _extract_parameters(self, command: str, action: str, data_info: Dict) -> Dict:
        """Extract parameters based on detected action"""
        params = {}
        columns = data_info.get('columns', [])
        
        if action == 'edit_cell':
            # Extract row, column, value
            row_match = re.search(r'row\s*(\d+)', command, re.IGNORECASE)
            if row_match:
                params['row'] = int(row_match.group(1))
            
            # Find column name
            for col in columns:
                if col.lower() in command.lower():
                    params['column'] = col
                    break
            
            # Extract value (after "to" or "with")
            value_match = re.search(r'(?:to|with)\s+(.+?)(?:\s|$)', command, re.IGNORECASE)
            if value_match:
                params['value'] = value_match.group(1).strip()
        
        elif action == 'create_column':
            # Extract column name
            name_match = re.search(r'(?:column|named?)\s+["\']?([^"\']+)["\']?', command, re.IGNORECASE)
            if name_match:
                params['column_name'] = name_match.group(1).strip()
            
            # Extract source columns
            source_cols = [col for col in columns if col.lower() in command.lower()]
            if source_cols:
                params['source_columns'] = source_cols
            
            # Detect operation
            if 'sum' in command.lower():
                params['operation'] = 'sum'
            elif 'concat' in command.lower() or 'combine' in command.lower():
                params['operation'] = 'concat'
            elif 'mean' in command.lower() or 'average' in command.lower():
                params['operation'] = 'mean'
        
        elif action == 'derive_features':
            if 'basic' in command.lower():
                params['feature_type'] = 'basic'
            elif 'rolling' in command.lower():
                params['feature_type'] = 'rolling'
            elif 'lag' in command.lower():
                params['feature_type'] = 'lag'
            else:
                params['feature_type'] = 'all'
            
            # Extract target columns
            target_cols = [col for col in columns if col.lower() in command.lower()]
            if target_cols:
                params['columns'] = target_cols
        
        elif action == 'smart_fill':
            # Find column
            for col in columns:
                if col.lower() in command.lower():
                    params['column'] = col
                    break
            
            # Detect method
            if 'forward' in command.lower() or 'ffill' in command.lower():
                params['method'] = 'forward'
            elif 'backward' in command.lower() or 'bfill' in command.lower():
                params['method'] = 'backward'
            elif 'mean' in command.lower():
                params['method'] = 'mean'
            elif 'median' in command.lower():
                params['method'] = 'median'
            elif 'mode' in command.lower():
                params['method'] = 'mode'
        
        return params
    
    def _llm_analyze_command(self, command: str, data_info: Dict) -> Dict:
        """Use LLM for complex command analysis"""
        
        system_prompt = f"""
        You are an expert data analysis assistant. Convert natural language commands into structured analysis instructions for a pandas DataFrame.
        
        Available data columns: {data_info.get('columns', [])}
        Data shape: {data_info.get('shape', 'Unknown')}
        Data types: {data_info.get('dtypes', {})}
        
        Available actions and their parameters:
        
        1. edit_cell: Edit a specific cell
           - row: int (row index)
           - column: str (column name)
           - value: any (new value)
        
        2. create_column: Create new column from existing ones
           - column_name: str
           - source_columns: list[str]
           - operation: str (sum, mean, concat, max, min)
           - formula: str (pandas expression, optional)
        
        3. derive_features: Create multiple derived features
           - feature_type: str (basic, rolling, lag, all)
           - columns: list[str] (target columns)
        
        4. apply_formula: Apply Excel-like formula
           - formula: str (Excel-like formula)
           - target_column: str
        
        5. batch_edit: Edit multiple cells
           - edits: list[dict] (each with row, column, value)
        
        6. smart_fill: Fill missing values intelligently
           - column: str
           - method: str (forward, backward, mean, median, mode)
        
        7. generate_code: Generate pandas code
           - operation_type: str (data_cleaning, feature_engineering, aggregation, pivot)
           - description: str
        
        8. execute_code: Execute custom pandas code
           - code: str (pandas code to execute)
        
        Return ONLY a valid JSON object with:
        - action: str (one of the above actions)
        - parameters: dict (specific parameters for the action)
        - confidence: str (high, medium, low)
        
        Examples:
        "Edit cell in row 5, column 'Price' to 100" -> {{"action": "edit_cell", "parameters": {{"row": 5, "column": "Price", "value": 100}}, "confidence": "high"}}
        "Create a new column 'Total' by summing 'Price' and 'Tax'" -> {{"action": "create_column", "parameters": {{"column_name": "Total", "source_columns": ["Price", "Tax"], "operation": "sum"}}, "confidence": "high"}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            result = response.choices[0].message.content.strip()
            
            # Clean up response if it has markdown formatting
            if result.startswith('```json'):
                result = result[7:-3].strip()
            elif result.startswith('```'):
                result = result[3:-3].strip()
            
            parsed_result = json.loads(result)
            parsed_result['method'] = 'llm_analysis'
            
            return parsed_result
            
        except Exception as e:
            logging.error(f"LLM analysis error: {e}")
            return {
                "action": "error", 
                "parameters": {"message": str(e)},
                "confidence": "low",
                "method": "error"
            }
    
    def generate_smart_suggestions(self, data_info: Dict) -> List[str]:
        """Generate smart suggestions based on data characteristics"""
        suggestions = []
        columns = data_info.get('columns', [])
        dtypes = data_info.get('dtypes', {})
        
        # Suggest based on data types
        numeric_cols = [col for col, dtype in dtypes.items() if 'int' in str(dtype) or 'float' in str(dtype)]
        text_cols = [col for col, dtype in dtypes.items() if 'object' in str(dtype)]
        
        if numeric_cols:
            suggestions.extend([
                f"Derive features for {', '.join(numeric_cols[:3])}",
                f"Calculate statistics for {numeric_cols[0]}",
                f"Create rolling averages for {numeric_cols[0]}"
            ])
        
        if text_cols:
            suggestions.extend([
                f"Clean text data in {text_cols[0]}",
                f"Create categories from {text_cols[0]}"
            ])
        
        if len(columns) > 1:
            suggestions.append("Create pivot table analysis")
            suggestions.append("Generate correlation matrix")
        
        return suggestions[:5]  # Return top 5 suggestions