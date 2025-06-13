from openai import AzureOpenAI
import json
import logging

from dotenv import load_dotenv
import os

load_dotenv()
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

class LLMClient:
    def __init__(self):
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY
        )
    
    def analyze_command(self, command: str, data_info: dict) -> dict:
        """Convert natural language command to data analysis instructions"""
        
        system_prompt = f"""
        You are a data analysis assistant. Convert natural language commands into structured analysis instructions.
        
        Available data columns: {data_info.get('columns', [])}
        Data shape: {data_info.get('shape', 'Unknown')}
        
        Return a JSON response with:
        - action: type of analysis (filter, group, calculate, chart, etc.)
        - parameters: specific parameters for the action
        - chart_type: if visualization is needed (bar, line, pie, etc.)
        
        Examples:
        "Show revenue by quarter" -> {{"action": "group", "parameters": {{"group_by": "quarter", "aggregate": "sum", "column": "revenue"}}, "chart_type": "bar"}}
        "Filter high revenue products" -> {{"action": "filter", "parameters": {{"column": "revenue", "condition": ">", "value": "mean"}}}}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1",  # Adjust model name to match your Azure deployment
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": command}
                ],
                temperature=0.1,
                max_tokens=2000
            )
            
            result = response.choices[0].message.content
            return json.loads(result)
        except Exception as e:
            logging.error(f"LLM analysis error: {e}")
            return {"action": "error", "message": str(e)}