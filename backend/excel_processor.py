import pandas as pd
import json
import numpy as np
from typing import Dict, Any, List
import logging

class ExcelProcessor:
    def __init__(self):
        self.data = None
        self.filename = None
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load Excel or CSV file"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            else:
                self.data = pd.read_excel(file_path)
            
            self.filename = file_path.split('/')[-1]
            
            # Convert dtypes to string representation for JSON serialization
            dtypes_dict = {col: str(dtype) for col, dtype in self.data.dtypes.items()}
            
            return {
                "success": True,
                "shape": self.data.shape,
                "columns": self.data.columns.tolist(),
                "dtypes": dtypes_dict,
                "preview": self.data.head().to_dict('records')
            }
        except Exception as e:
            logging.error(f"File loading error: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_analysis(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis based on LLM instruction"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        action = instruction.get('action')
        params = instruction.get('parameters', {})
        
        try:
            if action == 'group':
                result = self._group_analysis(params)
            elif action == 'filter':
                result = self._filter_analysis(params)
            elif action == 'calculate':
                result = self._calculate_analysis(params)
            elif action == 'chart':
                result = self._chart_data(params)
            else:
                result = {"error": f"Unknown action: {action}"}
            
            # Add chart type if specified
            if 'chart_type' in instruction:
                result['chart_type'] = instruction['chart_type']
            
            return result
        except Exception as e:
            logging.error(f"Analysis error: {e}")
            return {"error": str(e)}
    
    def _group_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Group by analysis"""
        group_by = params.get('group_by')
        aggregate = params.get('aggregate', 'sum')
        column = params.get('column')
        
        if group_by not in self.data.columns:
            return {"error": f"Column '{group_by}' not found"}
        
        if column and column not in self.data.columns:
            return {"error": f"Column '{column}' not found"}
        
        grouped = self.data.groupby(group_by)
        
        if column:
            if aggregate == 'sum':
                result = grouped[column].sum()
            elif aggregate == 'mean':
                result = grouped[column].mean()
            elif aggregate == 'count':
                result = grouped[column].count()
            else:
                result = grouped[column].sum()
        else:
            result = grouped.size()
        
        # Convert numpy types to Python native types for JSON serialization
        return {
            "data": {str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v 
                    for k, v in result.to_dict().items()},
            "labels": [str(label) for label in result.index.tolist()],
            "values": [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                      for v in result.values.tolist()]
        }
    
    def _filter_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data analysis"""
        column = params.get('column')
        condition = params.get('condition', '>')
        value = params.get('value')
        
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found"}
        
        if value == 'mean':
            value = self.data[column].mean()
        elif value == 'median':
            value = self.data[column].median()
        
        if condition == '>':
            filtered_data = self.data[self.data[column] > value]
        elif condition == '<':
            filtered_data = self.data[self.data[column] < value]
        elif condition == '==':
            filtered_data = self.data[self.data[column] == value]
        else:
            filtered_data = self.data
        
        # Convert to JSON-serializable format
        records = self._convert_records_to_json_serializable(filtered_data.to_dict('records'))
        preview = self._convert_records_to_json_serializable(filtered_data.head(10).to_dict('records'))
        
        return {
            "data": records,
            "count": len(filtered_data),
            "preview": preview
        }
    
    def _calculate_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics"""
        column = params.get('column')
        operation = params.get('operation', 'describe')
        
        if column and column not in self.data.columns:
            return {"error": f"Column '{column}' not found"}
        
        if operation == 'describe':
            if column:
                result = self.data[column].describe()
            else:
                result = self.data.describe()
            
            # Convert to JSON-serializable format
            if hasattr(result, 'to_dict'):
                stats = self._convert_dict_to_json_serializable(result.to_dict())
            else:
                stats = result
                
        elif operation == 'correlation':
            result = self.data.corr()
            stats = self._convert_dict_to_json_serializable(result.to_dict())
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        return {"statistics": stats}
    
    def _chart_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for charts"""
        x_column = params.get('x_column')
        y_column = params.get('y_column')
        
        if x_column not in self.data.columns:
            return {"error": f"Column '{x_column}' not found"}
        
        if y_column not in self.data.columns:
            return {"error": f"Column '{y_column}' not found"}
        
        # Convert to JSON-serializable format
        labels = [str(x) for x in self.data[x_column].tolist()]
        values = [float(v) if isinstance(v, (np.integer, np.floating)) else v 
                 for v in self.data[y_column].tolist()]
        
        return {
            "labels": labels,
            "values": values
        }
    
    def _convert_records_to_json_serializable(self, records: List[Dict]) -> List[Dict]:
        """Convert pandas records to JSON-serializable format"""
        json_records = []
        for record in records:
            json_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    json_record[key] = None
                elif isinstance(value, (np.integer, np.floating)):
                    json_record[key] = float(value)
                elif isinstance(value, np.bool_):
                    json_record[key] = bool(value)
                elif isinstance(value, (pd.Timestamp, np.datetime64)):
                    json_record[key] = str(value)
                else:
                    json_record[key] = value
            json_records.append(json_record)
        return json_records
    
    def _convert_dict_to_json_serializable(self, data: Dict) -> Dict:
        """Convert dictionary with numpy types to JSON-serializable format"""
        json_dict = {}
        for key, value in data.items():
            # Convert key to string if it's not already
            str_key = str(key)
            
            if isinstance(value, dict):
                json_dict[str_key] = self._convert_dict_to_json_serializable(value)
            elif pd.isna(value):
                json_dict[str_key] = None
            elif isinstance(value, (np.integer, np.floating)):
                json_dict[str_key] = float(value)
            elif isinstance(value, np.bool_):
                json_dict[str_key] = bool(value)
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                json_dict[str_key] = str(value)
            else:
                json_dict[str_key] = value
                
        return json_dict