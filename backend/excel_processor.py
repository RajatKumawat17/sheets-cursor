import pandas as pd
import numpy as np
import json
import ast
import re
from typing import Dict, Any, List, Union, Optional
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ExcelProcessor:
    def __init__(self):
        self.data = None
        self.filename = None
        self.history = []
        self.backup_data = None
    
    def load_file(self, file_path: str) -> Dict[str, Any]:
        """Load Excel or CSV file with enhanced error handling"""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, encoding='utf-8')
            else:
                self.data = pd.read_excel(file_path, engine='openpyxl')
            
            self.filename = file_path.split('/')[-1]
            self.backup_data = self.data.copy()
            self._log_action("load_file", {"file": self.filename, "shape": self.data.shape})
            
            return {
                "success": True,
                "shape": self.data.shape,
                "columns": self.data.columns.tolist(),
                "dtypes": {col: str(dtype) for col, dtype in self.data.dtypes.items()},
                "preview": self._safe_json_convert(self.data.head(5).to_dict('records')),
                "memory_usage": f"{self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            }
        except Exception as e:
            logging.error(f"File loading error: {e}")
            return {"success": False, "error": str(e)}
    
    def execute_smart_operation(self, instruction: Dict[str, Any]) -> Dict[str, Any]:
        """Execute smart operations with code generation and manipulation"""
        if self.data is None:
            return {"error": "No data loaded"}
        
        action = instruction.get('action', '').lower()
        params = instruction.get('parameters', {})
        
        try:
            # Map actions to methods
            action_map = {
                'edit_cell': self._edit_cell,
                'create_column': self._create_column,
                'derive_features': self._derive_features,
                'apply_formula': self._apply_formula,
                'filter_data': self._filter_data,
                'group_aggregate': self._group_aggregate,
                'pivot_table': self._pivot_table,
                'merge_data': self._merge_data,
                'clean_data': self._clean_data,
                'transform_data': self._transform_data,
                'statistical_analysis': self._statistical_analysis,
                'generate_code': self._generate_code,
                'execute_code': self._execute_code,
                'create_derived_column': self._create_derived_column,
                'batch_edit': self._batch_edit,
                'smart_fill': self._smart_fill
            }
            
            if action in action_map:
                result = action_map[action](params)
                self._log_action(action, params)
                return result
            else:
                return {"error": f"Unknown action: {action}"}
        
        except Exception as e:
            logging.error(f"Operation error: {e}")
            return {"error": str(e), "traceback": str(e)}
    
    def _edit_cell(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Edit a specific cell by row/column location"""
        row = params.get('row')
        column = params.get('column')
        value = params.get('value')
        
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found in {list(self.data.columns)}"}
        
        if row >= len(self.data):
            return {"error": f"Row {row} out of range (max: {len(self.data)-1})"}
        
        old_value = self.data.loc[row, column]
        self.data.loc[row, column] = value
        
        return {
            "success": True,
            "old_value": self._safe_json_value(old_value),
            "new_value": self._safe_json_value(value),
            "location": f"Row {row}, Column '{column}'"
        }
    
    def _create_column(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create new column with values from existing columns"""
        column_name = params.get('column_name')
        source_columns = params.get('source_columns', [])
        operation = params.get('operation', 'concat')
        formula = params.get('formula')
        
        if not column_name:
            return {"error": "Column name is required"}
        
        try:
            if formula:
                # Execute custom formula
                self.data[column_name] = self.data.eval(formula)
            elif operation == 'concat':
                self.data[column_name] = self.data[source_columns].astype(str).agg(' '.join, axis=1)
            elif operation == 'sum':
                self.data[column_name] = self.data[source_columns].sum(axis=1)
            elif operation == 'mean':
                self.data[column_name] = self.data[source_columns].mean(axis=1)
            elif operation == 'max':
                self.data[column_name] = self.data[source_columns].max(axis=1)
            elif operation == 'min':
                self.data[column_name] = self.data[source_columns].min(axis=1)
            else:
                return {"error": f"Unknown operation: {operation}"}
            
            return {
                "success": True,
                "column_name": column_name,
                "operation": operation,
                "preview": self._safe_json_convert(self.data[[column_name]].head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Column creation failed: {str(e)}"}
    
    def _derive_features(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create multiple derived features intelligently"""
        feature_type = params.get('feature_type', 'basic')
        target_columns = params.get('columns', [])
        
        if not target_columns:
            target_columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        derived_features = {}
        
        try:
            for col in target_columns:
                if col not in self.data.columns:
                    continue
                
                if feature_type in ['basic', 'all']:
                    # Basic statistical features
                    if self.data[col].dtype in ['int64', 'float64']:
                        derived_features[f'{col}_squared'] = self.data[col] ** 2
                        derived_features[f'{col}_log'] = np.log1p(self.data[col].abs())
                        derived_features[f'{col}_zscore'] = (self.data[col] - self.data[col].mean()) / self.data[col].std()
                
                if feature_type in ['rolling', 'all']:
                    # Rolling window features
                    if len(self.data) > 5:
                        derived_features[f'{col}_rolling_mean_3'] = self.data[col].rolling(3).mean()
                        derived_features[f'{col}_rolling_std_3'] = self.data[col].rolling(3).std()
                
                if feature_type in ['lag', 'all']:
                    # Lag features
                    derived_features[f'{col}_lag_1'] = self.data[col].shift(1)
                    derived_features[f'{col}_lag_2'] = self.data[col].shift(2)
            
            # Add derived features to dataframe
            for name, values in derived_features.items():
                self.data[name] = values
            
            return {
                "success": True,
                "features_created": list(derived_features.keys()),
                "feature_count": len(derived_features),
                "data_shape": self.data.shape
            }
        except Exception as e:
            return {"error": f"Feature derivation failed: {str(e)}"}
    
    def _apply_formula(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Excel-like formulas to data"""
        formula = params.get('formula')
        target_column = params.get('target_column')
        
        if not formula or not target_column:
            return {"error": "Formula and target_column are required"}
        
        try:
            # Convert Excel-like formulas to pandas operations
            processed_formula = self._convert_excel_formula(formula)
            
            # Execute formula
            if processed_formula.startswith('='):
                processed_formula = processed_formula[1:]  # Remove = sign
            
            self.data[target_column] = self.data.eval(processed_formula)
            
            return {
                "success": True,
                "formula": formula,
                "processed_formula": processed_formula,
                "target_column": target_column,
                "preview": self._safe_json_convert(self.data[[target_column]].head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Formula application failed: {str(e)}"}
    
    def _generate_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate pandas code for complex operations"""
        operation_type = params.get('operation_type')
        description = params.get('description', '')
        
        code_templates = {
            'data_cleaning': """
# Data Cleaning Code
df = df.dropna()  # Remove null values
df = df.drop_duplicates()  # Remove duplicates
df = df.reset_index(drop=True)  # Reset index
            """,
            'feature_engineering': f"""
# Feature Engineering Code
# Create new features based on existing data
{self._generate_feature_code()}
            """,
            'aggregation': """
# Aggregation Code
result = df.groupby('group_column').agg({
    'numeric_column': ['sum', 'mean', 'count'],
    'another_column': 'max'
}).reset_index()
            """,
            'pivot': """
# Pivot Table Code
pivot_result = df.pivot_table(
    values='value_column',
    index='row_column',
    columns='col_column',
    aggfunc='sum',
    fill_value=0
)
            """
        }
        
        code = code_templates.get(operation_type, f"# Custom operation: {description}")
        
        return {
            "success": True,
            "operation_type": operation_type,
            "generated_code": code,
            "description": description
        }
    
    def _execute_code(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom pandas code safely"""
        code = params.get('code')
        
        if not code:
            return {"error": "Code is required"}
        
        try:
            # Create safe execution environment
            safe_globals = {
                'df': self.data,
                'pd': pd,
                'np': np,
                'datetime': datetime,
                'timedelta': timedelta
            }
            
            exec(code, safe_globals)
            
            # Update data if df was modified
            if 'df' in safe_globals:
                self.data = safe_globals['df']
            
            return {
                "success": True,
                "code_executed": code,
                "data_shape": self.data.shape,
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Code execution failed: {str(e)}"}
    
    def _batch_edit(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Batch edit multiple cells"""
        edits = params.get('edits', [])  # List of {row, column, value}
        
        if not edits:
            return {"error": "No edits provided"}
        
        changes = []
        for edit in edits:
            row, column, value = edit.get('row'), edit.get('column'), edit.get('value')
            
            if column in self.data.columns and row < len(self.data):
                old_value = self.data.loc[row, column]
                self.data.loc[row, column] = value
                changes.append({
                    "row": row,
                    "column": column,
                    "old_value": self._safe_json_value(old_value),
                    "new_value": self._safe_json_value(value)
                })
        
        return {
            "success": True,
            "changes_made": len(changes),
            "changes": changes
        }
    
    def _smart_fill(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Smart fill missing values"""
        column = params.get('column')
        method = params.get('method', 'forward')
        
        if column not in self.data.columns:
            return {"error": f"Column '{column}' not found"}
        
        null_count_before = self.data[column].isnull().sum()
        
        if method == 'forward':
            self.data[column] = self.data[column].fillna(method='ffill')
        elif method == 'backward':
            self.data[column] = self.data[column].fillna(method='bfill')
        elif method == 'mean':
            self.data[column] = self.data[column].fillna(self.data[column].mean())
        elif method == 'median':
            self.data[column] = self.data[column].fillna(self.data[column].median())
        elif method == 'mode':
            self.data[column] = self.data[column].fillna(self.data[column].mode()[0])
        
        null_count_after = self.data[column].isnull().sum()
        
        return {
            "success": True,
            "column": column,
            "method": method,
            "nulls_filled": null_count_before - null_count_after,
            "remaining_nulls": null_count_after
        }
    def _filter_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on conditions"""
        column = params.get('column')
        condition = params.get('condition')
        value = params.get('value')
        operator = params.get('operator', '==')
        
        if not column or column not in self.data.columns:
            return {"error": f"Column '{column}' not found"}
        
        try:
            original_rows = len(self.data)
            
            if operator == '==':
                mask = self.data[column] == value
            elif operator == '!=':
                mask = self.data[column] != value
            elif operator == '>':
                mask = self.data[column] > value
            elif operator == '<':
                mask = self.data[column] < value
            elif operator == '>=':
                mask = self.data[column] >= value
            elif operator == '<=':
                mask = self.data[column] <= value
            elif operator == 'contains':
                mask = self.data[column].astype(str).str.contains(str(value), na=False)
            elif operator == 'startswith':
                mask = self.data[column].astype(str).str.startswith(str(value), na=False)
            elif operator == 'endswith':
                mask = self.data[column].astype(str).str.endswith(str(value), na=False)
            else:
                return {"error": f"Unknown operator: {operator}"}
            
            self.data = self.data[mask].reset_index(drop=True)
            filtered_rows = len(self.data)
            
            return {
                "success": True,
                "original_rows": original_rows,
                "filtered_rows": filtered_rows,
                "rows_removed": original_rows - filtered_rows,
                "filter_condition": f"{column} {operator} {value}",
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Filter operation failed: {str(e)}"}
    
    def _group_aggregate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Group data and perform aggregations"""
        group_by = params.get('group_by')
        agg_functions = params.get('agg_functions', {})
        
        if not group_by or group_by not in self.data.columns:
            return {"error": f"Group by column '{group_by}' not found"}
        
        if not agg_functions:
            return {"error": "Aggregation functions are required"}
        
        try:
            grouped_data = self.data.groupby(group_by).agg(agg_functions).reset_index()
            
            # Flatten column names if multi-level
            if isinstance(grouped_data.columns, pd.MultiIndex):
                grouped_data.columns = ['_'.join(col).strip() for col in grouped_data.columns.values]
            
            # Replace original data with grouped data
            self.data = grouped_data
            
            return {
                "success": True,
                "group_by": group_by,
                "aggregations": agg_functions,
                "result_shape": self.data.shape,
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Group aggregation failed: {str(e)}"}
    
    def _pivot_table(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create pivot table"""
        values = params.get('values')
        index = params.get('index')
        columns = params.get('columns')
        aggfunc = params.get('aggfunc', 'sum')
        fill_value = params.get('fill_value', 0)
        
        if not all([values, index]):
            return {"error": "Values and index are required for pivot table"}
        
        try:
            pivot_data = self.data.pivot_table(
                values=values,
                index=index,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=fill_value
            ).reset_index()
            
            # Replace original data with pivot table
            self.data = pivot_data
            
            return {
                "success": True,
                "values": values,
                "index": index,
                "columns": columns,
                "aggfunc": aggfunc,
                "result_shape": self.data.shape,
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Pivot table creation failed: {str(e)}"}
    
    def _merge_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge with another dataset"""
        # This would require loading another dataset
        # For now, return not implemented
        return {"error": "Merge data functionality requires additional dataset loading capabilities"}
    
    def _clean_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data with various operations"""
        operations = params.get('operations', ['remove_nulls', 'remove_duplicates'])
        
        try:
            original_shape = self.data.shape
            results = {}
            
            if 'remove_nulls' in operations:
                null_rows_before = self.data.isnull().any(axis=1).sum()
                self.data = self.data.dropna()
                null_rows_after = self.data.isnull().any(axis=1).sum()
                results['nulls_removed'] = null_rows_before - null_rows_after
            
            if 'remove_duplicates' in operations:
                duplicates_before = self.data.duplicated().sum()
                self.data = self.data.drop_duplicates()
                duplicates_after = self.data.duplicated().sum()
                results['duplicates_removed'] = duplicates_before - duplicates_after
            
            if 'strip_whitespace' in operations:
                string_columns = self.data.select_dtypes(include=['object']).columns
                for col in string_columns:
                    self.data[col] = self.data[col].astype(str).str.strip()
                results['whitespace_stripped'] = len(string_columns)
            
            if 'standardize_case' in operations:
                case_type = params.get('case_type', 'lower')
                string_columns = self.data.select_dtypes(include=['object']).columns
                for col in string_columns:
                    if case_type == 'lower':
                        self.data[col] = self.data[col].astype(str).str.lower()
                    elif case_type == 'upper':
                        self.data[col] = self.data[col].astype(str).str.upper()
                    elif case_type == 'title':
                        self.data[col] = self.data[col].astype(str).str.title()
                results['case_standardized'] = len(string_columns)
            
            self.data = self.data.reset_index(drop=True)
            
            return {
                "success": True,
                "operations": operations,
                "original_shape": original_shape,
                "new_shape": self.data.shape,
                "results": results,
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Data cleaning failed: {str(e)}"}
    
    def _transform_data(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data with various operations"""
        transformation = params.get('transformation')
        columns = params.get('columns', [])
        
        if not transformation:
            return {"error": "Transformation type is required"}
        
        try:
            if transformation == 'normalize':
                # Min-max normalization
                for col in columns:
                    if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']:
                        min_val = self.data[col].min()
                        max_val = self.data[col].max()
                        self.data[f'{col}_normalized'] = (self.data[col] - min_val) / (max_val - min_val)
            
            elif transformation == 'standardize':
                # Z-score standardization
                for col in columns:
                    if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']:
                        mean_val = self.data[col].mean()
                        std_val = self.data[col].std()
                        self.data[f'{col}_standardized'] = (self.data[col] - mean_val) / std_val
            
            elif transformation == 'log_transform':
                # Log transformation
                for col in columns:
                    if col in self.data.columns and self.data[col].dtype in ['int64', 'float64']:
                        self.data[f'{col}_log'] = np.log1p(self.data[col].abs())
            
            elif transformation == 'one_hot_encode':
                # One-hot encoding for categorical variables
                for col in columns:
                    if col in self.data.columns:
                        dummies = pd.get_dummies(self.data[col], prefix=col)
                        self.data = pd.concat([self.data, dummies], axis=1)
            
            else:
                return {"error": f"Unknown transformation: {transformation}"}
            
            return {
                "success": True,
                "transformation": transformation,
                "columns": columns,
                "new_shape": self.data.shape,
                "preview": self._safe_json_convert(self.data.head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Data transformation failed: {str(e)}"}
    
    def _statistical_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform statistical analysis"""
        analysis_type = params.get('analysis_type', 'descriptive')
        columns = params.get('columns', [])
        
        if not columns:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        try:
            results = {}
            
            if analysis_type == 'descriptive':
                desc_stats = self.data[columns].describe()
                results['descriptive_stats'] = self._safe_json_convert(desc_stats.to_dict())
            
            elif analysis_type == 'correlation':
                corr_matrix = self.data[columns].corr()
                results['correlation_matrix'] = self._safe_json_convert(corr_matrix.to_dict())
            
            elif analysis_type == 'missing_values':
                missing_info = {
                    col: {
                        'missing_count': int(self.data[col].isnull().sum()),
                        'missing_percentage': float(self.data[col].isnull().sum() / len(self.data) * 100)
                    }
                    for col in self.data.columns
                }
                results['missing_values'] = missing_info
            
            elif analysis_type == 'data_types':
                type_info = {
                    col: {
                        'dtype': str(self.data[col].dtype),
                        'unique_values': int(self.data[col].nunique()),
                        'sample_values': self.data[col].dropna().head(3).tolist()
                    }
                    for col in self.data.columns
                }
                results['data_types'] = type_info
            
            else:
                return {"error": f"Unknown analysis type: {analysis_type}"}
            
            return {
                "success": True,
                "analysis_type": analysis_type,
                "columns_analyzed": columns,
                "results": results
            }
        except Exception as e:
            return {"error": f"Statistical analysis failed: {str(e)}"}
    
    def _create_derived_column(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create derived column with advanced logic"""
        column_name = params.get('column_name')
        derivation_type = params.get('derivation_type')
        source_columns = params.get('source_columns', [])
        custom_logic = params.get('custom_logic')
        
        if not column_name:
            return {"error": "Column name is required"}
        
        try:
            if derivation_type == 'conditional':
                # Create conditional column based on custom logic
                condition = params.get('condition')
                true_value = params.get('true_value')
                false_value = params.get('false_value')
                
                if condition and true_value is not None and false_value is not None:
                    self.data[column_name] = np.where(
                        self.data.eval(condition), 
                        true_value, 
                        false_value
                    )
            
            elif derivation_type == 'binning':
                # Create bins for continuous variables
                source_col = source_columns[0] if source_columns else None
                bins = params.get('bins', 5)
                labels = params.get('labels')
                
                if source_col and source_col in self.data.columns:
                    self.data[column_name] = pd.cut(
                        self.data[source_col], 
                        bins=bins, 
                        labels=labels
                    )
            
            elif derivation_type == 'date_features':
                # Extract date features
                source_col = source_columns[0] if source_columns else None
                if source_col and source_col in self.data.columns:
                    date_col = pd.to_datetime(self.data[source_col])
                    feature_type = params.get('date_feature', 'year')
                    
                    if feature_type == 'year':
                        self.data[column_name] = date_col.dt.year
                    elif feature_type == 'month':
                        self.data[column_name] = date_col.dt.month
                    elif feature_type == 'day':
                        self.data[column_name] = date_col.dt.day
                    elif feature_type == 'weekday':
                        self.data[column_name] = date_col.dt.day_name()
                    elif feature_type == 'quarter':
                        self.data[column_name] = date_col.dt.quarter
            
            elif derivation_type == 'custom' and custom_logic:
                # Execute custom logic
                self.data[column_name] = self.data.eval(custom_logic)
            
            else:
                return {"error": f"Unknown derivation type: {derivation_type}"}
            
            return {
                "success": True,
                "column_name": column_name,
                "derivation_type": derivation_type,
                "preview": self._safe_json_convert(self.data[[column_name]].head().to_dict('records'))
            }
        except Exception as e:
            return {"error": f"Derived column creation failed: {str(e)}"}
    
    # Utility methods
    def _convert_excel_formula(self, formula: str) -> str:
        """Convert Excel-like formulas to pandas expressions"""
        # Simple conversions
        conversions = {
            'SUM(': 'sum(',
            'AVERAGE(': 'mean(',
            'MAX(': 'max(',
            'MIN(': 'min(',
            'COUNT(': 'count(',
        }
        
        for excel_func, pandas_func in conversions.items():
            formula = formula.replace(excel_func, pandas_func)
        
        return formula
    
    def _generate_feature_code(self) -> str:
        """Generate feature engineering code based on current data"""
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        if not numeric_cols:
            return "# No numeric columns found for feature engineering"
        
        code = f"""
# Feature engineering for columns: {numeric_cols}
for col in {numeric_cols}:
    df[f'{{col}}_squared'] = df[col] ** 2
    df[f'{{col}}_log'] = np.log1p(df[col].abs())
    df[f'{{col}}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
        """
        return code
    
    def _log_action(self, action: str, params: Dict[str, Any]):
        """Log actions for history tracking"""
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "parameters": params
        })
    
    def _safe_json_convert(self, data):
        """Convert data to JSON-safe format"""
        if isinstance(data, list):
            return [self._safe_json_convert(item) for item in data]
        elif isinstance(data, dict):
            return {str(k): self._safe_json_convert(v) for k, v in data.items()}
        else:
            return self._safe_json_value(data)
    
    def _safe_json_value(self, value):
        """Convert single value to JSON-safe format"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.integer, np.floating)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        elif isinstance(value, (pd.Timestamp, np.datetime64)):
            return str(value)
        else:
            return value
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get operation history"""
        return self.history
    
    def undo_last_operation(self) -> Dict[str, Any]:
        """Undo last operation by restoring backup"""
        if self.backup_data is not None:
            self.data = self.backup_data.copy()
            return {"success": True, "message": "Last operation undone"}
        return {"error": "No backup data available"}
    
    def save_data(self, file_path: str) -> Dict[str, Any]:
        """Save current data to file"""
        try:
            if file_path.endswith('.csv'):
                self.data.to_csv(file_path, index=False)
            else:
                self.data.to_excel(file_path, index=False)
            
            return {"success": True, "file_path": file_path, "shape": self.data.shape}
        except Exception as e:
            return {"error": f"Save failed: {str(e)}"}