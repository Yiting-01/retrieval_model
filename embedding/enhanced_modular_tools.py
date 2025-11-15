from __future__ import annotations
from enum import Enum
from typing import Literal, Dict, Optional, List, Union
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, validator
import pandas as pd
import re
from collections import deque

# Global variables for DataFrame history (undo functionality)
_df_history = deque(maxlen=10)  # Keep last 10 states
_last_operation = None

# Helper functions for undo functionality
def _save_df_state(operation_name: str):
    """Save current df state before modification."""
    global _df_history, _last_operation
    # Save current df state (assuming df is available in global scope)
    _df_history.append(df.copy())
    _last_operation = operation_name

def _restore_df_state():
    """Restore the previous df state."""
    global _df_history, _last_operation
    if _df_history:
        return _df_history.pop(), _last_operation
    return None, None

# Code template for helper functions to include in generated code
HELPER_FUNCTIONS_CODE = """
# Helper functions for DataFrame history (undo functionality)
from collections import deque
import pandas as pd

# Initialize global variables if they don't exist
if '_df_history' not in globals():
    _df_history = deque(maxlen=10)
if '_last_operation' not in globals():
    _last_operation = None

def _save_df_state(operation_name: str):
    \"\"\"Save current df state before modification.\"\"\"
    global _df_history, _last_operation, df
    try:
        _df_history.append(df.copy())
        _last_operation = operation_name
    except Exception as e:
        print(f"Warning: Could not save state: {e}")

def _restore_df_state():
    \"\"\"Restore the previous df state.\"\"\"
    global _df_history, _last_operation
    try:
        if _df_history:
            return _df_history.pop(), _last_operation
        return None, None
    except Exception as e:
        print(f"Warning: Could not restore state: {e}")
        return None, None
"""


# ===== EXISTING TOOLS =====

class SelectColumnInput(BaseModel):
    """Schema for selecting subset of columns."""
    
    column_names: list[str] = Field(..., description="List of column names to select")
    reasoning: str = Field(..., description="Explanation of why these columns are selected")


class GroupByInput(BaseModel):
    """Schema for grouping rows by column contents."""
    
    column_name: str = Field(..., description="Column name to group by")
    show_counts: bool = Field(True, description="Whether to show count of each group")
    modify_df: bool = Field(False, description="Whether to replace the current table with the group statistics for further analysis")


class PrintTableInput(BaseModel):
    """Schema for printing the current table state."""
    
    max_rows: int = Field(10, description="Maximum number of rows to display")
    show_all: bool = Field(False, description="Whether to show all rows regardless of max_rows")


class SelectColumnTool(BaseTool):
    """Tool to select a subset of columns relevant to the question."""
    
    name: str = "f_select_column"
    description: str = "Select a subset of columns. A column usually corresponds to an attribute in the table. This operation helps locate necessary attributes to answer the question."
    response_format: Literal["text"] = "text"
    args_schema: type[SelectColumnInput] = SelectColumnInput
    
    def _run(self, column_names: list[str], reasoning: str, **kwargs) -> str:
        return f"""
{HELPER_FUNCTIONS_CODE}

# Select specific columns from the current table
# Reasoning: {reasoning}
print(f"🔧 Selecting columns: {column_names}")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_select_column")

# Check if all columns exist
available_columns = df.columns.tolist()
missing_columns = [col for col in {column_names} if col not in available_columns]

if missing_columns:
    print(f"❌ Error: Columns not found: {{missing_columns}}")
    print(f"Available columns: {{available_columns}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    # Select the specified columns
    df = df[{column_names}].copy()
    print(f"✅ Successfully selected columns")
    print(f"📊 New table shape: {{df.shape}}")
    print(f"🔍 Selected columns preview:")
    print(df.head())
"""
    
    async def _arun(self, column_names: list[str], reasoning: str, **kwargs) -> str:
        return self._run(column_names, reasoning, **kwargs)


class GroupByTool(BaseTool):
    """Tool to group rows by column contents and show counts."""
    
    name: str = "f_group_by"
    description: str = "Group the rows by the contents of a specific column and provide the count of each enumeration value in that column. Can optionally replace the current table with the group statistics using modify_df=True for further analysis (e.g., sorting by counts)."
    response_format: Literal["text"] = "text"
    args_schema: type[GroupByInput] = GroupByInput
    
    def _run(self, column_name: str, show_counts: bool = True, modify_df: bool = False, **kwargs) -> str:
        return f"""
# Group rows by column '{column_name}'
print(f"🔧 Grouping by column '{column_name}'")
print(f"💭 Show counts: {show_counts}, Modify table: {modify_df}")

# Check if column exists
if '{column_name}' not in df.columns:
    print(f"❌ Error: Column '{column_name}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
else:
    # Group by the specified column
    grouped = df.groupby('{column_name}')
    
    print(f"✅ Successfully grouped by '{column_name}'")
    print(f"📊 Number of groups: {{len(grouped)}}")
    
    if {show_counts}:
        print(f"🔍 Group counts:")
        group_counts = df['{column_name}'].value_counts().sort_index()
        print(group_counts)
        
        # Also show group statistics if there are numeric columns
        numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns:
            print(f"\\n📈 Group statistics for numeric columns:")
            group_stats = df.groupby('{column_name}')[numeric_columns].agg(['count', 'mean', 'std']).round(2)
            print(group_stats)
    
    # Optionally modify the dataframe
    if {modify_df}:
        print(f"\\n🔄 Replacing current table with group statistics...")
        
        # Save current state before modification
        _save_df_state("f_group_by")
        
        try:
            # Create result DataFrame with group counts
            group_counts_df = df['{column_name}'].value_counts().sort_index().reset_index()
            group_counts_df.columns = ['{column_name}', 'count']
            
            # If there are numeric columns, also include their statistics
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            if numeric_columns and len(numeric_columns) > 0:
                print(f"📊 Including statistics for numeric columns: {{numeric_columns}}")
                group_stats = df.groupby('{column_name}')[numeric_columns].agg(['mean', 'std']).round(2)
                group_stats.columns = ['_'.join(col).strip() for col in group_stats.columns.values]
                group_stats = group_stats.reset_index()
                
                # Merge counts with statistics
                result_df = pd.merge(group_counts_df, group_stats, on='{column_name}')
            else:
                result_df = group_counts_df
            
            df = result_df
            print(f"✅ Table replaced with group statistics")
            print(f"📊 New table shape: {{df.shape}}")
            print(f"🔍 New table preview:")
            print(df.head())
            
        except Exception as e:
            print(f"❌ Error replacing table: {{e}}")
            # Restore state since operation failed
            _df_history.pop() if _df_history else None
    else:
        print(f"\\n💡 Use modify_df=True to replace current table with group statistics for further analysis")
"""
    
    async def _arun(self, column_name: str, show_counts: bool = True, modify_df: bool = False, **kwargs) -> str:
        return self._run(column_name, show_counts, modify_df, **kwargs)


class PrintTableTool(BaseTool):
    """Tool to print all rows of the current table state."""
    
    name: str = "print_table"
    description: str = "Print all rows of the selected/transformed table to see the current state and potentially provide final answer."
    response_format: Literal["text"] = "text"
    args_schema: type[PrintTableInput] = PrintTableInput
    
    def _run(self, max_rows: int = 10, show_all: bool = False, **kwargs) -> str:
        return f"""
# Print current table state
print(f"📋 Current Table State:")
print(f"📊 Shape: {{df.shape}}")
print(f"🔍 Columns: {{df.columns.tolist()}}")
print("-" * 60)

if {show_all} or len(df) <= {max_rows}:
    print("📄 All rows:")
    print(df.to_string())
else:
    print(f"📄 First {max_rows} rows:")
    print(df.head({max_rows}).to_string())
    if len(df) > {max_rows}:
        print(f"... and {{len(df) - {max_rows}}} more rows")

print("-" * 60)
print(f"✅ Table display complete")
"""
    
    async def _arun(self, max_rows: int = 10, show_all: bool = False, **kwargs) -> str:
        return self._run(max_rows, show_all, **kwargs)


# ===== NEW TOOLS - EASY TO ADD! =====

class CalculateAverageInput(BaseModel):
    """Schema for calculating average of a column by groups."""
    
    column_name: str = Field(..., description="Column name to calculate average for")
    group_by_column: str = Field(..., description="Column to group by before calculating average")
    modify_df: bool = Field(False, description="Whether to replace the current table with the grouped average results for further analysis")


class CalculateAverageTool(BaseTool):
    """Tool to calculate average of a column grouped by another column."""
    
    name: str = "f_calculate_average"
    description: str = "Calculate the average (mean) of a numeric column, grouped by another column. Perfect for finding which group has the highest/lowest average. Can optionally replace the current table with the grouped results using modify_df=True for further analysis (e.g., sorting by averages)."
    response_format: Literal["text"] = "text"
    args_schema: type[CalculateAverageInput] = CalculateAverageInput
    
    def _run(self, column_name: str, group_by_column: str, modify_df: bool = False, **kwargs) -> str:
        return f"""
# Calculate average of '{column_name}' grouped by '{group_by_column}'
print(f"🔧 Calculating average of '{column_name}' by '{group_by_column}'")
print(f"💭 Modify table: {modify_df}")

# Check if columns exist
if '{column_name}' not in df.columns:
    print(f"❌ Error: Column '{column_name}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
elif '{group_by_column}' not in df.columns:
    print(f"❌ Error: Column '{group_by_column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
else:
    # Calculate average by group
    avg_by_group = df.groupby('{group_by_column}')['{column_name}'].mean().round(2)
    
    print(f"✅ Average {column_name} by {group_by_column}:")
    print(avg_by_group)
    
    # Find the group with highest average
    highest_avg_group = avg_by_group.idxmax()
    highest_avg_value = avg_by_group.max()
    
    print(f"\\n🏆 ANSWER: The {group_by_column} with highest average {column_name} is:")
    print(f"   {group_by_column}: {{highest_avg_group}}")
    print(f"   Average {column_name}: {{highest_avg_value}}")
    
    # Optionally modify the dataframe
    if {modify_df}:
        print(f"\\n🔄 Replacing current table with grouped averages...")
        
        # Save current state before modification
        _save_df_state("f_calculate_average")
        
        try:
            # Convert series to dataframe and reset index
            df = avg_by_group.reset_index()
            df.columns = ['{group_by_column}', f'avg_{column_name}']
            
            print(f"✅ Table replaced with grouped averages")
            print(f"📊 New table shape: {{df.shape}}")
            print(f"🔍 New table preview:")
            print(df.head())
            
        except Exception as e:
            print(f"❌ Error replacing table: {{e}}")
            # Restore state since operation failed
            _df_history.pop() if _df_history else None
    else:
        print(f"\\n💡 Use modify_df=True to replace current table with these results for further analysis")
"""
    
    async def _arun(self, column_name: str, group_by_column: str, modify_df: bool = False, **kwargs) -> str:
        return self._run(column_name, group_by_column, modify_df, **kwargs)


class FilterRowsInput(BaseModel):
    """Schema for filtering rows based on conditions."""
    
    condition: str = Field(..., description="Pandas condition to filter rows (e.g., 'Age > 18', 'Sex == \"male\"')")
    reasoning: str = Field(..., description="Explanation of why this filtering is needed")


class FilterRowsTool(BaseTool):
    """Tool to filter rows based on conditions."""
    
    name: str = "f_filter_rows"
    description: str = "Filter rows based on specified conditions. Useful for removing missing data or focusing on specific subsets."
    response_format: Literal["text"] = "text"
    args_schema: type[FilterRowsInput] = FilterRowsInput
    
    def _run(self, condition: str, reasoning: str, **kwargs) -> str:
        return f"""
{HELPER_FUNCTIONS_CODE}

# Filter rows based on condition: {condition}
# Reasoning: {reasoning}
print(f"🔧 Filtering rows with condition: {condition}")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_filter_rows")

try:
    # Store the condition in a variable to avoid f-string quote conflicts
    filter_condition = {repr(condition)}
    print(f"Using condition: {{filter_condition}}")
    
    # Apply the filter using query method for safer condition handling
    original_shape = df.shape
    df = df.query(filter_condition).copy().reset_index(drop=True)
    new_shape = df.shape

    print(f"✅ Filtering completed")
    print(f"📊 Original shape: {{original_shape}}")
    print(f"📊 New shape: {{new_shape}}")
    print(f"📊 Removed {{original_shape[0] - new_shape[0]}} rows")
    print(f"🔍 Filtered data preview:")
    print(df.head())
except Exception as e:
    print(f"❌ Error applying condition: {{e}}")
    print(f"Available columns: {{df.columns.tolist()}}")
    print("Condition format example: Age > 18  or  Sex == 'male'")
    
    # Try alternative approach using boolean indexing if query fails
    try:
        print("🔄 Trying alternative boolean indexing approach...")
        
        # For simple equality conditions, try boolean indexing
        if "==" in filter_condition and ("'" in filter_condition or '"' in filter_condition):
            # Extract column and value for simple equality
            parts = filter_condition.split("==")
            if len(parts) == 2:
                col_name = parts[0].strip()
                value = parts[1].strip().strip("'\\\"")
                print(f"Trying boolean indexing: df['{{col_name}}'] == '{{value}}'")
                
                if col_name in df.columns:
                    original_shape = df.shape
                    df = df[df[col_name] == value].copy().reset_index(drop=True)
                    new_shape = df.shape
                    
                    print(f"✅ Alternative filtering completed")
                    print(f"📊 Original shape: {{original_shape}}")
                    print(f"📊 New shape: {{new_shape}}")
                    print(f"📊 Removed {{original_shape[0] - new_shape[0]}} rows")
                    print(f"🔍 Filtered data preview:")
                    print(df.head())
                else:
                    print(f"❌ Column '{{col_name}}' not found in dataframe")
            else:
                print("❌ Could not parse condition for boolean indexing")
        else:
            print("❌ Alternative approach not applicable for this condition type")
            # Restore state since alternative approach failed
            _df_history.pop() if _df_history else None
            
    except Exception as e2:
        print(f"❌ Alternative approach also failed: {{e2}}")
        print("Please check the condition syntax and column names")
        # Restore state since alternative approach failed
        _df_history.pop() if _df_history else None
"""
    
    async def _arun(self, condition: str, reasoning: str, **kwargs) -> str:
        return self._run(condition, reasoning, **kwargs)


class GetDataInfoInput(BaseModel):
    """Schema for getting basic information about the dataset."""
    
    show_missing: bool = Field(True, description="Whether to show missing value counts")
    show_dtypes: bool = Field(True, description="Whether to show data types")


class GetDataInfoTool(BaseTool):
    """Tool to get basic information about the dataset."""
    
    name: str = "f_get_data_info"
    description: str = "Get basic information about the dataset including shape, columns, data types, and missing values."
    response_format: Literal["text"] = "text"
    args_schema: type[GetDataInfoInput] = GetDataInfoInput
    
    def _run(self, show_missing: bool = True, show_dtypes: bool = True, **kwargs) -> str:
        return f"""
# Get basic dataset information
print(f"📊 Dataset Information:")
print(f"Shape: {{df.shape}}")
print(f"Columns: {{df.columns.tolist()}}")
print()

if {show_dtypes}:
    print(f"🔍 Data Types:")
    print(df.dtypes)
    print()

if {show_missing}:
    print(f"❓ Missing Values:")
    missing_counts = df.isnull().sum()
    print(missing_counts[missing_counts > 0])
    print()

print(f"🔍 First 5 rows:")
print(df.head())
"""
    
    async def _arun(self, show_missing: bool = True, show_dtypes: bool = True, **kwargs) -> str:
        return self._run(show_missing, show_dtypes, **kwargs)


# ===== ADDITIONAL TOOLS FROM ENHANCED_COT_TOOLS =====

class SelectRowInput(BaseModel):
    """Schema for selecting rows by indices or conditions."""
    
    indices: list[int] | None = Field(None, description="Explicit row indices to keep (0-based)")
    condition: str | None = Field(None, description="Single condition string (e.g., 'Age >= 30')")
    conditions: list[str] | None = Field(None, description="Multiple AND-connected conditions")
    reasoning: str = Field(..., description="Explanation of why this row selection is needed")


class SelectRowTool(BaseTool):
    """Tool to select rows by indices or conditions."""
    
    name: str = "f_select_row"
    description: str = "Return a row-filtered copy of the table. Can filter by explicit indices, single condition, or multiple AND-connected conditions."
    response_format: Literal["text"] = "text"
    args_schema: type[SelectRowInput] = SelectRowInput
    
    def _run(self, indices: list[int] | None = None, condition: str | None = None, 
             conditions: list[str] | None = None, reasoning: str = "", **kwargs) -> str:
        
        # Count non-None parameters
        params_provided = sum(x is not None for x in [indices, condition, conditions])
        
        if params_provided != 1:
            return f"""
# Error: Exactly one of indices, condition, or conditions must be provided
print(f"❌ Error: Exactly one parameter must be provided")
print(f"   indices: {indices}")
print(f"   condition: {condition}")
print(f"   conditions: {conditions}")
"""
        
        if indices is not None:
            return f"""
# Select rows by explicit indices: {indices}
# Reasoning: {reasoning}
print(f"🔧 Selecting rows by indices: {indices}")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_select_row")

# Check if indices are valid
max_index = len(df) - 1
invalid_indices = [i for i in {indices} if i < 0 or i > max_index]

if invalid_indices:
    print(f"❌ Error: Invalid indices {{invalid_indices}} (valid range: 0-{{max_index}})")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    # Select rows by indices
    original_shape = df.shape
    df = df.iloc[{indices}].copy().reset_index(drop=True)
    new_shape = df.shape
    
    print(f"✅ Row selection completed")
    print(f"📊 Original shape: {{original_shape}}")
    print(f"📊 New shape: {{new_shape}}")
    print(f"🔍 Selected rows preview:")
    print(df.head())
"""
        
        elif condition is not None:
            return f"""
# Select rows by single condition: {condition}
# Reasoning: {reasoning}
print(f"🔧 Filtering rows with condition: {condition}")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_select_row")

try:
    # Store the condition in a variable to avoid f-string quote conflicts
    filter_condition = {repr(condition)}
    print(f"Using condition: {{filter_condition}}")
    
    # Apply the condition using query method for safer condition handling
    original_shape = df.shape
    df = df.query(filter_condition).copy().reset_index(drop=True)
    new_shape = df.shape
    
    print(f"✅ Row filtering completed")
    print(f"📊 Original shape: {{original_shape}}")
    print(f"📊 New shape: {{new_shape}}")
    print(f"📊 Selected {{new_shape[0]}} rows")
    print(f"🔍 Filtered data preview:")
    print(df.head())
except Exception as e:
    print(f"❌ Error applying condition: {{e}}")
    print(f"Available columns: {{df.columns.tolist()}}")
    print("Condition format example: Age > 18  or  Sex == 'male'")
    
    # Try alternative approach using boolean indexing if query fails
    try:
        print("🔄 Trying alternative boolean indexing approach...")
        
        # For simple equality conditions, try boolean indexing
        if "==" in filter_condition and ("'" in filter_condition or '"' in filter_condition):
            # Extract column and value for simple equality
            parts = filter_condition.split("==")
            if len(parts) == 2:
                col_name = parts[0].strip()
                value = parts[1].strip().strip("'\\\"")
                print(f"Trying boolean indexing: df['{{col_name}}'] == '{{value}}'")
                
                if col_name in df.columns:
                    original_shape = df.shape
                    df = df[df[col_name] == value].copy().reset_index(drop=True)
                    new_shape = df.shape
                    
                    print(f"✅ Alternative filtering completed")
                    print(f"📊 Original shape: {{original_shape}}")
                    print(f"📊 New shape: {{new_shape}}")
                    print(f"📊 Selected {{new_shape[0]}} rows")
                    print(f"🔍 Filtered data preview:")
                    print(df.head())
                else:
                    print(f"❌ Column '{{col_name}}' not found in dataframe")
            else:
                print("❌ Could not parse condition for boolean indexing")
        else:
            print("❌ Alternative approach not applicable for this condition type")
            
    except Exception as e2:
        print(f"❌ Alternative approach also failed: {{e2}}")
        print("Please check the condition syntax and column names")
"""
        
        else:  # conditions is not None
            return f"""
# Select rows by multiple AND conditions: {conditions}
# Reasoning: {reasoning}
print(f"🔧 Filtering rows with multiple conditions: {conditions}")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_select_row")

try:
    # Apply all conditions with AND logic
    original_shape = df.shape
    
    # Store conditions in a variable to avoid f-string quote conflicts
    conditions_list = {repr(conditions)}
    combined_condition = " & ".join([f"({cond})" for cond in conditions_list])
    print(f"Combined condition: {{combined_condition}}")
    
    df = df.query(combined_condition).copy().reset_index(drop=True)
    new_shape = df.shape
    
    print(f"✅ Row filtering completed")
    print(f"📊 Original shape: {{original_shape}}")
    print(f"📊 New shape: {{new_shape}}")
    print(f"📊 Selected {{new_shape[0]}} rows")
    print(f"🔍 Filtered data preview:")
    print(df.head())
except Exception as e:
    print(f"❌ Error applying conditions: {{e}}")
    print(f"Available columns: {{df.columns.tolist()}}")
    print("Condition format example: Age > 18  or  Sex == 'male'")
    
    # Try alternative approach for simple conditions
    try:
        print("🔄 Trying alternative boolean indexing approach...")
        
        # For simple conditions, try to parse and apply one by one
        mask = pd.Series([True] * len(df))
        
        for cond in conditions_list:
            if "==" in cond and ("'" in cond or '"' in cond):
                # Extract column and value for simple equality
                parts = cond.split("==")
                if len(parts) == 2:
                    col_name = parts[0].strip()
                    value = parts[1].strip().strip("'\\\"")
                    
                    if col_name in df.columns:
                        mask = mask & (df[col_name] == value)
                    else:
                        print(f"❌ Column '{{col_name}}' not found in dataframe")
                        mask = pd.Series([False] * len(df))
                        break
                else:
                    print(f"❌ Could not parse condition: {{cond}}")
                    mask = pd.Series([False] * len(df))
                    break
            else:
                print(f"❌ Alternative approach not applicable for condition: {{cond}}")
                mask = pd.Series([False] * len(df))
                break
        
        if mask.any():
            original_shape = df.shape
            df = df[mask].copy().reset_index(drop=True)
            new_shape = df.shape
            
            print(f"✅ Alternative filtering completed")
            print(f"📊 Original shape: {{original_shape}}")
            print(f"📊 New shape: {{new_shape}}")
            print(f"📊 Selected {{new_shape[0]}} rows")
            print(f"🔍 Filtered data preview:")
            print(df.head())
        else:
            print("❌ No rows match the alternative filtering criteria")
            
    except Exception as e2:
        print(f"❌ Alternative approach also failed: {{e2}}")
        print("Please check the condition syntax and column names")
"""
    
    async def _arun(self, indices: list[int] | None = None, condition: str | None = None, 
                   conditions: list[str] | None = None, reasoning: str = "", **kwargs) -> str:
        return self._run(indices, condition, conditions, reasoning, **kwargs)


class SortByInput(BaseModel):
    """Schema for sorting table by columns."""
    
    columns: list[str] = Field(..., description="Column names to sort by (in order of priority)")
    order: str = Field("asc", description="Sort order: 'asc' for ascending, 'desc' for descending")


class SortByTool(BaseTool):
    """Tool to sort table lexicographically by columns."""
    
    name: str = "f_sort_by"
    description: str = "Sort table lexicographically by one or more columns in ascending or descending order."
    response_format: Literal["text"] = "text"
    args_schema: type[SortByInput] = SortByInput
    
    def _run(self, columns: list[str], order: str = "asc", **kwargs) -> str:
        return f"""
{HELPER_FUNCTIONS_CODE}

# Sort table by columns: {columns} in {order} order
print(f"🔧 Sorting by columns: {columns}")
print(f"📊 Sort order: {order}")

# Save current state before modification
_save_df_state("f_sort_by")

# Check if all columns exist
available_columns = df.columns.tolist()
missing_columns = [col for col in {columns} if col not in available_columns]

if missing_columns:
    print(f"❌ Error: Columns not found: {{missing_columns}}")
    print(f"Available columns: {{available_columns}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif "{order}" not in ["asc", "desc"]:
    print(f"❌ Error: Order must be 'asc' or 'desc', got '{order}'")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    # Sort the dataframe
    ascending = "{order}" == "asc"
    df = df.sort_values(by={columns}, ascending=ascending).reset_index(drop=True)
    
    print(f"✅ Successfully sorted by {columns}")
    print(f"📊 Table shape: {{df.shape}}")
    print(f"🔍 Sorted data preview:")
    print(df.head())
"""
    
    async def _arun(self, columns: list[str], order: str = "asc", **kwargs) -> str:
        return self._run(columns, order, **kwargs)


class AggregateInput(BaseModel):
    """Schema for computing aggregations."""
    
    column: str = Field(..., description="Column to aggregate")
    op: str = Field(..., description="Aggregation operation: count, sum, avg, min, max")
    group_by: str | None = Field(None, description="Optional column to group by before aggregating")
    modify_df: bool = Field(False, description="Whether to replace the current table with the aggregation results for further analysis")


class AggregateTool(BaseTool):
    """Tool to compute aggregations over columns."""
    
    name: str = "f_aggregate"
    description: str = "Compute aggregation (count, sum, avg, min, max) over a column, optionally grouped by another column. Can optionally replace the current table with the aggregation results using modify_df=True for further analysis (e.g., sorting by aggregated values)."
    response_format: Literal["text"] = "text"
    args_schema: type[AggregateInput] = AggregateInput
    
    def _run(self, column: str, op: str, group_by: str | None = None, modify_df: bool = False, **kwargs) -> str:
        valid_ops = ["count", "sum", "avg", "min", "max"]
        
        if group_by is None:
            return f"""
# Compute {op} of column '{column}'
print(f"🔧 Computing {op} of column '{column}'")
print(f"💭 Modify table: {modify_df}")

# Check if column exists and operation is valid
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
elif "{op}" not in {valid_ops}:
    print(f"❌ Error: Invalid operation '{op}'. Valid operations: {valid_ops}")
else:
    try:
        if "{op}" == "count":
            result = df['{column}'].count()
        elif "{op}" == "sum":
            result = df['{column}'].sum()
        elif "{op}" == "avg":
            result = df['{column}'].mean()
        elif "{op}" == "min":
            result = df['{column}'].min()
        elif "{op}" == "max":
            result = df['{column}'].max()
        
        print(f"✅ {op.title()} of '{column}': {{result}}")
        
        # Create result dataframe
        result_df = pd.DataFrame({{"{op}": [result]}})
        print(f"🔍 Result:")
        print(result_df)
        
        # Optionally modify the dataframe
        if {modify_df}:
            print(f"\\n🔄 Replacing current table with aggregation result...")
            
            # Save current state before modification
            _save_df_state("f_aggregate")
            
            try:
                df = result_df
                print(f"✅ Table replaced with aggregation result")
                print(f"📊 New table shape: {{df.shape}}")
                print(f"🔍 New table preview:")
                print(df.head())
                
            except Exception as e:
                print(f"❌ Error replacing table: {{e}}")
                # Restore state since operation failed
                _df_history.pop() if _df_history else None
        else:
            print(f"\\n💡 Use modify_df=True to replace current table with this result for further analysis")
        
    except Exception as e:
        print(f"❌ Error computing {op}: {{e}}")
        print(f"Column dtype: {{df['{column}'].dtype}}")
"""
        else:
            return f"""
# Compute {op} of column '{column}' grouped by '{group_by}'
print(f"🔧 Computing {op} of '{column}' grouped by '{group_by}'")
print(f"💭 Modify table: {modify_df}")

# Check if columns exist and operation is valid
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
elif '{group_by}' not in df.columns:
    print(f"❌ Error: Group by column '{group_by}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
elif "{op}" not in {valid_ops}:
    print(f"❌ Error: Invalid operation '{op}'. Valid operations: {valid_ops}")
else:
    try:
        if "{op}" == "count":
            result = df.groupby('{group_by}')['{column}'].count()
        elif "{op}" == "sum":
            result = df.groupby('{group_by}')['{column}'].sum()
        elif "{op}" == "avg":
            result = df.groupby('{group_by}')['{column}'].mean()
        elif "{op}" == "min":
            result = df.groupby('{group_by}')['{column}'].min()
        elif "{op}" == "max":
            result = df.groupby('{group_by}')['{column}'].max()
        
        print(f"✅ {op.title()} of '{column}' by '{group_by}':")
        print(result)
        
        # Convert to dataframe for consistency
        result_df = result.reset_index()
        result_df.columns = ['{group_by}', f'{op}_result']
        print(f"\\n🔍 Result as table:")
        print(result_df)
        
        # Optionally modify the dataframe
        if {modify_df}:
            print(f"\\n🔄 Replacing current table with grouped aggregation results...")
            
            # Save current state before modification
            _save_df_state("f_aggregate")
            
            try:
                df = result_df
                print(f"✅ Table replaced with grouped aggregation results")
                print(f"📊 New table shape: {{df.shape}}")
                print(f"🔍 New table preview:")
                print(df.head())
                
            except Exception as e:
                print(f"❌ Error replacing table: {{e}}")
                # Restore state since operation failed
                _df_history.pop() if _df_history else None
        else:
            print(f"\\n💡 Use modify_df=True to replace current table with these results for further analysis")
        
    except Exception as e:
        print(f"❌ Error computing grouped {op}: {{e}}")
        print(f"Column dtypes: {{df[['{column}', '{group_by}']].dtypes}}")
"""
    
    async def _arun(self, column: str, op: str, group_by: str | None = None, modify_df: bool = False, **kwargs) -> str:
        return self._run(column, op, group_by, modify_df, **kwargs)


class ComputeColumnInput(BaseModel):
    """Schema for computing new columns."""
    
    new_col: str = Field(..., description="Name of the new column to create")
    col_A: str = Field(..., description="Left operand column name")
    op: str = Field(..., description="Operation: add, sub, mul, div, ratio")
    col_B: str | int | float = Field(..., description="Right operand (column name or scalar value)")


class ComputeColumnTool(BaseTool):
    """Tool to create new columns with binary operations."""
    
    name: str = "f_compute_column"
    description: str = "Create a new column by applying binary operations between columns or column and scalar. Operations: add, sub, mul, div, ratio."
    response_format: Literal["text"] = "text"
    args_schema: type[ComputeColumnInput] = ComputeColumnInput
    
    def _run(self, new_col: str, col_A: str, op: str, col_B: str | int | float, **kwargs) -> str:
        valid_ops = ["add", "sub", "mul", "div", "ratio"]
        
        return f"""
{HELPER_FUNCTIONS_CODE}

# Create new column '{new_col}' = '{col_A}' {op} {col_B}
print(f"🔧 Creating column '{new_col}' = '{col_A}' {op} {col_B}")

# Save current state before modification
_save_df_state("f_compute_column")

# Check if operation is valid
if "{op}" not in {valid_ops}:
    print(f"❌ Error: Invalid operation '{op}'. Valid operations: {valid_ops}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif '{col_A}' not in df.columns:
    print(f"❌ Error: Column '{col_A}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        # Check if col_B is a column name or scalar
        if isinstance({col_B}, str) and {col_B} not in df.columns:
            print(f"❌ Error: Column '{col_B}' not found")
            print(f"Available columns: {{df.columns.tolist()}}")
            # Restore state since operation failed
            _df_history.pop() if _df_history else None
        else:
            # Perform the operation
            if "{op}" == "add":
                df['{new_col}'] = df['{col_A}'] + {col_B}
            elif "{op}" == "sub":
                df['{new_col}'] = df['{col_A}'] - {col_B}
            elif "{op}" == "mul":
                df['{new_col}'] = df['{col_A}'] * {col_B}
            elif "{op}" in ["div", "ratio"]:
                df['{new_col}'] = df['{col_A}'] / {col_B}
            
            print(f"✅ Successfully created column '{new_col}'")
            print(f"📊 New table shape: {{df.shape}}")
            print(f"🔍 New column preview:")
            print(df[['{col_A}', '{new_col}']].head())
            
    except Exception as e:
        print(f"❌ Error creating column: {{e}}")
        print(f"Check data types and values for division by zero")
        # Restore state since operation failed
        _df_history.pop() if _df_history else None
"""
    
    async def _arun(self, new_col: str, col_A: str, op: str, col_B: str | int | float, **kwargs) -> str:
        return self._run(new_col, col_A, op, col_B, **kwargs)


class DistinctCountInput(BaseModel):
    """Schema for counting distinct values."""
    
    column: str = Field(..., description="Column name to count distinct values for")


class DistinctCountTool(BaseTool):
    """Tool to count distinct values in a column."""
    
    name: str = "f_distinct_count"
    description: str = "Count the number of unique values in a column."
    response_format: Literal["text"] = "text"
    args_schema: type[DistinctCountInput] = DistinctCountInput
    
    def _run(self, column: str, **kwargs) -> str:
        return f"""
# Count distinct values in column '{column}'
print(f"🔧 Counting distinct values in column '{column}'")

# Check if column exists
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
else:
    # Count distinct values
    distinct_count = df['{column}'].nunique()
    
    print(f"✅ Number of distinct values in '{column}': {{distinct_count}}")
    
    # Show the distinct values if not too many
    if distinct_count <= 20:
        print(f"🔍 Distinct values:")
        distinct_values = df['{column}'].unique()
        print(distinct_values)
    else:
        print(f"🔍 First 10 distinct values:")
        distinct_values = df['{column}'].unique()[:10]
        print(distinct_values)
        print(f"... and {{distinct_count - 10}} more")
    
    # Create result dataframe
    result_df = pd.DataFrame({{"distinct_count": [distinct_count]}})
    print(f"\\n📊 Result:")
    print(result_df)
"""
    
    async def _arun(self, column: str, **kwargs) -> str:
        return self._run(column, **kwargs)


class FinalAnswerInput(BaseModel):
    """Schema for submitting the final answer."""
    answer: str = Field(..., description="The final answer to the user question")


class FinalAnswerTool(BaseTool):
    """Tool to deliver the final answer and end the analysis."""
    name: str = "f_final_answer"
    description: str = "Submit the final answer. Use exactly when you are done analysing."
    response_format: Literal["text"] = "text"
    args_schema: type[FinalAnswerInput] = FinalAnswerInput

    def _run(self, answer: str, **kwargs) -> str:
        # Simply print the answer; no dataframe manipulation.
        return f"FINAL ANSWER: {answer}"

    async def _arun(self, answer: str, **kwargs) -> str:
        return self._run(answer, **kwargs)


class RenameColumnInput(BaseModel):
    """Schema for renaming dataframe columns."""
    rename_map: Dict[str, str] = Field(
        ...,
        description="Dictionary mapping old column names to new names. Format: {'old_name': 'new_name'}",
        example={"old_name": "new_name"}
    )
    reasoning: str = Field(
        ...,
        description="Explanation for why these renames are being made",
        example="Standardizing column names for database ingestion"
    )

class RenameColumnTool(BaseTool):
    name: str = "f_rename_column"
    description: str = "Renames columns in a pandas DataFrame"
    args_schema: type[RenameColumnInput] = RenameColumnInput  # Uses our schema
    
    def _run(self, rename_map: Dict[str, str], reasoning: str, df: pd.DataFrame = None):
        """
        Args:
            rename_map: Dictionary of {old_name: new_name} pairs
            reasoning: Explanation for the renames
            df: Optional DataFrame to modify (passed by LangChain automatically)
            
        Returns:
            dict: {'status': str, 'new_columns': list, 'modified_rows': int}
        """
        if df is None:
            raise ValueError("No DataFrame provided")
            
        try:
            # Perform the rename
            df.rename(columns=rename_map, inplace=True)
            
            return {
                "status": "success",
                "new_columns": list(df.columns),
                "modified_rows": len(df),
                "reasoning": reasoning
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "original_columns": list(df.columns)
            }

class MissingValueStrategy(str, Enum):
    DROP = "drop"
    FILL = "fill"
    INTERPOLATE = "interpolate"

class HandleMissingValuesInput(BaseModel):
    """Schema for handling missing values in a DataFrame."""
    columns: Optional[list[str]] = Field(
        default=None,
        description="Specific columns to process (None means all columns)",
        example=["age", "income"]
    )
    strategy: MissingValueStrategy = Field(
        ...,
        description="How to handle missing values",
        example="fill"
    )
    fill_value: Optional[float | str] = Field(
        default=None,
        description="Value to use when strategy='fill'",
        example=0
    )
    reasoning: str = Field(
        ...,
        description="Why this missing value handling approach was chosen",
        example="Zeros are appropriate for this financial dataset"
    )

class HandleMissingValuesTool(BaseTool):
    name: str = "f_handle_missing_value"
    description: str = "Handles missing values in a pandas DataFrame"
    args_schema: type[HandleMissingValuesInput] = HandleMissingValuesInput
    
    def _run(self, 
             strategy: str, 
             reasoning: str,
             df: pd.DataFrame = None,
             columns: Optional[list[str]] = None,
             fill_value: Optional[float | str] = None):
        """
        Args:
            strategy: One of 'drop', 'fill', or 'interpolate'
            reasoning: Justification for the approach
            df: DataFrame to process (auto-injected by LangChain)
            columns: Specific columns to target
            fill_value: Value for filling when strategy='fill'
            
        Returns:
            dict: Processing results and metadata
        """
        if df is None:
            raise ValueError("No DataFrame provided")
            
        target_cols = columns if columns else df.columns
        results = {
            "original_missing": df[target_cols].isna().sum().to_dict(),
            "action": strategy,
            "reasoning": reasoning
        }
        
        try:
            if strategy == "drop":
                df.dropna(subset=target_cols, inplace=True)
            elif strategy == "fill":
                if fill_value is None:
                    raise ValueError("fill_value required for 'fill' strategy")
                df[target_cols] = df[target_cols].fillna(fill_value)
            elif strategy == "interpolate":
                df[target_cols] = df[target_cols].interpolate()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
            results.update({
                "status": "success",
                "remaining_missing": df[target_cols].isna().sum().to_dict(),
                "modified_rows": len(df),
                "new_dtypes": df[target_cols].dtypes.astype(str).to_dict()
            })
            
        except Exception as e:
            results.update({
                "status": "error",
                "error": str(e),
                "preserved_data_sample": df[target_cols].head().to_dict()
            })
            
        return results

class ManageDuplicatesStrategy(str, Enum):
    """Allowed strategies for the tool"""
    OPTION_A = "drop"
    OPTION_B = "keep"


class ManageDuplicatesInput(BaseModel):
    subset: Optional[List[str]] = Field(
        default = None,
        description = "Specific columns to consider for duplicates (None means all columns)",
        example = ["customer_id", "order_date"]
    )
    keep: Literal["first", "last", False] = Field(
        default="first",
        description="Which duplicates to keep: 'first', 'last', or False to drop all",
        example="first"
    )
    reset_index: bool = Field(
        default=True,
        description="Whether to reset the index after dropping duplicates",
        example=True
    )
    reasoning: str = Field(
        ...,
        description="Why duplicate removal is necessary",
        example="Ensure each customer has only one record per day"
    )
    
    """Schema for managing duplicates."""
    
    column: str = Field(..., description="Column to check for duplicates")
    strategy: ManageDuplicatesStrategy = Field(
        default = ManageDuplicatesStrategy.OPTION_A,
        description = "Action to take: 'drop' to remove duplicates, 'keep' to keep all instances"
    )
    # Documentation field
    reasoning: str = Field(..., description="Explanation of why the deduplication strategy is necessary for this case")

class ManageDuplicatesTool(BaseTool):
    name: str = "f_drop_duplicates"
    description: str = """Removes duplicate rows from a DataFrame. 
    Use when:
    - Preparing clean datasets for analysis
    - Removing redundant records
    - Ensuring unique constraints"""
    args_schema: type[ManageDuplicatesInput] = ManageDuplicatesInput

    def _run(
        self,
        reasoning: str,
        df: pd.DataFrame = None,
        subset: Optional[List[str]] = None,
        keep: str = "first",
        reset_index: bool = True,
        **kwargs
    ):
        """
        Args:
            df: DataFrame to process (auto-injected by LangChain)
            subset: Columns to consider for duplicates
            keep: Which duplicates to keep
            reset_index: Whether to reset index
            reasoning: Justification for the operation
            
        Returns:
            dict: Results with before/after stats
        """
        if df is None:
            raise ValueError("No DataFrame provided")

        results = {
            "action": "drop_duplicates",
            "params": {
                "subset": subset,
                "keep": keep,
                "reset_index": reset_index
            },
            "reasoning": reasoning,
            "original_shape": df.shape,
            "original_duplicates": df.duplicated(subset=subset).sum()
        }

        try:
            # Perform the operation
            df.drop_duplicates(
                subset=subset,
                keep=keep,
                inplace=True
            )
            
            if reset_index:
                df.reset_index(drop=True, inplace=True)

            results.update({
                "status": "success",
                "new_shape": df.shape,
                "remaining_duplicates": df.duplicated(subset=subset).sum(),
                "sample_data": df.head(2).to_dict(orient="records")
            })

        except Exception as e:
            results.update({
                "status": "error",
                "error": str(e),
                "preserved_data_sample": df.head(2).to_dict(orient="records")
            })

        return results
    
class PivotOperation(str, Enum):
    PIVOT = "pivot"
    MELT = "melt"  # Unpivot

class PivotTableInput(BaseModel):
    """Schema for reshaping tables via pivot/unpivot operations."""
    operation: PivotOperation = Field(
        ...,
        description="Whether to pivot (create columns) or melt (unpivot to long format)",
        example="pivot"
    )
    
    # Pivot parameters
    index: Optional[List[str]] = Field(
        default=None,
        description="Columns to use as index (for pivot) or identifier vars (for melt)",
        example=["date", "region"]
    )
    columns: Optional[List[str]] = Field(
        default=None,
        description="Columns whose values will become new columns (pivot only)",
        example=["product_type"]
    )
    values: Optional[List[str]] = Field(
        default=None,
        description="Columns to aggregate (pivot) or value columns (melt)",
        example=["sales", "quantity"]
    )
    
    # Melt-specific parameters
    var_name: Optional[str] = Field(
        default="variable",
        description="Name for melted variable column (melt only)",
        example="metric"
    )
    value_name: Optional[str] = Field(
        default="value",
        description="Name for melted value column (melt only)",
        example="measurement"
    )
    
    # Aggregation control
    aggfunc: Optional[Union[str, List[str]]] = Field(
        default="mean",
        description="Aggregation function(s) for pivot (e.g., 'sum', 'mean', ['sum', 'count'])",
        example="sum"
    )
    
    reasoning: str = Field(
        ...,
        description="Why this reshape operation is needed",
        example="Convert transaction records to wide format for analysis"
    )
    
class PivotTableTool(BaseTool):
    name: str = "f_pivot_table"
    description: str = """Reshapes tables by converting row values to columns (pivoting) 
    or columns to rows (unpivoting/melting). Use when:
    - Converting transaction data to analytical format
    - Preparing data for visualization
    - Normalizing denormalized tables"""
    args_schema: type[PivotTableInput] = PivotTableInput

    def _run(
        self,
        operation: str,
        reasoning: str,
        df: pd.DataFrame = None,
        index: Optional[List[str]] = None,
        columns: Optional[List[str]] = None,
        values: Optional[List[str]] = None,
        aggfunc: Union[str, List[str]] = "mean",
        var_name: str = "variable",
        value_name: str = "value",
        **kwargs
    ):
        """
        Args:
            df: DataFrame to reshape (auto-injected by LangChain)
            operation: 'pivot' or 'melt'
            index: Columns to preserve as identifiers
            columns: Columns to pivot (create new columns from)
            values: Columns to aggregate (pivot) or melt (unpivot)
            aggfunc: Aggregation function(s) for pivot
            var_name/value_name: Column names for melted data
            reasoning: Justification for the operation
        """
        if df is None:
            raise ValueError("No DataFrame provided")

        result = {
            "operation": operation,
            "reasoning": reasoning,
            "original_shape": df.shape,
            "original_columns": list(df.columns)
        }

        try:
            if operation == "pivot":
                # Pivot operation
                pivoted = df.pivot_table(
                    index=index,
                    columns=columns,
                    values=values,
                    aggfunc=aggfunc
                )
                # Flatten multi-index columns if created
                if isinstance(pivoted.columns, pd.MultiIndex):
                    pivoted.columns = ['_'.join(map(str, col)) for col in pivoted.columns.values]
                result_df = pivoted.reset_index()
                
            elif operation == "melt":
                # Unpivot operation
                result_df = df.melt(
                    id_vars=index,
                    value_vars=values,
                    var_name=var_name,
                    value_name=value_name
                )
            else:
                raise ValueError(f"Unknown operation: {operation}")

            result.update({
                "status": "success",
                "new_shape": result_df.shape,
                "new_columns": list(result_df.columns),
                "sample_data": result_df.head(3).to_dict(orient="records")
            })

        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
                "preserved_data_sample": df.head(2).to_dict(orient="records")
            })

        return result

class JoinType(str, Enum):
    INNER = "inner"
    LEFT = "left"
    RIGHT = "right"
    OUTER = "outer"
    CROSS = "cross"

class JoinTablesInput(BaseModel):
    """Schema for combining multiple tables using join operations."""
    join_type: JoinType = Field(
        default=JoinType.INNER,
        description="Type of join to perform",
        example="left"
    )
    
    # For standard joins
    left_on: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column(s) from left table to join on",
        example="customer_id"
    )
    right_on: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column(s) from right table to join on",
        example="client_id"
    )
    
    # For cross joins
    cross_join: bool = Field(
        default=False,
        description="Whether to perform a Cartesian product (ignore join keys)"
    )
    
    # Multi-table support
    suffix: Optional[str] = Field(
        default="_right",
        description="Suffix to append to overlapping columns",
        example="_store"
    )
    
    # Result handling
    drop_duplicates: bool = Field(
        default=True,
        description="Whether to drop duplicate columns after join"
    )
    
    reasoning: str = Field(
        ...,
        description="Why these tables should be joined",
        example="Combine customer profiles with transaction records"
    )
    
class JoinTablesTool(BaseTool):
    name: str = "f_join"
    description: str = """Combines multiple tables using join operations. 
    Use when:
    - Merging related datasets
    - Enriching data with additional attributes
    - Creating analytical datasets from normalized sources"""
    args_schema: type[JoinTablesInput] = JoinTablesInput

    def _run(
        self,
        join_type: str,
        reasoning: str,
        left_df: pd.DataFrame = None,
        right_df: pd.DataFrame = None,
        left_on: Optional[Union[str, List[str]]] = None,
        right_on: Optional[Union[str, List[str]]] = None,
        cross_join: bool = False,
        suffix: str = "_right",
        drop_duplicates: bool = True,
        **kwargs
    ):
        """
        Args:
            left_df: First table (auto-injected)
            right_df: Second table (auto-injected)
            join_type: Type of join (inner/left/right/outer/cross)
            left_on/right_on: Join key columns
            cross_join: Force Cartesian product
            suffix: Suffix for overlapping columns
            drop_duplicates: Remove duplicate columns
            reasoning: Justification for the join
        """
        if left_df is None or right_df is None:
            raise ValueError("Both left and right DataFrames must be provided")

        result = {
            "operation": "join",
            "join_type": join_type,
            "reasoning": reasoning,
            "left_shape": left_df.shape,
            "right_shape": right_df.shape,
            "left_columns": list(left_df.columns),
            "right_columns": list(right_df.columns)
        }

        try:
            if cross_join:
                # Cartesian product
                merged = left_df.assign(key=1).merge(
                    right_df.assign(key=1),
                    on="key"
                ).drop("key", axis=1)
            else:
                # Standard join
                merged = pd.merge(
                    left=left_df,
                    right=right_df,
                    how=join_type,
                    left_on=left_on,
                    right_on=right_on,
                    suffixes=("", suffix)
                )

            # Post-processing
            if drop_duplicates:
                merged = merged.loc[:, ~merged.columns.duplicated()]

            result.update({
                "status": "success",
                "merged_shape": merged.shape,
                "merged_columns": list(merged.columns),
                "sample_data": merged.head(3).to_dict(orient="records"),
                "join_key_overlap": f"{len(merged)} / {len(left_df)} rows matched"
            })

        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
                "left_sample": left_df.head(2).to_dict(orient="records"),
                "right_sample": right_df.head(2).to_dict(orient="records")
            })

        return result
    
class SampleRowsInput(BaseModel):
    """Schema for random row sampling."""
    n: Optional[int] = Field(
        default=None,
        description="Number of rows to sample. Specify either `n` or `frac`."
    )
    frac: Optional[float] = Field(
        default=None,
        description="Fraction of rows to sample. Specify either `n` or `frac`.",
        example=0.1
    )
    random_state: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility.",
        example=42
    )
    replace: bool = Field(
        default=False,
        description="Whether to sample with replacement."
    )
    reasoning: str = Field(
        ...,
        description="Why this sampling is needed.",
        example="Create a 10% holdout set for validation."
    )


class SampleRowsTool(BaseTool):
    name: str = "f_sample"
    description: str = """Randomly sample rows from a DataFrame.
    Use when:
    - You need a random subset for testing or validation.
    - You want to reduce dataset size for quick checks.
    - You want to bootstrap or resample data."""
    args_schema: type[SampleRowsInput] = SampleRowsInput

    def _run(
        self,
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = None,
        replace: bool = False,
        reasoning: str = "",
        df: pd.DataFrame = None,
        **kwargs
    ):
        """
        Args:
            df: The DataFrame to sample from (auto-injected)
            n: Number of rows to sample
            frac: Fraction of rows to sample
            random_state: Seed for reproducibility
            replace: Sample with replacement
            reasoning: Justification for sampling
        """
        if df is None:
            raise ValueError("A DataFrame must be provided for sampling.")

        if n is None and frac is None:
            raise ValueError("Either `n` or `frac` must be specified.")
        if n is not None and frac is not None:
            raise ValueError("Specify only one of `n` or `frac`.")

        result = {
            "operation": "sample",
            "reasoning": reasoning,
            "input_shape": df.shape,
            "input_columns": list(df.columns),
            "n": n,
            "frac": frac,
            "replace": replace,
            "random_state": random_state
        }

        try:
            sampled_df = df.sample(
                n=n,
                frac=frac,
                replace=replace,
                random_state=random_state
            )

            result.update({
                "status": "success",
                "sampled_shape": sampled_df.shape,
                "sampled_columns": list(sampled_df.columns),
                "sample_data": sampled_df.head(3).to_dict(orient="records")
            })

        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
                "input_sample": df.head(2).to_dict(orient="records")
            })

        return result


class BinValuesInput(BaseModel):
    """Schema for binning numeric values into discrete bins."""
    column: str = Field(
        ...,
        description="Name of the numeric column to bin.",
        example="age"
    )
    bins: Union[int, List[float]] = Field(
        ...,
        description="Number of bins OR list of bin edges. If int, equal-width bins are created. If list, custom edges are used.",
        example=5
    )
    labels: Optional[List[str]] = Field(
        default=None,
        description="Optional labels for the resulting bins.",
        example=["young", "middle-aged", "senior"]
    )
    include_lowest: bool = Field(
        default=True,
        description="Whether the first interval should be left-inclusive."
    )
    right: bool = Field(
        default=True,
        description="Indicates whether bins include the rightmost edge or not."
    )
    new_column: Optional[str] = Field(
        default=None,
        description="Optional name for the new binned column. If not provided, `_binned` is appended to original column name."
    )
    reasoning: str = Field(
        ...,
        description="Why these values should be discretized.",
        example="Bucket ages into age groups for cohort analysis."
    )


class BinValuesTool(BaseTool):
    name: str = "f_bin_values"
    description: str = """Discretize continuous numeric values into categorical bins.
    Use when:
    - You want to group numeric data into intervals.
    - You need age groups, income brackets, or other ranges.
    - You want to reduce cardinality for downstream analysis."""
    args_schema: type[BinValuesInput] = BinValuesInput

    def _run(
        self,
        column: str,
        bins: Union[int, List[float]],
        reasoning: str,
        df: pd.DataFrame = None,
        labels: Optional[List[str]] = None,
        include_lowest: bool = True,
        right: bool = True,
        new_column: Optional[str] = None,
        **kwargs
    ):
        """
        Args:
            df: The DataFrame to operate on (auto-injected)
            column: Name of the numeric column to bin
            bins: Number of bins OR explicit bin edges
            labels: Optional labels for bins
            include_lowest: Include lowest edge of first bin
            right: Include rightmost edge
            new_column: Name for the new binned column
            reasoning: Justification for binning
        """
        if df is None:
            raise ValueError("A DataFrame must be provided.")
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame.")

        result = {
            "operation": "bin_values",
            "reasoning": reasoning,
            "column": column,
            "bins": bins,
            "labels": labels,
            "include_lowest": include_lowest,
            "right": right,
            "new_column": new_column or f"{column}_binned",
            "input_shape": df.shape,
            "input_columns": list(df.columns)
        }

        try:
            binned = pd.cut(
                df[column],
                bins=bins,
                labels=labels,
                include_lowest=include_lowest,
                right=right
            )

            df[result["new_column"]] = binned

            result.update({
                "status": "success",
                "output_shape": df.shape,
                "output_columns": list(df.columns),
                "binned_sample": df[[column, result["new_column"]]].head(5).to_dict(orient="records"),
                "bin_counts": binned.value_counts(dropna=False).to_dict()
            })

        except Exception as e:
            result.update({
                "status": "error",
                "error": str(e),
                "input_sample": df.head(3).to_dict(orient="records")
            })

        return result
    
class StringOperationInput(BaseModel):
    """Schema for string operations on a DataFrame column."""
    column: str = Field(
        ...,
        description="Name of the string column to operate on.",
        example="customer_name"
    )

    operation: Literal[
        "lowercase",
        "uppercase",
        "titlecase",
        "substring",
        "concat",
        "replace"
    ] = Field(
        ...,
        description="Type of string operation to apply.",
        example="substring"
    )

    start: Optional[int] = Field(
        default=None,
        description="Start index for substring extraction.",
        example=0
    )
    end: Optional[int] = Field(
        default=None,
        description="End index for substring extraction.",
        example=5
    )

    other_column: Optional[str] = Field(
        default=None,
        description="Second column name for concatenation.",
        example="last_name"
    )
    separator: Optional[str] = Field(
        default=" ",
        description="Separator to use when concatenating columns.",
        example="_"
    )

    find: Optional[str] = Field(
        default=None,
        description="Substring or regex pattern to find for replacement.",
        example="Ltd"
    )
    replace_with: Optional[str] = Field(
        default=None,
        description="Replacement string for matched substrings.",
        example="Limited"
    )

    new_column: Optional[str] = Field(
        default=None,
        description="Optional name for the new column. If not provided, '_mod' is appended.",
        example="name_clean"
    )

    reasoning: str = Field(
        ...,
        description="Explanation of why this string operation is needed.",
        example="Standardize company names to lowercase."
    )


class StringOperationTool(BaseTool):
    name: str = "f_string_operation"
    description: str = """Apply string manipulation to a DataFrame column.
    Use when:
    - Standardizing text (lowercase, uppercase)
    - Extracting substrings
    - Concatenating multiple columns
    - Replacing substrings or regex patterns."""
    args_schema: type[StringOperationInput] = StringOperationInput

    def _run(
        self,
        column: str,
        operation: str,
        reasoning: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        other_column: Optional[str] = None,
        separator: Optional[str] = " ",
        find: Optional[str] = None,
        replace_with: Optional[str] = None,
        new_column: Optional[str] = None,
        **kwargs
    ) -> str:
        new_col = new_column or f"{column}_mod"
        
        if operation == "substring":
            return f"""
# Apply string operation '{operation}' to column '{column}'
# Reasoning: {reasoning}
print(f"🔧 Applying substring operation to column '{column}'")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_string_operation")

# Check if column exists
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif {start} is None:
    print(f"❌ Error: start parameter must be specified for substring operation")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        # Apply substring operation
        df['{new_col}'] = df['{column}'].astype(str).str[{start}:{end}]
        
        print(f"✅ Successfully applied substring to '{column}'")
        print(f"📊 New column: '{new_col}'")
        print(f"🔍 Result preview:")
        print(df[['{column}', '{new_col}']].head())
    except Exception as e:
        print(f"❌ Error applying substring: {{e}}")
        # Restore state since operation failed
        _df_history.pop() if _df_history else None
"""
        
        elif operation == "concat":
            return f"""
# Apply string operation '{operation}' to columns '{column}' and '{other_column}'
# Reasoning: {reasoning}
print(f"🔧 Concatenating columns '{column}' and '{other_column}'")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_string_operation")

# Check if columns exist
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif '{other_column}' not in df.columns:
    print(f"❌ Error: Column '{other_column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        # Apply concatenation
        df['{new_col}'] = df['{column}'].astype(str) + '{separator}' + df['{other_column}'].astype(str)
        
        print(f"✅ Successfully concatenated '{column}' and '{other_column}'")
        print(f"📊 New column: '{new_col}'")
        print(f"🔍 Result preview:")
        print(df[['{column}', '{other_column}', '{new_col}']].head())
    except Exception as e:
        print(f"❌ Error applying concatenation: {{e}}")
        # Restore state since operation failed
        _df_history.pop() if _df_history else None
"""
        
        elif operation == "replace":
            return f"""
# Apply string operation '{operation}' to column '{column}'
# Reasoning: {reasoning}
print(f"🔧 Applying replace operation to column '{column}'")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_string_operation")

# Check if column exists and parameters are provided
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif {repr(find)} is None or {repr(replace_with)} is None:
    print(f"❌ Error: Both find and replace_with parameters must be specified")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        # Apply replace operation
        find_pattern = {repr(find)}
        replace_value = {repr(replace_with)}
        df['{new_col}'] = df['{column}'].astype(str).str.replace(find_pattern, replace_value, regex=True)
        
        print(f"✅ Successfully applied replace to '{column}'")
        print(f"📊 Find pattern: {{find_pattern}}")
        print(f"📊 Replace with: {{replace_value}}")
        print(f"📊 New column: '{new_col}'")
        print(f"🔍 Result preview:")
        print(df[['{column}', '{new_col}']].head())
    except Exception as e:
        print(f"❌ Error applying replace: {{e}}")
        # Restore state since operation failed
        _df_history.pop() if _df_history else None
"""
        
        else:
            # Handle simple operations (lowercase, uppercase, titlecase)
            return f"""
# Apply string operation '{operation}' to column '{column}'
# Reasoning: {reasoning}
print(f"🔧 Applying '{operation}' operation to column '{column}'")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_string_operation")

# Check if column exists
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
elif '{operation}' not in ['lowercase', 'uppercase', 'titlecase']:
    print(f"❌ Error: Unsupported operation '{operation}'")
    print(f"Supported operations: lowercase, uppercase, titlecase, substring, concat, replace")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        # Apply the string operation
        if '{operation}' == 'lowercase':
            df['{new_col}'] = df['{column}'].astype(str).str.lower()
        elif '{operation}' == 'uppercase':
            df['{new_col}'] = df['{column}'].astype(str).str.upper()
        elif '{operation}' == 'titlecase':
            df['{new_col}'] = df['{column}'].astype(str).str.title()
        
        print(f"✅ Successfully applied '{operation}' to '{column}'")
        print(f"📊 New column: '{new_col}'")
        print(f"🔍 Result preview:")
        print(df[['{column}', '{new_col}']].head())
    except Exception as e:
                 print(f"❌ Error applying {operation}: {{e}}")
                 # Restore state since operation failed
                 _df_history.pop() if _df_history else None
"""
    
    async def _arun(
        self,
        column: str,
        operation: str,
        reasoning: str,
        start: Optional[int] = None,
        end: Optional[int] = None,
        other_column: Optional[str] = None,
        separator: Optional[str] = " ",
        find: Optional[str] = None,
        replace_with: Optional[str] = None,
        new_column: Optional[str] = None,
        **kwargs
    ) -> str:
        return self._run(column, operation, reasoning, start, end, other_column, separator, find, replace_with, new_column, **kwargs)

    
from pydantic import BaseModel, Field
from typing import Optional, Literal
import pandas as pd


class ProcessDatetimeInput(BaseModel):
    """Schema for processing datetime columns."""
    column: str = Field(
        ...,
        description="Name of the column with datetime strings or objects.",
        example="order_date"
    )

    operation: Literal[
        "parse",
        "extract_year",
        "extract_month",
        "extract_day",
        "extract_weekday",
        "to_timezone",
        "format"
    ] = Field(
        ...,
        description="Type of datetime operation to perform.",
        example="extract_year"
    )

    timezone: Optional[str] = Field(
        default=None,
        description="Timezone name (e.g., 'UTC', 'America/New_York'). Used for timezone conversion.",
        example="UTC"
    )

    date_format: Optional[str] = Field(
        default=None,
        description="Format string for output if formatting dates. E.g., '%Y-%m-%d'.",
        example="%Y-%m-%d"
    )

    new_column: Optional[str] = Field(
        default=None,
        description="Optional name for the new output column. If not provided, suffix is auto-added.",
        example="order_date_year"
    )

    reasoning: str = Field(
        ...,
        description="Why this datetime operation is needed.",
        example="Extract year for time-series grouping."
    )


class ProcessDatetimeTool(BaseTool):
    name: str = "f_process_datetime"
    description: str = """Process or transform datetime columns.
    Use when:
    - Parsing strings into datetime objects.
    - Extracting parts like year, month, day.
    - Converting timezone.
    - Formatting datetimes as strings."""
    response_format: Literal["text"] = "text"
    args_schema: type[ProcessDatetimeInput] = ProcessDatetimeInput

    def _run(
        self,
        column: str,
        operation: str,
        reasoning: str,
        timezone: Optional[str] = None,
        date_format: Optional[str] = None,
        new_column: Optional[str] = None,
        **kwargs
    ) -> str:
        new_col = new_column or f"{column}_{operation}"
        
        return f"""
# Process datetime column '{column}' with operation '{operation}'
# Reasoning: {reasoning}
print(f"🔧 Processing datetime column '{column}' with operation '{operation}'")
print(f"💭 Reasoning: {reasoning}")

# Save current state before modification
_save_df_state("f_process_datetime")

# Check if column exists
if '{column}' not in df.columns:
    print(f"❌ Error: Column '{column}' not found")
    print(f"Available columns: {{df.columns.tolist()}}")
    # Restore state since operation failed
    _df_history.pop() if _df_history else None
else:
    try:
        new_col_name = {repr(new_col)}
        
        if '{operation}' == 'parse':
            # Parse strings to datetime
            df[new_col_name] = pd.to_datetime(df['{column}'], errors='coerce')
            print(f"✅ Successfully parsed '{column}' to datetime")
            
        else:
            # First ensure column is datetime
            dt_series = pd.to_datetime(df['{column}'], errors='coerce')
            
            if '{operation}' == 'extract_year':
                df[new_col_name] = dt_series.dt.year
                print(f"✅ Successfully extracted year from '{column}'")
            elif '{operation}' == 'extract_month':
                df[new_col_name] = dt_series.dt.month
                print(f"✅ Successfully extracted month from '{column}'")
            elif '{operation}' == 'extract_day':
                df[new_col_name] = dt_series.dt.day
                print(f"✅ Successfully extracted day from '{column}'")
            elif '{operation}' == 'extract_weekday':
                df[new_col_name] = dt_series.dt.day_name()
                print(f"✅ Successfully extracted weekday from '{column}'")
            elif '{operation}' == 'to_timezone':
                if {repr(timezone)} is None:
                    print(f"❌ Error: timezone parameter required for timezone conversion")
                    _df_history.pop() if _df_history else None
                else:
                    timezone_str = {repr(timezone)}
                    df[new_col_name] = dt_series.dt.tz_localize('UTC').dt.tz_convert(timezone_str)
                    print(f"✅ Successfully converted '{column}' to timezone {{timezone_str}}")
            elif '{operation}' == 'format':
                if {repr(date_format)} is None:
                    print(f"❌ Error: date_format parameter required for formatting")
                    _df_history.pop() if _df_history else None
                else:
                    format_str = {repr(date_format)}
                    df[new_col_name] = dt_series.dt.strftime(format_str)
                    print(f"✅ Successfully formatted '{column}' with format {{format_str}}")
            else:
                print(f"❌ Error: Unsupported operation '{operation}'")
                print(f"Supported operations: parse, extract_year, extract_month, extract_day, extract_weekday, to_timezone, format")
                _df_history.pop() if _df_history else None
        
        if '{operation}' in ['parse', 'extract_year', 'extract_month', 'extract_day', 'extract_weekday'] or ({repr(timezone)} is not None and '{operation}' == 'to_timezone') or ({repr(date_format)} is not None and '{operation}' == 'format'):
            print(f"📊 New column: {{new_col_name}}")
            print(f"📊 Table shape: {{df.shape}}")
            print(f"🔍 Result preview:")
            print(df[['{column}', new_col_name]].head())
            
    except Exception as e:
        print(f"❌ Error processing datetime: {{e}}")
        print(f"Column dtype: {{df['{column}'].dtype if '{column}' in df.columns else 'N/A'}}")
        # Restore state since operation failed
        _df_history.pop() if _df_history else None
"""
    
    async def _arun(
        self,
        column: str,
        operation: str,
        reasoning: str,
        timezone: Optional[str] = None,
        date_format: Optional[str] = None,
        new_column: Optional[str] = None,
        **kwargs
    ) -> str:
        return self._run(column, operation, reasoning, timezone, date_format, new_column, **kwargs)


# ===== UNDO TOOL =====

class UndoInput(BaseModel):
    """Schema for undoing the last operation."""
    confirmation: bool = Field(True, description="Confirm that you want to undo the last operation")


class UndoTool(BaseTool):
    """Tool to undo the last table modification."""
    
    name: str = "f_undo"
    description: str = "Undo the last table modification operation. Use when you realize the last change was incorrect or want to try a different approach."
    response_format: Literal["text"] = "text"
    args_schema: type[UndoInput] = UndoInput
    
    def _run(self, confirmation: bool = True, **kwargs) -> str:
        return f"""
{HELPER_FUNCTIONS_CODE}

# Undo the last table operation
print(f"🔄 Attempting to undo last operation...")

if not {confirmation}:
    print(f"❌ Operation cancelled - confirmation required")
else:
    # Restore previous df state
    previous_df, last_op = _restore_df_state()
    
    if previous_df is not None:
        df = previous_df
        print(f"✅ Successfully undid last operation: {{last_op}}")
        print(f"📊 Restored table shape: {{df.shape}}")
        print(f"🔍 Current state preview:")
        print(df.head())
    else:
        print(f"❌ No previous state available to restore")
        print(f"💡 History is empty - no operations to undo")
"""
    
    async def _arun(self, confirmation: bool = True, **kwargs) -> str:
        return self._run(confirmation, **kwargs)


# ===== TOOL REGISTRY =====
# This is where the magic happens - just add tools to this list!

ENHANCED_MODULAR_TOOLS = [
    # Core tools
    SelectColumnTool(),
    GroupByTool(),
    PrintTableTool(),
    
    # Enhanced tools - easy to add!
    CalculateAverageTool(),
    FilterRowsTool(),
    GetDataInfoTool(),
    
    # Additional tools from enhanced_CoT_tools.py
    SelectRowTool(),
    SortByTool(),
    AggregateTool(),
    ComputeColumnTool(),
    DistinctCountTool(),
    FinalAnswerTool(),
    # RenameColumnTool(), 
    # HandleMissingValuesTool(), 
    # ManageDuplicatesTool(), 
    # PivotTableTool(), 
    # JoinTablesTool(),
    ProcessDatetimeTool(),
    StringOperationTool(),
    # BinValuesTool(),
    # SampleRowsTool(),
    
    # Utility tools
    UndoTool(),
]

# Want to add more tools? Just create them and add to the list above!
# LangChain's bind_tools() will automatically:
# 1. Extract tool schemas from Pydantic models
# 2. Include them in the LLM prompt
# 3. Validate tool calls
# 4. Route to the correct tool execution

# Example of how easy it is to add a new tool:
"""
class NewToolInput(BaseModel):
    param1: str = Field(..., description="Description of param1")
    param2: int = Field(default=10, description="Description of param2")

class NewTool(BaseTool):
    name: str = "f_new_tool"
    description: str = "Description of what this tool does"
    response_format: Literal["text"] = "text"
    args_schema: type[NewToolInput] = NewToolInput
    
    def _run(self, param1: str, param2: int = 10, **kwargs) -> str:
        return f"# Generated Python code here"
    
    async def _arun(self, param1: str, param2: int = 10, **kwargs) -> str:
        return self._run(param1, param2, **kwargs)

# Then just add: NewTool() to ENHANCED_MODULAR_TOOLS list!
""" 