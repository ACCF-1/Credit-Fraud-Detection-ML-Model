

#In[0] Libraries
'''import necessary libraries and setup parameters'''
import pandas as pd
import numpy as np

import sys
import os
import importlib
import datetime as dt

import sys
import os
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    sys.path.append(os.path.join(parent_dir, 'configurations'))
else:
    sys.path.append(os.path.join(os.getcwd(), 'configurations'))
    sys.path.append(os.path.join(os.getcwd(), 'scripts'))
import config as cfg

import utility_functions as uf
import specific_functions as sf

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")

from pandas.api.types import is_numeric_dtype, is_string_dtype, is_bool_dtype
from typing import Dict, Tuple


# In[1] cleaning functions
'''functions for data cleansing and validation'''

def convert_datatypes(
    df: pd.DataFrame,
    data_schema: pd.DataFrame
):
    # Define mapping of schema data types to pandas types
    dtype_mapping = {
        "string": "string",
        "int": "Int64",
        "float": "float64",
        "boolean": "boolean",
        "datetime": "datetime64[ns]"
    }
    schema_dict = data_schema.set_index("column_name")["data_type"].to_dict()

    # Convert data types
    for column, dtype in schema_dict.items():
        if dtype in dtype_mapping:
            df[column] = df[column].astype(dtype_mapping[dtype])
        else:
            print(f"Warning: Unknown data type '{dtype}' for column '{column}'")
    return df

def check_if_nulls_in_non_nullable_cols(
    df:pd.DataFrame, 
    data_schema:pd.DataFrame
):
    """ Check for null values in non-nullable columns of the DataFrame.
    Args:  
        df (pd.DataFrame): The DataFrame to check.
        data_schema (pd.DataFrame): The schema DataFrame containing column nullability information.
    Returns:
        None: Prints the status of non-nullable columns.
    """
    # Get non-nullable columns from the schema
    if 'nullable' not in data_schema.columns or 'column_name' not in data_schema.columns:
        print("Schema does not contain 'nullable' or 'column_name' columns.")
        return

    # Get non-nullable columns
    non_nullable_cols = data_schema[data_schema['nullable'] == 'N']['column_name'].tolist()
    print(f"Non-nullable columns: {non_nullable_cols}")
    # Check for nulls in non-nullable columns
    for col in non_nullable_cols:
        if df[col].isnull().any():
            print(f"Column '{col}' has null values, which is not allowed.")
        else:
            print(f"Column '{col}' is valid.")

def check_missing_columns(
        df: pd.DataFrame, 
        data_schema: pd.DataFrame
) -> bool:
    """ Check if the DataFrame has all required columns as per the schema.
    Args:  
        df (pd.DataFrame): The DataFrame to check.
        data_schema (pd.DataFrame): The schema DataFrame containing required columns.     
    Returns:
        bool: True if all required columns are present, False otherwise.
    """
    # Initialize validation status
    is_valid = True

    # Check for missing columns
    required_columns = data_schema['column_name'].tolist()
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        is_valid = False
    return is_valid

def check_non_nullable_violation(
    df: pd.DataFrame, 
    data_schema: pd.DataFrame
) -> bool:
    """ Check for null values in non-nullable columns of the DataFrame.
    Args:  
        df (pd.DataFrame): The DataFrame to check.
        data_schema (pd.DataFrame): The schema DataFrame containing column nullability information.
    Returns:
        bool: True if no nulls are found in non-nullable columns, False otherwise.
    """
    # Initialize validation status
    is_valid = True

    # Check for nulls in non-nullable columns
    non_nullable_cols = data_schema[data_schema['nullable'] == 'N']['column_name'].tolist()
    for col in non_nullable_cols:
        if col in df.columns and df[col].isnull().any():
            print(f"Column '{col}' has null values, which is not allowed.")
            is_valid = False
    return is_valid

def check_data_types(
    df: pd.DataFrame,   
    data_schema: pd.DataFrame
) -> bool:
    """ Check if the DataFrame columns match the expected data types from the schema.   
    Args:
        df (pd.DataFrame): The DataFrame to check.
        data_schema (pd.DataFrame): The schema DataFrame containing expected data types.    
    Returns:
        bool: True if all columns have the expected data types, False otherwise.
    """
    # Initialize validation status
    is_valid = True

    # Check for invalid data types
    dtype_mapping = {
        "string": "string",
        "int": "Int64",
        "float": "float64",
        "boolean": "bool",
        "datetime": "datetime64[ns]"
    }
    schema_dict = data_schema.set_index("column_name")["data_type"].to_dict()
    for column, expected_dtype in schema_dict.items():
        if column in df.columns:
            actual_dtype = str(df[column].dtype)
            if expected_dtype in dtype_mapping and actual_dtype != dtype_mapping[expected_dtype]:
                print(f"Column '{column}' has invalid data type. Expected: {dtype_mapping[expected_dtype]}, Found: {actual_dtype}")
                is_valid = False
    return is_valid

def check_non_negative_violations(
    df: pd.DataFrame,
    data_schema: pd.DataFrame   
) -> bool:
    """ Check for negative values in non-negative columns of the DataFrame.
    Args:
        df (pd.DataFrame): The DataFrame to check.
        data_schema (pd.DataFrame): The schema DataFrame containing column non-negativity information.
    Returns:
        bool: True if no negative values are found in non-negative columns, False otherwise.
    """
    # Initialize validation status
    is_valid = True
    
    # Check for negative values in non-negative columns
    negative_values = {}
    for column in df.select_dtypes(include=['number']).columns:
        has_negative = (df[column] < 0).any()
        negative_values[column] = has_negative
        if has_negative:
            print(f"Column '{column}' contains negative values.")
            is_valid = False

    return is_valid

def check_categorical_features(
        df: pd.DataFrame, 
        data_schema: pd.DataFrame, 
        high_cardinality_threshold: int = 50,
        low_cardinality_threshold: int = 30
) -> Tuple[Dict, Dict]:
    """
    Check dataset against schema to find:
    1. High cardinality categorical features
    2. Low cardinality features that should be categorical but aren't marked as such
    
    Parameters:
    - df: The dataset to check
    - data_schema: The schema dataframe
    - high_cardinality_threshold: Threshold for considering a categorical feature as high cardinality
    
    Returns:
    - Tuple of two dictionaries:
        1. High cardinality categorical features with their unique counts
        2. Low cardinality features not marked as categorical with their unique counts
    """
    # Get categorical columns from schema
    schema_categorical_cols = data_schema.loc[
        data_schema['is_categorical'] == 'Y', 'column_name'].tolist()
    
    # Initialize result dictionaries
    high_cardinality_categorical = {}
    low_cardinality_should_be_categorical = {}
    
    # Check each column in the dataset
    for col in df.columns:
        # Skip if column not in schema
        if col not in data_schema['column_name'].values:
            continue
            
        # Get unique values count (excluding nulls)
        unique_count = df[col].nunique()
        
        # Case 1: Column is marked as categorical in schema
        if col in schema_categorical_cols:
            if unique_count >= high_cardinality_threshold:
                high_cardinality_categorical[col] = unique_count
                
        # Case 2: Column is not marked as categorical but has low cardinality
        else:
            # Check if it's a potential categorical feature
            # We'll consider features with <= threshold unique values as potential categorical
            if unique_count <= low_cardinality_threshold:
                # Additional checks to avoid numerical features
                dtype = df[col].dtype
                if dtype in ['object', 'bool', 'category'] or (
                    dtype in ['int64', 'float64'] and unique_count <= low_cardinality_threshold):
                    low_cardinality_should_be_categorical[col] = unique_count
    
    return high_cardinality_categorical, low_cardinality_should_be_categorical


# In[1] Load Data
if __name__ == '__main__':
    # Load the raw data
    cf_raw_df = pd.read_csv(
        os.path.join(
            os.path.dirname(os.getcwd()),
            'data',
            'raw',
            'credit_fraud_data.csv'
        ),
    )
    data_schema = pd.read_excel(
        os.path.join(os.path.dirname(os.getcwd()),
        'configurations',
        'data_schema.xlsx'),
        sheet_name='schema'
    )
    cf_raw_df
    tgt_col = 'is_fraud'

    # In[2] cleansing and validation
    cf_raw_df = convert_datatypes(cf_raw_df, data_schema)
    data_overview = uf.data_overview(cf_raw_df, tgt_col)
    check_if_nulls_in_non_nullable_cols(cf_raw_df, data_schema)
    check_categorical_features(cf_raw_df, data_schema)
    check_missing_columns(cf_raw_df, data_schema)
    check_non_nullable_violation(cf_raw_df, data_schema)
    check_non_negative_violations(cf_raw_df, data_schema)
    check_data_types(cf_raw_df, data_schema)

    # In[3] Save the cleaned data
    print("Saving cleaned data...")
    cf_raw_df.to_csv(
        os.path.join(os.path.dirname(os.getcwd()), 'data', 'interim', 'credit_fraud_data_cleaned.csv'),
        index=False
    )
    print("Cleaned data saved...")

