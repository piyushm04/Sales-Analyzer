"""
Data Cleaning Module
Handles data preprocessing, cleaning, and transformation
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(file_path):
    """
    Load CSV data file
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ Data loaded successfully: {len(df)} rows")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def clean_data(df):
    """
    Clean and preprocess the dataset
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    if df is None or df.empty:
        return None
    
    df_clean = df.copy()
    
    # Convert date columns
    date_columns = ['Order Date', 'Ship Date']
    for col in date_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_datetime(df_clean[col], errors='coerce')
    
    # Ensure numeric columns are numeric
    numeric_columns = ['Sales', 'Profit', 'Quantity', 'Discount']
    for col in numeric_columns:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # Remove rows with missing critical data
    df_clean = df_clean.dropna(subset=['Sales', 'Order Date'])
    
    # Calculate additional metrics
    if 'Sales' in df_clean.columns and 'Profit' in df_clean.columns:
        df_clean['Profit Margin'] = (df_clean['Profit'] / df_clean['Sales'] * 100).round(2)
        df_clean['Profit Margin'] = df_clean['Profit Margin'].replace([np.inf, -np.inf], 0)
    
    # Extract time features
    if 'Order Date' in df_clean.columns:
        df_clean['Year'] = df_clean['Order Date'].dt.year
        df_clean['Month'] = df_clean['Order Date'].dt.month
        df_clean['Quarter'] = df_clean['Order Date'].dt.quarter
        df_clean['YearMonth'] = df_clean['Order Date'].dt.to_period('M')
        df_clean['YearQuarter'] = df_clean['Order Date'].dt.to_period('Q')
    
    # Create loss flag
    if 'Profit' in df_clean.columns:
        df_clean['Is Loss'] = df_clean['Profit'] < 0
    
    print(f"✅ Data cleaned: {len(df_clean)} rows remaining")
    return df_clean


def get_summary_stats(df):
    """
    Get summary statistics of the dataset
    
    Args:
        df: Cleaned DataFrame
        
    Returns:
        Dictionary with summary statistics
    """
    if df is None or df.empty:
        return {}
    
    stats = {
        'total_rows': len(df),
        'date_range': {
            'start': df['Order Date'].min() if 'Order Date' in df.columns else None,
            'end': df['Order Date'].max() if 'Order Date' in df.columns else None
        },
        'total_sales': df['Sales'].sum() if 'Sales' in df.columns else 0,
        'total_profit': df['Profit'].sum() if 'Profit' in df.columns else 0,
        'avg_profit_margin': df['Profit Margin'].mean() if 'Profit Margin' in df.columns else 0,
        'categories': df['Category'].nunique() if 'Category' in df.columns else 0,
        'regions': df['Region'].nunique() if 'Region' in df.columns else 0,
        'loss_making_orders': df['Is Loss'].sum() if 'Is Loss' in df.columns else 0
    }
    
    return stats


if __name__ == "__main__":
    # Test the cleaning functions
    print("Testing data cleaning module...")
    # This would be used with actual data file
    pass

