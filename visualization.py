"""
Visualization Module
Creates charts and visualizations for the dashboard
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')


def create_kpi_cards(df: pd.DataFrame) -> dict:
    """
    Calculate KPI metrics
    
    Args:
        df: Sales DataFrame
        
    Returns:
        Dictionary with KPI values
    """
    sales_col = 'Sales' if 'Sales' in df.columns else 'sales'
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    
    total_sales = df[sales_col].sum()
    total_profit = df[profit_col].sum()
    profit_margin = (total_profit / total_sales * 100) if total_sales > 0 else 0
    
    # Calculate YoY Growth
    if 'Order Date' in df.columns or 'order_date' in df.columns:
        date_col = 'Order Date' if 'Order Date' in df.columns else 'order_date'
        df[date_col] = pd.to_datetime(df[date_col])
        df['Year'] = df[date_col].dt.year
        
        if df['Year'].nunique() > 1:
            years = sorted(df['Year'].unique())
            if len(years) >= 2:
                current_year = years[-1]
                previous_year = years[-2]
                
                current_sales = df[df['Year'] == current_year][sales_col].sum()
                previous_sales = df[df['Year'] == previous_year][sales_col].sum()
                
                yoy_growth = ((current_sales - previous_sales) / previous_sales * 100) if previous_sales > 0 else 0
            else:
                yoy_growth = 0
        else:
            yoy_growth = 0
    else:
        yoy_growth = 0
    
    return {
        'total_sales': total_sales,
        'total_profit': total_profit,
        'profit_margin': profit_margin,
        'yoy_growth': yoy_growth
    }


def create_monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create monthly sales and profit trend chart
    
    Args:
        df: Sales DataFrame
        
    Returns:
        Plotly figure
    """
    date_col = 'Order Date' if 'Order Date' in df.columns else 'order_date'
    sales_col = 'Sales' if 'Sales' in df.columns else 'sales'
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    
    df[date_col] = pd.to_datetime(df[date_col])
    df['YearMonth'] = df[date_col].dt.to_period('M')
    
    monthly_data = df.groupby('YearMonth').agg({
        sales_col: 'sum',
        profit_col: 'sum'
    }).reset_index()
    
    monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'].astype(str))
    monthly_data = monthly_data.sort_values('Date')
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Date'],
            y=monthly_data[sales_col],
            name='Sales',
            line=dict(color='#1f77b4', width=3),
            mode='lines+markers'
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(
            x=monthly_data['Date'],
            y=monthly_data[profit_col],
            name='Profit',
            line=dict(color='#2ca02c', width=3),
            mode='lines+markers'
        ),
        secondary_y=True
    )
    
    fig.update_xaxes(title_text="Date")
    fig.update_yaxes(title_text="Sales ($)", secondary_y=False)
    fig.update_yaxes(title_text="Profit ($)", secondary_y=True)
    
    fig.update_layout(
        title="Monthly Sales & Profit Trend",
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_category_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create category-wise performance chart
    
    Args:
        df: Sales DataFrame
        
    Returns:
        Plotly figure
    """
    category_col = 'Category' if 'Category' in df.columns else 'category'
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    
    category_data = df.groupby(category_col).agg({
        profit_col: 'sum'
    }).reset_index()
    
    category_data = category_data.sort_values(profit_col, ascending=True)
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=category_data[category_col],
        x=category_data[profit_col],
        name='Profit',
        orientation='h',
        marker=dict(
            color=category_data[profit_col],
            colorscale='RdYlGn',
            showscale=True
        )
    ))
    
    fig.update_layout(
        title="Profit by Category",
        xaxis_title="Profit ($)",
        yaxis_title="Category",
        height=300,
        template='plotly_white'
    )
    
    return fig


def create_region_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Create region-wise performance heatmap
    
    Args:
        df: Sales DataFrame
        
    Returns:
        Plotly figure
    """
    region_col = 'Region' if 'Region' in df.columns else 'region'
    category_col = 'Category' if 'Category' in df.columns else 'category'
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    
    pivot_data = df.pivot_table(
        values=profit_col,
        index=region_col,
        columns=category_col,
        aggfunc='sum',
        fill_value=0
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn',
        text=pivot_data.values,
        texttemplate='$%{text:,.0f}',
        textfont={"size": 10},
        colorbar=dict(title="Profit ($)")
    ))
    
    fig.update_layout(
        title="Profit Heatmap: Region vs Category",
        xaxis_title="Category",
        yaxis_title="Region",
        height=400,
        template='plotly_white'
    )
    
    return fig


def create_top_products_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create top products chart
    
    Args:
        df: Sales DataFrame
        top_n: Number of top products to show
        
    Returns:
        Plotly figure
    """
    product_col = 'Product Name' if 'Product Name' in df.columns else 'product_name'
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    
    top_products = df.groupby(product_col)[profit_col].sum().nlargest(top_n).reset_index()
    top_products = top_products.sort_values(profit_col, ascending=True)
    
    fig = go.Figure(go.Bar(
        y=top_products[product_col],
        x=top_products[profit_col],
        orientation='h',
        marker=dict(
            color=top_products[profit_col],
            colorscale='Viridis',
            showscale=True
        ),
        text=top_products[profit_col],
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title=f"Top {top_n} Products by Profit",
        xaxis_title="Profit ($)",
        yaxis_title="Product",
        height=500,
        template='plotly_white'
    )
    
    return fig


def create_forecast_chart(historical_df: pd.DataFrame, forecast_df: pd.DataFrame) -> go.Figure:
    """
    Create forecast chart with historical and predicted data
    
    Args:
        historical_df: Historical sales data
        forecast_df: Forecasted sales data
        
    Returns:
        Plotly figure
    """
    date_col = 'Order Date' if 'Order Date' in historical_df.columns else 'order_date'
    sales_col = 'Sales' if 'Sales' in historical_df.columns else 'sales'
    
    historical_df[date_col] = pd.to_datetime(historical_df[date_col])
    historical_df['YearMonth'] = historical_df[date_col].dt.to_period('M')
    
    monthly_historical = historical_df.groupby('YearMonth')[sales_col].sum().reset_index()
    monthly_historical['Date'] = pd.to_datetime(monthly_historical['YearMonth'].astype(str))
    monthly_historical = monthly_historical.sort_values('Date')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_historical['Date'],
        y=monthly_historical[sales_col],
        name='Historical Sales',
        line=dict(color='#1f77b4', width=3),
        mode='lines+markers'
    ))
    
    if not forecast_df.empty and 'Date' in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecasted_Sales'],
            name='Forecasted Sales',
            line=dict(color='#ff7f0e', width=3, dash='dash'),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales ($)",
        hovermode='x unified',
        height=400,
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig


def create_loss_analysis_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create loss-making products/regions analysis
    
    Args:
        df: Sales DataFrame
        
    Returns:
        Plotly figure
    """
    profit_col = 'Profit' if 'Profit' in df.columns else 'profit'
    category_col = 'Category' if 'Category' in df.columns else 'category'
    
    loss_data = df[df[profit_col] < 0].groupby(category_col)[profit_col].sum().reset_index()
    loss_data = loss_data.sort_values(profit_col, ascending=True)
    
    fig = go.Figure(go.Bar(
        y=loss_data[category_col],
        x=loss_data[profit_col],
        orientation='h',
        marker=dict(color='#d62728'),
        text=loss_data[profit_col],
        texttemplate='$%{text:,.0f}',
        textposition='outside'
    ))
    
    fig.update_layout(
        title="Loss-Making Categories",
        xaxis_title="Total Loss ($)",
        yaxis_title="Category",
        height=300,
        template='plotly_white'
    )
    
    return fig


if __name__ == "__main__":
    print("Testing visualization module...")
    pass

