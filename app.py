"""
Sales & Profit Analytics Dashboard
Main Streamlit application
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import custom modules
from data_cleaning import load_data, clean_data, get_summary_stats
from database import DatabaseManager
from forecasting import SalesForecaster, simple_forecast
from visualization import (
    create_kpi_cards,
    create_monthly_trend_chart,
    create_category_performance_chart,
    create_region_heatmap,
    create_top_products_chart,
    create_forecast_chart,
    create_loss_analysis_chart
)

# Page configuration
st.set_page_config(
    page_title="Sales & Profit Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .kpi-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_clean' not in st.session_state:
    st.session_state.df_clean = None


def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">📊 Sales & Profit Analytics Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data source selection
        data_source = st.radio(
            "Data Source",
            ["Upload CSV File", "Use Sample Data", "Database Connection"],
            help="Choose your data source"
        )
        
        # File upload
        if data_source == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Upload Sales Data (CSV)",
                type=['csv'],
                help="Upload your Superstore sales dataset"
            )
            
            if uploaded_file is not None:
                if st.button("Load Data", type="primary"):
                    with st.spinner("Loading and cleaning data..."):
                        df = load_data(uploaded_file)
                        if df is not None:
                            df_clean = clean_data(df)
                            if df_clean is not None:
                                st.session_state.df = df
                                st.session_state.df_clean = df_clean
                                st.session_state.data_loaded = True
                                st.success("✅ Data loaded successfully!")
        
        # Sample data option
        elif data_source == "Use Sample Data":
            st.info("💡 Sample data option - generate synthetic data for demo")
            if st.button("Generate Sample Data", type="primary"):
                with st.spinner("Generating sample data..."):
                    df_clean = generate_sample_data()
                    st.session_state.df_clean = df_clean
                    st.session_state.data_loaded = True
                    st.success("✅ Sample data generated!")
        
        # Database connection
        elif data_source == "Database Connection":
            db_url = st.text_input(
                "Database URL",
                value="postgresql://postgres:postgres@localhost:5432/sales_analytics",
                help="PostgreSQL connection string"
            )
            
            if st.button("Connect to Database", type="primary"):
                with st.spinner("Connecting to database..."):
                    try:
                        db = DatabaseManager(db_url)
                        if db.connect():
                            st.session_state.db = db
                            st.success("✅ Database connected!")
                            
                            # Load data from database
                            query = "SELECT * FROM sales_data LIMIT 10000;"
                            df_clean = db.query_data(query)
                            if not df_clean.empty:
                                st.session_state.df_clean = df_clean
                                st.session_state.data_loaded = True
                                st.success("✅ Data loaded from database!")
                    except Exception as e:
                        st.error(f"❌ Database error: {str(e)}")
        
        st.divider()
        
        # Filters (shown only when data is loaded)
        if st.session_state.data_loaded:
            st.header("🔍 Filters")
            
            df_clean = st.session_state.df_clean
            
            # Date range filter
            if 'Order Date' in df_clean.columns or 'order_date' in df_clean.columns:
                date_col = 'Order Date' if 'Order Date' in df_clean.columns else 'order_date'
                df_clean[date_col] = pd.to_datetime(df_clean[date_col])
                
                min_date = df_clean[date_col].min().date()
                max_date = df_clean[date_col].max().date()
                
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                if len(date_range) == 2:
                    mask = (df_clean[date_col].dt.date >= date_range[0]) & \
                           (df_clean[date_col].dt.date <= date_range[1])
                    df_clean = df_clean[mask]
            
            # Category filter
            if 'Category' in df_clean.columns or 'category' in df_clean.columns:
                category_col = 'Category' if 'Category' in df_clean.columns else 'category'
                categories = ['All'] + sorted(df_clean[category_col].unique().tolist())
                selected_category = st.selectbox("Category", categories)
                
                if selected_category != 'All':
                    df_clean = df_clean[df_clean[category_col] == selected_category]
            
            # Region filter
            if 'Region' in df_clean.columns or 'region' in df_clean.columns:
                region_col = 'Region' if 'Region' in df_clean.columns else 'region'
                regions = ['All'] + sorted(df_clean[region_col].unique().tolist())
                selected_region = st.selectbox("Region", regions)
                
                if selected_region != 'All':
                    df_clean = df_clean[df_clean[region_col] == selected_region]
            
            # Customer Segment filter
            if 'Segment' in df_clean.columns or 'segment' in df_clean.columns:
                segment_col = 'Segment' if 'Segment' in df_clean.columns else 'segment'
                segments = ['All'] + sorted(df_clean[segment_col].unique().tolist())
                selected_segment = st.selectbox("Customer Segment", segments)
                
                if selected_segment != 'All':
                    df_clean = df_clean[df_clean[segment_col] == selected_segment]
            
            st.session_state.df_filtered = df_clean
    
    # Main content
    if not st.session_state.data_loaded:
        st.info("👈 Please load data using the sidebar options")
        st.markdown("""
        ### 📋 Expected Data Format
        
        Your CSV file should contain the following columns:
        - **Order Date**: Date of the order
        - **Sales**: Sales amount
        - **Profit**: Profit amount
        - **Quantity**: Quantity sold
        - **Category**: Product category
        - **Sub-Category**: Product sub-category
        - **Region**: Sales region
        - **State**: State/Province
        - **Segment**: Customer segment
        - **Customer Name**: Customer name
        - **Product Name**: Product name
        
        ### 🚀 Quick Start
        
        1. Upload your CSV file in the sidebar
        2. Click "Load Data"
        3. Explore the interactive dashboard!
        """)
    else:
        # Get filtered data
        df_display = st.session_state.get('df_filtered', st.session_state.df_clean)
        
        # KPI Cards
        st.header("📈 Key Performance Indicators")
        kpis = create_kpi_cards(df_display)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total Sales",
                value=f"${kpis['total_sales']:,.0f}",
                delta=f"{kpis['yoy_growth']:.1f}% YoY" if kpis['yoy_growth'] != 0 else None
            )
        
        with col2:
            st.metric(
                label="Total Profit",
                value=f"${kpis['total_profit']:,.0f}",
                delta=f"${kpis['total_profit']:,.0f}"
            )
        
        with col3:
            st.metric(
                label="Profit Margin",
                value=f"{kpis['profit_margin']:.2f}%",
                delta=f"{kpis['profit_margin']:.2f}%"
            )
        
        with col4:
            st.metric(
                label="Total Orders",
                value=f"{len(df_display):,}",
            )
        
        st.divider()
        
        # Charts Row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📅 Monthly Sales & Profit Trend")
            trend_fig = create_monthly_trend_chart(df_display)
            st.plotly_chart(trend_fig, use_container_width=True)
        
        with col2:
            st.subheader("📦 Profit by Category")
            category_fig = create_category_performance_chart(df_display)
            st.plotly_chart(category_fig, use_container_width=True)
        
        # Charts Row 2
        st.subheader("🗺️ Region vs Category Heatmap")
        heatmap_fig = create_region_heatmap(df_display)
        st.plotly_chart(heatmap_fig, use_container_width=True)
        
        # Charts Row 3
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top 10 Products by Profit")
            top_products_fig = create_top_products_chart(df_display, top_n=10)
            st.plotly_chart(top_products_fig, use_container_width=True)
        
        with col2:
            st.subheader("⚠️ Loss-Making Categories")
            loss_fig = create_loss_analysis_chart(df_display)
            st.plotly_chart(loss_fig, use_container_width=True)
        
        # Forecasting Section
        st.divider()
        st.header("🔮 Sales Forecasting")
        
        forecast_periods = st.slider(
            "Forecast Period (Months)",
            min_value=3,
            max_value=12,
            value=6,
            help="Number of months to forecast"
        )
        
        if st.button("Generate Forecast", type="primary"):
            with st.spinner("Training model and generating forecast..."):
                try:
                    # Use simple forecast for now
                    forecast_df = simple_forecast(df_display, periods=forecast_periods)
                    
                    if not forecast_df.empty:
                        forecast_fig = create_forecast_chart(df_display, forecast_df)
                        st.plotly_chart(forecast_fig, use_container_width=True)
                        
                        # Forecast summary
                        st.subheader("Forecast Summary")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_forecast = forecast_df['Forecasted_Sales'].mean()
                            st.metric("Average Monthly Forecast", f"${avg_forecast:,.0f}")
                        
                        with col2:
                            total_forecast = forecast_df['Forecasted_Sales'].sum()
                            st.metric("Total Forecasted Sales", f"${total_forecast:,.0f}")
                        
                        with col3:
                            # Calculate growth
                            date_col = 'Order Date' if 'Order Date' in df_display.columns else 'order_date'
                            sales_col = 'Sales' if 'Sales' in df_display.columns else 'sales'
                            
                            if date_col in df_display.columns:
                                df_display[date_col] = pd.to_datetime(df_display[date_col])
                                recent_months = df_display[df_display[date_col] >= df_display[date_col].max() - pd.DateOffset(months=3)]
                                recent_avg = recent_months[sales_col].mean() if len(recent_months) > 0 else 0
                                
                                if recent_avg > 0:
                                    growth = ((avg_forecast - recent_avg) / recent_avg) * 100
                                    st.metric("Expected Growth", f"{growth:.1f}%")
                    else:
                        st.error("Failed to generate forecast. Please check your data.")
                        
                except Exception as e:
                    st.error(f"Forecasting error: {str(e)}")
        
        # Data Summary
        st.divider()
        with st.expander("📊 Data Summary"):
            st.dataframe(df_display.head(100), use_container_width=True)
            st.write(f"**Total Rows:** {len(df_display):,}")


def generate_sample_data(n_rows: int = 1000) -> pd.DataFrame:
    """Generate sample sales data for demonstration"""
    np.random.seed(42)
    
    categories = ['Furniture', 'Office Supplies', 'Technology']
    sub_categories = {
        'Furniture': ['Chairs', 'Tables', 'Bookcases'],
        'Office Supplies': ['Paper', 'Binders', 'Storage'],
        'Technology': ['Phones', 'Computers', 'Accessories']
    }
    regions = ['West', 'East', 'Central', 'South']
    segments = ['Consumer', 'Corporate', 'Home Office']
    states = ['California', 'New York', 'Texas', 'Florida', 'Illinois']
    
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    
    data = {
        'Order Date': np.random.choice(dates, n_rows),
        'Sales': np.random.lognormal(mean=5, sigma=1, size=n_rows),
        'Quantity': np.random.randint(1, 10, n_rows),
        'Discount': np.random.uniform(0, 0.3, n_rows),
        'Category': np.random.choice(categories, n_rows),
        'Region': np.random.choice(regions, n_rows),
        'State': np.random.choice(states, n_rows),
        'Segment': np.random.choice(segments, n_rows),
        'Customer Name': [f'Customer_{i}' for i in range(n_rows)],
        'Product Name': [f'Product_{i}' for i in range(n_rows)]
    }
    
    df = pd.DataFrame(data)
    
    # Calculate profit (some categories have lower margins)
    profit_margins = {
        'Furniture': 0.05,  # Lower margin
        'Office Supplies': 0.15,
        'Technology': 0.20  # Higher margin
    }
    
    df['Profit'] = df.apply(
        lambda row: row['Sales'] * profit_margins[row['Category']] * (1 - row['Discount']) - 
                    np.random.uniform(0, row['Sales'] * 0.1),
        axis=1
    )
    
    # Add some loss-making orders
    loss_indices = np.random.choice(df.index, size=int(n_rows * 0.1), replace=False)
    df.loc[loss_indices, 'Profit'] = -np.abs(df.loc[loss_indices, 'Profit'])
    
    return clean_data(df)


if __name__ == "__main__":
    main()

