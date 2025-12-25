"""
Database Module
Handles PostgreSQL connection and data operations
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
import os
from typing import Optional


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize database connection
        
        Args:
            connection_string: PostgreSQL connection string
                Format: postgresql://user:password@host:port/database
        """
        if connection_string:
            self.connection_string = connection_string
        else:
            # Default connection (can be overridden with environment variables)
            self.connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://postgres:postgres@localhost:5432/sales_analytics'
            )
        
        self.engine = None
        self.conn = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.connection_string)
            self.conn = self.engine.connect()
            print("✅ Database connection established")
            return True
        except Exception as e:
            print(f"❌ Database connection error: {str(e)}")
            return False
    
    def create_schema(self):
        """Create database schema and tables"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS sales_data (
            id SERIAL PRIMARY KEY,
            order_id VARCHAR(50),
            order_date DATE,
            ship_date DATE,
            ship_mode VARCHAR(50),
            customer_id VARCHAR(50),
            customer_name VARCHAR(255),
            segment VARCHAR(50),
            country VARCHAR(100),
            city VARCHAR(100),
            state VARCHAR(100),
            postal_code VARCHAR(20),
            region VARCHAR(50),
            product_id VARCHAR(50),
            category VARCHAR(50),
            sub_category VARCHAR(100),
            product_name VARCHAR(500),
            sales DECIMAL(10, 2),
            quantity INTEGER,
            discount DECIMAL(5, 4),
            profit DECIMAL(10, 2),
            profit_margin DECIMAL(5, 2),
            year INTEGER,
            month INTEGER,
            quarter INTEGER,
            is_loss BOOLEAN,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_order_date ON sales_data(order_date);
        CREATE INDEX IF NOT EXISTS idx_category ON sales_data(category);
        CREATE INDEX IF NOT EXISTS idx_region ON sales_data(region);
        CREATE INDEX IF NOT EXISTS idx_customer_id ON sales_data(customer_id);
        """
        
        try:
            with self.engine.connect() as conn:
                # Execute each statement separately
                statements = create_table_sql.strip().split(';')
                for statement in statements:
                    if statement.strip():
                        conn.execute(text(statement))
                conn.commit()
            print("✅ Database schema created successfully")
            return True
        except Exception as e:
            print(f"❌ Schema creation error: {str(e)}")
            return False
    
    def insert_data(self, df: pd.DataFrame, table_name: str = 'sales_data', if_exists: str = 'replace'):
        """
        Insert DataFrame into database table
        
        Args:
            df: DataFrame to insert
            table_name: Target table name
            if_exists: 'replace', 'append', or 'fail'
        """
        try:
            # Map DataFrame columns to database columns
            column_mapping = {
                'Order ID': 'order_id',
                'Order Date': 'order_date',
                'Ship Date': 'ship_date',
                'Ship Mode': 'ship_mode',
                'Customer ID': 'customer_id',
                'Customer Name': 'customer_name',
                'Segment': 'segment',
                'Country': 'country',
                'City': 'city',
                'State': 'state',
                'Postal Code': 'postal_code',
                'Region': 'region',
                'Product ID': 'product_id',
                'Category': 'category',
                'Sub-Category': 'sub_category',
                'Product Name': 'product_name',
                'Sales': 'sales',
                'Quantity': 'quantity',
                'Discount': 'discount',
                'Profit': 'profit',
                'Profit Margin': 'profit_margin',
                'Year': 'year',
                'Month': 'month',
                'Quarter': 'quarter',
                'Is Loss': 'is_loss'
            }
            
            # Rename columns
            df_db = df.copy()
            for old_col, new_col in column_mapping.items():
                if old_col in df_db.columns:
                    df_db = df_db.rename(columns={old_col: new_col})
            
            # Select only columns that exist in both DataFrame and mapping
            available_cols = [col for col in df_db.columns if col in column_mapping.values()]
            df_db = df_db[available_cols]
            
            # Insert into database
            df_db.to_sql(
                table_name,
                self.engine,
                if_exists=if_exists,
                index=False,
                method='multi'
            )
            
            print(f"✅ Data inserted successfully: {len(df_db)} rows")
            return True
        except Exception as e:
            print(f"❌ Data insertion error: {str(e)}")
            return False
    
    def query_data(self, query: str) -> pd.DataFrame:
        """
        Execute SQL query and return DataFrame
        
        Args:
            query: SQL query string
            
        Returns:
            DataFrame with query results
        """
        try:
            df = pd.read_sql_query(query, self.engine)
            return df
        except Exception as e:
            print(f"❌ Query error: {str(e)}")
            return pd.DataFrame()
    
    def get_kpi_data(self) -> dict:
        """Get KPI metrics from database"""
        query = """
        SELECT 
            SUM(sales) as total_sales,
            SUM(profit) as total_profit,
            AVG(profit_margin) as avg_profit_margin,
            COUNT(*) as total_orders,
            SUM(CASE WHEN is_loss THEN 1 ELSE 0 END) as loss_orders
        FROM sales_data;
        """
        
        result = self.query_data(query)
        if not result.empty:
            return result.iloc[0].to_dict()
        return {}
    
    def get_category_performance(self) -> pd.DataFrame:
        """Get category-wise performance"""
        query = """
        SELECT 
            category,
            SUM(sales) as total_sales,
            SUM(profit) as total_profit,
            AVG(profit_margin) as avg_profit_margin,
            COUNT(*) as order_count
        FROM sales_data
        GROUP BY category
        ORDER BY total_profit DESC;
        """
        return self.query_data(query)
    
    def get_region_performance(self) -> pd.DataFrame:
        """Get region-wise performance"""
        query = """
        SELECT 
            region,
            SUM(sales) as total_sales,
            SUM(profit) as total_profit,
            AVG(profit_margin) as avg_profit_margin,
            COUNT(*) as order_count
        FROM sales_data
        GROUP BY region
        ORDER BY total_profit DESC;
        """
        return self.query_data(query)
    
    def get_monthly_trends(self) -> pd.DataFrame:
        """Get monthly sales and profit trends"""
        query = """
        SELECT 
            year,
            month,
            SUM(sales) as monthly_sales,
            SUM(profit) as monthly_profit,
            COUNT(*) as order_count
        FROM sales_data
        GROUP BY year, month
        ORDER BY year, month;
        """
        return self.query_data(query)
    
    def get_top_products(self, limit: int = 10) -> pd.DataFrame:
        """Get top products by profit"""
        query = f"""
        SELECT 
            product_name,
            category,
            sub_category,
            SUM(sales) as total_sales,
            SUM(profit) as total_profit,
            AVG(profit_margin) as avg_profit_margin
        FROM sales_data
        GROUP BY product_name, category, sub_category
        ORDER BY total_profit DESC
        LIMIT {limit};
        """
        return self.query_data(query)
    
    def get_top_customers(self, limit: int = 10) -> pd.DataFrame:
        """Get top customers by sales"""
        query = f"""
        SELECT 
            customer_name,
            segment,
            SUM(sales) as total_sales,
            SUM(profit) as total_profit,
            COUNT(*) as order_count
        FROM sales_data
        GROUP BY customer_name, segment
        ORDER BY total_sales DESC
        LIMIT {limit};
        """
        return self.query_data(query)
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
        if self.engine:
            self.engine.dispose()
        print("✅ Database connection closed")


if __name__ == "__main__":
    # Test database connection
    print("Testing database module...")
    db = DatabaseManager()
    if db.connect():
        db.create_schema()
        db.close()

