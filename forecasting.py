"""
Forecasting Module
Implements ML models for sales prediction
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SalesForecaster:
    """Sales forecasting using ML models"""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize forecaster
        
        Args:
            model_type: 'linear' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.is_trained = False
        
        if model_type == 'linear':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("Model type must be 'linear' or 'random_forest'")
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for forecasting
        
        Args:
            df: DataFrame with date and sales columns
            
        Returns:
            DataFrame with engineered features
        """
        df_features = df.copy()
        
        # Ensure date column is datetime
        if 'Order Date' in df_features.columns:
            date_col = 'Order Date'
        elif 'order_date' in df_features.columns:
            date_col = 'order_date'
        else:
            raise ValueError("Date column not found")
        
        df_features[date_col] = pd.to_datetime(df_features[date_col])
        
        # Aggregate by month
        df_features['YearMonth'] = df_features[date_col].dt.to_period('M')
        
        # Group by month and sum sales
        monthly_data = df_features.groupby('YearMonth').agg({
            'Sales' if 'Sales' in df_features.columns else 'sales': 'sum',
            'Profit' if 'Profit' in df_features.columns else 'profit': 'sum'
        }).reset_index()
        
        monthly_data['YearMonth'] = monthly_data['YearMonth'].astype(str)
        monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'])
        
        # Create time-based features
        monthly_data['Month'] = monthly_data['Date'].dt.month
        monthly_data['Quarter'] = monthly_data['Date'].dt.quarter
        monthly_data['Year'] = monthly_data['Date'].dt.year
        
        # Create sequential index
        monthly_data = monthly_data.sort_values('Date')
        monthly_data['TimeIndex'] = range(len(monthly_data))
        
        # Lag features
        sales_col = 'Sales' if 'Sales' in monthly_data.columns else 'sales'
        monthly_data['Sales_Lag1'] = monthly_data[sales_col].shift(1)
        monthly_data['Sales_Lag2'] = monthly_data[sales_col].shift(2)
        monthly_data['Sales_Lag3'] = monthly_data[sales_col].shift(3)
        
        # Rolling averages
        monthly_data['Sales_MA3'] = monthly_data[sales_col].rolling(window=3).mean()
        monthly_data['Sales_MA6'] = monthly_data[sales_col].rolling(window=6).mean()
        
        # Trend
        monthly_data['Trend'] = monthly_data['TimeIndex']
        
        # Remove rows with NaN (from lag features)
        monthly_data = monthly_data.dropna()
        
        return monthly_data
    
    def train(self, df: pd.DataFrame):
        """
        Train the forecasting model
        
        Args:
            df: DataFrame with historical sales data
        """
        try:
            # Prepare features
            monthly_data = self.prepare_features(df)
            
            if len(monthly_data) < 12:
                print("⚠️ Warning: Insufficient data for training (need at least 12 months)")
                return False
            
            # Select features
            feature_cols = [
                'Month', 'Quarter', 'Year', 'TimeIndex',
                'Sales_Lag1', 'Sales_Lag2', 'Sales_Lag3',
                'Sales_MA3', 'Sales_MA6', 'Trend'
            ]
            
            # Filter available features
            available_features = [col for col in feature_cols if col in monthly_data.columns]
            
            X = monthly_data[available_features]
            sales_col = 'Sales' if 'Sales' in monthly_data.columns else 'sales'
            y = monthly_data[sales_col]
            
            # Train model
            self.model.fit(X, y)
            self.is_trained = True
            
            # Calculate training metrics
            y_pred = self.model.predict(X)
            mae = mean_absolute_error(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            print(f"✅ Model trained successfully")
            print(f"   MAE: ${mae:,.2f}")
            print(f"   RMSE: ${rmse:,.2f}")
            print(f"   R² Score: {r2:.3f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Training error: {str(e)}")
            return False
    
    def forecast(self, periods: int = 6, last_date: pd.Timestamp = None) -> pd.DataFrame:
        """
        Generate sales forecast
        
        Args:
            periods: Number of months to forecast
            last_date: Last date in historical data
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.is_trained:
            print("❌ Model not trained. Please train the model first.")
            return pd.DataFrame()
        
        try:
            forecasts = []
            current_date = last_date if last_date else pd.Timestamp.now()
            
            for i in range(1, periods + 1):
                forecast_date = current_date + pd.DateOffset(months=i)
                
                features = {
                    'Month': forecast_date.month,
                    'Quarter': (forecast_date.month - 1) // 3 + 1,
                    'Year': forecast_date.year,
                    'TimeIndex': i,
                    'Trend': i
                }
                
                feature_vector = pd.DataFrame([features])
                prediction = self.model.predict(feature_vector)[0]
                
                forecasts.append({
                    'Date': forecast_date,
                    'Forecasted_Sales': max(0, prediction),
                    'Month': forecast_date.month,
                    'Year': forecast_date.year
                })
            
            forecast_df = pd.DataFrame(forecasts)
            return forecast_df
            
        except Exception as e:
            print(f"❌ Forecasting error: {str(e)}")
            return pd.DataFrame()
    
    def forecast_from_data(self, df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
        """
        Generate forecast directly from historical data
        
        Args:
            df: Historical sales data
            periods: Number of months to forecast
            
        Returns:
            DataFrame with forecasted values
        """
        if not self.is_trained:
            self.train(df)
        
        monthly_data = self.prepare_features(df)
        last_date = monthly_data['Date'].max()
        
        forecast = self.forecast(periods=periods, last_date=last_date)
        
        return forecast


def simple_forecast(df: pd.DataFrame, periods: int = 6) -> pd.DataFrame:
    """
    Simple forecasting using moving average and trend
    
    Args:
        df: Historical sales data
        periods: Number of months to forecast
        
    Returns:
        DataFrame with forecasted values
    """
    try:
        date_col = 'Order Date' if 'Order Date' in df.columns else 'order_date'
        sales_col = 'Sales' if 'Sales' in df.columns else 'sales'
        
        df[date_col] = pd.to_datetime(df[date_col])
        df['YearMonth'] = df[date_col].dt.to_period('M')
        
        monthly_data = df.groupby('YearMonth')[sales_col].sum().reset_index()
        monthly_data['Date'] = pd.to_datetime(monthly_data['YearMonth'].astype(str))
        monthly_data = monthly_data.sort_values('Date')
        
        monthly_data['TimeIndex'] = range(len(monthly_data))
        
        from sklearn.linear_model import LinearRegression
        X = monthly_data[['TimeIndex']]
        y = monthly_data[sales_col]
        
        model = LinearRegression()
        model.fit(X, y)
        
        last_date = monthly_data['Date'].max()
        last_index = monthly_data['TimeIndex'].max()
        
        forecasts = []
        for i in range(1, periods + 1):
            forecast_date = last_date + pd.DateOffset(months=i)
            future_index = last_index + i
            prediction = model.predict([[future_index]])[0]
            
            forecasts.append({
                'Date': forecast_date,
                'Forecasted_Sales': max(0, prediction),
                'Month': forecast_date.month,
                'Year': forecast_date.year
            })
        
        return pd.DataFrame(forecasts)
        
    except Exception as e:
        print(f"❌ Simple forecast error: {str(e)}")
        return pd.DataFrame()


if __name__ == "__main__":
    print("Testing forecasting module...")
    pass

