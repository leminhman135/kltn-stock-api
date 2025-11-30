"""
Prophet Model - Facebook's time series forecasting model
Mô hình mạnh mẽ cho dự đoán chuỗi thời gian với trend và seasonality
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProphetModel:
    """
    Facebook Prophet Implementation
    
    Ưu điểm:
    - Xử lý tốt missing data và outliers
    - Tự động phát hiện seasonality (yearly, weekly, daily)
    - Dễ sử dụng, không cần nhiều tuning
    - Xử lý tốt holidays và special events
    - Robust với dữ liệu phi tuyến tính
    
    Nhược điểm:
    - Chậm hơn so với ARIMA
    - Cần nhiều dữ liệu hơn (ít nhất vài tháng)
    - Không tốt cho dự đoán ngắn hạn (< 1 ngày)
    
    Cấu trúc:
    y(t) = g(t) + s(t) + h(t) + e(t)
    - g(t): trend (piecewise linear or logistic growth)
    - s(t): seasonality (Fourier series)
    - h(t): holidays effects
    - e(t): error term
    """
    
    def __init__(self, 
                 growth: str = 'linear',
                 yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True,
                 daily_seasonality: bool = False,
                 seasonality_mode: str = 'additive',
                 changepoint_prior_scale: float = 0.05):
        """
        Args:
            growth: 'linear' hoặc 'logistic'
            yearly_seasonality: Có sử dụng yearly seasonality
            weekly_seasonality: Có sử dụng weekly seasonality
            daily_seasonality: Có sử dụng daily seasonality
            seasonality_mode: 'additive' hoặc 'multiplicative'
            changepoint_prior_scale: Điều chỉnh flexibility của trend (0.001-0.5)
        """
        self.model = Prophet(
            growth=growth,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            seasonality_mode=seasonality_mode,
            changepoint_prior_scale=changepoint_prior_scale
        )
        
        self.fitted = False
        self.train_data = None
    
    def prepare_data(self, data: pd.Series, date_col: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Chuẩn bị dữ liệu cho Prophet
        Prophet yêu cầu DataFrame với 2 cột: 'ds' (date) và 'y' (value)
        
        Args:
            data: Series giá trị cần dự đoán
            date_col: Series ngày tháng (nếu data không có index datetime)
        
        Returns:
            DataFrame với format Prophet
        """
        df = pd.DataFrame()
        
        if date_col is not None:
            df['ds'] = pd.to_datetime(date_col)
        elif isinstance(data.index, pd.DatetimeIndex):
            df['ds'] = data.index
        else:
            raise ValueError("Cần có thông tin date: hoặc data có DatetimeIndex hoặc truyền date_col")
        
        df['y'] = data.values
        
        return df
    
    def add_regressor(self, name: str, prior_scale: float = 10.0, 
                     standardize: str = 'auto', mode: str = 'additive'):
        """
        Thêm regressor (biến ngoại sinh) vào mô hình
        
        Args:
            name: Tên regressor
            prior_scale: Độ mạnh của regressor
            standardize: 'auto', True, hoặc False
            mode: 'additive' hoặc 'multiplicative'
        """
        self.model.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize,
            mode=mode
        )
        logger.info(f"Added regressor: {name}")
    
    def add_seasonality(self, name: str, period: float, fourier_order: int,
                       prior_scale: float = 10.0, mode: str = 'additive'):
        """
        Thêm custom seasonality
        
        Args:
            name: Tên seasonality
            period: Chu kỳ (days) - vd: 30 cho monthly
            fourier_order: Số harmonics (càng cao càng flexible, thường 3-10)
            prior_scale: Độ mạnh của seasonality
            mode: 'additive' hoặc 'multiplicative'
        """
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale,
            mode=mode
        )
        logger.info(f"Added seasonality: {name} with period {period} days")
    
    def fit(self, train_data: pd.Series, date_col: Optional[pd.Series] = None,
            regressors: Optional[pd.DataFrame] = None):
        """
        Huấn luyện Prophet model
        
        Args:
            train_data: Chuỗi thời gian để train
            date_col: Cột ngày tháng
            regressors: DataFrame chứa các biến ngoại sinh
        """
        try:
            logger.info("Fitting Prophet model...")
            
            # Chuẩn bị dữ liệu
            df = self.prepare_data(train_data, date_col)
            
            # Thêm regressors nếu có
            if regressors is not None:
                for col in regressors.columns:
                    if col not in df.columns:
                        df[col] = regressors[col].values
            
            # Fit model
            self.model.fit(df)
            self.fitted = True
            self.train_data = df
            
            logger.info("Prophet model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting Prophet: {str(e)}")
            raise
    
    def predict(self, periods: int = 1, freq: str = 'D',
               future_regressors: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Dự đoán các bước tiếp theo
        
        Args:
            periods: Số periods cần dự đoán
            freq: Tần suất ('D' = daily, 'W' = weekly, 'M' = monthly)
            future_regressors: DataFrame regressors cho future periods
        
        Returns:
            DataFrame với predictions và confidence intervals
        """
        if not self.fitted:
            raise ValueError("Model chưa được huấn luyện")
        
        # Tạo future dataframe
        future = self.model.make_future_dataframe(periods=periods, freq=freq)
        
        # Thêm future regressors nếu có
        if future_regressors is not None:
            for col in future_regressors.columns:
                # Cần extend regressors cho cả historical và future
                if col in self.train_data.columns:
                    historical_values = self.train_data[col].values
                    future_values = future_regressors[col].values
                    combined = np.concatenate([historical_values, future_values])
                    future[col] = combined[:len(future)]
        
        # Dự đoán
        forecast = self.model.predict(future)
        
        return forecast
    
    def get_forecast_values(self, periods: int = 1, freq: str = 'D',
                           future_regressors: Optional[pd.DataFrame] = None) -> Dict[str, np.ndarray]:
        """
        Lấy giá trị dự đoán dạng numpy array
        
        Returns:
            Dictionary với 'predictions', 'lower_bound', 'upper_bound'
        """
        forecast = self.predict(periods, freq, future_regressors)
        
        # Chỉ lấy các dòng future (không phải historical)
        future_forecast = forecast.tail(periods)
        
        return {
            'predictions': future_forecast['yhat'].values,
            'lower_bound': future_forecast['yhat_lower'].values,
            'upper_bound': future_forecast['yhat_upper'].values,
            'dates': future_forecast['ds'].values
        }
    
    def evaluate(self, test_data: pd.Series, date_col: Optional[pd.Series] = None,
                regressors: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Đánh giá mô hình trên test set
        
        Returns:
            Dictionary với các metrics
        """
        if not self.fitted:
            raise ValueError("Model chưa được huấn luyện")
        
        # Chuẩn bị test data
        test_df = self.prepare_data(test_data, date_col)
        
        if regressors is not None:
            for col in regressors.columns:
                test_df[col] = regressors[col].values
        
        # Predict
        forecast = self.model.predict(test_df)
        predictions = forecast['yhat'].values
        actuals = test_data.values
        
        # Calculate metrics
        mae = np.mean(np.abs(actuals - predictions))
        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def cross_validate_model(self, initial: str = '730 days', 
                            period: str = '180 days',
                            horizon: str = '90 days') -> pd.DataFrame:
        """
        Thực hiện cross-validation
        
        Args:
            initial: Kích thước training set ban đầu
            period: Khoảng cách giữa các cutoff dates
            horizon: Forecast horizon
        
        Returns:
            DataFrame với cross-validation results
        """
        if not self.fitted:
            raise ValueError("Model chưa được huấn luyện")
        
        logger.info("Performing cross-validation...")
        
        df_cv = cross_validation(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon
        )
        
        df_metrics = performance_metrics(df_cv)
        
        logger.info("Cross-validation completed")
        return df_metrics
    
    def plot_components(self):
        """Plot các components của model (trend, seasonality)"""
        if not self.fitted:
            raise ValueError("Model chưa được huấn luyện")
        
        forecast = self.predict(periods=0)  # Get full historical forecast
        fig = self.model.plot_components(forecast)
        return fig


if __name__ == "__main__":
    # Test Prophet model
    from datetime import datetime, timedelta
    
    # Tạo dữ liệu mẫu với seasonality
    dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
    
    # Trend + yearly seasonality + weekly seasonality + noise
    t = np.arange(len(dates))
    trend = 100 + 0.05 * t
    yearly = 20 * np.sin(2 * np.pi * t / 365.25)
    weekly = 5 * np.sin(2 * np.pi * t / 7)
    noise = np.random.normal(0, 3, len(dates))
    
    data = pd.Series(trend + yearly + weekly + noise, index=dates)
    
    # Train/test split
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    print("Testing Prophet Model...")
    print(f"Training size: {len(train)}, Test size: {len(test)}")
    
    # Initialize and fit model
    prophet = ProphetModel(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    
    prophet.fit(train)
    
    # Predict
    forecast_result = prophet.get_forecast_values(periods=len(test))
    predictions = forecast_result['predictions']
    
    # Evaluate
    metrics = prophet.evaluate(test)
    print(f"\nEvaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
    
    print(f"\nSample predictions (first 5):")
    print(predictions[:5])
