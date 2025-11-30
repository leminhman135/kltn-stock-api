"""
ARIMA Model (AutoRegressive Integrated Moving Average)
Mô hình thống kê cổ điển cho dự đoán chuỗi thời gian
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ARIMAModel:
    """
    ARIMA Model Implementation
    
    ARIMA(p, d, q):
    - p: số lag observations (AR - AutoRegressive)
    - d: bậc sai phân (I - Integrated)
    - q: kích thước của moving average window (MA - Moving Average)
    
    Ưu điểm:
    - Đơn giản, dễ hiểu và giải thích
    - Hiệu quả với dữ liệu stationary
    - Không cần nhiều dữ liệu huấn luyện
    
    Nhược điểm:
    - Chỉ phù hợp với dữ liệu tuyến tính
    - Cần dữ liệu stationary
    - Không xử lý được nhiều biến đầu vào
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        """
        Args:
            order: (p, d, q) parameters cho ARIMA
        """
        self.order = order
        self.model = None
        self.fitted_model = None
        self.history = []
    
    def check_stationarity(self, data: pd.Series, 
                          significance_level: float = 0.05) -> Dict:
        """
        Kiểm tra tính dừng (stationarity) của chuỗi thời gian
        Sử dụng Augmented Dickey-Fuller test
        
        Returns:
            Dictionary với kết quả test
        """
        result = adfuller(data.dropna())
        
        is_stationary = result[1] < significance_level
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': is_stationary
        }
    
    def make_stationary(self, data: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Làm dữ liệu trở nên stationary bằng differencing
        
        Returns:
            (stationary_data, number_of_differences)
        """
        diff_data = data.copy()
        n_diff = 0
        
        for i in range(max_diff):
            stationarity = self.check_stationarity(diff_data)
            
            if stationarity['is_stationary']:
                break
            
            diff_data = diff_data.diff().dropna()
            n_diff += 1
        
        logger.info(f"Applied {n_diff} differencing to achieve stationarity")
        return diff_data, n_diff
    
    def auto_select_order(self, data: pd.Series, 
                         max_p: int = 5, max_q: int = 5) -> Tuple[int, int, int]:
        """
        Tự động chọn order tối ưu dựa trên AIC (Akaike Information Criterion)
        
        Returns:
            (p, d, q) tối ưu
        """
        logger.info("Auto-selecting ARIMA order...")
        
        # Xác định d
        _, d = self.make_stationary(data)
        
        best_aic = float('inf')
        best_order = (1, d, 1)
        
        for p in range(max_p + 1):
            for q in range(max_q + 1):
                try:
                    model = StatsARIMA(data, order=(p, d, q))
                    fitted = model.fit()
                    
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                
                except:
                    continue
        
        logger.info(f"Best order: {best_order} with AIC: {best_aic:.2f}")
        return best_order
    
    def fit(self, train_data: pd.Series, auto_order: bool = False):
        """
        Huấn luyện mô hình ARIMA
        
        Args:
            train_data: Chuỗi thời gian để huấn luyện
            auto_order: Tự động chọn order nếu True
        """
        try:
            if auto_order:
                self.order = self.auto_select_order(train_data)
            
            logger.info(f"Fitting ARIMA{self.order}...")
            
            self.model = StatsARIMA(train_data, order=self.order)
            self.fitted_model = self.model.fit()
            self.history = train_data.tolist()
            
            logger.info("Model fitted successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            logger.info(f"BIC: {self.fitted_model.bic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            raise
    
    def predict(self, steps: int = 1) -> np.ndarray:
        """
        Dự đoán các bước tiếp theo
        
        Args:
            steps: Số bước cần dự đoán
        
        Returns:
            Array các giá trị dự đoán
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được huấn luyện. Gọi fit() trước.")
        
        forecast = self.fitted_model.forecast(steps=steps)
        return np.array(forecast)
    
    def predict_with_confidence(self, steps: int = 1, 
                               alpha: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Dự đoán với confidence intervals
        
        Args:
            steps: Số bước cần dự đoán
            alpha: Mức ý nghĩa (0.05 cho 95% confidence interval)
        
        Returns:
            (predictions, lower_bounds, upper_bounds)
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        forecast_obj = self.fitted_model.get_forecast(steps=steps)
        predictions = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=alpha)
        
        return (
            predictions.values,
            conf_int.iloc[:, 0].values,
            conf_int.iloc[:, 1].values
        )
    
    def evaluate(self, test_data: pd.Series) -> Dict[str, float]:
        """
        Đánh giá mô hình trên tập test
        
        Returns:
            Dictionary với các metrics
        """
        predictions = self.predict(steps=len(test_data))
        
        mae = np.mean(np.abs(test_data - predictions))
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def get_summary(self) -> str:
        """Lấy thông tin tóm tắt của mô hình"""
        if self.fitted_model is None:
            return "Model chưa được huấn luyện"
        
        return str(self.fitted_model.summary())


class SARIMAXModel:
    """
    SARIMAX Model - ARIMA with Seasonal and Exogenous variables
    
    SARIMAX(p,d,q)(P,D,Q,s):
    - (p,d,q): ARIMA parameters
    - (P,D,Q,s): Seasonal parameters (s = seasonal period)
    - Exogenous variables: các biến ngoại sinh (technical indicators, sentiment, etc.)
    
    Ưu điểm:
    - Xử lý được seasonality
    - Cho phép thêm biến ngoại sinh
    - Linh hoạt hơn ARIMA cơ bản
    """
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1),
                 seasonal_order: Tuple[int, int, int, int] = (1, 1, 1, 12)):
        """
        Args:
            order: (p, d, q) cho ARIMA
            seasonal_order: (P, D, Q, s) cho seasonal component
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.fitted_model = None
    
    def fit(self, train_data: pd.Series, exog: Optional[pd.DataFrame] = None):
        """
        Huấn luyện SARIMAX model
        
        Args:
            train_data: Chuỗi thời gian target
            exog: DataFrame các biến ngoại sinh (optional)
        """
        try:
            logger.info(f"Fitting SARIMAX{self.order}x{self.seasonal_order}...")
            
            self.model = SARIMAX(
                train_data,
                exog=exog,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            self.fitted_model = self.model.fit(disp=False)
            
            logger.info("SARIMAX model fitted successfully")
            logger.info(f"AIC: {self.fitted_model.aic:.2f}")
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX: {str(e)}")
            raise
    
    def predict(self, steps: int = 1, exog: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Dự đoán với SARIMAX
        
        Args:
            steps: Số bước dự đoán
            exog: Dữ liệu biến ngoại sinh cho các bước dự đoán
        """
        if self.fitted_model is None:
            raise ValueError("Model chưa được huấn luyện")
        
        forecast = self.fitted_model.forecast(steps=steps, exog=exog)
        return np.array(forecast)
    
    def evaluate(self, test_data: pd.Series, exog: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """Đánh giá SARIMAX model"""
        predictions = self.predict(steps=len(test_data), exog=exog)
        
        mae = np.mean(np.abs(test_data - predictions))
        rmse = np.sqrt(np.mean((test_data - predictions) ** 2))
        mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }


if __name__ == "__main__":
    # Test ARIMA model
    from datetime import datetime, timedelta
    
    # Tạo dữ liệu mẫu
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    # Dữ liệu có xu hướng và noise
    trend = np.linspace(100, 150, 200)
    seasonal = 10 * np.sin(np.linspace(0, 4*np.pi, 200))
    noise = np.random.normal(0, 2, 200)
    data = pd.Series(trend + seasonal + noise, index=dates)
    
    # Chia train/test
    train_size = int(len(data) * 0.8)
    train, test = data[:train_size], data[train_size:]
    
    print("Testing ARIMA Model...")
    print(f"Training size: {len(train)}, Test size: {len(test)}")
    
    # Train model
    arima = ARIMAModel()
    stationarity = arima.check_stationarity(train)
    print(f"\nStationarity test: {stationarity}")
    
    arima.fit(train, auto_order=True)
    
    # Predict
    predictions = arima.predict(steps=len(test))
    
    # Evaluate
    metrics = arima.evaluate(test)
    print(f"\nEvaluation Metrics:")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAPE: {metrics['mape']:.2f}%")
