"""
LSTM và GRU Models - Deep Learning cho Time Series
Long Short-Term Memory và Gated Recurrent Unit
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataPreparator:
    """Chuẩn bị dữ liệu cho LSTM/GRU"""
    
    def __init__(self, lookback: int = 60, forecast_horizon: int = 1):
        """
        Args:
            lookback: Số timesteps nhìn lại (window size)
            forecast_horizon: Số bước dự đoán ra trước
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.fitted = False
    
    def create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tạo sequences cho training
        
        Args:
            data: Array 2D (samples, features)
        
        Returns:
            (X, y) where X shape = (samples, lookback, features)
                         y shape = (samples, forecast_horizon)
        """
        X, y = [], []
        
        for i in range(self.lookback, len(data) - self.forecast_horizon + 1):
            X.append(data[i - self.lookback:i])
            
            if self.forecast_horizon == 1:
                y.append(data[i, 0])  # Single-step prediction
            else:
                y.append(data[i:i + self.forecast_horizon, 0])  # Multi-step
        
        return np.array(X), np.array(y)
    
    def prepare_train_data(self, train_data: pd.DataFrame, 
                          target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """
        Chuẩn bị dữ liệu training
        
        Args:
            train_data: DataFrame với features
            target_col: Tên cột target để dự đoán
        
        Returns:
            (X_train, y_train)
        """
        # Đảm bảo target_col ở vị trí đầu tiên
        if target_col in train_data.columns:
            cols = [target_col] + [col for col in train_data.columns if col != target_col]
            train_data = train_data[cols]
        
        # Scale data
        scaled_data = self.scaler.fit_transform(train_data)
        self.fitted = True
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def prepare_test_data(self, test_data: pd.DataFrame,
                         target_col: str = 'close') -> Tuple[np.ndarray, np.ndarray]:
        """Chuẩn bị dữ liệu test"""
        if not self.fitted:
            raise ValueError("Scaler chưa được fit. Gọi prepare_train_data trước.")
        
        # Reorder columns như training
        if target_col in test_data.columns:
            cols = [target_col] + [col for col in test_data.columns if col != target_col]
            test_data = test_data[cols]
        
        # Scale
        scaled_data = self.scaler.transform(test_data)
        
        # Create sequences
        X, y = self.create_sequences(scaled_data)
        
        return X, y
    
    def inverse_transform_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Chuyển predictions về scale gốc"""
        # Tạo array với đúng số features như khi training
        n_features = self.scaler.n_features_in_
        
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        # Pad với zeros cho các features khác
        padded = np.zeros((len(predictions), n_features))
        padded[:, 0] = predictions.flatten()
        
        # Inverse transform
        inversed = self.scaler.inverse_transform(padded)
        
        return inversed[:, 0]


class LSTMModel:
    """
    LSTM Model Implementation
    
    Ưu điểm:
    - Xử lý tốt long-term dependencies
    - Tránh được vanishing gradient problem
    - Phù hợp với dữ liệu phi tuyến, phức tạp
    - Có thể học từ nhiều features
    
    Nhược điểm:
    - Cần nhiều dữ liệu
    - Training lâu
    - Dễ overfit
    - Khó giải thích
    
    Kiến trúc LSTM cell:
    - Forget gate: Quyết định thông tin nào bỏ qua
    - Input gate: Cập nhật cell state với thông tin mới
    - Output gate: Quyết định output dựa trên cell state
    """
    
    def __init__(self, 
                 lookback: int = 60,
                 forecast_horizon: int = 1,
                 units: List[int] = [50, 50],
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """
        Args:
            lookback: Số timesteps nhìn lại
            forecast_horizon: Số bước dự đoán
            units: List số units cho mỗi LSTM layer
            dropout: Dropout rate để tránh overfit
            bidirectional: Sử dụng Bidirectional LSTM
        """
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.units = units
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.model = None
        self.history = None
        self.data_preparator = TimeSeriesDataPreparator(lookback, forecast_horizon)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """
        Xây dựng kiến trúc LSTM
        
        Args:
            input_shape: (lookback, n_features)
        """
        model = Sequential()
        
        # First LSTM layer
        if self.bidirectional:
            model.add(Bidirectional(
                LSTM(self.units[0], return_sequences=len(self.units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(
                self.units[0],
                return_sequences=len(self.units) > 1,
                input_shape=input_shape
            ))
        
        model.add(Dropout(self.dropout))
        
        # Additional LSTM layers
        for i, units in enumerate(self.units[1:]):
            return_sequences = (i < len(self.units) - 2)
            
            if self.bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=return_sequences)))
            else:
                model.add(LSTM(units, return_sequences=return_sequences))
            
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(self.forecast_horizon))
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built LSTM model with {len(self.units)} layers")
        logger.info(f"Total parameters: {model.count_params():,}")
    
    def fit(self, train_data: pd.DataFrame, 
            target_col: str = 'close',
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1):
        """
        Huấn luyện LSTM model
        
        Args:
            train_data: DataFrame với features
            target_col: Cột target
            validation_split: Tỷ lệ validation
            epochs: Số epochs
            batch_size: Batch size
            verbose: Mức độ hiển thị (0, 1, 2)
        """
        try:
            logger.info("Preparing data for LSTM...")
            
            # Prepare data
            X_train, y_train = self.data_preparator.prepare_train_data(
                train_data, target_col
            )
            
            # Build model if not built
            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.build_model(input_shape)
            
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            logger.info("Training LSTM model...")
            
            # Train
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info("LSTM model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """
        Dự đoán với dữ liệu mới
        
        Args:
            data: DataFrame với features
            target_col: Cột target
        
        Returns:
            Predictions (đã inverse transform về scale gốc)
        """
        if self.model is None:
            raise ValueError("Model chưa được train")
        
        # Prepare data
        X, _ = self.data_preparator.prepare_test_data(data, target_col)
        
        # Predict
        predictions_scaled = self.model.predict(X, verbose=0)
        
        # Inverse transform
        predictions = self.data_preparator.inverse_transform_predictions(
            predictions_scaled
        )
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, 
                target_col: str = 'close') -> Dict[str, float]:
        """Đánh giá model trên test set"""
        # Get actual values
        actual = test_data[target_col].values[self.lookback:]
        
        # Predict
        predictions = self.predict(test_data, target_col)
        
        # Trim to same length
        min_len = min(len(actual), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]
        
        # Metrics
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    def save_model(self, filepath: str):
        """Lưu model"""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


class GRUModel:
    """
    GRU Model Implementation
    
    GRU (Gated Recurrent Unit) là phiên bản đơn giản hóa của LSTM
    
    Ưu điểm so với LSTM:
    - Ít parameters hơn (train nhanh hơn)
    - Hiệu quả tương đương LSTM trên nhiều tasks
    - Ít bị overfit hơn
    
    Nhược điểm:
    - Có thể kém hơn LSTM với long sequences
    
    Kiến trúc GRU cell:
    - Reset gate: Quyết định bỏ qua bao nhiêu past info
    - Update gate: Quyết định cập nhật state mới như thế nào
    """
    
    def __init__(self,
                 lookback: int = 60,
                 forecast_horizon: int = 1,
                 units: List[int] = [50, 50],
                 dropout: float = 0.2,
                 bidirectional: bool = False):
        """Tương tự LSTMModel"""
        self.lookback = lookback
        self.forecast_horizon = forecast_horizon
        self.units = units
        self.dropout = dropout
        self.bidirectional = bidirectional
        
        self.model = None
        self.history = None
        self.data_preparator = TimeSeriesDataPreparator(lookback, forecast_horizon)
    
    def build_model(self, input_shape: Tuple[int, int]):
        """Xây dựng kiến trúc GRU"""
        model = Sequential()
        
        # First GRU layer
        if self.bidirectional:
            model.add(Bidirectional(
                GRU(self.units[0], return_sequences=len(self.units) > 1),
                input_shape=input_shape
            ))
        else:
            model.add(GRU(
                self.units[0],
                return_sequences=len(self.units) > 1,
                input_shape=input_shape
            ))
        
        model.add(Dropout(self.dropout))
        
        # Additional GRU layers
        for i, units in enumerate(self.units[1:]):
            return_sequences = (i < len(self.units) - 2)
            
            if self.bidirectional:
                model.add(Bidirectional(GRU(units, return_sequences=return_sequences)))
            else:
                model.add(GRU(units, return_sequences=return_sequences))
            
            model.add(Dropout(self.dropout))
        
        # Output layer
        model.add(Dense(self.forecast_horizon))
        
        # Compile
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        self.model = model
        logger.info(f"Built GRU model with {len(self.units)} layers")
        logger.info(f"Total parameters: {model.count_params():,}")
    
    def fit(self, train_data: pd.DataFrame, 
            target_col: str = 'close',
            validation_split: float = 0.2,
            epochs: int = 100,
            batch_size: int = 32,
            verbose: int = 1):
        """Huấn luyện GRU model (code tương tự LSTM)"""
        try:
            logger.info("Preparing data for GRU...")
            
            X_train, y_train = self.data_preparator.prepare_train_data(
                train_data, target_col
            )
            
            if self.model is None:
                input_shape = (X_train.shape[1], X_train.shape[2])
                self.build_model(input_shape)
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7
                )
            ]
            
            logger.info("Training GRU model...")
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                callbacks=callbacks,
                verbose=verbose
            )
            
            logger.info("GRU model trained successfully")
            
        except Exception as e:
            logger.error(f"Error training GRU: {str(e)}")
            raise
    
    def predict(self, data: pd.DataFrame, target_col: str = 'close') -> np.ndarray:
        """Dự đoán (tương tự LSTM)"""
        if self.model is None:
            raise ValueError("Model chưa được train")
        
        X, _ = self.data_preparator.prepare_test_data(data, target_col)
        predictions_scaled = self.model.predict(X, verbose=0)
        predictions = self.data_preparator.inverse_transform_predictions(predictions_scaled)
        
        return predictions
    
    def evaluate(self, test_data: pd.DataFrame, 
                target_col: str = 'close') -> Dict[str, float]:
        """Đánh giá model"""
        actual = test_data[target_col].values[self.lookback:]
        predictions = self.predict(test_data, target_col)
        
        min_len = min(len(actual), len(predictions))
        actual = actual[:min_len]
        predictions = predictions[:min_len]
        
        mae = np.mean(np.abs(actual - predictions))
        rmse = np.sqrt(np.mean((actual - predictions) ** 2))
        mape = np.mean(np.abs((actual - predictions) / actual)) * 100
        
        return {'mae': mae, 'rmse': rmse, 'mape': mape}
    
    def save_model(self, filepath: str):
        """Lưu model"""
        if self.model is not None:
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model"""
        self.model = keras.models.load_model(filepath)
        logger.info(f"Model loaded from {filepath}")


if __name__ == "__main__":
    print("LSTM and GRU Models for Time Series Forecasting")
    print("=" * 60)
    print("\nKey Differences:")
    print("- LSTM: 3 gates (forget, input, output) - More parameters")
    print("- GRU: 2 gates (reset, update) - Faster, less parameters")
    print("\nBoth are excellent for:")
    print("- Non-linear time series")
    print("- Multi-variate forecasting")
    print("- Learning complex patterns")
