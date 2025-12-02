"""
Stock Price Prediction Service
Tích hợp tất cả các models: ARIMA, Prophet, LSTM, GRU, XGBoost, Ensemble
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
import joblib
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import các models
try:
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠️ scikit-learn not installed")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    logger.warning("⚠️ XGBoost not installed")

try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False
    logger.warning("⚠️ Prophet not installed")

try:
    from statsmodels.tsa.arima.model import ARIMA
    HAS_ARIMA = True
except ImportError:
    HAS_ARIMA = False
    logger.warning("⚠️ statsmodels not installed")

try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
    logger.warning("⚠️ TensorFlow not installed")


class TechnicalIndicators:
    """Tính các chỉ báo kỹ thuật"""
    
    @staticmethod
    def calculate_sma(prices: np.ndarray, window: int) -> np.ndarray:
        """Simple Moving Average"""
        result = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            result[i] = np.mean(prices[i - window + 1:i + 1])
        return result
    
    @staticmethod
    def calculate_ema(prices: np.ndarray, window: int) -> np.ndarray:
        """Exponential Moving Average"""
        alpha = 2 / (window + 1)
        result = np.zeros(len(prices))
        result[0] = prices[0]
        for i in range(1, len(prices)):
            result[i] = alpha * prices[i] + (1 - alpha) * result[i - 1]
        return result
    
    @staticmethod
    def calculate_rsi(prices: np.ndarray, window: int = 14) -> np.ndarray:
        """Relative Strength Index"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        result = np.full(len(prices), 50.0)
        
        for i in range(window, len(prices)):
            avg_gain = np.mean(gains[i - window:i])
            avg_loss = np.mean(losses[i - window:i])
            if avg_loss == 0:
                result[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                result[i] = 100 - (100 / (1 + rs))
        
        return result
    
    @staticmethod
    def calculate_macd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
        """MACD and Signal line"""
        ema_fast = TechnicalIndicators.calculate_ema(prices, fast)
        ema_slow = TechnicalIndicators.calculate_ema(prices, slow)
        macd = ema_fast - ema_slow
        macd_signal = TechnicalIndicators.calculate_ema(macd, signal)
        return macd, macd_signal
    
    @staticmethod
    def calculate_bollinger_bands(prices: np.ndarray, window: int = 20, num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Bollinger Bands: upper, middle, lower"""
        middle = TechnicalIndicators.calculate_sma(prices, window)
        std = np.full(len(prices), np.nan)
        for i in range(window - 1, len(prices)):
            std[i] = np.std(prices[i - window + 1:i + 1])
        
        upper = middle + num_std * std
        lower = middle - num_std * std
        return upper, middle, lower


class ARIMAPredictor:
    """ARIMA Model for Time Series Forecasting"""
    
    def __init__(self, order: Tuple[int, int, int] = (5, 1, 0)):
        self.order = order
        self.model = None
        self.fitted = False
    
    def fit(self, prices: np.ndarray):
        """Fit ARIMA model"""
        if not HAS_ARIMA:
            raise ImportError("statsmodels not installed")
        
        try:
            self.model = ARIMA(prices, order=self.order)
            self.model = self.model.fit()
            self.fitted = True
            logger.info(f"ARIMA{self.order} fitted successfully")
        except Exception as e:
            logger.error(f"ARIMA fit error: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> np.ndarray:
        """Predict next 'steps' values"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        forecast = self.model.forecast(steps=steps)
        return np.array(forecast)


class ProphetPredictor:
    """Facebook Prophet Model"""
    
    def __init__(self):
        self.model = None
        self.fitted = False
    
    def fit(self, df: pd.DataFrame):
        """
        Fit Prophet model
        df must have columns: 'ds' (date), 'y' (value)
        """
        if not HAS_PROPHET:
            raise ImportError("Prophet not installed")
        
        try:
            self.model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            self.model.fit(df)
            self.fitted = True
            logger.info("Prophet fitted successfully")
        except Exception as e:
            logger.error(f"Prophet fit error: {e}")
            self.fitted = False
    
    def predict(self, steps: int) -> pd.DataFrame:
        """Predict next 'steps' values"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        future = self.model.make_future_dataframe(periods=steps)
        forecast = self.model.predict(future)
        return forecast.tail(steps)[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]


class LSTMPredictor:
    """LSTM Deep Learning Model"""
    
    def __init__(self, lookback: int = 60, units: int = 50):
        self.lookback = lookback
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if HAS_SKLEARN else None
        self.fitted = False
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build LSTM architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
        
        model = keras.Sequential([
            keras.layers.LSTM(self.units, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.LSTM(self.units, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(25),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def fit(self, prices: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Fit LSTM model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
        
        try:
            # Scale data
            scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
            
            # Create sequences
            X, y = self._create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            # Build and train
            self.model = self._build_model((X.shape[1], 1))
            
            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss', patience=5, restore_best_weights=True
            )
            
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, 
                          callbacks=[early_stop], verbose=0)
            
            self.fitted = True
            logger.info("LSTM fitted successfully")
            
        except Exception as e:
            logger.error(f"LSTM fit error: {e}")
            self.fitted = False
    
    def predict(self, prices: np.ndarray, steps: int) -> np.ndarray:
        """Predict next 'steps' values"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        predictions = []
        current_seq = self.scaler.transform(prices[-self.lookback:].reshape(-1, 1))
        
        for _ in range(steps):
            # Reshape for prediction
            X = current_seq.reshape((1, self.lookback, 1))
            
            # Predict
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            
            # Update sequence
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)
        
        # Inverse transform
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()


class GRUPredictor:
    """GRU Deep Learning Model"""
    
    def __init__(self, lookback: int = 60, units: int = 50):
        self.lookback = lookback
        self.units = units
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1)) if HAS_SKLEARN else None
        self.fitted = False
    
    def _build_model(self, input_shape: Tuple[int, int]):
        """Build GRU architecture"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
        
        model = keras.Sequential([
            keras.layers.GRU(self.units, return_sequences=True, input_shape=input_shape),
            keras.layers.Dropout(0.2),
            keras.layers.GRU(self.units, return_sequences=False),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(25),
            keras.layers.Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for GRU"""
        X, y = [], []
        for i in range(self.lookback, len(data)):
            X.append(data[i - self.lookback:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)
    
    def fit(self, prices: np.ndarray, epochs: int = 50, batch_size: int = 32):
        """Fit GRU model"""
        if not HAS_TENSORFLOW:
            raise ImportError("TensorFlow not installed")
        
        try:
            scaled_data = self.scaler.fit_transform(prices.reshape(-1, 1))
            X, y = self._create_sequences(scaled_data)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            
            self.model = self._build_model((X.shape[1], 1))
            
            early_stop = keras.callbacks.EarlyStopping(
                monitor='loss', patience=5, restore_best_weights=True
            )
            
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size,
                          callbacks=[early_stop], verbose=0)
            
            self.fitted = True
            logger.info("GRU fitted successfully")
            
        except Exception as e:
            logger.error(f"GRU fit error: {e}")
            self.fitted = False
    
    def predict(self, prices: np.ndarray, steps: int) -> np.ndarray:
        """Predict next 'steps' values"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        predictions = []
        current_seq = self.scaler.transform(prices[-self.lookback:].reshape(-1, 1))
        
        for _ in range(steps):
            X = current_seq.reshape((1, self.lookback, 1))
            pred = self.model.predict(X, verbose=0)[0, 0]
            predictions.append(pred)
            current_seq = np.append(current_seq[1:], [[pred]], axis=0)
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions.flatten()


class XGBoostPredictor:
    """XGBoost Model với Feature Engineering"""
    
    def __init__(self, lookback: int = 30):
        self.lookback = lookback
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.fitted = False
    
    def _create_features(self, prices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create features từ price data"""
        features = []
        targets = []
        
        # Tính các indicators
        ti = TechnicalIndicators()
        sma_5 = ti.calculate_sma(prices, 5)
        sma_20 = ti.calculate_sma(prices, 20)
        rsi = ti.calculate_rsi(prices)
        macd, macd_signal = ti.calculate_macd(prices)
        
        for i in range(self.lookback, len(prices) - 1):
            row = []
            
            # Lag features
            for lag in [1, 2, 3, 5, 10]:
                row.append(prices[i] / prices[i - lag] - 1)  # Returns
            
            # Moving averages ratio
            row.append(prices[i] / sma_5[i] - 1)
            row.append(prices[i] / sma_20[i] - 1)
            row.append(sma_5[i] / sma_20[i] - 1)
            
            # Momentum indicators
            row.append((rsi[i] - 50) / 50)  # Normalized RSI
            row.append(macd[i] / prices[i])  # Normalized MACD
            row.append((macd[i] - macd_signal[i]) / prices[i])
            
            # Volatility
            row.append(np.std(prices[i - 10:i]) / prices[i])
            row.append(np.std(prices[i - 20:i]) / prices[i])
            
            features.append(row)
            targets.append((prices[i + 1] - prices[i]) / prices[i])  # Next day return
        
        return np.array(features), np.array(targets)
    
    def fit(self, prices: np.ndarray):
        """Fit XGBoost model"""
        if not HAS_XGBOOST:
            # Fallback to GradientBoosting
            if HAS_SKLEARN:
                self.model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.1
                )
            else:
                raise ImportError("Neither XGBoost nor sklearn installed")
        else:
            self.model = xgb.XGBRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                objective='reg:squarederror'
            )
        
        try:
            X, y = self._create_features(prices)
            X_scaled = self.scaler.fit_transform(X)
            self.model.fit(X_scaled, y)
            self.fitted = True
            logger.info("XGBoost fitted successfully")
        except Exception as e:
            logger.error(f"XGBoost fit error: {e}")
            self.fitted = False
    
    def predict(self, prices: np.ndarray, steps: int) -> np.ndarray:
        """Predict next 'steps' values"""
        if not self.fitted:
            raise ValueError("Model not fitted")
        
        predictions = []
        current_prices = prices.copy()
        
        for _ in range(steps):
            # Create features for latest data point
            X, _ = self._create_features(current_prices)
            if len(X) == 0:
                break
            
            X_scaled = self.scaler.transform(X[-1:])
            pred_return = self.model.predict(X_scaled)[0]
            
            # Clip extreme predictions
            pred_return = np.clip(pred_return, -0.05, 0.05)
            
            next_price = current_prices[-1] * (1 + pred_return)
            predictions.append(next_price)
            current_prices = np.append(current_prices, next_price)
        
        return np.array(predictions)


class EnsemblePredictor:
    """
    Ensemble Model - Kết hợp tất cả các models
    
    Models:
    - ARIMA: Statistical model
    - Prophet: Trend + Seasonality
    - LSTM: Deep Learning
    - GRU: Deep Learning (faster)
    - XGBoost: Feature-based ML
    
    Ensemble methods:
    - Weighted Average: dựa trên validation performance
    - Mean: Simple average
    """
    
    def __init__(self, use_models: List[str] = None):
        """
        Args:
            use_models: List of models to use. 
                       Options: ['arima', 'prophet', 'lstm', 'gru', 'xgboost']
                       Default: all available models
        """
        self.available_models = []
        self.models = {}
        self.weights = {}
        self.fitted = False
        
        # Default weights (can be updated after training)
        self.default_weights = {
            'arima': 0.15,
            'prophet': 0.20,
            'lstm': 0.25,
            'gru': 0.25,
            'xgboost': 0.15
        }
        
        # Determine which models to use
        if use_models is None:
            use_models = ['arima', 'prophet', 'lstm', 'gru', 'xgboost']
        
        # Initialize available models
        if 'arima' in use_models and HAS_ARIMA:
            self.models['arima'] = ARIMAPredictor()
            self.available_models.append('arima')
        
        if 'prophet' in use_models and HAS_PROPHET:
            self.models['prophet'] = ProphetPredictor()
            self.available_models.append('prophet')
        
        if 'lstm' in use_models and HAS_TENSORFLOW:
            self.models['lstm'] = LSTMPredictor()
            self.available_models.append('lstm')
        
        if 'gru' in use_models and HAS_TENSORFLOW:
            self.models['gru'] = GRUPredictor()
            self.available_models.append('gru')
        
        if 'xgboost' in use_models and (HAS_XGBOOST or HAS_SKLEARN):
            self.models['xgboost'] = XGBoostPredictor()
            self.available_models.append('xgboost')
        
        logger.info(f"Ensemble initialized with models: {self.available_models}")
    
    def fit(self, df: pd.DataFrame, target_col: str = 'close'):
        """
        Fit all models
        
        Args:
            df: DataFrame with at least 'date' and target_col
            target_col: Column to predict
        """
        prices = df[target_col].values
        
        training_results = {}
        
        # Fit ARIMA
        if 'arima' in self.models:
            try:
                self.models['arima'].fit(prices)
                training_results['arima'] = 'success'
            except Exception as e:
                training_results['arima'] = f'failed: {e}'
        
        # Fit Prophet
        if 'prophet' in self.models:
            try:
                prophet_df = df[['date', target_col]].copy()
                prophet_df.columns = ['ds', 'y']
                self.models['prophet'].fit(prophet_df)
                training_results['prophet'] = 'success'
            except Exception as e:
                training_results['prophet'] = f'failed: {e}'
        
        # Fit LSTM
        if 'lstm' in self.models:
            try:
                self.models['lstm'].fit(prices)
                training_results['lstm'] = 'success'
            except Exception as e:
                training_results['lstm'] = f'failed: {e}'
        
        # Fit GRU
        if 'gru' in self.models:
            try:
                self.models['gru'].fit(prices)
                training_results['gru'] = 'success'
            except Exception as e:
                training_results['gru'] = f'failed: {e}'
        
        # Fit XGBoost
        if 'xgboost' in self.models:
            try:
                self.models['xgboost'].fit(prices)
                training_results['xgboost'] = 'success'
            except Exception as e:
                training_results['xgboost'] = f'failed: {e}'
        
        # Update weights based on which models succeeded
        successful_models = [m for m, r in training_results.items() if r == 'success']
        if successful_models:
            total_weight = sum(self.default_weights[m] for m in successful_models)
            self.weights = {m: self.default_weights[m] / total_weight for m in successful_models}
        
        self.fitted = len(successful_models) > 0
        
        logger.info(f"Training results: {training_results}")
        logger.info(f"Final weights: {self.weights}")
        
        return training_results
    
    def predict(self, df: pd.DataFrame, steps: int = 7, 
                target_col: str = 'close') -> Dict:
        """
        Make ensemble predictions
        
        Returns:
            Dict with predictions, ranges, and model details
        """
        if not self.fitted:
            raise ValueError("Ensemble not fitted")
        
        prices = df[target_col].values
        last_price = float(prices[-1])
        
        # Get predictions from each model
        model_predictions = {}
        
        for model_name, model in self.models.items():
            if model_name not in self.weights:
                continue
            
            try:
                if model_name == 'arima':
                    preds = model.predict(steps)
                elif model_name == 'prophet':
                    forecast = model.predict(steps)
                    preds = forecast['yhat'].values
                elif model_name in ['lstm', 'gru', 'xgboost']:
                    preds = model.predict(prices, steps)
                
                # Clip extreme predictions (max ±15% từ giá cuối)
                max_price = last_price * 1.15
                min_price = last_price * 0.85
                preds = np.clip(preds, min_price, max_price)
                
                model_predictions[model_name] = preds
                logger.info(f"{model_name} predictions: {preds}")
                
            except Exception as e:
                logger.warning(f"{model_name} prediction error: {e}")
        
        if not model_predictions:
            raise ValueError("All models failed to predict")
        
        # Weighted ensemble
        ensemble_preds = np.zeros(steps)
        total_weight = 0
        
        for model_name, preds in model_predictions.items():
            weight = self.weights.get(model_name, 0)
            ensemble_preds += weight * preds[:steps]
            total_weight += weight
        
        if total_weight > 0:
            ensemble_preds /= total_weight
        
        # Calculate prediction ranges
        all_preds = np.array(list(model_predictions.values()))
        pred_std = np.std(all_preds, axis=0)
        
        prediction_ranges = []
        for i in range(steps):
            low = ensemble_preds[i] - 1.96 * pred_std[i]
            high = ensemble_preds[i] + 1.96 * pred_std[i]
            prediction_ranges.append({
                'low': round(max(last_price * 0.85, low), 2),
                'high': round(min(last_price * 1.15, high), 2)
            })
        
        # Generate dates (skip weekends)
        dates = []
        current_date = datetime.now()
        while len(dates) < steps:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                dates.append(current_date.strftime('%Y-%m-%d'))
        
        # Determine trend
        total_change = (ensemble_preds[-1] - last_price) / last_price
        if total_change > 0.02:
            trend = 'up'
        elif total_change < -0.02:
            trend = 'down'
        else:
            trend = 'sideways'
        
        # Calculate confidence based on model agreement
        pred_variance = np.mean(pred_std) / last_price
        confidence = max(0.3, min(0.9, 1 - pred_variance * 10))
        
        return {
            'symbol': df.get('symbol', 'UNKNOWN') if isinstance(df, dict) else 'UNKNOWN',
            'predictions': [round(p, 2) for p in ensemble_preds],
            'prediction_ranges': prediction_ranges,
            'dates': dates,
            'model': 'ensemble',
            'models_used': list(model_predictions.keys()),
            'weights': self.weights,
            'confidence': round(confidence, 2),
            'trend': trend,
            'last_price': last_price,
            'total_change_pct': round(total_change * 100, 2),
            'model_predictions': {k: [round(p, 2) for p in v] for k, v in model_predictions.items()}
        }


class PredictionService:
    """
    Main service for stock predictions
    """
    
    def __init__(self):
        self.model_cache = {}
        self.trained_symbols = set()
    
    def get_or_create_model(self, symbol: str) -> EnsemblePredictor:
        """Get cached model or create new one"""
        if symbol not in self.model_cache:
            self.model_cache[symbol] = EnsemblePredictor()
        return self.model_cache[symbol]
    
    def train(self, symbol: str, df: pd.DataFrame) -> Dict:
        """Train model for a symbol"""
        model = self.get_or_create_model(symbol)
        result = model.fit(df)
        if model.fitted:
            self.trained_symbols.add(symbol)
        return {
            'symbol': symbol,
            'trained': model.fitted,
            'models': model.available_models,
            'weights': model.weights,
            'results': result
        }
    
    def predict(self, symbol: str, df: pd.DataFrame, 
                steps: int = 7, train_if_needed: bool = True) -> Dict:
        """Make predictions for a symbol"""
        model = self.get_or_create_model(symbol)
        
        if not model.fitted and train_if_needed:
            self.train(symbol, df)
        
        if not model.fitted:
            # Fallback to simple prediction
            return self._simple_predict(df, steps)
        
        result = model.predict(df, steps)
        result['symbol'] = symbol
        return result
    
    def _simple_predict(self, df: pd.DataFrame, steps: int) -> Dict:
        """Simple fallback prediction"""
        prices = df['close'].values
        last_price = float(prices[-1])
        
        # Simple mean reversion + trend
        sma20 = np.mean(prices[-20:])
        trend = (prices[-1] - prices[-5]) / prices[-5] / 5  # Daily trend
        
        predictions = []
        current = last_price
        
        for i in range(steps):
            # Mean reversion
            reversion = (sma20 - current) * 0.05
            # Trend with decay
            trend_effect = trend * (0.8 ** i)
            # Next price
            change = reversion + trend_effect * current
            next_price = current + change
            next_price = np.clip(next_price, last_price * 0.85, last_price * 1.15)
            predictions.append(round(next_price, 2))
            current = next_price
        
        dates = []
        current_date = datetime.now()
        while len(dates) < steps:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                dates.append(current_date.strftime('%Y-%m-%d'))
        
        return {
            'predictions': predictions,
            'dates': dates,
            'model': 'simple',
            'confidence': 0.4,
            'last_price': last_price,
            'trend': 'up' if predictions[-1] > last_price else 'down' if predictions[-1] < last_price else 'sideways'
        }


# Global instance
_prediction_service = None

def get_prediction_service() -> PredictionService:
    """Get global prediction service instance"""
    global _prediction_service
    if _prediction_service is None:
        _prediction_service = PredictionService()
    return _prediction_service


def predict_stock(symbol: str, df: pd.DataFrame, steps: int = 7) -> Dict:
    """Quick function to predict stock"""
    service = get_prediction_service()
    return service.predict(symbol, df, steps)


def train_model(symbol: str, df: pd.DataFrame) -> Dict:
    """Quick function to train model"""
    service = get_prediction_service()
    return service.train(symbol, df)


if __name__ == "__main__":
    print("Stock Price Prediction Service")
    print("=" * 60)
    print("\nAvailable Models:")
    print(f"  - ARIMA: {HAS_ARIMA}")
    print(f"  - Prophet: {HAS_PROPHET}")
    print(f"  - LSTM/GRU: {HAS_TENSORFLOW}")
    print(f"  - XGBoost: {HAS_XGBOOST}")
    print(f"  - sklearn: {HAS_SKLEARN}")
    print("\nUsage:")
    print("  from prediction_service import predict_stock, train_model")
    print("  result = predict_stock('VNM', df, steps=7)")
