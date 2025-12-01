# src/model.py
"""
Lightweight ML Models for Stock Price Prediction
- ARIMA: Time series forecasting
- Random Forest: Feature-based prediction
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import logging

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# =====================================================
# ARIMA MODEL
# =====================================================

class ARIMAPredictor:
    """ARIMA model for time series prediction"""
    
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None
        self.model_fit = None
    
    def fit(self, prices: list):
        """Train ARIMA model on price data"""
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            if len(prices) < 30:
                logger.warning("Not enough data for ARIMA (need 30+ points)")
                return False
            
            self.model = ARIMA(prices, order=self.order)
            self.model_fit = self.model.fit()
            logger.info(f"ARIMA model trained successfully on {len(prices)} data points")
            return True
        except Exception as e:
            logger.error(f"ARIMA training error: {e}")
            return False
    
    def predict(self, steps: int = 7) -> list:
        """Predict next N days"""
        if self.model_fit is None:
            return []
        
        try:
            forecast = self.model_fit.forecast(steps=steps)
            return forecast.tolist()
        except Exception as e:
            logger.error(f"ARIMA prediction error: {e}")
            return []
    
    def get_metrics(self) -> dict:
        """Get model metrics"""
        if self.model_fit is None:
            return {}
        
        return {
            "aic": self.model_fit.aic,
            "bic": self.model_fit.bic,
            "order": self.order
        }


# =====================================================
# RANDOM FOREST MODEL
# =====================================================

class RandomForestPredictor:
    """Random Forest model with technical indicators"""
    
    def __init__(self, n_estimators=100, lookback=14):
        self.n_estimators = n_estimators
        self.lookback = lookback
        self.model = None
        self.scaler = None
        self.last_features = None
    
    def _create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicator features"""
        data = df.copy()
        
        # Price features
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Moving averages
        for window in [5, 10, 20]:
            data[f'sma_{window}'] = data['close'].rolling(window=window).mean()
            data[f'ema_{window}'] = data['close'].ewm(span=window).mean()
            data[f'sma_ratio_{window}'] = data['close'] / data[f'sma_{window}']
        
        # Volatility
        data['volatility_5'] = data['returns'].rolling(window=5).std()
        data['volatility_10'] = data['returns'].rolling(window=10).std()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = data['close'].ewm(span=12).mean()
        ema26 = data['close'].ewm(span=26).mean()
        data['macd'] = ema12 - ema26
        data['macd_signal'] = data['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        sma20 = data['close'].rolling(window=20).mean()
        std20 = data['close'].rolling(window=20).std()
        data['bb_upper'] = sma20 + (std20 * 2)
        data['bb_lower'] = sma20 - (std20 * 2)
        data['bb_position'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
        
        # Volume features (if available)
        if 'volume' in data.columns:
            data['volume_sma'] = data['volume'].rolling(window=10).mean()
            data['volume_ratio'] = data['volume'] / data['volume_sma']
        
        # Lag features
        for lag in range(1, 6):
            data[f'close_lag_{lag}'] = data['close'].shift(lag)
            data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
        
        # Target: next day close
        data['target'] = data['close'].shift(-1)
        
        return data.dropna()
    
    def fit(self, df: pd.DataFrame) -> bool:
        """Train Random Forest model"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            
            if len(df) < 50:
                logger.warning("Not enough data for RF (need 50+ points)")
                return False
            
            # Create features
            data = self._create_features(df)
            
            if len(data) < 30:
                logger.warning("Not enough data after feature creation")
                return False
            
            # Select features
            feature_cols = [col for col in data.columns if col not in ['date', 'target', 'open', 'high', 'low', 'close', 'volume']]
            
            X = data[feature_cols].values
            y = data['target'].values
            
            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(X_scaled, y)
            
            # Store last features for prediction
            self.last_features = X_scaled[-1:]
            self.feature_cols = feature_cols
            self.last_data = data.iloc[-1:]
            
            logger.info(f"Random Forest trained on {len(data)} samples, {len(feature_cols)} features")
            return True
            
        except Exception as e:
            logger.error(f"RF training error: {e}")
            return False
    
    def predict(self, steps: int = 7) -> list:
        """Predict next N days"""
        if self.model is None:
            return []
        
        try:
            predictions = []
            current_features = self.last_features.copy()
            last_close = self.last_data['close'].values[0]
            
            for _ in range(steps):
                pred = self.model.predict(current_features)[0]
                predictions.append(pred)
                
                # Simple feature update (shift lags)
                # In production, would recalculate all features
                current_features = current_features * (1 + np.random.normal(0, 0.001, current_features.shape))
                last_close = pred
            
            return predictions
            
        except Exception as e:
            logger.error(f"RF prediction error: {e}")
            return []
    
    def get_feature_importance(self) -> dict:
        """Get feature importance scores"""
        if self.model is None:
            return {}
        
        importance = dict(zip(self.feature_cols, self.model.feature_importances_))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])


# =====================================================
# ENSEMBLE PREDICTOR
# =====================================================

class StockPredictor:
    """Ensemble predictor combining ARIMA and Random Forest"""
    
    def __init__(self):
        self.arima = ARIMAPredictor()
        self.rf = RandomForestPredictor()
        self.weights = {'arima': 0.4, 'rf': 0.6}  # RF usually better for short-term
    
    def train(self, prices_df: pd.DataFrame) -> dict:
        """
        Train all models on price data
        
        Args:
            prices_df: DataFrame with columns ['date', 'open', 'high', 'low', 'close', 'volume']
        
        Returns:
            dict with training status for each model
        """
        results = {
            'arima': False,
            'rf': False,
            'message': ''
        }
        
        if len(prices_df) < 30:
            results['message'] = f"Not enough data: {len(prices_df)} rows (need 30+)"
            return results
        
        # Train ARIMA on close prices
        close_prices = prices_df['close'].tolist()
        results['arima'] = self.arima.fit(close_prices)
        
        # Train Random Forest on full data
        results['rf'] = self.rf.fit(prices_df)
        
        if results['arima'] or results['rf']:
            results['message'] = 'Training completed'
        else:
            results['message'] = 'Training failed for all models'
        
        return results
    
    def predict(self, steps: int = 7, model_type: str = 'ensemble') -> dict:
        """
        Generate predictions
        
        Args:
            steps: Number of days to predict
            model_type: 'arima', 'rf', or 'ensemble'
        
        Returns:
            dict with predictions and metadata
        """
        result = {
            'predictions': [],
            'model': model_type,
            'confidence': 0.0,
            'dates': []
        }
        
        # Generate future dates
        today = datetime.now()
        future_dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(steps)]
        result['dates'] = future_dates
        
        arima_preds = self.arima.predict(steps) if self.arima.model_fit else []
        rf_preds = self.rf.predict(steps) if self.rf.model else []
        
        if model_type == 'arima' and arima_preds:
            result['predictions'] = [float(x) for x in arima_preds]
            result['confidence'] = 0.65
        elif model_type == 'rf' and rf_preds:
            result['predictions'] = [float(x) for x in rf_preds]
            result['confidence'] = 0.72
        elif model_type == 'ensemble' and (arima_preds or rf_preds):
            # Weighted ensemble
            if arima_preds and rf_preds:
                ensemble = []
                for a, r in zip(arima_preds, rf_preds):
                    ensemble.append(float(a * self.weights['arima'] + r * self.weights['rf']))
                result['predictions'] = ensemble
                result['confidence'] = 0.75
            elif rf_preds:
                result['predictions'] = [float(x) for x in rf_preds]
                result['confidence'] = 0.72
            else:
                result['predictions'] = [float(x) for x in arima_preds]
                result['confidence'] = 0.65
        
        return result
    
    def get_metrics(self) -> dict:
        """Get metrics from all models"""
        return {
            'arima': self.arima.get_metrics(),
            'rf_features': self.rf.get_feature_importance() if self.rf.model else {}
        }


# =====================================================
# QUICK PREDICTION FUNCTION
# =====================================================

def quick_predict(prices: list, steps: int = 7) -> dict:
    """
    Quick prediction using simple moving average + trend
    Fallback when not enough data for ML models
    
    Args:
        prices: List of close prices
        steps: Days to predict
    
    Returns:
        dict with predictions
    """
    if len(prices) < 5:
        return {'predictions': [], 'error': 'Need at least 5 price points'}
    
    prices = np.array(prices)
    
    # Calculate trend
    recent = prices[-14:] if len(prices) >= 14 else prices
    trend = (recent[-1] - recent[0]) / len(recent)  # Average daily change
    
    # Calculate volatility
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns)
    
    # Generate predictions
    last_price = prices[-1]
    predictions = []
    
    for i in range(steps):
        # Trend + mean reversion + noise
        pred = last_price + trend * (i + 1) * (0.9 ** i)  # Dampening trend
        # Add some randomness based on volatility
        noise = np.random.normal(0, volatility * last_price * 0.1)
        pred += noise
        predictions.append(round(pred, 2))
    
    today = datetime.now()
    dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(steps)]
    
    return {
        'predictions': predictions,
        'dates': dates,
        'model': 'trend_ma',
        'confidence': 0.5,
        'trend': 'up' if trend > 0 else 'down',
        'volatility': round(volatility * 100, 2)
    }
