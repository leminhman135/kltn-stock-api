# src/model.py
"""
Stock Price Prediction using Machine Learning
Includes: Linear Regression, Random Forest, Gradient Boosting
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try import sklearn
try:
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
    logger.info("✅ scikit-learn loaded successfully")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠️ scikit-learn not installed. Using simple prediction.")


class StockMLModel:
    """Machine Learning model for stock price prediction"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.metrics = {}
        self.is_trained = False
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from OHLCV data"""
        if len(df) < 30:
            raise ValueError("Need at least 30 data points")
        
        data = df.copy()
        data = data.sort_values('date').reset_index(drop=True)
        
        close = data['close'].values
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        features = []
        targets = []
        
        for i in range(20, len(close) - 1):
            row = []
            current = close[i]
            
            # Lagged prices (normalized)
            row.append(close[i-1] / current)
            row.append(close[i-2] / current)
            row.append(close[i-3] / current)
            row.append(close[i-5] / current)
            row.append(close[i-10] / current)
            
            # Returns
            row.append((close[i] - close[i-1]) / close[i-1])
            row.append((close[i] - close[i-3]) / close[i-3])
            row.append((close[i] - close[i-5]) / close[i-5])
            
            # Moving averages (normalized)
            sma5 = np.mean(close[i-4:i+1])
            sma10 = np.mean(close[i-9:i+1])
            sma20 = np.mean(close[i-19:i+1])
            row.append(sma5 / current)
            row.append(sma10 / current)
            row.append(sma20 / current)
            
            # Volatility
            row.append(np.std(close[i-4:i+1]) / current)
            row.append(np.std(close[i-9:i+1]) / current)
            
            # Volume ratio
            avg_vol = np.mean(volume[i-9:i+1])
            row.append(volume[i] / avg_vol if avg_vol > 0 else 1.0)
            
            # Price position in range
            high_20 = np.max(close[i-19:i+1])
            low_20 = np.min(close[i-19:i+1])
            row.append((current - low_20) / (high_20 - low_20 + 0.001))
            
            features.append(row)
            targets.append((close[i+1] - close[i]) / close[i])
        
        return np.array(features), np.array(targets)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ML models"""
        if not HAS_SKLEARN:
            return {'success': False, 'error': 'scikit-learn not installed'}
        
        try:
            logger.info(f"🔄 Training models for {self.symbol}...")
            
            X, y = self.create_features(df)
            
            if len(X) < 50:
                return {'success': False, 'error': 'Not enough data (need 50+ points)'}
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            results = {}
            model_configs = {
                'linear': Ridge(alpha=1.0),
                'random_forest': RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
                'gradient_boost': GradientBoostingRegressor(n_estimators=50, max_depth=4, learning_rate=0.1, random_state=42)
            }
            
            for name, model in model_configs.items():
                try:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    mse = mean_squared_error(y_test, y_pred)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    direction_acc = np.sum((y_pred > 0) == (y_test > 0)) / len(y_test)
                    
                    self.models[name] = model
                    self.metrics[name] = {
                        'rmse': round(np.sqrt(mse) * 100, 2),
                        'mae': round(mae * 100, 2),
                        'r2': round(max(0, r2), 3),
                        'accuracy': round(direction_acc * 100, 1)
                    }
                    results[name] = {'trained': True, 'metrics': self.metrics[name]}
                    logger.info(f"  ✅ {name}: Accuracy={self.metrics[name]['accuracy']}%")
                except Exception as e:
                    results[name] = {'trained': False, 'error': str(e)}
            
            self.is_trained = len(self.models) > 0
            return {'success': True, 'symbol': self.symbol, 'data_points': len(df), 'models': results}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, steps: int = 7, model_type: str = 'ensemble') -> Dict:
        """Make predictions"""
        if not self.is_trained:
            return quick_predict(df['close'].tolist(), steps)
        
        try:
            X, _ = self.create_features(df)
            if len(X) == 0:
                return quick_predict(df['close'].tolist(), steps)
            
            latest_features = self.scaler.transform(X[-1:])
            last_price = float(df['close'].iloc[-1])
            
            predictions = []
            current_price = last_price
            
            for step in range(steps):
                if model_type == 'ensemble' and self.models:
                    pred_returns = []
                    weights = []
                    for name, model in self.models.items():
                        pred_return = model.predict(latest_features)[0]
                        acc = self.metrics.get(name, {}).get('accuracy', 50)
                        pred_returns.append(pred_return)
                        weights.append(acc)
                    weighted_return = sum(r * w for r, w in zip(pred_returns, weights)) / sum(weights)
                elif model_type in self.models:
                    weighted_return = self.models[model_type].predict(latest_features)[0]
                else:
                    weighted_return = 0.001
                
                next_price = current_price * (1 + weighted_return)
                predictions.append(round(float(next_price), 2))
                current_price = next_price
            
            # Generate dates (skip weekends)
            dates = []
            current_date = datetime.now()
            while len(dates) < steps:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:
                    dates.append(current_date.strftime('%Y-%m-%d'))
            
            avg_accuracy = np.mean([m.get('accuracy', 50) for m in self.metrics.values()])
            trend = 'up' if predictions[-1] > last_price * 1.02 else 'down' if predictions[-1] < last_price * 0.98 else 'sideways'
            
            return {
                'symbol': self.symbol,
                'predictions': predictions,
                'dates': dates,
                'model': model_type,
                'confidence': round(avg_accuracy / 100, 2),
                'trend': trend,
                'last_price': last_price,
                'metrics': self.metrics
            }
        except Exception as e:
            return quick_predict(df['close'].tolist(), steps)
    
    def get_metrics(self) -> Dict:
        return {'symbol': self.symbol, 'is_trained': self.is_trained, 'models': self.metrics}


def quick_predict(prices: list, steps: int = 7) -> dict:
    """Quick prediction using trend analysis (fallback)"""
    if len(prices) < 5:
        return {'predictions': [], 'error': 'Need at least 5 price points'}
    
    prices = np.array(prices, dtype=float)
    recent = prices[-14:] if len(prices) >= 14 else prices
    trend = (recent[-1] - recent[0]) / len(recent)
    
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns)
    
    sma5 = np.mean(prices[-5:])
    sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma5
    
    last_price = float(prices[-1])
    predictions = []
    
    for i in range(steps):
        pred = last_price + trend * (i + 1) * (0.85 ** i) + (sma20 - last_price) * 0.05 * (i + 1)
        pred += np.random.normal(0, volatility * last_price * 0.03)
        predictions.append(round(max(pred, last_price * 0.8), 2))
    
    dates = []
    current_date = datetime.now()
    while len(dates) < steps:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            dates.append(current_date.strftime('%Y-%m-%d'))
    
    trend_dir = 'up' if trend > volatility * last_price * 0.01 else 'down' if trend < -volatility * last_price * 0.01 else 'sideways'
    
    return {'predictions': predictions, 'dates': dates, 'model': 'technical', 'confidence': 0.6, 'trend': trend_dir, 'last_price': last_price}


# Model cache
_model_cache: Dict[str, StockMLModel] = {}

def get_model(symbol: str) -> StockMLModel:
    if symbol not in _model_cache:
        _model_cache[symbol] = StockMLModel(symbol)
    return _model_cache[symbol]

def train_model(symbol: str, df: pd.DataFrame) -> Dict:
    return get_model(symbol).train(df)

def predict_stock(symbol: str, df: pd.DataFrame, steps: int = 7, model_type: str = 'ensemble') -> Dict:
    model = get_model(symbol)
    if not model.is_trained:
        model.train(df)
    return model.predict(df, steps, model_type)
