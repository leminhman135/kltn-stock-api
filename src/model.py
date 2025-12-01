# src/model.py
"""
Simple Stock Price Prediction using Trend + Moving Average
Lightweight version - no heavy ML dependencies
"""

import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def quick_predict(prices: list, steps: int = 7) -> dict:
    """
    Quick prediction using trend analysis + moving average
    
    Args:
        prices: List of close prices (at least 5 points)
        steps: Days to predict (default 7)
    
    Returns:
        dict with predictions, dates, confidence, trend info
    """
    if len(prices) < 5:
        return {'predictions': [], 'error': 'Need at least 5 price points'}
    
    prices = np.array(prices, dtype=float)
    
    # Calculate trend from recent data
    recent = prices[-14:] if len(prices) >= 14 else prices
    trend = (recent[-1] - recent[0]) / len(recent)
    
    # Calculate volatility
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-14:]) if len(returns) >= 14 else np.std(returns)
    
    # Calculate moving averages
    sma5 = np.mean(prices[-5:]) if len(prices) >= 5 else prices[-1]
    sma10 = np.mean(prices[-10:]) if len(prices) >= 10 else sma5
    sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else sma10
    
    # Generate predictions
    last_price = float(prices[-1])
    predictions = []
    
    for i in range(steps):
        # Trend component with dampening
        trend_component = trend * (i + 1) * (0.85 ** i)
        
        # Mean reversion toward SMA20
        mean_reversion = (sma20 - last_price) * 0.05 * (i + 1)
        
        # Base prediction
        pred = last_price + trend_component + mean_reversion
        
        # Add controlled noise based on volatility
        noise = np.random.normal(0, volatility * last_price * 0.05)
        pred += noise
        
        # Ensure positive price
        pred = max(pred, last_price * 0.8)
        predictions.append(round(float(pred), 2))
    
    # Generate dates (skip weekends)
    today = datetime.now()
    dates = []
    day_count = 0
    current_date = today
    while day_count < steps:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current_date.strftime('%Y-%m-%d'))
            day_count += 1
    
    # Determine trend direction
    if trend > volatility * last_price * 0.01:
        trend_dir = 'up'
    elif trend < -volatility * last_price * 0.01:
        trend_dir = 'down'
    else:
        trend_dir = 'sideways'
    
    return {
        'predictions': predictions,
        'dates': dates,
        'model': 'trend_sma',
        'confidence': round(min(0.7, 0.5 + len(prices) / 500), 2),
        'trend': trend_dir,
        'volatility': round(float(volatility * 100), 2),
        'sma5': round(float(sma5), 2),
        'sma20': round(float(sma20), 2)
    }


# Dummy classes for backward compatibility
class StockPredictor:
    def train(self, df):
        return {'arima': False, 'rf': False, 'message': 'Use quick_predict instead'}
    
    def predict(self, steps=7, model_type='ensemble'):
        return {'predictions': [], 'dates': [], 'confidence': 0}
    
    def get_metrics(self):
        return {}
