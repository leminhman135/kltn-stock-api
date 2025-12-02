"""
Features Module - Advanced Feature Engineering for Stock Prediction

This module provides:
1. 100+ Technical Indicators (SMA, EMA, RSI, MACD, Bollinger, etc.)
2. Volume Analysis Features
3. Momentum and Volatility Indicators  
4. Statistical Features (skewness, kurtosis, autocorrelation)
5. Time-based Features
6. Lagged Features for ML models
"""

from .feature_engineering import (
    AdvancedFeatureEngineer,
    create_features_for_prediction
)

__all__ = [
    'AdvancedFeatureEngineer',
    'create_features_for_prediction'
]
