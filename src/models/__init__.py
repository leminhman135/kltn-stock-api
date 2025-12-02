"""
Models Module - Machine Learning & Deep Learning for Stock Prediction

Available Models:
1. ARIMA - Classical time series forecasting
2. Prophet - Facebook's forecasting library
3. XGBoost - Gradient boosting (in ensemble)
4. LSTM - Long Short-Term Memory neural network
5. GRU - Gated Recurrent Unit neural network  
6. CNN-LSTM - Hybrid convolutional + recurrent
7. Ensemble - Combination of multiple models

Usage:
    from src.models import LSTMModel, GRUModel, StockPredictor
    
    predictor = StockPredictor(model_type='lstm')
    predictor.fit(df, feature_cols=['close', 'volume', 'rsi'])
    predictions = predictor.predict(df)
"""

# Try importing deep learning models
try:
    from .deep_learning import (
        LSTMModel,
        GRUModel, 
        CNNLSTMModel,
        EnsemblePredictor,
        StockPredictor,
        DataPreprocessor,
        TENSORFLOW_AVAILABLE
    )
except ImportError:
    TENSORFLOW_AVAILABLE = False
    LSTMModel = None
    GRUModel = None
    CNNLSTMModel = None
    EnsemblePredictor = None
    StockPredictor = None
    DataPreprocessor = None

__all__ = [
    'LSTMModel',
    'GRUModel',
    'CNNLSTMModel', 
    'EnsemblePredictor',
    'StockPredictor',
    'DataPreprocessor',
    'TENSORFLOW_AVAILABLE'
]
