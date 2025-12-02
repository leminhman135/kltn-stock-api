"""
API Module - FastAPI Routers for Stock Prediction

Routers:
1. ml_endpoints - Basic ML predictions (ARIMA, Prophet, XGBoost)
2. advanced_ml_endpoints - Advanced ML (FinBERT, LSTM, GRU, Features)

Usage:
    from src.api import ml_router, advanced_ml_router
    
    app.include_router(ml_router)
    app.include_router(advanced_ml_router)
"""

from .ml_endpoints import router as ml_router
from .advanced_ml_endpoints import router as advanced_ml_router

__all__ = ['ml_router', 'advanced_ml_router']
