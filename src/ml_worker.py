"""
ML Worker Service
Handles heavy ML predictions in separate process/service
Can be called via HTTP or message queue
"""

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="KLTN ML Worker",
    version="1.0.0",
    description="ML prediction worker service"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy load ML models to reduce startup time
_models_loaded = False
_arima_model = None
_prophet_model = None
_lstm_model = None
_gru_model = None

def load_models():
    global _models_loaded, _arima_model, _prophet_model, _lstm_model, _gru_model
    
    if _models_loaded:
        return
    
    logger.info("🔄 Loading ML models...")
    
    try:
        from src.models.arima_model import ARIMAModel
        _arima_model = ARIMAModel
        logger.info("✅ ARIMA loaded")
    except Exception as e:
        logger.warning(f"⚠️ ARIMA failed: {e}")
    
    try:
        from src.models.prophet_model import ProphetModel
        _prophet_model = ProphetModel
        logger.info("✅ Prophet loaded")
    except Exception as e:
        logger.warning(f"⚠️ Prophet failed: {e}")
    
    try:
        from src.models.lstm_gru_models import LSTMModel, GRUModel
        _lstm_model = LSTMModel
        _gru_model = GRUModel
        logger.info("✅ LSTM/GRU loaded")
    except Exception as e:
        logger.warning(f"⚠️ LSTM/GRU failed: {e}")
    
    _models_loaded = True
    logger.info("✅ All models loaded!")

@app.get("/health")
async def health():
    return {"status": "healthy", "models_loaded": _models_loaded}

@app.post("/predict/{symbol}")
async def predict(
    symbol: str,
    days: int = 7,
    models: str = "all"
):
    """Run ML prediction for a stock"""
    load_models()
    
    # Import database
    from src.database.connection import get_db_session
    from src.database.models import StockPrice
    import pandas as pd
    
    results = {}
    
    with get_db_session() as db:
        # Get price data
        prices = db.query(StockPrice).filter(
            StockPrice.symbol == symbol.upper()
        ).order_by(StockPrice.date.desc()).limit(300).all()
        
        if not prices:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        df = pd.DataFrame([{
            'date': p.date,
            'close': p.close,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'volume': p.volume
        } for p in reversed(prices)])
        
        # Run ARIMA
        if _arima_model and ('all' in models or 'arima' in models):
            try:
                model = _arima_model()
                model.fit(df['close'])
                preds = model.predict(days)
                results['arima'] = {
                    'predictions': preds.tolist(),
                    'status': 'success'
                }
            except Exception as e:
                results['arima'] = {'status': 'error', 'error': str(e)}
        
        # Run Prophet
        if _prophet_model and ('all' in models or 'prophet' in models):
            try:
                model = _prophet_model()
                model.fit(df[['date', 'close']])
                preds = model.predict(days)
                results['prophet'] = {
                    'predictions': preds['yhat'].tolist(),
                    'status': 'success'
                }
            except Exception as e:
                results['prophet'] = {'status': 'error', 'error': str(e)}
        
        # Run LSTM
        if _lstm_model and ('all' in models or 'lstm' in models):
            try:
                model = _lstm_model(lookback=30)
                model.fit(df)
                preds = model.predict(days)
                results['lstm'] = {
                    'predictions': preds.tolist(),
                    'status': 'success'
                }
            except Exception as e:
                results['lstm'] = {'status': 'error', 'error': str(e)}
        
        # Run GRU
        if _gru_model and ('all' in models or 'gru' in models):
            try:
                model = _gru_model(lookback=30)
                model.fit(df)
                preds = model.predict(days)
                results['gru'] = {
                    'predictions': preds.tolist(),
                    'status': 'success'
                }
            except Exception as e:
                results['gru'] = {'status': 'error', 'error': str(e)}
    
    return {
        'symbol': symbol,
        'days': days,
        'results': results,
        'timestamp': datetime.utcnow().isoformat()
    }

@app.on_event("startup")
async def startup():
    logger.info("🚀 ML Worker starting...")
    # Optionally pre-load models
    # load_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081)
