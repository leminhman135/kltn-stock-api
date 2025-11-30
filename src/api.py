"""
FastAPI Backend - REST API cho Stock Prediction System
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Import các modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Stock Prediction API",
    description="API for stock price prediction using ensemble ML models",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class StockRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str


class PredictionRequest(BaseModel):
    symbol: str
    periods: int = 30
    model_type: str = "ensemble"  # 'arima', 'prophet', 'lstm', 'gru', 'ensemble'


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    strategy: str = "long_only"
    initial_capital: float = 100000
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class TrainingRequest(BaseModel):
    symbol: str
    model_types: List[str] = ["arima", "prophet", "lstm", "gru"]
    start_date: str
    end_date: str


# Global state (trong production nên dùng database)
models_cache = {}
data_cache = {}


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Stock Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "data": "/api/data/{symbol}",
            "predict": "/api/predict",
            "backtest": "/api/backtest",
            "train": "/api/train",
            "models": "/api/models",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(models_cache),
        "cached_symbols": len(data_cache)
    }


@app.post("/api/data/collect")
async def collect_data(request: StockRequest):
    """
    Thu thập dữ liệu giá cổ phiếu
    
    Args:
        symbol: Mã cổ phiếu
        start_date: Ngày bắt đầu (YYYY-MM-DD)
        end_date: Ngày kết thúc (YYYY-MM-DD)
    """
    try:
        from data_collection import YahooFinanceAPI
        
        logger.info(f"Collecting data for {request.symbol}")
        
        api = YahooFinanceAPI()
        df = api.get_stock_data(
            request.symbol,
            request.start_date,
            request.end_date
        )
        
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        
        # Cache data
        data_cache[request.symbol] = df
        
        return {
            "symbol": request.symbol,
            "records": len(df),
            "start_date": df['date'].min().isoformat(),
            "end_date": df['date'].max().isoformat(),
            "data": df.tail(10).to_dict(orient='records')
        }
    
    except Exception as e:
        logger.error(f"Error collecting data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/{symbol}")
async def get_data(symbol: str, limit: int = 100):
    """
    Lấy dữ liệu đã cache
    
    Args:
        symbol: Mã cổ phiếu
        limit: Số records trả về
    """
    if symbol not in data_cache:
        raise HTTPException(status_code=404, detail="Data not found. Call /api/data/collect first")
    
    df = data_cache[symbol]
    
    return {
        "symbol": symbol,
        "total_records": len(df),
        "data": df.tail(limit).to_dict(orient='records')
    }


@app.post("/api/predict")
async def predict(request: PredictionRequest):
    """
    Dự đoán giá cổ phiếu
    
    Args:
        symbol: Mã cổ phiếu
        periods: Số ngày dự đoán
        model_type: Loại mô hình
    """
    try:
        logger.info(f"Predicting {request.symbol} for {request.periods} periods using {request.model_type}")
        
        # Check if data exists
        if request.symbol not in data_cache:
            raise HTTPException(status_code=404, detail="Data not found. Collect data first")
        
        df = data_cache[request.symbol]
        
        # Simplified prediction logic (cần implement đầy đủ với models thực tế)
        # Đây chỉ là mock response
        
        last_price = df['close'].iloc[-1]
        dates = pd.date_range(
            start=df['date'].max() + timedelta(days=1),
            periods=request.periods,
            freq='D'
        )
        
        # Mock predictions
        predictions = last_price * (1 + np.random.randn(request.periods) * 0.02).cumsum() / 50
        
        return {
            "symbol": request.symbol,
            "model_type": request.model_type,
            "periods": request.periods,
            "last_actual_price": float(last_price),
            "predictions": [
                {
                    "date": date.isoformat(),
                    "predicted_price": float(pred),
                    "change_pct": float((pred - last_price) / last_price * 100)
                }
                for date, pred in zip(dates, predictions)
            ]
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error predicting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/backtest")
async def backtest(request: BacktestRequest):
    """
    Chạy backtest cho chiến lược
    
    Args:
        symbol: Mã cổ phiếu
        start_date, end_date: Khoảng thời gian backtest
        strategy: Chiến lược giao dịch
        initial_capital: Vốn ban đầu
        stop_loss, take_profit: Risk management
    """
    try:
        from backtesting import BacktestEngine
        
        logger.info(f"Backtesting {request.symbol} with {request.strategy} strategy")
        
        # Get data
        if request.symbol not in data_cache:
            raise HTTPException(status_code=404, detail="Data not found")
        
        df = data_cache[request.symbol]
        df_filtered = df[
            (df['date'] >= request.start_date) &
            (df['date'] <= request.end_date)
        ].copy()
        
        if df_filtered.empty:
            raise HTTPException(status_code=404, detail="No data in specified date range")
        
        # Mock predictions for backtest
        predictions = df_filtered['close'].values * (1 + np.random.randn(len(df_filtered)) * 0.01)
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission=0.001
        )
        
        results = engine.run_backtest(
            data=df_filtered.set_index('date'),
            predictions=predictions,
            strategy=request.strategy,
            stop_loss=request.stop_loss,
            take_profit=request.take_profit
        )
        
        # Get trades
        trades_df = engine.get_trades_df()
        
        return {
            "symbol": request.symbol,
            "strategy": request.strategy,
            "period": {
                "start": request.start_date,
                "end": request.end_date
            },
            "metrics": results,
            "trades": trades_df.to_dict(orient='records') if not trades_df.empty else []
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error backtesting: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train models (background task)
    
    Args:
        symbol: Mã cổ phiếu
        model_types: Danh sách models cần train
        start_date, end_date: Khoảng thời gian training
    """
    try:
        logger.info(f"Training models for {request.symbol}")
        
        # Add to background tasks
        # background_tasks.add_task(train_models_background, request)
        
        return {
            "message": "Training started",
            "symbol": request.symbol,
            "models": request.model_types,
            "status": "processing"
        }
    
    except Exception as e:
        logger.error(f"Error starting training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/models")
async def list_models():
    """Liệt kê các models đã được train"""
    return {
        "models": list(models_cache.keys()),
        "count": len(models_cache)
    }


@app.get("/api/models/{symbol}/{model_type}")
async def get_model_info(symbol: str, model_type: str):
    """Thông tin chi tiết về một model"""
    model_key = f"{symbol}_{model_type}"
    
    if model_key not in models_cache:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "symbol": symbol,
        "model_type": model_type,
        "status": "trained",
        "last_updated": datetime.now().isoformat()
    }


@app.get("/api/sentiment/{symbol}")
async def get_sentiment(symbol: str, days: int = 30):
    """
    Lấy sentiment analysis cho symbol
    
    Args:
        symbol: Mã cổ phiếu
        days: Số ngày gần nhất
    """
    try:
        # Mock sentiment data
        dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
        sentiments = np.random.randn(days) * 0.3  # -1 to 1
        
        return {
            "symbol": symbol,
            "period_days": days,
            "average_sentiment": float(np.mean(sentiments)),
            "daily_sentiment": [
                {
                    "date": date.isoformat(),
                    "sentiment_score": float(sent),
                    "sentiment_label": "positive" if sent > 0.1 else "negative" if sent < -0.1 else "neutral"
                }
                for date, sent in zip(dates, sentiments)
            ]
        }
    
    except Exception as e:
        logger.error(f"Error getting sentiment: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("Starting Stock Prediction API...")
    print("Documentation: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
