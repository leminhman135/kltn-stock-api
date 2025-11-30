"""
Enhanced FastAPI Backend with PostgreSQL Integration
REST API for Stock Prediction System
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta, date
import pandas as pd
import logging

from src.database.connection import get_db
from src.database.models import (
    Stock, StockPrice, TechnicalIndicator,
    SentimentAnalysis, ModelMetrics, Prediction
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="KLTN Stock Prediction API",
    description="Advanced API for Vietnamese stock price prediction using ML models and PostgreSQL",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =====================================================
# PYDANTIC MODELS
# =====================================================

class StockResponse(BaseModel):
    id: int
    symbol: str
    name: str
    exchange: str
    sector: Optional[str]
    is_active: bool
    
    class Config:
        from_attributes = True


class StockPriceResponse(BaseModel):
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    source: str
    
    class Config:
        from_attributes = True


class TechnicalIndicatorResponse(BaseModel):
    date: date
    sma_20: Optional[float]
    sma_50: Optional[float]
    rsi_14: Optional[float]
    macd: Optional[float]
    bb_upper: Optional[float]
    bb_lower: Optional[float]
    
    class Config:
        from_attributes = True


class PredictionResponse(BaseModel):
    target_date: date
    predicted_close: float
    confidence_upper: Optional[float]
    confidence_lower: Optional[float]
    model_name: str
    
    class Config:
        from_attributes = True


class SentimentResponse(BaseModel):
    date: date
    sentiment_score: float
    sentiment_label: str
    news_count: int
    confidence: float
    
    class Config:
        from_attributes = True


class ModelMetricsResponse(BaseModel):
    model_name: str
    stock_symbol: str
    mae: Optional[float]
    rmse: Optional[float]
    mape: Optional[float]
    r2_score: Optional[float]
    trained_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class PredictionRequest(BaseModel):
    symbol: str = Field(..., example="VNM")
    periods: int = Field(default=30, ge=1, le=90, example=30)
    model_type: str = Field(default="ensemble", example="ensemble")


class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    strategy: str = "long_only"
    initial_capital: float = 100000
    commission: float = 0.0015


# =====================================================
# ROOT & HEALTH ENDPOINTS
# =====================================================

@app.get("/", tags=["Root"])
async def root():
    """API Root - List all available endpoints"""
    return {
        "name": "KLTN Stock Prediction API",
        "version": "2.0.0",
        "database": "PostgreSQL",
        "endpoints": {
            "stocks": {
                "GET /api/stocks": "List all stocks",
                "GET /api/stocks/{symbol}": "Get stock details",
                "GET /api/stocks/search?q=query": "Search stocks"
            },
            "prices": {
                "GET /api/prices/{symbol}": "Get historical prices",
                "GET /api/prices/{symbol}/latest": "Get latest price",
                "GET /api/prices/{symbol}/ohlcv": "Get OHLCV data"
            },
            "indicators": {
                "GET /api/indicators/{symbol}": "Get technical indicators",
                "GET /api/indicators/{symbol}/latest": "Get latest indicators"
            },
            "predictions": {
                "POST /api/predictions/predict": "Create new prediction",
                "GET /api/predictions/{symbol}": "Get predictions for symbol",
                "GET /api/predictions/{symbol}/latest": "Get latest prediction"
            },
            "sentiment": {
                "GET /api/sentiment/{symbol}": "Get sentiment analysis",
                "GET /api/sentiment/{symbol}/latest": "Get latest sentiment"
            },
            "models": {
                "GET /api/models": "List all trained models",
                "GET /api/models/{symbol}": "Get models for symbol",
                "GET /api/models/{symbol}/{model_name}": "Get specific model metrics"
            },
            "backtest": {
                "POST /api/backtest": "Run backtest simulation"
            },
            "stats": {
                "GET /api/stats/overview": "System overview statistics",
                "GET /api/stats/stocks": "Stock statistics"
            }
        }
    }


@app.get("/api/health", tags=["Root"])
async def health_check():
    """Health check endpoint - No database required"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "2.0.0",
        "message": "API is running (database connection disabled for now)"
    }


# =====================================================
# STOCK ENDPOINTS
# =====================================================

@app.get("/api/stocks", response_model=List[StockResponse], tags=["Stocks"])
async def list_stocks(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    active_only: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get list of all stocks"""
    query = db.query(Stock)
    
    if active_only:
        query = query.filter(Stock.is_active == True)
    
    stocks = query.offset(skip).limit(limit).all()
    return stocks


@app.get("/api/stocks/{symbol}", response_model=StockResponse, tags=["Stocks"])
async def get_stock(symbol: str, db: Session = Depends(get_db)):
    """Get detailed information for a specific stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    return stock


@app.get("/api/stocks/search", response_model=List[StockResponse], tags=["Stocks"])
async def search_stocks(
    q: str = Query(..., min_length=1),
    db: Session = Depends(get_db)
):
    """Search stocks by symbol or name"""
    stocks = db.query(Stock).filter(
        (Stock.symbol.ilike(f"%{q}%")) | (Stock.name.ilike(f"%{q}%"))
    ).limit(20).all()
    
    return stocks


# =====================================================
# PRICE DATA ENDPOINTS
# =====================================================

@app.get("/api/prices/{symbol}", response_model=List[StockPriceResponse], tags=["Prices"])
async def get_prices(
    symbol: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: int = Query(252, ge=1, le=5000),
    db: Session = Depends(get_db)
):
    """Get historical prices for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(StockPrice).filter(StockPrice.stock_id == stock.id)
    
    if start_date:
        query = query.filter(StockPrice.date >= start_date)
    if end_date:
        query = query.filter(StockPrice.date <= end_date)
    
    prices = query.order_by(desc(StockPrice.date)).limit(limit).all()
    
    # Reverse to chronological order
    return list(reversed(prices))


@app.get("/api/prices/{symbol}/latest", response_model=StockPriceResponse, tags=["Prices"])
async def get_latest_price(symbol: str, db: Session = Depends(get_db)):
    """Get the most recent price for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    latest = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(desc(StockPrice.date)).first()
    
    if not latest:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")
    
    return latest


@app.get("/api/prices/{symbol}/ohlcv", tags=["Prices"])
async def get_ohlcv(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 252,
    db: Session = Depends(get_db)
):
    """Get OHLCV data in format suitable for charting libraries"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(StockPrice).filter(StockPrice.stock_id == stock.id)
    
    if start_date:
        query = query.filter(StockPrice.date >= start_date)
    if end_date:
        query = query.filter(StockPrice.date <= end_date)
    
    prices = query.order_by(StockPrice.date).limit(limit).all()
    
    return {
        "symbol": symbol,
        "data": [
            {
                "date": p.date.isoformat(),
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume
            }
            for p in prices
        ]
    }


# =====================================================
# TECHNICAL INDICATORS ENDPOINTS
# =====================================================

@app.get("/api/indicators/{symbol}", response_model=List[TechnicalIndicatorResponse], tags=["Indicators"])
async def get_indicators(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 252,
    db: Session = Depends(get_db)
):
    """Get technical indicators for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(TechnicalIndicator).filter(TechnicalIndicator.stock_id == stock.id)
    
    if start_date:
        query = query.filter(TechnicalIndicator.date >= start_date)
    if end_date:
        query = query.filter(TechnicalIndicator.date <= end_date)
    
    indicators = query.order_by(desc(TechnicalIndicator.date)).limit(limit).all()
    
    return list(reversed(indicators))


@app.get("/api/indicators/{symbol}/latest", response_model=TechnicalIndicatorResponse, tags=["Indicators"])
async def get_latest_indicators(symbol: str, db: Session = Depends(get_db)):
    """Get most recent technical indicators"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    latest = db.query(TechnicalIndicator).filter(
        TechnicalIndicator.stock_id == stock.id
    ).order_by(desc(TechnicalIndicator.date)).first()
    
    if not latest:
        raise HTTPException(status_code=404, detail=f"No indicators for {symbol}")
    
    return latest


# =====================================================
# PREDICTION ENDPOINTS
# =====================================================

@app.post("/api/predictions/predict", tags=["Predictions"])
async def create_prediction(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Create new predictions (triggers model inference)
    Note: This is a simplified version. Full implementation would load actual models.
    """
    stock = db.query(Stock).filter(Stock.symbol == request.symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {request.symbol} not found")
    
    # In production, this would trigger actual model inference
    # For now, return a task ID
    
    return {
        "status": "processing",
        "symbol": request.symbol,
        "periods": request.periods,
        "model_type": request.model_type,
        "message": "Prediction task queued. Check /api/predictions/{symbol} for results."
    }


@app.get("/api/predictions/{symbol}", response_model=List[PredictionResponse], tags=["Predictions"])
async def get_predictions(
    symbol: str,
    model_name: Optional[str] = None,
    limit: int = 30,
    db: Session = Depends(get_db)
):
    """Get predictions for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(Prediction).filter(Prediction.stock_id == stock.id)
    
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    
    predictions = query.order_by(desc(Prediction.target_date)).limit(limit).all()
    
    return list(reversed(predictions))


@app.get("/api/predictions/{symbol}/latest", response_model=PredictionResponse, tags=["Predictions"])
async def get_latest_prediction(
    symbol: str,
    model_name: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Get latest prediction"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(Prediction).filter(Prediction.stock_id == stock.id)
    
    if model_name:
        query = query.filter(Prediction.model_name == model_name)
    
    latest = query.order_by(desc(Prediction.target_date)).first()
    
    if not latest:
        raise HTTPException(status_code=404, detail="No predictions found")
    
    return latest


# =====================================================
# SENTIMENT ENDPOINTS
# =====================================================

@app.get("/api/sentiment/{symbol}", response_model=List[SentimentResponse], tags=["Sentiment"])
async def get_sentiment(
    symbol: str,
    start_date: Optional[str] = None,
    limit: int = 30,
    db: Session = Depends(get_db)
):
    """Get sentiment analysis for a stock"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    query = db.query(SentimentAnalysis).filter(SentimentAnalysis.stock_id == stock.id)
    
    if start_date:
        query = query.filter(SentimentAnalysis.date >= start_date)
    
    sentiments = query.order_by(desc(SentimentAnalysis.date)).limit(limit).all()
    
    return list(reversed(sentiments))


@app.get("/api/sentiment/{symbol}/latest", response_model=SentimentResponse, tags=["Sentiment"])
async def get_latest_sentiment(symbol: str, db: Session = Depends(get_db)):
    """Get latest sentiment"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    latest = db.query(SentimentAnalysis).filter(
        SentimentAnalysis.stock_id == stock.id
    ).order_by(desc(SentimentAnalysis.date)).first()
    
    if not latest:
        raise HTTPException(status_code=404, detail="No sentiment data found")
    
    return latest


# =====================================================
# MODEL ENDPOINTS
# =====================================================

@app.get("/api/models", response_model=List[ModelMetricsResponse], tags=["Models"])
async def list_models(
    active_only: bool = True,
    db: Session = Depends(get_db)
):
    """List all trained models"""
    query = db.query(ModelMetrics)
    
    if active_only:
        query = query.filter(ModelMetrics.is_active == True)
    
    models = query.order_by(desc(ModelMetrics.trained_at)).all()
    
    return models


@app.get("/api/models/{symbol}", response_model=List[ModelMetricsResponse], tags=["Models"])
async def get_models_for_stock(symbol: str, db: Session = Depends(get_db)):
    """Get all models trained for a specific stock"""
    models = db.query(ModelMetrics).filter(
        ModelMetrics.stock_symbol == symbol
    ).order_by(desc(ModelMetrics.trained_at)).all()
    
    if not models:
        raise HTTPException(status_code=404, detail=f"No models found for {symbol}")
    
    return models


@app.get("/api/models/{symbol}/{model_name}", response_model=ModelMetricsResponse, tags=["Models"])
async def get_specific_model(
    symbol: str,
    model_name: str,
    db: Session = Depends(get_db)
):
    """Get metrics for a specific model"""
    model = db.query(ModelMetrics).filter(
        ModelMetrics.stock_symbol == symbol,
        ModelMetrics.model_name == model_name,
        ModelMetrics.is_active == True
    ).first()
    
    if not model:
        raise HTTPException(status_code=404, detail=f"Model {model_name} for {symbol} not found")
    
    return model


# =====================================================
# STATISTICS ENDPOINTS
# =====================================================

@app.get("/api/stats/overview", tags=["Statistics"])
async def get_overview_stats(db: Session = Depends(get_db)):
    """Get system overview statistics"""
    return {
        "stocks": {
            "total": db.query(Stock).count(),
            "active": db.query(Stock).filter(Stock.is_active == True).count()
        },
        "price_records": db.query(StockPrice).count(),
        "indicators": db.query(TechnicalIndicator).count(),
        "predictions": db.query(Prediction).count(),
        "sentiment_records": db.query(SentimentAnalysis).count(),
        "models_trained": db.query(ModelMetrics).count(),
        "active_models": db.query(ModelMetrics).filter(ModelMetrics.is_active == True).count()
    }


@app.get("/api/stats/stocks", tags=["Statistics"])
async def get_stock_stats(db: Session = Depends(get_db)):
    """Get statistics for each stock"""
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    
    results = []
    for stock in stocks:
        price_count = db.query(StockPrice).filter(StockPrice.stock_id == stock.id).count()
        latest_price = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id
        ).order_by(desc(StockPrice.date)).first()
        
        results.append({
            "symbol": stock.symbol,
            "name": stock.name,
            "price_records": price_count,
            "latest_price": latest_price.close if latest_price else None,
            "latest_date": latest_price.date.isoformat() if latest_price else None
        })
    
    return results


# =====================================================
# BACKTEST ENDPOINT
# =====================================================

@app.post("/api/backtest", tags=["Backtest"])
async def run_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run backtest simulation
    Note: This is a simplified mock. Full implementation would use actual backtest engine.
    """
    stock = db.query(Stock).filter(Stock.symbol == request.symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {request.symbol} not found")
    
    # Get historical prices
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id,
        StockPrice.date >= request.start_date,
        StockPrice.date <= request.end_date
    ).order_by(StockPrice.date).all()
    
    if not prices:
        raise HTTPException(status_code=404, detail="No price data in specified range")
    
    # Mock backtest results
    return {
        "symbol": request.symbol,
        "strategy": request.strategy,
        "period": {
            "start": request.start_date,
            "end": request.end_date,
            "days": len(prices)
        },
        "initial_capital": request.initial_capital,
        "metrics": {
            "final_value": request.initial_capital * 1.15,
            "total_return": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "win_rate": 0.62,
            "total_trades": 45
        },
        "message": "This is a mock result. Implement actual backtest engine for real results."
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting KLTN Stock Prediction API v2.0...")
    print("📊 Documentation: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("🔗 Database: PostgreSQL")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
