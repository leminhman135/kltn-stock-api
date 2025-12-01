"""
Enhanced FastAPI Backend with PostgreSQL Integration
REST API for Stock Prediction System
"""

from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta, date
from contextlib import asynccontextmanager
import pandas as pd
import logging
import os

from src.database.connection import get_db, engine
from src.database.models import (
    Stock, StockPrice, TechnicalIndicator,
    SentimentAnalysis, ModelMetrics, Prediction, Base
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get the directory of the current file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(CURRENT_DIR, "static")


# =====================================================
# STARTUP EVENT - AUTO CREATE TABLES
# =====================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - runs on startup and shutdown"""
    # Startup
    logger.info("🚀 Starting KLTN Stock Prediction API...")
    try:
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created/verified successfully!")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down KLTN Stock Prediction API...")

# Initialize FastAPI app
app = FastAPI(
    title="KLTN Stock Prediction API",
    description="Advanced API for Vietnamese stock price prediction using ML models and PostgreSQL",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
# STATIC FILES & DASHBOARD
# =====================================================

# Mount static files if directory exists
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    logger.info(f"📁 Static files mounted from: {STATIC_DIR}")
else:
    logger.warning(f"⚠️ Static directory not found: {STATIC_DIR}")


@app.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
async def dashboard():
    """Serve the main dashboard page"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    logger.info(f"Looking for dashboard at: {index_path}")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    return {"error": "Dashboard not found", "path": index_path, "static_dir": STATIC_DIR}


# =====================================================
# ROOT & HEALTH ENDPOINTS
# =====================================================

@app.get("/", tags=["Root"])
async def root():
    """API Root - Redirect to dashboard or show API info"""
    # Return dashboard if exists, otherwise API info
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path, media_type="text/html")
    
    return {
        "name": "KLTN Stock Prediction API",
        "version": "2.0.0",
        "database": "PostgreSQL",
        "dashboard": "/dashboard",
        "static_dir": STATIC_DIR,
        "static_exists": os.path.exists(STATIC_DIR),
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


@app.get("/api/prices/{symbol}/by-date", response_model=StockPriceResponse, tags=["Prices"])
async def get_price_by_date(
    symbol: str, 
    date: str = Query(..., description="Ngày cần xem (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """Get price for a specific date"""
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    from datetime import datetime
    try:
        target_date = datetime.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")
    
    # Try exact date first
    price = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id,
        StockPrice.date == target_date
    ).first()
    
    # If no data for exact date, get closest previous date
    if not price:
        price = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id,
            StockPrice.date <= target_date
        ).order_by(desc(StockPrice.date)).first()
    
    if not price:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol} on or before {date}")
    
    return price


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
# NEWS & SENTIMENT ANALYSIS ENDPOINTS
# =====================================================

from src.news_service import news_service

@app.get("/api/news", tags=["News"])
async def get_market_news(limit: int = 20):
    """Lấy tin tức thị trường chung với phân tích sentiment"""
    try:
        news = news_service.get_all_news(symbol=None, limit=limit)
        return {
            "status": "success",
            "total": len(news),
            "news": [
                {
                    "title": n.title,
                    "summary": n.summary,
                    "url": n.url,
                    "source": n.source,
                    "published_at": n.published_at,
                    "sentiment": n.sentiment.value,
                    "sentiment_score": round(n.sentiment_score, 2),
                    "impact": n.impact_prediction
                }
                for n in news
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/{symbol}", tags=["News"])
async def get_stock_news(symbol: str, limit: int = 15):
    """Lấy tin tức cho một mã cổ phiếu cụ thể với phân tích sentiment"""
    try:
        news = news_service.get_all_news(symbol=symbol.upper(), limit=limit)
        summary = news_service.get_sentiment_summary(symbol.upper())
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "sentiment_summary": {
                "overall": summary["sentiment"],
                "avg_score": summary["avg_score"],
                "positive_count": summary["positive"],
                "negative_count": summary["negative"],
                "neutral_count": summary["neutral"],
                "recommendation": summary["recommendation"]
            },
            "total_news": len(news),
            "news": [
                {
                    "title": n.title,
                    "summary": n.summary,
                    "url": n.url,
                    "source": n.source,
                    "published_at": n.published_at,
                    "sentiment": n.sentiment.value,
                    "sentiment_score": round(n.sentiment_score, 2),
                    "impact": n.impact_prediction
                }
                for n in news
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/{symbol}/sentiment", tags=["News"])
async def get_news_sentiment(symbol: str):
    """Lấy tổng hợp sentiment từ tin tức cho một mã"""
    try:
        summary = news_service.get_sentiment_summary(symbol.upper())
        return {
            "status": "success",
            **summary
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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


# =====================================================
# DATABASE MANAGEMENT ENDPOINTS
# =====================================================

# VN30 stocks data
VN30_STOCKS = [
    {"symbol": "VNM", "name": "Công ty Cổ phần Sữa Việt Nam", "sector": "Consumer Goods", "exchange": "HOSE"},
    {"symbol": "VIC", "name": "Tập đoàn Vingroup", "sector": "Real Estate", "exchange": "HOSE"},
    {"symbol": "VHM", "name": "Vinhomes", "sector": "Real Estate", "exchange": "HOSE"},
    {"symbol": "VCB", "name": "Ngân hàng TMCP Ngoại thương Việt Nam", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "BID", "name": "Ngân hàng TMCP Đầu tư và Phát triển Việt Nam", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "CTG", "name": "Ngân hàng TMCP Công Thương Việt Nam", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "TCB", "name": "Ngân hàng TMCP Kỹ Thương Việt Nam", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "MBB", "name": "Ngân hàng TMCP Quân Đội", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "HPG", "name": "Tập đoàn Hòa Phát", "sector": "Steel", "exchange": "HOSE"},
    {"symbol": "FPT", "name": "FPT Corporation", "sector": "Technology", "exchange": "HOSE"},
    {"symbol": "MWG", "name": "Thế Giới Di Động", "sector": "Retail", "exchange": "HOSE"},
    {"symbol": "VPB", "name": "Ngân hàng TMCP Việt Nam Thịnh Vượng", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "PLX", "name": "Tập đoàn Xăng Dầu Việt Nam", "sector": "Energy", "exchange": "HOSE"},
    {"symbol": "VJC", "name": "Vietjet Air", "sector": "Aviation", "exchange": "HOSE"},
    {"symbol": "GAS", "name": "Tổng Công ty Khí Việt Nam", "sector": "Energy", "exchange": "HOSE"},
    {"symbol": "SAB", "name": "Tổng Công ty Cổ phần Bia - Rượu - Nước giải khát Sài Gòn", "sector": "Consumer Goods", "exchange": "HOSE"},
    {"symbol": "MSN", "name": "Tập đoàn Masan", "sector": "Consumer Goods", "exchange": "HOSE"},
    {"symbol": "VRE", "name": "Vincom Retail", "sector": "Real Estate", "exchange": "HOSE"},
    {"symbol": "NVL", "name": "Novaland", "sector": "Real Estate", "exchange": "HOSE"},
    {"symbol": "ACB", "name": "Ngân hàng TMCP Á Châu", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "GVR", "name": "Tập đoàn Công nghiệp Cao su Việt Nam", "sector": "Materials", "exchange": "HOSE"},
    {"symbol": "STB", "name": "Ngân hàng TMCP Sài Gòn Thương Tín", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "POW", "name": "Tổng Công ty Điện lực Dầu khí Việt Nam", "sector": "Energy", "exchange": "HOSE"},
    {"symbol": "BCM", "name": "Tổng Công ty Đầu tư và Phát triển Công nghiệp", "sector": "Industrial", "exchange": "HOSE"},
    {"symbol": "SSI", "name": "Công ty Cổ phần Chứng khoán SSI", "sector": "Securities", "exchange": "HOSE"},
    {"symbol": "VND", "name": "Công ty Cổ phần Chứng khoán VNDirect", "sector": "Securities", "exchange": "HOSE"},
    {"symbol": "TPB", "name": "Ngân hàng TMCP Tiên Phong", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "HDB", "name": "Ngân hàng TMCP Phát triển TP.HCM", "sector": "Banking", "exchange": "HOSE"},
    {"symbol": "PDR", "name": "Phát Đạt Real Estate", "sector": "Real Estate", "exchange": "HOSE"},
    {"symbol": "SHB", "name": "Ngân hàng TMCP Sài Gòn - Hà Nội", "sector": "Banking", "exchange": "HOSE"},
]


@app.post("/api/admin/init-db", tags=["Admin"])
async def init_database(db: Session = Depends(get_db)):
    """
    Initialize database with VN30 stocks.
    This creates sample stocks for testing.
    """
    try:
        # Check if stocks already exist
        existing_count = db.query(Stock).count()
        if existing_count > 0:
            return {
                "status": "skipped",
                "message": f"Database already has {existing_count} stocks",
                "hint": "Use /api/admin/reset-db to clear and reinitialize"
            }
        
        # Add VN30 stocks
        added = 0
        for stock_data in VN30_STOCKS:
            stock = Stock(
                symbol=stock_data["symbol"],
                name=stock_data["name"],
                sector=stock_data["sector"],
                exchange=stock_data["exchange"],
                is_active=True
            )
            db.add(stock)
            added += 1
        
        db.commit()
        
        return {
            "status": "success",
            "message": f"Added {added} VN30 stocks to database",
            "stocks_added": [s["symbol"] for s in VN30_STOCKS]
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/admin/seed-sample-data/{symbol}", tags=["Admin"])
async def seed_sample_data(
    symbol: str,
    days: int = Query(default=30, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """
    Seed sample price data for a stock (for testing purposes).
    Generates mock OHLCV data.
    """
    import random
    
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found. Initialize database first.")
    
    # Check if data exists
    existing = db.query(StockPrice).filter(StockPrice.stock_id == stock.id).count()
    if existing > 0:
        return {
            "status": "skipped",
            "message": f"Stock {symbol} already has {existing} price records",
            "hint": "Data already exists"
        }
    
    # Generate sample data
    base_price = random.uniform(20, 150) * 1000  # VND (20k - 150k)
    
    prices_added = []
    current_date = date.today() - timedelta(days=days)
    price = base_price
    
    for i in range(days):
        if current_date.weekday() < 5:  # Skip weekends
            # Random price movement
            change = random.uniform(-0.03, 0.03)
            price = price * (1 + change)
            
            high = price * random.uniform(1.01, 1.03)
            low = price * random.uniform(0.97, 0.99)
            open_price = random.uniform(low, high)
            volume = random.randint(100000, 5000000)
            
            stock_price = StockPrice(
                stock_id=stock.id,
                date=current_date,
                open=round(open_price, 0),
                high=round(high, 0),
                low=round(low, 0),
                close=round(price, 0),
                volume=volume,
                source="sample_data"
            )
            db.add(stock_price)
            prices_added.append(current_date.isoformat())
        
        current_date += timedelta(days=1)
    
    db.commit()
    
    return {
        "status": "success",
        "symbol": symbol,
        "records_added": len(prices_added),
        "date_range": {
            "from": prices_added[0] if prices_added else None,
            "to": prices_added[-1] if prices_added else None
        }
    }


@app.delete("/api/admin/reset-db", tags=["Admin"])
async def reset_database(
    confirm: bool = Query(False, description="Set to true to confirm reset"),
    db: Session = Depends(get_db)
):
    """
    Reset database - DELETE ALL DATA!
    Use with caution!
    """
    if not confirm:
        return {
            "status": "confirmation_required",
            "message": "Add ?confirm=true to URL to confirm database reset",
            "warning": "This will DELETE ALL DATA!"
        }
    
    try:
        # Delete in order due to foreign keys
        db.query(Prediction).delete()
        db.query(TechnicalIndicator).delete()
        db.query(SentimentAnalysis).delete()
        db.query(StockPrice).delete()
        db.query(ModelMetrics).delete()
        db.query(Stock).delete()
        
        db.commit()
        
        return {
            "status": "success",
            "message": "All data deleted successfully"
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/admin/db-status", tags=["Admin"])
async def get_db_status(db: Session = Depends(get_db)):
    """Get database connection status and table counts"""
    try:
        return {
            "status": "connected",
            "tables": {
                "stocks": db.query(Stock).count(),
                "stock_prices": db.query(StockPrice).count(),
                "technical_indicators": db.query(TechnicalIndicator).count(),
                "sentiment_analysis": db.query(SentimentAnalysis).count(),
                "predictions": db.query(Prediction).count(),
                "model_metrics": db.query(ModelMetrics).count()
            },
            "message": "Database connection successful"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# DATA COLLECTION ENDPOINTS - Fetch Real Data from VNDirect
# =====================================================

@app.post("/api/data/fetch/{symbol}", tags=["Data Collection"])
async def fetch_stock_data(
    symbol: str,
    days: int = Query(default=None, ge=7, le=1825, description="Số ngày dữ liệu (max 5 năm)"),
    from_date: str = Query(default=None, description="Từ ngày (YYYY-MM-DD)"),
    to_date: str = Query(default=None, description="Đến ngày (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Thu thập dữ liệu THỰC từ VNDirect API và lưu vào database.
    
    - **symbol**: Mã cổ phiếu (VNM, FPT, VCB, etc.)
    - **days**: Số ngày dữ liệu (nếu không dùng from_date/to_date)
    - **from_date**: Từ ngày (YYYY-MM-DD)
    - **to_date**: Đến ngày (YYYY-MM-DD)
    
    Dữ liệu bao gồm: Open, High, Low, Close, Volume
    """
    from src.data_collection import VNDirectAPI
    from datetime import datetime, timedelta
    
    # Check if stock exists in database
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if not stock:
        raise HTTPException(
            status_code=404, 
            detail=f"Stock {symbol} not found in database. Use /api/admin/init-db first."
        )
    
    try:
        # Initialize VNDirect API
        vndirect = VNDirectAPI()
        
        # Calculate date range
        if from_date and to_date:
            start_date = datetime.strptime(from_date, '%Y-%m-%d')
            end_date = datetime.strptime(to_date, '%Y-%m-%d')
        else:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days or 365)
        
        # Fetch data from VNDirect
        logger.info(f"Fetching data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
        df = vndirect.get_stock_price(
            symbol=symbol.upper(),
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=end_date.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            raise HTTPException(
                status_code=404, 
                detail=f"No data returned from VNDirect for {symbol}"
            )
        
        # Save to database
        records_added = 0
        records_updated = 0
        
        for _, row in df.iterrows():
            # Check if record already exists
            existing = db.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.date == row['date'].date()
            ).first()
            
            if existing:
                # Update existing record
                existing.open = float(row['Open'])
                existing.high = float(row['High'])
                existing.low = float(row['Low'])
                existing.close = float(row['Close'])
                existing.volume = float(row['Volume'])
                existing.source = "vndirect"
                records_updated += 1
            else:
                # Create new record
                price = StockPrice(
                    stock_id=stock.id,
                    date=row['date'].date(),
                    open=float(row['Open']),
                    high=float(row['High']),
                    low=float(row['Low']),
                    close=float(row['Close']),
                    volume=float(row['Volume']),
                    source="vndirect"
                )
                db.add(price)
                records_added += 1
        
        db.commit()
        
        # Get latest price for response
        latest = df.iloc[-1] if len(df) > 0 else None
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "records_added": records_added,
            "records_updated": records_updated,
            "total_records": len(df),
            "date_range": {
                "from": df['date'].min().strftime('%Y-%m-%d'),
                "to": df['date'].max().strftime('%Y-%m-%d')
            },
            "latest_price": {
                "date": latest['date'].strftime('%Y-%m-%d') if latest is not None else None,
                "close": float(latest['Close']) if latest is not None else None,
                "volume": float(latest['Volume']) if latest is not None else None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error fetching data for {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/data/fetch-all", tags=["Data Collection"])
async def fetch_all_stocks_data(
    days: int = Query(default=None, ge=7, le=1825),
    from_date: str = Query(default=None, description="Từ ngày (YYYY-MM-DD)"),
    to_date: str = Query(default=None, description="Đến ngày (YYYY-MM-DD)"),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Thu thập dữ liệu cho TẤT CẢ cổ phiếu trong database.
    
    - **days**: Số ngày dữ liệu (nếu không dùng from_date/to_date)
    - **from_date**: Từ ngày (YYYY-MM-DD)
    - **to_date**: Đến ngày (YYYY-MM-DD)
    
    ⚠️ Lưu ý: Quá trình này có thể mất vài phút!
    """
    from src.data_collection import VNDirectAPI
    from datetime import datetime, timedelta
    import time
    
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    
    if not stocks:
        raise HTTPException(
            status_code=404,
            detail="No stocks in database. Use /api/admin/init-db first."
        )
    
    vndirect = VNDirectAPI()
    
    # Calculate date range
    if from_date and to_date:
        start_date = datetime.strptime(from_date, '%Y-%m-%d')
        end_date = datetime.strptime(to_date, '%Y-%m-%d')
    else:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days or 365)
    
    results = []
    success_count = 0
    error_count = 0
    
    for stock in stocks:
        try:
            logger.info(f"Fetching {stock.symbol}...")
            df = vndirect.get_stock_price(
                symbol=stock.symbol,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                results.append({
                    "symbol": stock.symbol,
                    "status": "no_data",
                    "records": 0
                })
                error_count += 1
                continue
            
            # Save to database
            records_added = 0
            for _, row in df.iterrows():
                existing = db.query(StockPrice).filter(
                    StockPrice.stock_id == stock.id,
                    StockPrice.date == row['date'].date()
                ).first()
                
                if not existing:
                    price = StockPrice(
                        stock_id=stock.id,
                        date=row['date'].date(),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row['Volume']),
                        source="vndirect"
                    )
                    db.add(price)
                    records_added += 1
            
            db.commit()
            
            results.append({
                "symbol": stock.symbol,
                "status": "success",
                "records": records_added
            })
            success_count += 1
            
            # Rate limiting - avoid overloading VNDirect
            time.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error fetching {stock.symbol}: {str(e)}")
            results.append({
                "symbol": stock.symbol,
                "status": "error",
                "error": str(e)
            })
            error_count += 1
    
    return {
        "status": "completed",
        "summary": {
            "total_stocks": len(stocks),
            "success": success_count,
            "errors": error_count
        },
        "date_range": {
            "from": start_date.strftime('%Y-%m-%d'),
            "to": end_date.strftime('%Y-%m-%d')
        },
        "results": results
    }


@app.get("/api/data/realtime/{symbol}", tags=["Data Collection"])
async def get_realtime_price(symbol: str):
    """
    Lấy giá realtime từ VNDirect (không lưu database).
    
    Dùng để kiểm tra giá hiện tại nhanh.
    """
    from src.data_collection import VNDirectAPI
    
    try:
        vndirect = VNDirectAPI()
        
        # Get stock info (includes current price)
        info = vndirect.get_stock_info(symbol.upper())
        
        if not info:
            raise HTTPException(
                status_code=404,
                detail=f"Cannot get realtime data for {symbol}"
            )
        
        return {
            "symbol": symbol.upper(),
            "data": info,
            "source": "vndirect",
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/intraday/{symbol}", tags=["Data Collection"])
async def get_intraday_data(
    symbol: str,
    resolution: str = Query(default="5", description="1, 5, 15, 30, 60 (phút)")
):
    """
    Lấy dữ liệu intraday (trong ngày) từ VNDirect.
    
    - **resolution**: 1=1 phút, 5=5 phút, 15=15 phút, 30=30 phút, 60=1 giờ
    """
    from src.data_collection import VNDirectAPI
    
    try:
        vndirect = VNDirectAPI()
        df = vndirect.get_intraday_data(symbol.upper(), resolution=resolution)
        
        if df.empty:
            return {
                "symbol": symbol.upper(),
                "resolution": resolution,
                "data": [],
                "message": "No intraday data available (market may be closed)"
            }
        
        return {
            "symbol": symbol.upper(),
            "resolution": f"{resolution} min",
            "records": len(df),
            "data": df.to_dict('records'),
            "source": "vndirect"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ADVANCED BACKTEST ENDPOINT (using new engine)
# =====================================================

@app.post("/api/backtest/advanced", tags=["Backtest"])
async def run_advanced_backtest(request: BacktestRequest, db: Session = Depends(get_db)):
    """
    Run advanced backtest với Backtesting Engine đầy đủ
    
    Metrics:
    - Sharpe Ratio, Sortino Ratio
    - Max Drawdown
    - Win Rate, Profit Factor
    - VaR 95%
    """
    from src.backtest.backtesting_engine import BacktestingEngine, SignalGenerator, SignalType
    
    stock = db.query(Stock).filter(Stock.symbol == request.symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {request.symbol} not found")
    
    # Get historical prices
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id,
        StockPrice.date >= request.start_date,
        StockPrice.date <= request.end_date
    ).order_by(StockPrice.date).all()
    
    if len(prices) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 days of data")
    
    # Convert to DataFrame
    price_df = pd.DataFrame([{
        'date': p.date,
        'Open': p.open,
        'High': p.high,
        'Low': p.low,
        'Close': p.close,
        'Volume': p.volume
    } for p in prices])
    
    # Generate simple signals based on moving average crossover
    price_df['sma_10'] = price_df['Close'].rolling(10).mean()
    price_df['sma_30'] = price_df['Close'].rolling(30).mean()
    
    signals = {}
    for idx, row in price_df.iterrows():
        date = row['date']
        if pd.isna(row['sma_10']) or pd.isna(row['sma_30']):
            signals[date] = 'HOLD'
        elif row['sma_10'] > row['sma_30']:
            signals[date] = 'BUY'
        else:
            signals[date] = 'SELL'
    
    signals_series = pd.Series(signals)
    
    # Run backtest
    engine = BacktestingEngine(
        initial_capital=request.initial_capital,
        commission_rate=0.001,
        slippage=0.001
    )
    
    result = engine.run(
        price_df, 
        signals_series, 
        symbol=request.symbol,
        stop_loss_pct=0.05,
        take_profit_pct=0.10
    )
    
    return result.to_dict()


# =====================================================
# ETL ENDPOINTS
# =====================================================

@app.post("/api/etl/run/{symbol}", tags=["ETL"])
async def run_etl_pipeline(
    symbol: str,
    start_date: str = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(default=None, description="End date (YYYY-MM-DD)"),
    db: Session = Depends(get_db)
):
    """
    Chạy ETL pipeline cho một mã cổ phiếu
    
    ETL = Extract (VNDirect) -> Transform (Clean, Validate) -> Load (Database)
    """
    from src.etl.etl_pipeline import ETLPipeline, DatabaseLoader, VNDirectExtractor
    from datetime import datetime, timedelta
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    try:
        # Setup pipeline with database loader
        loader = DatabaseLoader(db)
        pipeline = ETLPipeline(loader=loader)
        
        # Run ETL
        result = pipeline.run(symbol.upper(), start_date, end_date)
        
        return result.to_dict()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/etl/validate/{symbol}", tags=["ETL"])
async def validate_stock_data(symbol: str, db: Session = Depends(get_db)):
    """
    Validate dữ liệu của một mã cổ phiếu trong database
    """
    from src.etl.etl_pipeline import DataValidator
    
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Get all prices
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(StockPrice.date).all()
    
    if not prices:
        return {"symbol": symbol, "status": "no_data", "message": "No price data found"}
    
    # Convert to DataFrame
    df = pd.DataFrame([{
        'date': p.date,
        'Open': p.open,
        'High': p.high,
        'Low': p.low,
        'Close': p.close,
        'Volume': p.volume
    } for p in prices])
    
    # Validate
    validator = DataValidator()
    result = validator.validate(df)
    
    return {
        "symbol": symbol.upper(),
        "validation": result.to_dict()
    }


# =====================================================
# SCHEDULER ENDPOINTS
# =====================================================

@app.get("/api/scheduler/status", tags=["Scheduler"])
async def get_scheduler_status():
    """Lấy trạng thái scheduler"""
    return {
        "status": "available",
        "scheduler_type": "render_cron",
        "message": "Use Render Cron Jobs or n8n for scheduling",
        "available_tasks": [
            {"name": "daily-data-fetch", "schedule": "0 7 * * 1-5", "description": "Fetch data 7AM UTC"},
            {"name": "weekly-full-update", "schedule": "0 0 * * 0", "description": "Full update Sunday"},
            {"name": "health-check", "schedule": "*/10 * * * *", "description": "Every 10 minutes"}
        ]
    }


@app.get("/api/scheduler/n8n-workflow", tags=["Scheduler"])
async def get_n8n_workflow():
    """
    Lấy template n8n workflow để import
    
    Hướng dẫn:
    1. Copy JSON output
    2. Mở n8n -> Workflows -> Import from JSON
    3. Paste và configure
    """
    from src.scheduler.scheduler_service import N8nIntegration
    
    # Get base URL from request
    base_url = "https://kltn-stock-api.onrender.com"
    
    n8n = N8nIntegration()
    workflow = n8n.generate_workflow_template(base_url)
    
    return {
        "message": "n8n Workflow Template - Import this JSON into n8n",
        "instructions": [
            "1. Open n8n (self-hosted or cloud)",
            "2. Go to Workflows -> Import from JSON",
            "3. Paste the 'workflow' object below",
            "4. Configure Slack/Discord node for notifications",
            "5. Activate the workflow"
        ],
        "workflow": workflow
    }


@app.get("/api/scheduler/render-config", tags=["Scheduler"])
async def get_render_cron_config():
    """
    Lấy cấu hình Render Cron Jobs
    
    Thêm vào render.yaml để tự động schedule
    """
    from src.scheduler.scheduler_service import generate_render_cron_config
    
    config = generate_render_cron_config("https://kltn-stock-api.onrender.com")
    
    return {
        "message": "Render Cron Jobs Configuration",
        "instructions": [
            "1. Copy the YAML content below",
            "2. Add to your render.yaml file",
            "3. Push to GitHub",
            "4. Render will auto-create cron jobs"
        ],
        "yaml_config": config
    }


# =====================================================
# EXTENDED DATA COLLECTION ENDPOINTS
# =====================================================

# --- Trading Data Endpoints ---

@app.get("/api/trading/detailed/{symbol}", tags=["Trading Data"])
async def get_detailed_trading_data(
    symbol: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy dữ liệu giao dịch chi tiết
    
    Bao gồm:
    - Giá OHLC
    - KL khớp lệnh, KL thỏa thuận
    - GT khớp lệnh, GT thỏa thuận
    - Giá tham chiếu, trần, sàn
    """
    from src.data_collection.trading_data import TradingDataCollector
    
    collector = TradingDataCollector()
    df = collector.get_detailed_trading_data(symbol, from_date, to_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No trading data found for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/trading/foreign/{symbol}", tags=["Trading Data"])
async def get_foreign_trading_data(
    symbol: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy dữ liệu giao dịch NDTNN
    
    Bao gồm:
    - Khối lượng mua/bán/ròng
    - Giá trị mua/bán/ròng
    - Room nước ngoài còn lại
    - % sở hữu nước ngoài
    """
    from src.data_collection.trading_data import TradingDataCollector
    
    collector = TradingDataCollector()
    df = collector.get_foreign_trading_data(symbol, from_date, to_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No foreign trading data found for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/trading/proprietary/{symbol}", tags=["Trading Data"])
async def get_proprietary_trading_data(
    symbol: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy dữ liệu giao dịch tự doanh
    
    Giao dịch của các CTCK tự thực hiện cho chính họ
    """
    from src.data_collection.trading_data import TradingDataCollector
    
    collector = TradingDataCollector()
    df = collector.get_proprietary_trading_data(symbol, from_date, to_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No proprietary trading data found for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/trading/summary/{symbol}", tags=["Trading Data"])
async def get_trading_summary(
    symbol: str,
    days: int = Query(30, description="Number of days")
):
    """
    Lấy tổng hợp dữ liệu giao dịch
    
    Bao gồm tổng hợp từ:
    - Giao dịch thông thường
    - Giao dịch NDTNN
    - Giao dịch tự doanh
    """
    from src.data_collection.trading_data import TradingDataCollector
    
    collector = TradingDataCollector()
    summary = collector.get_trading_summary(symbol, days)
    
    if not summary:
        raise HTTPException(status_code=404, detail=f"Cannot generate summary for {symbol}")
    
    return summary


@app.get("/api/trading/orderbook/{symbol}", tags=["Trading Data"])
async def get_order_book(symbol: str):
    """
    Lấy sổ lệnh (Order Book) realtime
    
    3 bước giá mua/bán tốt nhất
    """
    from src.data_collection.trading_data import TradingDataCollector
    
    collector = TradingDataCollector()
    order_book = collector.get_order_book(symbol)
    
    if not order_book:
        raise HTTPException(status_code=404, detail=f"Cannot get order book for {symbol}")
    
    return order_book


# --- Market Data Endpoints ---

@app.get("/api/market/index/{index_code}", tags=["Market Data"])
async def get_market_index(
    index_code: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy dữ liệu chỉ số thị trường
    
    Các chỉ số: VNINDEX, VN30, VN100, HNXINDEX, HNX30, UPCOM
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    df = collector.get_index_data(index_code.upper(), from_date, to_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data found for index {index_code}")
    
    return {
        "index_code": index_code.upper(),
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/market/index/realtime/{index_code}", tags=["Market Data"])
async def get_realtime_index(index_code: str = "VNINDEX"):
    """
    Lấy dữ liệu chỉ số realtime
    
    Bao gồm: Open, High, Low, Close, Volume, Value, Advances, Declines
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.get_realtime_index(index_code.upper())
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Cannot get realtime data for {index_code}")
    
    return data


@app.get("/api/market/indices/all", tags=["Market Data"])
async def get_all_indices_realtime():
    """
    Lấy dữ liệu realtime tất cả các chỉ số chính
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    df = collector.get_all_indices_realtime()
    
    return {
        "timestamp": datetime.now().isoformat(),
        "count": len(df),
        "data": df.to_dict(orient='records') if not df.empty else []
    }


@app.get("/api/market/freefloat/{symbol}", tags=["Market Data"])
async def get_freefloat(symbol: str):
    """
    Lấy thông tin tỷ lệ Freefloat
    
    % cổ phiếu tự do chuyển nhượng trên thị trường
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.get_freefloat_data(symbol)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Cannot get freefloat data for {symbol}")
    
    return data


@app.get("/api/market/foreign-ownership/{symbol}", tags=["Market Data"])
async def get_foreign_ownership(symbol: str):
    """
    Lấy thông tin tỷ lệ sở hữu nước ngoài
    
    Bao gồm: % sở hữu, room tối đa, room còn lại
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    data = collector.get_foreign_ownership(symbol)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Cannot get foreign ownership for {symbol}")
    
    return data


@app.get("/api/market/index-components/{index_code}", tags=["Market Data"])
async def get_index_components(index_code: str = "VN30"):
    """
    Lấy danh sách thành phần của chỉ số
    
    VD: VN30, HNX30, VNMIDCAP, etc.
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    df = collector.get_index_components(index_code.upper())
    
    return {
        "index_code": index_code.upper(),
        "count": len(df),
        "components": df.to_dict(orient='records') if not df.empty else []
    }


@app.get("/api/market/summary/{exchange}", tags=["Market Data"])
async def get_market_summary(exchange: str = "HOSE"):
    """
    Lấy tổng hợp thị trường
    
    Exchanges: HOSE, HNX, UPCOM
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    summary = collector.get_market_summary(exchange.upper())
    
    return summary


@app.get("/api/market/high-foreign-ownership", tags=["Market Data"])
async def get_stocks_high_foreign_ownership(
    exchange: str = Query("HOSE", description="Exchange: HOSE, HNX, UPCOM"),
    min_percent: float = Query(40, description="Minimum foreign ownership %")
):
    """
    Lấy danh sách CP có tỷ lệ sở hữu nước ngoài cao
    """
    from src.data_collection.market_data import MarketDataCollector
    
    collector = MarketDataCollector()
    df = collector.get_stocks_by_foreign_ownership(exchange.upper(), min_percent)
    
    return {
        "exchange": exchange.upper(),
        "min_foreign_percent": min_percent,
        "count": len(df),
        "stocks": df.to_dict(orient='records') if not df.empty else []
    }


# --- Financial Data Endpoints ---

@app.get("/api/financial/valuation/{symbol}", tags=["Financial Data"])
async def get_valuation_data(symbol: str):
    """
    Lấy dữ liệu định giá
    
    Bao gồm: P/E, P/B, P/S, EV/EBITDA, EPS, Book Value, Dividend Yield
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    data = collector.get_valuation_data(symbol)
    
    if not data:
        raise HTTPException(status_code=404, detail=f"Cannot get valuation data for {symbol}")
    
    return data


@app.get("/api/financial/fundamentals/{symbol}", tags=["Financial Data"])
async def get_fundamental_data(
    symbol: str,
    period_type: str = Query("quarter", description="Period: quarter or year"),
    periods: int = Query(8, description="Number of periods")
):
    """
    Lấy dữ liệu tài chính cơ bản
    
    Bao gồm: Doanh thu, Lợi nhuận, Biên lợi nhuận, ROE, ROA, Đòn bẩy
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_fundamental_data(symbol, period_type, periods)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Cannot get fundamentals for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "period_type": period_type,
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/financial/balance-sheet/{symbol}", tags=["Financial Data"])
async def get_balance_sheet(
    symbol: str,
    periods: int = Query(8, description="Number of periods")
):
    """
    Lấy Bảng cân đối kế toán
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_balance_sheet(symbol, periods)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Cannot get balance sheet for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/financial/income-statement/{symbol}", tags=["Financial Data"])
async def get_income_statement(
    symbol: str,
    periods: int = Query(8, description="Number of periods")
):
    """
    Lấy Báo cáo kết quả kinh doanh
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_income_statement(symbol, periods)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Cannot get income statement for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/financial/cash-flow/{symbol}", tags=["Financial Data"])
async def get_cash_flow(
    symbol: str,
    periods: int = Query(8, description="Number of periods")
):
    """
    Lấy Báo cáo lưu chuyển tiền tệ
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_cash_flow(symbol, periods)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Cannot get cash flow for {symbol}")
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "count": len(df),
        "data": df.to_dict(orient='records')
    }


@app.get("/api/financial/dividends/{symbol}", tags=["Financial Data"])
async def get_dividend_history(symbol: str):
    """
    Lấy lịch sử chi trả cổ tức
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_dividend_history(symbol)
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "count": len(df),
        "data": df.to_dict(orient='records') if not df.empty else []
    }


@app.get("/api/financial/peer-comparison/{symbol}", tags=["Financial Data"])
async def get_peer_comparison(symbol: str):
    """
    So sánh với các công ty cùng ngành
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    df = collector.get_peer_comparison(symbol)
    
    return {
        "symbol": symbol.upper().replace('.VN', ''),
        "count": len(df),
        "peers": df.to_dict(orient='records') if not df.empty else []
    }


@app.get("/api/financial/summary/{symbol}", tags=["Financial Data"])
async def get_financial_summary(symbol: str):
    """
    Lấy tổng hợp dữ liệu tài chính
    
    Bao gồm: Định giá, Chỉ số tài chính, BCĐKT, Dòng tiền
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    collector = FinancialDataCollector()
    summary = collector.get_financial_summary(symbol)
    
    if not summary:
        raise HTTPException(status_code=404, detail=f"Cannot get financial summary for {symbol}")
    
    return summary


@app.post("/api/financial/dcf-valuation", tags=["Financial Data"])
async def calculate_dcf_valuation(
    fcf: float = Query(..., description="Free Cash Flow hiện tại (tỷ VND)"),
    growth_rate: float = Query(10, description="Tỷ lệ tăng trưởng FCF (%)"),
    discount_rate: float = Query(12, description="Tỷ lệ chiết khấu/WACC (%)"),
    terminal_growth: float = Query(3, description="Tỷ lệ tăng trưởng vĩnh viễn (%)"),
    years: int = Query(10, description="Số năm dự báo"),
    shares: float = Query(1000, description="Số cổ phiếu (triệu CP)")
):
    """
    Tính giá trị nội tại theo DCF (Discounted Cash Flow)
    """
    from src.data_collection.financial_data import DCFValuation
    
    result = DCFValuation.calculate_intrinsic_value(
        fcf=fcf,
        growth_rate=growth_rate,
        discount_rate=discount_rate,
        terminal_growth=terminal_growth,
        years=years,
        shares_outstanding=shares
    )
    
    return result


@app.post("/api/financial/stock-screening", tags=["Financial Data"])
async def screen_stocks(
    exchange: str = Query("HOSE", description="Exchange: HOSE, HNX"),
    pe_max: float = Query(None, description="P/E tối đa"),
    pb_max: float = Query(None, description="P/B tối đa"),
    roe_min: float = Query(None, description="ROE tối thiểu (%)"),
    market_cap_min: float = Query(None, description="Vốn hóa tối thiểu (tỷ VND)")
):
    """
    Lọc cổ phiếu theo tiêu chí tài chính
    """
    from src.data_collection.financial_data import FinancialDataCollector
    
    criteria = {'exchange': exchange}
    if pe_max: criteria['pe_max'] = pe_max
    if pb_max: criteria['pb_max'] = pb_max
    if roe_min: criteria['roe_min'] = roe_min
    if market_cap_min: criteria['market_cap_min'] = market_cap_min
    
    collector = FinancialDataCollector()
    df = collector.screen_stocks(criteria)
    
    return {
        "criteria": criteria,
        "count": len(df),
        "stocks": df.to_dict(orient='records') if not df.empty else []
    }


# --- Industry Data Endpoints ---

@app.get("/api/industry/sectors", tags=["Industry Data"])
async def get_all_sectors():
    """
    Lấy danh sách các ngành
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    
    return {
        "sectors": collector.get_all_sector_names(),
        "vn_sectors": list(collector.VN_SECTORS.keys()),
        "icb_sectors": collector.SECTORS
    }


@app.get("/api/industry/sector/{sector_name}", tags=["Industry Data"])
async def get_sector_stocks(sector_name: str):
    """
    Lấy danh sách CP trong một ngành
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    stocks = collector.get_stocks_by_sector(sector_name)
    
    if not stocks:
        raise HTTPException(status_code=404, detail=f"Sector '{sector_name}' not found")
    
    return {
        "sector": sector_name,
        "count": len(stocks),
        "stocks": stocks
    }


@app.get("/api/industry/performance/{sector_name}", tags=["Industry Data"])
async def get_sector_performance(
    sector_name: str,
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy hiệu suất ngành
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    df = collector.get_vn_sector_performance(sector_name, from_date, to_date)
    
    if df.empty:
        raise HTTPException(status_code=404, detail=f"Cannot get performance for {sector_name}")
    
    return {
        "sector": sector_name,
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "stocks": df.to_dict(orient='records')
    }


@app.get("/api/industry/all-performance", tags=["Industry Data"])
async def get_all_sectors_performance(
    from_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    to_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    """
    Lấy hiệu suất tất cả các ngành
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    df = collector.get_all_sectors_performance(from_date, to_date)
    
    return {
        "from_date": from_date,
        "to_date": to_date,
        "count": len(df),
        "sectors": df.to_dict(orient='records') if not df.empty else []
    }


@app.get("/api/industry/market-breadth/{exchange}", tags=["Industry Data"])
async def get_market_breadth(exchange: str = "HOSE"):
    """
    Lấy độ rộng thị trường
    
    Advances, Declines, Unchanged, Ceiling, Floor
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    data = collector.get_market_breadth(exchange.upper())
    
    return data


@app.get("/api/industry/supply-demand/{exchange}", tags=["Industry Data"])
async def get_supply_demand(exchange: str = "HOSE"):
    """
    Lấy dữ liệu cung cầu thị trường
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    data = collector.get_supply_demand(exchange.upper())
    
    return data


@app.get("/api/industry/sector-rotation", tags=["Industry Data"])
async def get_sector_rotation(periods: int = Query(4, description="Number of periods (weeks)")):
    """
    Phân tích sector rotation (dòng tiền luân chuyển)
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    analysis = collector.get_sector_rotation_analysis(periods)
    
    return analysis


@app.post("/api/industry/comparison", tags=["Industry Data"])
async def compare_stocks_in_industry(symbols: List[str]):
    """
    So sánh các công ty trong cùng ngành
    
    Body: ["VCB", "BID", "CTG", "TCB", "MBB"]
    """
    from src.data_collection.industry_data import IndustryDataCollector
    
    collector = IndustryDataCollector()
    df = collector.get_industry_comparison(symbols)
    
    return {
        "count": len(df),
        "comparison": df.to_dict(orient='records') if not df.empty else []
    }


if __name__ == "__main__":
    import uvicorn
    
    print("Starting KLTN Stock Prediction API v2.0...")
    print("📊 Documentation: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("🔗 Database: SQLite/PostgreSQL")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
