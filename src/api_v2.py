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

# Mount static files
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/dashboard", tags=["Dashboard"], include_in_schema=False)
async def dashboard():
    """Serve the main dashboard page"""
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"error": "Dashboard not found"}


# =====================================================
# ROOT & HEALTH ENDPOINTS
# =====================================================

@app.get("/", tags=["Root"])
async def root():
    """API Root - Redirect to dashboard or show API info"""
    # Return dashboard if exists, otherwise API info
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    
    return {
        "name": "KLTN Stock Prediction API",
        "version": "2.0.0",
        "database": "PostgreSQL",
        "dashboard": "/dashboard",
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
    days: int = Query(default=365, ge=7, le=1825, description="Số ngày dữ liệu (max 5 năm)"),
    db: Session = Depends(get_db)
):
    """
    Thu thập dữ liệu THỰC từ VNDirect API và lưu vào database.
    
    - **symbol**: Mã cổ phiếu (VNM, FPT, VCB, etc.)
    - **days**: Số ngày dữ liệu cần lấy (mặc định 365 ngày = 1 năm)
    
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
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Fetch data from VNDirect
        logger.info(f"Fetching data for {symbol} from VNDirect...")
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
    days: int = Query(default=365, ge=7, le=1825),
    background_tasks: BackgroundTasks = None,
    db: Session = Depends(get_db)
):
    """
    Thu thập dữ liệu cho TẤT CẢ cổ phiếu trong database.
    
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
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
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


if __name__ == "__main__":
    import uvicorn
    
    print("Starting KLTN Stock Prediction API v2.0...")
    print("📊 Documentation: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("🔗 Database: PostgreSQL")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
