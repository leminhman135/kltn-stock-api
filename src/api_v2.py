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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.database.connection import get_db, engine
from src.database.models import (
    Stock, StockPrice, TechnicalIndicator,
    SentimentAnalysis, ModelMetrics, Prediction, 
    NewsArticle, AnalyzedNews, SentimentSummary, Base
)
# Import extended models so they get created on startup
try:
    from src.database.extended_models import (
        ForeignTrading, ProprietaryTrading, DetailedTrading,
        MarketIndex, StockOwnership, IndexComponent,
        Valuation, FinancialStatement, FinancialRatio,
        SectorStatistics, MarketBreadth, SupplyDemand
    )
    logger.info("✅ Extended database models imported")
except ImportError as e:
    logger.warning(f"⚠️ Extended models not available: {e}")

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
    
    # Database initialization
    try:
        # Create all tables (checkfirst=True to skip existing tables/indexes)
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("✅ Database tables created/verified successfully!")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Continue anyway - maybe DB is already set up
    
    # Initialize scheduler (optional, non-blocking)
    try:
        import importlib
        # Check if scheduler module exists
        scheduler_module = importlib.import_module('src.scheduler.daily_scheduler')
        init_scheduler = getattr(scheduler_module, 'init_scheduler', None)
        
        if init_scheduler:
            scheduler = init_scheduler()
            logger.info("✅ Background scheduler started successfully!")
            logger.info(f"   → Next run: {scheduler.get_next_run_time()}")
        else:
            logger.warning("⚠️ Scheduler init function not found")
    except ImportError as e:
        logger.warning(f"⚠️ Scheduler module not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Scheduler initialization failed (non-fatal): {e}")
        logger.info("   → API will continue without automatic scheduling")
    
    yield
    
    # Shutdown
    logger.info("👋 Shutting down KLTN Stock Prediction API...")
    try:
        import importlib
        scheduler_module = importlib.import_module('src.scheduler.daily_scheduler')
        get_scheduler = getattr(scheduler_module, 'get_scheduler', None)
        
        if get_scheduler:
            scheduler = get_scheduler()
            if scheduler:
                scheduler.stop()
                logger.info("✅ Scheduler stopped")
    except Exception as e:
        logger.warning(f"⚠️ Scheduler shutdown warning: {e}")

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
# INCLUDE ML MODELS ROUTER
# =====================================================
try:
    from src.api.ml_endpoints import router as ml_router
    app.include_router(ml_router)
    logger.info("✅ ML Models router loaded successfully")
except ImportError as e:
    logger.warning(f"⚠️ ML Models router not loaded: {e}")

# Include Advanced ML Router (FinBERT, LSTM, GRU, Features)
try:
    from src.api.advanced_ml_endpoints import router as advanced_ml_router
    app.include_router(advanced_ml_router)
    logger.info("✅ Advanced ML router loaded (FinBERT, Deep Learning)")
except ImportError as e:
    logger.warning(f"⚠️ Advanced ML router not loaded: {e}")


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


@app.get("/api/prices/{symbol}/historical", tags=["Prices"])
async def get_historical_prices(
    symbol: str,
    from_date: str = Query(..., description="Ngày bắt đầu (YYYY-MM-DD), VD: 2020-01-01"),
    to_date: str = Query(..., description="Ngày kết thúc (YYYY-MM-DD), VD: 2024-12-31"),
    source: str = Query("auto", description="Nguồn dữ liệu: 'database', 'api', hoặc 'auto'"),
    db: Session = Depends(get_db)
):
    """
    Lấy dữ liệu giá lịch sử - hỗ trợ tìm kiếm nhiều năm
    
    - **from_date**: Ngày bắt đầu (có thể từ nhiều năm trước, VD: 2015-01-01)
    - **to_date**: Ngày kết thúc
    - **source**: 
        - 'database': Chỉ lấy từ database
        - 'api': Lấy trực tiếp từ VNDirect API
        - 'auto': Ưu tiên database, nếu không có thì gọi API
    
    Ví dụ:
    - /api/prices/VNM/historical?from_date=2020-01-01&to_date=2024-12-31
    - /api/prices/FPT/historical?from_date=2015-01-01&to_date=2023-12-31&source=api
    """
    from src.data_collection import TradingDataCollector
    
    clean_symbol = symbol.upper().replace('.VN', '')
    
    result = {
        "symbol": clean_symbol,
        "from_date": from_date,
        "to_date": to_date,
        "source_used": None,
        "count": 0,
        "data": []
    }
    
    # Try database first if source is 'database' or 'auto'
    if source in ['database', 'auto']:
        stock = db.query(Stock).filter(Stock.symbol == clean_symbol).first()
        
        if stock:
            prices = db.query(StockPrice).filter(
                StockPrice.stock_id == stock.id,
                StockPrice.date >= from_date,
                StockPrice.date <= to_date
            ).order_by(StockPrice.date).all()
            
            if prices:
                result["source_used"] = "database"
                result["count"] = len(prices)
                result["data"] = [
                    {
                        "date": p.date.isoformat(),
                        "open": p.open,
                        "high": p.high,
                        "low": p.low,
                        "close": p.close,
                        "volume": p.volume,
                        "change_percent": p.change_percent
                    }
                    for p in prices
                ]
                return result
    
    # If no data in database or source is 'api', fetch from VNDirect
    if source in ['api', 'auto']:
        try:
            collector = TradingDataCollector()
            df = collector.get_detailed_trading_data(clean_symbol, from_date, to_date)
            
            if not df.empty:
                result["source_used"] = "vndirect_api"
                result["count"] = len(df)
                result["data"] = df.to_dict(orient='records')
                return result
        except Exception as e:
            if source == 'api':
                raise HTTPException(status_code=500, detail=f"API error: {str(e)}")
    
    # No data found
    if result["count"] == 0:
        raise HTTPException(
            status_code=404, 
            detail=f"Không tìm thấy dữ liệu cho {clean_symbol} từ {from_date} đến {to_date}"
        )
    
    return result


@app.get("/api/prices/{symbol}/years", tags=["Prices"])
async def get_prices_by_year(
    symbol: str,
    year: int = Query(..., description="Năm cần xem (VD: 2020, 2021, 2022...)"),
    db: Session = Depends(get_db)
):
    """
    Lấy toàn bộ dữ liệu giá của một năm cụ thể
    
    - **year**: Năm cần lấy dữ liệu (2015-2025)
    
    Ví dụ: /api/prices/VNM/years?year=2023
    """
    from src.data_collection import TradingDataCollector
    
    clean_symbol = symbol.upper().replace('.VN', '')
    from_date = f"{year}-01-01"
    to_date = f"{year}-12-31"
    
    # Try database first
    stock = db.query(Stock).filter(Stock.symbol == clean_symbol).first()
    
    if stock:
        prices = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id,
            StockPrice.date >= from_date,
            StockPrice.date <= to_date
        ).order_by(StockPrice.date).all()
        
        if prices:
            return {
                "symbol": clean_symbol,
                "year": year,
                "source": "database",
                "trading_days": len(prices),
                "first_date": prices[0].date.isoformat(),
                "last_date": prices[-1].date.isoformat(),
                "year_open": prices[0].open,
                "year_close": prices[-1].close,
                "year_high": max(p.high for p in prices),
                "year_low": min(p.low for p in prices),
                "year_change_percent": round(((prices[-1].close / prices[0].open) - 1) * 100, 2),
                "total_volume": sum(p.volume for p in prices),
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
    
    # Fetch from API
    try:
        collector = TradingDataCollector()
        df = collector.get_detailed_trading_data(clean_symbol, from_date, to_date)
        
        if not df.empty:
            return {
                "symbol": clean_symbol,
                "year": year,
                "source": "vndirect_api",
                "trading_days": len(df),
                "first_date": df['date'].iloc[0] if 'date' in df.columns else from_date,
                "last_date": df['date'].iloc[-1] if 'date' in df.columns else to_date,
                "data": df.to_dict(orient='records')
            }
    except Exception as e:
        pass
    
    raise HTTPException(
        status_code=404, 
        detail=f"Không tìm thấy dữ liệu cho {clean_symbol} năm {year}"
    )


@app.get("/api/prices/{symbol}/range", tags=["Prices"])
async def get_prices_date_range(
    symbol: str,
    years: int = Query(5, description="Số năm gần nhất (mặc định 5 năm)"),
    db: Session = Depends(get_db)
):
    """
    Lấy dữ liệu giá trong khoảng N năm gần nhất
    
    - **years**: Số năm cần lấy (1-10 năm)
    
    Ví dụ: /api/prices/VNM/range?years=5 (lấy 5 năm gần nhất)
    """
    from src.data_collection import TradingDataCollector
    
    clean_symbol = symbol.upper().replace('.VN', '')
    
    to_date = datetime.now().strftime('%Y-%m-%d')
    from_date = (datetime.now() - timedelta(days=years*365)).strftime('%Y-%m-%d')
    
    # Try database first
    stock = db.query(Stock).filter(Stock.symbol == clean_symbol).first()
    
    if stock:
        prices = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id,
            StockPrice.date >= from_date,
            StockPrice.date <= to_date
        ).order_by(StockPrice.date).all()
        
        if prices and len(prices) > 100:  # At least 100 days of data
            return {
                "symbol": clean_symbol,
                "years": years,
                "from_date": from_date,
                "to_date": to_date,
                "source": "database",
                "trading_days": len(prices),
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
    
    # Fetch from API
    try:
        collector = TradingDataCollector()
        df = collector.get_detailed_trading_data(clean_symbol, from_date, to_date)
        
        if not df.empty:
            return {
                "symbol": clean_symbol,
                "years": years,
                "from_date": from_date,
                "to_date": to_date,
                "source": "vndirect_api",
                "trading_days": len(df),
                "data": df.to_dict(orient='records')
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lấy dữ liệu: {str(e)}")
    
    raise HTTPException(
        status_code=404, 
        detail=f"Không tìm thấy dữ liệu cho {clean_symbol}"
    )


# =====================================================
# BẢNG GIÁ THEO NGÀY (Market Board by Date)
# =====================================================

@app.get("/api/market-board/dates", tags=["Market Board"])
async def get_available_dates(
    limit: int = Query(30, description="Số ngày gần nhất"),
    db: Session = Depends(get_db)
):
    """
    📅 Lấy danh sách các ngày có dữ liệu giao dịch
    
    Trả về danh sách các ngày có thể xem bảng giá.
    Mỗi ngày là 1 trang riêng biệt.
    """
    # Lấy các ngày có dữ liệu
    dates = db.query(StockPrice.date).distinct().order_by(
        desc(StockPrice.date)
    ).limit(limit).all()
    
    if not dates:
        raise HTTPException(status_code=404, detail="Không có dữ liệu")
    
    date_list = [d[0].isoformat() for d in dates]
    
    return {
        "status": "ok",
        "total_dates": len(date_list),
        "latest_date": date_list[0] if date_list else None,
        "oldest_date": date_list[-1] if date_list else None,
        "dates": date_list
    }


@app.get("/api/market/board-compare", tags=["Market Board"])
async def compare_dates(
    date1: str = Query(..., description="Ngày 1 (YYYY-MM-DD)"),
    date2: str = Query(..., description="Ngày 2 (YYYY-MM-DD)"),
    symbol: Optional[str] = Query(None, description="Mã CP cụ thể (optional)"),
    db: Session = Depends(get_db)
):
    """
    📊 So sánh bảng giá giữa 2 ngày
    
    Ví dụ: /api/market/board-compare?date1=2024-11-28&date2=2024-11-29
    """
    from datetime import datetime as dt
    
    try:
        d1 = dt.strptime(date1, '%Y-%m-%d').date()
        d2 = dt.strptime(date2, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày")
    
    # Query cho cả 2 ngày
    query1 = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == d1,
        Stock.is_active == True
    )
    
    query2 = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == d2,
        Stock.is_active == True
    )
    
    if symbol:
        query1 = query1.filter(Stock.symbol == symbol.upper())
        query2 = query2.filter(Stock.symbol == symbol.upper())
    
    results1 = {stock.symbol: price for stock, price in query1.all()}
    results2 = {stock.symbol: price for stock, price in query2.all()}
    
    if not results1 or not results2:
        raise HTTPException(status_code=404, detail="Không có dữ liệu")
    
    # Build comparison
    comparison = []
    for sym in results1.keys():
        if sym in results2:
            p1 = results1[sym]
            p2 = results2[sym]
            
            price_change = p2.close - p1.close
            price_change_pct = (price_change / p1.close * 100) if p1.close > 0 else 0
            volume_change = p2.volume - p1.volume
            volume_change_pct = (volume_change / p1.volume * 100) if p1.volume > 0 else 0
            
            comparison.append({
                "symbol": sym,
                "date1_close": p1.close,
                "date2_close": p2.close,
                "price_change": round(price_change, 2),
                "price_change_percent": round(price_change_pct, 2),
                "date1_volume": p1.volume,
                "date2_volume": p2.volume,
                "volume_change": volume_change,
                "volume_change_percent": round(volume_change_pct, 2)
            })
    
    # Sort by price change
    comparison.sort(key=lambda x: x["price_change_percent"], reverse=True)
    
    return {
        "date1": date1,
        "date2": date2,
        "total_stocks": len(comparison),
        "comparison": comparison
    }


@app.get("/api/market-board/{date}", tags=["Market Board"])
async def get_market_board_by_date(
    date: str,
    exchange: Optional[str] = Query(None, description="Sàn: HOSE, HNX, UPCOM"),
    sort_by: str = Query("symbol", description="Sắp xếp: symbol, change_percent, volume, value"),
    order: str = Query("asc", description="Thứ tự: asc, desc"),
    db: Session = Depends(get_db)
):
    """
    📊 Lấy BẢNG GIÁ của TẤT CẢ cổ phiếu trong MỘT NGÀY cụ thể
    
    - **date**: Ngày cần xem (YYYY-MM-DD)
    - **exchange**: Lọc theo sàn (HOSE, HNX, UPCOM)
    - **sort_by**: Cột sắp xếp
    - **order**: Tăng/giảm dần
    
    Ví dụ:
    - /api/market-board/2024-11-28 → Xem bảng giá ngày 28/11/2024
    - /api/market-board/2024-11-29?sort_by=change_percent&order=desc → Top tăng mạnh nhất
    """
    from datetime import datetime as dt
    
    try:
        target_date = dt.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày. Dùng YYYY-MM-DD")
    
    # Query stocks với giá của ngày được chọn
    query = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == target_date,
        Stock.is_active == True
    )
    
    if exchange:
        query = query.filter(Stock.exchange == exchange.upper())
    
    results = query.all()
    
    if not results:
        raise HTTPException(
            status_code=404, 
            detail=f"Không có dữ liệu bảng giá ngày {date}"
        )
    
    # Build market board data
    board_data = []
    total_volume = 0
    total_value = 0
    advances = 0  # Số mã tăng
    declines = 0  # Số mã giảm
    unchanged = 0  # Số mã đứng giá
    
    for stock, price in results:
        # Tính thay đổi giá
        change = price.close - price.open
        change_percent = (change / price.open * 100) if price.open > 0 else 0
        
        # Tính giá trị giao dịch (ước tính)
        value = price.close * price.volume
        
        if change > 0:
            advances += 1
        elif change < 0:
            declines += 1
        else:
            unchanged += 1
        
        total_volume += price.volume
        total_value += value
        
        board_data.append({
            "symbol": stock.symbol,
            "name": stock.name,
            "exchange": stock.exchange,
            "sector": stock.sector,
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume,
            "value": round(value, 0),
            "change": round(change, 2),
            "change_percent": round(change_percent, 2),
        })
    
    # Sắp xếp
    reverse = (order == "desc")
    if sort_by == "change_percent":
        board_data.sort(key=lambda x: x["change_percent"], reverse=reverse)
    elif sort_by == "volume":
        board_data.sort(key=lambda x: x["volume"], reverse=reverse)
    elif sort_by == "value":
        board_data.sort(key=lambda x: x["value"], reverse=reverse)
    else:
        board_data.sort(key=lambda x: x["symbol"], reverse=reverse)
    
    return {
        "status": "ok",
        "date": date,
        "exchange": exchange or "ALL",
        "summary": {
            "total_stocks": len(board_data),
            "advances": advances,
            "declines": declines,
            "unchanged": unchanged,
            "total_volume": total_volume,
            "total_value": total_value
        },
        "sort_by": sort_by,
        "order": order,
        "data": board_data
    }


@app.get("/api/market-board/{date}/top-gainers", tags=["Market Board"])
async def get_top_gainers(
    date: str,
    limit: int = Query(10, description="Số lượng top"),
    exchange: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    📈 Top cổ phiếu TĂNG GIÁ mạnh nhất trong ngày
    """
    from datetime import datetime as dt
    
    try:
        target_date = dt.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày")
    
    query = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == target_date,
        Stock.is_active == True
    )
    
    if exchange:
        query = query.filter(Stock.exchange == exchange.upper())
    
    results = query.all()
    
    # Calculate change percent và sort
    gainers = []
    for stock, price in results:
        change_percent = ((price.close - price.open) / price.open * 100) if price.open > 0 else 0
        if change_percent > 0:
            gainers.append({
                "symbol": stock.symbol,
                "name": stock.name,
                "exchange": stock.exchange,
                "close": price.close,
                "change_percent": round(change_percent, 2),
                "volume": price.volume
            })
    
    gainers.sort(key=lambda x: x["change_percent"], reverse=True)
    
    return {
        "date": date,
        "exchange": exchange or "ALL",
        "total": len(gainers),
        "top_gainers": gainers[:limit]
    }


@app.get("/api/market-board/{date}/top-losers", tags=["Market Board"])
async def get_top_losers(
    date: str,
    limit: int = Query(10, description="Số lượng top"),
    exchange: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    📉 Top cổ phiếu GIẢM GIÁ mạnh nhất trong ngày
    """
    from datetime import datetime as dt
    
    try:
        target_date = dt.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày")
    
    query = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == target_date,
        Stock.is_active == True
    )
    
    if exchange:
        query = query.filter(Stock.exchange == exchange.upper())
    
    results = query.all()
    
    # Calculate change percent và sort
    losers = []
    for stock, price in results:
        change_percent = ((price.close - price.open) / price.open * 100) if price.open > 0 else 0
        if change_percent < 0:
            losers.append({
                "symbol": stock.symbol,
                "name": stock.name,
                "exchange": stock.exchange,
                "close": price.close,
                "change_percent": round(change_percent, 2),
                "volume": price.volume
            })
    
    losers.sort(key=lambda x: x["change_percent"])  # Sort ascending (most negative first)
    
    return {
        "date": date,
        "exchange": exchange or "ALL",
        "total": len(losers),
        "top_losers": losers[:limit]
    }


@app.get("/api/market-board/{date}/top-volume", tags=["Market Board"])
async def get_top_volume(
    date: str,
    limit: int = Query(10, description="Số lượng top"),
    exchange: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    🔥 Top cổ phiếu có KHỐI LƯỢNG giao dịch cao nhất trong ngày
    """
    from datetime import datetime as dt
    
    try:
        target_date = dt.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày")
    
    query = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == target_date,
        Stock.is_active == True
    )
    
    if exchange:
        query = query.filter(Stock.exchange == exchange.upper())
    
    results = query.all()
    
    # Sort by volume
    volume_list = []
    for stock, price in results:
        change_percent = ((price.close - price.open) / price.open * 100) if price.open > 0 else 0
        volume_list.append({
            "symbol": stock.symbol,
            "name": stock.name,
            "exchange": stock.exchange,
            "close": price.close,
            "change_percent": round(change_percent, 2),
            "volume": price.volume,
            "value": round(price.close * price.volume, 0)
        })
    
    volume_list.sort(key=lambda x: x["volume"], reverse=True)
    
    return {
        "date": date,
        "exchange": exchange or "ALL",
        "total": len(volume_list),
        "top_volume": volume_list[:limit]
    }


@app.get("/api/market-board/{date}/export", tags=["Market Board"])
async def export_market_board(
    date: str,
    format: str = Query("json", description="Định dạng: json, csv"),
    db: Session = Depends(get_db)
):
    """
    📁 Export bảng giá của ngày ra file (JSON hoặc CSV)
    """
    from datetime import datetime as dt
    from fastapi.responses import Response
    
    try:
        target_date = dt.strptime(date, '%Y-%m-%d').date()
    except ValueError:
        raise HTTPException(status_code=400, detail="Sai định dạng ngày")
    
    results = db.query(Stock, StockPrice).join(
        StockPrice, Stock.id == StockPrice.stock_id
    ).filter(
        StockPrice.date == target_date,
        Stock.is_active == True
    ).all()
    
    if not results:
        raise HTTPException(status_code=404, detail=f"Không có dữ liệu ngày {date}")
    
    board_data = []
    for stock, price in results:
        change_percent = ((price.close - price.open) / price.open * 100) if price.open > 0 else 0
        board_data.append({
            "symbol": stock.symbol,
            "name": stock.name,
            "exchange": stock.exchange,
            "open": price.open,
            "high": price.high,
            "low": price.low,
            "close": price.close,
            "volume": price.volume,
            "change_percent": round(change_percent, 2)
        })
    
    board_data.sort(key=lambda x: x["symbol"])
    
    if format == "csv":
        import csv
        import io
        
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=board_data[0].keys())
        writer.writeheader()
        writer.writerows(board_data)
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=market_board_{date}.csv"
            }
        )
    
    return {
        "date": date,
        "total": len(board_data),
        "data": board_data
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

# Import ML models with error handling
try:
    from src.model import StockMLModel, quick_predict, train_model as train_ml_model, predict_stock, get_model
    ML_AVAILABLE = True
    logger.info("✅ ML models loaded successfully")
except ImportError as e:
    ML_AVAILABLE = False
    logger.warning(f"⚠️ ML models not available: {e}")
    
    # Fallback quick_predict
    def quick_predict(prices, steps=7):
        import numpy as np
        from datetime import datetime, timedelta
        if len(prices) < 5:
            return {'predictions': [], 'error': 'Need at least 5 prices'}
        prices = np.array(prices)
        last = prices[-1]
        trend = (prices[-1] - prices[0]) / len(prices) if len(prices) > 1 else 0
        preds = [round(last + trend * (i+1) * 0.9**i, 2) for i in range(steps)]
        today = datetime.now()
        dates = [(today + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(steps)]
        return {'predictions': preds, 'dates': dates, 'model': 'simple_trend', 'confidence': 0.4}


@app.post("/api/predictions/train/{symbol}", tags=["Predictions"])
async def train_model_endpoint(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    Train ML models on stock price data
    Returns training status and metrics
    """
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    # Get price data
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(StockPrice.date).all()
    
    if len(prices) < 50:
        raise HTTPException(status_code=400, detail=f"Not enough data: {len(prices)} rows (need 50+)")
    
    # Convert to DataFrame
    import pandas as pd
    df = pd.DataFrame([{
        'date': p.date,
        'open': p.open,
        'high': p.high,
        'low': p.low,
        'close': p.close,
        'volume': p.volume or 0
    } for p in prices])
    
    # Train model
    if ML_AVAILABLE:
        result = train_ml_model(symbol, df)
        if result.get('success'):
            return {
                "status": "success",
                "symbol": symbol,
                "data_points": len(prices),
                "models": result.get('models', {}),
                "message": "Models trained successfully"
            }
        else:
            raise HTTPException(status_code=500, detail=result.get('error', 'Training failed'))
    else:
        raise HTTPException(status_code=500, detail="ML library not available")


@app.post("/api/predictions/predict", tags=["Predictions"])
async def create_prediction(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """
    Generate price predictions using ML models
    Automatically trains model if not cached
    """
    try:
        stock = db.query(Stock).filter(Stock.symbol == request.symbol).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {request.symbol} not found")
        
        # Get price data
        prices = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id
        ).order_by(StockPrice.date).all()
        
        if len(prices) < 5:
            raise HTTPException(status_code=400, detail=f"Not enough data: {len(prices)} rows")
        
        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame([{
            'date': p.date,
            'open': p.open,
            'high': p.high,
            'low': p.low,
            'close': p.close,
            'volume': p.volume or 0
        } for p in prices])
        
        # Use ML prediction if available
        if ML_AVAILABLE and len(prices) >= 50:
            result = predict_stock(request.symbol, df, steps=request.periods, model_type=request.model_type)
        else:
            # Fallback to quick prediction
            close_prices = [p.close for p in prices]
            result = quick_predict(close_prices, steps=request.periods)
        
        result['symbol'] = request.symbol
        result['data_points'] = len(prices)
        
        # Save predictions to database
        from datetime import datetime
        try:
            for pred_date, pred_price in zip(result.get('dates', []), result.get('predictions', [])):
                prediction = Prediction(
                    stock_id=stock.id,
                    target_date=datetime.strptime(pred_date, '%Y-%m-%d').date(),
                    predicted_close=float(pred_price),
                    confidence=float(result.get('confidence', 0.5)),
                    model_name=str(result.get('model', 'ensemble')),
                    created_at=datetime.now()
                )
                db.merge(prediction)
            db.commit()
        except Exception as db_err:
            logger.warning(f"Could not save predictions to DB: {db_err}")
            db.rollback()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Fallback to quick predict
        try:
            close_prices = [p.close for p in prices]
            result = quick_predict(close_prices, steps=request.periods)
            result['symbol'] = request.symbol
            result['error_fallback'] = str(e)
            return result
        except:
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/predictions/quick/{symbol}", tags=["Predictions"])
async def quick_prediction(
    symbol: str,
    days: int = 7,
    db: Session = Depends(get_db)
):
    """
    Quick prediction using simple trend + MA
    Fast but less accurate
    """
    stock = db.query(Stock).filter(Stock.symbol == symbol).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(StockPrice.date).limit(100).all()
    
    if len(prices) < 5:
        raise HTTPException(status_code=400, detail="Not enough price data")
    
    close_prices = [p.close for p in prices]
    result = quick_predict(close_prices, steps=days)
    result['symbol'] = symbol
    result['last_price'] = prices[-1].close
    result['last_date'] = str(prices[-1].date)
    
    return result


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
from src.news_relevance import relevance_model

@app.get("/api/news", tags=["News"])
async def get_market_news(limit: int = 50):
    """Lấy tin tức thị trường chung với phân tích sentiment (optimized for 3 days)"""
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
async def get_stock_news(symbol: str, limit: int = 50):
    """Lấy tin tức cho một mã cổ phiếu cụ thể với phân tích sentiment và relevance"""
    try:
        news = news_service.get_all_news(symbol=symbol.upper(), limit=limit)
        summary = news_service.get_sentiment_summary(symbol.upper())
        
        # Calculate relevance scores for each news
        news_with_relevance = []
        for n in news:
            text = f"{n.title} {n.summary}"
            relevance = relevance_model.calculate_relevance_score(text, symbol.upper())
            
            news_with_relevance.append({
                "title": n.title,
                "summary": n.summary,
                "url": n.url,
                "source": n.source,
                "published_at": n.published_at,
                "sentiment": n.sentiment.value,
                "sentiment_score": round(n.sentiment_score, 2),
                "impact": n.impact_prediction,
                "relevance_score": relevance["relevance_score"],
                "relevance_confidence": relevance["confidence"],
                "relevance_explanation": relevance["explanation"],
                "matched_features": relevance["matched_features"]
            })
        
        # Sort by relevance score (highest first)
        news_with_relevance.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "sentiment_summary": {
                "overall": summary.get("overall", "neutral"),
                "avg_score": summary.get("avg_score", 0),
                "positive_count": summary.get("positive_count", 0),
                "negative_count": summary.get("negative_count", 0),
                "neutral_count": summary.get("neutral_count", 0),
                "recommendation": summary.get("recommendation", "Đang phân tích...")
            },
            "total_news": len(news_with_relevance),
            "news": news_with_relevance
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


@app.get("/api/news/features/sentiment", tags=["News"])
async def get_sentiment_features():
    """Lấy danh sách các từ khóa được dùng để phân tích sentiment"""
    try:
        from src.news_service import SentimentAnalyzer
        analyzer = SentimentAnalyzer()
        
        return {
            "status": "success",
            "method": "keyword-based",
            "description": "Phân tích sentiment dựa trên từ khóa tiếng Việt",
            "features": {
                "positive_keywords": {
                    "count": len(analyzer.POSITIVE_KEYWORDS),
                    "examples": analyzer.POSITIVE_KEYWORDS[:20],
                    "categories": [
                        "Tài chính (lợi nhuận, doanh thu, tăng trưởng)",
                        "Kinh doanh (mở rộng, hợp tác, dự án mới)",
                        "Thị trường (tăng điểm, mua ròng, breakout)",
                        "Đánh giá (khuyến nghị mua, nâng rating)"
                    ]
                },
                "negative_keywords": {
                    "count": len(analyzer.NEGATIVE_KEYWORDS),
                    "examples": analyzer.NEGATIVE_KEYWORDS[:20],
                    "categories": [
                        "Tài chính (thua lỗ, nợ xấu, giảm lợi nhuận)",
                        "Kinh doanh (đóng cửa, sa thải, tranh chấp)",
                        "Thị trường (giảm sàn, bán tháo, breakdown)",
                        "Đánh giá (cảnh báo, hạ rating, rủi ro)"
                    ]
                },
                "modifiers": {
                    "count": len(analyzer.STRONG_MODIFIERS),
                    "examples": analyzer.STRONG_MODIFIERS,
                    "description": "Từ nhấn mạnh (tăng hệ số 1.5x)"
                }
            },
            "scoring": {
                "formula": "(positive_count - negative_count) / total_count",
                "range": "[-1.0, 1.0]",
                "classification": {
                    "positive": "score > 0.2",
                    "neutral": "-0.2 <= score <= 0.2",
                    "negative": "score < -0.2"
                }
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/news/features/relevance/{symbol}", tags=["News"])
async def get_relevance_features(symbol: str):
    """Lấy thông tin về features dùng để tính relevance score cho mã cổ phiếu"""
    try:
        features = relevance_model.get_features_explanation(symbol.upper())
        return {
            "status": "success",
            **features
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

@app.post("/api/data/sync-daily", tags=["Data Collection"])
async def sync_daily_data(
    db: Session = Depends(get_db)
):
    """
    🔄 Tự động cập nhật dữ liệu MỚI NHẤT cho tất cả cổ phiếu.
    
    Endpoint này sẽ:
    1. Kiểm tra ngày cuối cùng có dữ liệu trong DB
    2. Fetch dữ liệu từ ngày đó đến hôm nay
    3. Lưu vào database
    
    **Dùng cho:**
    - Cron job hàng ngày (Render Cron)
    - Đảm bảo dữ liệu luôn cập nhật liên tục
    
    **Ví dụ:**
    - DB có dữ liệu đến 28/11 → Sync sẽ fetch từ 28/11 đến hôm nay (1/12)
    """
    from src.data_collection import VNDirectAPI
    import time
    
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    
    if not stocks:
        raise HTTPException(
            status_code=404,
            detail="No stocks in database. Use /api/admin/init-db first."
        )
    
    vndirect = VNDirectAPI()
    today = datetime.now()
    results = []
    total_new_records = 0
    
    for stock in stocks:
        try:
            # Tìm ngày cuối cùng có dữ liệu trong DB
            last_price = db.query(StockPrice).filter(
                StockPrice.stock_id == stock.id
            ).order_by(desc(StockPrice.date)).first()
            
            if last_price:
                # Có dữ liệu → fetch từ ngày cuối + 1
                start_date = datetime.combine(last_price.date, datetime.min.time()) + timedelta(days=1)
            else:
                # Chưa có dữ liệu → fetch 30 ngày gần nhất
                start_date = today - timedelta(days=30)
            
            # Nếu ngày bắt đầu >= hôm nay thì skip (đã có dữ liệu mới nhất)
            if start_date.date() >= today.date():
                results.append({
                    "symbol": stock.symbol,
                    "status": "up_to_date",
                    "last_date": last_price.date.isoformat() if last_price else None,
                    "new_records": 0
                })
                continue
            
            # Fetch dữ liệu
            df = vndirect.get_stock_price(
                symbol=stock.symbol,
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=today.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                results.append({
                    "symbol": stock.symbol,
                    "status": "no_new_data",
                    "last_date": last_price.date.isoformat() if last_price else None,
                    "new_records": 0
                })
                continue
            
            # Lưu vào database
            new_records = 0
            for _, row in df.iterrows():
                # Skip nếu đã tồn tại
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
                    new_records += 1
            
            db.commit()
            total_new_records += new_records
            
            results.append({
                "symbol": stock.symbol,
                "status": "synced",
                "last_date": df['date'].max().strftime('%Y-%m-%d'),
                "new_records": new_records
            })
            
            # Rate limiting
            time.sleep(0.3)
            
        except Exception as e:
            logger.error(f"Error syncing {stock.symbol}: {str(e)}")
            results.append({
                "symbol": stock.symbol,
                "status": "error",
                "error": str(e)
            })
    
    return {
        "status": "completed",
        "timestamp": today.isoformat(),
        "summary": {
            "total_stocks": len(stocks),
            "synced": len([r for r in results if r["status"] == "synced"]),
            "up_to_date": len([r for r in results if r["status"] == "up_to_date"]),
            "errors": len([r for r in results if r["status"] == "error"]),
            "total_new_records": total_new_records
        },
        "results": results
    }


@app.post("/api/data/sync/{symbol}", tags=["Data Collection"])
async def sync_stock_data(
    symbol: str,
    db: Session = Depends(get_db)
):
    """
    🔄 Tự động cập nhật dữ liệu MỚI NHẤT cho MỘT cổ phiếu.
    
    - Kiểm tra ngày cuối trong DB
    - Fetch từ ngày đó đến hôm nay
    - Lưu dữ liệu mới
    
    **Ví dụ:** VNM có dữ liệu đến 28/11 → Sync sẽ fetch 29/11, 30/11, 1/12
    """
    from src.data_collection import VNDirectAPI
    
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if not stock:
        raise HTTPException(
            status_code=404,
            detail=f"Stock {symbol} not found. Use /api/admin/init-db first."
        )
    
    vndirect = VNDirectAPI()
    today = datetime.now()
    
    # Tìm ngày cuối cùng có dữ liệu
    last_price = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(desc(StockPrice.date)).first()
    
    if last_price:
        start_date = datetime.combine(last_price.date, datetime.min.time()) + timedelta(days=1)
        last_date_str = last_price.date.isoformat()
    else:
        start_date = today - timedelta(days=365)  # Fetch 1 năm nếu chưa có dữ liệu
        last_date_str = None
    
    # Nếu đã có dữ liệu mới nhất
    if start_date.date() >= today.date():
        return {
            "status": "up_to_date",
            "symbol": symbol.upper(),
            "message": f"Data is already up to date (last: {last_date_str})",
            "last_date": last_date_str,
            "new_records": 0
        }
    
    try:
        # Fetch dữ liệu mới
        df = vndirect.get_stock_price(
            symbol=symbol.upper(),
            from_date=start_date.strftime('%Y-%m-%d'),
            to_date=today.strftime('%Y-%m-%d')
        )
        
        if df.empty:
            return {
                "status": "no_new_data",
                "symbol": symbol.upper(),
                "message": "No new data available from VNDirect",
                "last_date": last_date_str,
                "new_records": 0
            }
        
        # Lưu vào database
        new_records = 0
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
                new_records += 1
        
        db.commit()
        
        return {
            "status": "synced",
            "symbol": symbol.upper(),
            "message": f"Successfully synced {new_records} new records",
            "previous_last_date": last_date_str,
            "new_last_date": df['date'].max().strftime('%Y-%m-%d'),
            "new_records": new_records,
            "date_range": {
                "from": df['date'].min().strftime('%Y-%m-%d'),
                "to": df['date'].max().strftime('%Y-%m-%d')
            }
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error syncing {symbol}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/status", tags=["Data Collection"])
async def get_data_status(db: Session = Depends(get_db)):
    """
    📊 Kiểm tra trạng thái dữ liệu của tất cả cổ phiếu.
    
    Hiển thị:
    - Ngày đầu tiên và cuối cùng có dữ liệu
    - Số ngày dữ liệu
    - Có cần sync không
    """
    stocks = db.query(Stock).filter(Stock.is_active == True).all()
    today = datetime.now().date()
    
    results = []
    needs_sync_count = 0
    
    for stock in stocks:
        # Lấy ngày đầu và cuối
        first_price = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id
        ).order_by(StockPrice.date).first()
        
        last_price = db.query(StockPrice).filter(
            StockPrice.stock_id == stock.id
        ).order_by(desc(StockPrice.date)).first()
        
        # Đếm số records
        total_records = db.query(func.count(StockPrice.id)).filter(
            StockPrice.stock_id == stock.id
        ).scalar()
        
        # Kiểm tra cần sync không
        needs_sync = False
        days_behind = 0
        if last_price:
            days_behind = (today - last_price.date).days
            needs_sync = days_behind > 1  # Cần sync nếu thiếu > 1 ngày
        else:
            needs_sync = True
            days_behind = -1  # Không có dữ liệu
        
        if needs_sync:
            needs_sync_count += 1
        
        results.append({
            "symbol": stock.symbol,
            "first_date": first_price.date.isoformat() if first_price else None,
            "last_date": last_price.date.isoformat() if last_price else None,
            "total_records": total_records,
            "days_behind": days_behind,
            "needs_sync": needs_sync
        })
    
    # Sort by days_behind (những mã cần sync nhất lên trước)
    results.sort(key=lambda x: x["days_behind"], reverse=True)
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "today": today.isoformat(),
        "summary": {
            "total_stocks": len(stocks),
            "needs_sync": needs_sync_count,
            "up_to_date": len(stocks) - needs_sync_count
        },
        "stocks": results
    }


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
        "message": "Use Render Cron Jobs or UptimeRobot for scheduling",
        "keep_alive": {
            "service": "UptimeRobot",
            "url": "https://uptimerobot.com",
            "interval": "5-10 minutes"
        },
        "available_tasks": [
            {"name": "daily-data-fetch", "schedule": "0 7 * * 1-5", "description": "Fetch data 7AM UTC"},
            {"name": "weekly-full-update", "schedule": "0 0 * * 0", "description": "Full update Sunday"},
            {"name": "health-check", "schedule": "*/10 * * * *", "description": "Every 10 minutes"}
        ]
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


# =====================================================
# FINBERT SENTIMENT ENDPOINTS (Pre-computed offline)
# =====================================================

@app.get("/api/finbert/sentiment/{symbol}", tags=["FinBERT Sentiment"])
async def get_finbert_sentiment(
    symbol: str,
    days: int = Query(7, description="Số ngày gần đây"),
    db: Session = Depends(get_db)
):
    """
    🤖 Lấy kết quả phân tích FinBERT sentiment đã tính toán offline
    
    Dữ liệu này được tính toán trước bằng script offline và lưu vào DB.
    Chạy script: python scripts/analyze_news_finbert.py
    
    Returns:
    - Sentiment summary: Overall sentiment, avg score
    - Recent news: List of analyzed news with FinBERT scores
    """
    from sqlalchemy import text
    
    clean_symbol = symbol.upper().replace('.VN', '')
    
    try:
        # Check if analyzed_news table exists
        check_table = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'analyzed_news'
            )
        """)
        result = db.execute(check_table).scalar()
        
        if not result:
            return {
                "status": "not_available",
                "symbol": clean_symbol,
                "message": "FinBERT analysis not available. Run offline script first.",
                "hint": "python scripts/analyze_news_finbert.py --symbols " + clean_symbol
            }
        
        # Get sentiment summary
        summary_query = text("""
            SELECT * FROM sentiment_summary 
            WHERE symbol = :symbol 
            ORDER BY date DESC 
            LIMIT :days
        """)
        summary_result = db.execute(summary_query, {"symbol": clean_symbol, "days": days}).fetchall()
        
        # Get recent analyzed news
        news_query = text("""
            SELECT title, summary, url, source, published_at, 
                   sentiment, sentiment_score, positive_score, negative_score, neutral_score
            FROM analyzed_news 
            WHERE symbol = :symbol 
            ORDER BY published_at DESC 
            LIMIT 20
        """)
        news_result = db.execute(news_query, {"symbol": clean_symbol}).fetchall()
        
        if not summary_result and not news_result:
            return {
                "status": "no_data",
                "symbol": clean_symbol,
                "message": f"No FinBERT data for {clean_symbol}. Run analysis script.",
                "hint": "python scripts/analyze_news_finbert.py --symbols " + clean_symbol
            }
        
        # Calculate overall sentiment
        if summary_result:
            total_positive = sum(row[3] for row in summary_result)  # positive_count
            total_negative = sum(row[4] for row in summary_result)  # negative_count
            total_neutral = sum(row[5] for row in summary_result)   # neutral_count
            avg_score = sum(row[6] for row in summary_result) / len(summary_result)  # avg_score
            
            if avg_score > 0.3:
                overall = "positive"
                recommendation = "📈 Tâm lý thị trường tích cực, có thể xem xét mua"
            elif avg_score < -0.3:
                overall = "negative"
                recommendation = "📉 Tâm lý thị trường tiêu cực, nên thận trọng"
            else:
                overall = "neutral"
                recommendation = "⚖️ Tâm lý trung lập, theo dõi thêm diễn biến"
        else:
            total_positive = total_negative = total_neutral = 0
            avg_score = 0
            overall = "unknown"
            recommendation = "Chưa đủ dữ liệu"
        
        return {
            "status": "ok",
            "symbol": clean_symbol,
            "model": "ProsusAI/finbert",
            "analysis_type": "pre_computed_offline",
            "sentiment_summary": {
                "overall": overall,
                "avg_score": round(avg_score, 3),
                "positive_count": total_positive,
                "negative_count": total_negative,
                "neutral_count": total_neutral,
                "total_news": total_positive + total_negative + total_neutral,
                "recommendation": recommendation
            },
            "daily_summary": [
                {
                    "date": str(row[2]),  # date
                    "positive": row[3],
                    "negative": row[4],
                    "neutral": row[5],
                    "avg_score": round(row[6], 3),
                    "overall": row[7],
                    "news_count": row[8]
                }
                for row in summary_result
            ],
            "recent_news": [
                {
                    "title": row[0],
                    "summary": row[1][:200] + "..." if row[1] and len(row[1]) > 200 else row[1],
                    "url": row[2],
                    "source": row[3],
                    "published_at": str(row[4]),
                    "sentiment": row[5],
                    "sentiment_score": round(row[6], 3),
                    "scores": {
                        "positive": round(row[7], 3),
                        "negative": round(row[8], 3),
                        "neutral": round(row[9], 3)
                    }
                }
                for row in news_result
            ]
        }
        
    except Exception as e:
        logger.error(f"FinBERT query error: {e}")
        return {
            "status": "error",
            "symbol": clean_symbol,
            "message": str(e),
            "hint": "Ensure offline analysis has been run"
        }


@app.get("/api/finbert/summary", tags=["FinBERT Sentiment"])
async def get_finbert_market_summary(
    db: Session = Depends(get_db)
):
    """
    📊 Tổng hợp FinBERT sentiment cho toàn thị trường
    
    Lấy sentiment tổng hợp của tất cả các mã đã phân tích.
    """
    from sqlalchemy import text
    
    try:
        # Check if table exists
        check_table = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'sentiment_summary'
            )
        """)
        result = db.execute(check_table).scalar()
        
        if not result:
            return {
                "status": "not_available",
                "message": "FinBERT analysis tables not created yet",
                "hint": "Run: python scripts/analyze_news_finbert.py"
            }
        
        # Get latest summary for each symbol
        query = text("""
            SELECT DISTINCT ON (symbol) 
                symbol, date, positive_count, negative_count, neutral_count, 
                avg_score, overall_sentiment, news_count
            FROM sentiment_summary
            ORDER BY symbol, date DESC
        """)
        results = db.execute(query).fetchall()
        
        if not results:
            return {
                "status": "no_data",
                "message": "No FinBERT data available",
                "hint": "Run offline analysis script"
            }
        
        # Calculate market-wide metrics
        total_positive = sum(row[2] for row in results)
        total_negative = sum(row[3] for row in results)
        total_neutral = sum(row[4] for row in results)
        avg_market_score = sum(row[5] for row in results) / len(results)
        
        positive_stocks = sum(1 for row in results if row[6] == 'positive')
        negative_stocks = sum(1 for row in results if row[6] == 'negative')
        neutral_stocks = sum(1 for row in results if row[6] == 'neutral')
        
        return {
            "status": "ok",
            "model": "ProsusAI/finbert",
            "market_sentiment": {
                "avg_score": round(avg_market_score, 3),
                "total_news_analyzed": total_positive + total_negative + total_neutral,
                "positive_news": total_positive,
                "negative_news": total_negative,
                "neutral_news": total_neutral
            },
            "stock_breakdown": {
                "total_stocks": len(results),
                "positive_outlook": positive_stocks,
                "negative_outlook": negative_stocks,
                "neutral_outlook": neutral_stocks
            },
            "stocks": [
                {
                    "symbol": row[0],
                    "date": str(row[1]),
                    "sentiment": row[6],
                    "score": round(row[5], 3),
                    "news_count": row[7]
                }
                for row in sorted(results, key=lambda x: x[5], reverse=True)
            ]
        }
        
    except Exception as e:
        logger.error(f"FinBERT market summary error: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/api/finbert/status", tags=["FinBERT Sentiment"])
async def get_finbert_status(db: Session = Depends(get_db)):
    """
    📋 Kiểm tra trạng thái FinBERT analysis
    
    Xem đã phân tích bao nhiêu tin, ngày cuối cùng update.
    """
    from sqlalchemy import text
    
    try:
        # Check tables exist
        check_analyzed = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'analyzed_news'
            )
        """)
        check_summary = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'sentiment_summary'
            )
        """)
        
        analyzed_exists = db.execute(check_analyzed).scalar()
        summary_exists = db.execute(check_summary).scalar()
        
        if not analyzed_exists or not summary_exists:
            return {
                "status": "not_initialized",
                "tables_created": {
                    "analyzed_news": analyzed_exists,
                    "sentiment_summary": summary_exists
                },
                "message": "Run offline script to initialize: python scripts/analyze_news_finbert.py"
            }
        
        # Get stats
        news_count = db.execute(text("SELECT COUNT(*) FROM analyzed_news")).scalar()
        symbols_count = db.execute(text("SELECT COUNT(DISTINCT symbol) FROM analyzed_news")).scalar()
        
        latest_news = db.execute(text(
            "SELECT MAX(analyzed_at) FROM analyzed_news"
        )).scalar()
        
        latest_summary = db.execute(text(
            "SELECT MAX(updated_at) FROM sentiment_summary"
        )).scalar()
        
        # Get symbols analyzed
        symbols = db.execute(text(
            "SELECT DISTINCT symbol FROM analyzed_news ORDER BY symbol"
        )).fetchall()
        
        return {
            "status": "ready",
            "model": "ProsusAI/finbert",
            "statistics": {
                "total_news_analyzed": news_count,
                "symbols_covered": symbols_count,
                "symbols": [s[0] for s in symbols]
            },
            "last_update": {
                "news_analysis": str(latest_news) if latest_news else None,
                "summary_update": str(latest_summary) if latest_summary else None
            },
            "instructions": {
                "update_data": "python scripts/analyze_news_finbert.py",
                "specific_symbols": "python scripts/analyze_news_finbert.py --symbols VNM FPT",
                "more_days": "python scripts/analyze_news_finbert.py --days 30"
            }
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


# =====================================================
# SCHEDULER & INDICATORS ENDPOINTS
# =====================================================

@app.post("/api/scheduler/start", tags=["Scheduler"])
async def start_scheduler():
    """
    Khởi động background scheduler
    - Daily update: Mon-Fri at 18:00
    - Weekly catch-up: Sunday at 10:00
    """
    try:
        from src.scheduler.daily_scheduler import init_scheduler
        
        scheduler = init_scheduler()
        next_run = scheduler.get_next_run_time()
        
        return {
            "status": "success",
            "message": "Scheduler started successfully",
            "next_run": next_run,
            "jobs": [
                {
                    "name": "Daily Stock Price & Indicators Update",
                    "schedule": "Mon-Fri at 18:00"
                },
                {
                    "name": "Weekly Catch-up Update",
                    "schedule": "Sunday at 10:00"
                }
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")


@app.post("/api/scheduler/run-now", tags=["Scheduler"])
async def trigger_daily_update():
    """
    Chạy daily update job ngay lập tức (manual trigger)
    - Fetch stock prices
    - Calculate technical indicators
    """
    try:
        from src.scheduler.daily_scheduler import get_scheduler
        
        scheduler = get_scheduler()
        
        # Run job in background
        import threading
        thread = threading.Thread(target=scheduler.run_now)
        thread.start()
        
        return {
            "status": "success",
            "message": "Daily update job triggered. Running in background...",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/scheduler/status", tags=["Scheduler"])
async def get_scheduler_status():
    """Lấy trạng thái scheduler"""
    try:
        from src.scheduler.daily_scheduler import get_scheduler
        
        scheduler = get_scheduler()
        next_run = scheduler.get_next_run_time()
        
        return {
            "status": "running",
            "next_run": next_run,
            "jobs": [
                {
                    "id": "daily_price_update",
                    "name": "Daily Stock Price & Indicators Update",
                    "schedule": "Mon-Fri at 18:00"
                },
                {
                    "id": "weekly_catchup",
                    "name": "Weekly Catch-up Update",
                    "schedule": "Sunday at 10:00"
                }
            ]
        }
    except Exception as e:
        return {
            "status": "stopped",
            "message": "Scheduler not started yet. Use POST /api/scheduler/start to start.",
            "error": str(e)
        }


@app.post("/api/indicators/calculate", tags=["Indicators"])
async def calculate_indicators_for_all(
    days: int = Query(default=365, description="Số ngày dữ liệu để tính"),
    db: Session = Depends(get_db)
):
    """
    Tính toán technical indicators cho tất cả stocks
    
    Indicators:
    - Moving Averages: SMA20, SMA50, SMA200, EMA12, EMA26
    - RSI (14)
    - MACD (12,26,9)
    - Bollinger Bands (20,2)
    - Stochastic Oscillator
    - ATR (14)
    - OBV
    - ADX
    - CCI
    - Williams %R
    """
    try:
        from src.features.indicators_processor import IndicatorsProcessor
        
        processor = IndicatorsProcessor(db)
        result = processor.process_all_stocks(days=days)
        
        return {
            "status": "success",
            "message": "Indicators calculated successfully",
            "result": result,
            "indicators": [
                "SMA (20, 50, 200)",
                "EMA (12, 26)",
                "RSI (14)",
                "MACD (12, 26, 9)",
                "Bollinger Bands (20, 2)",
                "Stochastic Oscillator",
                "ATR (14)",
                "OBV",
                "ADX",
                "CCI",
                "Williams %R"
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/indicators/calculate/{symbol}", tags=["Indicators"])
async def calculate_indicators_for_symbol(
    symbol: str,
    days: int = Query(default=365, description="Số ngày dữ liệu"),
    db: Session = Depends(get_db)
):
    """Tính toán technical indicators cho một symbol cụ thể"""
    try:
        from src.features.indicators_processor import IndicatorsProcessor
        
        # Find stock
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        processor = IndicatorsProcessor(db)
        success = processor.process_stock(stock.id, days=days)
        
        if success:
            return {
                "status": "success",
                "symbol": symbol.upper(),
                "message": f"Indicators calculated for {symbol}"
            }
        else:
            return {
                "status": "failed",
                "symbol": symbol.upper(),
                "message": f"Failed to calculate indicators for {symbol}"
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/indicators/{symbol}", tags=["Indicators"])
async def get_indicators_for_symbol(
    symbol: str,
    limit: int = Query(default=30, description="Số ngày gần nhất"),
    db: Session = Depends(get_db)
):
    """Lấy technical indicators đã tính cho một symbol"""
    try:
        # Find stock
        stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
        if not stock:
            raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
        
        # Get indicators
        indicators = db.query(TechnicalIndicator).filter(
            TechnicalIndicator.stock_id == stock.id
        ).order_by(desc(TechnicalIndicator.date)).limit(limit).all()
        
        if not indicators:
            return {
                "symbol": symbol.upper(),
                "indicators": [],
                "message": "No indicators found. Use POST /api/indicators/calculate to calculate."
            }
        
        data = []
        for ind in indicators:
            data.append({
                "date": ind.date.isoformat(),
                "sma_20": ind.sma_20,
                "sma_50": ind.sma_50,
                "sma_200": ind.sma_200,
                "ema_12": ind.ema_12,
                "ema_26": ind.ema_26,
                "rsi_14": ind.rsi_14,
                "macd": ind.macd,
                "macd_signal": ind.macd_signal,
                "macd_histogram": ind.macd_histogram,
                "bb_upper": ind.bb_upper,
                "bb_middle": ind.bb_middle,
                "bb_lower": ind.bb_lower,
                "stoch_k": ind.stoch_k,
                "stoch_d": ind.stoch_d,
                "atr_14": ind.atr_14,
                "obv": ind.obv,
                "adx": ind.adx,
                "plus_di": ind.plus_di,
                "minus_di": ind.minus_di,
                "cci": ind.cci,
                "williams_r": ind.williams_r
            })
        
        return {
            "symbol": symbol.upper(),
            "records": len(data),
            "indicators": data
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    print("Starting KLTN Stock Prediction API v2.0...")
    print("📊 Documentation: http://localhost:8000/docs")
    print("📚 ReDoc: http://localhost:8000/redoc")
    print("🔗 Database: SQLite/PostgreSQL")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
