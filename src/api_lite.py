"""
Lightweight FastAPI - Data Only (No ML)
For fast startup and low memory usage
"""

from fastapi import FastAPI, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
from datetime import datetime, timedelta
from typing import List, Optional
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from src.database.connection import get_db
from src.database.models import Stock, StockPrice, Prediction

app = FastAPI(
    title="KLTN Stock API - Lite",
    version="2.0.0-lite",
    description="Lightweight API for stock data (no ML)"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# HEALTH CHECK
# ============================================================================

@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "version": "2.0.0-lite", "timestamp": datetime.utcnow().isoformat()}

@app.get("/")
async def root():
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return {"message": "KLTN Stock API Lite", "docs": "/docs"}

# ============================================================================
# STOCK DATA ENDPOINTS
# ============================================================================

@app.get("/api/stocks")
async def get_stocks(db: Session = Depends(get_db)):
    stocks = db.query(Stock).order_by(Stock.symbol).all()
    return [{"symbol": s.symbol, "name": s.name, "exchange": s.exchange} for s in stocks]

@app.get("/api/prices/{symbol}")
async def get_prices(
    symbol: str,
    limit: int = Query(60, ge=1, le=500),
    db: Session = Depends(get_db)
):
    prices = db.query(StockPrice).filter(
        StockPrice.symbol == symbol.upper()
    ).order_by(StockPrice.date.desc()).limit(limit).all()
    
    return [{
        "date": p.date.isoformat() if p.date else None,
        "open": p.open,
        "high": p.high,
        "low": p.low,
        "close": p.close,
        "volume": p.volume
    } for p in reversed(prices)]

@app.get("/api/stats")
async def get_stats(db: Session = Depends(get_db)):
    total_stocks = db.query(func.count(Stock.id)).scalar() or 0
    total_records = db.query(func.count(StockPrice.id)).scalar() or 0
    
    latest = db.query(func.max(StockPrice.date)).scalar()
    
    return {
        "total_stocks": total_stocks,
        "total_records": total_records,
        "latest_date": latest.isoformat() if latest else None
    }

# ============================================================================
# STATIC FILES
# ============================================================================

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
