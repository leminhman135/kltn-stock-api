"""Database helper functions for data collection."""

from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import and_
import logging

from src.database.connection import SessionLocal
from src.database.models import Stock, StockPrice, TechnicalIndicator, NewsArticle

logger = logging.getLogger(__name__)


def save_stock_prices_to_db(
    symbol: str,
    df: pd.DataFrame,
    source: str = "vndirect"
) -> int:
    """
    Save stock prices to database.
    
    Args:
        symbol: Stock symbol (e.g., 'VNM', 'VIC')
        df: DataFrame with columns: date, open, high, low, close, volume
        source: Data source name
    
    Returns:
        Number of records saved
    """
    if df.empty:
        return 0
    
    db = SessionLocal()
    saved_count = 0
    
    try:
        # Get or create stock
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            stock = Stock(
                symbol=symbol,
                name=symbol,  # Can be updated later with real name
                exchange="HOSE",
                is_active=True
            )
            db.add(stock)
            db.commit()
            db.refresh(stock)
            logger.info(f"Created new stock entry: {symbol}")
        
        # Prepare price data
        for _, row in df.iterrows():
            try:
                # Check if record exists
                existing = db.query(StockPrice).filter(
                    and_(
                        StockPrice.stock_id == stock.id,
                        StockPrice.date == row['date']
                    )
                ).first()
                
                if existing:
                    # Update existing record
                    existing.open = float(row['open'])
                    existing.high = float(row['high'])
                    existing.low = float(row['low'])
                    existing.close = float(row['close'])
                    existing.volume = float(row['volume'])
                    existing.source = source
                else:
                    # Create new record
                    price = StockPrice(
                        stock_id=stock.id,
                        date=row['date'],
                        open=float(row['open']),
                        high=float(row['high']),
                        low=float(row['low']),
                        close=float(row['close']),
                        volume=float(row['volume']),
                        source=source
                    )
                    db.add(price)
                    saved_count += 1
            
            except Exception as e:
                logger.error(f"Error saving row for {symbol} on {row['date']}: {e}")
                continue
        
        db.commit()
        logger.info(f"✅ Saved {saved_count} new price records for {symbol} to database")
        return saved_count
    
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error saving {symbol} to database: {e}")
        return 0
    finally:
        db.close()


def save_technical_indicators_to_db(
    symbol: str,
    df: pd.DataFrame
) -> int:
    """
    Save technical indicators to database.
    
    Args:
        symbol: Stock symbol
        df: DataFrame with indicators columns
    
    Returns:
        Number of records saved
    """
    if df.empty:
        return 0
    
    db = SessionLocal()
    saved_count = 0
    
    try:
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            logger.error(f"Stock {symbol} not found in database")
            return 0
        
        indicator_columns = {
            'SMA_20': 'sma_20',
            'SMA_50': 'sma_50',
            'SMA_200': 'sma_200',
            'EMA_12': 'ema_12',
            'EMA_26': 'ema_26',
            'RSI': 'rsi_14',
            'MACD': 'macd',
            'MACD_Signal': 'macd_signal',
            'MACD_Histogram': 'macd_histogram',
            'BB_Upper': 'bb_upper',
            'BB_Middle': 'bb_middle',
            'BB_Lower': 'bb_lower',
            'ATR': 'atr_14',
            'OBV': 'obv',
            'ADX': 'adx_14'
        }
        
        for _, row in df.iterrows():
            try:
                # Check if record exists
                existing = db.query(TechnicalIndicator).filter(
                    and_(
                        TechnicalIndicator.stock_id == stock.id,
                        TechnicalIndicator.date == row['date']
                    )
                ).first()
                
                # Prepare indicator data
                indicator_data = {
                    'stock_id': stock.id,
                    'date': row['date']
                }
                
                for df_col, db_col in indicator_columns.items():
                    if df_col in row and pd.notna(row[df_col]):
                        indicator_data[db_col] = float(row[df_col])
                
                if existing:
                    # Update existing
                    for key, value in indicator_data.items():
                        if key not in ['stock_id', 'date']:
                            setattr(existing, key, value)
                else:
                    # Create new
                    indicator = TechnicalIndicator(**indicator_data)
                    db.add(indicator)
                    saved_count += 1
            
            except Exception as e:
                logger.error(f"Error saving indicators for {symbol} on {row['date']}: {e}")
                continue
        
        db.commit()
        logger.info(f"✅ Saved {saved_count} indicator records for {symbol} to database")
        return saved_count
    
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error saving indicators for {symbol}: {e}")
        return 0
    finally:
        db.close()


def save_news_to_db(
    news_df: pd.DataFrame
) -> int:
    """
    Save news articles to database.
    
    Args:
        news_df: DataFrame with columns: symbol, source, title, summary, link, date_str
    
    Returns:
        Number of records saved
    """
    if news_df.empty:
        return 0
    
    db = SessionLocal()
    saved_count = 0
    
    try:
        for _, row in news_df.iterrows():
            try:
                # Check if article exists
                existing = db.query(NewsArticle).filter(
                    NewsArticle.url == row['link']
                ).first()
                
                if not existing:
                    # Parse date
                    try:
                        published_date = datetime.strptime(row['date_str'], '%Y-%m-%d %H:%M:%S')
                    except:
                        published_date = datetime.now()
                    
                    article = NewsArticle(
                        stock_symbol=row['symbol'],
                        title=row['title'],
                        summary=row.get('summary', ''),
                        url=row['link'],
                        source=row['source'],
                        published_date=published_date
                    )
                    db.add(article)
                    saved_count += 1
            
            except Exception as e:
                logger.error(f"Error saving news article: {e}")
                continue
        
        db.commit()
        logger.info(f"✅ Saved {saved_count} news articles to database")
        return saved_count
    
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error saving news to database: {e}")
        return 0
    finally:
        db.close()


def get_stock_prices_from_db(
    symbol: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = 1000
) -> pd.DataFrame:
    """
    Retrieve stock prices from database.
    
    Args:
        symbol: Stock symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        limit: Maximum records to return
    
    Returns:
        DataFrame with price data
    """
    db = SessionLocal()
    
    try:
        stock = db.query(Stock).filter(Stock.symbol == symbol).first()
        if not stock:
            logger.warning(f"Stock {symbol} not found in database")
            return pd.DataFrame()
        
        query = db.query(StockPrice).filter(StockPrice.stock_id == stock.id)
        
        if start_date:
            query = query.filter(StockPrice.date >= start_date)
        if end_date:
            query = query.filter(StockPrice.date <= end_date)
        
        query = query.order_by(StockPrice.date.desc()).limit(limit)
        
        prices = query.all()
        
        if not prices:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for price in prices:
            data.append({
                'date': price.date,
                'open': price.open,
                'high': price.high,
                'low': price.low,
                'close': price.close,
                'volume': price.volume,
                'source': price.source
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('date').reset_index(drop=True)
        
        logger.info(f"Retrieved {len(df)} price records for {symbol} from database")
        return df
    
    except Exception as e:
        logger.error(f"Error retrieving prices for {symbol}: {e}")
        return pd.DataFrame()
    finally:
        db.close()
