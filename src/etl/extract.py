"""
ETL Module 1: Extract - Thu thập dữ liệu từ Database
Đọc dữ liệu giá và tin tức từ PostgreSQL Database

QUAN TRỌNG:
- Đọc từ raw_stock_data table (dữ liệu thô từ API)
- Hoặc đọc từ stock_prices table (dữ liệu đã xử lý)
"""

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import desc, text
from datetime import datetime, timedelta
from typing import Optional
import logging

from src.database.connection import get_db
from src.database.models import Stock, StockPrice, NewsArticle, AnalyzedNews
from src.etl.config_loader import get_config

logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract data from PostgreSQL Database"""
    
    def __init__(self, db: Session = None):
        self.db = db or next(get_db())
        self.config = get_config()
    
    # =====================================================
    # EXTRACT FROM RAW DATA TABLE (Dữ liệu thô từ API)
    # =====================================================
    
    def extract_raw_price_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        unprocessed_only: bool = True
    ) -> pd.DataFrame:
        """
        Extract RAW price data from raw_stock_data table
        
        Đây là dữ liệu thô từ API, chưa qua transform
        
        Args:
            symbol: Stock symbol filter
            start_date: Start date filter
            end_date: End date filter
            unprocessed_only: Only get unprocessed data
        
        Returns:
            DataFrame with raw price data
        """
        try:
            logger.info(f"📥 Extracting RAW data (symbol={symbol}, unprocessed_only={unprocessed_only})")
            
            # Build query
            conditions = []
            params = {}
            
            if symbol:
                conditions.append("symbol = :symbol")
                params['symbol'] = symbol.upper()
            
            if start_date:
                conditions.append("date >= :start_date")
                params['start_date'] = start_date.date()
            
            if end_date:
                conditions.append("date <= :end_date")
                params['end_date'] = end_date.date()
            
            if unprocessed_only:
                conditions.append("processed = FALSE")
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = text(f"""
                SELECT 
                    id, symbol, date, 
                    open, high, low, close, volume,
                    value, change_percent, change_point,
                    source, collected_at
                FROM raw_stock_data
                {where_clause}
                ORDER BY symbol, date
            """)
            
            result = self.db.execute(query, params)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if df.empty:
                logger.warning("⚠️  No raw data found")
                return pd.DataFrame()
            
            logger.info(f"✅ Extracted {len(df)} raw records from raw_stock_data table")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error extracting raw data: {str(e)}")
            return pd.DataFrame()
    
    # =====================================================
    # EXTRACT FROM PROCESSED TABLE (Dữ liệu đã xử lý)
    # =====================================================
    
    def extract_price_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract stock price data
        
        Args:
            symbol: Stock symbol (None = all stocks)
            start_date: Start date filter
            end_date: End date filter
            limit: Max number of records
        
        Returns:
            DataFrame with columns: [id, stock_id, symbol, date, open, high, low, close, volume]
        """
        try:
            logger.info(f"📥 Extracting price data (symbol={symbol})")
            
            # Build query
            query = self.db.query(StockPrice, Stock.symbol).join(
                Stock, StockPrice.stock_id == Stock.id
            )
            
            # Apply filters
            if symbol:
                query = query.filter(Stock.symbol == symbol.upper())
            
            if start_date:
                query = query.filter(StockPrice.date >= start_date.date())
            
            if end_date:
                query = query.filter(StockPrice.date <= end_date.date())
            
            # Order and limit
            query = query.order_by(Stock.symbol, StockPrice.date)
            
            if limit:
                query = query.limit(limit)
            
            # Execute query
            results = query.all()
            
            if not results:
                logger.warning("⚠️  No price data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for price, symbol_name in results:
                data.append({
                    'id': price.id,
                    'stock_id': price.stock_id,
                    'symbol': symbol_name,
                    'date': price.date,
                    'open': price.open,
                    'high': price.high,
                    'low': price.low,
                    'close': price.close,
                    'volume': price.volume
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Extracted {len(df)} price records")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error extracting price data: {str(e)}")
            return pd.DataFrame()
    
    def extract_news_data(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        unprocessed_only: bool = False,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Extract news article data
        
        Args:
            symbol: Stock symbol filter
            start_date: Start date filter
            unprocessed_only: Only get news without sentiment analysis
            limit: Max number of records
        
        Returns:
            DataFrame with columns: [id, symbol, title, content, summary, url, source, published_at]
        """
        try:
            logger.info(f"📥 Extracting news data (symbol={symbol}, unprocessed_only={unprocessed_only})")
            
            # Build query
            query = self.db.query(NewsArticle)
            
            # Apply filters
            if symbol:
                query = query.filter(NewsArticle.symbol == symbol.upper())
            
            if start_date:
                query = query.filter(NewsArticle.published_at >= start_date)
            
            if unprocessed_only:
                # Get news that don't have sentiment analysis yet
                analyzed_ids = self.db.query(AnalyzedNews.news_id).distinct()
                query = query.filter(~NewsArticle.id.in_(analyzed_ids))
            
            # Order and limit
            query = query.order_by(desc(NewsArticle.published_at))
            
            if limit:
                query = query.limit(limit)
            
            # Execute query
            results = query.all()
            
            if not results:
                logger.warning("⚠️  No news data found")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for news in results:
                data.append({
                    'id': news.id,
                    'symbol': news.symbol,
                    'title': news.title,
                    'content': news.content,
                    'summary': news.summary,
                    'url': news.url,
                    'source': news.source,
                    'published_at': news.published_at,
                    'created_at': news.created_at
                })
            
            df = pd.DataFrame(data)
            logger.info(f"✅ Extracted {len(df)} news records")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error extracting news data: {str(e)}")
            return pd.DataFrame()
    
    def extract_latest_prices(self, days: int = 30) -> pd.DataFrame:
        """
        Extract latest price data for all stocks
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with latest price data
        """
        start_date = datetime.now() - timedelta(days=days)
        return self.extract_price_data(start_date=start_date)
    
    def extract_latest_news(self, days: int = 7) -> pd.DataFrame:
        """
        Extract latest news for all stocks
        
        Args:
            days: Number of days to look back
        
        Returns:
            DataFrame with latest news
        """
        start_date = datetime.now() - timedelta(days=days)
        return self.extract_news_data(start_date=start_date)
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    # Test extractor
    print("Testing Data Extractor...")
    print("-" * 80)
    
    extractor = DataExtractor()
    
    # Test 1: Extract price data for VNM
    print("\n1. Extract VNM price data (last 30 days)")
    df_price = extractor.extract_price_data(
        symbol="VNM",
        start_date=datetime.now() - timedelta(days=30)
    )
    print(f"   Result: {len(df_price)} records")
    if not df_price.empty:
        print(df_price.head())
    
    # Test 2: Extract latest news
    print("\n2. Extract latest news (last 7 days)")
    df_news = extractor.extract_latest_news(days=7)
    print(f"   Result: {len(df_news)} records")
    if not df_news.empty:
        print(df_news[['symbol', 'title', 'published_at']].head())
    
    # Test 3: Extract unprocessed news
    print("\n3. Extract unprocessed news")
    df_unprocessed = extractor.extract_news_data(unprocessed_only=True, limit=10)
    print(f"   Result: {len(df_unprocessed)} records")
    
    extractor.close()
    print("\n✅ Extractor test completed")
