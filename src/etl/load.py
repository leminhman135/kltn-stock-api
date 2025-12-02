"""
ETL Module 3: Load - Lưu dữ liệu vào Database
Load cleaned data into PostgreSQL
"""

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import and_
from datetime import datetime
from typing import List, Dict, Optional
import logging

from src.database.connection import get_db
from src.database.models import Stock, StockPrice, NewsArticle
from src.etl.config_loader import get_config

logger = logging.getLogger(__name__)


class DataLoader:
    """Load data into PostgreSQL Database"""
    
    def __init__(self, db: Session = None):
        self.db = db or next(get_db())
        self.config = get_config()
        self.batch_size = self.config.get('etl.batch_size', 1000)
    
    # =====================================================
    # LOAD PRICE DATA
    # =====================================================
    
    def load_price_data(self, df: pd.DataFrame) -> Dict:
        """
        Load price data into database
        
        Strategy: Upsert (Insert new, Update existing)
        
        Args:
            df: Cleaned price DataFrame
        
        Returns:
            dict: Statistics (inserted, updated, failed)
        """
        if df.empty:
            logger.warning("⚠️  No price data to load")
            return {'inserted': 0, 'updated': 0, 'failed': 0}
        
        logger.info(f"💾 Loading {len(df)} price records to database")
        
        stats = {
            'inserted': 0,
            'updated': 0,
            'failed': 0
        }
        
        try:
            for idx, row in df.iterrows():
                try:
                    # Check if record exists
                    existing = self.db.query(StockPrice).filter(
                        and_(
                            StockPrice.stock_id == row['stock_id'],
                            StockPrice.date == row['date']
                        )
                    ).first()
                    
                    if existing:
                        # Update existing record
                        existing.open = float(row['open'])
                        existing.high = float(row['high'])
                        existing.low = float(row['low'])
                        existing.close = float(row['close'])
                        existing.volume = int(row['volume'])
                        stats['updated'] += 1
                    else:
                        # Insert new record
                        price_record = StockPrice(
                            stock_id=row['stock_id'],
                            date=row['date'],
                            open=float(row['open']),
                            high=float(row['high']),
                            low=float(row['low']),
                            close=float(row['close']),
                            volume=int(row['volume'])
                        )
                        self.db.add(price_record)
                        stats['inserted'] += 1
                    
                    # Commit in batches
                    if (stats['inserted'] + stats['updated']) % self.batch_size == 0:
                        self.db.commit()
                        logger.info(f"  → Committed batch: {stats['inserted']} inserted, {stats['updated']} updated")
                
                except Exception as e:
                    logger.error(f"Error loading record {idx}: {str(e)}")
                    stats['failed'] += 1
                    continue
            
            # Final commit
            self.db.commit()
            
            logger.info(
                f"✅ Load complete: {stats['inserted']} inserted, "
                f"{stats['updated']} updated, {stats['failed']} failed"
            )
            
            return stats
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error loading price data: {str(e)}")
            stats['failed'] = len(df)
            return stats
    
    # =====================================================
    # LOAD NEWS DATA
    # =====================================================
    
    def load_news_data(self, df: pd.DataFrame) -> Dict:
        """
        Load news data into database
        
        Strategy: Update clean_content for existing records
        
        Args:
            df: Cleaned news DataFrame
        
        Returns:
            dict: Statistics (updated, failed)
        """
        if df.empty:
            logger.warning("⚠️  No news data to load")
            return {'updated': 0, 'failed': 0}
        
        logger.info(f"💾 Loading {len(df)} news records to database")
        
        stats = {
            'updated': 0,
            'failed': 0
        }
        
        try:
            for idx, row in df.iterrows():
                try:
                    # Find existing record
                    existing = self.db.query(NewsArticle).filter(
                        NewsArticle.id == row['id']
                    ).first()
                    
                    if existing:
                        # Update clean content
                        existing.content = row.get('clean_content', existing.content)
                        stats['updated'] += 1
                    else:
                        # Skip if not found (should already exist from data collection)
                        logger.warning(f"News record {row['id']} not found, skipping")
                        stats['failed'] += 1
                        continue
                    
                    # Commit in batches
                    if stats['updated'] % self.batch_size == 0:
                        self.db.commit()
                        logger.info(f"  → Committed batch: {stats['updated']} updated")
                
                except Exception as e:
                    logger.error(f"Error loading news {idx}: {str(e)}")
                    stats['failed'] += 1
                    continue
            
            # Final commit
            self.db.commit()
            
            logger.info(
                f"✅ Load complete: {stats['updated']} updated, "
                f"{stats['failed']} failed"
            )
            
            return stats
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error loading news data: {str(e)}")
            stats['failed'] = len(df)
            return stats
    
    # =====================================================
    # BULK INSERT (FASTER FOR LARGE DATA)
    # =====================================================
    
    def bulk_insert_prices(self, df: pd.DataFrame) -> int:
        """
        Bulk insert price data (faster but no upsert)
        
        WARNING: This will fail if duplicate records exist
        Use for initial data load only
        
        Args:
            df: Cleaned price DataFrame
        
        Returns:
            Number of records inserted
        """
        if df.empty:
            return 0
        
        logger.info(f"💾 Bulk inserting {len(df)} price records")
        
        try:
            # Convert DataFrame to list of dicts
            records = df.to_dict('records')
            
            # Prepare objects
            objects = []
            for record in records:
                objects.append(StockPrice(
                    stock_id=record['stock_id'],
                    date=record['date'],
                    open=float(record['open']),
                    high=float(record['high']),
                    low=float(record['low']),
                    close=float(record['close']),
                    volume=int(record['volume'])
                ))
            
            # Bulk insert
            self.db.bulk_save_objects(objects)
            self.db.commit()
            
            logger.info(f"✅ Bulk insert complete: {len(objects)} records")
            return len(objects)
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Bulk insert failed: {str(e)}")
            return 0
    
    # =====================================================
    # HELPER METHODS
    # =====================================================
    
    def get_stock_id(self, symbol: str) -> Optional[int]:
        """
        Get stock_id from symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            stock_id or None if not found
        """
        try:
            stock = self.db.query(Stock).filter(
                Stock.symbol == symbol.upper()
            ).first()
            
            return stock.id if stock else None
        
        except Exception as e:
            logger.error(f"Error getting stock_id for {symbol}: {str(e)}")
            return None
    
    def check_duplicate_prices(self, symbol: str, date: datetime) -> bool:
        """
        Check if price record exists
        
        Args:
            symbol: Stock symbol
            date: Price date
        
        Returns:
            True if exists, False otherwise
        """
        try:
            stock_id = self.get_stock_id(symbol)
            if not stock_id:
                return False
            
            existing = self.db.query(StockPrice).filter(
                and_(
                    StockPrice.stock_id == stock_id,
                    StockPrice.date == date
                )
            ).first()
            
            return existing is not None
        
        except Exception as e:
            logger.error(f"Error checking duplicate: {str(e)}")
            return False
    
    def get_latest_price_date(self, symbol: str) -> Optional[datetime]:
        """
        Get latest price date for a symbol
        
        Args:
            symbol: Stock symbol
        
        Returns:
            Latest date or None
        """
        try:
            stock_id = self.get_stock_id(symbol)
            if not stock_id:
                return None
            
            latest = self.db.query(StockPrice.date).filter(
                StockPrice.stock_id == stock_id
            ).order_by(StockPrice.date.desc()).first()
            
            return latest[0] if latest else None
        
        except Exception as e:
            logger.error(f"Error getting latest date: {str(e)}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    # Test loader
    print("Testing Data Loader...")
    print("-" * 80)
    
    loader = DataLoader()
    
    # Test 1: Check stock_id
    print("\n1. Get stock_id for VNM")
    stock_id = loader.get_stock_id('VNM')
    print(f"   VNM stock_id: {stock_id}")
    
    # Test 2: Check latest price date
    print("\n2. Get latest price date for VNM")
    latest_date = loader.get_latest_price_date('VNM')
    print(f"   Latest date: {latest_date}")
    
    # Test 3: Check duplicate
    print("\n3. Check if price exists")
    if latest_date:
        exists = loader.check_duplicate_prices('VNM', latest_date)
        print(f"   Duplicate exists: {exists}")
    
    # Test 4: Load test data (commented out to avoid actual insert)
    print("\n4. Test load (dry run)")
    test_df = pd.DataFrame({
        'stock_id': [stock_id],
        'symbol': ['VNM'],
        'date': [datetime.now().date()],
        'open': [80000.0],
        'high': [81000.0],
        'low': [79000.0],
        'close': [80500.0],
        'volume': [1000000]
    })
    print(f"   Prepared {len(test_df)} test records")
    print("   (Skipping actual insert)")
    
    # stats = loader.load_price_data(test_df)
    # print(f"   Stats: {stats}")
    
    loader.close()
    print("\n✅ Loader test completed")
