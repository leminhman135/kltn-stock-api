"""
Raw Data Collector - Thu thập và lưu trữ dữ liệu thô vào SQL
Fetch data từ API (VNDirect, SSI) và lưu vào raw_stock_data table
"""

import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text, Table, Column, Integer, String, Float, Date, DateTime, MetaData, create_engine
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging
import json

from src.database.connection import get_db, engine
from src.database.models import Stock
from src.data_collection import VNDirectAPI
from src.etl.config_loader import get_config

logger = logging.getLogger(__name__)


class RawDataCollector:
    """
    Thu thập dữ liệu thô từ API và lưu vào SQL
    
    Workflow:
    1. Fetch data từ VNDirect/SSI API
    2. Lưu raw data vào raw_stock_data table
    3. ETL pipeline sẽ đọc từ raw_stock_data để transform
    """
    
    def __init__(self, db: Session = None):
        self.db = db or next(get_db())
        self.config = get_config()
        self.api = VNDirectAPI()
        
        # Ensure raw data table exists
        self._create_raw_data_table()
    
    def _create_raw_data_table(self):
        """
        Tạo table raw_stock_data nếu chưa tồn tại
        
        Table này lưu dữ liệu thô từ API trước khi transform
        """
        try:
            # SQLite compatible syntax
            create_table_sql = text("""
                CREATE TABLE IF NOT EXISTS raw_stock_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    date DATE NOT NULL,
                    
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    
                    value REAL,
                    change_percent REAL,
                    change_point REAL,
                    
                    source TEXT,
                    raw_json TEXT,
                    collected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed INTEGER DEFAULT 0,
                    
                    UNIQUE(symbol, date, source)
                )
            """)
            
            index_sqls = [
                text("CREATE INDEX IF NOT EXISTS idx_raw_stock_symbol ON raw_stock_data(symbol)"),
                text("CREATE INDEX IF NOT EXISTS idx_raw_stock_date ON raw_stock_data(date)"),
                text("CREATE INDEX IF NOT EXISTS idx_raw_stock_processed ON raw_stock_data(processed)")
            ]
            
            self.db.execute(create_table_sql)
            for index_sql in index_sqls:
                self.db.execute(index_sql)
            self.db.commit()
            logger.info("✅ Raw data table ready")
            
        except Exception as e:
            logger.error(f"❌ Error creating raw data table: {str(e)}")
            self.db.rollback()
    
    # =====================================================
    # COLLECT RAW DATA FROM API
    # =====================================================
    
    def collect_price_data(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Thu thập dữ liệu giá thô từ API và lưu vào raw_stock_data
        
        Args:
            symbol: Mã cổ phiếu
            from_date: Ngày bắt đầu (mặc định: 30 ngày trước)
            to_date: Ngày kết thúc (mặc định: hôm nay)
        
        Returns:
            dict: Statistics (collected, inserted, updated, failed)
        """
        try:
            logger.info(f"📡 Collecting raw data for {symbol} from API...")
            
            # Default date range
            if not to_date:
                to_date = datetime.now()
            if not from_date:
                from_date = to_date - timedelta(days=30)
            
            # Format dates for API
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            # Fetch from VNDirect API
            df = self.api.get_stock_price(
                symbol=symbol.upper(),
                from_date=from_str,
                to_date=to_str
            )
            
            if df.empty:
                logger.warning(f"⚠️  No data returned from API for {symbol}")
                return {
                    'status': 'no_data',
                    'collected': 0,
                    'inserted': 0,
                    'updated': 0,
                    'failed': 0
                }
            
            logger.info(f"✅ API returned {len(df)} records")
            
            # Save to raw_stock_data table
            stats = self._save_raw_dataframe(symbol, df, source='vndirect')
            
            return stats
        
        except Exception as e:
            logger.error(f"❌ Error collecting data for {symbol}: {str(e)}")
            return {
                'status': 'error',
                'error': str(e),
                'collected': 0,
                'inserted': 0,
                'updated': 0,
                'failed': 0
            }
    
    def _save_raw_dataframe(
        self,
        symbol: str,
        df: pd.DataFrame,
        source: str = 'vndirect'
    ) -> Dict:
        """
        Lưu DataFrame vào raw_stock_data table
        
        Args:
            symbol: Stock symbol
            df: DataFrame from API (columns: date, Open, High, Low, Close, Volume)
            source: Data source name
        
        Returns:
            dict: Statistics
        """
        stats = {
            'status': 'success',
            'collected': len(df),
            'inserted': 0,
            'updated': 0,
            'failed': 0
        }
        
        try:
            for _, row in df.iterrows():
                try:
                    # Parse date
                    if isinstance(row['date'], str):
                        date_obj = datetime.strptime(row['date'][:10], '%Y-%m-%d').date()
                    else:
                        date_obj = row['date'].date() if hasattr(row['date'], 'date') else row['date']
                    
                    # Parse price fields from DataFrame
                    open_price = float(row.get('Open', 0) or 0)
                    high_price = float(row.get('High', 0) or 0)
                    low_price = float(row.get('Low', 0) or 0)
                    close_price = float(row.get('Close', 0) or 0)
                    volume = int(row.get('Volume', 0) or 0)
                    
                    # Optional fields (may not exist in DataFrame)
                    value = float(row.get('value', 0) or 0)
                    change_percent = float(row.get('changePercent', 0) or 0)
                    change_point = float(row.get('change', 0) or 0)
                    
                    # Convert row to JSON string for audit
                    raw_json = row.to_json(force_ascii=False)
                    
                    # Check if record exists
                    check_query = text("""
                        SELECT id FROM raw_stock_data
                        WHERE symbol = :symbol AND date = :date AND source = :source
                    """)
                    existing = self.db.execute(
                        check_query,
                        {
                            'symbol': symbol.upper(),
                            'date': date_obj,
                            'source': source
                        }
                    ).fetchone()
                    
                    if existing:
                        # Update existing record
                        update_query = text("""
                            UPDATE raw_stock_data SET
                                open = :open,
                                high = :high,
                                low = :low,
                                close = :close,
                                volume = :volume,
                                value = :value,
                                change_percent = :change_percent,
                                change_point = :change_point,
                                raw_json = :raw_json,
                                collected_at = CURRENT_TIMESTAMP,
                                processed = 0
                            WHERE symbol = :symbol AND date = :date AND source = :source
                        """)
                        
                        self.db.execute(update_query, {
                            'symbol': symbol.upper(),
                            'date': date_obj,
                            'source': source,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'value': value,
                            'change_percent': change_percent,
                            'change_point': change_point,
                            'raw_json': raw_json
                        })
                        
                        stats['updated'] += 1
                    else:
                        # Insert new record
                        insert_query = text("""
                            INSERT INTO raw_stock_data (
                                symbol, date, open, high, low, close, volume,
                                value, change_percent, change_point, source, raw_json
                            ) VALUES (
                                :symbol, :date, :open, :high, :low, :close, :volume,
                                :value, :change_percent, :change_point, :source, :raw_json
                            )
                        """)
                        
                        self.db.execute(insert_query, {
                            'symbol': symbol.upper(),
                            'date': date_obj,
                            'open': open_price,
                            'high': high_price,
                            'low': low_price,
                            'close': close_price,
                            'volume': volume,
                            'value': value,
                            'change_percent': change_percent,
                            'change_point': change_point,
                            'source': source,
                            'raw_json': raw_json
                        })
                        
                        stats['inserted'] += 1
                    
                except Exception as e:
                    logger.error(f"Error saving record: {str(e)}")
                    stats['failed'] += 1
                    continue
            
            # Commit all changes
            self.db.commit()
            
            logger.info(
                f"💾 Saved raw data: {stats['inserted']} inserted, "
                f"{stats['updated']} updated, {stats['failed']} failed"
            )
            
            return stats
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error saving raw data: {str(e)}")
            stats['status'] = 'error'
            stats['error'] = str(e)
            return stats
    
    # =====================================================
    # BATCH COLLECTION
    # =====================================================
    
    def collect_multiple_symbols(
        self,
        symbols: List[str],
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> Dict:
        """
        Thu thập dữ liệu cho nhiều mã cổ phiếu
        
        Args:
            symbols: List of stock symbols
            from_date: Start date
            to_date: End date
        
        Returns:
            dict: Combined statistics
        """
        logger.info("=" * 80)
        logger.info("📡 BATCH RAW DATA COLLECTION")
        logger.info(f"   Symbols: {', '.join(symbols)}")
        logger.info(f"   Date range: {from_date} to {to_date}")
        logger.info("=" * 80)
        
        total_stats = {
            'status': 'success',
            'symbols_processed': 0,
            'total_collected': 0,
            'total_inserted': 0,
            'total_updated': 0,
            'total_failed': 0,
            'symbol_details': {}
        }
        
        for symbol in symbols:
            logger.info(f"\n📊 Processing {symbol}...")
            
            stats = self.collect_price_data(symbol, from_date, to_date)
            
            total_stats['symbols_processed'] += 1
            total_stats['total_collected'] += stats.get('collected', 0)
            total_stats['total_inserted'] += stats.get('inserted', 0)
            total_stats['total_updated'] += stats.get('updated', 0)
            total_stats['total_failed'] += stats.get('failed', 0)
            total_stats['symbol_details'][symbol] = stats
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ BATCH COLLECTION COMPLETED")
        logger.info(f"   Symbols processed: {total_stats['symbols_processed']}")
        logger.info(f"   Total collected: {total_stats['total_collected']}")
        logger.info(f"   Total inserted: {total_stats['total_inserted']}")
        logger.info(f"   Total updated: {total_stats['total_updated']}")
        logger.info(f"   Total failed: {total_stats['total_failed']}")
        logger.info("=" * 80)
        
        return total_stats
    
    def collect_all_stocks(
        self,
        days: int = 30
    ) -> Dict:
        """
        Thu thập dữ liệu cho tất cả stocks trong database
        
        Args:
            days: Number of days to look back
        
        Returns:
            dict: Statistics
        """
        try:
            # Get all stock symbols from database
            stocks = self.db.query(Stock.symbol).all()
            symbols = [s[0] for s in stocks]
            
            logger.info(f"📊 Found {len(symbols)} stocks in database")
            
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Collect data for all symbols
            return self.collect_multiple_symbols(symbols, from_date, to_date)
        
        except Exception as e:
            logger.error(f"❌ Error collecting all stocks: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # =====================================================
    # QUERY RAW DATA
    # =====================================================
    
    def get_unprocessed_data(
        self,
        symbol: Optional[str] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Lấy raw data chưa được xử lý
        
        Args:
            symbol: Stock symbol filter
            limit: Max records
        
        Returns:
            DataFrame with unprocessed raw data
        """
        try:
            query = text("""
                SELECT * FROM raw_stock_data
                WHERE processed = FALSE
                {symbol_filter}
                ORDER BY date DESC
                {limit_clause}
            """.format(
                symbol_filter=f"AND symbol = '{symbol.upper()}'" if symbol else "",
                limit_clause=f"LIMIT {limit}" if limit else ""
            ))
            
            result = self.db.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            logger.info(f"✅ Found {len(df)} unprocessed records")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error querying unprocessed data: {str(e)}")
            return pd.DataFrame()
    
    def mark_as_processed(
        self,
        record_ids: List[int]
    ) -> int:
        """
        Đánh dấu raw data đã được xử lý
        
        Args:
            record_ids: List of record IDs
        
        Returns:
            Number of records marked
        """
        try:
            if not record_ids:
                return 0
            
            ids_str = ','.join(map(str, record_ids))
            
            update_query = text(f"""
                UPDATE raw_stock_data
                SET processed = TRUE
                WHERE id IN ({ids_str})
            """)
            
            result = self.db.execute(update_query)
            self.db.commit()
            
            count = result.rowcount
            logger.info(f"✅ Marked {count} records as processed")
            
            return count
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error marking as processed: {str(e)}")
            return 0
    
    def close(self):
        """Close database connection"""
        if self.db:
            self.db.close()


if __name__ == "__main__":
    # Test raw data collector
    print("Testing Raw Data Collector...")
    print("-" * 80)
    
    collector = RawDataCollector()
    
    # Test 1: Collect data for VNM
    print("\n1. Collect raw data for VNM (last 30 days)")
    stats = collector.collect_price_data('VNM')
    print(f"   Result: {stats}")
    
    # Test 2: Collect for multiple symbols
    print("\n2. Collect for multiple symbols")
    batch_stats = collector.collect_multiple_symbols(['VNM', 'HPG', 'VCB'], days=7)
    print(f"   Result: {batch_stats}")
    
    # Test 3: Get unprocessed data
    print("\n3. Get unprocessed raw data")
    df_unprocessed = collector.get_unprocessed_data(limit=10)
    print(f"   Found {len(df_unprocessed)} unprocessed records")
    
    collector.close()
    print("\n✅ Raw data collector test completed")
