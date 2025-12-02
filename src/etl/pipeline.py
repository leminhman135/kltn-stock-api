"""
ETL Pipeline Orchestrator - Kết hợp Extract, Transform, Load
Main pipeline for end-to-end data processing
"""

import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import logging

from src.database.connection import get_db
from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer
from src.etl.load import DataLoader
from src.etl.config_loader import get_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ETLOrchestrator:
    """
    ETL Pipeline Orchestrator
    
    Coordinates Extract → Transform → Load process for:
    - Stock price data
    - News article data
    """
    
    def __init__(self):
        self.config = get_config()
        self.db = next(get_db())
        
        # Initialize ETL components
        self.extractor = DataExtractor(self.db)
        self.transformer = DataTransformer()
        self.loader = DataLoader(self.db)
    
    # =====================================================
    # PRICE DATA ETL
    # =====================================================
    
    def run_price_etl(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict:
        """
        Run full ETL pipeline for price data
        
        Args:
            symbol: Stock symbol (None = all stocks)
            start_date: Start date filter
            end_date: End date filter
        
        Returns:
            dict: Pipeline statistics
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING PRICE DATA ETL PIPELINE")
        logger.info(f"   Symbol: {symbol or 'ALL'}")
        logger.info(f"   Date range: {start_date} to {end_date}")
        logger.info("=" * 80)
        
        pipeline_start = datetime.now()
        
        try:
            # PHASE 1: EXTRACT
            logger.info("\n📥 PHASE 1: EXTRACT")
            df = self.extractor.extract_price_data(symbol, start_date, end_date)
            
            if df.empty:
                logger.warning("⚠️  No data extracted, pipeline stopped")
                return {
                    'status': 'no_data',
                    'extracted': 0,
                    'transformed': 0,
                    'loaded': {'inserted': 0, 'updated': 0, 'failed': 0}
                }
            
            extracted_count = len(df)
            logger.info(f"✅ Extracted {extracted_count} records")
            
            # PHASE 2: TRANSFORM
            logger.info("\n🔄 PHASE 2: TRANSFORM")
            df = self.transformer.transform_price_data(df)
            transformed_count = len(df)
            logger.info(f"✅ Transformed {transformed_count} records")
            
            # Validation
            validation = self.transformer.validate_price_data(df)
            logger.info(f"📊 Validation: {validation}")
            
            # PHASE 3: LOAD
            logger.info("\n💾 PHASE 3: LOAD")
            load_stats = self.loader.load_price_data(df)
            logger.info(f"✅ Load complete: {load_stats}")
            
            # SUMMARY
            duration = (datetime.now() - pipeline_start).total_seconds()
            
            result = {
                'status': 'success',
                'extracted': extracted_count,
                'transformed': transformed_count,
                'loaded': load_stats,
                'validation': validation,
                'duration_seconds': round(duration, 2)
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ PRICE DATA ETL PIPELINE COMPLETED")
            logger.info(f"   Extracted: {extracted_count} records")
            logger.info(f"   Transformed: {transformed_count} records")
            logger.info(f"   Inserted: {load_stats['inserted']} records")
            logger.info(f"   Updated: {load_stats['updated']} records")
            logger.info(f"   Failed: {load_stats['failed']} records")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info("=" * 80)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ ETL Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # =====================================================
    # NEWS DATA ETL
    # =====================================================
    
    def run_news_etl(
        self,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        unprocessed_only: bool = True
    ) -> Dict:
        """
        Run full ETL pipeline for news data
        
        Args:
            symbol: Stock symbol filter
            start_date: Start date filter
            unprocessed_only: Only process news without sentiment
        
        Returns:
            dict: Pipeline statistics
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING NEWS DATA ETL PIPELINE")
        logger.info(f"   Symbol: {symbol or 'ALL'}")
        logger.info(f"   Date: {start_date}")
        logger.info(f"   Unprocessed only: {unprocessed_only}")
        logger.info("=" * 80)
        
        pipeline_start = datetime.now()
        
        try:
            # PHASE 1: EXTRACT
            logger.info("\n📥 PHASE 1: EXTRACT")
            df = self.extractor.extract_news_data(symbol, start_date, unprocessed_only)
            
            if df.empty:
                logger.warning("⚠️  No data extracted, pipeline stopped")
                return {
                    'status': 'no_data',
                    'extracted': 0,
                    'transformed': 0,
                    'loaded': {'updated': 0, 'failed': 0}
                }
            
            extracted_count = len(df)
            logger.info(f"✅ Extracted {extracted_count} records")
            
            # PHASE 2: TRANSFORM
            logger.info("\n🔄 PHASE 2: TRANSFORM")
            df = self.transformer.transform_news_data(df)
            transformed_count = len(df)
            logger.info(f"✅ Transformed {transformed_count} records")
            
            # Validation
            validation = self.transformer.validate_news_data(df)
            logger.info(f"📊 Validation: {validation}")
            
            # PHASE 3: LOAD
            logger.info("\n💾 PHASE 3: LOAD")
            load_stats = self.loader.load_news_data(df)
            logger.info(f"✅ Load complete: {load_stats}")
            
            # SUMMARY
            duration = (datetime.now() - pipeline_start).total_seconds()
            
            result = {
                'status': 'success',
                'extracted': extracted_count,
                'transformed': transformed_count,
                'loaded': load_stats,
                'validation': validation,
                'duration_seconds': round(duration, 2),
                'processed_news_ids': df['id'].tolist() if not df.empty else []
            }
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ NEWS DATA ETL PIPELINE COMPLETED")
            logger.info(f"   Extracted: {extracted_count} records")
            logger.info(f"   Transformed: {transformed_count} records")
            logger.info(f"   Updated: {load_stats['updated']} records")
            logger.info(f"   Failed: {load_stats['failed']} records")
            logger.info(f"   Duration: {duration:.2f}s")
            logger.info("=" * 80)
            
            return result
        
        except Exception as e:
            logger.error(f"❌ ETL Pipeline failed: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    # =====================================================
    # CONVENIENCE METHODS
    # =====================================================
    
    def run_daily_price_update(self) -> Dict:
        """
        Run daily price update (last 7 days)
        
        Returns:
            Pipeline statistics
        """
        start_date = datetime.now() - timedelta(days=7)
        return self.run_price_etl(start_date=start_date)
    
    def run_daily_news_update(self) -> Dict:
        """
        Run daily news update (last 7 days, unprocessed only)
        
        Returns:
            Pipeline statistics
        """
        start_date = datetime.now() - timedelta(days=7)
        return self.run_news_etl(start_date=start_date, unprocessed_only=True)
    
    def run_full_etl(self, symbol: Optional[str] = None) -> Dict:
        """
        Run both price and news ETL
        
        Args:
            symbol: Stock symbol (None = all stocks)
        
        Returns:
            Combined statistics
        """
        logger.info("\n" + "=" * 80)
        logger.info("🚀 RUNNING FULL ETL PIPELINE (PRICE + NEWS)")
        logger.info("=" * 80 + "\n")
        
        # Run price ETL
        price_result = self.run_price_etl(symbol=symbol)
        
        # Run news ETL
        news_result = self.run_news_etl(symbol=symbol, unprocessed_only=True)
        
        return {
            'price_etl': price_result,
            'news_etl': news_result
        }
    
    def close(self):
        """Close all connections"""
        self.extractor.close()
        self.loader.close()
        if self.db:
            self.db.close()


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def run_price_etl(symbol: Optional[str] = None, days: int = 30) -> Dict:
    """
    Helper: Run price ETL for recent data
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
    
    Returns:
        Pipeline statistics
    """
    orchestrator = ETLOrchestrator()
    try:
        start_date = datetime.now() - timedelta(days=days)
        return orchestrator.run_price_etl(symbol=symbol, start_date=start_date)
    finally:
        orchestrator.close()


def run_news_etl(symbol: Optional[str] = None, days: int = 7) -> Dict:
    """
    Helper: Run news ETL for recent data
    
    Args:
        symbol: Stock symbol
        days: Number of days to look back
    
    Returns:
        Pipeline statistics
    """
    orchestrator = ETLOrchestrator()
    try:
        start_date = datetime.now() - timedelta(days=days)
        return orchestrator.run_news_etl(symbol=symbol, start_date=start_date, unprocessed_only=True)
    finally:
        orchestrator.close()


def run_full_etl(symbol: Optional[str] = None) -> Dict:
    """
    Helper: Run complete ETL pipeline
    
    Args:
        symbol: Stock symbol (None = all stocks)
    
    Returns:
        Combined statistics
    """
    orchestrator = ETLOrchestrator()
    try:
        return orchestrator.run_full_etl(symbol=symbol)
    finally:
        orchestrator.close()


if __name__ == "__main__":
    print("Testing ETL Orchestrator...")
    print("-" * 80)
    
    # Test full ETL for VNM
    result = run_full_etl(symbol="VNM")
    
    print("\n📊 ETL Results:")
    print(f"Price ETL: {result['price_etl']}")
    print(f"News ETL: {result['news_etl']}")
    
    print("\n✅ ETL Orchestrator test completed")
