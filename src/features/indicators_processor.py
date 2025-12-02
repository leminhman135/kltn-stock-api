"""
Module xử lý tính toán và lưu trữ Technical Indicators vào Database
"""

import pandas as pd
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import desc
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from src.database.connection import get_db, engine
from src.database.models import Stock, StockPrice, TechnicalIndicator
from src.features.technical_indicators import TechnicalIndicators

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndicatorsProcessor:
    """Xử lý tính toán và lưu trữ Technical Indicators"""
    
    def __init__(self, db: Session):
        self.db = db
        self.calculator = TechnicalIndicators()
    
    def get_stock_price_data(self, stock_id: int, days: int = 365) -> pd.DataFrame:
        """
        Lấy dữ liệu giá từ database
        
        Args:
            stock_id: ID của stock
            days: Số ngày lấy về (mặc định 365 ngày)
        
        Returns:
            DataFrame với dữ liệu giá OHLCV
        """
        try:
            # Lấy dữ liệu giá từ DB
            prices = self.db.query(StockPrice).filter(
                StockPrice.stock_id == stock_id
            ).order_by(desc(StockPrice.date)).limit(days).all()
            
            if not prices:
                logger.warning(f"No price data found for stock_id={stock_id}")
                return pd.DataFrame()
            
            # Convert sang DataFrame
            data = []
            for p in prices:
                data.append({
                    'date': p.date,
                    'open': p.open,
                    'high': p.high,
                    'low': p.low,
                    'close': p.close,
                    'volume': p.volume
                })
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Loaded {len(df)} price records for stock_id={stock_id}")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error loading price data: {str(e)}")
            return pd.DataFrame()
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tính toán tất cả các chỉ báo kỹ thuật
        
        Args:
            df: DataFrame với dữ liệu giá OHLCV
        
        Returns:
            DataFrame với tất cả indicators đã tính
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        try:
            # Moving Averages
            result_df['sma_20'] = self.calculator.calculate_sma(df, window=20)
            result_df['sma_50'] = self.calculator.calculate_sma(df, window=50)
            result_df['sma_200'] = self.calculator.calculate_sma(df, window=200)
            result_df['ema_12'] = self.calculator.calculate_ema(df, window=12)
            result_df['ema_26'] = self.calculator.calculate_ema(df, window=26)
            
            # RSI
            result_df['rsi_14'] = self.calculator.calculate_rsi(df, window=14)
            
            # MACD
            macd_df = self.calculator.calculate_macd(df)
            result_df['macd'] = macd_df['macd']
            result_df['macd_signal'] = macd_df['macd_signal']
            result_df['macd_histogram'] = macd_df['macd_histogram']
            
            # Bollinger Bands
            bb_df = self.calculator.calculate_bollinger_bands(df)
            result_df['bb_upper'] = bb_df['bb_upper']
            result_df['bb_middle'] = bb_df['bb_middle']
            result_df['bb_lower'] = bb_df['bb_lower']
            
            # Stochastic Oscillator
            stoch_df = self.calculator.calculate_stochastic_oscillator(df)
            result_df['stoch_k'] = stoch_df['stoch_k']
            result_df['stoch_d'] = stoch_df['stoch_d']
            
            # ATR
            result_df['atr_14'] = self.calculator.calculate_atr(df, window=14)
            
            # OBV
            result_df['obv'] = self.calculator.calculate_obv(df)
            
            # ADX
            adx_df = self.calculator.calculate_adx(df)
            result_df['adx'] = adx_df['adx']
            result_df['plus_di'] = adx_df['plus_di']
            result_df['minus_di'] = adx_df['minus_di']
            
            # CCI
            result_df['cci'] = self.calculator.calculate_cci(df)
            
            # Williams %R
            result_df['williams_r'] = self.calculator.calculate_williams_r(df)
            
            logger.info(f"✅ Calculated indicators for {len(result_df)} records")
            return result_df
        
        except Exception as e:
            logger.error(f"❌ Error calculating indicators: {str(e)}")
            return df
    
    def save_indicators_to_db(self, stock_id: int, df: pd.DataFrame) -> int:
        """
        Lưu indicators vào database
        
        Args:
            stock_id: ID của stock
            df: DataFrame với indicators đã tính
        
        Returns:
            Số lượng records đã lưu
        """
        if df.empty:
            return 0
        
        saved_count = 0
        
        try:
            for idx, row in df.iterrows():
                # Skip nếu có NaN values (thường ở đầu series do rolling)
                if pd.isna(row['sma_20']):
                    continue
                
                # Kiểm tra xem đã có record này chưa
                existing = self.db.query(TechnicalIndicator).filter(
                    TechnicalIndicator.stock_id == stock_id,
                    TechnicalIndicator.date == row['date']
                ).first()
                
                if existing:
                    # Update existing record
                    existing.sma_20 = float(row['sma_20']) if not pd.isna(row['sma_20']) else None
                    existing.sma_50 = float(row['sma_50']) if not pd.isna(row['sma_50']) else None
                    existing.sma_200 = float(row['sma_200']) if not pd.isna(row['sma_200']) else None
                    existing.ema_12 = float(row['ema_12']) if not pd.isna(row['ema_12']) else None
                    existing.ema_26 = float(row['ema_26']) if not pd.isna(row['ema_26']) else None
                    existing.rsi_14 = float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None
                    existing.macd = float(row['macd']) if not pd.isna(row['macd']) else None
                    existing.macd_signal = float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None
                    existing.macd_histogram = float(row['macd_histogram']) if not pd.isna(row['macd_histogram']) else None
                    existing.bb_upper = float(row['bb_upper']) if not pd.isna(row['bb_upper']) else None
                    existing.bb_middle = float(row['bb_middle']) if not pd.isna(row['bb_middle']) else None
                    existing.bb_lower = float(row['bb_lower']) if not pd.isna(row['bb_lower']) else None
                    existing.stoch_k = float(row['stoch_k']) if not pd.isna(row['stoch_k']) else None
                    existing.stoch_d = float(row['stoch_d']) if not pd.isna(row['stoch_d']) else None
                    existing.atr_14 = float(row['atr_14']) if not pd.isna(row['atr_14']) else None
                    existing.obv = float(row['obv']) if not pd.isna(row['obv']) else None
                    existing.adx = float(row['adx']) if not pd.isna(row['adx']) else None
                    existing.plus_di = float(row['plus_di']) if not pd.isna(row['plus_di']) else None
                    existing.minus_di = float(row['minus_di']) if not pd.isna(row['minus_di']) else None
                    existing.cci = float(row['cci']) if not pd.isna(row['cci']) else None
                    existing.williams_r = float(row['williams_r']) if not pd.isna(row['williams_r']) else None
                else:
                    # Create new record
                    indicator = TechnicalIndicator(
                        stock_id=stock_id,
                        date=row['date'],
                        sma_20=float(row['sma_20']) if not pd.isna(row['sma_20']) else None,
                        sma_50=float(row['sma_50']) if not pd.isna(row['sma_50']) else None,
                        sma_200=float(row['sma_200']) if not pd.isna(row['sma_200']) else None,
                        ema_12=float(row['ema_12']) if not pd.isna(row['ema_12']) else None,
                        ema_26=float(row['ema_26']) if not pd.isna(row['ema_26']) else None,
                        rsi_14=float(row['rsi_14']) if not pd.isna(row['rsi_14']) else None,
                        macd=float(row['macd']) if not pd.isna(row['macd']) else None,
                        macd_signal=float(row['macd_signal']) if not pd.isna(row['macd_signal']) else None,
                        macd_histogram=float(row['macd_histogram']) if not pd.isna(row['macd_histogram']) else None,
                        bb_upper=float(row['bb_upper']) if not pd.isna(row['bb_upper']) else None,
                        bb_middle=float(row['bb_middle']) if not pd.isna(row['bb_middle']) else None,
                        bb_lower=float(row['bb_lower']) if not pd.isna(row['bb_lower']) else None,
                        stoch_k=float(row['stoch_k']) if not pd.isna(row['stoch_k']) else None,
                        stoch_d=float(row['stoch_d']) if not pd.isna(row['stoch_d']) else None,
                        atr_14=float(row['atr_14']) if not pd.isna(row['atr_14']) else None,
                        obv=float(row['obv']) if not pd.isna(row['obv']) else None,
                        adx=float(row['adx']) if not pd.isna(row['adx']) else None,
                        plus_di=float(row['plus_di']) if not pd.isna(row['plus_di']) else None,
                        minus_di=float(row['minus_di']) if not pd.isna(row['minus_di']) else None,
                        cci=float(row['cci']) if not pd.isna(row['cci']) else None,
                        williams_r=float(row['williams_r']) if not pd.isna(row['williams_r']) else None
                    )
                    self.db.add(indicator)
                
                saved_count += 1
            
            self.db.commit()
            logger.info(f"✅ Saved {saved_count} indicator records for stock_id={stock_id}")
            return saved_count
        
        except Exception as e:
            self.db.rollback()
            logger.error(f"❌ Error saving indicators: {str(e)}")
            return 0
    
    def process_stock(self, stock_id: int, days: int = 365) -> bool:
        """
        Xử lý đầy đủ cho một stock: load data → calculate → save
        
        Args:
            stock_id: ID của stock
            days: Số ngày dữ liệu
        
        Returns:
            True nếu thành công
        """
        try:
            logger.info(f"🔄 Processing indicators for stock_id={stock_id}")
            
            # 1. Load price data
            df = self.get_stock_price_data(stock_id, days=days)
            if df.empty:
                logger.warning(f"⚠️ No data to process for stock_id={stock_id}")
                return False
            
            # 2. Calculate indicators
            df_with_indicators = self.calculate_all_indicators(df)
            
            # 3. Save to database
            saved = self.save_indicators_to_db(stock_id, df_with_indicators)
            
            if saved > 0:
                logger.info(f"✅ Successfully processed stock_id={stock_id}: {saved} records")
                return True
            else:
                logger.warning(f"⚠️ No indicators saved for stock_id={stock_id}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Error processing stock_id={stock_id}: {str(e)}")
            return False
    
    def process_all_stocks(self, days: int = 365) -> dict:
        """
        Xử lý tất cả stocks trong database
        
        Returns:
            Dict với thống kê: {success: int, failed: int, total: int}
        """
        try:
            stocks = self.db.query(Stock).filter(Stock.is_active == True).all()
            
            if not stocks:
                logger.warning("⚠️ No active stocks found in database")
                return {'success': 0, 'failed': 0, 'total': 0}
            
            success_count = 0
            failed_count = 0
            
            logger.info(f"🚀 Starting indicator calculation for {len(stocks)} stocks")
            
            for stock in stocks:
                if self.process_stock(stock.id, days=days):
                    success_count += 1
                else:
                    failed_count += 1
            
            result = {
                'success': success_count,
                'failed': failed_count,
                'total': len(stocks)
            }
            
            logger.info(f"✅ Indicator calculation complete: {success_count}/{len(stocks)} succeeded")
            return result
        
        except Exception as e:
            logger.error(f"❌ Error in process_all_stocks: {str(e)}")
            return {'success': 0, 'failed': 0, 'total': 0}


def run_indicator_calculation():
    """Helper function để chạy từ scheduler hoặc command line"""
    db = next(get_db())
    try:
        processor = IndicatorsProcessor(db)
        result = processor.process_all_stocks(days=365)
        logger.info(f"📊 Indicator calculation result: {result}")
        return result
    finally:
        db.close()


if __name__ == "__main__":
    # Test module
    result = run_indicator_calculation()
    print(f"\nIndicator Calculation Results:")
    print(f"  Success: {result['success']}")
    print(f"  Failed: {result['failed']}")
    print(f"  Total: {result['total']}")
