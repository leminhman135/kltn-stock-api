"""
ETL Module 2: Transform - Làm sạch và chuẩn hóa dữ liệu
Xử lý dữ liệu giá và tin tức
"""

import pandas as pd
import numpy as np
import re
import unicodedata
from bs4 import BeautifulSoup
from typing import Optional
import logging

from src.etl.config_loader import get_config

logger = logging.getLogger(__name__)


class DataTransformer:
    """Transform and clean data"""
    
    def __init__(self):
        self.config = get_config()
    
    # =====================================================
    # PRICE DATA TRANSFORMATION
    # =====================================================
    
    def transform_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform price data
        
        Operations:
        1. Remove duplicates
        2. Handle missing values
        3. Validate price ranges
        4. Validate OHLC relationships
        5. Normalize data types
        6. Sort data
        
        Args:
            df: Raw price DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("⚠️  Empty DataFrame, skipping transformation")
            return df
        
        logger.info(f"🔄 Transforming {len(df)} price records")
        
        original_count = len(df)
        
        try:
            # 1. Remove duplicates (keep last)
            df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')
            
            # 2. Handle missing values
            # Forward fill for prices (use previous day's price)
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns:
                    df[col] = df.groupby('symbol')[col].fillna(method='ffill')
            
            # Fill volume with 0
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0)
            
            # 3. Validate price ranges (prices must be positive)
            for col in price_cols:
                if col in df.columns:
                    df = df[df[col] > 0]
            
            # 4. Validate OHLC relationships
            if all(col in df.columns for col in price_cols):
                df = df[
                    (df['high'] >= df['low']) &
                    (df['high'] >= df['open']) &
                    (df['high'] >= df['close']) &
                    (df['low'] <= df['open']) &
                    (df['low'] <= df['close'])
                ]
            
            # 5. Normalize data types
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date']).dt.date
            
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].str.upper().str.strip()
            
            for col in price_cols:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            
            if 'volume' in df.columns:
                df['volume'] = df['volume'].astype(int)
            
            # 6. Sort by symbol and date
            df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
            
            removed = original_count - len(df)
            logger.info(f"✅ Transformed: {len(df)} valid records ({removed} removed)")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error transforming price data: {str(e)}")
            return df
    
    # =====================================================
    # NEWS DATA TRANSFORMATION
    # =====================================================
    
    def transform_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and transform news data
        
        Operations:
        1. Clean text content (remove HTML, special chars)
        2. Normalize Vietnamese text
        3. Create clean_content field
        4. Validate content length
        5. Remove duplicates
        6. Normalize data types
        
        Args:
            df: Raw news DataFrame
        
        Returns:
            Cleaned DataFrame
        """
        if df.empty:
            logger.warning("⚠️  Empty DataFrame, skipping transformation")
            return df
        
        logger.info(f"🔄 Transforming {len(df)} news records")
        
        original_count = len(df)
        
        try:
            # 1. Create clean_content field
            df['clean_content'] = df.apply(
                lambda row: self._clean_text(
                    row.get('content') or row.get('summary') or row.get('title') or ''
                ),
                axis=1
            )
            
            # 2. Clean title
            if 'title' in df.columns:
                df['title'] = df['title'].apply(self._clean_text)
            
            # 3. Normalize symbol
            if 'symbol' in df.columns:
                df['symbol'] = df['symbol'].str.upper().str.strip()
            
            # 4. Normalize dates
            if 'published_at' in df.columns:
                df['published_at'] = pd.to_datetime(df['published_at'])
            
            # 5. Validate content length (minimum 50 characters)
            min_length = self.config.get('etl.news.min_content_length', 50)
            df = df[df['clean_content'].str.len() >= min_length]
            
            # 6. Remove duplicates (same title + symbol)
            if 'title' in df.columns and 'symbol' in df.columns:
                df = df.drop_duplicates(subset=['symbol', 'title'], keep='first')
            
            # 7. Sort by published date
            if 'published_at' in df.columns:
                df = df.sort_values('published_at', ascending=False)
            
            df = df.reset_index(drop=True)
            
            removed = original_count - len(df)
            logger.info(f"✅ Transformed: {len(df)} valid records ({removed} removed)")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error transforming news data: {str(e)}")
            return df
    
    # =====================================================
    # TEXT CLEANING UTILITIES
    # =====================================================
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text content
        
        Steps:
        1. Remove HTML tags
        2. Remove URLs and emails
        3. Normalize Unicode
        4. Remove special characters
        5. Remove extra whitespace
        
        Args:
            text: Raw text
        
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        try:
            # 1. Remove HTML tags
            soup = BeautifulSoup(text, 'html.parser')
            text = soup.get_text()
            
            # 2. Normalize Unicode (NFC form for Vietnamese)
            text = unicodedata.normalize('NFC', text)
            
            # 3. Remove URLs
            text = re.sub(
                r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
                '',
                text
            )
            
            # 4. Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # 5. Remove special characters (keep Vietnamese characters)
            # Keep: letters, numbers, spaces, basic punctuation
            text = re.sub(r'[^\w\s\.,;:!?\-\(\)àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', '', text, flags=re.IGNORECASE)
            
            # 6. Remove extra whitespace
            text = ' '.join(text.split())
            
            # 7. Trim
            text = text.strip()
            
            return text
        
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return ""
    
    def _normalize_vietnamese(self, text: str) -> str:
        """
        Normalize Vietnamese text
        
        - Convert to lowercase
        - Normalize Unicode composition
        - Remove tone marks (optional)
        """
        if not text:
            return ""
        
        # Normalize Unicode
        text = unicodedata.normalize('NFC', text)
        
        return text.strip()
    
    # =====================================================
    # DATA VALIDATION
    # =====================================================
    
    def validate_price_data(self, df: pd.DataFrame) -> dict:
        """
        Validate price data quality
        
        Returns:
            dict: Validation report with statistics
        """
        report = {
            'total_records': len(df),
            'valid_records': 0,
            'issues': []
        }
        
        if df.empty:
            report['issues'].append("Empty DataFrame")
            return report
        
        # Check required columns
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            report['issues'].append(f"Missing columns: {missing_cols}")
            return report
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            report['null_counts'] = null_counts.to_dict()
        
        # Check for negative prices
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()
        if negative_prices > 0:
            report['issues'].append(f"{negative_prices} records with negative prices")
        
        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).sum()
        
        if invalid_ohlc > 0:
            report['issues'].append(f"{invalid_ohlc} records with invalid OHLC relationships")
        
        report['valid_records'] = len(df) - negative_prices - invalid_ohlc
        
        return report
    
    def validate_news_data(self, df: pd.DataFrame) -> dict:
        """
        Validate news data quality
        
        Returns:
            dict: Validation report with statistics
        """
        report = {
            'total_records': len(df),
            'valid_records': 0,
            'issues': []
        }
        
        if df.empty:
            report['issues'].append("Empty DataFrame")
            return report
        
        # Check required columns
        required_cols = ['symbol', 'title', 'published_at']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            report['issues'].append(f"Missing columns: {missing_cols}")
            return report
        
        # Check for null values
        null_counts = df[required_cols].isnull().sum()
        if null_counts.sum() > 0:
            report['null_counts'] = null_counts.to_dict()
        
        # Check content length
        if 'clean_content' in df.columns:
            short_content = (df['clean_content'].str.len() < 50).sum()
            if short_content > 0:
                report['issues'].append(f"{short_content} records with too short content")
        
        # Check for duplicates
        duplicates = df.duplicated(subset=['symbol', 'title']).sum()
        if duplicates > 0:
            report['issues'].append(f"{duplicates} duplicate records")
        
        report['valid_records'] = len(df) - duplicates
        
        return report


if __name__ == "__main__":
    # Test transformer
    print("Testing Data Transformer...")
    print("-" * 80)
    
    transformer = DataTransformer()
    
    # Test 1: Transform price data
    print("\n1. Test price data transformation")
    test_price_df = pd.DataFrame({
        'symbol': ['VNM', 'VNM', 'HPG'],
        'date': ['2024-01-01', '2024-01-02', '2024-01-01'],
        'open': [80000, 81000, 25000],
        'high': [81000, 82000, 26000],
        'low': [79000, 80000, 24000],
        'close': [80500, 81500, 25500],
        'volume': [1000000, 1200000, 2000000]
    })
    
    transformed = transformer.transform_price_data(test_price_df)
    print(f"   Original: {len(test_price_df)} records")
    print(f"   Transformed: {len(transformed)} records")
    print(transformed)
    
    # Test 2: Transform news data
    print("\n2. Test news data transformation")
    test_news_df = pd.DataFrame({
        'symbol': ['VNM', 'HPG'],
        'title': ['<b>Vinamilk tăng trưởng mạnh</b>', 'Hòa Phát báo lãi cao'],
        'content': ['<html><p>Nội dung tin tức... http://example.com</p></html>', 'Nội dung về Hòa Phát...'],
        'published_at': ['2024-01-15 10:00:00', '2024-01-15 11:00:00']
    })
    
    transformed_news = transformer.transform_news_data(test_news_df)
    print(f"   Original: {len(test_news_df)} records")
    print(f"   Transformed: {len(transformed_news)} records")
    print(transformed_news[['symbol', 'title', 'clean_content']])
    
    # Test 3: Validate data
    print("\n3. Test data validation")
    price_report = transformer.validate_price_data(transformed)
    print(f"   Price validation: {price_report}")
    
    news_report = transformer.validate_news_data(transformed_news)
    print(f"   News validation: {news_report}")
    
    print("\n✅ Transformer test completed")
