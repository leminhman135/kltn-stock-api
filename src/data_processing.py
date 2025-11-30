"""
Module xử lý dữ liệu: ETL Pipeline
Extract - Transform - Load
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import json
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataExtractor:
    """Extract: Trích xuất dữ liệu từ nhiều nguồn"""
    
    def __init__(self, data_dir: str = './data'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def extract_from_csv(self, filepath: str) -> pd.DataFrame:
        """Trích xuất dữ liệu từ CSV"""
        try:
            df = pd.read_csv(filepath)
            logger.info(f"Extracted {len(df)} records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error extracting CSV: {str(e)}")
            return pd.DataFrame()
    
    def extract_from_dict(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Kết hợp nhiều DataFrame thành một"""
        try:
            dfs = []
            for symbol, df in data_dict.items():
                df_copy = df.copy()
                if 'symbol' not in df_copy.columns:
                    df_copy['symbol'] = symbol
                dfs.append(df_copy)
            
            combined_df = pd.concat(dfs, ignore_index=True)
            logger.info(f"Combined {len(data_dict)} DataFrames into one with {len(combined_df)} records")
            return combined_df
        except Exception as e:
            logger.error(f"Error combining DataFrames: {str(e)}")
            return pd.DataFrame()


class DataTransformer:
    """Transform: Biến đổi, làm sạch, chuẩn hóa dữ liệu"""
    
    def __init__(self):
        self.scaler = None
    
    def clean_price_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Làm sạch dữ liệu giá cổ phiếu"""
        df_clean = df.copy()
        
        if 'date' in df_clean.columns:
            df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        column_mapping = {
            'Open': 'open', 'High': 'high', 'Low': 'low',
            'Close': 'close', 'Volume': 'volume', 'Adj Close': 'adj_close'
        }
        df_clean.rename(columns=column_mapping, inplace=True)
        
        important_cols = ['open', 'high', 'low', 'close', 'volume']
        df_clean.dropna(subset=[col for col in important_cols if col in df_clean.columns], inplace=True)
        
        for col in ['open', 'high', 'low', 'close']:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]
        
        if 'date' in df_clean.columns:
            df_clean.sort_values('date', inplace=True)
            df_clean.reset_index(drop=True, inplace=True)
        
        logger.info(f"Cleaned data: {len(df)} -> {len(df_clean)} records")
        return df_clean
    
    def handle_missing_values(self, df: pd.DataFrame, method: str = 'ffill') -> pd.DataFrame:
        """Xử lý giá trị thiếu"""
        df_filled = df.copy()
        
        if method == 'ffill':
            df_filled.fillna(method='ffill', inplace=True)
        elif method == 'interpolate':
            numeric_cols = df_filled.select_dtypes(include=[np.number]).columns
            df_filled[numeric_cols] = df_filled[numeric_cols].interpolate(method='linear')
        
        return df_filled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Tạo các features cơ bản"""
        df_features = df.copy()
        
        if 'close' in df_features.columns:
            df_features['returns'] = df_features['close'].pct_change()
            df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
        
        if 'high' in df_features.columns and 'low' in df_features.columns:
            df_features['hl_range'] = df_features['high'] - df_features['low']
        
        if 'volume' in df_features.columns:
            df_features['volume_change'] = df_features['volume'].pct_change()
        
        return df_features


class DataLoader:
    """Load: Tải dữ liệu vào nơi lưu trữ"""
    
    def __init__(self, output_dir: str = './data/processed'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_to_csv(self, df: pd.DataFrame, filename: str) -> bool:
        """Lưu vào CSV"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.info(f"Saved {len(df)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving to CSV: {str(e)}")
            return False


class ETLPipeline:
    """Pipeline ETL tổng hợp"""
    
    def __init__(self, data_dir: str = './data', output_dir: str = './data/processed'):
        self.extractor = DataExtractor(data_dir)
        self.transformer = DataTransformer()
        self.loader = DataLoader(output_dir)
    
    def process_price_data(self, data_dict: Dict[str, pd.DataFrame], 
                          save_format: str = 'csv') -> pd.DataFrame:
        """Xử lý dữ liệu giá cổ phiếu hoàn chỉnh"""
        logger.info("Starting ETL pipeline for price data...")
        
        df = self.extractor.extract_from_dict(data_dict)
        
        if df.empty:
            logger.error("No data to process")
            return df
        
        df = self.transformer.clean_price_data(df)
        df = self.transformer.handle_missing_values(df, method='ffill')
        df = self.transformer.create_features(df)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'stock_prices_processed_{timestamp}.{save_format}'
        
        if save_format == 'csv':
            self.loader.load_to_csv(df, filename)
        
        logger.info("ETL pipeline completed successfully")
        return df


def preprocess_data(data):
    """Legacy function for backward compatibility"""
    df = pd.DataFrame(data)
    df.fillna(method='ffill', inplace=True)
    return df
