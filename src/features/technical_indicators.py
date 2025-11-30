"""
Module tính toán các chỉ báo kỹ thuật (Technical Indicators)
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Tính toán các chỉ báo kỹ thuật phổ biến"""
    
    @staticmethod
    def calculate_sma(df: pd.DataFrame, column: str = 'close', 
                      window: int = 20) -> pd.Series:
        """
        Simple Moving Average (SMA)
        
        Args:
            df: DataFrame chứa dữ liệu
            column: Tên cột để tính (mặc định 'close')
            window: Số ngày cho moving average
        
        Returns:
            Series chứa giá trị SMA
        """
        return df[column].rolling(window=window).mean()
    
    @staticmethod
    def calculate_ema(df: pd.DataFrame, column: str = 'close', 
                      window: int = 20) -> pd.Series:
        """
        Exponential Moving Average (EMA)
        Trọng số cao hơn cho dữ liệu gần đây
        """
        return df[column].ewm(span=window, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(df: pd.DataFrame, column: str = 'close', 
                     window: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        Dao động từ 0-100, RSI > 70: overbought, RSI < 30: oversold
        
        Formula:
        RSI = 100 - (100 / (1 + RS))
        RS = Average Gain / Average Loss
        """
        delta = df[column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame, column: str = 'close',
                      fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """
        Moving Average Convergence Divergence (MACD)
        
        Returns:
            DataFrame với các cột: macd, macd_signal, macd_histogram
        """
        ema_fast = df[column].ewm(span=fast, adjust=False).mean()
        ema_slow = df[column].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return pd.DataFrame({
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        })
    
    @staticmethod
    def calculate_bollinger_bands(df: pd.DataFrame, column: str = 'close',
                                 window: int = 20, num_std: float = 2) -> pd.DataFrame:
        """
        Bollinger Bands
        
        Returns:
            DataFrame với các cột: bb_middle, bb_upper, bb_lower
        """
        sma = df[column].rolling(window=window).mean()
        std = df[column].rolling(window=window).std()
        
        bb_upper = sma + (std * num_std)
        bb_lower = sma - (std * num_std)
        
        return pd.DataFrame({
            'bb_middle': sma,
            'bb_upper': bb_upper,
            'bb_lower': bb_lower
        })
    
    @staticmethod
    def calculate_stochastic_oscillator(df: pd.DataFrame, window: int = 14,
                                       smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
        """
        Stochastic Oscillator
        Dao động từ 0-100
        
        Returns:
            DataFrame với các cột: stoch_k, stoch_d
        """
        low_min = df['low'].rolling(window=window).min()
        high_max = df['high'].rolling(window=window).max()
        
        stoch_k = 100 * ((df['close'] - low_min) / (high_max - low_min))
        stoch_k = stoch_k.rolling(window=smooth_k).mean()
        stoch_d = stoch_k.rolling(window=smooth_d).mean()
        
        return pd.DataFrame({
            'stoch_k': stoch_k,
            'stoch_d': stoch_d
        })
    
    @staticmethod
    def calculate_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        Đo lường độ biến động của thị trường
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=window).mean()
        
        return atr
    
    @staticmethod
    def calculate_obv(df: pd.DataFrame) -> pd.Series:
        """
        On-Balance Volume (OBV)
        Tích lũy khối lượng dựa trên hướng giá
        """
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        return obv
    
    @staticmethod
    def calculate_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        """
        Average Directional Index (ADX)
        Đo lường sức mạnh của xu hướng
        
        Returns:
            DataFrame với các cột: adx, plus_di, minus_di
        """
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = TechnicalIndicators.calculate_atr(df, window=1)
        
        plus_di = 100 * (plus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
        minus_di = 100 * (minus_dm.rolling(window=window).mean() / tr.rolling(window=window).mean())
        
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=window).mean()
        
        return pd.DataFrame({
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di
        })
    
    @staticmethod
    def calculate_cci(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        Đo lường độ lệch của giá so với trung bình
        """
        tp = (df['high'] + df['low'] + df['close']) / 3  # Typical Price
        sma_tp = tp.rolling(window=window).mean()
        mad = tp.rolling(window=window).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        
        return cci
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Williams %R
        Dao động từ -100 đến 0
        """
        highest_high = df['high'].rolling(window=window).max()
        lowest_low = df['low'].rolling(window=window).min()
        
        williams_r = -100 * ((highest_high - df['close']) / (highest_high - lowest_low))
        
        return williams_r
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame, column: str = 'close', 
                          window: int = 10) -> pd.Series:
        """
        Momentum
        Tốc độ thay đổi giá
        """
        return df[column].diff(window)
    
    @staticmethod
    def calculate_roc(df: pd.DataFrame, column: str = 'close', 
                     window: int = 12) -> pd.Series:
        """
        Rate of Change (ROC)
        Phần trăm thay đổi giá
        """
        return ((df[column] - df[column].shift(window)) / df[column].shift(window)) * 100
    
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm tất cả chỉ báo kỹ thuật vào DataFrame
        
        Args:
            df: DataFrame với các cột: open, high, low, close, volume
        
        Returns:
            DataFrame với tất cả chỉ báo kỹ thuật
        """
        df_with_indicators = df.copy()
        
        try:
            # Moving Averages
            df_with_indicators['sma_5'] = self.calculate_sma(df, window=5)
            df_with_indicators['sma_10'] = self.calculate_sma(df, window=10)
            df_with_indicators['sma_20'] = self.calculate_sma(df, window=20)
            df_with_indicators['sma_50'] = self.calculate_sma(df, window=50)
            df_with_indicators['sma_200'] = self.calculate_sma(df, window=200)
            
            df_with_indicators['ema_12'] = self.calculate_ema(df, window=12)
            df_with_indicators['ema_26'] = self.calculate_ema(df, window=26)
            
            # RSI
            df_with_indicators['rsi'] = self.calculate_rsi(df)
            
            # MACD
            macd_df = self.calculate_macd(df)
            df_with_indicators = pd.concat([df_with_indicators, macd_df], axis=1)
            
            # Bollinger Bands
            bb_df = self.calculate_bollinger_bands(df)
            df_with_indicators = pd.concat([df_with_indicators, bb_df], axis=1)
            
            # Stochastic Oscillator
            if 'high' in df.columns and 'low' in df.columns:
                stoch_df = self.calculate_stochastic_oscillator(df)
                df_with_indicators = pd.concat([df_with_indicators, stoch_df], axis=1)
            
            # ATR
            if 'high' in df.columns and 'low' in df.columns:
                df_with_indicators['atr'] = self.calculate_atr(df)
            
            # OBV
            if 'volume' in df.columns:
                df_with_indicators['obv'] = self.calculate_obv(df)
            
            # ADX
            if 'high' in df.columns and 'low' in df.columns:
                adx_df = self.calculate_adx(df)
                df_with_indicators = pd.concat([df_with_indicators, adx_df], axis=1)
            
            # CCI
            if 'high' in df.columns and 'low' in df.columns:
                df_with_indicators['cci'] = self.calculate_cci(df)
            
            # Williams %R
            if 'high' in df.columns and 'low' in df.columns:
                df_with_indicators['williams_r'] = self.calculate_williams_r(df)
            
            # Momentum & ROC
            df_with_indicators['momentum'] = self.calculate_momentum(df)
            df_with_indicators['roc'] = self.calculate_roc(df)
            
            logger.info(f"Added {len(df_with_indicators.columns) - len(df.columns)} technical indicators")
            
        except Exception as e:
            logger.error(f"Error calculating indicators: {str(e)}")
        
        return df_with_indicators


if __name__ == "__main__":
    # Test với dữ liệu mẫu
    dates = pd.date_range('2023-01-01', periods=200)
    df_test = pd.DataFrame({
        'date': dates,
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 105,
        'low': np.random.randn(200).cumsum() + 95,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000000, 10000000, 200)
    })
    
    ti = TechnicalIndicators()
    df_with_indicators = ti.add_all_indicators(df_test)
    
    print(f"\nOriginal columns: {len(df_test.columns)}")
    print(f"With indicators: {len(df_with_indicators.columns)}")
    print(f"\nNew indicators added: {df_with_indicators.columns.tolist()[len(df_test.columns):]}")
