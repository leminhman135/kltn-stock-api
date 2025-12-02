"""
Advanced Feature Engineering for Stock Price Prediction
Xây dựng đặc trưng chuyên sâu cho dự đoán giá cổ phiếu

Features:
1. Technical Indicators (Chỉ báo kỹ thuật)
2. Price Patterns (Mẫu hình giá)
3. Volume Analysis (Phân tích khối lượng)
4. Momentum Features (Đặc trưng động lượng)
5. Volatility Features (Đặc trưng biến động)
6. Statistical Features (Đặc trưng thống kê)
7. Time-based Features (Đặc trưng thời gian)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedFeatureEngineer:
    """
    Xây dựng đặc trưng nâng cao cho dự đoán giá cổ phiếu
    
    Các nhóm đặc trưng:
    - Technical: RSI, MACD, Bollinger Bands, Stochastic, ATR, ADX, etc.
    - Price: Returns, Log returns, Price ratios, Gap analysis
    - Volume: Volume ratio, OBV, VWAP, Volume profile
    - Momentum: ROC, MOM, Williams %R, CCI
    - Volatility: Historical volatility, Parkinson, Garman-Klass
    - Statistical: Skewness, Kurtosis, Z-score
    - Time: Day of week, Month, Quarter, Seasonality
    """
    
    def __init__(self):
        self.feature_names = []
        
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo tất cả các đặc trưng
        
        Args:
            df: DataFrame với columns ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame với đầy đủ features
        """
        logger.info("🔧 Creating advanced features...")
        
        result = df.copy()
        
        # 1. Technical Indicators
        result = self._add_technical_indicators(result)
        
        # 2. Price Features
        result = self._add_price_features(result)
        
        # 3. Volume Features
        result = self._add_volume_features(result)
        
        # 4. Momentum Features
        result = self._add_momentum_features(result)
        
        # 5. Volatility Features
        result = self._add_volatility_features(result)
        
        # 6. Statistical Features
        result = self._add_statistical_features(result)
        
        # 7. Time Features
        result = self._add_time_features(result)
        
        # 8. Lagged Features
        result = self._add_lagged_features(result)
        
        # Store feature names
        self.feature_names = [col for col in result.columns if col not in df.columns]
        
        logger.info(f"✅ Created {len(self.feature_names)} features")
        
        return result
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các chỉ báo kỹ thuật"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']
        
        # ==========================================
        # MOVING AVERAGES
        # ==========================================
        
        # Simple Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = close.rolling(window=period).mean()
        
        # Exponential Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'ema_{period}'] = close.ewm(span=period, adjust=False).mean()
        
        # Price to MA ratios
        df['price_to_sma20'] = close / df['sma_20']
        df['price_to_sma50'] = close / df['sma_50']
        df['sma_cross_20_50'] = (df['sma_20'] - df['sma_50']) / df['sma_50']
        
        # ==========================================
        # RSI - Relative Strength Index
        # ==========================================
        
        for period in [7, 14, 21]:
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / (loss + 1e-10)
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # RSI Divergence
        df['rsi_ma'] = df['rsi_14'].rolling(window=5).mean()
        df['rsi_divergence'] = df['rsi_14'] - df['rsi_ma']
        
        # ==========================================
        # MACD - Moving Average Convergence Divergence
        # ==========================================
        
        exp1 = close.ewm(span=12, adjust=False).mean()
        exp2 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_cross'] = np.where(df['macd'] > df['macd_signal'], 1, -1)
        
        # ==========================================
        # BOLLINGER BANDS
        # ==========================================
        
        for period in [20]:
            sma = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            
            df[f'bb_upper_{period}'] = sma + (std * 2)
            df[f'bb_lower_{period}'] = sma - (std * 2)
            df[f'bb_middle_{period}'] = sma
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / sma
            df[f'bb_position_{period}'] = (close - df[f'bb_lower_{period}']) / (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}'] + 1e-10)
        
        # ==========================================
        # STOCHASTIC OSCILLATOR
        # ==========================================
        
        for period in [14]:
            lowest_low = low.rolling(window=period).min()
            highest_high = high.rolling(window=period).max()
            
            df[f'stoch_k_{period}'] = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
            df[f'stoch_d_{period}'] = df[f'stoch_k_{period}'].rolling(window=3).mean()
            df[f'stoch_cross_{period}'] = df[f'stoch_k_{period}'] - df[f'stoch_d_{period}']
        
        # ==========================================
        # ATR - Average True Range
        # ==========================================
        
        for period in [14]:
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f'atr_{period}'] = tr.rolling(window=period).mean()
            df[f'atr_percent_{period}'] = df[f'atr_{period}'] / close * 100
        
        # ==========================================
        # ADX - Average Directional Index
        # ==========================================
        
        period = 14
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
        minus_dm = np.abs(minus_dm.where((minus_dm > plus_dm) & (minus_dm < 0), 0))
        
        tr = pd.concat([high - low, np.abs(high - close.shift()), np.abs(low - close.shift())], axis=1).max(axis=1)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (tr.rolling(window=period).mean() + 1e-10))
        
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx'] = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        df['adx'] = df['adx'].rolling(window=period).mean()
        
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng giá"""
        
        close = df['close']
        open_price = df['open']
        high = df['high']
        low = df['low']
        
        # Returns
        df['return_1d'] = close.pct_change(1)
        df['return_5d'] = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)
        
        # Log returns
        df['log_return_1d'] = np.log(close / close.shift(1))
        df['log_return_5d'] = np.log(close / close.shift(5))
        
        # Price ratios
        df['high_low_ratio'] = high / (low + 1e-10)
        df['close_open_ratio'] = close / (open_price + 1e-10)
        
        # Candlestick features
        df['candle_body'] = close - open_price
        df['candle_body_percent'] = df['candle_body'] / (open_price + 1e-10) * 100
        df['upper_shadow'] = high - pd.concat([close, open_price], axis=1).max(axis=1)
        df['lower_shadow'] = pd.concat([close, open_price], axis=1).min(axis=1) - low
        df['candle_range'] = high - low
        
        # Gap analysis
        df['gap'] = open_price - close.shift(1)
        df['gap_percent'] = df['gap'] / (close.shift(1) + 1e-10) * 100
        
        # Price position in range
        df['price_position'] = (close - low) / (high - low + 1e-10)
        
        # Higher highs, lower lows
        df['higher_high'] = (high > high.shift(1)).astype(int)
        df['lower_low'] = (low < low.shift(1)).astype(int)
        
        # Consecutive up/down days
        df['up_day'] = (close > close.shift(1)).astype(int)
        df['consecutive_up'] = df['up_day'].groupby((df['up_day'] != df['up_day'].shift()).cumsum()).cumsum()
        df['consecutive_down'] = (1 - df['up_day']).groupby(((1 - df['up_day']) != (1 - df['up_day']).shift()).cumsum()).cumsum()
        
        return df
    
    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng khối lượng"""
        
        close = df['close']
        volume = df['volume']
        
        # Volume Moving Averages
        for period in [5, 10, 20]:
            df[f'volume_sma_{period}'] = volume.rolling(window=period).mean()
        
        # Volume Ratio
        df['volume_ratio_5'] = volume / (df['volume_sma_5'] + 1e-10)
        df['volume_ratio_20'] = volume / (df['volume_sma_20'] + 1e-10)
        
        # Volume Change
        df['volume_change'] = volume.pct_change()
        df['volume_change_5d'] = volume.pct_change(5)
        
        # On-Balance Volume (OBV)
        df['obv'] = (np.sign(close.diff()) * volume).cumsum()
        df['obv_sma_10'] = df['obv'].rolling(window=10).mean()
        df['obv_ratio'] = df['obv'] / (df['obv_sma_10'] + 1e-10)
        
        # Volume Price Trend (VPT)
        df['vpt'] = (volume * close.pct_change()).cumsum()
        
        # Money Flow
        typical_price = (df['high'] + df['low'] + close) / 3
        money_flow = typical_price * volume
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        df['mfi_14'] = 100 - (100 / (1 + positive_flow.rolling(14).sum() / (negative_flow.rolling(14).sum() + 1e-10)))
        
        # VWAP - Volume Weighted Average Price
        df['vwap'] = (volume * (df['high'] + df['low'] + close) / 3).cumsum() / (volume.cumsum() + 1e-10)
        df['price_to_vwap'] = close / (df['vwap'] + 1e-10)
        
        # Volume Spike
        df['volume_spike'] = (volume > df['volume_sma_20'] * 2).astype(int)
        
        # Accumulation/Distribution
        clv = ((close - df['low']) - (df['high'] - close)) / (df['high'] - df['low'] + 1e-10)
        df['ad_line'] = (clv * volume).cumsum()
        
        return df
    
    def _add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng động lượng"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = (close - close.shift(period)) / (close.shift(period) + 1e-10) * 100
        
        # Momentum
        for period in [10, 20]:
            df[f'momentum_{period}'] = close - close.shift(period)
        
        # Williams %R
        for period in [14]:
            highest_high = high.rolling(window=period).max()
            lowest_low = low.rolling(window=period).min()
            df[f'williams_r_{period}'] = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
        
        # Commodity Channel Index (CCI)
        for period in [20]:
            typical_price = (high + low + close) / 3
            sma_tp = typical_price.rolling(window=period).mean()
            mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
            df[f'cci_{period}'] = (typical_price - sma_tp) / (0.015 * mad + 1e-10)
        
        # Ultimate Oscillator
        bp = close - pd.concat([low, close.shift(1)], axis=1).min(axis=1)
        tr = pd.concat([high - low, np.abs(high - close.shift(1)), np.abs(low - close.shift(1))], axis=1).max(axis=1)
        
        avg7 = bp.rolling(7).sum() / (tr.rolling(7).sum() + 1e-10)
        avg14 = bp.rolling(14).sum() / (tr.rolling(14).sum() + 1e-10)
        avg28 = bp.rolling(28).sum() / (tr.rolling(28).sum() + 1e-10)
        
        df['ultimate_oscillator'] = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
        
        # Awesome Oscillator
        df['ao'] = (high + low).rolling(5).mean() / 2 - (high + low).rolling(34).mean() / 2
        
        return df
    
    def _add_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng biến động"""
        
        close = df['close']
        high = df['high']
        low = df['low']
        open_price = df['open']
        
        # Historical Volatility
        log_returns = np.log(close / close.shift(1))
        
        for period in [5, 10, 20, 30]:
            df[f'volatility_{period}d'] = log_returns.rolling(window=period).std() * np.sqrt(252)
        
        # Parkinson Volatility (using high-low range)
        for period in [20]:
            hl_log = np.log(high / low)
            df[f'parkinson_vol_{period}'] = np.sqrt((1 / (4 * np.log(2))) * (hl_log ** 2).rolling(period).mean()) * np.sqrt(252)
        
        # Garman-Klass Volatility
        for period in [20]:
            hl = np.log(high / low) ** 2
            co = np.log(close / open_price) ** 2
            gk = 0.5 * hl - (2 * np.log(2) - 1) * co
            df[f'gk_vol_{period}'] = np.sqrt(gk.rolling(period).mean()) * np.sqrt(252)
        
        # Volatility Ratio
        df['volatility_ratio'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-10)
        
        # Volatility Trend
        df['volatility_trend'] = df['volatility_20d'] - df['volatility_20d'].shift(5)
        
        # Range Percent
        df['range_percent'] = (high - low) / (close + 1e-10) * 100
        df['range_percent_sma'] = df['range_percent'].rolling(10).mean()
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng thống kê"""
        
        close = df['close']
        returns = close.pct_change()
        
        # Rolling statistics
        for period in [20]:
            # Mean
            df[f'return_mean_{period}'] = returns.rolling(window=period).mean()
            
            # Std
            df[f'return_std_{period}'] = returns.rolling(window=period).std()
            
            # Skewness
            df[f'skewness_{period}'] = returns.rolling(window=period).skew()
            
            # Kurtosis
            df[f'kurtosis_{period}'] = returns.rolling(window=period).kurt()
            
            # Z-score
            mean = close.rolling(window=period).mean()
            std = close.rolling(window=period).std()
            df[f'zscore_{period}'] = (close - mean) / (std + 1e-10)
        
        # Percentile rank
        df['percentile_rank_20'] = close.rolling(20).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        
        # Distance from 52-week high/low (using 252 trading days)
        if len(df) >= 252:
            df['dist_52w_high'] = (close - close.rolling(252).max()) / (close.rolling(252).max() + 1e-10)
            df['dist_52w_low'] = (close - close.rolling(252).min()) / (close.rolling(252).min() + 1e-10)
        else:
            max_period = min(len(df), 252)
            df['dist_52w_high'] = (close - close.rolling(max_period).max()) / (close.rolling(max_period).max() + 1e-10)
            df['dist_52w_low'] = (close - close.rolling(max_period).min()) / (close.rolling(max_period).min() + 1e-10)
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng thời gian"""
        
        if isinstance(df.index, pd.DatetimeIndex):
            date_index = df.index
        else:
            # Try to convert index to datetime
            try:
                date_index = pd.to_datetime(df.index)
            except:
                return df
        
        # Day of week (0 = Monday, 4 = Friday)
        df['day_of_week'] = date_index.dayofweek
        
        # Is Monday/Friday
        df['is_monday'] = (date_index.dayofweek == 0).astype(int)
        df['is_friday'] = (date_index.dayofweek == 4).astype(int)
        
        # Month
        df['month'] = date_index.month
        
        # Quarter
        df['quarter'] = date_index.quarter
        
        # Day of month
        df['day_of_month'] = date_index.day
        
        # Is month start/end
        df['is_month_start'] = date_index.is_month_start.astype(int)
        df['is_month_end'] = date_index.is_month_end.astype(int)
        
        # Week of year
        df['week_of_year'] = date_index.isocalendar().week.values
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Cyclical encoding for day of week
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 5)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 5)
        
        return df
    
    def _add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm các đặc trưng trễ (lagged features)"""
        
        # Lagged returns
        for lag in [1, 2, 3, 5, 10]:
            df[f'return_lag_{lag}'] = df['return_1d'].shift(lag)
        
        # Lagged volume ratio
        for lag in [1, 2, 3]:
            df[f'volume_ratio_lag_{lag}'] = df['volume_ratio_5'].shift(lag)
        
        # Lagged RSI
        for lag in [1, 2]:
            df[f'rsi_14_lag_{lag}'] = df['rsi_14'].shift(lag)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Nhóm features theo category"""
        
        groups = {
            'trend': [f for f in self.feature_names if any(x in f.lower() for x in ['sma', 'ema', 'macd', 'adx'])],
            'momentum': [f for f in self.feature_names if any(x in f.lower() for x in ['rsi', 'roc', 'momentum', 'stoch', 'cci', 'williams'])],
            'volatility': [f for f in self.feature_names if any(x in f.lower() for x in ['volatility', 'atr', 'bb_', 'range'])],
            'volume': [f for f in self.feature_names if any(x in f.lower() for x in ['volume', 'obv', 'vwap', 'mfi', 'ad_'])],
            'price': [f for f in self.feature_names if any(x in f.lower() for x in ['return', 'gap', 'candle', 'shadow', 'position'])],
            'statistical': [f for f in self.feature_names if any(x in f.lower() for x in ['skew', 'kurt', 'zscore', 'percentile'])],
            'time': [f for f in self.feature_names if any(x in f.lower() for x in ['day_', 'month', 'quarter', 'week', 'sin', 'cos'])],
        }
        
        return groups
    
    def select_top_features(self, df: pd.DataFrame, target_col: str = 'close', 
                           n_features: int = 50) -> List[str]:
        """
        Chọn top N features dựa trên correlation với target
        """
        from scipy import stats
        
        correlations = {}
        
        for col in self.feature_names:
            if col in df.columns and col != target_col:
                # Remove NaN
                valid_mask = ~(df[col].isna() | df[target_col].isna())
                if valid_mask.sum() > 30:
                    corr, _ = stats.pearsonr(df.loc[valid_mask, col], df.loc[valid_mask, target_col])
                    correlations[col] = abs(corr)
        
        # Sort by absolute correlation
        sorted_features = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        return [f[0] for f in sorted_features[:n_features]]


# Convenience function
def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Quick function to create all features"""
    engineer = AdvancedFeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    # Test
    import numpy as np
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    n = len(dates)
    
    np.random.seed(42)
    df = pd.DataFrame({
        'open': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'high': 0,
        'low': 0,
        'close': 100 + np.cumsum(np.random.randn(n) * 0.5),
        'volume': np.random.randint(1000000, 10000000, n)
    }, index=dates)
    
    df['high'] = df[['open', 'close']].max(axis=1) + np.random.rand(n) * 2
    df['low'] = df[['open', 'close']].min(axis=1) - np.random.rand(n) * 2
    
    # Create features
    engineer = AdvancedFeatureEngineer()
    df_features = engineer.create_all_features(df)
    
    print(f"\n📊 Total features created: {len(engineer.feature_names)}")
    print(f"\n📋 Feature groups:")
    
    groups = engineer.get_feature_importance_groups()
    for group_name, features in groups.items():
        print(f"  {group_name}: {len(features)} features")
    
    print(f"\n✅ Sample features:")
    print(df_features[['close', 'sma_20', 'rsi_14', 'macd', 'volatility_20d']].tail())
