# src/model.py
"""
Stock Price Prediction using Machine Learning
Includes: Linear Regression, Random Forest, Gradient Boosting
With: RSI, MACD, Bollinger Bands, Mean Reversion
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try import sklearn
try:
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split, TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    HAS_SKLEARN = True
    logger.info("✅ scikit-learn loaded successfully")
except ImportError:
    HAS_SKLEARN = False
    logger.warning("⚠️ scikit-learn not installed. Using simple prediction.")


class StockMLModel:
    """Machine Learning model for stock price prediction"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.models = {}
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.metrics = {}
        self.is_trained = False
        self.feature_names = []
    
    def calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices[-(period+1):])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, prices: np.ndarray) -> Tuple[float, float]:
        """Calculate MACD and Signal line"""
        if len(prices) < 26:
            return 0.0, 0.0
        
        def ema(data, span):
            alpha = 2 / (span + 1)
            result = data[0]
            for price in data[1:]:
                result = alpha * price + (1 - alpha) * result
            return result
        
        ema12 = ema(prices[-26:], 12)
        ema26 = ema(prices[-26:], 26)
        macd = ema12 - ema26
        # Normalize by price
        return macd / prices[-1], 0.0
    
    def calculate_bollinger_position(self, prices: np.ndarray, period: int = 20) -> float:
        """Calculate position within Bollinger Bands (-1 to 1)"""
        if len(prices) < period:
            return 0.0
        recent = prices[-period:]
        sma = np.mean(recent)
        std = np.std(recent)
        if std == 0:
            return 0.0
        upper = sma + 2 * std
        lower = sma - 2 * std
        current = prices[-1]
        # Normalize to -1 (below lower) to 1 (above upper)
        position = (current - sma) / (2 * std)
        return np.clip(position, -1, 1)
    
    def create_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Create features from OHLCV data with technical indicators"""
        if len(df) < 30:
            raise ValueError("Need at least 30 data points")
        
        data = df.copy()
        data = data.sort_values('date').reset_index(drop=True)
        
        close = data['close'].values
        high = data['high'].values if 'high' in data.columns else close
        low = data['low'].values if 'low' in data.columns else close
        volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
        
        features = []
        targets = []
        
        self.feature_names = [
            'lag_1', 'lag_2', 'lag_3', 'lag_5', 'lag_10',
            'return_1d', 'return_3d', 'return_5d', 'return_10d',
            'sma5_ratio', 'sma10_ratio', 'sma20_ratio',
            'vol_5d', 'vol_10d', 'vol_20d',
            'volume_ratio',
            'price_position_20d',
            'rsi_14',
            'macd_norm',
            'bollinger_pos',
            'high_low_range',
            'distance_from_high',
            'distance_from_low',
            'mean_reversion_signal'
        ]
        
        for i in range(25, len(close) - 1):
            row = []
            current = close[i]
            
            # Lagged prices (normalized) - trend indicators
            row.append(close[i-1] / current)
            row.append(close[i-2] / current)
            row.append(close[i-3] / current)
            row.append(close[i-5] / current)
            row.append(close[i-10] / current)
            
            # Returns - momentum indicators
            row.append((close[i] - close[i-1]) / close[i-1])  # 1-day return
            row.append((close[i] - close[i-3]) / close[i-3])  # 3-day return
            row.append((close[i] - close[i-5]) / close[i-5])  # 5-day return
            row.append((close[i] - close[i-10]) / close[i-10])  # 10-day return
            
            # Moving averages (normalized) - trend/support levels
            sma5 = np.mean(close[i-4:i+1])
            sma10 = np.mean(close[i-9:i+1])
            sma20 = np.mean(close[i-19:i+1])
            row.append(sma5 / current)
            row.append(sma10 / current)
            row.append(sma20 / current)
            
            # Volatility at different timeframes
            row.append(np.std(close[i-4:i+1]) / current)
            row.append(np.std(close[i-9:i+1]) / current)
            row.append(np.std(close[i-19:i+1]) / current)
            
            # Volume ratio
            avg_vol = np.mean(volume[i-9:i+1])
            row.append(volume[i] / avg_vol if avg_vol > 0 else 1.0)
            
            # Price position in 20-day range
            high_20 = np.max(close[i-19:i+1])
            low_20 = np.min(close[i-19:i+1])
            row.append((current - low_20) / (high_20 - low_20 + 0.001))
            
            # RSI - overbought/oversold indicator (important for reversal)
            rsi = self.calculate_rsi(close[:i+1])
            row.append((rsi - 50) / 50)  # Normalize to -1 to 1
            
            # MACD - momentum/trend indicator
            macd_norm, _ = self.calculate_macd(close[:i+1])
            row.append(macd_norm)
            
            # Bollinger Bands position - mean reversion indicator
            bb_pos = self.calculate_bollinger_position(close[:i+1])
            row.append(bb_pos)
            
            # High-Low range (volatility measure)
            row.append((high[i] - low[i]) / current if high[i] != low[i] else 0)
            
            # Distance from recent high/low (reversal signals)
            recent_high = np.max(high[i-9:i+1])
            recent_low = np.min(low[i-9:i+1])
            row.append((recent_high - current) / current)  # Distance from high (potential upside)
            row.append((current - recent_low) / current)   # Distance from low (potential downside)
            
            # Mean reversion signal: price far from SMA20 tends to revert
            mean_reversion = (sma20 - current) / current
            row.append(mean_reversion)
            
            features.append(row)
            # Target: next day's return
            targets.append((close[i+1] - close[i]) / close[i])
        
        return np.array(features), np.array(targets)
    
    def train(self, df: pd.DataFrame) -> Dict:
        """Train ML models with TimeSeriesSplit for better validation"""
        if not HAS_SKLEARN:
            return {'success': False, 'error': 'scikit-learn not installed'}
        
        try:
            logger.info(f"🔄 Training models for {self.symbol}...")
            
            X, y = self.create_features(df)
            
            if len(X) < 60:
                return {'success': False, 'error': 'Not enough data (need 60+ points)'}
            
            # Use TimeSeriesSplit for proper time-series validation
            tscv = TimeSeriesSplit(n_splits=3)
            
            results = {}
            model_configs = {
                'linear': Ridge(alpha=10.0),  # Higher regularization
                'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
                'random_forest': RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=6,  # Reduced depth to prevent overfitting
                    min_samples_leaf=5,
                    random_state=42, 
                    n_jobs=-1
                ),
                'gradient_boost': GradientBoostingRegressor(
                    n_estimators=100, 
                    max_depth=3,  # Shallow trees
                    learning_rate=0.05,  # Lower learning rate
                    subsample=0.8,
                    random_state=42
                )
            }
            
            # Calculate baseline metrics
            baseline_preds = np.zeros(len(y))  # Predict no change
            baseline_acc = np.sum((baseline_preds >= 0) == (y >= 0)) / len(y)
            logger.info(f"  📊 Baseline accuracy (predict no change): {baseline_acc*100:.1f}%")
            
            for name, model in model_configs.items():
                try:
                    # Cross-validation scores
                    cv_scores = []
                    cv_direction_acc = []
                    
                    for train_idx, test_idx in tscv.split(X):
                        X_train_cv, X_test_cv = X[train_idx], X[test_idx]
                        y_train_cv, y_test_cv = y[train_idx], y[test_idx]
                        
                        scaler_cv = StandardScaler()
                        X_train_scaled = scaler_cv.fit_transform(X_train_cv)
                        X_test_scaled = scaler_cv.transform(X_test_cv)
                        
                        model_cv = type(model)(**model.get_params())
                        model_cv.fit(X_train_scaled, y_train_cv)
                        y_pred_cv = model_cv.predict(X_test_scaled)
                        
                        cv_scores.append(r2_score(y_test_cv, y_pred_cv))
                        cv_direction_acc.append(np.mean((y_pred_cv >= 0) == (y_test_cv >= 0)))
                    
                    # Final train on all data
                    X_scaled = self.scaler.fit_transform(X)
                    model.fit(X_scaled, y)
                    
                    # Metrics from cross-validation
                    avg_r2 = np.mean(cv_scores)
                    avg_direction_acc = np.mean(cv_direction_acc)
                    
                    # Only accept model if better than baseline
                    if avg_direction_acc > baseline_acc * 0.95:  # At least 95% of baseline
                        self.models[name] = model
                        self.metrics[name] = {
                            'r2': round(max(0, avg_r2), 3),
                            'accuracy': round(avg_direction_acc * 100, 1),
                            'vs_baseline': round((avg_direction_acc / baseline_acc - 1) * 100, 1)
                        }
                        results[name] = {'trained': True, 'metrics': self.metrics[name]}
                        logger.info(f"  ✅ {name}: Accuracy={self.metrics[name]['accuracy']}% (vs baseline: {self.metrics[name]['vs_baseline']:+.1f}%)")
                    else:
                        results[name] = {'trained': False, 'reason': 'worse than baseline'}
                        logger.info(f"  ❌ {name}: Rejected (accuracy={avg_direction_acc*100:.1f}% < baseline)")
                        
                except Exception as e:
                    results[name] = {'trained': False, 'error': str(e)}
            
            self.is_trained = len(self.models) > 0
            
            # Store baseline for reference
            self.metrics['baseline'] = {'accuracy': round(baseline_acc * 100, 1)}
            
            return {'success': True, 'symbol': self.symbol, 'data_points': len(df), 'models': results, 'baseline_accuracy': round(baseline_acc * 100, 1)}
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return {'success': False, 'error': str(e)}
    
    def predict(self, df: pd.DataFrame, steps: int = 7, model_type: str = 'ensemble') -> Dict:
        """Make predictions with mean reversion and uncertainty"""
        if not self.is_trained:
            return quick_predict(df['close'].tolist(), steps)
        
        try:
            data = df.copy().sort_values('date').reset_index(drop=True)
            close = data['close'].values
            high = data['high'].values if 'high' in data.columns else close
            low = data['low'].values if 'low' in data.columns else close
            volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
            
            last_price = float(close[-1])
            sma20 = np.mean(close[-20:])
            
            # Calculate current market condition indicators
            current_rsi = self.calculate_rsi(close)
            current_bb = self.calculate_bollinger_position(close)
            
            predictions = []
            prediction_ranges = []
            current_price = last_price
            
            # Volatility for uncertainty bounds
            returns = np.diff(close[-30:]) / close[-31:-1]
            daily_vol = np.std(returns)
            
            for step in range(steps):
                # Get fresh features for current state
                temp_close = np.append(close, current_price) if step > 0 else close
                temp_high = np.append(high, current_price) if step > 0 else high
                temp_low = np.append(low, current_price) if step > 0 else low
                temp_volume = np.append(volume, volume[-1]) if step > 0 else volume
                
                # Create single feature row for prediction
                feature_row = self._create_single_feature(temp_close, temp_high, temp_low, temp_volume)
                
                if feature_row is None:
                    break
                
                feature_scaled = self.scaler.transform(feature_row.reshape(1, -1))
                
                if model_type == 'ensemble' and self.models:
                    pred_returns = []
                    weights = []
                    for name, model in self.models.items():
                        pred_return = model.predict(feature_scaled)[0]
                        # Clip extreme predictions - much tighter bounds
                        pred_return = np.clip(pred_return, -0.015, 0.015)  # Max 1.5% per day
                        acc = self.metrics.get(name, {}).get('accuracy', 50)
                        pred_returns.append(pred_return)
                        weights.append(max(acc - 45, 1))  # Weight by excess accuracy over random
                    
                    if sum(weights) > 0:
                        weighted_return = sum(r * w for r, w in zip(pred_returns, weights)) / sum(weights)
                    else:
                        weighted_return = np.mean(pred_returns)
                elif model_type in self.models:
                    weighted_return = self.models[model_type].predict(feature_scaled)[0]
                    weighted_return = np.clip(weighted_return, -0.015, 0.015)
                else:
                    weighted_return = 0.0
                
                # Apply mean reversion pressure - STRONGER
                # If price is far above SMA20, reduce upward predictions
                price_vs_sma = (current_price - sma20) / sma20
                mean_reversion_factor = 1.0
                
                if price_vs_sma > 0.02 and weighted_return > 0:  # Price 2%+ above SMA, predicting up
                    mean_reversion_factor = max(0.1, 1 - abs(price_vs_sma) * 5)
                elif price_vs_sma < -0.02 and weighted_return < 0:  # Price 2%+ below SMA, predicting down
                    mean_reversion_factor = max(0.1, 1 - abs(price_vs_sma) * 5)
                
                # Strong mean reversion: pull back toward SMA
                if abs(price_vs_sma) > 0.03:
                    # Add mean reversion component
                    reversion_pull = -price_vs_sma * 0.15  # Pull 15% of the gap per day
                    weighted_return = weighted_return * 0.5 + reversion_pull * 0.5
                
                # Apply RSI-based adjustment (overbought/oversold) - STRONGER
                rsi_adjustment = 1.0
                if current_rsi > 65 and weighted_return > 0:  # Overbought, reduce upward
                    rsi_adjustment = max(0.2, 1 - (current_rsi - 50) / 100)
                elif current_rsi < 35 and weighted_return < 0:  # Oversold, reduce downward
                    rsi_adjustment = max(0.2, 1 - (50 - current_rsi) / 100)
                
                # Apply Bollinger adjustment - STRONGER
                bb_adjustment = 1.0
                if current_bb > 0.5 and weighted_return > 0:  # Near upper band
                    bb_adjustment = max(0.3, 1 - current_bb)
                elif current_bb < -0.5 and weighted_return < 0:  # Near lower band
                    bb_adjustment = max(0.3, 1 + current_bb)
                
                # Final adjusted return
                adjusted_return = weighted_return * mean_reversion_factor * rsi_adjustment * bb_adjustment
                
                # Add decay for longer predictions (less confident further out)
                decay = 0.7 ** step  # Stronger decay
                adjusted_return *= decay
                
                # Final clip - max 1% move per day after all adjustments
                adjusted_return = np.clip(adjusted_return, -0.01, 0.01)
                
                next_price = current_price * (1 + adjusted_return)
                
                # HARD LIMIT: Giới hạn tổng mức thay đổi tối đa ±15% so với giá gốc
                max_total_change = 0.15  # 15%
                max_price = last_price * (1 + max_total_change)
                min_price = last_price * (1 - max_total_change)
                next_price = np.clip(next_price, min_price, max_price)
                
                predictions.append(round(float(next_price), 2))
                
                # Calculate prediction range (confidence interval)
                uncertainty = daily_vol * np.sqrt(step + 1) * current_price * 1.96  # 95% CI
                pred_low = max(min_price, next_price - uncertainty)
                pred_high = min(max_price, next_price + uncertainty)
                prediction_ranges.append({
                    'low': round(pred_low, 2),
                    'high': round(pred_high, 2)
                })
                
                current_price = next_price
            
            # Generate dates (skip weekends)
            dates = []
            current_date = datetime.now()
            while len(dates) < steps:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:
                    dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Calculate confidence based on model performance
            avg_accuracy = np.mean([m.get('accuracy', 50) for m in self.metrics.values() if 'accuracy' in m])
            baseline_acc = self.metrics.get('baseline', {}).get('accuracy', 50)
            
            # Confidence = how much better than random
            confidence = min(0.9, max(0.3, (avg_accuracy - 45) / 50))
            
            # Determine trend
            if len(predictions) > 0:
                total_change = (predictions[-1] - last_price) / last_price
                if total_change > 0.02:
                    trend = 'up'
                elif total_change < -0.02:
                    trend = 'down'
                else:
                    trend = 'sideways'
            else:
                trend = 'unknown'
            
            return {
                'symbol': self.symbol,
                'predictions': predictions,
                'prediction_ranges': prediction_ranges,
                'dates': dates,
                'model': model_type,
                'confidence': round(confidence, 2),
                'trend': trend,
                'last_price': last_price,
                'sma20': round(sma20, 2),
                'rsi': round(current_rsi, 1),
                'bollinger_position': round(current_bb, 2),
                'metrics': {k: v for k, v in self.metrics.items() if k != 'baseline'},
                'baseline_accuracy': baseline_acc
            }
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return quick_predict(df['close'].tolist(), steps)
    
    def _create_single_feature(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> np.ndarray:
        """Create feature row for single prediction"""
        if len(close) < 26:
            return None
        
        i = len(close) - 1
        current = close[i]
        row = []
        
        # Lagged prices
        row.append(close[i-1] / current)
        row.append(close[i-2] / current)
        row.append(close[i-3] / current)
        row.append(close[i-5] / current)
        row.append(close[i-10] / current)
        
        # Returns
        row.append((close[i] - close[i-1]) / close[i-1])
        row.append((close[i] - close[i-3]) / close[i-3])
        row.append((close[i] - close[i-5]) / close[i-5])
        row.append((close[i] - close[i-10]) / close[i-10])
        
        # SMAs
        sma5 = np.mean(close[i-4:i+1])
        sma10 = np.mean(close[i-9:i+1])
        sma20 = np.mean(close[i-19:i+1])
        row.append(sma5 / current)
        row.append(sma10 / current)
        row.append(sma20 / current)
        
        # Volatility
        row.append(np.std(close[i-4:i+1]) / current)
        row.append(np.std(close[i-9:i+1]) / current)
        row.append(np.std(close[i-19:i+1]) / current)
        
        # Volume
        avg_vol = np.mean(volume[i-9:i+1])
        row.append(volume[i] / avg_vol if avg_vol > 0 else 1.0)
        
        # Price position
        high_20 = np.max(close[i-19:i+1])
        low_20 = np.min(close[i-19:i+1])
        row.append((current - low_20) / (high_20 - low_20 + 0.001))
        
        # RSI
        rsi = self.calculate_rsi(close)
        row.append((rsi - 50) / 50)
        
        # MACD
        macd_norm, _ = self.calculate_macd(close)
        row.append(macd_norm)
        
        # Bollinger
        bb_pos = self.calculate_bollinger_position(close)
        row.append(bb_pos)
        
        # High-Low range
        row.append((high[i] - low[i]) / current if high[i] != low[i] else 0)
        
        # Distance from high/low
        recent_high = np.max(high[i-9:i+1])
        recent_low = np.min(low[i-9:i+1])
        row.append((recent_high - current) / current)
        row.append((current - recent_low) / current)
        
        # Mean reversion
        row.append((sma20 - current) / current)
        
        return np.array(row)
    
    def get_metrics(self) -> Dict:
        return {'symbol': self.symbol, 'is_trained': self.is_trained, 'models': self.metrics}


def quick_predict(prices: list, steps: int = 7) -> dict:
    """Quick prediction using technical analysis (fallback) - balanced approach"""
    if len(prices) < 5:
        return {'predictions': [], 'error': 'Need at least 5 price points'}
    
    prices = np.array(prices, dtype=float)
    last_price = float(prices[-1])
    
    # Calculate indicators
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    
    # Trend (with decay for mean reversion)
    recent = prices[-10:] if len(prices) >= 10 else prices
    short_trend = (recent[-1] - recent[0]) / (len(recent) * recent[0]) if len(recent) > 1 else 0
    
    # Moving averages
    sma5 = np.mean(prices[-5:])
    sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    
    # RSI-like momentum
    if len(returns) >= 14:
        gains = returns[-14:][returns[-14:] > 0]
        losses = -returns[-14:][returns[-14:] < 0]
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0.001
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    else:
        rsi = 50
    
    # Price position relative to range
    high_20 = np.max(prices[-20:]) if len(prices) >= 20 else np.max(prices)
    low_20 = np.min(prices[-20:]) if len(prices) >= 20 else np.min(prices)
    price_position = (last_price - low_20) / (high_20 - low_20 + 0.001)
    
    predictions = []
    current_price = last_price
    
    # HARD LIMIT: Maximum total change ±15%
    max_total_change = 0.15
    max_price = last_price * (1 + max_total_change)
    min_price = last_price * (1 - max_total_change)
    
    for i in range(steps):
        # Base prediction from trend - much weaker
        trend_component = short_trend * (0.5 ** i) * 0.3  # Weak trend, strong decay
        
        # Mean reversion component (pull toward SMA20) - this is key
        price_vs_sma = (current_price - sma20) / sma20
        mean_reversion = -price_vs_sma * 0.1  # Pull 10% of gap toward SMA
        
        # RSI adjustment - stronger
        rsi_adj = 0
        if rsi > 60:  # Overbought - expect pullback
            rsi_adj = -0.003 * (rsi - 50) / 50
        elif rsi < 40:  # Oversold - expect bounce
            rsi_adj = 0.003 * (50 - rsi) / 50
        
        # Position adjustment (if near high, less likely to go higher)
        position_adj = 0
        if price_position > 0.7:  # Near highs
            position_adj = -0.002 * (price_position - 0.5)
        elif price_position < 0.3:  # Near lows
            position_adj = 0.002 * (0.5 - price_position)
        
        # Combine all factors
        expected_return = trend_component + mean_reversion + rsi_adj + position_adj
        
        # Add small random noise
        random_component = np.random.normal(0, volatility * 0.1)
        
        # Clip extreme moves - max 1% per day
        total_return = np.clip(expected_return + random_component, -0.01, 0.01)
        
        next_price = current_price * (1 + total_return)
        
        # Enforce hard limit ±15% from original price
        next_price = np.clip(next_price, min_price, max_price)
        
        predictions.append(round(float(next_price), 2))
        current_price = next_price
    
    # Generate dates (skip weekends)
    dates = []
    current_date = datetime.now()
    while len(dates) < steps:
        current_date += timedelta(days=1)
        if current_date.weekday() < 5:
            dates.append(current_date.strftime('%Y-%m-%d'))
    
    # Determine trend based on overall prediction
    total_change = (predictions[-1] - last_price) / last_price
    if total_change > 0.02:
        trend = 'up'
    elif total_change < -0.02:
        trend = 'down'
    else:
        trend = 'sideways'
    
    return {
        'predictions': predictions, 
        'dates': dates, 
        'model': 'technical', 
        'confidence': 0.45,  # Lower confidence for simple model
        'trend': trend, 
        'last_price': last_price,
        'rsi': round(rsi, 1),
        'sma20': round(sma20, 2),
        'note': 'Fallback model - limited accuracy'
    }


# Model cache
_model_cache: Dict[str, StockMLModel] = {}

def get_model(symbol: str) -> StockMLModel:
    if symbol not in _model_cache:
        _model_cache[symbol] = StockMLModel(symbol)
    return _model_cache[symbol]

def train_model(symbol: str, df: pd.DataFrame) -> Dict:
    return get_model(symbol).train(df)

def predict_stock(symbol: str, df: pd.DataFrame, steps: int = 7, model_type: str = 'ensemble') -> Dict:
    model = get_model(symbol)
    if not model.is_trained:
        model.train(df)
    return model.predict(df, steps, model_type)
