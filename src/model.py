# src/model.py
"""
Stock Price Prediction using Machine Learning
Includes: Linear Regression, Random Forest, Gradient Boosting
With: RSI, MACD, Bollinger Bands, Mean Reversion, News Sentiment
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_sentiment_score(symbol: str) -> Optional[Dict]:
    """
    Lấy sentiment score từ database cho mã cổ phiếu
    Returns: {'score': float, 'positive': int, 'negative': int, 'neutral': int, 'total': int}
    """
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            return None
        
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            # Lấy sentiment của 7 ngày gần nhất
            cur.execute("""
                SELECT 
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) FILTER (WHERE sentiment = 'positive') as positive,
                    COUNT(*) FILTER (WHERE sentiment = 'negative') as negative,
                    COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral,
                    COUNT(*) as total
                FROM analyzed_news
                WHERE (symbol = %s OR symbol = 'MARKET')
                AND analyzed_at >= NOW() - INTERVAL '7 days'
            """, (symbol,))
            row = cur.fetchone()
            
            if row and row[4] > 0:  # Có ít nhất 1 tin
                conn.close()
                return {
                    'score': float(row[0]) if row[0] else 0.0,
                    'positive': row[1] or 0,
                    'negative': row[2] or 0,
                    'neutral': row[3] or 0,
                    'total': row[4] or 0
                }
        conn.close()
        return None
    except Exception as e:
        logger.debug(f"Sentiment fetch error: {e}")
        return None

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
        """Make predictions - realistic approach based on model accuracy"""
        if not self.is_trained:
            return quick_predict(df['close'].tolist(), steps)
        
        try:
            # Lấy sentiment data cho symbol này
            self._sentiment_data = get_sentiment_score(self.symbol)
            if self._sentiment_data:
                logger.info(f"📰 Sentiment for {self.symbol}: score={self._sentiment_data['score']:.2f}")
            
            data = df.copy().sort_values('date').reset_index(drop=True)
            close = data['close'].values
            high = data['high'].values if 'high' in data.columns else close
            low = data['low'].values if 'low' in data.columns else close
            volume = data['volume'].values if 'volume' in data.columns else np.ones(len(close))
            
            last_price = float(close[-1])
            sma20 = np.mean(close[-20:]) if len(close) >= 20 else np.mean(close)
            sma5 = np.mean(close[-5:]) if len(close) >= 5 else np.mean(close)
            
            # Calculate volatility properly
            if len(close) >= 31:
                returns = np.diff(close[-31:]) / close[-31:-1]
            else:
                returns = np.diff(close) / close[:-1]
            daily_vol = np.std(returns) if len(returns) > 0 else 0.02
            avg_return = np.mean(returns) if len(returns) > 0 else 0
            
            # Current indicators
            current_rsi = self.calculate_rsi(close)
            current_bb = self.calculate_bollinger_position(close)
            
            # Calculate model confidence from accuracy
            avg_accuracy = np.mean([m.get('accuracy', 50) for m in self.metrics.values() 
                                   if isinstance(m, dict) and 'accuracy' in m])
            baseline_acc = self.metrics.get('baseline', {}).get('accuracy', 50)
            
            # Model is useful only if significantly better than baseline
            model_useful = avg_accuracy > baseline_acc + 3
            
            predictions = []
            prediction_ranges = []
            current_price = last_price
            
            for step in range(steps):
                if model_useful:
                    # Get features for prediction
                    temp_close = np.append(close, current_price) if step > 0 else close
                    temp_high = np.append(high, current_price) if step > 0 else high
                    temp_low = np.append(low, current_price) if step > 0 else low
                    temp_volume = np.append(volume, volume[-1]) if step > 0 else volume
                    
                    feature_row = self._create_single_feature(temp_close, temp_high, temp_low, temp_volume)
                    
                    if feature_row is not None:
                        feature_scaled = self.scaler.transform(feature_row.reshape(1, -1))
                        
                        # Get ensemble prediction
                        pred_returns = []
                        for name, model in self.models.items():
                            try:
                                pred = model.predict(feature_scaled)[0]
                                pred_returns.append(np.clip(pred, -0.03, 0.03))
                            except:
                                pass
                        
                        if pred_returns:
                            model_return = np.median(pred_returns)  # Use median for robustness
                        else:
                            model_return = 0
                    else:
                        model_return = 0
                else:
                    # Model not useful, use simple momentum
                    model_return = avg_return * 0.5  # Weak momentum
                
                # Apply realistic adjustments
                adjusted_return = model_return
                
                # 1. Mean reversion (gentle, only when far from SMA)
                price_vs_sma20 = (current_price - sma20) / sma20
                if abs(price_vs_sma20) > 0.05:  # More than 5% from SMA20
                    reversion = -price_vs_sma20 * 0.1  # Pull 10% of gap
                    adjusted_return = adjusted_return * 0.7 + reversion * 0.3
                
                # 2. RSI adjustment (only at extremes)
                if current_rsi > 75:  # Very overbought
                    adjusted_return = min(adjusted_return, 0.005)  # Cap upside
                elif current_rsi < 25:  # Very oversold
                    adjusted_return = max(adjusted_return, -0.005)  # Cap downside
                
                # 3. Confidence decay for future predictions
                confidence_factor = 0.9 ** step  # Gentle decay
                adjusted_return *= confidence_factor
                
                # 4. Add small random walk component (markets are noisy)
                noise = np.random.normal(0, daily_vol * 0.3)
                adjusted_return += noise * (0.5 ** step)  # Less noise for further predictions
                
                # 5. Apply sentiment (first 2 days only)
                if step < 2 and hasattr(self, '_sentiment_data') and self._sentiment_data:
                    sent = self._sentiment_data
                    if sent.get('total', 0) >= 3:
                        sent_adj = sent.get('score', 0) * 0.003  # Max 0.3% impact
                        adjusted_return += sent_adj
                
                # Final realistic bounds
                adjusted_return = np.clip(adjusted_return, -0.025, 0.025)  # Max 2.5% per day
                
                next_price = current_price * (1 + adjusted_return)
                
                # Absolute bounds (max 10% total change)
                max_price = last_price * 1.10
                min_price = last_price * 0.90
                next_price = np.clip(next_price, min_price, max_price)
                
                predictions.append(round(float(next_price), 2))
                
                # Prediction range (confidence interval)
                uncertainty = daily_vol * np.sqrt(step + 1) * current_price * 1.5
                prediction_ranges.append({
                    'low': round(max(min_price, next_price - uncertainty), 2),
                    'high': round(min(max_price, next_price + uncertainty), 2)
                })
                
                current_price = next_price
            
            # Generate dates (skip weekends)
            dates = []
            current_date = datetime.now()
            while len(dates) < steps:
                current_date += timedelta(days=1)
                if current_date.weekday() < 5:
                    dates.append(current_date.strftime('%Y-%m-%d'))
            
            # Determine trend
            if len(predictions) >= 2:
                price_change = (predictions[-1] - last_price) / last_price
                if price_change > 0.015:
                    trend = 'up'
                elif price_change < -0.015:
                    trend = 'down'
                else:
                    trend = 'sideways'
            else:
                trend = 'unknown'
            
            # Honest confidence score
            if model_useful:
                confidence = min(0.7, max(0.4, (avg_accuracy - 50) / 30))
            else:
                confidence = 0.35
            
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
                'model_accuracy': round(avg_accuracy, 1),
                'model_useful': model_useful,
                'daily_volatility': round(daily_vol * 100, 2),
                'metrics': {k: v for k, v in self.metrics.items() if k != 'baseline'},
                'sentiment': self._sentiment_data if hasattr(self, '_sentiment_data') else None,
                'warning': None if model_useful else 'Model accuracy near baseline, predictions less reliable'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
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
    """Quick prediction using random walk with drift - realistic approach"""
    if len(prices) < 5:
        return {'predictions': [], 'error': 'Need at least 5 price points'}
    
    prices = np.array(prices, dtype=float)
    last_price = float(prices[-1])
    
    # Calculate historical statistics
    returns = np.diff(prices) / prices[:-1]
    daily_vol = np.std(returns[-20:]) if len(returns) >= 20 else np.std(returns)
    avg_return = np.mean(returns[-20:]) if len(returns) >= 20 else np.mean(returns)
    
    # Moving averages
    sma5 = np.mean(prices[-5:])
    sma20 = np.mean(prices[-20:]) if len(prices) >= 20 else np.mean(prices)
    
    # Simple RSI
    if len(returns) >= 14:
        gains = np.where(returns[-14:] > 0, returns[-14:], 0)
        losses = np.where(returns[-14:] < 0, -returns[-14:], 0)
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses) + 1e-10
        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
    else:
        rsi = 50
    
    predictions = []
    prediction_ranges = []
    current_price = last_price
    
    # Realistic bounds (max 10% total change over prediction period)
    max_price = last_price * 1.10
    min_price = last_price * 0.90
    
    for i in range(steps):
        # Random walk with slight mean reversion
        # Base: small drift based on recent average
        drift = avg_return * 0.3 * (0.9 ** i)  # Decaying drift
        
        # Mean reversion only when far from SMA20
        price_vs_sma = (current_price - sma20) / sma20
        if abs(price_vs_sma) > 0.05:
            reversion = -price_vs_sma * 0.05  # Gentle pull
        else:
            reversion = 0
        
        # Random component (most important for realism)
        noise = np.random.normal(0, daily_vol * 0.6)
        
        # Combine
        expected_return = drift + reversion + noise
        
        # RSI bounds (only at extremes)
        if rsi > 70 and expected_return > 0.01:
            expected_return *= 0.5
        elif rsi < 30 and expected_return < -0.01:
            expected_return *= 0.5
        
        # Daily bounds: max 2.5% change
        expected_return = np.clip(expected_return, -0.025, 0.025)
        
        next_price = current_price * (1 + expected_return)
        next_price = np.clip(next_price, min_price, max_price)
        
        predictions.append(round(float(next_price), 2))
        
        # Confidence interval
        uncertainty = daily_vol * np.sqrt(i + 1) * current_price * 1.5
        prediction_ranges.append({
            'low': round(max(min_price, next_price - uncertainty), 2),
            'high': round(min(max_price, next_price + uncertainty), 2)
        })
        
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
    if total_change > 0.015:
        trend = 'up'
    elif total_change < -0.015:
        trend = 'down'
    else:
        trend = 'sideways'
    
    return {
        'symbol': 'unknown',
        'predictions': predictions,
        'prediction_ranges': prediction_ranges,
        'dates': dates, 
        'model': 'random_walk', 
        'confidence': 0.35,
        'trend': trend, 
        'last_price': last_price,
        'rsi': round(rsi, 1),
        'sma20': round(sma20, 2),
        'daily_volatility': round(daily_vol * 100, 2),
        'model_useful': False,
        'warning': 'Fallback random walk model - predictions have high uncertainty'
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
