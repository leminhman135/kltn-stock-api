

import os
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["Advanced ML"])


# ============ Pydantic Models ============

class SentimentRequest(BaseModel):
    texts: List[str]
    
class SentimentResponse(BaseModel):
    results: List[Dict]
    method: str
    processing_time_ms: float

class DeepPredictRequest(BaseModel):
    symbol: str
    model_type: str = "lstm"  # lstm, gru, cnn_lstm, ensemble
    forecast_days: int = 5
    sequence_length: int = 60

class DeepPredictResponse(BaseModel):
    symbol: str
    predictions: List[float]
    dates: List[str]
    model_type: str
    confidence: float
    uncertainty: Optional[List[float]] = None
    training_metrics: Optional[Dict] = None


# ============ Helper Functions ============

def get_stock_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch stock data from database"""
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            raise HTTPException(status_code=500, detail="Database not configured")
        
        conn = psycopg2.connect(db_url)
        query = f"""
            SELECT date, open, high, low, close, volume
            FROM stock_prices
            WHERE symbol = %s
            ORDER BY date DESC
            LIMIT %s
        """
        
        df = pd.read_sql(query, conn, params=(symbol, days))
        conn.close()
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail=f"No data for {symbol}")
        
        df = df.sort_values('date').reset_index(drop=True)
        return df
        
    except Exception as e:
        logger.error(f"Database error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Sentiment Endpoints ============

@router.post("/sentiment", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    """
    Phân tích sentiment với FinBERT
    
    - Sử dụng ProsusAI/finbert cho tiếng Anh
    - Fallback keyword-based cho tiếng Việt
    - Returns: positive, negative, neutral với confidence score
    """
    import time
    start_time = time.time()
    
    try:
        from src.sentiment import get_analyzer
        
        analyzer = get_analyzer()
        results = analyzer.analyze_batch(request.texts)
        
        processing_time = (time.time() - start_time) * 1000
        
        return SentimentResponse(
            results=results,
            method=results[0].get('method', 'unknown') if results else 'unknown',
            processing_time_ms=round(processing_time, 2)
        )
        
    except ImportError:
        # Fallback if sentiment module not available
        results = [{'label': 'neutral', 'score': 0.5, 'sentiment_score': 0.0, 'method': 'unavailable'} 
                   for _ in request.texts]
        return SentimentResponse(
            results=results,
            method='unavailable',
            processing_time_ms=0
        )


@router.get("/sentiment/{symbol}")
async def get_symbol_sentiment(
    symbol: str,
    days: int = Query(default=7, ge=1, le=30)
):
    """
    Lấy sentiment tổng hợp cho mã cổ phiếu từ database
    """
    try:
        import psycopg2
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            return {"symbol": symbol, "sentiment": None, "error": "Database not configured"}
        
        conn = psycopg2.connect(db_url)
        with conn.cursor() as cur:
            cur.execute("""
                SELECT 
                    date_trunc('day', analyzed_at) as date,
                    AVG(sentiment_score) as avg_score,
                    COUNT(*) as news_count,
                    COUNT(*) FILTER (WHERE sentiment = 'positive') as positive,
                    COUNT(*) FILTER (WHERE sentiment = 'negative') as negative,
                    COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral
                FROM analyzed_news
                WHERE (symbol = %s OR symbol = 'MARKET')
                AND analyzed_at >= NOW() - INTERVAL '%s days'
                GROUP BY date_trunc('day', analyzed_at)
                ORDER BY date DESC
            """, (symbol, days))
            
            rows = cur.fetchall()
        conn.close()
        
        daily_sentiment = []
        for row in rows:
            daily_sentiment.append({
                'date': row[0].strftime('%Y-%m-%d'),
                'avg_score': round(float(row[1]), 3) if row[1] else 0,
                'news_count': row[2],
                'positive': row[3],
                'negative': row[4],
                'neutral': row[5]
            })
        
        # Overall sentiment
        if daily_sentiment:
            overall_score = np.mean([d['avg_score'] for d in daily_sentiment])
            total_news = sum(d['news_count'] for d in daily_sentiment)
            
            if overall_score > 0.2:
                sentiment_label = 'bullish'
            elif overall_score < -0.2:
                sentiment_label = 'bearish'
            else:
                sentiment_label = 'neutral'
        else:
            overall_score = 0
            total_news = 0
            sentiment_label = 'no_data'
        
        return {
            'symbol': symbol,
            'days': days,
            'overall_sentiment': {
                'score': round(overall_score, 3),
                'label': sentiment_label,
                'total_news': total_news
            },
            'daily': daily_sentiment
        }
        
    except Exception as e:
        logger.error(f"Sentiment fetch error: {e}")
        return {"symbol": symbol, "error": str(e)}


# ============ Deep Learning Endpoints ============

@router.post("/deep-predict", response_model=DeepPredictResponse)
async def deep_learning_predict(request: DeepPredictRequest):
    """
    Dự đoán giá cổ phiếu bằng Deep Learning (LSTM/GRU/CNN-LSTM)
    
    Models:
    - lstm: LSTM with attention mechanism
    - gru: Bidirectional GRU
    - cnn_lstm: Hybrid CNN-LSTM
    - ensemble: Combination of all models
    """
    try:
        from src.models import TENSORFLOW_AVAILABLE, StockPredictor
        
        if not TENSORFLOW_AVAILABLE:
            raise HTTPException(
                status_code=503, 
                detail="TensorFlow not available. Using fallback models."
            )
        
        # Get data
        df = get_stock_data(request.symbol, days=500)
        
        if len(df) < 100:
            raise HTTPException(
                status_code=400,
                detail=f"Need at least 100 data points, got {len(df)}"
            )
        
        # Add technical features
        from src.features import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer(df)
        df_features = engineer.create_all_features()
        
        # Select features for model
        feature_cols = ['close', 'volume', 'rsi_14', 'macd', 'bb_percent_b', 
                       'return_1d', 'return_5d', 'volatility_20', 'sma_ratio_20']
        
        # Filter to available columns
        available_cols = [col for col in feature_cols if col in df_features.columns]
        if 'close' not in available_cols:
            available_cols = ['close'] + available_cols
        
        # Create and train model
        predictor = StockPredictor(
            model_type=request.model_type,
            sequence_length=request.sequence_length,
            forecast_horizon=request.forecast_days
        )
        
        training_metrics = predictor.fit(
            df_features,
            feature_cols=available_cols,
            target_col='close',
            epochs=50,
            batch_size=32
        )
        
        # Make predictions
        predictions = predictor.predict(df_features)
        
        # Generate dates
        dates = []
        current_date = datetime.now()
        while len(dates) < request.forecast_days:
            current_date += timedelta(days=1)
            if current_date.weekday() < 5:
                dates.append(current_date.strftime('%Y-%m-%d'))
        
        return DeepPredictResponse(
            symbol=request.symbol,
            predictions=predictions.tolist(),
            dates=dates,
            model_type=request.model_type,
            confidence=0.7,
            training_metrics=training_metrics
        )
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=503, detail="Deep learning modules not available")
    except Exception as e:
        logger.error(f"Deep predict error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/features/{symbol}")
async def get_features(
    symbol: str,
    days: int = Query(default=100, ge=30, le=500)
):
    """
    Lấy engineered features cho mã cổ phiếu
    
    Returns 100+ technical indicators và features
    """
    try:
        df = get_stock_data(symbol, days)
        
        from src.features import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer(df)
        df_features = engineer.create_all_features()
        
        # Get latest values
        latest = df_features.iloc[-1].to_dict()
        
        # Clean NaN values
        for key, value in latest.items():
            if pd.isna(value):
                latest[key] = None
            elif isinstance(value, (np.int64, np.float64)):
                latest[key] = float(value)
        
        # Feature summary
        feature_groups = {
            'trend': ['sma_5', 'sma_10', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal'],
            'momentum': ['rsi_14', 'stoch_k', 'stoch_d', 'cci', 'williams_r', 'roc_10'],
            'volatility': ['bb_upper', 'bb_lower', 'bb_percent_b', 'atr_14', 'volatility_20'],
            'volume': ['volume_sma', 'volume_ratio', 'obv', 'vwap'],
            'returns': ['return_1d', 'return_5d', 'return_10d', 'return_20d']
        }
        
        summary = {}
        for group, cols in feature_groups.items():
            summary[group] = {col: latest.get(col) for col in cols if col in latest}
        
        return {
            'symbol': symbol,
            'date': str(df_features['date'].iloc[-1]) if 'date' in df_features.columns else None,
            'total_features': len(df_features.columns),
            'data_points': len(df_features),
            'summary': summary,
            'all_features': latest
        }
        
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-status")
async def get_model_status():
    """
    Kiểm tra trạng thái các ML models
    """
    status = {
        'sklearn': False,
        'tensorflow': False,
        'transformers': False,
        'torch': False,
        'prophet': False,
        'statsmodels': False
    }
    
    try:
        from sklearn.ensemble import RandomForestRegressor
        status['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import tensorflow as tf
        status['tensorflow'] = True
        status['tensorflow_version'] = tf.__version__
        status['gpu_available'] = len(tf.config.list_physical_devices('GPU')) > 0
    except ImportError:
        pass
    
    try:
        import transformers
        status['transformers'] = True
        status['transformers_version'] = transformers.__version__
    except ImportError:
        pass
    
    try:
        import torch
        status['torch'] = True
        status['torch_version'] = torch.__version__
        status['torch_cuda'] = torch.cuda.is_available()
    except ImportError:
        pass
    
    try:
        from prophet import Prophet
        status['prophet'] = True
    except ImportError:
        pass
    
    try:
        import statsmodels
        status['statsmodels'] = True
    except ImportError:
        pass
    
    # Available models
    available_models = []
    if status['sklearn']:
        available_models.extend(['linear', 'random_forest', 'gradient_boost', 'xgboost'])
    if status['tensorflow']:
        available_models.extend(['lstm', 'gru', 'cnn_lstm'])
    if status['statsmodels']:
        available_models.append('arima')
    if status['prophet']:
        available_models.append('prophet')
    if status['transformers']:
        available_models.append('finbert')
    
    return {
        'libraries': status,
        'available_models': available_models,
        'deep_learning_ready': status['tensorflow'],
        'sentiment_ready': status['transformers'] and status['torch']
    }


@router.post("/train/{symbol}")
async def train_model(
    symbol: str,
    model_type: str = Query(default="ensemble", regex="^(lstm|gru|cnn_lstm|ensemble)$")
):
    """
    Train deep learning model cho symbol cụ thể
    """
    try:
        from src.models import TENSORFLOW_AVAILABLE
        
        if not TENSORFLOW_AVAILABLE:
            return {"success": False, "error": "TensorFlow not available"}
        
        # Get data
        df = get_stock_data(symbol, days=500)
        
        # Add features
        from src.features import AdvancedFeatureEngineer
        
        engineer = AdvancedFeatureEngineer(df)
        df_features = engineer.create_all_features()
        
        # Train model
        from src.models import StockPredictor
        
        predictor = StockPredictor(model_type=model_type)
        
        feature_cols = ['close', 'volume', 'rsi_14', 'macd', 'return_1d']
        available_cols = [col for col in feature_cols if col in df_features.columns]
        
        metrics = predictor.fit(df_features, feature_cols=available_cols, epochs=100)
        
        return {
            'success': True,
            'symbol': symbol,
            'model_type': model_type,
            'data_points': len(df),
            'features_used': len(available_cols),
            'metrics': metrics
        }
        
    except Exception as e:
        logger.error(f"Training error: {e}")
        return {"success": False, "error": str(e)}
