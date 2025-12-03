"""
ML Models API Endpoints
Tích hợp ARIMA, Prophet, LSTM, GRU, Ensemble vào FastAPI
"""

from fastapi import APIRouter, HTTPException, Query, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import Optional, List
from datetime import datetime, timedelta, date
from pydantic import BaseModel
import pandas as pd
import numpy as np
import logging
import json
import os

# Local imports
from src.database.connection import get_db
from src.database.models import Stock, StockPrice, ModelMetrics, Prediction

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml", tags=["ML Models"])


# =====================================================
# PYDANTIC MODELS
# =====================================================

class PredictionRequest(BaseModel):
    symbol: str
    days: int = 5
    model_type: str = "arima"  # arima, prophet, lstm, gru, ensemble


class TrainRequest(BaseModel):
    symbol: str
    model_type: str  # arima, prophet, lstm, gru
    epochs: int = 100
    lookback: int = 60


class ModelComparisonRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    models: List[str] = ["arima", "prophet", "lstm", "gru"]


class EnsemblePredictionRequest(BaseModel):
    symbol: str
    days: int = 5
    ensemble_type: str = "weighted"  # average, weighted, stacking
    include_sentiment: bool = True


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def get_stock_data(db: Session, symbol: str, days: int = 365) -> pd.DataFrame:
    """Lấy dữ liệu stock từ database"""
    stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
    if not stock:
        raise HTTPException(status_code=404, detail=f"Stock {symbol} not found")
    
    prices = db.query(StockPrice).filter(
        StockPrice.stock_id == stock.id
    ).order_by(desc(StockPrice.date)).limit(days).all()
    
    if not prices:
        raise HTTPException(status_code=404, detail=f"No price data for {symbol}")
    
    df = pd.DataFrame([{
        'date': p.date,
        'open': float(p.open),
        'high': float(p.high),
        'low': float(p.low),
        'close': float(p.close),
        'volume': float(p.volume)
    } for p in prices])
    
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    return df


def save_model_metrics(db: Session, symbol: str, model_name: str, 
                       metrics: dict, predictions: list = None):
    """Lưu metrics của model vào database"""
    # Lưu metrics
    model_metrics = ModelMetrics(
        symbol=symbol.upper(),
        model_name=model_name,
        mae=metrics.get('mae', 0),
        rmse=metrics.get('rmse', 0),
        mape=metrics.get('mape', 0),
        accuracy=100 - metrics.get('mape', 0),  # Rough estimate
        train_date=datetime.now(),
        is_active=True
    )
    
    db.add(model_metrics)
    db.commit()
    
    return model_metrics.id


# =====================================================
# ARIMA ENDPOINTS
# =====================================================

@router.get("/arima/predict/{symbol}")
async def arima_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    auto_order: bool = Query(False, description="Tự động tìm order tối ưu"),
    db: Session = Depends(get_db)  # Will be injected from main app
):
    """
    📈 Dự đoán giá cổ phiếu bằng ARIMA
    
    ARIMA(p,d,q):
    - p: AutoRegressive order
    - d: Differencing order  
    - q: Moving Average order
    
    **Ưu điểm**: Đơn giản, nhanh, hiệu quả với dữ liệu stationary
    **Nhược điểm**: Chỉ phù hợp dữ liệu tuyến tính
    """
    try:
        from src.models.arima_model import ARIMAModel
        
        # Get data
        df = get_stock_data(db, symbol)
        
        if len(df) < 60:
            raise HTTPException(status_code=400, detail="Need at least 60 days of data")
        
        # Prepare data
        close_prices = df['close']
        
        # Initialize and fit ARIMA
        arima = ARIMAModel(order=(5, 1, 2))  # Default order
        arima.fit(close_prices, auto_order=auto_order)
        
        # Predict
        predictions, lower_bounds, upper_bounds = arima.predict_with_confidence(steps=days)
        
        # Generate future dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Calculate metrics on last 20% of data
        test_size = int(len(df) * 0.2)
        test_data = close_prices[-test_size:]
        arima_test = ARIMAModel(order=arima.order)
        arima_test.fit(close_prices[:-test_size])
        metrics = arima_test.evaluate(test_data)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": "ARIMA",
            "model_params": {
                "order": arima.order,
                "aic": float(arima.fitted_model.aic),
                "bic": float(arima.fitted_model.bic)
            },
            "current_price": float(close_prices.iloc[-1]),
            "predictions": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": float(pred),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "change_percent": float((pred - close_prices.iloc[-1]) / close_prices.iloc[-1] * 100)
                }
                for date, pred, lower, upper in zip(future_dates, predictions, lower_bounds, upper_bounds)
            ],
            "metrics": {
                "mae": round(metrics['mae'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mape": round(metrics['mape'], 2)
            },
            "recommendation": "BUY" if predictions[-1] > close_prices.iloc[-1] * 1.02 else 
                             ("SELL" if predictions[-1] < close_prices.iloc[-1] * 0.98 else "HOLD")
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="ARIMA module not available. Install statsmodels.")
    except Exception as e:
        logger.error(f"ARIMA prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# PROPHET ENDPOINTS
# =====================================================

@router.get("/prophet/predict/{symbol}")
async def prophet_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    yearly_seasonality: bool = Query(True),
    weekly_seasonality: bool = Query(True),
    db: Session = Depends(get_db)
):
    """
    📊 Dự đoán giá cổ phiếu bằng Facebook Prophet
    
    Prophet tự động phát hiện:
    - Trend (xu hướng)
    - Yearly seasonality (mùa vụ năm)
    - Weekly seasonality (mùa vụ tuần)
    - Holiday effects
    
    **Ưu điểm**: Robust với missing data, tự động detect seasonality
    **Nhược điểm**: Cần nhiều dữ liệu hơn ARIMA
    """
    try:
        from src.models.prophet_model import ProphetModel
        
        # Get data
        df = get_stock_data(db, symbol)
        
        if len(df) < 90:
            raise HTTPException(status_code=400, detail="Prophet needs at least 90 days of data")
        
        # Prepare data
        close_prices = df['close']
        
        # Initialize Prophet
        prophet = ProphetModel(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            changepoint_prior_scale=0.05
        )
        
        # Fit
        prophet.fit(close_prices)
        
        # Predict
        forecast = prophet.get_forecast_values(periods=days)
        
        # Get metrics
        test_size = int(len(df) * 0.2)
        test_data = close_prices[-test_size:]
        prophet_test = ProphetModel(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality
        )
        prophet_test.fit(close_prices[:-test_size])
        metrics = prophet_test.evaluate(test_data)
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": "Prophet",
            "model_params": {
                "yearly_seasonality": yearly_seasonality,
                "weekly_seasonality": weekly_seasonality,
                "growth": "linear"
            },
            "current_price": float(close_prices.iloc[-1]),
            "predictions": [
                {
                    "date": pd.Timestamp(date).strftime("%Y-%m-%d"),
                    "predicted_price": float(pred),
                    "lower_bound": float(lower),
                    "upper_bound": float(upper),
                    "change_percent": float((pred - close_prices.iloc[-1]) / close_prices.iloc[-1] * 100)
                }
                for date, pred, lower, upper in zip(
                    forecast['dates'], 
                    forecast['predictions'], 
                    forecast['lower_bound'], 
                    forecast['upper_bound']
                )
            ],
            "metrics": {
                "mae": round(metrics['mae'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mape": round(metrics['mape'], 2)
            },
            "recommendation": "BUY" if forecast['predictions'][-1] > close_prices.iloc[-1] * 1.02 else 
                             ("SELL" if forecast['predictions'][-1] < close_prices.iloc[-1] * 0.98 else "HOLD")
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="Prophet not available. Install prophet.")
    except Exception as e:
        logger.error(f"Prophet prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# LSTM ENDPOINTS
# =====================================================

@router.get("/lstm/predict/{symbol}")
async def lstm_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    lookback: int = Query(60, description="Số ngày lookback"),
    use_pretrained: bool = Query(True, description="Sử dụng model đã train sẵn"),
    db: Session = Depends(get_db)
):
    """
    🧠 Dự đoán giá cổ phiếu bằng LSTM (Deep Learning)
    
    LSTM (Long Short-Term Memory):
    - Xử lý tốt long-term dependencies
    - Phù hợp với dữ liệu phi tuyến
    
    **Ưu điểm**: Mạnh mẽ với patterns phức tạp
    **Nhược điểm**: Cần nhiều dữ liệu, training lâu
    
    ⚠️ Note: Nếu chưa có model, sẽ train nhanh với dữ liệu hiện có
    """
    try:
        from src.models.lstm_gru_models import LSTMModel
        
        # Get data
        df = get_stock_data(db, symbol)
        
        if len(df) < lookback + 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Need at least {lookback + 30} days of data for LSTM"
            )
        
        # Check for pretrained model
        model_path = f"models/lstm_{symbol.upper()}.h5"
        
        lstm = LSTMModel(
            lookback=lookback,
            forecast_horizon=1,
            units=[50, 50],
            dropout=0.2
        )
        
        # Prepare features
        feature_df = df[['close', 'volume', 'high', 'low']].copy()
        
        # Add technical indicators
        feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
        feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
        feature_df['rsi'] = calculate_rsi(feature_df['close'])
        feature_df = feature_df.dropna()
        
        if use_pretrained and os.path.exists(model_path):
            lstm.load_model(model_path)
            logger.info(f"Loaded pretrained LSTM for {symbol}")
        else:
            # Quick train with minimal epochs
            lstm.fit(
                feature_df, 
                target_col='close',
                epochs=30,
                batch_size=32,
                verbose=0
            )
        
        # Predict next days
        predictions = []
        current_data = feature_df.copy()
        
        for i in range(days):
            pred = lstm.predict(current_data, target_col='close')
            if len(pred) > 0:
                predictions.append(float(pred[-1]))
            
        # Generate dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Get metrics from last prediction
        metrics = lstm.evaluate(feature_df[-int(len(feature_df)*0.2):], target_col='close')
        
        current_price = float(df['close'].iloc[-1])
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": "LSTM",
            "model_params": {
                "lookback": lookback,
                "layers": 2,
                "units": [50, 50],
                "dropout": 0.2,
                "pretrained": use_pretrained and os.path.exists(model_path)
            },
            "current_price": current_price,
            "predictions": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": pred,
                    "change_percent": (pred - current_price) / current_price * 100
                }
                for date, pred in zip(future_dates, predictions)
            ] if predictions else [],
            "metrics": {
                "mae": round(metrics['mae'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mape": round(metrics['mape'], 2)
            },
            "recommendation": "BUY" if predictions and predictions[-1] > current_price * 1.02 else 
                             ("SELL" if predictions and predictions[-1] < current_price * 0.98 else "HOLD"),
            "note": "For better results, train model offline with more epochs"
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="LSTM not available. Install tensorflow.")
    except Exception as e:
        logger.error(f"LSTM prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# GRU ENDPOINTS
# =====================================================

@router.get("/gru/predict/{symbol}")
async def gru_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    lookback: int = Query(60, description="Số ngày lookback"),
    use_pretrained: bool = Query(True, description="Sử dụng model đã train sẵn"),
    db: Session = Depends(get_db)
):
    """
    🧠 Dự đoán giá cổ phiếu bằng GRU (Deep Learning)
    
    GRU (Gated Recurrent Unit):
    - Phiên bản đơn giản hóa của LSTM
    - Ít parameters, train nhanh hơn
    
    **Ưu điểm**: Nhanh hơn LSTM, ít overfit
    **Nhược điểm**: Có thể kém hơn với long sequences
    """
    try:
        from src.models.lstm_gru_models import GRUModel
        
        # Get data
        df = get_stock_data(db, symbol)
        
        if len(df) < lookback + 30:
            raise HTTPException(
                status_code=400, 
                detail=f"Need at least {lookback + 30} days of data for GRU"
            )
        
        # Check for pretrained model
        model_path = f"models/gru_{symbol.upper()}.h5"
        
        gru = GRUModel(
            lookback=lookback,
            forecast_horizon=1,
            units=[50, 50],
            dropout=0.2
        )
        
        # Prepare features
        feature_df = df[['close', 'volume', 'high', 'low']].copy()
        feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
        feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
        feature_df['rsi'] = calculate_rsi(feature_df['close'])
        feature_df = feature_df.dropna()
        
        if use_pretrained and os.path.exists(model_path):
            gru.load_model(model_path)
            logger.info(f"Loaded pretrained GRU for {symbol}")
        else:
            # Quick train
            gru.fit(
                feature_df, 
                target_col='close',
                epochs=30,
                batch_size=32,
                verbose=0
            )
        
        # Predict
        predictions = []
        current_data = feature_df.copy()
        
        for i in range(days):
            pred = gru.predict(current_data, target_col='close')
            if len(pred) > 0:
                predictions.append(float(pred[-1]))
        
        # Generate dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Get metrics
        metrics = gru.evaluate(feature_df[-int(len(feature_df)*0.2):], target_col='close')
        
        current_price = float(df['close'].iloc[-1])
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": "GRU",
            "model_params": {
                "lookback": lookback,
                "layers": 2,
                "units": [50, 50],
                "dropout": 0.2,
                "pretrained": use_pretrained and os.path.exists(model_path)
            },
            "current_price": current_price,
            "predictions": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": pred,
                    "change_percent": (pred - current_price) / current_price * 100
                }
                for date, pred in zip(future_dates, predictions)
            ] if predictions else [],
            "metrics": {
                "mae": round(metrics['mae'], 2),
                "rmse": round(metrics['rmse'], 2),
                "mape": round(metrics['mape'], 2)
            },
            "recommendation": "BUY" if predictions and predictions[-1] > current_price * 1.02 else 
                             ("SELL" if predictions and predictions[-1] < current_price * 0.98 else "HOLD"),
            "note": "For better results, train model offline with more epochs"
        }
        
    except ImportError:
        raise HTTPException(status_code=500, detail="GRU not available. Install tensorflow.")
    except Exception as e:
        logger.error(f"GRU prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# ENSEMBLE ENDPOINTS
# =====================================================

@router.get("/ensemble/predict/{symbol}")
async def ensemble_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    ensemble_type: str = Query("weighted", description="Loại ensemble: average, weighted, stacking"),
    include_sentiment: bool = Query(True, description="Kết hợp FinBERT sentiment"),
    db: Session = Depends(get_db)
):
    """
    🔮 Dự đoán tổng hợp (Ensemble) kết hợp nhiều models
    
    **Ensemble Types:**
    - average: Trung bình đơn giản
    - weighted: Trọng số theo performance
    - stacking: Meta-learning với Random Forest
    
    **Features:**
    - Kết hợp ARIMA + Prophet + (LSTM/GRU nếu có)
    - Tích hợp FinBERT sentiment
    - Điều chỉnh prediction theo tin tức
    
    ⭐ Recommended: Sử dụng ensemble_type="weighted" cho kết quả tốt nhất
    """
    try:
        from src.models.arima_model import ARIMAModel
        from src.models.prophet_model import ProphetModel
        from src.models.ensemble import WeightedAverageEnsemble, StackingEnsemble, SimpleAverageEnsemble
        
        # Get data
        df = get_stock_data(db, symbol)
        close_prices = df['close']
        current_price = float(close_prices.iloc[-1])
        
        # Collect predictions from each model
        all_predictions = {}
        all_metrics = {}
        
        # 1. ARIMA
        try:
            arima = ARIMAModel(order=(5, 1, 2))
            arima.fit(close_prices)
            arima_preds = arima.predict(steps=days)
            all_predictions['arima'] = arima_preds
            
            # Evaluate
            test_size = int(len(close_prices) * 0.2)
            arima_test = ARIMAModel(order=(5, 1, 2))
            arima_test.fit(close_prices[:-test_size])
            all_metrics['arima'] = arima_test.evaluate(close_prices[-test_size:])
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")
        
        # 2. Prophet
        try:
            prophet = ProphetModel()
            prophet.fit(close_prices)
            prophet_result = prophet.get_forecast_values(periods=days)
            all_predictions['prophet'] = prophet_result['predictions']
            
            # Evaluate
            prophet_test = ProphetModel()
            prophet_test.fit(close_prices[:-test_size])
            all_metrics['prophet'] = prophet_test.evaluate(close_prices[-test_size:])
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
        
        if len(all_predictions) < 2:
            raise HTTPException(
                status_code=500, 
                detail="Need at least 2 models for ensemble"
            )
        
        # Calculate ensemble prediction
        if ensemble_type == "average":
            ensemble_preds = np.mean(list(all_predictions.values()), axis=0)
        elif ensemble_type == "weighted":
            # Weight by inverse MAE
            weights = {}
            total_inv_mae = 0
            for name, metrics in all_metrics.items():
                inv_mae = 1.0 / (metrics['mae'] + 1e-6)
                weights[name] = inv_mae
                total_inv_mae += inv_mae
            
            weights = {k: v / total_inv_mae for k, v in weights.items()}
            
            ensemble_preds = np.zeros(days)
            for name, preds in all_predictions.items():
                ensemble_preds += weights[name] * np.array(preds[:days])
        else:
            ensemble_preds = np.mean(list(all_predictions.values()), axis=0)
        
        # Get sentiment adjustment if requested
        sentiment_adjustment = 0
        sentiment_info = None
        
        if include_sentiment:
            try:
                from sqlalchemy import text
                
                query = text("""
                    SELECT avg_score, overall_sentiment 
                    FROM sentiment_summary 
                    WHERE symbol = :symbol 
                    ORDER BY date DESC 
                    LIMIT 1
                """)
                result = db.execute(query, {"symbol": symbol.upper()}).fetchone()
                
                if result:
                    sentiment_score = result[0]
                    sentiment_label = result[1]
                    
                    # Adjust prediction based on sentiment (-2% to +2%)
                    sentiment_adjustment = sentiment_score * 0.02  # Max 2% adjustment
                    ensemble_preds = ensemble_preds * (1 + sentiment_adjustment)
                    
                    sentiment_info = {
                        "score": round(sentiment_score, 3),
                        "label": sentiment_label,
                        "adjustment_percent": round(sentiment_adjustment * 100, 2)
                    }
            except Exception as e:
                logger.warning(f"Sentiment query failed: {e}")
        
        # Generate dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Calculate ensemble metrics (average of individual metrics)
        avg_mae = np.mean([m['mae'] for m in all_metrics.values()])
        avg_rmse = np.mean([m['rmse'] for m in all_metrics.values()])
        avg_mape = np.mean([m['mape'] for m in all_metrics.values()])
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": f"Ensemble ({ensemble_type})",
            "models_used": list(all_predictions.keys()),
            "current_price": current_price,
            "predictions": [
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "predicted_price": float(pred),
                    "change_percent": float((pred - current_price) / current_price * 100)
                }
                for date, pred in zip(future_dates, ensemble_preds)
            ],
            "individual_predictions": {
                name: [float(p) for p in preds[:days]]
                for name, preds in all_predictions.items()
            },
            "weights": weights if ensemble_type == "weighted" else None,
            "sentiment": sentiment_info,
            "metrics": {
                "ensemble": {
                    "mae": round(avg_mae, 2),
                    "rmse": round(avg_rmse, 2),
                    "mape": round(avg_mape, 2)
                },
                "individual": {
                    name: {
                        "mae": round(m['mae'], 2),
                        "rmse": round(m['rmse'], 2),
                        "mape": round(m['mape'], 2)
                    }
                    for name, m in all_metrics.items()
                }
            },
            "recommendation": "BUY" if ensemble_preds[-1] > current_price * 1.02 else 
                             ("SELL" if ensemble_preds[-1] < current_price * 0.98 else "HOLD"),
            "confidence": "HIGH" if len(all_predictions) >= 3 else "MEDIUM"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# MODEL COMPARISON ENDPOINT
# =====================================================

@router.get("/compare/{symbol}")
async def compare_models(
    symbol: str,
    days: int = Query(30, ge=7, le=90, description="Số ngày để đánh giá"),
    db: Session = Depends(get_db)
):
    """
    📊 So sánh performance của các ML models
    
    So sánh ARIMA vs Prophet vs LSTM vs GRU trên:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Square Error)
    - MAPE (Mean Absolute Percentage Error)
    
    Giúp chọn model tốt nhất cho mã cổ phiếu cụ thể.
    """
    try:
        from src.models.arima_model import ARIMAModel
        from src.models.prophet_model import ProphetModel
        
        # Get data
        df = get_stock_data(db, symbol, days=365)
        
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Need at least 100 days of data")
        
        close_prices = df['close']
        
        # Split train/test
        test_size = min(days, int(len(df) * 0.2))
        train = close_prices[:-test_size]
        test = close_prices[-test_size:]
        
        results = {}
        
        # 1. ARIMA
        try:
            arima = ARIMAModel(order=(5, 1, 2))
            arima.fit(train)
            arima_metrics = arima.evaluate(test)
            results['arima'] = {
                "model": "ARIMA(5,1,2)",
                "description": "Classical time series model",
                **arima_metrics
            }
        except Exception as e:
            results['arima'] = {"error": str(e)}
        
        # 2. Prophet
        try:
            prophet = ProphetModel()
            prophet.fit(train)
            prophet_metrics = prophet.evaluate(test)
            results['prophet'] = {
                "model": "Facebook Prophet",
                "description": "Decomposable time series model",
                **prophet_metrics
            }
        except Exception as e:
            results['prophet'] = {"error": str(e)}
        
        # 3. LSTM (if available and model exists)
        try:
            from src.models.lstm_gru_models import LSTMModel
            model_path = f"models/lstm_{symbol.upper()}.h5"
            
            if os.path.exists(model_path):
                lstm = LSTMModel(lookback=60)
                lstm.load_model(model_path)
                
                feature_df = df[['close', 'volume', 'high', 'low']].copy()
                feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
                feature_df = feature_df.dropna()
                
                lstm_metrics = lstm.evaluate(feature_df[-test_size:], target_col='close')
                results['lstm'] = {
                    "model": "LSTM",
                    "description": "Deep learning (pre-trained)",
                    **lstm_metrics
                }
            else:
                results['lstm'] = {
                    "status": "not_available",
                    "message": "No pre-trained LSTM model found",
                    "hint": "Run offline training script"
                }
        except Exception as e:
            results['lstm'] = {"status": "not_available", "error": str(e)}
        
        # 4. GRU (similar to LSTM)
        try:
            from src.models.lstm_gru_models import GRUModel
            model_path = f"models/gru_{symbol.upper()}.h5"
            
            if os.path.exists(model_path):
                gru = GRUModel(lookback=60)
                gru.load_model(model_path)
                
                feature_df = df[['close', 'volume', 'high', 'low']].copy()
                feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
                feature_df = feature_df.dropna()
                
                gru_metrics = gru.evaluate(feature_df[-test_size:], target_col='close')
                results['gru'] = {
                    "model": "GRU",
                    "description": "Deep learning (pre-trained)",
                    **gru_metrics
                }
            else:
                results['gru'] = {
                    "status": "not_available",
                    "message": "No pre-trained GRU model found",
                    "hint": "Run offline training script"
                }
        except Exception as e:
            results['gru'] = {"status": "not_available", "error": str(e)}
        
        # Find best model
        valid_results = {k: v for k, v in results.items() if 'mae' in v}
        
        if valid_results:
            best_by_mae = min(valid_results.items(), key=lambda x: x[1]['mae'])
            best_by_rmse = min(valid_results.items(), key=lambda x: x[1]['rmse'])
            best_by_mape = min(valid_results.items(), key=lambda x: x[1]['mape'])
        else:
            best_by_mae = best_by_rmse = best_by_mape = (None, {})
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "test_period": {
                "days": test_size,
                "from": str(test.index[0]),
                "to": str(test.index[-1])
            },
            "results": {
                name: {
                    **data,
                    "mae": round(data.get('mae', 0), 2),
                    "rmse": round(data.get('rmse', 0), 2),
                    "mape": round(data.get('mape', 0), 2)
                } if 'mae' in data else data
                for name, data in results.items()
            },
            "ranking": {
                "by_mae": best_by_mae[0],
                "by_rmse": best_by_rmse[0],
                "by_mape": best_by_mape[0]
            },
            "recommendation": {
                "best_model": best_by_mape[0] if best_by_mape[0] else "unknown",
                "reason": f"Lowest MAPE: {best_by_mape[1].get('mape', 'N/A')}%" if best_by_mape[0] else "Not enough data"
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI technical indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


# =====================================================
# BACKTESTING WITH ML MODELS
# =====================================================

@router.post("/backtest/{symbol}")
async def backtest_ml_strategy(
    symbol: str,
    model_type: str = Query("arima", description="Model: arima, prophet, ensemble"),
    days: int = Query(180, ge=30, le=365, description="Số ngày backtest"),
    initial_capital: float = Query(100_000_000, description="Vốn ban đầu (VND)"),
    threshold: float = Query(0.02, description="Ngưỡng signal (%)"),
    stop_loss: float = Query(0.05, description="Stop loss (%)"),
    take_profit: float = Query(0.10, description="Take profit (%)"),
    db: Session = Depends(get_db)
):
    """
    📊 Backtest chiến lược giao dịch với ML models
    
    Quy trình:
    1. Sử dụng predictions từ model để tạo signals (BUY/SELL/HOLD)
    2. Mô phỏng giao dịch với vốn ban đầu
    3. Tính các metrics: Sharpe, Sortino, Max Drawdown, Win Rate
    
    **Ví dụ chiến lược:**
    - BUY khi predicted_price > current_price + 2%
    - SELL khi predicted_price < current_price - 2%
    - Sử dụng Stop Loss 5% và Take Profit 10%
    
    **Metrics quan trọng:**
    - Sharpe Ratio > 1: Tốt, > 2: Rất tốt
    - Max Drawdown < 20%: Chấp nhận được
    - Win Rate > 50%: Profitable strategy
    """
    try:
        from src.backtest.backtesting_engine import BacktestingEngine, SignalGenerator
        from src.models.arima_model import ARIMAModel
        
        # Get data
        df = get_stock_data(db, symbol, days=days + 60)  # Extra for training
        
        if len(df) < days + 30:
            raise HTTPException(status_code=400, detail=f"Need at least {days + 30} days of data")
        
        close_prices = df['close']
        
        # Split train/test
        train_size = 60
        train_data = close_prices[:train_size]
        test_data_df = df.iloc[train_size:]
        
        # Generate predictions using chosen model
        all_predictions = {}
        
        if model_type == "arima":
            from src.models.arima_model import ARIMAModel
            
            for i in range(len(test_data_df) - 1):
                current_train = close_prices[:train_size + i]
                
                arima = ARIMAModel(order=(5, 1, 2))
                arima.fit(current_train)
                pred = arima.predict(steps=1)[0]
                
                next_date = test_data_df.index[i + 1]
                all_predictions[next_date] = float(pred)
        
        elif model_type == "prophet":
            from src.models.prophet_model import ProphetModel
            
            for i in range(len(test_data_df) - 1):
                current_train = close_prices[:train_size + i]
                
                prophet = ProphetModel()
                prophet.fit(current_train)
                result = prophet.get_forecast_values(periods=1)
                pred = result['predictions'][0]
                
                next_date = test_data_df.index[i + 1]
                all_predictions[next_date] = float(pred)
        
        elif model_type == "ensemble":
            # Use both ARIMA and Prophet
            from src.models.arima_model import ARIMAModel
            from src.models.prophet_model import ProphetModel
            
            for i in range(len(test_data_df) - 1):
                current_train = close_prices[:train_size + i]
                
                # ARIMA
                arima = ARIMAModel(order=(5, 1, 2))
                arima.fit(current_train)
                arima_pred = arima.predict(steps=1)[0]
                
                # Prophet  
                prophet = ProphetModel()
                prophet.fit(current_train)
                prophet_result = prophet.get_forecast_values(periods=1)
                prophet_pred = prophet_result['predictions'][0]
                
                # Average
                pred = (float(arima_pred) + float(prophet_pred)) / 2
                
                next_date = test_data_df.index[i + 1]
                all_predictions[next_date] = pred
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown model_type: {model_type}")
        
        # Convert to Series
        predictions_series = pd.Series(all_predictions)
        
        # Generate signals
        test_prices = test_data_df['close']
        signals = SignalGenerator.from_predictions(test_prices, predictions_series, threshold)
        
        # Prepare data for backtest
        test_data_for_bt = test_data_df.reset_index()
        test_data_for_bt.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        test_data_for_bt.columns = ['date', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Run backtest
        engine = BacktestingEngine(
            initial_capital=initial_capital,
            commission_rate=0.001,
            slippage=0.001
        )
        
        result = engine.run(
            test_data_for_bt,
            signals,
            symbol=symbol.upper(),
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit
        )
        
        # Return results
        result_dict = result.to_dict()
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "model": model_type,
            "backtest_params": {
                "days": days,
                "initial_capital": initial_capital,
                "threshold": threshold,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            },
            "results": result_dict,
            "interpretation": {
                "sharpe": "Excellent" if result.sharpe_ratio > 2 else ("Good" if result.sharpe_ratio > 1 else ("Acceptable" if result.sharpe_ratio > 0 else "Poor")),
                "max_drawdown": "Low risk" if result.max_drawdown_pct < 10 else ("Moderate" if result.max_drawdown_pct < 20 else "High risk"),
                "profitability": "Profitable" if result.total_return_pct > 0 else "Loss"
            },
            "recommendation": (
                "✅ Strategy shows promise" if result.sharpe_ratio > 1 and result.win_rate > 50
                else "⚠️ Strategy needs optimization" if result.total_return_pct > 0
                else "❌ Strategy not recommended"
            )
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# STATUS ENDPOINT
# =====================================================

@router.get("/status")
async def ml_models_status():
    """
    📋 Kiểm tra trạng thái các ML models
    
    Xem modules nào đã được cài đặt và sẵn sàng sử dụng.
    """
    status = {
        "arima": False,
        "prophet": False,
        "lstm_gru": False,
        "ensemble": False,
        "finbert": False
    }
    
    try:
        from statsmodels.tsa.arima.model import ARIMA
        status["arima"] = True
    except ImportError:
        pass
    
    try:
        from prophet import Prophet
        status["prophet"] = True
    except ImportError:
        pass
    
    try:
        import tensorflow
        status["lstm_gru"] = True
    except ImportError:
        pass
    
    try:
        from src.models.ensemble import StackingEnsemble
        status["ensemble"] = True
    except ImportError:
        pass
    
    try:
        from transformers import AutoTokenizer
        status["finbert"] = True
    except ImportError:
        pass
    
    # Check for pretrained models
    pretrained_models = []
    if os.path.exists("models"):
        for f in os.listdir("models"):
            if f.endswith(".h5") or f.endswith(".pkl"):
                pretrained_models.append(f)
    
    return {
        "status": "ok",
        "modules": status,
        "all_available": all(status.values()),
        "pretrained_models": pretrained_models,
        "endpoints": [
            "/api/ml/arima/predict/{symbol}",
            "/api/ml/prophet/predict/{symbol}",
            "/api/ml/lstm/predict/{symbol}",
            "/api/ml/gru/predict/{symbol}",
            "/api/ml/ensemble/predict/{symbol}",
            "/api/ml/unified/predict/{symbol}",
            "/api/ml/compare/{symbol}",
            "/api/ml/backtest/{symbol}"
        ]
    }


# =====================================================
# UNIFIED PREDICTION ENDPOINT (New Integrated Approach)
# =====================================================

@router.get("/unified/predict/{symbol}")
async def unified_predict(
    symbol: str,
    days: int = Query(5, ge=1, le=30, description="Số ngày dự đoán"),
    models: str = Query("all", description="Models: all, arima, prophet, lstm, gru, xgboost hoặc comma-separated"),
    include_sentiment: bool = Query(True, description="Tích hợp FinBERT sentiment"),
    include_technicals: bool = Query(True, description="Bao gồm technical indicators"),
    db: Session = Depends(get_db)
):
    """
    🚀 **UNIFIED PREDICTION** - Dự đoán tích hợp với tất cả models
    
    **Đây là endpoint được khuyến nghị sử dụng** vì nó:
    - Tự động chọn và chạy các model phù hợp
    - Kết hợp ARIMA + Prophet + LSTM + GRU + XGBoost
    - Tích hợp FinBERT sentiment từ tin tức
    - Tính toán confidence score và recommendation
    - Cung cấp detailed breakdown cho từng model
    
    **Response bao gồm:**
    - Dự đoán ensemble từ nhiều models
    - Individual predictions từ từng model
    - Sentiment analysis từ tin tức
    - Technical indicators hiện tại
    - Confidence level và trading recommendation
    
    **Ví dụ:**
    - `/api/ml/unified/predict/VNM?days=7` - Dự đoán VNM 7 ngày
    - `/api/ml/unified/predict/FPT?models=arima,prophet` - Chỉ dùng ARIMA và Prophet
    """
    try:
        # Get stock data
        df = get_stock_data(db, symbol, days=365)
        
        if len(df) < 60:
            raise HTTPException(
                status_code=400,
                detail=f"Không đủ dữ liệu lịch sử. Cần ít nhất 60 ngày, hiện có {len(df)} ngày."
            )
        
        # Ensure DatetimeIndex for Prophet compatibility
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        close_prices = df['close']
        current_price = float(close_prices.iloc[-1])
        
        # Determine which models to use
        if models == "all":
            model_list = ["arima", "prophet", "lstm", "gru", "xgboost"]
        else:
            model_list = [m.strip().lower() for m in models.split(",")]
        
        # Collect predictions
        predictions = {}
        metrics = {}
        model_details = {}
        
        test_size = int(len(close_prices) * 0.2)
        train_data = close_prices[:-test_size]
        test_data = close_prices[-test_size:]
        
        # 1. ARIMA
        if "arima" in model_list:
            arima_orders = [(5, 1, 2), (2, 1, 2), (3, 1, 1), (1, 1, 1)]  # Fallback orders
            arima_success = False
            
            for order in arima_orders:
                try:
                    from src.models.arima_model import ARIMAModel
                    arima = ARIMAModel(order=order)
                    arima.fit(close_prices)
                    arima_preds = arima.predict(steps=days)
                    predictions['arima'] = arima_preds.tolist() if hasattr(arima_preds, 'tolist') else list(arima_preds)
                    
                    # Evaluate
                    arima_test = ARIMAModel(order=order)
                    arima_test.fit(train_data)
                    metrics['arima'] = arima_test.evaluate(test_data)
                    model_details['arima'] = {
                        "name": f"ARIMA{order}",
                        "type": "Classical Time Series",
                        "order": str(order),
                        "description": "AutoRegressive Integrated Moving Average"
                    }
                    arima_success = True
                    logger.info(f"ARIMA succeeded with order {order}")
                    break
                except Exception as e:
                    logger.warning(f"ARIMA order {order} failed: {e}")
                    continue
            
            if not arima_success:
                model_details['arima'] = {"error": "All ARIMA orders failed"}
        
        # 2. Prophet
        if "prophet" in model_list:
            try:
                from src.models.prophet_model import ProphetModel
                prophet = ProphetModel()
                prophet.fit(close_prices)
                prophet_result = prophet.get_forecast_values(periods=days)
                predictions['prophet'] = prophet_result['predictions']
                
                # Evaluate
                prophet_test = ProphetModel()
                prophet_test.fit(train_data)
                metrics['prophet'] = prophet_test.evaluate(test_data)
                model_details['prophet'] = {
                    "name": "Facebook Prophet",
                    "type": "Decomposable Time Series",
                    "description": "Additive model với trend và seasonality"
                }
            except Exception as e:
                logger.warning(f"Prophet failed: {e}")
                model_details['prophet'] = {"error": str(e)}
        
        # 3. LSTM
        if "lstm" in model_list:
            try:
                from src.models.lstm_gru_models import LSTMModel
                model_path = f"models/lstm_{symbol.upper()}.h5"
                
                lstm = LSTMModel(lookback=60, forecast_horizon=1, units=[50, 50])
                
                # Prepare features
                feature_df = df[['close', 'volume', 'high', 'low']].copy()
                feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
                feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
                feature_df['rsi'] = calculate_rsi(feature_df['close'])
                feature_df = feature_df.dropna()
                
                # Validate data size (need at least lookback + test_size)
                min_required = 60 + test_size + 10  # lookback + test + buffer
                if len(feature_df) < min_required:
                    raise ValueError(f"Not enough data: {len(feature_df)} rows, need {min_required}")
                
                if os.path.exists(model_path):
                    lstm.load_model(model_path)
                    pretrained = True
                else:
                    lstm.fit(feature_df, target_col='close', epochs=30, batch_size=32, verbose=0)
                    pretrained = False
                
                # Predict
                lstm_preds = []
                try:
                    pred = lstm.predict(feature_df, target_col='close')
                    if len(pred) > 0:
                        # Use last prediction value for all forecast days
                        last_pred = float(pred[-1])
                        lstm_preds = [last_pred] * days
                except Exception as pred_err:
                    logger.warning(f"LSTM prediction failed: {pred_err}")
                    lstm_preds = []
                
                if len(lstm_preds) > 0:
                    predictions['lstm'] = lstm_preds
                    try:
                        metrics['lstm'] = lstm.evaluate(feature_df[-test_size:], target_col='close')
                    except:
                        metrics['lstm'] = {"error": "Evaluation failed"}
                else:
                    raise ValueError("No predictions generated")
                model_details['lstm'] = {
                    "name": "LSTM",
                    "type": "Deep Learning",
                    "layers": 2,
                    "units": [50, 50],
                    "pretrained": pretrained,
                    "description": "Long Short-Term Memory Neural Network"
                }
            except Exception as e:
                logger.warning(f"LSTM failed: {e}")
                model_details['lstm'] = {"error": str(e)}
        
        # 4. GRU
        if "gru" in model_list:
            try:
                from src.models.lstm_gru_models import GRUModel
                model_path = f"models/gru_{symbol.upper()}.h5"
                
                gru = GRUModel(lookback=60, forecast_horizon=1, units=[50, 50])
                
                # Prepare features
                feature_df = df[['close', 'volume', 'high', 'low']].copy()
                feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
                feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
                feature_df['rsi'] = calculate_rsi(feature_df['close'])
                feature_df = feature_df.dropna()
                
                # Validate data size
                min_required = 60 + test_size + 10
                if len(feature_df) < min_required:
                    raise ValueError(f"Not enough data: {len(feature_df)} rows, need {min_required}")
                
                if os.path.exists(model_path):
                    gru.load_model(model_path)
                    pretrained = True
                else:
                    gru.fit(feature_df, target_col='close', epochs=30, batch_size=32, verbose=0)
                    pretrained = False
                
                # Predict
                gru_preds = []
                try:
                    pred = gru.predict(feature_df, target_col='close')
                    if len(pred) > 0:
                        # Use last prediction value for all forecast days
                        last_pred = float(pred[-1])
                        gru_preds = [last_pred] * days
                except Exception as pred_err:
                    logger.warning(f"GRU prediction failed: {pred_err}")
                    gru_preds = []
                
                if len(gru_preds) > 0:
                    predictions['gru'] = gru_preds
                    try:
                        metrics['gru'] = gru.evaluate(feature_df[-test_size:], target_col='close')
                    except:
                        metrics['gru'] = {"error": "Evaluation failed"}
                else:
                    raise ValueError("No predictions generated")
                model_details['gru'] = {
                    "name": "GRU",
                    "type": "Deep Learning",
                    "layers": 2,
                    "units": [50, 50],
                    "pretrained": pretrained,
                    "description": "Gated Recurrent Unit Neural Network"
                }
            except Exception as e:
                logger.warning(f"GRU failed: {e}")
                model_details['gru'] = {"error": str(e)}
        
        # 5. XGBoost
        if "xgboost" in model_list:
            try:
                import xgboost as xgb
                
                # Create features
                feature_df = df.copy()
                feature_df['sma_5'] = feature_df['close'].rolling(5).mean()
                feature_df['sma_20'] = feature_df['close'].rolling(20).mean()
                feature_df['rsi'] = calculate_rsi(feature_df['close'])
                feature_df['return'] = feature_df['close'].pct_change()
                feature_df['volatility'] = feature_df['return'].rolling(10).std()
                feature_df['target'] = feature_df['close'].shift(-1)
                feature_df = feature_df.dropna()
                
                feature_cols = ['close', 'volume', 'sma_5', 'sma_20', 'rsi', 'return', 'volatility']
                X = feature_df[feature_cols].values
                y = feature_df['target'].values
                
                # Train
                xgb_model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=42
                )
                xgb_model.fit(X[:-test_size], y[:-test_size])
                
                # Predict
                xgb_preds = []
                last_features = X[-1].reshape(1, -1)
                for _ in range(days):
                    pred = xgb_model.predict(last_features)[0]
                    xgb_preds.append(float(pred))
                    # Update features for next prediction
                    last_features[0, 0] = pred  # Update close
                
                predictions['xgboost'] = xgb_preds
                
                # Evaluate
                y_pred = xgb_model.predict(X[-test_size:])
                metrics['xgboost'] = {
                    'mae': float(np.mean(np.abs(y_pred - y[-test_size:]))),
                    'rmse': float(np.sqrt(np.mean((y_pred - y[-test_size:]) ** 2))),
                    'mape': float(np.mean(np.abs((y[-test_size:] - y_pred) / y[-test_size:])) * 100)
                }
                model_details['xgboost'] = {
                    "name": "XGBoost",
                    "type": "Gradient Boosting",
                    "n_estimators": 100,
                    "max_depth": 5,
                    "description": "Extreme Gradient Boosting"
                }
            except Exception as e:
                logger.warning(f"XGBoost failed: {e}")
                model_details['xgboost'] = {"error": str(e)}
        
        # Check if we have predictions
        if not predictions:
            raise HTTPException(
                status_code=500,
                detail="Không có model nào chạy thành công. Vui lòng kiểm tra cấu hình."
            )
        
        # Calculate weighted ensemble prediction
        if len(predictions) == 0:
            raise HTTPException(
                status_code=400,
                detail=f"All models failed. Errors: {', '.join([f'{k}: {v.get(\"error\", \"unknown\")}' for k, v in model_details.items() if 'error' in v])}"
            )
        
        weights = {}
        total_inv_mae = 0
        for name, m in metrics.items():
            if 'mae' in m:
                inv_mae = 1.0 / (m['mae'] + 1e-6)
                weights[name] = inv_mae
                total_inv_mae += inv_mae
        
        if total_inv_mae > 0:
            weights = {k: v / total_inv_mae for k, v in weights.items()}
        else:
            # Equal weights if no metrics
            weights = {k: 1.0 / len(predictions) for k in predictions.keys()}
        
        # Weighted ensemble
        ensemble_preds = np.zeros(days)
        for name, preds in predictions.items():
            w = weights.get(name, 1.0 / len(predictions))
            for i, p in enumerate(preds[:days]):
                ensemble_preds[i] += w * p
        
        # Get sentiment if requested
        sentiment_info = None
        sentiment_adjustment = 0
        
        if include_sentiment:
            try:
                from sqlalchemy import text
                # Query from sentiment_analysis table using stock_id
                # First get stock_id
                stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
                if stock:
                    query = text("""
                        SELECT sentiment_label, confidence, analyzed_at 
                        FROM sentiment_analysis 
                        WHERE stock_id = :stock_id 
                        ORDER BY date DESC 
                        LIMIT 5
                    """)
                    result = db.execute(query, {"stock_id": stock.id}).fetchall()
                else:
                    result = []
                
                if result:
                    sentiments = [r[0] for r in result]
                    confidences = [r[1] for r in result]
                    
                    # Calculate average sentiment
                    sentiment_scores = []
                    for s, c in zip(sentiments, confidences):
                        if s == 'positive':
                            sentiment_scores.append(1.0 * c)
                        elif s == 'negative':
                            sentiment_scores.append(-1.0 * c)
                        else:
                            sentiment_scores.append(0.0)
                    
                    avg_sentiment = np.mean(sentiment_scores)
                    sentiment_adjustment = avg_sentiment * 0.02 * current_price  # 2% adjustment
                    
                    sentiment_info = {
                        "average_score": float(avg_sentiment),
                        "samples": len(result),
                        "last_analyzed": str(result[0][2]) if result else None,
                        "interpretation": "Positive" if avg_sentiment > 0.3 else ("Negative" if avg_sentiment < -0.3 else "Neutral"),
                        "adjustment_percent": round(avg_sentiment * 2, 2)
                    }
                    
                    # Apply sentiment adjustment to ensemble
                    ensemble_preds = ensemble_preds + sentiment_adjustment
            except Exception as e:
                logger.warning(f"Sentiment integration failed: {e}")
        
        # Technical indicators
        technicals = None
        if include_technicals:
            try:
                sma_10 = float(df['close'].rolling(10).mean().iloc[-1])
                sma_30 = float(df['close'].rolling(30).mean().iloc[-1])
                rsi = float(calculate_rsi(df['close']).iloc[-1])
                
                technicals = {
                    "sma_10": round(sma_10, 2),
                    "sma_30": round(sma_30, 2),
                    "rsi": round(rsi, 2),
                    "price_vs_sma10": round((current_price - sma_10) / sma_10 * 100, 2),
                    "price_vs_sma30": round((current_price - sma_30) / sma_30 * 100, 2),
                    "trend": "Uptrend" if sma_10 > sma_30 else "Downtrend",
                    "rsi_signal": "Overbought" if rsi > 70 else ("Oversold" if rsi < 30 else "Neutral")
                }
            except Exception as e:
                logger.warning(f"Technical indicators failed: {e}")
        
        # Generate dates
        last_date = df.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=days)
        
        # Calculate confidence
        model_count = len(predictions)
        avg_mape = np.mean([m.get('mape', 10) for m in metrics.values()]) if metrics else 10
        confidence = min(0.95, max(0.3, 1 - avg_mape / 100)) * (model_count / 5)
        confidence = min(0.95, confidence)
        
        confidence_level = "HIGH" if confidence > 0.7 else ("MEDIUM" if confidence > 0.5 else "LOW")
        
        # Generate recommendation
        final_pred = float(ensemble_preds[-1])
        pct_change = (final_pred - current_price) / current_price * 100
        
        if pct_change > 3:
            recommendation = "STRONG BUY"
        elif pct_change > 1:
            recommendation = "BUY"
        elif pct_change < -3:
            recommendation = "STRONG SELL"
        elif pct_change < -1:
            recommendation = "SELL"
        else:
            recommendation = "HOLD"
        
        return {
            "status": "success",
            "symbol": symbol.upper(),
            "current_price": current_price,
            "prediction_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            
            # Main prediction
            "ensemble_prediction": {
                "type": "weighted_average",
                "predictions": [
                    {
                        "date": future_dates[i].strftime("%Y-%m-%d"),
                        "predicted_price": round(float(ensemble_preds[i]), 2),
                        "change_percent": round((float(ensemble_preds[i]) - current_price) / current_price * 100, 2)
                    }
                    for i in range(len(ensemble_preds))
                ],
                "final_price": round(final_pred, 2),
                "total_change_percent": round(pct_change, 2)
            },
            
            # Individual model predictions
            "individual_predictions": {
                name: {
                    "model_info": model_details.get(name, {}),
                    "predictions": [round(float(p), 2) for p in preds[:days]],
                    "weight_in_ensemble": round(weights.get(name, 0), 4),
                    "metrics": {
                        "mae": round(metrics[name]['mae'], 2),
                        "rmse": round(metrics[name]['rmse'], 2),
                        "mape": round(metrics[name]['mape'], 2)
                    } if name in metrics and 'mae' in metrics[name] else None
                }
                for name, preds in predictions.items()
            },
            
            # Model weights
            "model_weights": {k: round(v, 4) for k, v in weights.items()},
            
            # Sentiment
            "sentiment": sentiment_info,
            
            # Technical indicators
            "technical_indicators": technicals,
            
            # Recommendation
            "analysis": {
                "recommendation": recommendation,
                "confidence": round(confidence, 2),
                "confidence_level": confidence_level,
                "models_used": list(predictions.keys()),
                "models_failed": [k for k, v in model_details.items() if "error" in v]
            },
            
            # Metadata
            "metadata": {
                "data_points": len(df),
                "training_split": f"{100-int(test_size/len(df)*100)}%/{int(test_size/len(df)*100)}%",
                "forecast_horizon": days,
                "models_requested": model_list,
                "sentiment_integrated": include_sentiment and sentiment_info is not None
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified prediction error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

