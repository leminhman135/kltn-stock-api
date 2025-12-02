"""
Upload Predictions Script
Chạy predictions từ models đã train và upload lên database

Usage:
    python scripts/upload_predictions.py --symbol VNM --days 5
    python scripts/upload_predictions.py --all --days 5
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# VN30 stocks
VN30_SYMBOLS = [
    "VNM", "VIC", "VHM", "VCB", "BID", "CTG", "TCB", "MBB", "HPG", "FPT",
    "MWG", "VPB", "PLX", "VJC", "GAS", "SAB", "MSN", "VRE", "NVL", "ACB"
]


class PredictionUploader:
    """Upload predictions từ trained models lên database"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
    
    def get_stock_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Lấy dữ liệu stock từ database"""
        from src.database.connection import get_db_session
        from src.database.models import Stock, StockPrice
        from sqlalchemy import desc
        
        with get_db_session() as db:
            stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
            
            if not stock:
                raise ValueError(f"Stock {symbol} not found")
            
            prices = db.query(StockPrice).filter(
                StockPrice.stock_id == stock.id
            ).order_by(desc(StockPrice.date)).limit(days).all()
            
            if not prices:
                raise ValueError(f"No price data for {symbol}")
            
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
            
            return df, stock.id
    
    def predict_arima(self, symbol: str, df: pd.DataFrame, days: int) -> list:
        """Predict with ARIMA"""
        import joblib
        
        model_path = self.models_dir / f"arima_{symbol.upper()}.pkl"
        
        if not model_path.exists():
            # Train on the fly
            from src.models.arima_model import ARIMAModel
            arima = ARIMAModel(order=(5, 1, 2))
            arima.fit(df['close'])
            predictions, lower, upper = arima.predict_with_confidence(steps=days)
        else:
            # Load pretrained
            saved = joblib.load(model_path)
            from src.models.arima_model import ARIMAModel
            arima = ARIMAModel(order=saved['order'])
            arima.fit(df['close'])
            predictions, lower, upper = arima.predict_with_confidence(steps=days)
        
        return predictions.tolist(), lower.tolist(), upper.tolist()
    
    def predict_prophet(self, symbol: str, df: pd.DataFrame, days: int) -> list:
        """Predict with Prophet"""
        import joblib
        
        model_path = self.models_dir / f"prophet_{symbol.upper()}.pkl"
        
        from src.models.prophet_model import ProphetModel
        prophet = ProphetModel()
        prophet.fit(df['close'])
        result = prophet.get_forecast_values(periods=days)
        
        return result['predictions'].tolist(), result['lower_bound'].tolist(), result['upper_bound'].tolist()
    
    def predict_lstm(self, symbol: str, df: pd.DataFrame, days: int) -> list:
        """Predict with LSTM"""
        model_path = self.models_dir / f"lstm_{symbol.upper()}.h5"
        
        if not model_path.exists():
            logger.warning(f"No LSTM model for {symbol}")
            return None, None, None
        
        from src.models.lstm_gru_models import LSTMModel
        import joblib
        
        lstm = LSTMModel(lookback=60)
        lstm.load_model(str(model_path))
        
        # Load scaler
        scaler_path = self.models_dir / f"lstm_{symbol.upper()}_scaler.pkl"
        if scaler_path.exists():
            lstm.data_preparator.scaler = joblib.load(scaler_path)
            lstm.data_preparator.fitted = True
        
        # Add features
        feature_df = df[['close', 'volume', 'high', 'low']].copy()
        feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
        feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
        feature_df = feature_df.dropna()
        
        predictions = []
        for i in range(days):
            pred = lstm.predict(feature_df, target_col='close')
            if len(pred) > 0:
                predictions.append(float(pred[-1]))
        
        return predictions, None, None
    
    def predict_gru(self, symbol: str, df: pd.DataFrame, days: int) -> list:
        """Predict with GRU"""
        model_path = self.models_dir / f"gru_{symbol.upper()}.h5"
        
        if not model_path.exists():
            logger.warning(f"No GRU model for {symbol}")
            return None, None, None
        
        from src.models.lstm_gru_models import GRUModel
        import joblib
        
        gru = GRUModel(lookback=60)
        gru.load_model(str(model_path))
        
        # Load scaler
        scaler_path = self.models_dir / f"gru_{symbol.upper()}_scaler.pkl"
        if scaler_path.exists():
            gru.data_preparator.scaler = joblib.load(scaler_path)
            gru.data_preparator.fitted = True
        
        # Add features
        feature_df = df[['close', 'volume', 'high', 'low']].copy()
        feature_df['sma_10'] = feature_df['close'].rolling(10).mean()
        feature_df['sma_30'] = feature_df['close'].rolling(30).mean()
        feature_df = feature_df.dropna()
        
        predictions = []
        for i in range(days):
            pred = gru.predict(feature_df, target_col='close')
            if len(pred) > 0:
                predictions.append(float(pred[-1]))
        
        return predictions, None, None
    
    def ensemble_predict(self, symbol: str, df: pd.DataFrame, days: int) -> list:
        """Ensemble prediction combining all models"""
        all_predictions = {}
        weights = {}
        
        # ARIMA
        try:
            arima_preds, _, _ = self.predict_arima(symbol, df, days)
            if arima_preds:
                all_predictions['arima'] = arima_preds
                weights['arima'] = 0.3  # Default weight
        except Exception as e:
            logger.warning(f"ARIMA failed: {e}")
        
        # Prophet
        try:
            prophet_preds, _, _ = self.predict_prophet(symbol, df, days)
            if prophet_preds:
                all_predictions['prophet'] = prophet_preds
                weights['prophet'] = 0.3
        except Exception as e:
            logger.warning(f"Prophet failed: {e}")
        
        # LSTM
        try:
            lstm_preds, _, _ = self.predict_lstm(symbol, df, days)
            if lstm_preds:
                all_predictions['lstm'] = lstm_preds
                weights['lstm'] = 0.2
        except Exception as e:
            logger.warning(f"LSTM failed: {e}")
        
        # GRU
        try:
            gru_preds, _, _ = self.predict_gru(symbol, df, days)
            if gru_preds:
                all_predictions['gru'] = gru_preds
                weights['gru'] = 0.2
        except Exception as e:
            logger.warning(f"GRU failed: {e}")
        
        if not all_predictions:
            raise ValueError("No models available for ensemble")
        
        # Weighted average
        total_weight = sum(weights[k] for k in all_predictions.keys())
        normalized_weights = {k: weights[k]/total_weight for k in all_predictions.keys()}
        
        ensemble_preds = np.zeros(days)
        for name, preds in all_predictions.items():
            ensemble_preds += normalized_weights[name] * np.array(preds[:days])
        
        return ensemble_preds.tolist(), all_predictions, normalized_weights
    
    def save_predictions_to_db(self, symbol: str, stock_id: int, 
                                predictions: dict, days: int):
        """Lưu predictions vào database"""
        from src.database.connection import get_db_session
        from src.database.models import Prediction, ModelMetrics
        from datetime import date as dt_date
        
        with get_db_session() as db:
            today = datetime.now().date()
            
            for model_name, preds in predictions.items():
                if preds is None:
                    continue
                
                for i, pred_value in enumerate(preds[:days]):
                    pred_date = today + timedelta(days=i+1)
                    
                    # Check if prediction already exists
                    existing = db.query(Prediction).filter(
                        Prediction.stock_id == stock_id,
                        Prediction.prediction_date == pred_date,
                        Prediction.model_name == model_name
                    ).first()
                    
                    if existing:
                        existing.predicted_price = float(pred_value)
                        existing.confidence = 0.7  # Default confidence
                        existing.created_at = datetime.now()
                    else:
                        new_pred = Prediction(
                            stock_id=stock_id,
                            model_name=model_name,
                            prediction_date=pred_date,
                            predicted_price=float(pred_value),
                            confidence=0.7
                        )
                        db.add(new_pred)
            
            db.commit()
            logger.info(f"✅ Saved predictions for {symbol} to database")
    
    def run_predictions(self, symbol: str, days: int = 5, save_to_db: bool = True):
        """Chạy predictions cho một symbol"""
        logger.info(f"Running predictions for {symbol}...")
        
        try:
            # Get data
            df, stock_id = self.get_stock_data(symbol)
            
            predictions = {}
            
            # ARIMA
            try:
                arima_preds, _, _ = self.predict_arima(symbol, df, days)
                predictions['arima'] = arima_preds
                logger.info(f"  ARIMA: {arima_preds}")
            except Exception as e:
                logger.warning(f"  ARIMA failed: {e}")
            
            # Prophet
            try:
                prophet_preds, _, _ = self.predict_prophet(symbol, df, days)
                predictions['prophet'] = prophet_preds
                logger.info(f"  Prophet: {prophet_preds}")
            except Exception as e:
                logger.warning(f"  Prophet failed: {e}")
            
            # LSTM
            try:
                lstm_preds, _, _ = self.predict_lstm(symbol, df, days)
                if lstm_preds:
                    predictions['lstm'] = lstm_preds
                    logger.info(f"  LSTM: {lstm_preds}")
            except Exception as e:
                logger.warning(f"  LSTM failed: {e}")
            
            # GRU
            try:
                gru_preds, _, _ = self.predict_gru(symbol, df, days)
                if gru_preds:
                    predictions['gru'] = gru_preds
                    logger.info(f"  GRU: {gru_preds}")
            except Exception as e:
                logger.warning(f"  GRU failed: {e}")
            
            # Ensemble
            try:
                ensemble_preds, _, _ = self.ensemble_predict(symbol, df, days)
                predictions['ensemble'] = ensemble_preds
                logger.info(f"  Ensemble: {ensemble_preds}")
            except Exception as e:
                logger.warning(f"  Ensemble failed: {e}")
            
            # Save to database
            if save_to_db and predictions:
                self.save_predictions_to_db(symbol, stock_id, predictions, days)
            
            return predictions
            
        except Exception as e:
            logger.error(f"❌ Predictions failed for {symbol}: {e}")
            return None


def main():
    parser = argparse.ArgumentParser(description="Upload predictions to database")
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Stock symbol'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Run for all VN30 stocks'
    )
    
    parser.add_argument(
        '--days', '-d',
        type=int,
        default=5,
        help='Number of days to predict'
    )
    
    parser.add_argument(
        '--models-dir', '-m',
        type=str,
        default='models',
        help='Directory containing trained models'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save to database (dry run)'
    )
    
    args = parser.parse_args()
    
    # Parse symbols
    if args.all:
        symbols = VN30_SYMBOLS
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        print("Please specify --symbol or --all")
        return
    
    uploader = PredictionUploader(models_dir=args.models_dir)
    
    logger.info(f"🚀 Running predictions for {len(symbols)} symbols")
    logger.info(f"Days: {args.days}")
    logger.info(f"Models dir: {args.models_dir}")
    
    results = {}
    
    for symbol in symbols:
        predictions = uploader.run_predictions(
            symbol=symbol,
            days=args.days,
            save_to_db=not args.no_save
        )
        if predictions:
            results[symbol] = predictions
    
    # Summary
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    
    for symbol, preds in results.items():
        print(f"\n{symbol}:")
        for model, values in preds.items():
            if values:
                print(f"  {model}: {[round(v, 2) for v in values[:3]]}...")


if __name__ == "__main__":
    main()
