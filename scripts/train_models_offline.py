"""
Offline Training Script for ML Models
Train ARIMA, Prophet, LSTM, GRU models và lưu weights

Usage:
    python scripts/train_models_offline.py --symbol VNM --models arima,prophet,lstm,gru
    python scripts/train_models_offline.py --all  # Train cho tất cả VN30
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
    "MWG", "VPB", "PLX", "VJC", "GAS", "SAB", "MSN", "VRE", "NVL", "ACB",
    "GVR", "STB", "POW", "BCM", "SSI", "VND", "TPB", "HDB", "PDR", "SHB"
]


class ModelTrainer:
    """Train và lưu các ML models"""
    
    def __init__(self, output_dir: str = "models"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.results = {
            "trained_at": datetime.now().isoformat(),
            "models": {}
        }
    
    def get_stock_data(self, symbol: str, days: int = 365) -> pd.DataFrame:
        """Lấy dữ liệu từ database hoặc API"""
        try:
            from src.database.connection import get_db_session
            from src.database.models import Stock, StockPrice
            from sqlalchemy import desc
            
            with get_db_session() as db:
                stock = db.query(Stock).filter(Stock.symbol == symbol.upper()).first()
                
                if not stock:
                    logger.warning(f"Stock {symbol} not found in database")
                    return self._fetch_from_api(symbol, days)
                
                prices = db.query(StockPrice).filter(
                    StockPrice.stock_id == stock.id
                ).order_by(desc(StockPrice.date)).limit(days).all()
                
                if not prices or len(prices) < 100:
                    logger.warning(f"Not enough data in DB for {symbol}, fetching from API")
                    return self._fetch_from_api(symbol, days)
                
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
                
                logger.info(f"Loaded {len(df)} days of data for {symbol} from database")
                return df
                
        except Exception as e:
            logger.warning(f"Database error: {e}, fetching from API")
            return self._fetch_from_api(symbol, days)
    
    def _fetch_from_api(self, symbol: str, days: int) -> pd.DataFrame:
        """Fetch dữ liệu từ VNDirect API"""
        try:
            from src.data_collection import VNDirectAPI
            
            vndirect = VNDirectAPI()
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            df = vndirect.get_stock_price(
                symbol=symbol.upper(),
                from_date=start_date.strftime('%Y-%m-%d'),
                to_date=end_date.strftime('%Y-%m-%d')
            )
            
            if df.empty:
                raise ValueError(f"No data available for {symbol}")
            
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            # Standardize column names
            df.columns = [c.lower() for c in df.columns]
            
            logger.info(f"Fetched {len(df)} days of data for {symbol} from API")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return pd.DataFrame()
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm technical indicators làm features"""
        df = df.copy()
        
        # Moving averages
        df['sma_10'] = df['close'].rolling(10).mean()
        df['sma_30'] = df['close'].rolling(30).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['close'].ewm(span=12, adjust=False).mean()
        exp2 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_std'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
        df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
        
        # Price change
        df['price_change'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        # Drop NaN
        df = df.dropna()
        
        return df
    
    def train_arima(self, symbol: str, df: pd.DataFrame) -> dict:
        """Train và lưu ARIMA model"""
        try:
            from src.models.arima_model import ARIMAModel
            import joblib
            
            logger.info(f"Training ARIMA for {symbol}...")
            
            close_prices = df['close']
            
            # Auto-select order
            arima = ARIMAModel()
            arima.fit(close_prices, auto_order=True)
            
            # Evaluate
            test_size = int(len(close_prices) * 0.2)
            train = close_prices[:-test_size]
            test = close_prices[-test_size:]
            
            arima_test = ARIMAModel(order=arima.order)
            arima_test.fit(train)
            metrics = arima_test.evaluate(test)
            
            # Save model
            model_path = self.output_dir / f"arima_{symbol.upper()}.pkl"
            joblib.dump({
                'order': arima.order,
                'fitted_model': arima.fitted_model,
                'trained_at': datetime.now().isoformat()
            }, model_path)
            
            result = {
                "status": "success",
                "model_path": str(model_path),
                "order": arima.order,
                "aic": float(arima.fitted_model.aic),
                "bic": float(arima.fitted_model.bic),
                "metrics": {
                    "mae": round(metrics['mae'], 4),
                    "rmse": round(metrics['rmse'], 4),
                    "mape": round(metrics['mape'], 4)
                }
            }
            
            logger.info(f"✅ ARIMA trained for {symbol}: MAE={metrics['mae']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ ARIMA training failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def train_prophet(self, symbol: str, df: pd.DataFrame) -> dict:
        """Train và lưu Prophet model"""
        try:
            from src.models.prophet_model import ProphetModel
            import joblib
            
            logger.info(f"Training Prophet for {symbol}...")
            
            close_prices = df['close']
            
            # Initialize and train
            prophet = ProphetModel(
                yearly_seasonality=True,
                weekly_seasonality=True,
                changepoint_prior_scale=0.05
            )
            prophet.fit(close_prices)
            
            # Evaluate
            test_size = int(len(close_prices) * 0.2)
            train = close_prices[:-test_size]
            test = close_prices[-test_size:]
            
            prophet_test = ProphetModel(
                yearly_seasonality=True,
                weekly_seasonality=True
            )
            prophet_test.fit(train)
            metrics = prophet_test.evaluate(test)
            
            # Save model (Prophet models are large, save only essential parts)
            model_path = self.output_dir / f"prophet_{symbol.upper()}.pkl"
            joblib.dump({
                'model': prophet.model,
                'train_data': prophet.train_data,
                'trained_at': datetime.now().isoformat()
            }, model_path)
            
            result = {
                "status": "success",
                "model_path": str(model_path),
                "metrics": {
                    "mae": round(metrics['mae'], 4),
                    "rmse": round(metrics['rmse'], 4),
                    "mape": round(metrics['mape'], 4)
                }
            }
            
            logger.info(f"✅ Prophet trained for {symbol}: MAE={metrics['mae']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Prophet training failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def train_lstm(self, symbol: str, df: pd.DataFrame, 
                   epochs: int = 100, lookback: int = 60) -> dict:
        """Train và lưu LSTM model"""
        try:
            from src.models.lstm_gru_models import LSTMModel
            
            logger.info(f"Training LSTM for {symbol} (epochs={epochs}, lookback={lookback})...")
            
            # Add features
            feature_df = self.add_features(df)
            
            # Select features
            features = ['close', 'volume', 'sma_10', 'sma_30', 'rsi', 'macd']
            feature_df = feature_df[[col for col in features if col in feature_df.columns]]
            
            if len(feature_df) < lookback + 50:
                raise ValueError(f"Not enough data: {len(feature_df)} rows")
            
            # Initialize and train
            lstm = LSTMModel(
                lookback=lookback,
                forecast_horizon=1,
                units=[100, 50],
                dropout=0.2
            )
            
            lstm.fit(
                feature_df,
                target_col='close',
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate
            test_size = int(len(feature_df) * 0.2)
            metrics = lstm.evaluate(feature_df[-test_size:], target_col='close')
            
            # Save model
            model_path = self.output_dir / f"lstm_{symbol.upper()}.h5"
            lstm.save_model(str(model_path))
            
            # Save scaler separately
            scaler_path = self.output_dir / f"lstm_{symbol.upper()}_scaler.pkl"
            import joblib
            joblib.dump(lstm.data_preparator.scaler, scaler_path)
            
            result = {
                "status": "success",
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "params": {
                    "lookback": lookback,
                    "epochs": epochs,
                    "units": [100, 50],
                    "dropout": 0.2
                },
                "metrics": {
                    "mae": round(metrics['mae'], 4),
                    "rmse": round(metrics['rmse'], 4),
                    "mape": round(metrics['mape'], 4)
                }
            }
            
            logger.info(f"✅ LSTM trained for {symbol}: MAE={metrics['mae']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ LSTM training failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def train_gru(self, symbol: str, df: pd.DataFrame,
                  epochs: int = 100, lookback: int = 60) -> dict:
        """Train và lưu GRU model"""
        try:
            from src.models.lstm_gru_models import GRUModel
            
            logger.info(f"Training GRU for {symbol} (epochs={epochs}, lookback={lookback})...")
            
            # Add features
            feature_df = self.add_features(df)
            
            # Select features
            features = ['close', 'volume', 'sma_10', 'sma_30', 'rsi', 'macd']
            feature_df = feature_df[[col for col in features if col in feature_df.columns]]
            
            if len(feature_df) < lookback + 50:
                raise ValueError(f"Not enough data: {len(feature_df)} rows")
            
            # Initialize and train
            gru = GRUModel(
                lookback=lookback,
                forecast_horizon=1,
                units=[100, 50],
                dropout=0.2
            )
            
            gru.fit(
                feature_df,
                target_col='close',
                epochs=epochs,
                batch_size=32,
                validation_split=0.2,
                verbose=1
            )
            
            # Evaluate
            test_size = int(len(feature_df) * 0.2)
            metrics = gru.evaluate(feature_df[-test_size:], target_col='close')
            
            # Save model
            model_path = self.output_dir / f"gru_{symbol.upper()}.h5"
            gru.save_model(str(model_path))
            
            # Save scaler
            scaler_path = self.output_dir / f"gru_{symbol.upper()}_scaler.pkl"
            import joblib
            joblib.dump(gru.data_preparator.scaler, scaler_path)
            
            result = {
                "status": "success",
                "model_path": str(model_path),
                "scaler_path": str(scaler_path),
                "params": {
                    "lookback": lookback,
                    "epochs": epochs,
                    "units": [100, 50],
                    "dropout": 0.2
                },
                "metrics": {
                    "mae": round(metrics['mae'], 4),
                    "rmse": round(metrics['rmse'], 4),
                    "mape": round(metrics['mape'], 4)
                }
            }
            
            logger.info(f"✅ GRU trained for {symbol}: MAE={metrics['mae']:.2f}")
            return result
            
        except Exception as e:
            logger.error(f"❌ GRU training failed for {symbol}: {e}")
            return {"status": "error", "error": str(e)}
    
    def train_all_models(self, symbol: str, models: list = None,
                        epochs: int = 100, lookback: int = 60) -> dict:
        """Train tất cả models cho một symbol"""
        if models is None:
            models = ['arima', 'prophet', 'lstm', 'gru']
        
        logger.info(f"=" * 60)
        logger.info(f"Training models for {symbol}: {models}")
        logger.info(f"=" * 60)
        
        # Get data
        df = self.get_stock_data(symbol)
        
        if df.empty or len(df) < 100:
            return {
                "symbol": symbol,
                "status": "error",
                "error": "Not enough data"
            }
        
        results = {
            "symbol": symbol,
            "data_points": len(df),
            "date_range": {
                "from": str(df.index[0]),
                "to": str(df.index[-1])
            },
            "models": {}
        }
        
        # Train each model
        if 'arima' in models:
            results['models']['arima'] = self.train_arima(symbol, df)
        
        if 'prophet' in models:
            results['models']['prophet'] = self.train_prophet(symbol, df)
        
        if 'lstm' in models:
            results['models']['lstm'] = self.train_lstm(symbol, df, epochs, lookback)
        
        if 'gru' in models:
            results['models']['gru'] = self.train_gru(symbol, df, epochs, lookback)
        
        # Update global results
        self.results['models'][symbol] = results
        
        return results
    
    def save_results(self):
        """Lưu kết quả training"""
        results_path = self.output_dir / "training_results.json"
        
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train ML models offline")
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        help='Stock symbol to train (e.g., VNM, FPT)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., VNM,FPT,VCB)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Train for all VN30 stocks'
    )
    
    parser.add_argument(
        '--models', '-m',
        type=str,
        default='arima,prophet',
        help='Comma-separated list of models to train (arima,prophet,lstm,gru)'
    )
    
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of epochs for LSTM/GRU training'
    )
    
    parser.add_argument(
        '--lookback', '-l',
        type=int,
        default=60,
        help='Lookback window for LSTM/GRU'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='models',
        help='Output directory for saved models'
    )
    
    args = parser.parse_args()
    
    # Parse models
    models = [m.strip().lower() for m in args.models.split(',')]
    
    # Parse symbols
    if args.all:
        symbols = VN30_SYMBOLS
    elif args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(',')]
    elif args.symbol:
        symbols = [args.symbol.upper()]
    else:
        print("Please specify --symbol, --symbols, or --all")
        return
    
    # Initialize trainer
    trainer = ModelTrainer(output_dir=args.output)
    
    logger.info(f"🚀 Starting offline training")
    logger.info(f"Symbols: {symbols}")
    logger.info(f"Models: {models}")
    logger.info(f"Output: {args.output}")
    
    # Train
    for symbol in symbols:
        try:
            trainer.train_all_models(
                symbol=symbol,
                models=models,
                epochs=args.epochs,
                lookback=args.lookback
            )
        except Exception as e:
            logger.error(f"Failed to train {symbol}: {e}")
    
    # Save results
    trainer.save_results()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    for symbol, result in trainer.results['models'].items():
        print(f"\n{symbol}:")
        if 'models' in result:
            for model_name, model_result in result['models'].items():
                if model_result.get('status') == 'success':
                    metrics = model_result.get('metrics', {})
                    print(f"  ✅ {model_name.upper()}: MAE={metrics.get('mae', 'N/A')}, MAPE={metrics.get('mape', 'N/A')}%")
                else:
                    print(f"  ❌ {model_name.upper()}: {model_result.get('error', 'Failed')}")
    
    print("\n" + "=" * 60)
    print(f"Models saved to: {args.output}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
