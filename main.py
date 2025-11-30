"""
Main Orchestration Script - Stock Price Prediction System
Điều phối toàn bộ pipeline từ data collection đến prediction
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import các modules
from data_collection import DataCollectionPipeline
from data_processing import ETLPipeline
from features.technical_indicators import TechnicalIndicators
from features.sentiment_analysis import SentimentAnalysisPipeline
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel
from models.lstm_gru_models import LSTMModel, GRUModel
from models.ensemble import EnsembleFactory, evaluate_ensemble
from backtesting import BacktestEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StockPredictionSystem:
    """
    Hệ thống dự đoán giá cổ phiếu tổng hợp
    
    Pipeline:
    1. Data Collection: Thu thập từ APIs và web scraping
    2. Data Processing: ETL, cleaning, feature engineering
    3. Technical Indicators: Tính toán các chỉ báo kỹ thuật
    4. Sentiment Analysis: Phân tích cảm tính tin tức
    5. Model Training: Train ARIMA, Prophet, LSTM, GRU
    6. Ensemble: Kết hợp models với meta-learning
    7. Backtesting: Kiểm định ngược
    8. Prediction: Dự đoán giá tương lai
    """
    
    def __init__(self, config: dict = None):
        """
        Args:
            config: Dictionary chứa cấu hình hệ thống
        """
        self.config = config or self._default_config()
        
        # Initialize components
        self.data_collector = DataCollectionPipeline()
        self.etl = ETLPipeline()
        self.tech_indicators = TechnicalIndicators()
        self.sentiment_analyzer = None  # Lazy loading
        
        self.models = {}
        self.ensemble = None
        
        logger.info("StockPredictionSystem initialized")
    
    def _default_config(self) -> dict:
        """Cấu hình mặc định"""
        return {
            'symbols': ['VNM.VN', 'VIC.VN', 'HPG.VN'],
            'start_date': '2022-01-01',
            'end_date': datetime.now().strftime('%Y-%m-%d'),
            'train_split': 0.8,
            'models_to_train': ['arima', 'prophet', 'lstm', 'gru'],
            'ensemble_type': 'stacking',
            'backtest_strategy': 'long_only',
            'initial_capital': 100000,
        }
    
    def run_full_pipeline(self, symbol: str):
        """
        Chạy toàn bộ pipeline cho một symbol
        
        Args:
            symbol: Mã cổ phiếu
        """
        logger.info(f"=" * 60)
        logger.info(f"Starting full pipeline for {symbol}")
        logger.info(f"=" * 60)
        
        # 1. Data Collection
        logger.info("\n📊 Step 1: Data Collection")
        price_data, news_data = self.collect_data(symbol)
        
        if price_data.empty:
            logger.error(f"No data collected for {symbol}")
            return None
        
        # 2. Data Processing (ETL)
        logger.info("\n🔄 Step 2: Data Processing (ETL)")
        processed_data = self.process_data({symbol: price_data})
        
        # 3. Feature Engineering
        logger.info("\n⚙️ Step 3: Feature Engineering")
        data_with_features = self.add_features(processed_data)
        
        # 4. Sentiment Analysis
        logger.info("\n💭 Step 4: Sentiment Analysis")
        if not news_data.empty:
            sentiment_data = self.analyze_sentiment(news_data)
            # Merge sentiment với price data
            # data_with_features = self.merge_sentiment(data_with_features, sentiment_data)
        
        # 5. Train/Test Split
        train_size = int(len(data_with_features) * self.config['train_split'])
        train_data = data_with_features.iloc[:train_size]
        test_data = data_with_features.iloc[train_size:]
        
        logger.info(f"Train size: {len(train_data)}, Test size: {len(test_data)}")
        
        # 6. Model Training
        logger.info("\n🤖 Step 5: Model Training")
        self.train_models(train_data, symbol)
        
        # 7. Ensemble
        logger.info("\n🎯 Step 6: Creating Ensemble")
        self.create_ensemble(train_data, test_data)
        
        # 8. Backtesting
        logger.info("\n🔄 Step 7: Backtesting")
        backtest_results = self.run_backtest(test_data)
        
        # 9. Future Predictions
        logger.info("\n🔮 Step 8: Future Predictions")
        predictions = self.predict_future(data_with_features, periods=30)
        
        logger.info(f"\n✅ Pipeline completed for {symbol}")
        
        return {
            'data': data_with_features,
            'train_data': train_data,
            'test_data': test_data,
            'models': self.models,
            'ensemble': self.ensemble,
            'backtest_results': backtest_results,
            'predictions': predictions
        }
    
    def collect_data(self, symbol: str):
        """Thu thập dữ liệu"""
        try:
            price_data, news_data = self.data_collector.collect_all_data(
                [symbol],
                self.config['start_date'],
                self.config['end_date']
            )
            
            price_df = price_data.get(symbol, pd.DataFrame())
            
            logger.info(f"Collected {len(price_df)} price records, {len(news_data)} news articles")
            return price_df, news_data
        
        except Exception as e:
            logger.error(f"Error collecting data: {str(e)}")
            return pd.DataFrame(), pd.DataFrame()
    
    def process_data(self, data_dict: dict) -> pd.DataFrame:
        """Xử lý dữ liệu với ETL pipeline"""
        try:
            processed_df = self.etl.process_price_data(data_dict, save_format='csv')
            logger.info(f"Processed {len(processed_df)} records")
            return processed_df
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            return pd.DataFrame()
    
    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm technical indicators"""
        try:
            df_with_features = self.tech_indicators.add_all_indicators(df)
            logger.info(f"Added {len(df_with_features.columns) - len(df.columns)} indicators")
            return df_with_features
        except Exception as e:
            logger.error(f"Error adding features: {str(e)}")
            return df
    
    def analyze_sentiment(self, news_df: pd.DataFrame) -> pd.DataFrame:
        """Phân tích sentiment"""
        try:
            if self.sentiment_analyzer is None:
                self.sentiment_analyzer = SentimentAnalysisPipeline()
            
            _, daily_sentiment = self.sentiment_analyzer.process_news(news_df)
            logger.info(f"Analyzed sentiment for {len(daily_sentiment)} days")
            return daily_sentiment
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return pd.DataFrame()
    
    def train_models(self, train_data: pd.DataFrame, symbol: str):
        """Train các models"""
        models_to_train = self.config['models_to_train']
        
        for model_name in models_to_train:
            try:
                logger.info(f"Training {model_name.upper()}...")
                
                if model_name == 'arima':
                    model = ARIMAModel()
                    model.fit(train_data['close'], auto_order=True)
                    self.models[model_name] = model
                
                elif model_name == 'prophet':
                    model = ProphetModel()
                    model.fit(train_data['close'])
                    self.models[model_name] = model
                
                elif model_name == 'lstm':
                    model = LSTMModel(lookback=60, units=[50, 50])
                    model.fit(train_data, epochs=50, verbose=0)
                    self.models[model_name] = model
                
                elif model_name == 'gru':
                    model = GRUModel(lookback=60, units=[50, 50])
                    model.fit(train_data, epochs=50, verbose=0)
                    self.models[model_name] = model
                
                logger.info(f"{model_name.upper()} training completed")
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
    
    def create_ensemble(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """Tạo ensemble model"""
        try:
            ensemble_type = self.config['ensemble_type']
            
            self.ensemble = EnsembleFactory.create_ensemble(
                ensemble_type=ensemble_type,
                models=self.models,
                meta_model_type='ridge'
            )
            
            # Fit ensemble trên validation data
            if hasattr(self.ensemble, 'fit'):
                self.ensemble.fit(test_data[:len(test_data)//2])
            
            logger.info(f"Created {ensemble_type} ensemble")
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {str(e)}")
    
    def run_backtest(self, test_data: pd.DataFrame) -> dict:
        """Chạy backtesting"""
        try:
            # Get predictions
            if self.ensemble:
                predictions = self.ensemble.predict(test_data)
            else:
                predictions = test_data['close'].values
            
            # Run backtest
            engine = BacktestEngine(
                initial_capital=self.config['initial_capital'],
                commission=0.001
            )
            
            results = engine.run_backtest(
                data=test_data,
                predictions=predictions,
                strategy=self.config['backtest_strategy']
            )
            
            logger.info(f"Backtest completed: Return {results['total_return_pct']:.2f}%, "
                       f"Sharpe {results['sharpe_ratio']:.2f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in backtesting: {str(e)}")
            return {}
    
    def predict_future(self, data: pd.DataFrame, periods: int = 30) -> np.ndarray:
        """Dự đoán tương lai"""
        try:
            if self.ensemble:
                predictions = self.ensemble.predict(data.tail(periods))
            else:
                predictions = np.array([])
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions
            
        except Exception as e:
            logger.error(f"Error predicting: {str(e)}")
            return np.array([])


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Stock Price Prediction System')
    parser.add_argument('--symbol', type=str, default='VNM.VN',
                       help='Stock symbol to analyze')
    parser.add_argument('--start-date', type=str, default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--mode', type=str, default='full',
                       choices=['full', 'predict', 'backtest', 'train'],
                       help='Execution mode')
    
    args = parser.parse_args()
    
    # Configure
    config = {
        'symbols': [args.symbol],
        'start_date': args.start_date,
        'end_date': args.end_date or datetime.now().strftime('%Y-%m-%d'),
        'train_split': 0.8,
        'models_to_train': ['arima', 'prophet'],  # Start with simpler models
        'ensemble_type': 'weighted',
        'backtest_strategy': 'long_only',
        'initial_capital': 100000,
    }
    
    # Initialize system
    system = StockPredictionSystem(config)
    
    # Run
    if args.mode == 'full':
        results = system.run_full_pipeline(args.symbol)
        
        if results:
            print("\n" + "="*60)
            print("PIPELINE RESULTS")
            print("="*60)
            print(f"Symbol: {args.symbol}")
            print(f"Data points: {len(results['data'])}")
            print(f"Models trained: {len(results['models'])}")
            
            if results['backtest_results']:
                br = results['backtest_results']
                print(f"\nBacktest Results:")
                print(f"  Total Return: {br['total_return_pct']:.2f}%")
                print(f"  Sharpe Ratio: {br['sharpe_ratio']:.2f}")
                print(f"  Max Drawdown: {br['max_drawdown_pct']:.2f}%")
                print(f"  Win Rate: {br['win_rate_pct']:.2f}%")
    
    else:
        print(f"Mode '{args.mode}' not yet implemented")


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║          STOCK PRICE PREDICTION SYSTEM v1.0               ║
    ║                                                           ║
    ║  Comprehensive AI/ML system for stock price forecasting  ║
    ║                                                           ║
    ║  Features:                                                ║
    ║  - Multiple models: ARIMA, Prophet, LSTM, GRU             ║
    ║  - Ensemble learning with meta-learning                   ║
    ║  - Sentiment analysis with FinBERT                        ║
    ║  - Backtesting engine                                     ║
    ║  - Technical indicators                                   ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    main()
