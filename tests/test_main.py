"""
Unit Tests for KLTN Stock Prediction API
=========================================

Các test cases kiểm tra chức năng của các module chính:
- Data Collection
- Technical Indicators
- ML Models
- API Endpoints
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.scheduler.daily_scheduler import init_scheduler

# Start background scheduler
scheduler = init_scheduler()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def sample_price_data():
    """Generate sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    
    base_price = 50.0
    returns = np.random.normal(0, 0.02, 100)
    prices = base_price * np.cumprod(1 + returns)
    
    data = pd.DataFrame({
        'date': dates,
        'Open': prices * (1 + np.random.uniform(-0.01, 0.01, 100)),
        'High': prices * (1 + np.random.uniform(0, 0.02, 100)),
        'Low': prices * (1 - np.random.uniform(0, 0.02, 100)),
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, 100)
    })
    
    return data


@pytest.fixture
def sample_news_data():
    """Generate sample news data for testing"""
    return [
        {
            'title': 'VNM công bố lợi nhuận quý 3 tăng 15%',
            'content': 'Vinamilk báo cáo kết quả kinh doanh tích cực...',
            'source': 'CafeF',
            'published_at': '2024-11-28'
        },
        {
            'title': 'Thị trường chứng khoán giảm điểm',
            'content': 'VN-Index giảm 10 điểm trong phiên giao dịch...',
            'source': 'VnExpress',
            'published_at': '2024-11-27'
        }
    ]


# =============================================================================
# Technical Indicators Tests
# =============================================================================

class TestTechnicalIndicators:
    """Test cases for technical indicators calculation"""
    
    def test_sma_calculation(self, sample_price_data):
        """Test Simple Moving Average calculation"""
        from features.technical_indicators import TechnicalIndicators
        
        ti = TechnicalIndicators()
        result = ti.calculate_sma(sample_price_data['Close'], window=10)
        
        # Check output type
        assert isinstance(result, pd.Series)
        
        # Check length
        assert len(result) == len(sample_price_data)
        
        # First 9 values should be NaN (window - 1)
        assert result[:9].isna().all()
        
        # Check calculation
        expected = sample_price_data['Close'][:10].mean()
        assert abs(result.iloc[9] - expected) < 0.01
    
    def test_rsi_calculation(self, sample_price_data):
        """Test RSI calculation"""
        from features.technical_indicators import TechnicalIndicators
        
        ti = TechnicalIndicators()
        result = ti.calculate_rsi(sample_price_data['Close'], period=14)
        
        # RSI should be between 0 and 100
        valid_values = result.dropna()
        assert (valid_values >= 0).all()
        assert (valid_values <= 100).all()
    
    def test_macd_calculation(self, sample_price_data):
        """Test MACD calculation"""
        from features.technical_indicators import TechnicalIndicators
        
        ti = TechnicalIndicators()
        macd, signal, histogram = ti.calculate_macd(sample_price_data['Close'])
        
        # Check output types
        assert isinstance(macd, pd.Series)
        assert isinstance(signal, pd.Series)
        assert isinstance(histogram, pd.Series)
        
        # Histogram should be MACD - Signal
        valid_idx = ~(macd.isna() | signal.isna())
        np.testing.assert_array_almost_equal(
            histogram[valid_idx], 
            (macd - signal)[valid_idx]
        )
    
    def test_bollinger_bands(self, sample_price_data):
        """Test Bollinger Bands calculation"""
        from features.technical_indicators import TechnicalIndicators
        
        ti = TechnicalIndicators()
        upper, middle, lower = ti.calculate_bollinger_bands(
            sample_price_data['Close'], 
            window=20, 
            std_dev=2
        )
        
        # Upper should always be >= middle >= lower
        valid_idx = ~(upper.isna() | middle.isna() | lower.isna())
        assert (upper[valid_idx] >= middle[valid_idx]).all()
        assert (middle[valid_idx] >= lower[valid_idx]).all()


# =============================================================================
# Sentiment Analysis Tests
# =============================================================================

class TestSentimentAnalysis:
    """Test cases for sentiment analysis"""
    
    def test_keyword_sentiment(self):
        """Test keyword-based sentiment analysis"""
        from features.sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer()
        
        # Positive text
        positive_text = "Cổ phiếu tăng mạnh, lợi nhuận đạt kỷ lục"
        result = analyzer.analyze_keyword(positive_text)
        assert result['sentiment'] == 'positive'
        
        # Negative text
        negative_text = "Thị trường sụt giảm, thua lỗ nghiêm trọng"
        result = analyzer.analyze_keyword(negative_text)
        assert result['sentiment'] == 'negative'
        
        # Neutral text
        neutral_text = "Báo cáo thường niên công ty ABC"
        result = analyzer.analyze_keyword(neutral_text)
        assert result['sentiment'] == 'neutral'
    
    @pytest.mark.skip(reason="Requires FinBERT model download")
    def test_finbert_sentiment(self):
        """Test FinBERT sentiment analysis"""
        from features.sentiment_analysis import SentimentAnalyzer
        
        analyzer = SentimentAnalyzer(use_finbert=True)
        
        text = "Company reports record profits for Q3 2024"
        result = analyzer.analyze_finbert(text)
        
        assert 'sentiment' in result
        assert 'score' in result
        assert result['sentiment'] in ['positive', 'negative', 'neutral']


# =============================================================================
# ML Models Tests
# =============================================================================

class TestARIMAModel:
    """Test cases for ARIMA model"""
    
    def test_arima_train(self, sample_price_data):
        """Test ARIMA model training"""
        from models.arima_model import ARIMAPredictor
        
        model = ARIMAPredictor()
        model.fit(sample_price_data['Close'].values)
        
        assert model.model is not None
        assert model.is_fitted
    
    def test_arima_predict(self, sample_price_data):
        """Test ARIMA prediction"""
        from models.arima_model import ARIMAPredictor
        
        model = ARIMAPredictor()
        model.fit(sample_price_data['Close'].values)
        
        predictions = model.predict(days=7)
        
        # Check output
        assert len(predictions) == 7
        assert all(p > 0 for p in predictions)  # Prices should be positive


class TestLSTMModel:
    """Test cases for LSTM model"""
    
    def test_data_preparation(self, sample_price_data):
        """Test LSTM data preparation"""
        from models.lstm_gru_models import LSTMPredictor
        
        model = LSTMPredictor(sequence_length=60)
        X, y = model.prepare_data(sample_price_data['Close'].values)
        
        # Check shapes
        assert X.shape[0] == len(sample_price_data) - 60
        assert X.shape[1] == 60
        assert X.shape[2] == 1
        assert len(y) == len(sample_price_data) - 60
    
    @pytest.mark.skip(reason="Slow test - requires TensorFlow")
    def test_lstm_train_predict(self, sample_price_data):
        """Test LSTM training and prediction"""
        from models.lstm_gru_models import LSTMPredictor
        
        model = LSTMPredictor(sequence_length=60, epochs=5)
        model.fit(sample_price_data['Close'].values)
        
        predictions = model.predict(days=7)
        
        assert len(predictions) == 7


class TestEnsembleModel:
    """Test cases for Ensemble model"""
    
    def test_weighted_average(self):
        """Test weighted average ensemble"""
        from models.ensemble import WeightedAverageEnsemble
        
        # Mock predictions
        predictions = {
            'arima': [50, 51, 52],
            'prophet': [49, 50, 51],
            'lstm': [51, 52, 53],
            'gru': [50, 51, 52]
        }
        
        ensemble = WeightedAverageEnsemble()
        result = ensemble.combine(predictions)
        
        assert len(result) == 3
        # Result should be weighted average
        assert all(p > 0 for p in result)
    
    def test_inverse_rmse_weights(self):
        """Test weight calculation based on inverse RMSE"""
        from models.ensemble import WeightedAverageEnsemble
        
        rmse_scores = {
            'arima': 3.45,
            'prophet': 3.12,
            'lstm': 2.34,
            'gru': 2.51
        }
        
        ensemble = WeightedAverageEnsemble()
        weights = ensemble.calculate_weights(rmse_scores)
        
        # Sum of weights should be 1
        assert abs(sum(weights.values()) - 1.0) < 0.01
        
        # Lower RMSE should have higher weight
        assert weights['lstm'] > weights['arima']


# =============================================================================
# Backtesting Tests
# =============================================================================

class TestBacktestingEngine:
    """Test cases for backtesting engine"""
    
    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        from backtest.backtesting_engine import BacktestingEngine
        
        engine = BacktestingEngine(initial_capital=100000000)
        
        # Sample returns
        returns = pd.Series([0.01, -0.005, 0.02, -0.01, 0.015])
        
        sharpe = engine.calculate_sharpe_ratio(returns)
        
        # Sharpe ratio should be a number
        assert isinstance(sharpe, float)
    
    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        from backtest.backtesting_engine import BacktestingEngine
        
        engine = BacktestingEngine(initial_capital=100000000)
        
        # Sample equity curve
        equity = pd.Series([100, 105, 98, 103, 95, 110])
        
        max_dd = engine.calculate_max_drawdown(equity)
        
        # Max drawdown should be negative
        assert max_dd <= 0
        
        # Should be the maximum peak-to-trough decline
        assert max_dd == pytest.approx(-0.0952, rel=0.01)  # (95-105)/105
    
    def test_win_rate(self):
        """Test win rate calculation"""
        from backtest.backtesting_engine import BacktestingEngine
        
        engine = BacktestingEngine(initial_capital=100000000)
        
        # Sample trades
        trades = [
            {'profit': 1000},
            {'profit': -500},
            {'profit': 2000},
            {'profit': -300},
            {'profit': 1500}
        ]
        
        win_rate = engine.calculate_win_rate(trades)
        
        # 3 winning trades out of 5
        assert win_rate == 0.6


# =============================================================================
# API Endpoint Tests
# =============================================================================

class TestAPIEndpoints:
    """Test cases for FastAPI endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        from fastapi.testclient import TestClient
        from api_v2 import app
        return TestClient(app)
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
    
    def test_get_stocks(self, client):
        """Test get stocks endpoint"""
        response = client.get("/api/stocks")
        
        assert response.status_code == 200
        assert isinstance(response.json(), list)
    
    def test_get_prices(self, client):
        """Test get prices endpoint"""
        response = client.get("/api/prices/VNM?limit=10")
        
        # May return 200 with data or 404 if no data
        assert response.status_code in [200, 404]
        
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, list)
            if len(data) > 0:
                assert 'date' in data[0]
                assert 'close' in data[0]
    
    def test_prediction_endpoint(self, client):
        """Test prediction endpoint"""
        response = client.get("/api/predictions/quick/VNM?days=7")
        
        # May return 200 or 404/500 depending on data availability
        assert response.status_code in [200, 404, 500]
    
    def test_ml_status(self, client):
        """Test ML models status endpoint"""
        response = client.get("/api/ml/status")
        
        assert response.status_code == 200
        data = response.json()
        assert 'status' in data


# =============================================================================
# Data Validation Tests
# =============================================================================

class TestDataValidation:
    """Test cases for data validation"""
    
    def test_price_validation(self, sample_price_data):
        """Test price data validation"""
        from etl.etl_pipeline import DataValidator
        
        validator = DataValidator()
        result = validator.validate(sample_price_data)
        
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_invalid_prices(self):
        """Test validation with invalid prices"""
        from etl.etl_pipeline import DataValidator
        
        # Create invalid data (negative prices)
        invalid_data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=10),
            'Open': [50, -10, 52, 53, 54, 55, 56, 57, 58, 59],  # Invalid negative
            'High': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            'Low': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
            'Close': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            'Volume': [1000000] * 10
        })
        
        validator = DataValidator()
        result = validator.validate(invalid_data)
        
        assert not result.is_valid
        assert len(result.errors) > 0
    
    def test_missing_values(self):
        """Test validation with missing values"""
        from etl.etl_pipeline import DataValidator
        
        # Create data with missing values
        data = pd.DataFrame({
            'date': pd.date_range(start='2024-01-01', periods=10),
            'Open': [50, None, 52, 53, 54, 55, 56, 57, 58, 59],  # Missing value
            'High': [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            'Low': [49, 50, 51, 52, 53, 54, 55, 56, 57, 58],
            'Close': [50, 51, 52, 53, 54, 55, 56, 57, 58, 59],
            'Volume': [1000000] * 10
        })
        
        validator = DataValidator()
        result = validator.validate(data)
        
        assert len(result.warnings) > 0  # Should warn about missing values


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for full workflow"""
    
    @pytest.mark.skip(reason="Requires database connection")
    def test_full_prediction_flow(self, sample_price_data):
        """Test full prediction flow from data to prediction"""
        from models.ensemble import EnsemblePredictor
        from features.technical_indicators import TechnicalIndicators
        
        # 1. Calculate technical indicators
        ti = TechnicalIndicators()
        features = ti.calculate_all(sample_price_data)
        
        # 2. Train ensemble model
        ensemble = EnsemblePredictor()
        ensemble.fit(sample_price_data['Close'].values)
        
        # 3. Make predictions
        predictions = ensemble.predict(days=7)
        
        # 4. Validate output
        assert len(predictions) == 7
        assert all(p > 0 for p in predictions)


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
