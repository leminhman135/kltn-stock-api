"""SQLAlchemy models for KLTN stock prediction system."""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, 
    Text, Boolean, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from .connection import Base


class Stock(Base):
    """Stock information table."""
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200), nullable=False)
    exchange = Column(String(50), default="HOSE")  # HOSE, HNX, UPCOM
    sector = Column(String(100))
    industry = Column(String(100))
    market_cap = Column(Float)
    outstanding_shares = Column(Float)
    is_active = Column(Boolean, default=True)
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    prices = relationship("StockPrice", back_populates="stock", cascade="all, delete-orphan")
    indicators = relationship("TechnicalIndicator", back_populates="stock", cascade="all, delete-orphan")
    sentiments = relationship("SentimentAnalysis", back_populates="stock", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="stock", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Stock(symbol='{self.symbol}', name='{self.name}')>"


class StockPrice(Base):
    """Historical stock prices (OHLCV data)."""
    __tablename__ = "stock_prices"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # OHLCV
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    
    # Additional fields
    adjusted_close = Column(Float)  # Adjusted for dividends/splits
    change_percent = Column(Float)
    
    # Data source tracking
    source = Column(String(50), default="vndirect")  # vndirect, yahoo, manual
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="prices")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_stock_date'),
        Index('ix_stock_prices_stock_date', 'stock_id', 'date'),
    )
    
    def __repr__(self):
        return f"<StockPrice(stock_id={self.stock_id}, date='{self.date}', close={self.close})>"


class TechnicalIndicator(Base):
    """Technical indicators calculated from price data."""
    __tablename__ = "technical_indicators"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Moving Averages
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Momentum Indicators
    rsi_14 = Column(Float)  # Relative Strength Index
    macd = Column(Float)  # MACD line
    macd_signal = Column(Float)  # Signal line
    macd_histogram = Column(Float)
    
    # Volatility Indicators
    bb_upper = Column(Float)  # Bollinger Bands Upper
    bb_middle = Column(Float)  # Bollinger Bands Middle
    bb_lower = Column(Float)  # Bollinger Bands Lower
    atr_14 = Column(Float)  # Average True Range
    
    # Volume Indicators
    obv = Column(Float)  # On-Balance Volume
    
    # Trend Indicators
    adx_14 = Column(Float)  # Average Directional Index
    
    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="indicators")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_indicator_stock_date'),
        Index('ix_indicators_stock_date', 'stock_id', 'date'),
    )
    
    def __repr__(self):
        return f"<TechnicalIndicator(stock_id={self.stock_id}, date='{self.date}')>"


class SentimentAnalysis(Base):
    """Sentiment analysis results from news and social media."""
    __tablename__ = "sentiment_analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False, index=True)
    
    # Sentiment scores
    sentiment_score = Column(Float)  # -1 to 1 (negative to positive)
    sentiment_label = Column(String(20))  # negative, neutral, positive
    confidence = Column(Float)  # 0 to 1
    
    # News aggregation
    news_count = Column(Integer, default=0)
    positive_count = Column(Integer, default=0)
    negative_count = Column(Integer, default=0)
    neutral_count = Column(Integer, default=0)
    
    # Source breakdown
    sources = Column(JSON)  # {"cafef": 5, "vietstock": 3, "ndh": 2}
    
    # Sample headlines
    top_headlines = Column(JSON)  # List of top 5 headlines
    
    # Model used
    model_name = Column(String(100), default="finbert")  # finbert, phobert, etc.
    
    # Metadata
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="sentiments")
    
    # Unique constraint
    __table_args__ = (
        UniqueConstraint('stock_id', 'date', name='uq_sentiment_stock_date'),
        Index('ix_sentiment_stock_date', 'stock_id', 'date'),
    )
    
    def __repr__(self):
        return f"<SentimentAnalysis(stock_id={self.stock_id}, date='{self.date}', score={self.sentiment_score})>"


class ModelMetrics(Base):
    """Training metrics for different prediction models."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)  # ARIMA, Prophet, LSTM, GRU, Ensemble
    stock_symbol = Column(String(20), nullable=False, index=True)
    version = Column(String(50))  # Model version (e.g., "v1.0", "20241130")
    
    # Performance metrics
    mae = Column(Float)  # Mean Absolute Error
    rmse = Column(Float)  # Root Mean Squared Error
    mape = Column(Float)  # Mean Absolute Percentage Error
    r2_score = Column(Float)  # R² Score
    
    # Training info
    train_start_date = Column(Date)
    train_end_date = Column(Date)
    test_start_date = Column(Date)
    test_end_date = Column(Date)
    training_samples = Column(Integer)
    test_samples = Column(Integer)
    
    # Model parameters
    hyperparameters = Column(JSON)  # Store model config as JSON
    
    # File paths
    model_file_path = Column(String(500))  # Path to saved model file
    
    # Status
    is_active = Column(Boolean, default=True)  # Current production model
    
    # Metadata
    trained_at = Column(DateTime, default=datetime.utcnow)
    training_duration_seconds = Column(Float)
    
    # Index for finding active models
    __table_args__ = (
        Index('ix_model_active', 'model_name', 'stock_symbol', 'is_active'),
    )
    
    def __repr__(self):
        return f"<ModelMetrics(model='{self.model_name}', symbol='{self.stock_symbol}', rmse={self.rmse})>"


class Prediction(Base):
    """Stock price predictions from various models."""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    model_id = Column(Integer, ForeignKey("model_metrics.id"))
    
    # Prediction details
    prediction_date = Column(Date, nullable=False)  # When prediction was made
    target_date = Column(Date, nullable=False, index=True)  # Date being predicted
    
    # Predicted values
    predicted_close = Column(Float, nullable=False)
    predicted_high = Column(Float)
    predicted_low = Column(Float)
    
    # Confidence intervals
    confidence_upper = Column(Float)  # 95% confidence upper bound
    confidence_lower = Column(Float)  # 95% confidence lower bound
    
    # Actual values (filled after target_date)
    actual_close = Column(Float)
    prediction_error = Column(Float)  # actual - predicted
    
    # Metadata
    model_name = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    stock = relationship("Stock", back_populates="predictions")
    
    # Index for fast queries
    __table_args__ = (
        Index('ix_predictions_stock_target', 'stock_id', 'target_date'),
        Index('ix_predictions_model_target', 'model_id', 'target_date'),
    )
    
    def __repr__(self):
        return f"<Prediction(stock_id={self.stock_id}, target='{self.target_date}', predicted={self.predicted_close})>"


class NewsArticle(Base):
    """Scraped news articles for sentiment analysis."""
    __tablename__ = "news_articles"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_symbol = Column(String(20), nullable=False, index=True)
    
    # Article content
    title = Column(String(500), nullable=False)
    summary = Column(Text)
    content = Column(Text)
    url = Column(String(1000), unique=True)
    
    # Source info
    source = Column(String(100))  # cafef, vietstock, ndh, etc.
    author = Column(String(200))
    published_date = Column(DateTime, index=True)
    
    # Sentiment (from analysis)
    sentiment_score = Column(Float)
    sentiment_label = Column(String(20))
    
    # Metadata
    scraped_at = Column(DateTime, default=datetime.utcnow)
    
    # Index for fast queries
    __table_args__ = (
        Index('ix_news_symbol_date', 'stock_symbol', 'published_date'),
    )
    
    def __repr__(self):
        return f"<NewsArticle(symbol='{self.stock_symbol}', title='{self.title[:50]}...')>"
