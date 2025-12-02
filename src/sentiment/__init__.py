"""
Sentiment Analysis Module for Vietnamese Stock Market

This module provides:
1. FinBERT-based sentiment analysis for financial news
2. Keyword-based fallback for Vietnamese text
3. Trading signals generation from sentiment
4. Sentiment aggregation and feature engineering
"""

from .finbert_analyzer import (
    FinBERTSentimentAnalyzer,
    SentimentTradingSignals,
    get_analyzer,
    analyze_sentiment,
    TRANSFORMERS_AVAILABLE,
    TORCH_AVAILABLE
)

__all__ = [
    'FinBERTSentimentAnalyzer',
    'SentimentTradingSignals', 
    'get_analyzer',
    'analyze_sentiment',
    'TRANSFORMERS_AVAILABLE',
    'TORCH_AVAILABLE'
]
