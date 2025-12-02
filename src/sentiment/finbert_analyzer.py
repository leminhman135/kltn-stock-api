"""
FinBERT Sentiment Analysis for Vietnamese Stock Market

FinBERT là mô hình BERT được fine-tune cho phân tích sentiment tài chính.
Module này tích hợp FinBERT để phân tích tin tức cổ phiếu Việt Nam.

Features:
1. Phân tích sentiment từ tiêu đề và nội dung tin tức
2. Aggregate sentiment theo ngày/tuần
3. Sentiment scoring cho trading signals
4. Multi-language support (EN + VN translation)
"""

import os
import logging
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check if transformers is available
TRANSFORMERS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    logger.warning("PyTorch not available. Sentiment analysis will use fallback.")

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("Transformers not available. Sentiment analysis will use fallback.")


class FinBERTSentimentAnalyzer:
    """
    FinBERT Sentiment Analyzer cho thị trường chứng khoán
    
    Sử dụng model ProsusAI/finbert - được train trên dữ liệu tài chính
    
    Labels:
    - positive: Tin tức tích cực (bullish)
    - negative: Tin tức tiêu cực (bearish)  
    - neutral: Tin tức trung lập
    
    Usage:
        analyzer = FinBERTSentimentAnalyzer()
        result = analyzer.analyze("VNM báo cáo lợi nhuận tăng 20%")
        # {'label': 'positive', 'score': 0.85, 'sentiment_score': 0.85}
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = None):
        """
        Args:
            model_name: Tên model trên HuggingFace
            device: 'cuda', 'cpu', hoặc None (auto-detect)
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        self.device = device
        self.is_loaded = False
        
        # Sentiment mapping
        self.label_to_score = {
            'positive': 1.0,
            'neutral': 0.0,
            'negative': -1.0
        }
        
        # Vietnamese to English translation for common financial terms
        self.vn_positive_keywords = [
            'tăng', 'lợi nhuận', 'tích cực', 'khởi sắc', 'đột phá', 'vượt kỳ vọng',
            'cổ tức', 'chia thưởng', 'mở rộng', 'hợp đồng lớn', 'tăng trưởng',
            'kỷ lục', 'thành công', 'phục hồi', 'bứt phá', 'lạc quan'
        ]
        
        self.vn_negative_keywords = [
            'giảm', 'thua lỗ', 'khó khăn', 'sụt giảm', 'thiệt hại', 'rủi ro',
            'phá sản', 'nợ xấu', 'thanh tra', 'vi phạm', 'cảnh báo', 'lo ngại',
            'bán tháo', 'hoảng loạn', 'khủng hoảng', 'suy thoái', 'đình trệ'
        ]
    
    def load_model(self) -> bool:
        """Load FinBERT model"""
        
        if not TRANSFORMERS_AVAILABLE or not TORCH_AVAILABLE:
            logger.warning("Transformers/PyTorch not available, using keyword-based fallback")
            return False
        
        try:
            logger.info(f"🔄 Loading FinBERT model: {self.model_name}")
            
            # Auto-detect device
            if self.device is None:
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == 'cuda' else -1,
                max_length=512,
                truncation=True
            )
            
            self.is_loaded = True
            logger.info(f"✅ FinBERT loaded successfully on {self.device}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load FinBERT: {e}")
            return False
    
    def analyze(self, text: str) -> Dict:
        """
        Phân tích sentiment của một đoạn text
        
        Args:
            text: Tiêu đề hoặc nội dung tin tức
            
        Returns:
            {
                'label': 'positive'/'negative'/'neutral',
                'score': float (0-1),
                'sentiment_score': float (-1 to 1),
                'method': 'finbert' or 'keyword'
            }
        """
        if not text or not text.strip():
            return {
                'label': 'neutral',
                'score': 0.0,
                'sentiment_score': 0.0,
                'method': 'empty'
            }
        
        # Try FinBERT first
        if self.is_loaded and self.pipeline:
            return self._analyze_with_finbert(text)
        
        # Fallback to keyword-based analysis
        return self._analyze_with_keywords(text)
    
    def _analyze_with_finbert(self, text: str) -> Dict:
        """Phân tích bằng FinBERT model"""
        try:
            # Truncate text if too long
            if len(text) > 500:
                text = text[:500]
            
            result = self.pipeline(text)[0]
            
            label = result['label'].lower()
            score = result['score']
            
            # Map to sentiment score (-1 to 1)
            sentiment_score = self.label_to_score.get(label, 0) * score
            
            return {
                'label': label,
                'score': score,
                'sentiment_score': sentiment_score,
                'method': 'finbert'
            }
            
        except Exception as e:
            logger.error(f"FinBERT analysis error: {e}")
            return self._analyze_with_keywords(text)
    
    def _analyze_with_keywords(self, text: str) -> Dict:
        """Fallback: Phân tích bằng keywords cho tiếng Việt"""
        
        text_lower = text.lower()
        
        positive_count = sum(1 for kw in self.vn_positive_keywords if kw in text_lower)
        negative_count = sum(1 for kw in self.vn_negative_keywords if kw in text_lower)
        
        total = positive_count + negative_count
        
        if total == 0:
            return {
                'label': 'neutral',
                'score': 0.5,
                'sentiment_score': 0.0,
                'method': 'keyword'
            }
        
        # Calculate sentiment
        sentiment_score = (positive_count - negative_count) / total
        
        if sentiment_score > 0.2:
            label = 'positive'
            score = min(0.5 + sentiment_score * 0.5, 1.0)
        elif sentiment_score < -0.2:
            label = 'negative'
            score = min(0.5 + abs(sentiment_score) * 0.5, 1.0)
        else:
            label = 'neutral'
            score = 0.5
        
        return {
            'label': label,
            'score': score,
            'sentiment_score': sentiment_score,
            'method': 'keyword'
        }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict]:
        """Phân tích nhiều texts cùng lúc"""
        
        results = []
        
        if self.is_loaded and self.pipeline:
            try:
                # Batch processing with FinBERT
                batch_results = self.pipeline(texts[:100])  # Limit batch size
                
                for result in batch_results:
                    label = result['label'].lower()
                    score = result['score']
                    sentiment_score = self.label_to_score.get(label, 0) * score
                    
                    results.append({
                        'label': label,
                        'score': score,
                        'sentiment_score': sentiment_score,
                        'method': 'finbert'
                    })
                
                return results
                
            except Exception as e:
                logger.error(f"Batch analysis error: {e}")
        
        # Fallback to individual analysis
        return [self.analyze(text) for text in texts]
    
    def analyze_news_dataframe(self, df: pd.DataFrame, 
                               text_col: str = 'title',
                               content_col: str = None) -> pd.DataFrame:
        """
        Phân tích sentiment cho DataFrame tin tức
        
        Args:
            df: DataFrame với tin tức
            text_col: Column chứa tiêu đề
            content_col: Column chứa nội dung (optional)
            
        Returns:
            DataFrame với thêm columns sentiment
        """
        result_df = df.copy()
        
        # Analyze titles
        texts = result_df[text_col].fillna('').tolist()
        title_sentiments = self.analyze_batch(texts)
        
        result_df['sentiment_label'] = [s['label'] for s in title_sentiments]
        result_df['sentiment_score'] = [s['sentiment_score'] for s in title_sentiments]
        result_df['sentiment_confidence'] = [s['score'] for s in title_sentiments]
        result_df['sentiment_method'] = [s['method'] for s in title_sentiments]
        
        # Analyze content if provided
        if content_col and content_col in result_df.columns:
            contents = result_df[content_col].fillna('').tolist()
            content_sentiments = self.analyze_batch(contents)
            
            result_df['content_sentiment_score'] = [s['sentiment_score'] for s in content_sentiments]
            
            # Combined score (weighted average)
            result_df['combined_sentiment'] = (
                0.4 * result_df['sentiment_score'] + 
                0.6 * result_df['content_sentiment_score']
            )
        
        return result_df
    
    def aggregate_daily_sentiment(self, df: pd.DataFrame, 
                                  date_col: str = 'date',
                                  sentiment_col: str = 'sentiment_score') -> pd.DataFrame:
        """
        Tổng hợp sentiment theo ngày
        
        Args:
            df: DataFrame với sentiment đã phân tích
            date_col: Column chứa date
            sentiment_col: Column chứa sentiment score
            
        Returns:
            DataFrame với daily aggregated sentiment
        """
        daily = df.groupby(date_col).agg({
            sentiment_col: ['mean', 'std', 'count'],
            'sentiment_label': lambda x: x.value_counts().to_dict()
        }).reset_index()
        
        # Flatten column names
        daily.columns = [
            date_col, 
            'avg_sentiment', 
            'sentiment_std', 
            'news_count',
            'label_distribution'
        ]
        
        # Add sentiment signal
        daily['sentiment_signal'] = pd.cut(
            daily['avg_sentiment'],
            bins=[-1, -0.3, 0.3, 1],
            labels=['bearish', 'neutral', 'bullish']
        )
        
        return daily
    
    def calculate_sentiment_features(self, sentiment_df: pd.DataFrame,
                                     price_df: pd.DataFrame,
                                     date_col: str = 'date') -> pd.DataFrame:
        """
        Tính các đặc trưng sentiment cho prediction
        
        Returns DataFrame với các features:
        - sentiment_ma_3: Moving average 3 ngày
        - sentiment_ma_7: Moving average 7 ngày
        - sentiment_momentum: Thay đổi sentiment
        - sentiment_divergence: Divergence với giá
        - news_volume: Số lượng tin tức (càng nhiều càng quan trọng)
        """
        # Merge sentiment with prices
        if date_col in sentiment_df.columns and date_col in price_df.columns:
            merged = pd.merge(price_df, sentiment_df, on=date_col, how='left')
        else:
            merged = price_df.copy()
            merged['avg_sentiment'] = 0
            merged['news_count'] = 0
        
        # Fill missing sentiment
        merged['avg_sentiment'] = merged['avg_sentiment'].fillna(0)
        merged['news_count'] = merged['news_count'].fillna(0)
        
        # Sentiment features
        merged['sentiment_ma_3'] = merged['avg_sentiment'].rolling(3).mean()
        merged['sentiment_ma_7'] = merged['avg_sentiment'].rolling(7).mean()
        merged['sentiment_momentum'] = merged['avg_sentiment'] - merged['avg_sentiment'].shift(1)
        merged['sentiment_acceleration'] = merged['sentiment_momentum'] - merged['sentiment_momentum'].shift(1)
        
        # News volume features
        merged['news_volume_ma'] = merged['news_count'].rolling(7).mean()
        merged['news_volume_spike'] = (merged['news_count'] > merged['news_volume_ma'] * 2).astype(int)
        
        # Sentiment divergence with price
        if 'close' in merged.columns:
            price_return = merged['close'].pct_change()
            merged['sentiment_price_divergence'] = merged['sentiment_momentum'] - price_return
        
        return merged


class SentimentTradingSignals:
    """
    Tạo trading signals từ sentiment analysis
    """
    
    @staticmethod
    def generate_signals(df: pd.DataFrame, 
                        sentiment_col: str = 'avg_sentiment',
                        threshold_bullish: float = 0.3,
                        threshold_bearish: float = -0.3) -> pd.DataFrame:
        """
        Generate trading signals từ sentiment
        
        Signals:
        - 1: Buy (bullish sentiment)
        - 0: Hold (neutral)
        - -1: Sell (bearish sentiment)
        """
        result = df.copy()
        
        conditions = [
            (result[sentiment_col] > threshold_bullish),
            (result[sentiment_col] < threshold_bearish)
        ]
        choices = [1, -1]
        
        result['sentiment_signal'] = np.select(conditions, choices, default=0)
        
        # Signal strength based on confidence
        if 'sentiment_std' in result.columns:
            # Lower std = higher confidence
            result['signal_strength'] = 1 - result['sentiment_std'].clip(0, 1)
        else:
            result['signal_strength'] = np.abs(result[sentiment_col])
        
        return result
    
    @staticmethod
    def backtest_signals(df: pd.DataFrame,
                        signal_col: str = 'sentiment_signal',
                        price_col: str = 'close') -> Dict:
        """
        Backtest sentiment signals
        
        Returns performance metrics
        """
        result_df = df.copy()
        
        # Calculate returns
        result_df['return'] = result_df[price_col].pct_change()
        result_df['signal_return'] = result_df[signal_col].shift(1) * result_df['return']
        
        # Performance metrics
        total_return = (1 + result_df['signal_return']).prod() - 1
        sharpe_ratio = result_df['signal_return'].mean() / (result_df['signal_return'].std() + 1e-10) * np.sqrt(252)
        
        # Win rate
        positive_returns = result_df[result_df['signal_return'] > 0]
        win_rate = len(positive_returns) / len(result_df[result_df['signal_return'] != 0]) if len(result_df[result_df['signal_return'] != 0]) > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'num_trades': (result_df[signal_col].diff() != 0).sum()
        }


# Global instance for easy access
_analyzer = None

def get_analyzer() -> FinBERTSentimentAnalyzer:
    """Get or create global analyzer instance"""
    global _analyzer
    if _analyzer is None:
        _analyzer = FinBERTSentimentAnalyzer()
        _analyzer.load_model()
    return _analyzer


def analyze_sentiment(text: str) -> Dict:
    """Quick function to analyze sentiment"""
    analyzer = get_analyzer()
    return analyzer.analyze(text)


if __name__ == "__main__":
    # Test
    print("🔥 Testing FinBERT Sentiment Analyzer")
    print("="*50)
    
    analyzer = FinBERTSentimentAnalyzer()
    loaded = analyzer.load_model()
    
    test_texts = [
        "VNM báo cáo lợi nhuận quý 3 tăng 25% so với cùng kỳ năm trước",
        "Cổ phiếu HPG giảm mạnh do lo ngại về giá thép thế giới",
        "FPT ký hợp đồng lớn với đối tác Nhật Bản trị giá 100 triệu USD",
        "Ngân hàng VCB bị thanh tra về vấn đề nợ xấu",
        "Thị trường chứng khoán Việt Nam giao dịch ổn định",
        "Apple's quarterly earnings exceeded analyst expectations",
        "The company announced major layoffs affecting 10,000 employees"
    ]
    
    print(f"\nMethod: {'FinBERT' if loaded else 'Keyword-based fallback'}\n")
    
    for text in test_texts:
        result = analyzer.analyze(text)
        emoji = "🟢" if result['label'] == 'positive' else ("🔴" if result['label'] == 'negative' else "⚪")
        print(f"{emoji} [{result['label']:8s}] ({result['sentiment_score']:+.2f}) {text[:60]}...")
    
    print("\n✅ Test completed!")
