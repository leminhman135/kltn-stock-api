"""
Module phân tích cảm tính (Sentiment Analysis) sử dụng FinBERT
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from datetime import datetime
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTSentimentAnalyzer:
    """
    Phân tích cảm tính tin tức tài chính sử dụng FinBERT
    
    FinBERT là mô hình BERT được fine-tune trên dữ liệu tài chính,
    có khả năng phân loại văn bản thành 3 lớp:
    - Positive (Tích cực)
    - Negative (Tiêu cực)  
    - Neutral (Trung lập)
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        """
        Khởi tạo FinBERT model
        
        Args:
            model_name: Tên mô hình trên HuggingFace
                       'ProsusAI/finbert' - Model chính thức
                       'yiyanghkust/finbert-tone' - Alternative
        """
        try:
            logger.info(f"Loading FinBERT model: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Chuyển model sang evaluation mode
            self.model.eval()
            
            # Sử dụng GPU nếu có
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading FinBERT model: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Làm sạch văn bản"""
        if not isinstance(text, str):
            return ""
        
        # Xóa URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Xóa HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Xóa ký tự đặc biệt nhưng giữ dấu câu
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Xóa khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text.strip()
    
    def predict_sentiment(self, text: str) -> Dict[str, float]:
        """
        Dự đoán cảm tính của một đoạn văn bản
        
        Args:
            text: Văn bản cần phân tích
        
        Returns:
            Dictionary với scores: {
                'positive': float,
                'negative': float,
                'neutral': float,
                'sentiment': str,  # 'positive', 'negative', hoặc 'neutral'
                'score': float     # Confidence score của sentiment chính
            }
        """
        try:
            # Làm sạch văn bản
            cleaned_text = self.clean_text(text)
            
            if not cleaned_text:
                return {
                    'positive': 0.0,
                    'negative': 0.0,
                    'neutral': 1.0,
                    'sentiment': 'neutral',
                    'score': 1.0
                }
            
            # Tokenize
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Chuyển sang device
            inputs = {key: val.to(self.device) for key, val in inputs.items()}
            
            # Dự đoán
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            # Chuyển về CPU và numpy
            scores = predictions.cpu().numpy()[0]
            
            # Map scores (thứ tự có thể khác nhau tùy model)
            labels = ['positive', 'negative', 'neutral']
            sentiment_scores = {label: float(score) for label, score in zip(labels, scores)}
            
            # Xác định sentiment chính
            main_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            main_score = sentiment_scores[main_sentiment]
            
            result = {
                **sentiment_scores,
                'sentiment': main_sentiment,
                'score': main_score
            }
            
            return result
        
        except Exception as e:
            logger.error(f"Error predicting sentiment: {str(e)}")
            return {
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'sentiment': 'neutral',
                'score': 0.0
            }
    
    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """
        Phân tích cảm tính cho nhiều văn bản (hiệu quả hơn)
        
        Args:
            texts: Danh sách văn bản
            batch_size: Số văn bản xử lý cùng lúc
        
        Returns:
            List of sentiment dictionaries
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            for text in batch_texts:
                sentiment = self.predict_sentiment(text)
                results.append(sentiment)
            
            if (i + batch_size) % 100 == 0:
                logger.info(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")
        
        return results
    
    def analyze_news_dataframe(self, df: pd.DataFrame, 
                              text_column: str = 'title') -> pd.DataFrame:
        """
        Phân tích cảm tính cho DataFrame tin tức
        
        Args:
            df: DataFrame chứa tin tức
            text_column: Tên cột chứa văn bản cần phân tích
        
        Returns:
            DataFrame với các cột sentiment được thêm vào
        """
        df_sentiment = df.copy()
        
        if text_column not in df_sentiment.columns:
            logger.error(f"Column '{text_column}' not found in DataFrame")
            return df_sentiment
        
        logger.info(f"Analyzing sentiment for {len(df_sentiment)} news articles...")
        
        # Phân tích cảm tính
        texts = df_sentiment[text_column].fillna('').tolist()
        sentiments = self.analyze_batch(texts)
        
        # Thêm vào DataFrame
        df_sentiment['sentiment_positive'] = [s['positive'] for s in sentiments]
        df_sentiment['sentiment_negative'] = [s['negative'] for s in sentiments]
        df_sentiment['sentiment_neutral'] = [s['neutral'] for s in sentiments]
        df_sentiment['sentiment_label'] = [s['sentiment'] for s in sentiments]
        df_sentiment['sentiment_score'] = [s['score'] for s in sentiments]
        
        logger.info("Sentiment analysis completed")
        
        return df_sentiment


class SentimentAggregator:
    """Tổng hợp điểm cảm tính theo ngày cho từng mã cổ phiếu"""
    
    @staticmethod
    def aggregate_by_date(df: pd.DataFrame, date_column: str = 'date',
                         symbol_column: str = 'symbol') -> pd.DataFrame:
        """
        Tổng hợp điểm sentiment theo ngày
        
        Args:
            df: DataFrame với sentiment scores
            date_column: Tên cột ngày
            symbol_column: Tên cột mã cổ phiếu
        
        Returns:
            DataFrame với điểm sentiment trung bình theo ngày
        """
        if date_column not in df.columns:
            logger.error(f"Column '{date_column}' not found")
            return pd.DataFrame()
        
        # Chuyển đổi date
        df_agg = df.copy()
        df_agg[date_column] = pd.to_datetime(df_agg[date_column], errors='coerce')
        
        # Chỉ giữ các hàng có ngày hợp lệ
        df_agg = df_agg.dropna(subset=[date_column])
        
        # Group by date và symbol
        group_cols = [date_column]
        if symbol_column in df_agg.columns:
            group_cols.append(symbol_column)
        
        aggregated = df_agg.groupby(group_cols).agg({
            'sentiment_positive': 'mean',
            'sentiment_negative': 'mean',
            'sentiment_neutral': 'mean',
            'sentiment_score': 'mean',
            'sentiment_label': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'neutral'
        }).reset_index()
        
        # Tính sentiment tổng hợp
        aggregated['daily_sentiment'] = (
            aggregated['sentiment_positive'] - aggregated['sentiment_negative']
        )
        
        # Đếm số bài viết
        news_count = df_agg.groupby(group_cols).size().reset_index(name='news_count')
        aggregated = aggregated.merge(news_count, on=group_cols, how='left')
        
        return aggregated
    
    @staticmethod
    def create_sentiment_features(df: pd.DataFrame, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Tạo các đặc trưng cảm tính với moving averages
        
        Args:
            df: DataFrame với daily sentiment scores
            windows: Danh sách số ngày cho moving average
        
        Returns:
            DataFrame với sentiment features
        """
        df_features = df.copy()
        
        if 'daily_sentiment' not in df_features.columns:
            return df_features
        
        # Sắp xếp theo ngày
        if 'date' in df_features.columns:
            df_features = df_features.sort_values('date')
        
        # Tính moving averages
        for window in windows:
            df_features[f'sentiment_ma_{window}'] = (
                df_features['daily_sentiment'].rolling(window=window).mean()
            )
        
        # Sentiment momentum
        df_features['sentiment_momentum'] = df_features['daily_sentiment'].diff()
        
        # Cumulative sentiment
        df_features['sentiment_cumsum'] = df_features['daily_sentiment'].cumsum()
        
        return df_features


class SentimentAnalysisPipeline:
    """Pipeline tổng hợp cho sentiment analysis"""
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        self.analyzer = FinBERTSentimentAnalyzer(model_name)
        self.aggregator = SentimentAggregator()
    
    def process_news(self, news_df: pd.DataFrame, 
                    text_column: str = 'title',
                    date_column: str = 'date',
                    symbol_column: str = 'symbol') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Xử lý tin tức: phân tích cảm tính và tổng hợp theo ngày
        
        Returns:
            Tuple[DataFrame, DataFrame]: (news_with_sentiment, daily_sentiment)
        """
        # Phân tích cảm tính
        news_with_sentiment = self.analyzer.analyze_news_dataframe(
            news_df, text_column=text_column
        )
        
        # Tổng hợp theo ngày
        daily_sentiment = self.aggregator.aggregate_by_date(
            news_with_sentiment,
            date_column=date_column,
            symbol_column=symbol_column
        )
        
        # Tạo features
        daily_sentiment = self.aggregator.create_sentiment_features(daily_sentiment)
        
        return news_with_sentiment, daily_sentiment


if __name__ == "__main__":
    # Test sentiment analysis
    sample_news = pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10),
        'symbol': ['AAPL'] * 10,
        'title': [
            'Apple reports record quarterly revenue',
            'Stock market crashes amid economic concerns',
            'Tech giant announces new product line',
            'Analysts downgrade stock rating',
            'Company exceeds earnings expectations',
            'Market volatility continues',
            'Strong demand drives growth',
            'Regulatory challenges ahead',
            'Innovation leads to market share gains',
            'Economic uncertainty weighs on outlook'
        ]
    })
    
    print("Testing Sentiment Analysis Pipeline...")
    print("Note: First run will download the FinBERT model (~400MB)")
    
    try:
        pipeline = SentimentAnalysisPipeline()
        news_sentiment, daily_sentiment = pipeline.process_news(sample_news)
        
        print(f"\nProcessed {len(news_sentiment)} news articles")
        print("\nSample results:")
        print(news_sentiment[['title', 'sentiment_label', 'sentiment_score']].head())
        
        print("\nDaily aggregated sentiment:")
        print(daily_sentiment[['date', 'daily_sentiment', 'news_count']].head())
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("Note: FinBERT requires transformers and torch libraries")
