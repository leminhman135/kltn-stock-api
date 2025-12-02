"""
Sentiment Analysis Pipeline - Quy trình xử lý tin tức chuẩn với FinBERT

Quy trình 7 bước:
1. Thu thập tin tức từ báo tài chính
2. Làm sạch văn bản
3. Tokenization theo chuẩn BERT
4. Nhúng (embedding) bằng FinBERT
5. Dự đoán sentiment: positive – neutral – negative
6. Chuyển sentiment theo ngày về dạng số
7. Gộp vào mô hình dự báo giá
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import re

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers/torch not installed. Run: pip install transformers torch")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NewsCollector:
    """
    Bước 1: Thu thập tin tức từ báo tài chính
    """
    
    def __init__(self):
        from src.news_service import news_service
        self.news_service = news_service
    
    def collect_news(self, symbol: str = None, days: int = 30, limit: int = 100) -> pd.DataFrame:
        """
        Thu thập tin tức từ nhiều nguồn RSS
        
        Args:
            symbol: Mã cổ phiếu (None = tất cả tin)
            days: Số ngày lấy tin
            limit: Số lượng tin tối đa
        
        Returns:
            DataFrame với columns: [date, symbol, title, summary, url, source]
        """
        logger.info(f"📰 Bước 1: Thu thập tin tức cho {symbol or 'thị trường'}")
        
        news_articles = self.news_service.get_all_news(symbol=symbol, limit=limit)
        
        if not news_articles:
            logger.warning("Không có tin tức")
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for article in news_articles:
            data.append({
                'date': article.published_at,
                'symbol': article.symbol or symbol or 'MARKET',
                'title': article.title,
                'summary': article.summary,
                'url': article.url,
                'source': article.source
            })
        
        df = pd.DataFrame(data)
        
        # Parse date
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Filter by days
        cutoff_date = datetime.now() - timedelta(days=days)
        df = df[df['date'] >= cutoff_date]
        
        logger.info(f"✓ Thu thập được {len(df)} tin tức")
        return df


class TextCleaner:
    """
    Bước 2: Làm sạch văn bản
    """
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Làm sạch văn bản cho FinBERT
        
        - Xóa URLs, HTML tags
        - Xóa ký tự đặc biệt
        - Chuẩn hóa khoảng trắng
        - Giữ dấu câu quan trọng
        """
        if not isinstance(text, str) or not text:
            return ""
        
        # Xóa URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Xóa HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Xóa email
        text = re.sub(r'\S+@\S+', '', text)
        
        # Xóa ký tự đặc biệt nhưng GIỮ dấu câu quan trọng
        text = re.sub(r'[^\w\s.,!?%$€£¥-]', ' ', text)
        
        # Chuẩn hóa số
        text = re.sub(r'\d+', ' NUMBER ', text)
        
        # Xóa khoảng trắng thừa
        text = ' '.join(text.split())
        
        return text.strip()
    
    def clean_dataframe(self, df: pd.DataFrame, columns: List[str] = ['title', 'summary']) -> pd.DataFrame:
        """
        Làm sạch nhiều cột trong DataFrame
        """
        logger.info(f"🧹 Bước 2: Làm sạch văn bản")
        
        df_clean = df.copy()
        
        for col in columns:
            if col in df_clean.columns:
                df_clean[f'{col}_clean'] = df_clean[col].apply(self.clean)
                logger.info(f"  ✓ Cleaned {col}")
        
        # Combine title + summary
        if 'title_clean' in df_clean.columns and 'summary_clean' in df_clean.columns:
            df_clean['text_clean'] = df_clean['title_clean'] + ' ' + df_clean['summary_clean']
            df_clean['text_clean'] = df_clean['text_clean'].str.strip()
        elif 'title_clean' in df_clean.columns:
            df_clean['text_clean'] = df_clean['title_clean']
        
        logger.info(f"✓ Làm sạch hoàn tất")
        return df_clean


class FinBERTTokenizer:
    """
    Bước 3: Tokenization theo chuẩn BERT
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        
        logger.info(f"🔤 Bước 3: Load FinBERT tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"✓ Tokenizer ready: {model_name}")
    
    def tokenize(self, text: str, max_length: int = 512) -> Dict:
        """
        Tokenize văn bản theo chuẩn BERT
        
        Returns:
            {
                'input_ids': tensor,
                'attention_mask': tensor,
                'token_type_ids': tensor (optional)
            }
        """
        return self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding='max_length',
            return_attention_mask=True
        )
    
    def tokenize_batch(self, texts: List[str], max_length: int = 512) -> Dict:
        """Tokenize nhiều văn bản cùng lúc"""
        return self.tokenizer(
            texts,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True,
            return_attention_mask=True
        )


class FinBERTEmbedder:
    """
    Bước 4: Nhúng (embedding) bằng FinBERT
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        
        logger.info(f"🧠 Bước 4: Load FinBERT model")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        
        # Evaluation mode
        self.model.eval()
        
        # GPU nếu có
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        logger.info(f"✓ Model ready on {self.device}")
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Lấy embeddings từ FinBERT (hidden states)
        
        Returns:
            numpy array shape (768,) cho FinBERT base
        """
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Lấy [CLS] token embedding từ last hidden state
            # Shape: (1, seq_len, hidden_size) -> lấy (1, 0, hidden_size)
            embeddings = outputs.hidden_states[-1][:, 0, :].cpu().numpy()
            
        return embeddings.squeeze()  # (768,)


class SentimentPredictor:
    """
    Bước 5: Dự đoán sentiment: positive – neutral – negative
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers not installed")
        
        logger.info(f"🎯 Bước 5: Load FinBERT cho sentiment prediction")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Label mapping (FinBERT output order)
        self.id2label = {0: 'positive', 1: 'negative', 2: 'neutral'}
        
        logger.info(f"✓ Predictor ready on {self.device}")
    
    def predict(self, text: str) -> Dict:
        """
        Dự đoán sentiment cho một văn bản
        
        Returns:
            {
                'sentiment': str,  # 'positive', 'negative', 'neutral'
                'positive': float,
                'negative': float,
                'neutral': float,
                'confidence': float
            }
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=512,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()[0]
        
        # Map to labels
        scores = {
            'positive': float(probs[0]),
            'negative': float(probs[1]),
            'neutral': float(probs[2])
        }
        
        # Tìm sentiment chính
        sentiment = max(scores, key=scores.get)
        confidence = scores[sentiment]
        
        return {
            'sentiment': sentiment,
            **scores,
            'confidence': confidence
        }
    
    def predict_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict]:
        """Dự đoán nhiều văn bản"""
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()
            
            # Process batch results
            for prob in probs:
                scores = {
                    'positive': float(prob[0]),
                    'negative': float(prob[1]),
                    'neutral': float(prob[2])
                }
                sentiment = max(scores, key=scores.get)
                
                results.append({
                    'sentiment': sentiment,
                    **scores,
                    'confidence': scores[sentiment]
                })
        
        return results


class SentimentNumericalConverter:
    """
    Bước 6: Chuyển sentiment theo ngày về dạng số
    """
    
    @staticmethod
    def sentiment_to_score(sentiment: str, positive: float, negative: float) -> float:
        """
        Chuyển sentiment thành điểm số [-1, 1]
        
        Formula: score = positive - negative
        """
        return positive - negative
    
    def convert_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Thêm cột sentiment_score vào DataFrame
        """
        logger.info(f"🔢 Bước 6: Chuyển sentiment về dạng số")
        
        df_score = df.copy()
        
        # Calculate sentiment score
        df_score['sentiment_score'] = df_score.apply(
            lambda row: self.sentiment_to_score(
                row['sentiment'], 
                row['positive'], 
                row['negative']
            ),
            axis=1
        )
        
        logger.info(f"✓ Sentiment score range: [{df_score['sentiment_score'].min():.2f}, {df_score['sentiment_score'].max():.2f}]")
        
        return df_score
    
    def aggregate_by_date(self, df: pd.DataFrame, date_col: str = 'date', 
                         symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        Tổng hợp sentiment theo ngày
        
        Returns:
            DataFrame với cột: date, symbol, daily_sentiment_score, news_count
        """
        logger.info(f"📊 Tổng hợp sentiment theo ngày")
        
        # Group by date + symbol
        aggregated = df.groupby([date_col, symbol_col]).agg({
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            date_col, symbol_col,
            'daily_sentiment_mean', 'daily_sentiment_std', 
            'daily_sentiment_min', 'daily_sentiment_max',
            'daily_positive', 'daily_negative', 'daily_neutral',
            'daily_sentiment_mode'
        ]
        
        # Count news per day
        news_count = df.groupby([date_col, symbol_col]).size().reset_index(name='news_count')
        aggregated = aggregated.merge(news_count, on=[date_col, symbol_col])
        
        logger.info(f"✓ Tổng hợp {len(aggregated)} ngày")
        
        return aggregated


class ModelIntegrator:
    """
    Bước 7: Gộp vào mô hình dự báo giá
    """
    
    def merge_with_price_data(self, price_df: pd.DataFrame, sentiment_df: pd.DataFrame,
                              on: List[str] = ['date', 'symbol']) -> pd.DataFrame:
        """
        Merge sentiment data vào price data
        
        Args:
            price_df: DataFrame giá cổ phiếu (date, symbol, open, high, low, close, volume)
            sentiment_df: DataFrame sentiment (date, symbol, daily_sentiment_mean, ...)
        
        Returns:
            Merged DataFrame
        """
        logger.info(f"🔗 Bước 7: Gộp sentiment vào dữ liệu giá")
        
        # Ensure date columns are datetime
        price_df['date'] = pd.to_datetime(price_df['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Merge
        merged = price_df.merge(
            sentiment_df,
            on=on,
            how='left'
        )
        
        # Fill missing sentiment with neutral (0)
        sentiment_cols = [col for col in merged.columns if 'sentiment' in col.lower() or 'daily_' in col.lower()]
        for col in sentiment_cols:
            if merged[col].dtype in [np.float64, np.float32]:
                merged[col].fillna(0, inplace=True)
            else:
                merged[col].fillna('neutral', inplace=True)
        
        # Fill news_count with 0
        if 'news_count' in merged.columns:
            merged['news_count'].fillna(0, inplace=True)
        
        logger.info(f"✓ Merged {len(merged)} rows")
        logger.info(f"  Sentiment coverage: {(merged['news_count'] > 0).sum() / len(merged) * 100:.1f}%")
        
        return merged
    
    def create_sentiment_features(self, df: pd.DataFrame, windows: List[int] = [3, 7, 14]) -> pd.DataFrame:
        """
        Tạo features sentiment cho model ML
        
        - Moving averages
        - Momentum (change)
        - Volatility
        """
        logger.info(f"⚙️ Tạo sentiment features")
        
        df_feat = df.copy()
        
        # Sort by date
        df_feat = df_feat.sort_values('date')
        
        # Moving averages
        for window in windows:
            df_feat[f'sentiment_ma_{window}'] = df_feat['daily_sentiment_mean'].rolling(window=window).mean()
        
        # Momentum (first difference)
        df_feat['sentiment_momentum'] = df_feat['daily_sentiment_mean'].diff()
        
        # Volatility (rolling std)
        df_feat['sentiment_volatility'] = df_feat['daily_sentiment_mean'].rolling(window=7).std()
        
        # Cumulative sentiment
        df_feat['sentiment_cumsum'] = df_feat['daily_sentiment_mean'].cumsum()
        
        # Positive/Negative ratio
        df_feat['pos_neg_ratio'] = df_feat['daily_positive'] / (df_feat['daily_negative'] + 1e-5)
        
        logger.info(f"✓ Created {len([c for c in df_feat.columns if 'sentiment' in c])} sentiment features")
        
        return df_feat


class SentimentPipeline:
    """
    Pipeline tổng hợp - Chạy đầy đủ 7 bước
    """
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        """
        Khởi tạo pipeline
        """
        logger.info("="*60)
        logger.info("🚀 Khởi tạo Sentiment Pipeline với FinBERT")
        logger.info("="*60)
        
        self.collector = NewsCollector()
        self.cleaner = TextCleaner()
        self.predictor = SentimentPredictor(model_name)
        self.converter = SentimentNumericalConverter()
        self.integrator = ModelIntegrator()
        
        logger.info("✓ Pipeline ready")
    
    def process_news(self, symbol: str, days: int = 30, limit: int = 100) -> pd.DataFrame:
        """
        Xử lý tin tức: Bước 1-6
        
        Returns:
            DataFrame với sentiment scores theo ngày
        """
        # Step 1: Collect
        news_df = self.collector.collect_news(symbol=symbol, days=days, limit=limit)
        
        if news_df.empty:
            logger.warning("No news to process")
            return pd.DataFrame()
        
        # Step 2: Clean
        news_df = self.cleaner.clean_dataframe(news_df)
        
        # Step 3-5: Tokenize + Embed + Predict (all in predict_batch)
        logger.info("🎯 Bước 3-5: Tokenization + Embedding + Prediction")
        texts = news_df['text_clean'].fillna('').tolist()
        sentiments = self.predictor.predict_batch(texts, batch_size=16)
        
        # Add to DataFrame
        for key in ['sentiment', 'positive', 'negative', 'neutral', 'confidence']:
            news_df[key] = [s[key] for s in sentiments]
        
        logger.info(f"✓ Predicted {len(sentiments)} articles")
        
        # Step 6: Convert to numerical
        news_df = self.converter.convert_dataframe(news_df)
        daily_sentiment = self.converter.aggregate_by_date(news_df)
        
        return daily_sentiment
    
    def process_and_merge(self, symbol: str, price_df: pd.DataFrame, 
                         days: int = 30, limit: int = 100) -> pd.DataFrame:
        """
        Pipeline đầy đủ: Bước 1-7
        
        Returns:
            Price DataFrame với sentiment features
        """
        # Steps 1-6
        daily_sentiment = self.process_news(symbol=symbol, days=days, limit=limit)
        
        if daily_sentiment.empty:
            logger.warning("No sentiment data, returning price data only")
            return price_df
        
        # Step 7: Merge
        merged = self.integrator.merge_with_price_data(price_df, daily_sentiment)
        merged = self.integrator.create_sentiment_features(merged)
        
        return merged


# ============ Test & Demo ============

def test_pipeline():
    """Test pipeline với dữ liệu mẫu"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST SENTIMENT PIPELINE")
    logger.info("="*60)
    
    # Sample news
    sample_texts = [
        "Vinamilk công bố lợi nhuận quý 3 tăng 25% so với cùng kỳ năm ngoái",
        "Thị trường chứng khoán sụt giảm mạnh do lo ngại lãi suất tăng",
        "FPT ký hợp đồng xuất khẩu phần mềm 50 triệu USD sang Nhật Bản",
        "Ngân hàng Nhà nước cảnh báo rủi ro tín dụng bất động sản",
        "VIC ra mắt dự án Vinhomes Ocean Park 3 quy mô 500 hecta"
    ]
    
    # Test predictor
    predictor = SentimentPredictor()
    
    logger.info("\nDự đoán sentiment:")
    for i, text in enumerate(sample_texts, 1):
        result = predictor.predict(text)
        logger.info(f"\n{i}. {text[:60]}...")
        logger.info(f"   → {result['sentiment'].upper()} (confidence: {result['confidence']:.2f})")
        logger.info(f"   → Scores: pos={result['positive']:.2f}, neg={result['negative']:.2f}, neu={result['neutral']:.2f}")


if __name__ == "__main__":
    if not HAS_TRANSFORMERS:
        print("❌ Please install: pip install transformers torch")
    else:
        test_pipeline()
