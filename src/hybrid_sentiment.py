"""
Hybrid Sentiment Analyzer - Kết hợp Keyword-based (tiếng Việt) + FinBERT (tiếng Anh)

Chiến lược:
1. Dùng keyword-based cho tin tiếng Việt (nhanh, chính xác hơn)
2. Dùng FinBERT cho tin tiếng Anh (nếu có)
3. Dùng translation nếu cần phân tích sâu
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from src.news_service import SentimentAnalyzer as KeywordAnalyzer

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    HAS_FINBERT = True
except ImportError:
    HAS_FINBERT = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridSentimentAnalyzer:
    """
    Hybrid Sentiment Analyzer cho tiếng Việt
    
    Method 1 (Primary): Keyword-based - Chính xác cao cho tiếng Việt
    Method 2 (Fallback): FinBERT - Cho tin tiếng Anh
    """
    
    def __init__(self, use_finbert: bool = False):
        """
        Args:
            use_finbert: Có sử dụng FinBERT không (tốn RAM hơn)
        """
        logger.info("🚀 Khởi tạo Hybrid Sentiment Analyzer")
        
        # Primary: Keyword-based cho tiếng Việt
        self.keyword_analyzer = KeywordAnalyzer()
        logger.info("✓ Keyword-based analyzer ready (Vietnamese)")
        
        # Optional: FinBERT cho tiếng Anh
        self.finbert = None
        self.finbert_tokenizer = None
        self.device = None
        
        if use_finbert and HAS_FINBERT:
            try:
                self.finbert_tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                self.finbert = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')
                self.finbert.eval()
                self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.finbert.to(self.device)
                logger.info(f"✓ FinBERT ready on {self.device} (English)")
            except Exception as e:
                logger.warning(f"Failed to load FinBERT: {e}")
                self.finbert = None
    
    def _is_vietnamese(self, text: str) -> bool:
        """
        Kiểm tra văn bản có phải tiếng Việt không
        
        Simple heuristic: Tìm các ký tự có dấu tiếng Việt
        """
        vietnamese_chars = 'àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        vietnamese_chars += vietnamese_chars.upper()
        
        for char in vietnamese_chars:
            if char in text:
                return True
        
        return False
    
    def analyze(self, text: str, method: str = 'auto') -> Dict:
        """
        Phân tích sentiment với hybrid approach
        
        Args:
            text: Văn bản cần phân tích
            method: 'auto', 'keyword', 'finbert'
        
        Returns:
            {
                'sentiment': str,
                'positive': float,
                'negative': float,
                'neutral': float,
                'sentiment_score': float,  # [-1, 1]
                'confidence': float,
                'method': str,
                'explanation': str
            }
        """
        if not text or not isinstance(text, str):
            return self._neutral_result(method='none')
        
        # Auto-detect language
        if method == 'auto':
            is_vietnamese = self._is_vietnamese(text)
            method = 'keyword' if is_vietnamese else 'finbert'
        
        # Method 1: Keyword-based (fast, accurate for Vietnamese)
        if method == 'keyword':
            sentiment, score, impact = self.keyword_analyzer.analyze(text)
            
            # Convert to standard format
            return {
                'sentiment': sentiment.value,
                'positive': max(0, score),
                'negative': abs(min(0, score)),
                'neutral': 1 - abs(score),
                'sentiment_score': score,
                'confidence': abs(score),
                'method': 'keyword-based',
                'explanation': impact
            }
        
        # Method 2: FinBERT (for English)
        elif method == 'finbert' and self.finbert is not None:
            return self._analyze_with_finbert(text)
        
        # Fallback: keyword
        else:
            sentiment, score, impact = self.keyword_analyzer.analyze(text)
            return {
                'sentiment': sentiment.value,
                'positive': max(0, score),
                'negative': abs(min(0, score)),
                'neutral': 1 - abs(score),
                'sentiment_score': score,
                'confidence': abs(score),
                'method': 'keyword-based (fallback)',
                'explanation': impact
            }
    
    def _analyze_with_finbert(self, text: str) -> Dict:
        """Phân tích với FinBERT"""
        try:
            with torch.no_grad():
                inputs = self.finbert_tokenizer(
                    text,
                    return_tensors='pt',
                    truncation=True,
                    max_length=512,
                    padding=True
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                outputs = self.finbert(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probs.cpu().numpy()[0]
            
            # FinBERT output: [positive, negative, neutral]
            positive = float(probs[0])
            negative = float(probs[1])
            neutral = float(probs[2])
            
            # Determine sentiment
            scores = {'positive': positive, 'negative': negative, 'neutral': neutral}
            sentiment = max(scores, key=scores.get)
            
            # Sentiment score [-1, 1]
            sentiment_score = positive - negative
            
            # Explanation
            if abs(sentiment_score) > 0.5:
                explanation = f"FinBERT high confidence: {sentiment}"
            elif abs(sentiment_score) > 0.2:
                explanation = f"FinBERT moderate confidence: {sentiment}"
            else:
                explanation = f"FinBERT low confidence: neutral"
            
            return {
                'sentiment': sentiment,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'sentiment_score': sentiment_score,
                'confidence': scores[sentiment],
                'method': 'finbert',
                'explanation': explanation
            }
        
        except Exception as e:
            logger.error(f"FinBERT error: {e}")
            return self._neutral_result(method='finbert-error')
    
    def _neutral_result(self, method: str) -> Dict:
        """Return neutral result"""
        return {
            'sentiment': 'neutral',
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0,
            'sentiment_score': 0.0,
            'confidence': 1.0,
            'method': method,
            'explanation': 'No analysis'
        }
    
    def analyze_batch(self, texts: List[str], method: str = 'auto') -> List[Dict]:
        """
        Phân tích nhiều văn bản
        
        Tự động phân loại Việt/Anh và dùng method phù hợp
        """
        results = []
        
        for text in texts:
            result = self.analyze(text, method=method)
            results.append(result)
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_col: str = 'text') -> pd.DataFrame:
        """
        Phân tích sentiment cho DataFrame
        
        Adds columns: sentiment, positive, negative, neutral, sentiment_score, 
                     confidence, method, explanation
        """
        logger.info(f"🔍 Analyzing sentiment for {len(df)} texts")
        
        df_result = df.copy()
        
        texts = df[text_col].fillna('').tolist()
        sentiments = self.analyze_batch(texts)
        
        # Add columns
        for key in sentiments[0].keys():
            df_result[key] = [s[key] for s in sentiments]
        
        # Statistics
        method_counts = df_result['method'].value_counts()
        logger.info(f"✓ Methods used: {method_counts.to_dict()}")
        
        sentiment_counts = df_result['sentiment'].value_counts()
        logger.info(f"✓ Sentiments: {sentiment_counts.to_dict()}")
        
        return df_result
    
    def aggregate_by_date(self, df: pd.DataFrame, date_col: str = 'date',
                         symbol_col: str = 'symbol') -> pd.DataFrame:
        """
        Tổng hợp sentiment theo ngày
        
        Returns:
            DataFrame với daily sentiment scores
        """
        logger.info(f"📊 Aggregating sentiment by date")
        
        # Ensure date column
        df['date'] = pd.to_datetime(df[date_col])
        
        # Group by date + symbol
        agg_dict = {
            'sentiment_score': ['mean', 'std', 'min', 'max'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean',
            'confidence': 'mean',
            'sentiment': lambda x: x.mode()[0] if len(x) > 0 else 'neutral'
        }
        
        if symbol_col in df.columns:
            group_cols = ['date', symbol_col]
        else:
            group_cols = ['date']
        
        aggregated = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Flatten columns
        aggregated.columns = [
            group_cols[0], group_cols[1] if len(group_cols) > 1 else None,
            'daily_sentiment_mean', 'daily_sentiment_std',
            'daily_sentiment_min', 'daily_sentiment_max',
            'daily_positive', 'daily_negative', 'daily_neutral',
            'daily_confidence', 'daily_sentiment_mode'
        ]
        aggregated.columns = [c for c in aggregated.columns if c is not None]
        
        # Count news
        news_count = df.groupby(group_cols).size().reset_index(name='news_count')
        aggregated = aggregated.merge(news_count, on=group_cols)
        
        logger.info(f"✓ Aggregated {len(aggregated)} days")
        
        return aggregated


# ============ Integration với Pipeline cũ ============

class EnhancedSentimentPipeline:
    """
    Pipeline nâng cấp với Hybrid Analyzer
    
    7 Bước:
    1. Thu thập tin tức (news_service)
    2. Làm sạch văn bản
    3. Tokenization (tự động trong analyzer)
    4. Embedding (tự động trong analyzer)
    5. Dự đoán sentiment (Hybrid: keyword/FinBERT)
    6. Chuyển về dạng số
    7. Gộp vào model
    """
    
    def __init__(self, use_finbert: bool = False):
        """
        Args:
            use_finbert: Có load FinBERT không (tốn RAM)
        """
        logger.info("="*60)
        logger.info("🚀 Enhanced Sentiment Pipeline với Hybrid Analyzer")
        logger.info("="*60)
        
        self.analyzer = HybridSentimentAnalyzer(use_finbert=use_finbert)
        
        logger.info("✓ Pipeline ready")
    
    def process_news_dataframe(self, news_df: pd.DataFrame, 
                               text_col: str = 'text') -> pd.DataFrame:
        """
        Xử lý DataFrame tin tức
        
        Input: DataFrame với columns [date, symbol, text, ...]
        Output: DataFrame with sentiment analysis
        """
        # Step 2: Clean (if needed)
        if 'text_clean' not in news_df.columns:
            from src.sentiment_pipeline import TextCleaner
            cleaner = TextCleaner()
            news_df['text_clean'] = news_df[text_col].apply(cleaner.clean)
            text_col = 'text_clean'
        
        # Steps 3-5: Analyze
        news_df = self.analyzer.analyze_dataframe(news_df, text_col=text_col)
        
        # Step 6: Aggregate by date
        daily_sentiment = self.analyzer.aggregate_by_date(news_df)
        
        return news_df, daily_sentiment
    
    def merge_with_price_data(self, price_df: pd.DataFrame, 
                             sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """
        Step 7: Merge sentiment vào price data
        """
        logger.info("🔗 Merging sentiment with price data")
        
        # Ensure date types
        price_df['date'] = pd.to_datetime(price_df['date'])
        sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
        
        # Determine merge keys
        merge_keys = ['date']
        if 'symbol' in price_df.columns and 'symbol' in sentiment_df.columns:
            merge_keys.append('symbol')
        
        # Merge
        merged = price_df.merge(sentiment_df, on=merge_keys, how='left')
        
        # Fill missing values
        sentiment_cols = [c for c in merged.columns if 'sentiment' in c or 'daily_' in c]
        for col in sentiment_cols:
            if merged[col].dtype in [np.float64, np.float32]:
                merged[col].fillna(0, inplace=True)
        
        if 'news_count' in merged.columns:
            merged[col].fillna(0, inplace=True)
        
        logger.info(f"✓ Merged {len(merged)} rows")
        logger.info(f"  Sentiment coverage: {(merged.get('news_count', pd.Series([0])) > 0).sum() / len(merged) * 100:.1f}%")
        
        return merged


# ============ Test ============

def test_hybrid():
    """Test hybrid analyzer"""
    logger.info("\n" + "="*60)
    logger.info("🧪 TEST HYBRID SENTIMENT ANALYZER")
    logger.info("="*60)
    
    analyzer = HybridSentimentAnalyzer(use_finbert=False)
    
    test_cases = [
        "Vinamilk công bố lợi nhuận quý 3 tăng 25% so với cùng kỳ",
        "Thị trường chứng khoán sụt giảm mạnh do lo ngại lãi suất",
        "FPT ký hợp đồng xuất khẩu phần mềm 50 triệu USD",
        "Ngân hàng cảnh báo rủi ro tín dụng bất động sản",
        "VIC ra mắt dự án Vinhomes Ocean Park 3 quy mô lớn"
    ]
    
    logger.info("\n📊 Sentiment Analysis Results:")
    for i, text in enumerate(test_cases, 1):
        result = analyzer.analyze(text)
        
        logger.info(f"\n{i}. {text}")
        logger.info(f"   Sentiment: {result['sentiment'].upper()} ({result['method']})")
        logger.info(f"   Score: {result['sentiment_score']:.2f} (confidence: {result['confidence']:.2f})")
        logger.info(f"   Scores: pos={result['positive']:.2f}, neg={result['negative']:.2f}, neu={result['neutral']:.2f}")
        logger.info(f"   → {result['explanation']}")


if __name__ == "__main__":
    test_hybrid()
