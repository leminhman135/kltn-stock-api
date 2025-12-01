"""
PhoBERT Vietnamese Sentiment Analyzer - Offline Script
Chạy trên máy local, phân tích tin tức tiếng Việt và lưu kết quả vào database
Sử dụng PhoBERT - model được train cho tiếng Việt
"""

import os
import sys
import logging
import argparse
import feedparser
import psycopg2
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VietnameseSentimentAnalyzer:
    """Vietnamese Sentiment Analyzer using PhoBERT"""
    
    def __init__(self):
        logger.info("🔄 Loading Vietnamese Sentiment model (first time may take a while)...")
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            import torch.nn.functional as F
            
            self.torch = torch
            self.F = F
            
            # PhoBERT Vietnamese Sentiment model
            # Model được fine-tune cho sentiment analysis tiếng Việt
            model_name = "wonrax/phobert-base-vietnamese-sentiment"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # Move to GPU if available
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            device_name = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
            logger.info(f"✅ PhoBERT Vietnamese Sentiment loaded on {device_name}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            logger.info("Run: pip install transformers torch")
            raise
    
    def analyze(self, text: str) -> Dict:
        """Analyze Vietnamese text sentiment"""
        if not text or len(text.strip()) < 10:
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34
            }
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text[:256],  # PhoBERT max 256 tokens
                return_tensors="pt",
                truncation=True,
                max_length=256,
                padding=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with self.torch.no_grad():
                outputs = self.model(**inputs)
                probs = self.F.softmax(outputs.logits, dim=1)[0]
            
            # Model output: NEG (0), POS (1), NEU (2)
            neg_score = probs[0].item()
            pos_score = probs[1].item()
            neu_score = probs[2].item()
            
            # Determine sentiment
            scores = {'negative': neg_score, 'positive': pos_score, 'neutral': neu_score}
            sentiment = max(scores, key=scores.get)
            
            # Calculate overall score (-1 to 1)
            overall_score = pos_score - neg_score
            
            return {
                'sentiment': sentiment,
                'score': round(overall_score, 4),
                'positive': round(pos_score, 4),
                'negative': round(neg_score, 4),
                'neutral': round(neu_score, 4)
            }
            
        except Exception as e:
            logger.warning(f"Analysis error: {e}")
            return {
                'sentiment': 'neutral',
                'score': 0.0,
                'positive': 0.33,
                'negative': 0.33,
                'neutral': 0.34
            }


class DatabaseManager:
    """Manage database connections and operations"""
    
    def __init__(self):
        self.db_url = os.getenv('DATABASE_URL')
        if not self.db_url:
            raise ValueError("DATABASE_URL not found in environment variables")
        
        # Test connection and create tables
        self._create_tables()
    
    def get_connection(self):
        """Get database connection"""
        return psycopg2.connect(self.db_url)
    
    def _create_tables(self):
        """Create required tables if not exist"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Create analyzed_news table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS analyzed_news (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20),
                        title TEXT NOT NULL,
                        summary TEXT,
                        url TEXT,
                        source VARCHAR(100),
                        published_at TIMESTAMP,
                        sentiment VARCHAR(20),
                        sentiment_score FLOAT,
                        positive_score FLOAT,
                        negative_score FLOAT,
                        neutral_score FLOAT,
                        analyzed_at TIMESTAMP DEFAULT NOW(),
                        model_version VARCHAR(50) DEFAULT 'finbert-v1',
                        UNIQUE(url)
                    )
                """)
                
                # Create sentiment_summary table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS sentiment_summary (
                        id SERIAL PRIMARY KEY,
                        symbol VARCHAR(20),
                        date DATE,
                        positive_count INT DEFAULT 0,
                        negative_count INT DEFAULT 0,
                        neutral_count INT DEFAULT 0,
                        avg_score FLOAT DEFAULT 0,
                        overall_sentiment VARCHAR(20),
                        news_count INT DEFAULT 0,
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(symbol, date)
                    )
                """)
                
                # Create index
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analyzed_news_symbol 
                    ON analyzed_news(symbol)
                """)
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_analyzed_news_published 
                    ON analyzed_news(published_at)
                """)
                
                conn.commit()
                logger.info("✅ Database tables ready")
        except Exception as e:
            conn.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            conn.close()
    
    def save_news(self, news: Dict) -> bool:
        """Save analyzed news to database"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO analyzed_news 
                    (symbol, title, summary, url, source, published_at,
                     sentiment, sentiment_score, positive_score, negative_score, neutral_score)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (url) DO UPDATE SET
                        sentiment = EXCLUDED.sentiment,
                        sentiment_score = EXCLUDED.sentiment_score,
                        positive_score = EXCLUDED.positive_score,
                        negative_score = EXCLUDED.negative_score,
                        neutral_score = EXCLUDED.neutral_score,
                        analyzed_at = NOW()
                """, (
                    news.get('symbol'),
                    news.get('title'),
                    news.get('summary'),
                    news.get('url'),
                    news.get('source'),
                    news.get('published_at'),
                    news.get('sentiment'),
                    news.get('sentiment_score'),
                    news.get('positive_score'),
                    news.get('negative_score'),
                    news.get('neutral_score')
                ))
                conn.commit()
                return True
        except Exception as e:
            conn.rollback()
            logger.warning(f"Error saving news: {e}")
            return False
        finally:
            conn.close()
    
    def update_summary(self, symbol: str, date: datetime):
        """Update daily sentiment summary"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                # Get counts for the day
                cur.execute("""
                    SELECT 
                        COUNT(*) FILTER (WHERE sentiment = 'positive') as pos,
                        COUNT(*) FILTER (WHERE sentiment = 'negative') as neg,
                        COUNT(*) FILTER (WHERE sentiment = 'neutral') as neu,
                        AVG(sentiment_score) as avg_score,
                        COUNT(*) as total
                    FROM analyzed_news
                    WHERE symbol = %s 
                    AND DATE(published_at) = %s
                """, (symbol, date.date()))
                
                row = cur.fetchone()
                pos, neg, neu, avg_score, total = row
                
                # Determine overall sentiment
                if pos > neg and pos > neu:
                    overall = 'positive'
                elif neg > pos and neg > neu:
                    overall = 'negative'
                else:
                    overall = 'neutral'
                
                # Upsert summary
                cur.execute("""
                    INSERT INTO sentiment_summary 
                    (symbol, date, positive_count, negative_count, neutral_count, 
                     avg_score, overall_sentiment, news_count, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
                    ON CONFLICT (symbol, date) DO UPDATE SET
                        positive_count = EXCLUDED.positive_count,
                        negative_count = EXCLUDED.negative_count,
                        neutral_count = EXCLUDED.neutral_count,
                        avg_score = EXCLUDED.avg_score,
                        overall_sentiment = EXCLUDED.overall_sentiment,
                        news_count = EXCLUDED.news_count,
                        updated_at = NOW()
                """, (symbol, date.date(), pos or 0, neg or 0, neu or 0, 
                      avg_score or 0, overall, total or 0))
                
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.warning(f"Error updating summary: {e}")
        finally:
            conn.close()
    
    def get_stats(self) -> Dict:
        """Get current statistics"""
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE sentiment = 'positive') as positive,
                        COUNT(*) FILTER (WHERE sentiment = 'negative') as negative,
                        COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral,
                        COUNT(DISTINCT symbol) as symbols
                    FROM analyzed_news
                """)
                row = cur.fetchone()
                return {
                    'total': row[0] or 0,
                    'positive': row[1] or 0,
                    'negative': row[2] or 0,
                    'neutral': row[3] or 0,
                    'symbols': row[4] or 0
                }
        finally:
            conn.close()


class NewsFetcher:
    """Fetch news from RSS feeds"""
    
    # Vietnamese stock news RSS feeds - Expanded list
    RSS_FEEDS = {
        # CafeF
        'cafef_ck': 'https://cafef.vn/rss/thi-truong-chung-khoan.rss',
        'cafef_dn': 'https://cafef.vn/rss/doanh-nghiep.rss',
        'cafef_vimo': 'https://cafef.vn/rss/vi-mo-dau-tu.rss',
        # VnEconomy
        'vneconomy': 'https://vneconomy.vn/chung-khoan.rss',
        'vneconomy_dn': 'https://vneconomy.vn/doanh-nghiep.rss',
        # VnExpress
        'vnexpress_kd': 'https://vnexpress.net/rss/kinh-doanh.rss',
        'vnexpress_ck': 'https://vnexpress.net/rss/chung-khoan.rss',
        # Thanh Nien
        'thanhnien': 'https://thanhnien.vn/rss/tai-chinh-kinh-doanh.rss',
        # Tuoi Tre
        'tuoitre': 'https://tuoitre.vn/rss/kinh-doanh.rss',
        # Dan Tri
        'dantri': 'https://dantri.com.vn/rss/kinh-doanh.rss',
        # NDH
        'ndh': 'https://ndh.vn/rss/chung-khoan-1045.rss',
        # Vietstock
        'vietstock': 'https://finance.vietstock.vn/rss/tin-noi-bat.rss',
        # 24h
        '24h': 'https://cdn.24h.com.vn/upload/rss/taichinh.rss',
    }
    
    # VN30 stock symbols
    VN30_SYMBOLS = [
        'VNM', 'VIC', 'VHM', 'VCB', 'BID', 'CTG', 'TCB', 'MBB', 'HPG', 'FPT',
        'MWG', 'VPB', 'PLX', 'VJC', 'GAS', 'SAB', 'MSN', 'VRE', 'NVL', 'ACB',
        'GVR', 'STB', 'POW', 'BCM', 'SSI', 'VND', 'TPB', 'HDB', 'PDR', 'SHB'
    ]
    
    def __init__(self, days: int = 7):
        self.days = days
        self.cutoff_date = datetime.now() - timedelta(days=days)
    
    def fetch_all(self, symbols: List[str] = None) -> List[Dict]:
        """Fetch news from all RSS feeds"""
        symbols = symbols or self.VN30_SYMBOLS
        all_news = []
        
        for source, url in self.RSS_FEEDS.items():
            try:
                logger.info(f"📡 Fetching from {source}...")
                feed = feedparser.parse(url)
                
                for entry in feed.entries[:50]:  # Limit per source
                    try:
                        # Parse date
                        published = None
                        if hasattr(entry, 'published_parsed') and entry.published_parsed:
                            published = datetime(*entry.published_parsed[:6])
                        elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                            published = datetime(*entry.updated_parsed[:6])
                        else:
                            published = datetime.now()
                        
                        # Skip old news
                        if published < self.cutoff_date:
                            continue
                        
                        title = entry.get('title', '')
                        summary = entry.get('summary', entry.get('description', ''))
                        
                        # Match symbols in title/summary
                        text = f"{title} {summary}".upper()
                        matched_symbols = [s for s in symbols if s in text]
                        
                        if matched_symbols:
                            for symbol in matched_symbols:
                                all_news.append({
                                    'symbol': symbol,
                                    'title': title,
                                    'summary': summary[:500] if summary else '',
                                    'url': entry.get('link', ''),
                                    'source': source,
                                    'published_at': published
                                })
                        else:
                            # General market news
                            all_news.append({
                                'symbol': 'MARKET',
                                'title': title,
                                'summary': summary[:500] if summary else '',
                                'url': entry.get('link', ''),
                                'source': source,
                                'published_at': published
                            })
                    except Exception as e:
                        continue
                        
                logger.info(f"  ✅ Got {len(feed.entries)} articles from {source}")
                
            except Exception as e:
                logger.warning(f"  ⚠️ Failed to fetch {source}: {e}")
        
        # Remove duplicates by URL
        seen_urls = set()
        unique_news = []
        for news in all_news:
            if news['url'] not in seen_urls:
                seen_urls.add(news['url'])
                unique_news.append(news)
        
        logger.info(f"📰 Total unique articles: {len(unique_news)}")
        return unique_news


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='FinBERT News Sentiment Analyzer')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to analyze')
    parser.add_argument('--days', type=int, default=7, help='Days of news to analyze')
    parser.add_argument('--limit', type=int, default=100, help='Max news to analyze')
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 FinBERT News Sentiment Analyzer")
    print("=" * 60)
    
    # Initialize database
    try:
        db = DatabaseManager()
        logger.info("✅ Database connected")
    except Exception as e:
        logger.error(f"❌ Database error: {e}")
        return
    
    # Get current stats
    stats_before = db.get_stats()
    logger.info(f"📊 Current: {stats_before['total']} news, {stats_before['symbols']} symbols")
    
    # Fetch news
    fetcher = NewsFetcher(days=args.days)
    news_list = fetcher.fetch_all(args.symbols)
    
    if not news_list:
        logger.info("❌ No news found")
        return
    
    # Limit
    news_list = news_list[:args.limit]
    logger.info(f"📰 Will analyze {len(news_list)} articles")
    
    # Load Vietnamese Sentiment model (PhoBERT)
    try:
        analyzer = VietnameseSentimentAnalyzer()
    except Exception as e:
        logger.error(f"❌ Failed to load PhoBERT: {e}")
        return
    
    # Analyze and save
    saved = 0
    symbols_dates = set()
    
    for i, news in enumerate(news_list):
        # Analyze
        text = f"{news['title']}. {news.get('summary', '')}"
        result = analyzer.analyze(text)
        
        # Add sentiment to news
        news['sentiment'] = result['sentiment']
        news['sentiment_score'] = result['score']
        news['positive_score'] = result['positive']
        news['negative_score'] = result['negative']
        news['neutral_score'] = result['neutral']
        
        # Save to database
        if db.save_news(news):
            saved += 1
            symbols_dates.add((news['symbol'], news['published_at']))
        
        # Log progress
        emoji = '📈' if result['sentiment'] == 'positive' else '📉' if result['sentiment'] == 'negative' else '➡️'
        logger.info(f"[{i+1}/{len(news_list)}] {emoji} {news['symbol']}: {result['sentiment']} ({result['score']:.2f})")
    
    # Update summaries
    logger.info("📊 Updating daily summaries...")
    for symbol, pub_date in symbols_dates:
        db.update_summary(symbol, pub_date)
    
    # Final stats
    stats_after = db.get_stats()
    
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Saved: {saved} news articles")
    print(f"📈 Positive: {stats_after['positive']}")
    print(f"📉 Negative: {stats_after['negative']}")
    print(f"➡️ Neutral: {stats_after['neutral']}")
    print(f"📁 Total in DB: {stats_after['total']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
