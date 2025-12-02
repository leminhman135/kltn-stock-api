"""
Script chạy Sentiment Pipeline đầy đủ cho mã cổ phiếu

Usage:
    python scripts/run_sentiment_pipeline.py VNM --days 30
    python scripts/run_sentiment_pipeline.py --all --days 7
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline_for_symbol(symbol: str, days: int = 30, save_to_db: bool = False):
    """
    Chạy sentiment pipeline cho một mã cổ phiếu
    
    7 Bước:
    1. Thu thập tin tức
    2. Làm sạch văn bản
    3. Tokenization
    4. Embedding
    5. Dự đoán sentiment
    6. Chuyển về dạng số
    7. Export kết quả
    """
    logger.info("="*80)
    logger.info(f"🚀 SENTIMENT PIPELINE - {symbol}")
    logger.info("="*80)
    
    from src.hybrid_sentiment import EnhancedSentimentPipeline
    from src.news_service import news_service
    
    # Initialize pipeline
    pipeline = EnhancedSentimentPipeline(use_finbert=False)
    
    # Step 1: Collect news
    logger.info(f"\n📰 Bước 1: Thu thập tin tức cho {symbol}")
    news_articles = news_service.get_all_news(symbol=symbol, limit=100)
    
    if not news_articles:
        logger.warning(f"❌ Không có tin tức cho {symbol}")
        return None
    
    # Convert to DataFrame
    news_data = []
    for article in news_articles:
        news_data.append({
            'date': article.published_at,
            'symbol': symbol,
            'title': article.title,
            'summary': article.summary,
            'text': f"{article.title} {article.summary}",
            'url': article.url,
            'source': article.source
        })
    
    news_df = pd.DataFrame(news_data)
    news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
    
    # Filter by days
    from datetime import timedelta
    cutoff = datetime.now() - timedelta(days=days)
    news_df = news_df[news_df['date'] >= cutoff]
    
    logger.info(f"✓ Thu thập {len(news_df)} tin tức trong {days} ngày qua")
    
    if len(news_df) == 0:
        logger.warning("❌ Không có tin tức gần đây")
        return None
    
    # Steps 2-6: Process
    logger.info(f"\n🔄 Bước 2-6: Xử lý & phân tích sentiment")
    news_analyzed, daily_sentiment = pipeline.process_news_dataframe(news_df, text_col='text')
    
    # Display results
    logger.info(f"\n" + "="*80)
    logger.info("📊 KẾT QUẢ PHÂN TÍCH")
    logger.info("="*80)
    
    # Overall stats
    sentiment_counts = news_analyzed['sentiment'].value_counts()
    logger.info(f"\n📈 Tổng hợp toàn bộ tin tức:")
    for sentiment, count in sentiment_counts.items():
        pct = count / len(news_analyzed) * 100
        logger.info(f"  {sentiment.upper()}: {count} tin ({pct:.1f}%)")
    
    avg_score = news_analyzed['sentiment_score'].mean()
    logger.info(f"\n💯 Điểm sentiment trung bình: {avg_score:.3f}")
    
    if avg_score > 0.2:
        logger.info(f"  → 🟢 TIN TỨC TÍCH CỰC cho {symbol}")
    elif avg_score < -0.2:
        logger.info(f"  → 🔴 TIN TỨC TIÊU CỰC cho {symbol}")
    else:
        logger.info(f"  → 🟡 TIN TỨC TRUNG LẬP cho {symbol}")
    
    # Daily sentiment
    logger.info(f"\n📅 Sentiment theo ngày (gần nhất):")
    logger.info(daily_sentiment.head(10).to_string())
    
    # Step 7: Save results
    logger.info(f"\n💾 Bước 7: Lưu kết quả")
    
    # Save to CSV
    output_dir = "data/sentiment_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    news_file = f"{output_dir}/{symbol}_news_{timestamp}.csv"
    news_analyzed.to_csv(news_file, index=False, encoding='utf-8-sig')
    logger.info(f"  ✓ News analysis: {news_file}")
    
    daily_file = f"{output_dir}/{symbol}_daily_{timestamp}.csv"
    daily_sentiment.to_csv(daily_file, index=False, encoding='utf-8-sig')
    logger.info(f"  ✓ Daily sentiment: {daily_file}")
    
    # Save to database (optional)
    if save_to_db:
        logger.info(f"\n💾 Lưu vào database...")
        try:
            save_to_database(news_analyzed, daily_sentiment, symbol)
            logger.info(f"  ✓ Database updated")
        except Exception as e:
            logger.error(f"  ❌ Database error: {e}")
    
    logger.info(f"\n✅ HOÀN THÀNH pipeline cho {symbol}")
    
    return {
        'symbol': symbol,
        'news_count': len(news_analyzed),
        'days': days,
        'avg_sentiment_score': avg_score,
        'sentiment_distribution': sentiment_counts.to_dict(),
        'news_file': news_file,
        'daily_file': daily_file
    }


def save_to_database(news_df: pd.DataFrame, daily_df: pd.DataFrame, symbol: str):
    """
    Lưu kết quả vào database
    
    Tables:
    - analyzed_news: Chi tiết từng tin
    - daily_sentiment: Tổng hợp theo ngày
    """
    try:
        from sqlalchemy import create_engine
        import os
        
        db_url = os.getenv('DATABASE_URL')
        if not db_url:
            logger.warning("DATABASE_URL not found")
            return
        
        engine = create_engine(db_url)
        
        # Save news analysis
        news_df.to_sql(
            'analyzed_news',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        # Save daily sentiment
        daily_df.to_sql(
            'daily_sentiment',
            engine,
            if_exists='append',
            index=False,
            method='multi'
        )
        
        logger.info(f"✓ Saved to database")
        
    except Exception as e:
        logger.error(f"Database save error: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(description='Run Sentiment Analysis Pipeline')
    
    parser.add_argument('symbol', nargs='?', type=str, help='Stock symbol (e.g., VNM, VIC)')
    parser.add_argument('--all', action='store_true', help='Run for all major symbols')
    parser.add_argument('--days', type=int, default=30, help='Number of days to analyze')
    parser.add_argument('--db', action='store_true', help='Save to database')
    
    args = parser.parse_args()
    
    # List of major symbols
    major_symbols = [
        'VNM', 'VIC', 'VHM', 'HPG', 'FPT', 'MWG', 
        'VCB', 'BID', 'CTG', 'TCB', 'MBB',
        'MSN', 'SAB', 'PLX', 'VJC', 'SSI'
    ]
    
    if args.all:
        logger.info(f"🔄 Running pipeline for {len(major_symbols)} symbols")
        results = []
        
        for symbol in major_symbols:
            try:
                result = run_pipeline_for_symbol(symbol, days=args.days, save_to_db=args.db)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
        
        # Summary
        logger.info(f"\n" + "="*80)
        logger.info(f"📊 TỔNG KẾT")
        logger.info("="*80)
        logger.info(f"Processed: {len(results)} symbols")
        
        for result in results:
            logger.info(f"  {result['symbol']}: {result['news_count']} news, "
                       f"avg_score={result['avg_sentiment_score']:.3f}")
    
    elif args.symbol:
        run_pipeline_for_symbol(args.symbol.upper(), days=args.days, save_to_db=args.db)
    
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
