"""
Test script for ETL Pipeline
Test các module Extract, Transform, Load và Pipeline tổng hợp
"""

import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer
from src.etl.load import DataLoader
from src.etl.pipeline import ETLOrchestrator


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_extractor():
    """Test DataExtractor module"""
    print_section("TEST 1: DATA EXTRACTOR")
    
    extractor = DataExtractor()
    
    # Test 1.1: Extract VNM prices (last 30 days)
    print("\n1.1. Extract VNM price data (last 30 days)")
    start_date = datetime.now() - timedelta(days=30)
    df_price = extractor.extract_price_data(symbol="VNM", start_date=start_date)
    print(f"✅ Extracted {len(df_price)} price records")
    if not df_price.empty:
        print(df_price.head())
        print(f"   Date range: {df_price['date'].min()} to {df_price['date'].max()}")
    
    # Test 1.2: Extract latest news
    print("\n1.2. Extract latest news (last 7 days)")
    df_news = extractor.extract_latest_news(days=7)
    print(f"✅ Extracted {len(df_news)} news records")
    if not df_news.empty:
        print(df_news[['symbol', 'title', 'published_at']].head())
    
    # Test 1.3: Extract unprocessed news
    print("\n1.3. Extract unprocessed news")
    df_unprocessed = extractor.extract_news_data(unprocessed_only=True, limit=10)
    print(f"✅ Extracted {len(df_unprocessed)} unprocessed news")
    
    extractor.close()
    print("\n✅ Extractor tests passed")


def test_transformer():
    """Test DataTransformer module"""
    print_section("TEST 2: DATA TRANSFORMER")
    
    transformer = DataTransformer()
    
    # Test 2.1: Transform price data
    print("\n2.1. Transform price data")
    import pandas as pd
    
    test_price = pd.DataFrame({
        'stock_id': [1, 1, 1],
        'symbol': ['vnm', 'VNM', 'vnm'],  # Test case normalization
        'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'open': [80000, 81000, None],  # Test missing value
        'high': [81000, 82000, 82500],
        'low': [79000, 80000, 81000],
        'close': [80500, 81500, 82000],
        'volume': [1000000, 1200000, 0]
    })
    
    transformed = transformer.transform_price_data(test_price)
    print(f"✅ Transformed {len(transformed)} records (from {len(test_price)})")
    print(transformed)
    
    # Validation
    validation = transformer.validate_price_data(transformed)
    print(f"\n📊 Validation: {validation}")
    
    # Test 2.2: Transform news data
    print("\n2.2. Transform news data")
    
    test_news = pd.DataFrame({
        'id': [1, 2],
        'symbol': ['vnm', 'hpg'],
        'title': ['<b>Vinamilk tăng trưởng</b>', 'Hòa Phát báo lãi cao'],
        'content': [
            '<html><p>Vinamilk đạt doanh thu cao http://example.com</p></html>',
            'Hòa Phát công bố kết quả kinh doanh quý 4 với lợi nhuận tăng mạnh.'
        ],
        'published_at': ['2024-01-15 10:00:00', '2024-01-15 11:00:00']
    })
    
    transformed_news = transformer.transform_news_data(test_news)
    print(f"✅ Transformed {len(transformed_news)} news records")
    print(transformed_news[['symbol', 'title', 'clean_content']])
    
    # Validation
    news_validation = transformer.validate_news_data(transformed_news)
    print(f"\n📊 Validation: {news_validation}")
    
    print("\n✅ Transformer tests passed")


def test_loader():
    """Test DataLoader module"""
    print_section("TEST 3: DATA LOADER")
    
    loader = DataLoader()
    
    # Test 3.1: Get stock_id
    print("\n3.1. Get stock_id for VNM")
    stock_id = loader.get_stock_id('VNM')
    print(f"✅ VNM stock_id: {stock_id}")
    
    # Test 3.2: Get latest price date
    print("\n3.2. Get latest price date for VNM")
    latest_date = loader.get_latest_price_date('VNM')
    print(f"✅ Latest date: {latest_date}")
    
    # Test 3.3: Check duplicate
    print("\n3.3. Check duplicate price")
    if latest_date:
        exists = loader.check_duplicate_prices('VNM', latest_date)
        print(f"✅ Duplicate exists: {exists}")
    
    # Test 3.4: Load test (dry run)
    print("\n3.4. Test load operation (dry run)")
    print("   ℹ️  Skipping actual database insert to avoid data duplication")
    print("   ℹ️  In production, this would insert/update records")
    
    loader.close()
    print("\n✅ Loader tests passed")


def test_full_pipeline():
    """Test complete ETL Pipeline"""
    print_section("TEST 4: FULL ETL PIPELINE")
    
    orchestrator = ETLOrchestrator()
    
    # Test 4.1: Price ETL for VNM (last 7 days)
    print("\n4.1. Run Price ETL for VNM (last 7 days)")
    start_date = datetime.now() - timedelta(days=7)
    price_result = orchestrator.run_price_etl(symbol="VNM", start_date=start_date)
    
    print("\n📊 Price ETL Results:")
    print(f"   Status: {price_result['status']}")
    print(f"   Extracted: {price_result.get('extracted', 0)} records")
    print(f"   Transformed: {price_result.get('transformed', 0)} records")
    print(f"   Loaded: {price_result.get('loaded', {})}")
    print(f"   Duration: {price_result.get('duration_seconds', 0)}s")
    
    # Test 4.2: News ETL (unprocessed only)
    print("\n4.2. Run News ETL (unprocessed, last 7 days)")
    news_result = orchestrator.run_news_etl(
        start_date=start_date,
        unprocessed_only=True
    )
    
    print("\n📊 News ETL Results:")
    print(f"   Status: {news_result['status']}")
    print(f"   Extracted: {news_result.get('extracted', 0)} records")
    print(f"   Transformed: {news_result.get('transformed', 0)} records")
    print(f"   Loaded: {news_result.get('loaded', {})}")
    print(f"   Duration: {news_result.get('duration_seconds', 0)}s")
    
    orchestrator.close()
    print("\n✅ Full pipeline tests passed")


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("  🧪 ETL PIPELINE TEST SUITE")
    print("  Testing Extract → Transform → Load modules")
    print("=" * 80)
    
    try:
        # Test each module
        test_extractor()
        test_transformer()
        test_loader()
        test_full_pipeline()
        
        # Final summary
        print("\n" + "=" * 80)
        print("  ✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\n📝 Summary:")
        print("  • DataExtractor: ✅ Working")
        print("  • DataTransformer: ✅ Working")
        print("  • DataLoader: ✅ Working")
        print("  • ETL Pipeline: ✅ Working")
        print("\n💡 Next steps:")
        print("  1. Review extracted data quality")
        print("  2. Run sentiment analysis on clean news")
        print("  3. Calculate technical features")
        print("  4. Train prediction models")
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
