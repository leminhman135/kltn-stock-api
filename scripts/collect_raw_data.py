"""
Script thu thập dữ liệu thô từ API và lưu vào SQL
Usage:
    python scripts/collect_raw_data.py --symbols VNM HPG VCB --days 30
    python scripts/collect_raw_data.py --all --days 7
"""

import sys
import os
import argparse
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.etl.raw_data_collector import RawDataCollector


def main():
    parser = argparse.ArgumentParser(description='Collect raw stock data from API to SQL')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to collect')
    parser.add_argument('--all', action='store_true', help='Collect all stocks from database')
    parser.add_argument('--days', type=int, default=30, help='Number of days to collect (default: 30)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("📡 RAW DATA COLLECTION FROM API TO SQL")
    print("=" * 80)
    
    # Initialize collector
    collector = RawDataCollector()
    
    try:
        if args.all:
            # Collect all stocks
            print(f"\n📊 Collecting data for ALL stocks (last {args.days} days)")
            stats = collector.collect_all_stocks(days=args.days)
        
        elif args.symbols:
            # Collect specific symbols
            print(f"\n📊 Collecting data for: {', '.join(args.symbols)}")
            print(f"📅 Date range: Last {args.days} days")
            
            to_date = datetime.now()
            from_date = to_date - timedelta(days=args.days)
            
            stats = collector.collect_multiple_symbols(
                symbols=args.symbols,
                from_date=from_date,
                to_date=to_date
            )
        
        else:
            print("\n⚠️  Please specify --symbols or --all")
            parser.print_help()
            return
        
        # Print summary
        print("\n" + "=" * 80)
        print("📊 COLLECTION SUMMARY")
        print("=" * 80)
        print(f"Status: {stats['status']}")
        print(f"Symbols processed: {stats.get('symbols_processed', 0)}")
        print(f"Total collected: {stats.get('total_collected', 0)}")
        print(f"Total inserted: {stats.get('total_inserted', 0)}")
        print(f"Total updated: {stats.get('total_updated', 0)}")
        print(f"Total failed: {stats.get('total_failed', 0)}")
        
        if 'symbol_details' in stats:
            print("\n📋 Details by symbol:")
            for symbol, detail in stats['symbol_details'].items():
                print(f"  {symbol}: collected={detail.get('collected', 0)}, "
                      f"inserted={detail.get('inserted', 0)}, "
                      f"updated={detail.get('updated', 0)}")
        
        print("=" * 80)
        print("\n✅ Raw data collection completed!")
        print(f"💾 Data saved to: raw_stock_data table in PostgreSQL")
        print(f"🔄 Next step: Run ETL pipeline to transform and load data")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        collector.close()


if __name__ == "__main__":
    main()
