"""
Script để test Indicators Processor và Daily Scheduler
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.database.connection import get_db
from src.features.indicators_processor import IndicatorsProcessor
from src.scheduler.daily_scheduler import DailyDataScheduler

def test_indicators_processor():
    """Test tính toán indicators"""
    print("=" * 80)
    print("TEST 1: INDICATORS PROCESSOR")
    print("=" * 80)
    
    db = next(get_db())
    
    try:
        processor = IndicatorsProcessor(db)
        
        # Test với 1 stock
        from src.database.models import Stock
        stock = db.query(Stock).first()
        
        if not stock:
            print("❌ No stocks in database. Please run init-db first.")
            return
        
        print(f"\n📊 Testing indicators calculation for {stock.symbol}...")
        success = processor.process_stock(stock.id, days=365)
        
        if success:
            print(f"✅ Indicators calculated successfully for {stock.symbol}")
        else:
            print(f"❌ Failed to calculate indicators for {stock.symbol}")
        
        # Test với all stocks
        print(f"\n📊 Testing indicators calculation for ALL stocks...")
        result = processor.process_all_stocks(days=365)
        
        print(f"\n📈 Results:")
        print(f"   Success: {result['success']}")
        print(f"   Failed: {result['failed']}")
        print(f"   Total: {result['total']}")
        
    finally:
        db.close()


def test_scheduler():
    """Test scheduler"""
    print("\n" + "=" * 80)
    print("TEST 2: DAILY SCHEDULER")
    print("=" * 80)
    
    scheduler = DailyDataScheduler()
    
    print("\n📅 Scheduler initialized")
    print("   Jobs configured:")
    print("   - Daily update: Mon-Fri at 18:00")
    print("   - Weekly catch-up: Sunday at 10:00")
    
    # Test chạy ngay
    choice = input("\n⚠️  Run daily update job now? (y/n): ")
    
    if choice.lower() == 'y':
        print("\n🚀 Running daily update job...")
        scheduler.run_now()
        print("\n✅ Job completed")
    else:
        print("\n⏭️  Skipped manual run")
    
    # Start scheduler
    choice = input("\n⚠️  Start background scheduler? (y/n): ")
    
    if choice.lower() == 'y':
        scheduler.start()
        print(f"\n✅ Scheduler started")
        print(f"   Next run: {scheduler.get_next_run_time()}")
        print("\n💡 Scheduler is now running in background.")
        print("   Press Ctrl+C to stop.")
        
        try:
            import time
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping scheduler...")
            scheduler.stop()
            print("✅ Scheduler stopped")
    else:
        print("\n⏭️  Scheduler not started")


def main():
    """Main test function"""
    print("\n" + "=" * 80)
    print("🧪 TESTING: INDICATORS PROCESSOR & SCHEDULER")
    print("=" * 80)
    
    print("\nWhat would you like to test?")
    print("1. Indicators Processor only")
    print("2. Scheduler only")
    print("3. Both")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ")
    
    if choice == '1':
        test_indicators_processor()
    elif choice == '2':
        test_scheduler()
    elif choice == '3':
        test_indicators_processor()
        test_scheduler()
    elif choice == '4':
        print("\n👋 Exiting...")
        return
    else:
        print("\n❌ Invalid choice")
        return
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
