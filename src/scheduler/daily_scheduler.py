"""
Scheduler - Tự động cập nhật dữ liệu giá và tính indicators hàng ngày
Chạy vào 18:00 mỗi ngày (sau khi thị trường đóng cửa)
"""

import logging

# Try to import apscheduler (optional dependency)
try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    SCHEDULER_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ apscheduler not installed. Scheduler features disabled.")
    BackgroundScheduler = None
    CronTrigger = None
    SCHEDULER_AVAILABLE = False

from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import desc

from src.database.connection import get_db
from src.database.models import Stock, StockPrice
from src.data_collection import VNDirectAPI
from src.features.indicators_processor import IndicatorsProcessor
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DailyDataScheduler:
    """Scheduler để tự động cập nhật dữ liệu hàng ngày"""
    
    def __init__(self):
        if not SCHEDULER_AVAILABLE:
            raise ImportError("apscheduler is not installed. Install with: pip install apscheduler")
        self.scheduler = BackgroundScheduler()
        self.vndirect = VNDirectAPI()
        
    def fetch_and_save_stock_prices(self, db: Session) -> dict:
        """
        Tải dữ liệu giá mới nhất cho tất cả stocks và lưu vào DB
        
        Returns:
            Dict với thống kê: {success: int, failed: int, total: int, new_records: int}
        """
        try:
            stocks = db.query(Stock).filter(Stock.is_active == True).all()
            
            if not stocks:
                logger.warning("⚠️ No active stocks found")
                return {'success': 0, 'failed': 0, 'total': 0, 'new_records': 0}
            
            success_count = 0
            failed_count = 0
            total_new_records = 0
            today = datetime.now()
            
            logger.info(f"🔄 Starting daily price update for {len(stocks)} stocks")
            
            for stock in stocks:
                try:
                    # Tìm ngày cuối cùng có dữ liệu
                    last_price = db.query(StockPrice).filter(
                        StockPrice.stock_id == stock.id
                    ).order_by(desc(StockPrice.date)).first()
                    
                    if last_price:
                        # Fetch từ ngày cuối + 1
                        start_date = datetime.combine(last_price.date, datetime.min.time()) + timedelta(days=1)
                    else:
                        # Không có dữ liệu → fetch 30 ngày gần nhất
                        start_date = today - timedelta(days=30)
                    
                    # Nếu start_date > today thì skip (đã up to date)
                    if start_date.date() > today.date():
                        logger.info(f"✓ {stock.symbol}: Already up to date")
                        success_count += 1
                        continue
                    
                    # Fetch data từ VNDirect
                    df = self.vndirect.get_stock_price(
                        symbol=stock.symbol,
                        from_date=start_date.strftime('%Y-%m-%d'),
                        to_date=today.strftime('%Y-%m-%d')
                    )
                    
                    if df.empty:
                        logger.warning(f"⚠️ {stock.symbol}: No new data")
                        failed_count += 1
                        continue
                    
                    # Lưu vào database
                    new_records = 0
                    for idx, row in df.iterrows():
                        # Kiểm tra xem đã có record này chưa
                        existing = db.query(StockPrice).filter(
                            StockPrice.stock_id == stock.id,
                            StockPrice.date == row['date'].date()
                        ).first()
                        
                        if not existing:
                            price_record = StockPrice(
                                stock_id=stock.id,
                                date=row['date'].date(),
                                open=float(row['Open']),
                                high=float(row['High']),
                                low=float(row['Low']),
                                close=float(row['Close']),
                                volume=int(row['Volume']) if not pd.isna(row['Volume']) else 0
                            )
                            db.add(price_record)
                            new_records += 1
                    
                    db.commit()
                    total_new_records += new_records
                    
                    logger.info(f"✅ {stock.symbol}: Saved {new_records} new records")
                    success_count += 1
                    
                except Exception as e:
                    logger.error(f"❌ {stock.symbol}: Error - {str(e)}")
                    failed_count += 1
                    db.rollback()
                    continue
            
            result = {
                'success': success_count,
                'failed': failed_count,
                'total': len(stocks),
                'new_records': total_new_records
            }
            
            logger.info(f"📊 Daily price update complete: {success_count}/{len(stocks)} succeeded, {total_new_records} new records")
            return result
        
        except Exception as e:
            logger.error(f"❌ Error in fetch_and_save_stock_prices: {str(e)}")
            return {'success': 0, 'failed': 0, 'total': 0, 'new_records': 0}
    
    def daily_update_job(self):
        """
        Job chạy hàng ngày:
        1. Tải dữ liệu giá mới
        2. Tính toán technical indicators
        """
        logger.info("=" * 80)
        logger.info(f"🚀 DAILY UPDATE JOB STARTED - {datetime.now()}")
        logger.info("=" * 80)
        
        db = next(get_db())
        
        try:
            # Step 1: Fetch and save stock prices
            logger.info("📥 Step 1: Fetching stock prices...")
            price_result = self.fetch_and_save_stock_prices(db)
            logger.info(f"   → {price_result['new_records']} new price records saved")
            
            # Step 2: Calculate technical indicators
            logger.info("📊 Step 2: Calculating technical indicators...")
            processor = IndicatorsProcessor(db)
            indicator_result = processor.process_all_stocks(days=365)
            logger.info(f"   → {indicator_result['success']} stocks processed")
            
            # Summary
            logger.info("=" * 80)
            logger.info("✅ DAILY UPDATE JOB COMPLETED")
            logger.info(f"   Price Update: {price_result['success']}/{price_result['total']} succeeded")
            logger.info(f"   Indicators: {indicator_result['success']}/{indicator_result['total']} succeeded")
            logger.info(f"   New Records: {price_result['new_records']}")
            logger.info("=" * 80)
            
        except Exception as e:
            logger.error(f"❌ Error in daily_update_job: {str(e)}")
        finally:
            db.close()
    
    def start(self):
        """
        Khởi động scheduler
        - Chạy daily job vào 18:00 mỗi ngày (Thứ 2 - Thứ 6)
        - Có thể thêm job chạy vào cuối tuần để cập nhật dữ liệu bị miss
        """
        # Job chính: Chạy vào 18:00 từ Thứ 2 đến Thứ 6
        self.scheduler.add_job(
            self.daily_update_job,
            trigger=CronTrigger(
                day_of_week='mon-fri',  # Thứ 2 - Thứ 6
                hour=18,
                minute=0
            ),
            id='daily_price_update',
            name='Daily Stock Price & Indicators Update',
            replace_existing=True
        )
        
        # Job phụ: Chạy vào 10:00 Chủ nhật để cập nhật dữ liệu tuần trước (nếu bị miss)
        self.scheduler.add_job(
            self.daily_update_job,
            trigger=CronTrigger(
                day_of_week='sun',  # Chủ nhật
                hour=10,
                minute=0
            ),
            id='weekly_catchup',
            name='Weekly Catch-up Update',
            replace_existing=True
        )
        
        self.scheduler.start()
        logger.info("🎯 Scheduler started successfully")
        logger.info("   → Daily update: Mon-Fri at 18:00")
        logger.info("   → Weekly catch-up: Sunday at 10:00")
    
    def stop(self):
        """Dừng scheduler"""
        self.scheduler.shutdown()
        logger.info("🛑 Scheduler stopped")
    
    def run_now(self):
        """Chạy job ngay lập tức (để test hoặc manual trigger)"""
        logger.info("▶️ Running daily update job manually...")
        self.daily_update_job()
    
    def get_next_run_time(self) -> str:
        """Lấy thời gian chạy tiếp theo"""
        jobs = self.scheduler.get_jobs()
        if jobs:
            next_run = min([job.next_run_time for job in jobs if job.next_run_time])
            return next_run.strftime('%Y-%m-%d %H:%M:%S')
        return "No jobs scheduled"


# Global scheduler instance
scheduler_instance = None


def init_scheduler():
    """Khởi tạo và start scheduler"""
    global scheduler_instance
    if not SCHEDULER_AVAILABLE:
        logger.warning("⚠️ Scheduler not available (apscheduler not installed)")
        return None
    if scheduler_instance is None:
        scheduler_instance = DailyDataScheduler()
        scheduler_instance.start()
    return scheduler_instance


def get_scheduler():
    """Lấy scheduler instance (singleton)"""
    global scheduler_instance
    if scheduler_instance is None:
        return init_scheduler()
    return scheduler_instance


if __name__ == "__main__":
    # Test scheduler
    print("Testing Daily Data Scheduler...")
    print("-" * 80)
    
    scheduler = DailyDataScheduler()
    
    # Chạy ngay để test
    print("\n▶️ Running update job now (test mode)...\n")
    scheduler.run_now()
    
    print("\n" + "=" * 80)
    print("Test completed. To run scheduler in background:")
    print("  1. scheduler.start()  # Start background scheduler")
    print("  2. Keep script running")
    print("  3. Jobs will run automatically at scheduled times")
    print("=" * 80)
