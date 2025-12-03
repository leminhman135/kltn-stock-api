# Cron Worker for Railway
# This runs scheduled tasks separately from the main API

import os
import sys
import time
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_daily_update():
    """Run daily stock data update"""
    try:
        from src.scheduler.daily_scheduler import DailyScheduler
        
        logger.info("🚀 Starting daily update...")
        scheduler = DailyScheduler()
        scheduler.run_daily_update()
        logger.info("✅ Daily update completed!")
        
    except Exception as e:
        logger.error(f"❌ Daily update failed: {e}")

def run_scheduler():
    """Run APScheduler for cron jobs"""
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
        
        scheduler = BlockingScheduler()
        
        # Mon-Fri at 18:00 Vietnam time (11:00 UTC)
        scheduler.add_job(
            run_daily_update,
            CronTrigger(day_of_week='mon-fri', hour=11, minute=0),
            id='daily_update_weekday',
            name='Daily Stock Update (Weekday)'
        )
        
        # Sunday at 10:00 Vietnam time (03:00 UTC)
        scheduler.add_job(
            run_daily_update,
            CronTrigger(day_of_week='sun', hour=3, minute=0),
            id='daily_update_sunday',
            name='Daily Stock Update (Sunday)'
        )
        
        logger.info("📅 Scheduler started with jobs:")
        for job in scheduler.get_jobs():
            logger.info(f"  - {job.name}: {job.trigger}")
        
        scheduler.start()
        
    except ImportError as e:
        logger.warning(f"⚠️ APScheduler not available: {e}")
        logger.info("Running in simple loop mode...")
        
        # Fallback: simple loop checking time
        while True:
            now = datetime.utcnow()
            # Check if it's 11:00 UTC (18:00 VN) on weekday
            if now.hour == 11 and now.minute == 0 and now.weekday() < 5:
                run_daily_update()
            # Check if it's 03:00 UTC (10:00 VN) on Sunday
            elif now.hour == 3 and now.minute == 0 and now.weekday() == 6:
                run_daily_update()
            
            time.sleep(60)  # Check every minute

if __name__ == "__main__":
    logger.info("🔄 Cron Worker starting...")
    
    # Run once on startup
    if os.getenv("RUN_ON_STARTUP", "false").lower() == "true":
        run_daily_update()
    
    # Start scheduler
    run_scheduler()
