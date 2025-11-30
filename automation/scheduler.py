"""
Automation Scheduler - Tự động hóa thu thập dữ liệu và training
Sử dụng Python Schedule thay vì Apache Airflow (đơn giản hơn)
"""

import schedule
import time
import logging
from datetime import datetime, timedelta
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_collection import VNDirectAPI, YahooFinanceAPI
from src.data_processing import DataTransformer
from src.features.technical_indicators import TechnicalIndicators
from src.features.sentiment_analysis import FinBERTSentimentAnalyzer
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation/logs/scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ==================== CONFIGURATION ====================
STOCKS = ['VNM', 'VIC', 'HPG', 'VCB', 'FPT', 'VHM', 'MSN', 'CTG', 'TCB', 'BID']
DATA_DIR = 'data'
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# Tạo thư mục nếu chưa có
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs('automation/logs', exist_ok=True)


# ==================== TASK 1: DATA COLLECTION ====================
def collect_stock_data():
    """
    ETL Pipeline - Extract: Thu thập dữ liệu từ API
    Chạy mỗi ngày lúc 18:00 (sau khi thị trường đóng cửa)
    """
    logger.info("="*60)
    logger.info("🚀 STARTING DATA COLLECTION")
    logger.info("="*60)
    
    vndirect = VNDirectAPI()
    yahoo = YahooFinanceAPI()
    
    success_count = 0
    fail_count = 0
    
    for symbol in STOCKS:
        try:
            logger.info(f"📊 Collecting data for {symbol}...")
            
            # Get data từ VNDirect (ưu tiên)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            
            df = vndirect.get_stock_price(f'{symbol}.VN', start_date, end_date)
            
            # Fallback to Yahoo Finance nếu VNDirect fail
            if df.empty:
                logger.warning(f"⚠️ VNDirect failed for {symbol}, trying Yahoo Finance...")
                df = yahoo.get_stock_data(f'{symbol}.VN', start_date, end_date)
            
            if not df.empty:
                # Lưu raw data
                raw_file = os.path.join(RAW_DIR, f'{symbol}_raw_{datetime.now().strftime("%Y%m%d")}.csv')
                df.to_csv(raw_file, index=False)
                logger.info(f"✅ Saved {len(df)} records to {raw_file}")
                success_count += 1
            else:
                logger.error(f"❌ No data collected for {symbol}")
                fail_count += 1
                
        except Exception as e:
            logger.error(f"❌ Error collecting {symbol}: {e}")
            fail_count += 1
    
    logger.info(f"✅ Collection complete: {success_count} success, {fail_count} failed")
    return success_count, fail_count


# ==================== TASK 2: DATA PROCESSING (ETL - Transform) ====================
def process_and_clean_data():
    """
    ETL Pipeline - Transform: Làm sạch và chuẩn hóa dữ liệu
    Chạy sau khi thu thập xong (18:30)
    """
    logger.info("="*60)
    logger.info("🔄 STARTING DATA PROCESSING & CLEANING")
    logger.info("="*60)
    
    transformer = DataTransformer()
    success_count = 0
    
    # Tìm file raw mới nhất
    today = datetime.now().strftime("%Y%m%d")
    
    for symbol in STOCKS:
        try:
            raw_file = os.path.join(RAW_DIR, f'{symbol}_raw_{today}.csv')
            
            if not os.path.exists(raw_file):
                logger.warning(f"⚠️ Raw file not found for {symbol}")
                continue
            
            # Load raw data
            df = pd.read_csv(raw_file)
            logger.info(f"📖 Processing {symbol}: {len(df)} records")
            
            # Clean data
            df_clean = transformer.clean_price_data(df)
            
            # Handle missing values
            df_clean = transformer.handle_missing_values(df_clean)
            
            # Lưu processed data
            processed_file = os.path.join(PROCESSED_DIR, f'{symbol}_processed.csv')
            df_clean.to_csv(processed_file, index=False)
            
            logger.info(f"✅ Processed {symbol}: {len(df_clean)} records saved to {processed_file}")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error processing {symbol}: {e}")
    
    logger.info(f"✅ Processing complete: {success_count} stocks processed")
    return success_count


# ==================== TASK 3: FEATURE ENGINEERING ====================
def calculate_technical_indicators():
    """
    Tính toán các chỉ báo kỹ thuật
    Chạy sau khi processing xong (19:00)
    """
    logger.info("="*60)
    logger.info("📈 CALCULATING TECHNICAL INDICATORS")
    logger.info("="*60)
    
    tech_indicators = TechnicalIndicators()
    success_count = 0
    
    for symbol in STOCKS:
        try:
            processed_file = os.path.join(PROCESSED_DIR, f'{symbol}_processed.csv')
            
            if not os.path.exists(processed_file):
                logger.warning(f"⚠️ Processed file not found for {symbol}")
                continue
            
            # Load processed data
            df = pd.read_csv(processed_file)
            logger.info(f"📊 Calculating indicators for {symbol}...")
            
            # Calculate all indicators
            df_with_indicators = tech_indicators.calculate_all(df)
            
            # Lưu với indicators
            indicators_file = os.path.join(PROCESSED_DIR, f'{symbol}_with_indicators.csv')
            df_with_indicators.to_csv(indicators_file, index=False)
            
            logger.info(f"✅ Calculated indicators for {symbol}: {len(df_with_indicators.columns)} columns")
            success_count += 1
            
        except Exception as e:
            logger.error(f"❌ Error calculating indicators for {symbol}: {e}")
    
    logger.info(f"✅ Indicator calculation complete: {success_count} stocks")
    return success_count


# ==================== TASK 4: SENTIMENT ANALYSIS ====================
def analyze_sentiment():
    """
    Phân tích cảm xúc từ tin tức (nếu có data)
    Chạy mỗi ngày lúc 20:00
    """
    logger.info("="*60)
    logger.info("💭 ANALYZING SENTIMENT")
    logger.info("="*60)
    
    try:
        sentiment_analyzer = FinBERTSentimentAnalyzer()
        
        # TODO: Implement news scraping
        # Hiện tại chỉ log, sẽ implement sau
        logger.info("⚠️ Sentiment analysis requires news data (to be implemented)")
        logger.info("📝 Will use web scraping in next phase")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Sentiment analysis error: {e}")
        return 0


# ==================== TASK 5: MODEL TRAINING ====================
def train_models_weekly():
    """
    Train lại models mỗi tuần
    Chạy vào Chủ Nhật lúc 02:00
    """
    logger.info("="*60)
    logger.info("🤖 STARTING WEEKLY MODEL TRAINING")
    logger.info("="*60)
    
    try:
        # Import models
        from src.models.arima_model import ARIMAModel
        from src.models.prophet_model import ProphetModel
        from src.models.lstm_gru_models import LSTMModel
        
        success_count = 0
        
        for symbol in STOCKS[:3]:  # Train top 3 stocks để tiết kiệm thời gian
            try:
                logger.info(f"🎓 Training models for {symbol}...")
                
                # Load data với indicators
                indicators_file = os.path.join(PROCESSED_DIR, f'{symbol}_with_indicators.csv')
                
                if not os.path.exists(indicators_file):
                    logger.warning(f"⚠️ Data not found for {symbol}")
                    continue
                
                df = pd.read_csv(indicators_file)
                
                # Train ARIMA
                logger.info(f"  Training ARIMA for {symbol}...")
                arima = ARIMAModel()
                arima_fit = arima.train(df['close'])
                
                # Train Prophet
                logger.info(f"  Training Prophet for {symbol}...")
                prophet = ProphetModel()
                prophet_model = prophet.train(df[['date', 'close']].rename(columns={'date': 'ds', 'close': 'y'}))
                
                logger.info(f"✅ Models trained for {symbol}")
                success_count += 1
                
            except Exception as e:
                logger.error(f"❌ Error training models for {symbol}: {e}")
        
        logger.info(f"✅ Training complete: {success_count} stocks")
        return success_count
        
    except Exception as e:
        logger.error(f"❌ Model training error: {e}")
        return 0


# ==================== TASK 6: DATA BACKUP ====================
def backup_data():
    """
    Backup dữ liệu mỗi tuần
    Chạy vào Chủ Nhật lúc 03:00
    """
    logger.info("="*60)
    logger.info("💾 BACKING UP DATA")
    logger.info("="*60)
    
    try:
        backup_dir = os.path.join(DATA_DIR, 'backups', datetime.now().strftime('%Y%m%d'))
        os.makedirs(backup_dir, exist_ok=True)
        
        # Copy processed data
        import shutil
        for file in os.listdir(PROCESSED_DIR):
            if file.endswith('.csv'):
                src = os.path.join(PROCESSED_DIR, file)
                dst = os.path.join(backup_dir, file)
                shutil.copy2(src, dst)
        
        logger.info(f"✅ Backup complete: {backup_dir}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Backup error: {e}")
        return False


# ==================== TASK 7: CLEANUP OLD FILES ====================
def cleanup_old_files():
    """
    Xóa file raw cũ hơn 30 ngày
    Chạy mỗi tuần vào Chủ Nhật lúc 04:00
    """
    logger.info("="*60)
    logger.info("🗑️  CLEANING UP OLD FILES")
    logger.info("="*60)
    
    try:
        cutoff_date = datetime.now() - timedelta(days=30)
        deleted_count = 0
        
        for file in os.listdir(RAW_DIR):
            if file.endswith('.csv'):
                file_path = os.path.join(RAW_DIR, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if file_time < cutoff_date:
                    os.remove(file_path)
                    logger.info(f"🗑️  Deleted old file: {file}")
                    deleted_count += 1
        
        logger.info(f"✅ Cleanup complete: {deleted_count} files deleted")
        return deleted_count
        
    except Exception as e:
        logger.error(f"❌ Cleanup error: {e}")
        return 0


# ==================== SCHEDULER SETUP ====================
def setup_schedule():
    """
    Thiết lập lịch trình chạy tự động
    """
    logger.info("🔧 Setting up schedule...")
    
    # DAILY TASKS
    schedule.every().day.at("18:00").do(collect_stock_data)           # Sau giờ đóng cửa
    schedule.every().day.at("18:30").do(process_and_clean_data)       # ETL Transform
    schedule.every().day.at("19:00").do(calculate_technical_indicators)
    schedule.every().day.at("20:00").do(analyze_sentiment)
    
    # WEEKLY TASKS (Sunday)
    schedule.every().sunday.at("02:00").do(train_models_weekly)
    schedule.every().sunday.at("03:00").do(backup_data)
    schedule.every().sunday.at("04:00").do(cleanup_old_files)
    
    logger.info("✅ Schedule configured:")
    logger.info("  📊 Daily 18:00 - Data Collection")
    logger.info("  🔄 Daily 18:30 - Data Processing")
    logger.info("  📈 Daily 19:00 - Technical Indicators")
    logger.info("  💭 Daily 20:00 - Sentiment Analysis")
    logger.info("  🤖 Sunday 02:00 - Model Training")
    logger.info("  💾 Sunday 03:00 - Data Backup")
    logger.info("  🗑️  Sunday 04:00 - Cleanup")


# ==================== MANUAL RUN ====================
def run_all_tasks_now():
    """
    Chạy tất cả tasks ngay lập tức (for testing)
    """
    logger.info("="*60)
    logger.info("🚀 RUNNING ALL TASKS MANUALLY")
    logger.info("="*60)
    
    collect_stock_data()
    process_and_clean_data()
    calculate_technical_indicators()
    analyze_sentiment()
    
    logger.info("="*60)
    logger.info("✅ ALL TASKS COMPLETED")
    logger.info("="*60)


# ==================== MAIN ====================
def main():
    """
    Main scheduler loop
    """
    logger.info("="*60)
    logger.info("🤖 AUTOMATION SCHEDULER STARTED")
    logger.info("="*60)
    
    # Setup schedule
    setup_schedule()
    
    # Run immediately on start (optional)
    # run_all_tasks_now()
    
    # Main loop
    logger.info("⏰ Scheduler is running... (Press Ctrl+C to stop)")
    
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
            
    except KeyboardInterrupt:
        logger.info("\n⏹️  Scheduler stopped by user")
    except Exception as e:
        logger.error(f"❌ Scheduler error: {e}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Stock Prediction Automation Scheduler')
    parser.add_argument('--run-now', action='store_true', help='Run all tasks immediately')
    parser.add_argument('--collect-only', action='store_true', help='Run data collection only')
    parser.add_argument('--process-only', action='store_true', help='Run data processing only')
    parser.add_argument('--train-only', action='store_true', help='Run model training only')
    
    args = parser.parse_args()
    
    if args.run_now:
        run_all_tasks_now()
    elif args.collect_only:
        collect_stock_data()
    elif args.process_only:
        process_and_clean_data()
    elif args.train_only:
        train_models_weekly()
    else:
        main()  # Run scheduler
