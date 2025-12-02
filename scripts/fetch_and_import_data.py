"""
Script để fetch dữ liệu giao dịch và import vào Railway PostgreSQL
"""
import os
import sys
from datetime import datetime, timedelta

# Thêm path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Import collectors
from src.data_collection.trading_data import TradingDataCollector
from src.database.models import Stock, StockPrice

# ==========================================
# CẤU HÌNH
# ==========================================

# Railway PostgreSQL URL (public)
POSTGRES_URL = os.getenv("RAILWAY_DATABASE_URL", "")

if not POSTGRES_URL:
    print("❌ Chưa có RAILWAY_DATABASE_URL!")
    POSTGRES_URL = input("Paste DATABASE_URL (public): ").strip()

# Fix Railway URL format
if POSTGRES_URL.startswith("postgres://"):
    POSTGRES_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

print(f"📍 Target: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else 'unknown'}")

# ==========================================
# KẾT NỐI
# ==========================================

print("\n🔌 Connecting to PostgreSQL...")
try:
    engine = create_engine(POSTGRES_URL, echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()
    print("✅ Connected to PostgreSQL")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    sys.exit(1)

# ==========================================
# LẤY DANH SÁCH STOCKS
# ==========================================

print("\n📋 Getting stock list...")
stocks = session.query(Stock).filter(Stock.is_active == True).all()
print(f"✅ Found {len(stocks)} active stocks")

if len(stocks) == 0:
    print("❌ No stocks found. Please run migrate_to_postgres.py first.")
    sys.exit(1)

# ==========================================
# FETCH VÀ IMPORT DATA
# ==========================================

print("\n📥 Fetching trading data from VNDirect...")

collector = TradingDataCollector()

# Lấy data 1 năm
end_date = datetime.now()
start_date = end_date - timedelta(days=365)

start_str = start_date.strftime("%Y-%m-%d")
end_str = end_date.strftime("%Y-%m-%d")

total_imported = 0
failed_symbols = []

for stock in stocks:
    symbol = stock.symbol
    print(f"\n  📊 Fetching {symbol}...", end=" ")
    
    try:
        # Fetch từ VNDirect
        df = collector.get_detailed_trading_data(symbol, start_str, end_str)
        
        if df is None or len(df) == 0:
            print("⏭️ No data")
            failed_symbols.append(symbol)
            continue
        
        # Chuẩn bị data để insert
        records = []
        for _, row in df.iterrows():
            try:
                record = StockPrice(
                    stock_id=stock.id,
                    date=pd.to_datetime(row.get('date', row.name)).date() if 'date' in row else row.name.date(),
                    open=float(row.get('open', 0)),
                    high=float(row.get('high', 0)),
                    low=float(row.get('low', 0)),
                    close=float(row.get('close', 0)),
                    volume=float(row.get('volume', 0)),
                    source='vndirect'
                )
                records.append(record)
            except Exception as e:
                continue
        
        if records:
            # Xóa data cũ của symbol này
            session.execute(
                text("DELETE FROM stock_prices WHERE stock_id = :stock_id"),
                {"stock_id": stock.id}
            )
            
            # Insert data mới
            session.bulk_save_objects(records)
            session.commit()
            
            print(f"✅ {len(records)} rows")
            total_imported += len(records)
        else:
            print("⏭️ No valid records")
            failed_symbols.append(symbol)
            
    except Exception as e:
        print(f"❌ Error: {str(e)[:50]}")
        failed_symbols.append(symbol)
        session.rollback()

# ==========================================
# KẾT QUẢ
# ==========================================

print("\n" + "="*50)
print("📊 IMPORT SUMMARY")
print("="*50)
print(f"✅ Total imported: {total_imported} rows")
print(f"📈 Stocks with data: {len(stocks) - len(failed_symbols)}/{len(stocks)}")

if failed_symbols:
    print(f"⚠️ Failed symbols: {', '.join(failed_symbols[:10])}")
    if len(failed_symbols) > 10:
        print(f"   ... and {len(failed_symbols) - 10} more")

# Verify
print("\n🔍 Verifying...")
count = session.execute(text("SELECT COUNT(*) FROM stock_prices")).scalar()
print(f"📈 Total stock_prices in DB: {count}")

session.close()
print("\n🎉 Done!")
