"""
Script để migrate data từ SQLite local sang PostgreSQL (Railway)
"""
import os
import sqlite3
import sys

# Thêm path để import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import pandas as pd

# ==========================================
# CẤU HÌNH - THAY ĐỔI THEO DATABASE CỦA BẠN
# ==========================================

# SQLite source (local)
SQLITE_PATH = "kltn_stocks.db"

# PostgreSQL target (Railway) - THAY THẾ BẰNG URL CỦA BẠN
# Lấy từ Railway Dashboard > PostgreSQL > Variables > DATABASE_URL
POSTGRES_URL = os.getenv("RAILWAY_DATABASE_URL", "")

if not POSTGRES_URL:
    print("❌ Chưa có RAILWAY_DATABASE_URL!")
    print("Cách 1: Set environment variable:")
    print('  $env:RAILWAY_DATABASE_URL = "postgresql://postgres:xxx@xxx.railway.app:5432/railway"')
    print("\nCách 2: Nhập trực tiếp:")
    POSTGRES_URL = input("Paste DATABASE_URL từ Railway: ").strip()

if not POSTGRES_URL:
    print("❌ Không có DATABASE_URL. Thoát.")
    sys.exit(1)

# Fix Railway URL format
if POSTGRES_URL.startswith("postgres://"):
    POSTGRES_URL = POSTGRES_URL.replace("postgres://", "postgresql://", 1)

print(f"📍 Target: {POSTGRES_URL.split('@')[1] if '@' in POSTGRES_URL else 'unknown'}")

# ==========================================
# KẾT NỐI DATABASES
# ==========================================

print("\n🔌 Connecting to databases...")

# SQLite
sqlite_conn = sqlite3.connect(SQLITE_PATH)
print(f"✅ Connected to SQLite: {SQLITE_PATH}")

# PostgreSQL
try:
    pg_engine = create_engine(POSTGRES_URL, echo=False)
    pg_conn = pg_engine.connect()
    print("✅ Connected to PostgreSQL")
except Exception as e:
    print(f"❌ PostgreSQL connection failed: {e}")
    sys.exit(1)

# ==========================================
# LẤY DANH SÁCH TABLES TỪ SQLITE
# ==========================================

cursor = sqlite_conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
tables = [row[0] for row in cursor.fetchall()]

print(f"\n📋 Found {len(tables)} tables: {', '.join(tables)}")

# ==========================================
# TẠO TABLES VÀ IMPORT DATA
# ==========================================

from src.database.connection import Base
from src.database.models import Stock, StockPrice, Prediction, ModelMetrics

# Tạo tables trong PostgreSQL
print("\n🔨 Creating tables in PostgreSQL...")
Base.metadata.create_all(pg_engine)
print("✅ Tables created")

# Import từng table theo thứ tự (stocks trước, rồi stock_prices)
print("\n📤 Importing data...")

# Thứ tự import quan trọng do foreign key
import_order = ['stocks', 'stock_prices', 'news_articles', 'sentiment_analysis', 
                'technical_indicators', 'predictions', 'model_metrics']

for table_name in import_order:
    if table_name not in tables:
        continue
    try:
        # Đọc data từ SQLite
        df = pd.read_sql(f"SELECT * FROM {table_name}", sqlite_conn)
        
        if len(df) == 0:
            print(f"  ⏭️  {table_name}: Empty, skipping")
            continue
        
        # Fix boolean columns cho PostgreSQL
        if 'is_active' in df.columns:
            df['is_active'] = df['is_active'].astype(bool)
        
        # Clear existing data trước khi import
        try:
            pg_conn.execute(text(f"DELETE FROM {table_name}"))
            pg_conn.commit()
        except:
            pass
        
        # Import vào PostgreSQL - chia nhỏ batch để tránh lỗi
        batch_size = 100
        total_rows = len(df)
        imported = 0
        
        for i in range(0, total_rows, batch_size):
            batch = df.iloc[i:i+batch_size]
            batch.to_sql(table_name, pg_engine, if_exists='append', index=False)
            imported += len(batch)
        
        print(f"  ✅ {table_name}: {imported} rows imported")
        
    except Exception as e:
        print(f"  ❌ {table_name}: Error - {e}")

# ==========================================
# VERIFY
# ==========================================

print("\n🔍 Verifying import...")

Session = sessionmaker(bind=pg_engine)
session = Session()

try:
    # Check stocks
    stock_count = session.execute(text("SELECT COUNT(*) FROM stocks")).scalar()
    print(f"  📊 Stocks: {stock_count}")
    
    # Check prices
    price_count = session.execute(text("SELECT COUNT(*) FROM stock_prices")).scalar()
    print(f"  📈 Stock Prices: {price_count}")
    
    # Check predictions
    try:
        pred_count = session.execute(text("SELECT COUNT(*) FROM predictions")).scalar()
        print(f"  🔮 Predictions: {pred_count}")
    except:
        print("  🔮 Predictions: Table not found")
    
except Exception as e:
    print(f"  ⚠️ Verify error: {e}")

# ==========================================
# DONE
# ==========================================

sqlite_conn.close()
pg_conn.close()
session.close()

print("\n" + "="*50)
print("🎉 Migration completed!")
print("="*50)
print("\nBạn có thể test API Railway bằng cách gọi:")
print("  https://YOUR-RAILWAY-URL.up.railway.app/api/health")
print("  https://YOUR-RAILWAY-URL.up.railway.app/api/stocks")
