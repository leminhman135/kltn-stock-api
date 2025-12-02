# 📊 Quy trình Lưu trữ Dữ liệu Thô vào SQL

## 🎯 Vấn đề

Khi **trích xuất dữ liệu thô từ API**, cần đảm bảo dữ liệu được **lưu trữ vào SQL Database** trước khi xử lý (transform). Điều này đảm bảo:

1. ✅ **Data persistence** - Dữ liệu không bị mất nếu quá trình ETL lỗi
2. ✅ **Audit trail** - Có thể tracking dữ liệu gốc từ API
3. ✅ **Reprocessing** - Có thể re-run ETL mà không cần gọi API lại
4. ✅ **Data lineage** - Biết được dữ liệu đến từ đâu, khi nào

---

## 🏗️ Kiến trúc Mới

### **WORKFLOW CŨ** (Không lưu raw data):
```
API → DataFrame (memory) → Transform → Load to stock_prices
       ↑ MẤT DATA NẾU ETL LỖI
```

### **WORKFLOW MỚI** (Lưu raw data vào SQL):
```
API → raw_stock_data table (SQL) → Extract → Transform → Load to stock_prices
      ↑ PERSISTENT, AUDITABLE, REPROCESSABLE
```

---

## 📁 Cấu trúc Files Mới

```
D:\KLTN\
├── src/
│   └── etl/
│       ├── raw_data_collector.py  ✨ MỚI - Thu thập raw data từ API
│       ├── extract.py              🔧 CẬP NHẬT - Thêm extract_raw_price_data()
│       ├── transform.py            ⚪ GIỮ NGUYÊN
│       ├── load.py                 ⚪ GIỮ NGUYÊN
│       └── pipeline.py             ⚪ GIỮ NGUYÊN
│
└── scripts/
    └── collect_raw_data.py        ✨ MỚI - Script thu thập raw data
```

---

## 📊 Database Schema

### **TABLE: raw_stock_data** (Dữ liệu thô từ API)

```sql
CREATE TABLE raw_stock_data (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Raw OHLCV data
    open FLOAT,
    high FLOAT,
    low FLOAT,
    close FLOAT,
    volume BIGINT,
    
    -- Additional raw fields from API
    value FLOAT,              -- Giá trị giao dịch
    change_percent FLOAT,     -- % thay đổi
    change_point FLOAT,       -- Điểm thay đổi
    
    -- Metadata
    source VARCHAR(50),       -- 'vndirect', 'ssi', etc.
    raw_json TEXT,            -- JSON gốc từ API (audit trail)
    collected_at TIMESTAMP DEFAULT NOW(),
    processed BOOLEAN DEFAULT FALSE,  -- Đã xử lý bởi ETL chưa
    
    UNIQUE(symbol, date, source)  -- Tránh duplicate
);

-- Indexes for performance
CREATE INDEX idx_raw_stock_symbol ON raw_stock_data(symbol);
CREATE INDEX idx_raw_stock_date ON raw_stock_data(date);
CREATE INDEX idx_raw_stock_processed ON raw_stock_data(processed);
```

### **So sánh với stock_prices table** (Dữ liệu đã xử lý)

| Field | raw_stock_data | stock_prices | Ghi chú |
|-------|---------------|--------------|---------|
| symbol | ✅ VARCHAR(20) | ❌ (dùng stock_id FK) | Raw giữ symbol trực tiếp |
| date | ✅ DATE | ✅ DATE | Giống nhau |
| OHLCV | ✅ | ✅ | Giống nhau |
| value | ✅ | ❌ | Raw có thêm field này |
| change_percent | ✅ | ❌ | Raw có thêm |
| source | ✅ | ❌ | Tracking nguồn API |
| raw_json | ✅ | ❌ | Audit trail - JSON gốc |
| processed | ✅ | ❌ | Flag đã xử lý |

---

## 🚀 Cách Sử dụng

### **BƯỚC 1: Thu thập Raw Data từ API**

```bash
# Kích hoạt virtual environment
& D:\KLTN\venv\Scripts\Activate.ps1

# Thu thập cho VNM (30 ngày gần đây)
python scripts/collect_raw_data.py --symbols VNM --days 30

# Thu thập cho nhiều mã
python scripts/collect_raw_data.py --symbols VNM HPG VCB FPT --days 7

# Thu thập TẤT CẢ mã trong database
python scripts/collect_raw_data.py --all --days 30
```

**Output**:
```
================================================================================
📡 RAW DATA COLLECTION FROM API TO SQL
================================================================================

📊 Collecting data for: VNM, HPG, VCB
📅 Date range: Last 30 days

📊 Processing VNM...
📡 Collecting raw data for VNM from API...
✅ API returned 20 records
💾 Saved raw data: 20 inserted, 0 updated, 0 failed

📊 Processing HPG...
📡 Collecting raw data for HPG from API...
✅ API returned 20 records
💾 Saved raw data: 20 inserted, 0 updated, 0 failed

================================================================================
✅ BATCH COLLECTION COMPLETED
   Symbols processed: 3
   Total collected: 60
   Total inserted: 60
   Total updated: 0
   Total failed: 0
================================================================================

✅ Raw data collection completed!
💾 Data saved to: raw_stock_data table in PostgreSQL
🔄 Next step: Run ETL pipeline to transform and load data
```

---

### **BƯỚC 2: Chạy ETL Pipeline để Transform**

```bash
# ETL sẽ đọc từ raw_stock_data, transform, và load vào stock_prices
python scripts/test_etl_pipeline.py
```

**Hoặc dùng Python code**:

```python
from src.etl.extract import DataExtractor
from src.etl.transform import DataTransformer
from src.etl.load import DataLoader

# Extract raw data
extractor = DataExtractor()
df_raw = extractor.extract_raw_price_data(
    symbol='VNM',
    unprocessed_only=True  # Chỉ lấy dữ liệu chưa xử lý
)

# Transform
transformer = DataTransformer()
df_clean = transformer.transform_price_data(df_raw)

# Load to stock_prices
loader = DataLoader()
stats = loader.load_price_data(df_clean)

print(f"Loaded: {stats}")

# Đánh dấu raw data đã xử lý
from src.etl.raw_data_collector import RawDataCollector
collector = RawDataCollector()
collector.mark_as_processed(df_raw['id'].tolist())
```

---

## 📊 Workflow Chi tiết

```
┌────────────────────────────────────────────────────────────────┐
│ PHASE 1: COLLECT RAW DATA FROM API                            │
└────────────────────────────────────────────────────────────────┘

    📡 VNDirect API
    ├─ GET /historical_quotes
    │  └─ symbol=VNM, from=2024-11-01, to=2024-12-02
    │
    ▼
    {
      "data": [
        {
          "date": "2024-11-03",
          "open": 57.7,
          "high": 58.3,
          "low": 57.3,
          "close": 57.3,
          "volume": 2642700,
          "value": 153200000000,
          "changePercent": -0.52,
          "change": -0.3
        },
        ...
      ]
    }
    │
    ▼
    ┌─────────────────────────────────────────────────────────┐
    │ raw_stock_data table (PostgreSQL)                       │
    ├─────────────────────────────────────────────────────────┤
    │ id | symbol | date       | open | high | low | close   │
    │ 1  | VNM    | 2024-11-03 | 57.7 | 58.3 | 57.3 | 57.3  │
    │ 2  | VNM    | 2024-11-04 | 57.6 | 57.9 | 56.3 | 57.3  │
    │ ...                                                      │
    ├─────────────────────────────────────────────────────────┤
    │ source: 'vndirect'                                       │
    │ raw_json: '{...}' ← JSON gốc từ API                     │
    │ processed: FALSE ← Chưa xử lý                           │
    │ collected_at: 2024-12-02 21:30:00                       │
    └─────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PHASE 2: ETL PIPELINE EXTRACT                                 │
└────────────────────────────────────────────────────────────────┘

    SELECT * FROM raw_stock_data
    WHERE processed = FALSE
      AND symbol = 'VNM'
    ORDER BY date
    │
    ▼
    DataFrame (in memory)
    ┌──────────────────────────────────────┐
    │ symbol | date       | open  | close │
    │ VNM    | 2024-11-03 | 57.7  | 57.3  │
    │ VNM    | 2024-11-04 | 57.6  | 57.3  │
    └──────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PHASE 3: TRANSFORM                                            │
└────────────────────────────────────────────────────────────────┘

    • Remove duplicates
    • Handle missing values
    • Validate OHLC relationships
    • Normalize data types
    │
    ▼
    Clean DataFrame

┌────────────────────────────────────────────────────────────────┐
│ PHASE 4: LOAD TO stock_prices                                 │
└────────────────────────────────────────────────────────────────┘

    INSERT INTO stock_prices (...)
    VALUES (...)
    ON CONFLICT DO UPDATE
    │
    ▼
    ┌─────────────────────────────────────────────────────────┐
    │ stock_prices table (Processed data)                     │
    ├─────────────────────────────────────────────────────────┤
    │ id | stock_id | date       | open | high | low | close│
    │ 1  | 1        | 2024-11-03 | 57.7 | 58.3 | 57.3| 57.3 │
    │ 2  | 1        | 2024-11-04 | 57.6 | 57.9 | 56.3| 57.3 │
    └─────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────┐
│ PHASE 5: MARK AS PROCESSED                                    │
└────────────────────────────────────────────────────────────────┘

    UPDATE raw_stock_data
    SET processed = TRUE
    WHERE id IN (1, 2, ...)
    │
    ▼
    ✅ Raw data marked as processed
```

---

## 🔍 Query Examples

### **1. Xem raw data chưa xử lý**

```sql
SELECT * FROM raw_stock_data
WHERE processed = FALSE
ORDER BY collected_at DESC
LIMIT 10;
```

### **2. Xem raw data cho VNM**

```sql
SELECT symbol, date, open, high, low, close, volume, source, collected_at
FROM raw_stock_data
WHERE symbol = 'VNM'
ORDER BY date DESC
LIMIT 20;
```

### **3. So sánh raw vs processed data**

```sql
SELECT 
    r.symbol,
    r.date,
    r.close AS raw_close,
    p.close AS processed_close,
    r.source,
    r.collected_at
FROM raw_stock_data r
LEFT JOIN stock_prices p ON (
    r.symbol = (SELECT symbol FROM stocks WHERE id = p.stock_id)
    AND r.date = p.date
)
WHERE r.symbol = 'VNM'
ORDER BY r.date DESC
LIMIT 10;
```

### **4. Audit trail - Xem JSON gốc từ API**

```sql
SELECT symbol, date, raw_json
FROM raw_stock_data
WHERE symbol = 'VNM' AND date = '2024-11-03';
```

**Output**:
```json
{
  "date": "2024-11-03",
  "open": 57.7,
  "high": 58.3,
  "low": 57.3,
  "close": 57.3,
  "volume": 2642700,
  "value": 153200000000,
  "changePercent": -0.52,
  "change": -0.3,
  "adOpen": 57700,
  "adHigh": 58300,
  ...
}
```

---

## 🎓 Lợi ích

| Lợi ích | Mô tả | Ví dụ |
|---------|-------|-------|
| **Data Persistence** | Dữ liệu từ API được lưu vĩnh viễn | Nếu ETL lỗi, không mất data |
| **Audit Trail** | Có JSON gốc để kiểm tra | Debug khi giá sai |
| **Reprocessing** | Re-run ETL không cần gọi API lại | Tiết kiệm API quota |
| **Data Lineage** | Biết data từ đâu, khi nào | Tracking nguồn gốc |
| **Version Control** | Có thể lưu nhiều version từ nhiều source | VNDirect vs SSI |
| **Performance** | ETL chạy nhanh hơn (đọc từ DB thay vì API) | < 1s thay vì 5-10s |

---

## 📅 Scheduled Collection

### **Tự động thu thập mỗi ngày**

```python
# File: src/scheduler/daily_scheduler.py

from apscheduler.schedulers.background import BackgroundScheduler
from src.etl.raw_data_collector import RawDataCollector

def daily_collect_raw_data():
    """Thu thập raw data mỗi ngày lúc 18:00"""
    collector = RawDataCollector()
    try:
        stats = collector.collect_all_stocks(days=1)
        print(f"✅ Collected raw data: {stats['total_collected']} records")
    finally:
        collector.close()

# Schedule
scheduler = BackgroundScheduler()
scheduler.add_job(
    daily_collect_raw_data,
    trigger='cron',
    hour=18,
    minute=0,
    day_of_week='mon-fri'
)
scheduler.start()
```

---

## 🐛 Troubleshooting

### **Lỗi: Table 'raw_stock_data' does not exist**

```bash
# Solution: Table sẽ tự động tạo khi chạy RawDataCollector lần đầu
python scripts/collect_raw_data.py --symbols VNM --days 1
```

### **Lỗi: Duplicate key violation**

```bash
# Lỗi này xảy ra khi cố insert duplicate (symbol, date, source)
# Solution: Code đã handle bằng UPSERT (INSERT ... ON CONFLICT DO UPDATE)
```

### **Lỗi: API rate limit exceeded**

```bash
# Solution: Thu thập từng mã một, hoặc giảm số ngày
python scripts/collect_raw_data.py --symbols VNM --days 7  # Thay vì 30
```

---

## 📊 Statistics & Monitoring

### **Check raw data status**

```python
from src.etl.raw_data_collector import RawDataCollector

collector = RawDataCollector()

# Get unprocessed count
df = collector.get_unprocessed_data()
print(f"Unprocessed records: {len(df)}")

# Get by symbol
df_vnm = collector.get_unprocessed_data(symbol='VNM')
print(f"VNM unprocessed: {len(df_vnm)}")
```

### **SQL Query**

```sql
-- Raw data statistics
SELECT 
    source,
    COUNT(*) as total_records,
    COUNT(*) FILTER (WHERE processed = TRUE) as processed,
    COUNT(*) FILTER (WHERE processed = FALSE) as unprocessed,
    MIN(date) as oldest_date,
    MAX(date) as newest_date
FROM raw_stock_data
GROUP BY source;
```

---

## 🎯 Tóm tắt Workflow

```
1. COLLECT RAW DATA (scripts/collect_raw_data.py)
   ↓
   📡 API → raw_stock_data table (SQL)
   
2. EXTRACT (src/etl/extract.py)
   ↓
   📥 SELECT * FROM raw_stock_data WHERE processed = FALSE
   
3. TRANSFORM (src/etl/transform.py)
   ↓
   🔄 Clean, validate, normalize
   
4. LOAD (src/etl/load.py)
   ↓
   💾 INSERT INTO stock_prices
   
5. MARK PROCESSED
   ↓
   ✅ UPDATE raw_stock_data SET processed = TRUE
```

---

**Tác giả**: KLTN Stock Prediction System  
**Version**: 2.0  
**Last Updated**: December 2, 2024  
**Feature**: Raw Data Storage in SQL
