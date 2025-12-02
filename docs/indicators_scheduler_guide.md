# Module Đặc trưng Kỹ thuật & Scheduler

## 📋 Tổng quan

Module này bao gồm:
1. **Indicators Processor**: Tính toán các chỉ báo kỹ thuật từ dữ liệu giá trong Database
2. **Daily Scheduler**: Tự động cập nhật dữ liệu giá và tính indicators hàng ngày

## 🎯 Chức năng

### 1. Technical Indicators Processor

**File**: `src/features/indicators_processor.py`

**Các chỉ báo được tính**:
- **Moving Averages**: SMA (20, 50, 200), EMA (12, 26)
- **RSI**: Relative Strength Index (14)
- **MACD**: Moving Average Convergence Divergence (12, 26, 9)
- **Bollinger Bands**: (20, 2)
- **Stochastic Oscillator**: %K, %D
- **ATR**: Average True Range (14)
- **OBV**: On-Balance Volume
- **ADX**: Average Directional Index
- **CCI**: Commodity Channel Index
- **Williams %R**

**Quy trình**:
```
1. Đọc dữ liệu giá từ Database (bảng stock_prices)
2. Tính toán các chỉ báo kỹ thuật
3. Lưu kết quả vào Database (bảng technical_indicators)
```

### 2. Daily Scheduler

**File**: `src/scheduler/daily_scheduler.py`

**Lịch chạy**:
- **Thứ 2 - Thứ 6**: 18:00 (sau khi thị trường đóng cửa)
- **Chủ nhật**: 10:00 (catch-up dữ liệu tuần trước)

**Nhiệm vụ tự động**:
1. Fetch dữ liệu giá mới nhất từ VNDirect API
2. Lưu vào Database
3. Tính toán Technical Indicators
4. Cập nhật Database

## 🚀 Sử dụng

### 1. API Endpoints

#### Start Scheduler
```bash
POST /api/scheduler/start
```
Khởi động background scheduler

#### Trigger Manual Update
```bash
POST /api/scheduler/run-now
```
Chạy update job ngay lập tức

#### Check Scheduler Status
```bash
GET /api/scheduler/status
```

#### Calculate Indicators (All Stocks)
```bash
POST /api/indicators/calculate?days=365
```

#### Calculate Indicators (Single Stock)
```bash
POST /api/indicators/calculate/VNM?days=365
```

#### Get Indicators
```bash
GET /api/indicators/VNM?limit=30
```

### 2. Python Script

#### Test Module
```bash
python scripts/test_indicators_scheduler.py
```

#### Run Indicators Calculation
```python
from src.database.connection import get_db
from src.features.indicators_processor import run_indicator_calculation

result = run_indicator_calculation()
print(f"Success: {result['success']}/{result['total']}")
```

#### Start Scheduler Manually
```python
from src.scheduler.daily_scheduler import init_scheduler

scheduler = init_scheduler()
# Scheduler sẽ chạy background
```

### 3. Từ Command Line

#### Cài đặt dependencies
```bash
pip install apscheduler>=3.10.0
```

#### Chạy test
```bash
python scripts/test_indicators_scheduler.py
```

## 📊 Database Schema

### Bảng `technical_indicators`

```sql
CREATE TABLE technical_indicators (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    date DATE NOT NULL,
    
    -- Moving Averages
    sma_20 FLOAT,
    sma_50 FLOAT,
    sma_200 FLOAT,
    ema_12 FLOAT,
    ema_26 FLOAT,
    
    -- Momentum
    rsi_14 FLOAT,
    macd FLOAT,
    macd_signal FLOAT,
    macd_histogram FLOAT,
    
    -- Volatility
    bb_upper FLOAT,
    bb_middle FLOAT,
    bb_lower FLOAT,
    atr_14 FLOAT,
    
    -- Oscillators
    stoch_k FLOAT,
    stoch_d FLOAT,
    williams_r FLOAT,
    
    -- Volume
    obv FLOAT,
    
    -- Trend
    adx FLOAT,
    plus_di FLOAT,
    minus_di FLOAT,
    cci FLOAT,
    
    created_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(stock_id, date)
);
```

## 🔧 Configuration

### Scheduler Settings

Trong `src/scheduler/daily_scheduler.py`:

```python
# Daily job: Mon-Fri at 18:00
self.scheduler.add_job(
    self.daily_update_job,
    trigger=CronTrigger(
        day_of_week='mon-fri',
        hour=18,
        minute=0
    )
)

# Weekly catch-up: Sunday at 10:00
self.scheduler.add_job(
    self.daily_update_job,
    trigger=CronTrigger(
        day_of_week='sun',
        hour=10,
        minute=0
    )
)
```

### Indicators Settings

Trong `src/features/indicators_processor.py`:

```python
# Moving Averages
result_df['sma_20'] = calculator.calculate_sma(df, window=20)
result_df['sma_50'] = calculator.calculate_sma(df, window=50)

# RSI
result_df['rsi_14'] = calculator.calculate_rsi(df, window=14)

# MACD
macd_df = calculator.calculate_macd(df, fast=12, slow=26, signal=9)
```

## 📝 Ví dụ

### Tính indicators cho VNM
```python
from src.database.connection import get_db
from src.features.indicators_processor import IndicatorsProcessor

db = next(get_db())
processor = IndicatorsProcessor(db)

# Tính cho stock_id = 1 (VNM)
success = processor.process_stock(stock_id=1, days=365)

if success:
    print("✅ Indicators calculated")
```

### Start scheduler và để chạy background
```python
from src.scheduler.daily_scheduler import DailyDataScheduler

scheduler = DailyDataScheduler()
scheduler.start()

# Scheduler đang chạy background
# Nhấn Ctrl+C để dừng
import time
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    scheduler.stop()
```

### Trigger manual update qua API
```bash
curl -X POST http://localhost:8000/api/scheduler/run-now
```

### Lấy indicators data
```bash
curl http://localhost:8000/api/indicators/VNM?limit=30
```

## 🐛 Troubleshooting

### Lỗi: "No price data found"
- Kiểm tra bảng `stock_prices` có dữ liệu chưa
- Chạy `/api/data/sync-daily` để fetch dữ liệu

### Lỗi: "Scheduler not started"
- Gọi `POST /api/scheduler/start` trước
- Hoặc chạy `init_scheduler()` trong Python

### Indicators có giá trị None
- Bình thường với dữ liệu đầu (do rolling window)
- Cần ít nhất 200 ngày dữ liệu cho SMA200

## 📚 Tham khảo

- **TechnicalIndicators class**: `src/features/technical_indicators.py`
- **Database models**: `src/database/models.py`
- **API documentation**: http://localhost:8000/docs

## 📦 Dependencies

```
apscheduler>=3.10.0  # Background scheduler
pandas>=2.3.0        # Data processing
numpy>=2.3.0         # Numerical computing
sqlalchemy>=2.0.0    # Database ORM
```

## ✅ Checklist Triển khai

- [x] Tạo `IndicatorsProcessor` class
- [x] Tạo `DailyDataScheduler` class
- [x] Thêm API endpoints
- [x] Tạo test script
- [x] Update `requirements.txt`
- [x] Viết documentation

## 🎉 Kết quả

Module này cho phép:
- ✅ Tự động cập nhật dữ liệu giá hàng ngày
- ✅ Tự động tính toán 15+ technical indicators
- ✅ Lưu trữ dữ liệu vào Database
- ✅ Truy vấn qua REST API
- ✅ Lên lịch chạy tự động
