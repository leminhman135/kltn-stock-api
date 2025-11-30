# KLTN - Báo Cáo Tình Trạng Dự Án

**Ngày cập nhật**: 30/11/2025
**Đề tài**: Dự đoán giá cổ phiếu sử dụng AI & Machine Learning

---

## 📋 CHECKLIST THEO YÊU CẦU

### 1️⃣ NGHIÊN CỨU MÔ HÌNH (Research & Comparison)

#### ✅ **Đã Có - Các mô hình đã implement:**

| Mô hình | File | Trạng thái | Ghi chú |
|---------|------|-----------|---------|
| **ARIMA** | `src/models/arima_model.py` | ✅ Hoàn chỉnh | 339 dòng, có auto_arima, seasonal |
| **Prophet** | `src/models/prophet_model.py` | ✅ Hoàn chỉnh | Facebook Prophet, xử lý seasonality |
| **LSTM** | `src/models/lstm_gru_models.py` | ✅ Hoàn chỉnh | Deep learning, sequential data |
| **GRU** | `src/models/lstm_gru_models.py` | ✅ Hoàn chỉnh | Faster than LSTM |
| **Ensemble** | `src/models/ensemble.py` | ✅ Hoàn chỉnh | Kết hợp nhiều mô hình |

**Kiến trúc & So sánh:**
- ✅ ARIMA: Documented (statsmodels)
- ✅ Prophet: Documented (additive model)
- ✅ LSTM: Documented (RNN architecture)
- ✅ GRU: Documented (simplified LSTM)
- ✅ Ensemble: Simple averaging/weighted

#### ❌ **Chưa Có - Cần bổ sung:**

| Nội dung | Mức độ | Ghi chú |
|----------|--------|---------|
| **So sánh chi tiết** | ⚠️ Quan trọng | Viết paper/report so sánh 4 mô hình |
| **Biểu đồ kiến trúc** | ⚠️ Quan trọng | Vẽ architecture diagram |
| **Benchmark results** | ⚠️ Quan trọng | Bảng so sánh MAE, RMSE, Training time |

---

### 2️⃣ FINBERT - PHÂN TÍCH CẢM TÍNH

#### ✅ **Đã Có:**
- File: `src/features/sentiment_analysis.py` (177 dòng)
- ✅ FinBERT model loaded
- ✅ Analyze text sentiment
- ✅ Batch processing

#### ❌ **Chưa Có - Cần bổ sung:**

| Nội dung | File cần tạo | Mức độ |
|----------|-------------|--------|
| **Kiến trúc FinBERT** | `docs/FINBERT_ARCHITECTURE.md` | 🔴 Quan trọng |
| **Cách hoạt động** | Same as above | 🔴 Quan trọng |
| **Fine-tuning guide** | `docs/FINBERT_FINETUNING.md` | ⚠️ Tùy chọn |
| **Vietnamese sentiment** | Update sentiment_analysis.py | 🔴 Quan trọng (VN stocks) |

**Action items:**
```python
# Cần implement:
# 1. Tiếng Việt sentiment (PhoBERT hoặc ViT5)
# 2. Document FinBERT architecture
# 3. Explain BERT → FinBERT adaptation
```

---

### 3️⃣ META-LEARNING & ENSEMBLE

#### ✅ **Đã Có:**
- File: `src/models/ensemble.py`
- ✅ Simple averaging
- ✅ Weighted ensemble
- ✅ Combine predictions

#### ❌ **Chưa Có - Cần bổ sung:**

| Kỹ thuật | Mức độ | Ghi chú |
|----------|--------|---------|
| **Stacking** | 🔴 Quan trọng | Meta-model trên top |
| **Blending** | ⚠️ Tùy chọn | Similar to stacking |
| **Boosting** | ⚠️ Tùy chọn | AdaBoost, XGBoost |
| **Voting** | ✅ Có rồi | Trong ensemble.py |

**Action items:**
```python
# Cần tạo: src/models/meta_learning.py
# - Stacking ensemble
# - Cross-validation strategy
# - Meta-features generation
```

---

### 4️⃣ BACKTESTING ENGINE

#### ✅ **Đã Có:**
- File: `src/backtesting.py` (395 dòng)
- ✅ Long/Short strategies
- ✅ Mean reversion
- ✅ Performance metrics (Sharpe, Max Drawdown)
- ✅ Trade history
- ✅ Web UI integration

#### ❌ **Chưa Có - Cần bổ sung:**

| Nội dung | Mức độ | Ghi chú |
|----------|--------|---------|
| **Transaction costs** | ⚠️ Quan trọng | Phí môi giới, thuế |
| **Slippage** | ⚠️ Quan trọng | Chênh lệch giá thực tế |
| **Walk-forward analysis** | 🔴 Quan trọng | Rolling window validation |
| **Risk metrics** | ⚠️ Tùy chọn | VaR, CVaR, Sortino |

**Action items:**
```python
# Update src/backtesting.py:
# - Add transaction_cost parameter
# - Implement slippage model
# - Add walk_forward_analysis()
```

---

### 5️⃣ TỰ ĐỘNG HÓA (Automation)

#### ✅ **Đã Có:**
- ❌ **KHÔNG CÓ** Cronjob
- ❌ **KHÔNG CÓ** Apache Airflow
- ❌ **KHÔNG CÓ** Scheduling system

#### 🔴 **CẦN XÂY DỰNG GẤP:**

| Công cụ | File cần tạo | Mức độ |
|---------|-------------|--------|
| **Cronjob script** | `automation/daily_collection.sh` | 🔴 Quan trọng |
| **Airflow DAG** | `airflow/dags/stock_pipeline.py` | 🔴 Quan trọng |
| **Scheduler** | `automation/scheduler.py` | 🔴 Quan trọng |
| **Docker compose** | `docker-compose.yml` | ⚠️ Tùy chọn |

**Action items:**
```bash
# Cần tạo:
1. automation/
   ├── daily_collection.sh      # Cronjob cho Linux
   ├── daily_collection.ps1      # Task Scheduler cho Windows
   └── scheduler.py              # Python APScheduler

2. airflow/
   ├── dags/
   │   ├── data_collection_dag.py
   │   ├── model_training_dag.py
   │   └── prediction_dag.py
   └── docker-compose.yml
```

---

### 6️⃣ THU THẬP DỮ LIỆU (Data Collection)

#### ✅ **Đã Có:**
- File: `src/data_collection.py` (460+ dòng)
- ✅ **API Module**: Yahoo Finance, VNDirect dchart
- ✅ **Web Scraping**: BeautifulSoup, Scrapy
- ✅ Multiple data sources

**Chi tiết:**
| Nguồn | Loại | Trạng thái |
|-------|------|-----------|
| Yahoo Finance API | ✅ API | Hoạt động tốt |
| VNDirect dchart | ✅ API | Hoạt động tốt (11 endpoints) |
| CafeF scraping | ✅ Scraping | BeautifulSoup ready |
| VNDirect news | ✅ Scraping | Template ready |

#### ⚠️ **Cần cải thiện:**
- ⚠️ Error handling & retry logic
- ⚠️ Rate limiting
- ⚠️ Proxy rotation (nếu cần)

---

### 7️⃣ ETL PIPELINE (Extract-Transform-Load)

#### ✅ **Đã Có:**
- File: `src/data_processing.py` (259 dòng)
- ✅ **Extract**: From APIs & scraping
- ✅ **Transform**: Clean, normalize
- ✅ **Load**: To CSV (local)

**Chi tiết:**
```python
# src/data_processing.py includes:
- DataProcessor class
- clean_data()
- normalize_prices()
- handle_missing_values()
- feature_engineering()
```

#### ❌ **Chưa Có - Cần bổ sung:**

| Component | File cần tạo | Mức độ |
|-----------|-------------|--------|
| **Database integration** | 🔴 CẦN GẤP | Không có DB |
| **PostgreSQL/MySQL** | `src/database/connection.py` | 🔴 Quan trọng |
| **MongoDB** | `src/database/nosql.py` | ⚠️ Tùy chọn |
| **Data validation** | `src/data_validation.py` | ⚠️ Quan trọng |

---

### 8️⃣ ĐẶC TRƯNG KỸ THUẬT (Technical Features)

#### ✅ **Đã Có:**
- File: `src/features/technical_indicators.py` (458 dòng)
- ✅ MACD, RSI, Bollinger Bands
- ✅ Moving Averages (SMA, EMA, WMA)
- ✅ Stochastic, Williams %R
- ✅ ATR, ADX, OBV
- ✅ 25+ indicators

**Hoàn chỉnh**: ✅ **100%**

---

### 9️⃣ PHÂN TÍCH CẢM TÍNH (Sentiment Analysis)

#### ✅ **Đã Có:**
- File: `src/features/sentiment_analysis.py`
- ✅ FinBERT integration
- ✅ Batch processing
- ✅ Score calculation

#### ❌ **Chưa Có - Cần bổ sung:**

| Nội dung | Mức độ | Ghi chú |
|----------|--------|---------|
| **Vietnamese NLP** | 🔴 Quan trọng | PhoBERT, ViT5 |
| **News aggregation** | 🔴 Quan trọng | Daily sentiment scores |
| **Database storage** | 🔴 Quan trọng | Store sentiment by date |

**Action items:**
```python
# Update src/features/sentiment_analysis.py:
# - Add Vietnamese model (PhoBERT)
# - Add aggregate_daily_sentiment()
# - Add store_to_database()
```

---

### 🔟 ỨNG DỤNG WEB (Web Application)

#### ✅ **Đã Có:**
- File: `src/web_app.py` (2500+ dòng)
- ✅ **8 pages**: Home, Market, Data, Check, Predict, Backtest, Sentiment, Training
- ✅ **Navigation bar**: Horizontal menu
- ✅ **Backtesting UI**: Interactive
- ✅ **Charts**: Plotly candlestick, line charts
- ✅ **Data validation**: Compare VNDirect vs Yahoo
- ✅ **Theme**: Ocean blue with orange accents
- ✅ **Responsive**: Professional design

**Hoàn chỉnh**: ✅ **90%**

#### ⚠️ **Cần cải thiện:**
- ⚠️ User authentication
- ⚠️ Save user preferences
- ⚠️ Portfolio management
- ⚠️ Export reports (PDF)

---

### 1️⃣1️⃣ API ENDPOINT (REST API)

#### ✅ **Đã Có:**
- File: `src/api.py` (43 dòng)
- ⚠️ **Chỉ có template** - Chưa implement đầy đủ

#### 🔴 **CẦN XÂY DỰNG GẤP:**

**Endpoints cần có:**
```python
# src/api.py - Cần mở rộng

GET  /api/stocks/{symbol}           # Stock info
GET  /api/stocks/{symbol}/price     # Historical prices
GET  /api/stocks/{symbol}/predict   # Predictions
POST /api/predict                   # Predict multiple stocks
GET  /api/indicators/{symbol}       # Technical indicators
GET  /api/sentiment/{symbol}        # Sentiment scores
POST /api/backtest                  # Run backtest
GET  /api/models                    # List available models
POST /api/models/train              # Train model
```

**Framework:** FastAPI hoặc Flask

---

### 1️⃣2️⃣ LƯU TRỮ DATABASE ONLINE

#### ❌ **CHƯA CÓ - CẦN XÂY DỰNG:**

**Options:**

| Database | Ưu điểm | Nhược điểm | Khuyến nghị |
|----------|---------|-----------|-------------|
| **PostgreSQL** | Relational, SQL, ACID | Setup phức tạp | ⭐ **Recommended** |
| **MongoDB** | NoSQL, flexible schema | Không có transactions | ⚠️ Backup option |
| **MySQL** | Popular, stable | Slower than PostgreSQL | ⚠️ Alternative |
| **SQLite** | Simple, file-based | Not for production | ❌ Local only |
| **Firebase** | Real-time, cloud | Cost, vendor lock-in | ⚠️ For prototyping |

**🔴 Recommended Stack:**
```
PostgreSQL (TimescaleDB) + Supabase/Railway
```

**Cần tạo:**
```
src/database/
├── __init__.py
├── connection.py          # Database connection pool
├── models.py              # SQLAlchemy models
├── crud.py                # CRUD operations
└── migrations/            # Alembic migrations
    ├── env.py
    └── versions/
```

**Schema tables cần có:**
```sql
-- stocks (thông tin cổ phiếu)
-- prices (giá lịch sử)
-- indicators (chỉ số kỹ thuật)
-- sentiment (điểm cảm tính)
-- predictions (dự đoán)
-- models (model metadata)
-- backtests (kết quả backtest)
-- users (nếu có authentication)
```

---

## 📊 TỔNG KẾT

### ✅ Đã Hoàn Thành (70%):

1. ✅ **4 mô hình**: ARIMA, Prophet, LSTM, GRU
2. ✅ **Ensemble model**
3. ✅ **Backtesting engine**: Basic
4. ✅ **Data collection**: APIs + Scraping
5. ✅ **ETL pipeline**: Extract, Transform (Load local only)
6. ✅ **Technical indicators**: 25+ indicators
7. ✅ **Sentiment analysis**: FinBERT (English only)
8. ✅ **Web UI**: 8 pages, professional design
9. ✅ **Data validation**: Compare sources

### 🔴 Cần Làm Gấp (Quan trọng):

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 🔴 P0 | **Database online** (PostgreSQL) | 2-3 ngày | Critical |
| 🔴 P0 | **Automation** (Airflow/Cronjob) | 2-3 ngày | Critical |
| 🔴 P0 | **API endpoints** (FastAPI) | 2 ngày | Critical |
| 🔴 P1 | **Model comparison report** | 1 ngày | Important |
| 🔴 P1 | **FinBERT documentation** | 1 ngày | Important |
| 🔴 P1 | **Vietnamese sentiment** (PhoBERT) | 2 ngày | Important |

### ⚠️ Cần Cải Thiện (Tùy chọn):

| Priority | Task | Effort |
|----------|------|--------|
| ⚠️ P2 | Meta-learning (Stacking) | 1-2 ngày |
| ⚠️ P2 | Walk-forward analysis | 1 ngày |
| ⚠️ P2 | Transaction costs in backtest | 0.5 ngày |
| ⚠️ P3 | User authentication | 1 ngày |
| ⚠️ P3 | Portfolio management | 2 ngày |

---

## 🚀 ROADMAP KẾ HOẠCH

### **Week 1: Database & Infrastructure**
```
Day 1-2: Setup PostgreSQL + Supabase/Railway
Day 3-4: Create schema, models, migrations
Day 5: Test CRUD operations
```

### **Week 2: Automation & API**
```
Day 1-2: Airflow DAG setup
Day 3-4: FastAPI endpoints
Day 5: Testing & integration
```

### **Week 3: Documentation & Improvements**
```
Day 1: Model comparison report
Day 2: FinBERT architecture docs
Day 3: Vietnamese sentiment (PhoBERT)
Day 4-5: Final testing & deployment
```

---

## 📝 FILES CẦN TẠO

```
KLTN/
├── automation/                    # 🔴 NEW
│   ├── scheduler.py
│   ├── daily_collection.sh
│   └── airflow/
│       ├── docker-compose.yml
│       └── dags/
│           ├── data_collection_dag.py
│           ├── training_dag.py
│           └── prediction_dag.py
│
├── src/
│   ├── database/                  # 🔴 NEW
│   │   ├── __init__.py
│   │   ├── connection.py
│   │   ├── models.py
│   │   ├── crud.py
│   │   └── migrations/
│   │
│   ├── api.py                     # ⚠️ EXPAND (hiện tại 43 dòng)
│   │
│   ├── models/
│   │   └── meta_learning.py      # 🔴 NEW
│   │
│   └── features/
│       └── vietnamese_sentiment.py # 🔴 NEW
│
├── docs/                          # 🔴 NEW
│   ├── MODEL_COMPARISON.md
│   ├── FINBERT_ARCHITECTURE.md
│   └── API_DOCUMENTATION.md
│
├── tests/                         # 🔴 NEW
│   ├── test_models.py
│   ├── test_api.py
│   └── test_database.py
│
├── .env.example                   # 🔴 NEW
├── docker-compose.yml             # 🔴 NEW
└── requirements-prod.txt          # 🔴 NEW
```

---

## ✅ ACTION ITEMS - BẮT ĐẦU NGAY

### 1. Setup Database (Priority 0)
```bash
# Install PostgreSQL client
pip install psycopg2-binary sqlalchemy alembic

# Create database structure
mkdir -p src/database
# ... create files ...
```

### 2. Setup Automation (Priority 0)
```bash
# Install Airflow
pip install apache-airflow apache-airflow-providers-postgres

# Create DAG structure
mkdir -p airflow/dags
# ... create DAG files ...
```

### 3. Expand API (Priority 0)
```bash
# Install FastAPI
pip install fastapi uvicorn pydantic

# Expand src/api.py
# ... add all endpoints ...
```

---

**Kết luận**: Dự án đã hoàn thành **~70%**. Cần tập trung vào:
1. 🔴 **Database online** (PostgreSQL)
2. 🔴 **Automation** (Airflow)
3. 🔴 **API expansion** (FastAPI)

Sau đó mới làm phần documentation và improvements.
