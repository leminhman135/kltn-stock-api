# BÁO CÁO TÀI LIỆU CÔNG NGHỆ
## Hệ Thống Dự Đoán Giá Cổ Phiếu Việt Nam (KLTN Stock API)

---

## 📋 MỤC LỤC

1. [Tổng Quan Hệ Thống](#1-tổng-quan-hệ-thống)
2. [Kiến Trúc Hệ Thống](#2-kiến-trúc-hệ-thống)
3. [Công Nghệ Sử Dụng](#3-công-nghệ-sử-dụng)
4. [Cấu Trúc Dự Án](#4-cấu-trúc-dự-án)
5. [Database Schema](#5-database-schema)
6. [API Endpoints](#6-api-endpoints)
7. [Mô Hình Machine Learning](#7-mô-hình-machine-learning)
8. [Phân Tích Sentiment](#8-phân-tích-sentiment)
9. [Triển Khai & Vận Hành](#9-triển-khai--vận-hành)
10. [Bảo Mật & Hiệu Năng](#10-bảo-mật--hiệu-năng)

---

## 1. TỔNG QUAN HỆ THỐNG

### 1.1 Giới thiệu

**KLTN Stock API** là hệ thống dự đoán giá cổ phiếu Việt Nam sử dụng các kỹ thuật Machine Learning kết hợp với phân tích kỹ thuật (Technical Analysis) và phân tích cảm xúc thị trường (Sentiment Analysis).

### 1.2 Mục tiêu

| Mục tiêu | Mô tả |
|----------|-------|
| **Thu thập dữ liệu** | Tự động lấy dữ liệu giá cổ phiếu từ VNDirect API |
| **Phân tích kỹ thuật** | Tính toán 20+ chỉ báo kỹ thuật (RSI, MACD, Bollinger Bands...) |
| **Phân tích sentiment** | Phân tích cảm xúc tin tức từ các nguồn uy tín |
| **Dự đoán giá** | Sử dụng ML models để dự đoán xu hướng giá |
| **API RESTful** | Cung cấp API cho ứng dụng frontend |

### 1.3 Phạm vi

- **Thị trường**: Sàn HOSE, HNX, UPCOM (Việt Nam)
- **Cổ phiếu hỗ trợ**: 30 mã VN30 + có thể mở rộng
- **Dữ liệu**: Historical data từ 2020 đến hiện tại
- **Dự đoán**: 1-30 ngày tới

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Kiến trúc tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Web Browser  │  Mobile App  │  Postman/API Client  │  Frontend │
└───────────────┴──────────────┴───────────────────────┴───────────┘
                                    │
                                    ▼ HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY (FastAPI)                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  /stocks │ │ /prices  │ │/predict  │ │/sentiment│           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │/indicators│ │  /news   │ │/backtest │ │  /admin  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
┌───────────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   BUSINESS LOGIC      │ │  ML MODELS      │ │ DATA COLLECTION │
├───────────────────────┤ ├─────────────────┤ ├─────────────────┤
│ • Technical Indicators│ │ • Random Forest │ │ • VNDirect API  │
│ • Sentiment Analysis  │ │ • Gradient Boost│ │ • RSS Feeds     │
│ • Backtesting Engine  │ │ • Ridge/Elastic │ │ • Web Scraping  │
│ • ETL Pipeline        │ │ • Ensemble      │ │ • CafeF, VST    │
└───────────────────────┘ └─────────────────┘ └─────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      DATA LAYER (PostgreSQL)                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │  stocks  │ │stock_    │ │technical_│ │sentiment_│           │
│  │          │ │prices    │ │indicators│ │analysis  │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │predictions│ │model_    │ │news_     │                        │
│  │          │ │metrics   │ │articles  │                        │
│  └──────────┘ └──────────┘ └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Luồng dữ liệu (Data Flow)

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   VNDirect  │───▶│    ETL      │───▶│  Database   │───▶│     API     │
│     API     │    │  Pipeline   │    │ PostgreSQL  │    │   FastAPI   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
                         │                   │
                         ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Technical  │    │     ML      │
                   │ Indicators  │    │   Models    │
                   └─────────────┘    └─────────────┘
                         │                   │
                         └─────────┬─────────┘
                                   ▼
                          ┌─────────────┐
                          │ Predictions │
                          └─────────────┘
```

---

## 3. CÔNG NGHỆ SỬ DỤNG

### 3.1 Backend Technologies

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **Python** | 3.11.0 | Ngôn ngữ lập trình chính |
| **FastAPI** | 0.117.1 | Web framework - REST API |
| **Uvicorn** | 0.38.0 | ASGI server |
| **Pydantic** | 2.12.5 | Data validation |

### 3.2 Database

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **PostgreSQL** | 15+ | Database chính |
| **SQLAlchemy** | 2.0.44 | ORM (Object-Relational Mapping) |
| **Alembic** | 1.17.2 | Database migrations |
| **psycopg2-binary** | 2.9.11 | PostgreSQL adapter |

### 3.3 Machine Learning & Data Science

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **scikit-learn** | 1.6.1 | ML algorithms (RF, GB, Ridge) |
| **pandas** | 2.3.3 | Data manipulation |
| **numpy** | 2.3.5 | Numerical computing |

### 3.4 Data Collection

| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| **requests** | 2.32.5 | HTTP client |
| **BeautifulSoup4** | 4.14.2 | Web scraping |

### 3.5 Infrastructure

| Công nghệ | Mục đích |
|-----------|----------|
| **Render.com** | Cloud hosting (Web + Database) |
| **UptimeRobot** | Keep-alive monitoring |
| **GitHub** | Version control & CI/CD |

---

## 4. CẤU TRÚC DỰ ÁN

```
KLTN/
├── 📁 src/                          # Source code chính
│   ├── 📄 api_v2.py                 # FastAPI application (3,693 lines)
│   ├── 📄 model.py                  # ML models (727 lines)
│   ├── 📄 news_service.py           # News scraping & sentiment (537 lines)
│   ├── 📄 data_collection.py        # Data collection utilities
│   ├── 📄 analysis.py               # Data analysis
│   │
│   ├── 📁 database/                 # Database layer
│   │   ├── 📄 connection.py         # Database connection
│   │   ├── 📄 models.py             # SQLAlchemy models
│   │   ├── 📄 extended_models.py    # Extended models
│   │   └── 📄 helpers.py            # Database helpers
│   │
│   ├── 📁 models/                   # ML model implementations
│   │   ├── 📄 arima_model.py        # ARIMA model
│   │   ├── 📄 prophet_model.py      # Prophet model
│   │   ├── 📄 lstm_gru_models.py    # Deep learning models
│   │   └── 📄 ensemble.py           # Ensemble methods
│   │
│   ├── 📁 features/                 # Feature engineering
│   │   ├── 📄 technical_indicators.py
│   │   └── 📄 sentiment_analysis.py
│   │
│   ├── 📁 etl/                      # ETL pipeline
│   ├── 📁 backtest/                 # Backtesting engine
│   ├── 📁 scheduler/                # Task scheduling
│   └── 📁 static/                   # Static files (dashboard)
│
├── 📁 data/                         # Data files
├── 📁 docs/                         # Documentation
├── 📁 scripts/                      # Utility scripts
│
├── 📄 main.py                       # Entry point
├── 📄 requirements.txt              # Python dependencies
├── 📄 render.yaml                   # Render deployment config
├── 📄 build.sh                      # Build script
└── 📄 start.sh                      # Start script
```

---

## 5. DATABASE SCHEMA

### 5.1 Entity Relationship Diagram

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│     stocks      │       │   stock_prices  │       │ technical_      │
├─────────────────┤       ├─────────────────┤       │ indicators      │
│ PK id           │──────<│ FK stock_id     │       ├─────────────────┤
│    symbol       │       │ PK id           │       │ FK stock_id     │
│    name         │       │    date         │       │ PK id           │
│    exchange     │       │    open         │       │    date         │
│    sector       │       │    high         │       │    sma_20       │
│    is_active    │       │    low          │       │    rsi_14       │
│    created_at   │       │    close        │       │    macd         │
│    updated_at   │       │    volume       │       │    bb_upper     │
└─────────────────┘       │    source       │       │    bb_lower     │
         │                └─────────────────┘       │    calculated_at│
         │                                          └─────────────────┘
         │
         │                ┌─────────────────┐       ┌─────────────────┐
         │                │ sentiment_      │       │   predictions   │
         │                │ analysis        │       ├─────────────────┤
         └───────────────<├─────────────────┤       │ FK stock_id     │
                         │ FK stock_id     │       │ FK model_id     │
                         │ PK id           │       │ PK id           │
                         │    date         │       │    prediction_  │
                         │    sentiment_   │       │    date         │
                         │    score        │       │    target_date  │
                         │    sentiment_   │       │    predicted_   │
                         │    label        │       │    close        │
                         │    news_count   │       │    confidence_  │
                         │    model_name   │       │    upper        │
                         └─────────────────┘       │    confidence_  │
                                                   │    lower        │
                                                   └─────────────────┘

┌─────────────────┐       ┌─────────────────┐
│  model_metrics  │       │  news_articles  │
├─────────────────┤       ├─────────────────┤
│ PK id           │       │ PK id           │
│    model_name   │       │    stock_symbol │
│    stock_symbol │       │    title        │
│    mae          │       │    summary      │
│    rmse         │       │    url          │
│    mape         │       │    source       │
│    r2_score     │       │    sentiment_   │
│    hyperparams  │       │    score        │
│    is_active    │       │    published_at │
│    trained_at   │       │    scraped_at   │
└─────────────────┘       └─────────────────┘
```

### 5.2 Chi tiết bảng chính

#### stocks
```sql
CREATE TABLE stocks (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) UNIQUE NOT NULL,
    name VARCHAR(200) NOT NULL,
    exchange VARCHAR(50) DEFAULT 'HOSE',
    sector VARCHAR(100),
    industry VARCHAR(100),
    market_cap FLOAT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);
```

#### stock_prices
```sql
CREATE TABLE stock_prices (
    id SERIAL PRIMARY KEY,
    stock_id INTEGER REFERENCES stocks(id),
    date DATE NOT NULL,
    open FLOAT NOT NULL,
    high FLOAT NOT NULL,
    low FLOAT NOT NULL,
    close FLOAT NOT NULL,
    volume FLOAT NOT NULL,
    source VARCHAR(50) DEFAULT 'vndirect',
    UNIQUE(stock_id, date)
);
CREATE INDEX ix_stock_prices_date ON stock_prices(stock_id, date);
```

---

## 6. API ENDPOINTS

### 6.1 Tổng quan API

| Nhóm | Số endpoints | Mô tả |
|------|--------------|-------|
| **Root** | 2 | Health check, root info |
| **Stocks** | 4 | CRUD operations cho stocks |
| **Prices** | 8 | Historical price data |
| **Market Board** | 6 | Market overview by date |
| **Indicators** | 2 | Technical indicators |
| **Predictions** | 5 | ML predictions |
| **Sentiment** | 3 | Sentiment analysis |
| **News** | 3 | News articles |
| **Models** | 3 | Model management |
| **Backtest** | 2 | Backtesting |
| **Data Collection** | 6 | Data fetching |
| **Admin** | 4 | Database management |
| **Trading Data** | 5 | Trading information |
| **Market Data** | 8 | Market indices |
| **Financial Data** | 10 | Financial statements |
| **Industry** | 8 | Sector analysis |

### 6.2 API Endpoints chi tiết

#### 📊 Stock Endpoints
```
GET  /api/stocks                 # Danh sách cổ phiếu
GET  /api/stocks/{symbol}        # Chi tiết cổ phiếu
GET  /api/stocks/search?q=xxx    # Tìm kiếm
```

#### 💰 Price Endpoints
```
GET  /api/prices/{symbol}                    # Lịch sử giá
GET  /api/prices/{symbol}/latest             # Giá mới nhất
GET  /api/prices/{symbol}/by-date?date=xxx   # Giá theo ngày
GET  /api/prices/{symbol}/historical         # Giá lịch sử
GET  /api/prices/{symbol}/ohlcv              # Dữ liệu OHLCV
```

#### 🤖 Prediction Endpoints
```
POST /api/predictions/train/{symbol}    # Train model
POST /api/predictions/predict           # Tạo dự đoán
GET  /api/predictions/quick/{symbol}    # Dự đoán nhanh
GET  /api/predictions/{symbol}          # Lấy dự đoán
GET  /api/predictions/{symbol}/latest   # Dự đoán mới nhất
```

#### 📰 Sentiment Endpoints
```
GET  /api/sentiment/{symbol}          # Sentiment history
GET  /api/sentiment/{symbol}/latest   # Latest sentiment
GET  /api/news/{symbol}               # Tin tức
GET  /api/news/{symbol}/sentiment     # Tin + sentiment
```

#### 📈 Technical Indicators
```
GET  /api/indicators/{symbol}         # All indicators
GET  /api/indicators/{symbol}/latest  # Latest indicators
```

### 6.3 Response Format

```json
{
    "success": true,
    "data": { ... },
    "message": "Success",
    "timestamp": "2024-12-02T10:30:00Z"
}
```

### 6.4 Error Response

```json
{
    "detail": "Stock not found",
    "status_code": 404
}
```

---

## 7. MÔ HÌNH MACHINE LEARNING

### 7.1 Tổng quan Models

| Model | Loại | Ưu điểm | Nhược điểm |
|-------|------|---------|------------|
| **Ridge Regression** | Linear | Nhanh, stable | Linear only |
| **ElasticNet** | Linear | Regularization | Linear only |
| **Random Forest** | Ensemble | Non-linear, robust | Slow training |
| **Gradient Boosting** | Ensemble | High accuracy | Overfitting risk |
| **Ensemble (Voting)** | Meta | Best of all | Complexity |

### 7.2 Feature Engineering

#### Technical Indicators (Input Features)
```python
features = {
    # Trend Indicators
    'sma_5', 'sma_10', 'sma_20', 'sma_50',
    'ema_12', 'ema_26',
    
    # Momentum Indicators
    'rsi_14',           # Relative Strength Index
    'macd',             # MACD line
    'macd_signal',      # Signal line
    'macd_histogram',   # MACD histogram
    
    # Volatility Indicators
    'bb_upper',         # Bollinger Upper
    'bb_middle',        # Bollinger Middle
    'bb_lower',         # Bollinger Lower
    'bb_width',         # Bandwidth
    
    # Price Features
    'price_change',     # Daily change
    'price_change_pct', # % change
    'high_low_range',   # Daily range
    
    # Volume Features
    'volume_change',    # Volume change
    'volume_ma_ratio',  # Volume vs MA
    
    # Sentiment Features
    'sentiment_score',  # -1 to 1
    'news_count',       # Number of news
}
```

### 7.3 Model Pipeline

```python
# Training Pipeline
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Raw Data   │───▶│   Feature   │───▶│   Train/    │
│  (Prices)   │    │ Engineering │    │   Test      │
└─────────────┘    └─────────────┘    │   Split     │
                                      └─────────────┘
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                        ▼                        ▼
            ┌─────────────┐          ┌─────────────┐          ┌─────────────┐
            │   Ridge     │          │   Random    │          │  Gradient   │
            │ Regression  │          │   Forest    │          │  Boosting   │
            └─────────────┘          └─────────────┘          └─────────────┘
                   │                        │                        │
                   └────────────────────────┼────────────────────────┘
                                            ▼
                                     ┌─────────────┐
                                     │  Ensemble   │
                                     │   (Vote)    │
                                     └─────────────┘
                                            │
                                            ▼
                                     ┌─────────────┐
                                     │ Prediction  │
                                     └─────────────┘
```

### 7.4 Metrics đánh giá

| Metric | Công thức | Ý nghĩa |
|--------|-----------|---------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$ | Mean Absolute Error |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$ | Root Mean Square Error |
| **MAPE** | $\frac{100\%}{n}\sum_{i=1}^{n}|\frac{y_i - \hat{y}_i}{y_i}|$ | Mean Absolute Percentage Error |
| **R²** | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of Determination |

---

## 8. PHÂN TÍCH SENTIMENT

### 8.1 Nguồn dữ liệu tin tức

| Nguồn | Loại | URL |
|-------|------|-----|
| **CafeF** | RSS/Scraping | cafef.vn |
| **VietStock** | RSS | vietstock.vn |
| **VnExpress Kinh doanh** | RSS | vnexpress.net |
| **NDH** | Scraping | ndh.vn |
| **Thanh Niên Tài chính** | RSS | thanhnien.vn |

### 8.2 Sentiment Analysis Algorithm

```python
class SentimentAnalyzer:
    # Positive keywords (60+ từ khóa)
    POSITIVE_KEYWORDS = [
        "tăng trưởng", "lợi nhuận tăng", "vượt kế hoạch",
        "cổ tức cao", "lãi kỷ lục", "triển vọng tốt",
        "khuyến nghị mua", "breakout", "uptrend", ...
    ]
    
    # Negative keywords (60+ từ khóa)
    NEGATIVE_KEYWORDS = [
        "thua lỗ", "giảm lợi nhuận", "nợ xấu",
        "phá sản", "bán tháo", "downtrend",
        "cảnh báo", "rủi ro cao", ...
    ]
    
    # Strong modifiers (tăng/giảm score)
    STRONG_MODIFIERS = [
        "kỷ lục", "đột biến", "lịch sử",
        "chưa từng có", "mạnh nhất", ...
    ]
```

### 8.3 Sentiment Score Calculation

```
Score = (positive_count - negative_count) / total_keywords * multiplier

Trong đó:
- positive_count: Số từ khóa tích cực
- negative_count: Số từ khóa tiêu cực
- multiplier: 1.5 nếu có strong modifier, 1.0 nếu không

Label:
- score > 0.1  → POSITIVE
- score < -0.1 → NEGATIVE
- else         → NEUTRAL
```

---

## 9. TRIỂN KHAI & VẬN HÀNH

### 9.1 Infrastructure trên Render.com

```yaml
# render.yaml
services:
  - type: web
    name: kltn-stock-api
    env: python
    region: singapore
    plan: free
    branch: main
    buildCommand: "./build.sh"
    startCommand: "./start.sh"
    healthCheckPath: /api/health

databases:
  - name: kltn-postgres
    plan: free
    region: singapore
```

### 9.2 Deployment Flow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   GitHub    │───▶│   Render    │───▶│   Build     │───▶│   Deploy    │
│   Push      │    │   Webhook   │    │   Process   │    │   Live      │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 9.3 Keep-Alive với UptimeRobot

```
┌─────────────────┐          ┌─────────────────┐
│   UptimeRobot   │──────────│   KLTN API      │
│   (Free tier)   │  ping    │   /api/health   │
│   5-10 phút     │──────────│                 │
└─────────────────┘          └─────────────────┘
```

**Tại sao cần Keep-Alive?**
- Render Free tier: API sleep sau 15 phút không hoạt động
- UptimeRobot ping mỗi 5-10 phút → API luôn active

### 9.4 Environment Variables

| Variable | Mô tả | Ví dụ |
|----------|-------|-------|
| `DATABASE_URL` | PostgreSQL connection | `postgres://user:pass@host/db` |
| `PORT` | Server port | `10000` |
| `PYTHON_VERSION` | Python version | `3.11.0` |

### 9.5 Build & Start Scripts

**build.sh**
```bash
#!/bin/bash
pip install --upgrade pip
pip install -r requirements.txt
```

**start.sh**
```bash
#!/bin/bash
uvicorn src.api_v2:app --host 0.0.0.0 --port ${PORT:-10000}
```

---

## 10. BẢO MẬT & HIỆU NĂNG

### 10.1 Security Measures

| Measure | Implementation |
|---------|----------------|
| **CORS** | Configured in FastAPI middleware |
| **HTTPS** | Enforced by Render.com |
| **Input Validation** | Pydantic models |
| **SQL Injection** | SQLAlchemy ORM (parameterized queries) |
| **Rate Limiting** | Can be added via middleware |

### 10.2 Performance Optimization

| Technique | Mô tả |
|-----------|-------|
| **Database Indexing** | Indexes trên các cột thường query |
| **Connection Pooling** | SQLAlchemy pool |
| **Async Endpoints** | FastAPI async/await |
| **Response Caching** | Can be added for static data |
| **Lazy Loading** | Load data when needed |

### 10.3 Database Indexes

```sql
-- Performance indexes
CREATE INDEX ix_stock_prices_stock_date ON stock_prices(stock_id, date);
CREATE INDEX ix_indicators_stock_date ON technical_indicators(stock_id, date);
CREATE INDEX ix_sentiment_stock_date ON sentiment_analysis(stock_id, date);
CREATE INDEX ix_predictions_stock_target ON predictions(stock_id, target_date);
CREATE INDEX ix_news_symbol_date ON news_articles(stock_symbol, published_date);
```

### 10.4 Monitoring & Logging

```python
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Startup
logger.info("🚀 Starting KLTN Stock Prediction API...")

# Success
logger.info("✅ Database tables created successfully!")

# Error
logger.error(f"❌ Database initialization error: {e}")
```

---

## 📊 THỐNG KÊ DỰ ÁN

| Metric | Giá trị |
|--------|---------|
| **Tổng số dòng code** | ~6,000+ lines |
| **Số API endpoints** | 70+ endpoints |
| **Số bảng database** | 7 tables |
| **Số models ML** | 4 models + ensemble |
| **Số technical indicators** | 20+ indicators |
| **Số nguồn tin tức** | 5+ sources |

---

## 🔗 LINKS

- **API Live**: https://kltn-stock-api.onrender.com
- **API Docs**: https://kltn-stock-api.onrender.com/docs
- **GitHub**: https://github.com/leminhman135/kltn-stock-api
- **Monitoring**: UptimeRobot Dashboard

---

## 📝 CHANGELOG

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2024-12 | PostgreSQL migration, Full API |
| 1.0.0 | 2024-11 | Initial release with SQLite |

---

*Tài liệu được cập nhật: 02/12/2024*
