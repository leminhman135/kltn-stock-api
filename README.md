# 📈 HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU VIỆT NAM SỬ DỤNG MACHINE LEARNING

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Khóa luận tốt nghiệp** - Khoa Công nghệ Thông tin
> 
> **Đề tài:** Xây dựng hệ thống dự đoán giá cổ phiếu thị trường Việt Nam sử dụng các mô hình Machine Learning kết hợp phân tích cảm tính tin tức tài chính

---

## 📋 MỤC LỤC

1. [Giới thiệu](#-giới-thiệu)
2. [Mục tiêu đề tài](#-mục-tiêu-đề-tài)
3. [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
4. [Công nghệ sử dụng](#-công-nghệ-sử-dụng)
5. [Cài đặt](#-cài-đặt)
6. [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
7. [API Documentation](#-api-documentation)
8. [Các mô hình ML](#-các-mô-hình-ml)
9. [Kết quả đánh giá](#-kết-quả-đánh-giá)
10. [Triển khai](#-triển-khai)

---

## 🎯 GIỚI THIỆU

### Lý do chọn đề tài

Thị trường chứng khoán Việt Nam ngày càng phát triển với hơn **5 triệu tài khoản giao dịch** (2024). Tuy nhiên, nhiều nhà đầu tư cá nhân thiếu công cụ phân tích chuyên nghiệp, dẫn đến quyết định đầu tư dựa trên cảm tính.

**Vấn đề cần giải quyết:**
- ❌ Khó khăn trong việc thu thập và xử lý dữ liệu giá cổ phiếu
- ❌ Thiếu công cụ phân tích kỹ thuật tự động
- ❌ Không có hệ thống phân tích tin tức và cảm tính thị trường
- ❌ Không có khả năng backtest chiến lược đầu tư

**Giải pháp đề xuất:**
- ✅ Xây dựng hệ thống thu thập dữ liệu tự động từ VNDirect API
- ✅ Phát triển các mô hình ML dự đoán giá: LSTM, GRU, Prophet, ARIMA
- ✅ Tích hợp FinBERT để phân tích cảm tính tin tức tài chính
- ✅ Xây dựng Backtesting Engine đánh giá chiến lược
- ✅ Cung cấp API và giao diện web trực quan

---

## 🎯 MỤC TIÊU ĐỀ TÀI

### Mục tiêu chính
1. **Xây dựng hệ thống thu thập dữ liệu** tự động từ các nguồn tài chính Việt Nam
2. **Phát triển pipeline ETL** để xử lý, chuẩn hóa và lưu trữ dữ liệu
3. **Xây dựng các mô hình Machine Learning** dự đoán giá cổ phiếu
4. **Tích hợp phân tích cảm tính** sử dụng FinBERT
5. **Phát triển Backtesting Engine** đánh giá hiệu suất chiến lược
6. **Xây dựng API RESTful** cung cấp dịch vụ dự đoán
7. **Thiết kế giao diện web** trực quan hóa dữ liệu và kết quả

### Phạm vi
- **Đối tượng:** 30 mã cổ phiếu VN30 trên sàn HOSE
- **Dữ liệu:** Giá OHLCV, tin tức tài chính tiếng Việt
- **Thời gian dữ liệu:** 2020 - 2025

---

## 🏗 KIẾN TRÚC HỆ THỐNG

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                         │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│   VNDirect API  │   News Scraper  │        Scheduler Service        │
│   (OHLCV Data)  │   (CafeF, etc)  │   (Cronjob / APScheduler)       │
└────────┬────────┴────────┬────────┴─────────────────────────────────┘
         │                 │
         ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ETL PIPELINE                                 │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│    Extract      │    Transform    │              Load               │
│  (Raw Data)     │ (Clean/Validate)│         (PostgreSQL)            │
└────────┬────────┴────────┬────────┴────────┬────────────────────────┘
         │                 │                 │
         ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING                             │
├─────────────────────────────────┬───────────────────────────────────┤
│     Technical Indicators        │      Sentiment Analysis           │
│  (MACD, RSI, Bollinger, SMA)    │    (FinBERT Vietnamese)           │
└────────────────┬────────────────┴───────────────┬───────────────────┘
                 │                                │
                 ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ML MODELS LAYER                                │
├──────────┬──────────┬──────────┬──────────┬────────────────────────┤
│   ARIMA  │  Prophet │   LSTM   │   GRU    │   Ensemble Model       │
│          │          │          │          │  (Weighted Average)    │
└────────┬─┴────────┬─┴────────┬─┴────────┬─┴──────────┬─────────────┘
         │          │          │          │            │
         └──────────┴──────────┴──────────┴────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EVALUATION & BACKTESTING                        │
├─────────────────────────────────────────────────────────────────────┤
│  Metrics: RMSE, MAE, MAPE, R²  │  Sharpe Ratio, Sortino, Max DD    │
└────────────────────────────────┴────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         API LAYER (FastAPI)                          │
├─────────────────────────────────────────────────────────────────────┤
│   /api/stocks    /api/prices    /api/predictions    /api/backtest   │
│   /api/ml/*      /api/news      /api/finbert        /api/indicators │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         WEB UI (Dashboard)                           │
├─────────────────────────────────────────────────────────────────────┤
│   Stock Grid  │  Price Chart  │  Prediction  │  News & Sentiment   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 CÔNG NGHỆ SỬ DỤNG

### Backend
| Công nghệ | Phiên bản | Mục đích |
|-----------|-----------|----------|
| Python | 3.9+ | Ngôn ngữ chính |
| FastAPI | 0.100+ | Web framework |
| SQLAlchemy | 2.0+ | ORM |
| PostgreSQL | 14+ | Database production |
| SQLite | 3 | Database development |

### Machine Learning
| Thư viện | Mục đích |
|----------|----------|
| TensorFlow/Keras | LSTM, GRU models |
| Prophet | Time series forecasting |
| statsmodels | ARIMA/SARIMAX |
| scikit-learn | Preprocessing, metrics |
| transformers | FinBERT sentiment |

### Data Processing
| Thư viện | Mục đích |
|----------|----------|
| pandas | Data manipulation |
| numpy | Numerical computing |
| ta-lib | Technical indicators |
| BeautifulSoup | Web scraping |

### Frontend
| Công nghệ | Mục đích |
|-----------|----------|
| HTML5/CSS3 | Structure & styling |
| JavaScript | Interactivity |
| Chart.js | Data visualization |

### Deployment
| Platform | Mục đích |
|----------|----------|
| Render.com | Cloud hosting |
| GitHub Actions | CI/CD |
| UptimeRobot | Keep-alive monitoring |

---

## ⚙️ CÀI ĐẶT

### Yêu cầu hệ thống
- Python 3.9 hoặc cao hơn
- pip (Python package manager)
- Git
- PostgreSQL (production) hoặc SQLite (development)

### Bước 1: Clone repository
```bash
git clone https://github.com/leminhman135/kltn-stock-api.git
cd kltn-stock-api
```

### Bước 2: Tạo virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Bước 3: Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### Bước 4: Cấu hình environment
```bash
# Copy file .env.example thành .env
cp scripts/.env.example scripts/.env

# Chỉnh sửa các biến môi trường
# DATABASE_URL=postgresql://user:password@host:port/dbname
```

### Bước 5: Khởi tạo database
```bash
# Chạy API server
python -m uvicorn src.api_v2:app --reload

# Truy cập API để init database
# POST http://localhost:8000/api/admin/init-db
```

### Bước 6: Thu thập dữ liệu
```bash
# Fetch dữ liệu từ VNDirect
# POST http://localhost:8000/api/data/fetch-all?days=365
```

---

## 📖 HƯỚNG DẪN SỬ DỤNG

### Chạy ứng dụng
```bash
# Development
uvicorn src.api_v2:app --reload --host 0.0.0.0 --port 8000

# Production
gunicorn src.api_v2:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Truy cập ứng dụng
- **Web UI:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Các tính năng chính

#### 1. Dashboard
- Xem danh sách 30 mã VN30
- Biểu đồ giá realtime
- Thống kê tổng quan

#### 2. Bảng giá
- Bảng giá chi tiết với giá trần/sàn
- Lọc theo ngày
- Khối lượng giao dịch

#### 3. Dự đoán giá
- Chọn mã cổ phiếu
- Chọn mô hình (Ensemble, LSTM, GRU, Prophet, ARIMA)
- Xem kết quả dự đoán với biểu đồ

#### 4. Phân tích tin tức
- Tin tức mới nhất
- Phân tích sentiment bằng FinBERT
- Tổng hợp điểm cảm tính

---

## 📡 API DOCUMENTATION

### Base URL
```
https://kltn-stock-api.onrender.com/api
```

### Endpoints chính

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| GET | `/stocks` | Danh sách cổ phiếu |
| GET | `/prices/{symbol}` | Lịch sử giá |
| POST | `/ml/arima/predict/{symbol}` | Dự đoán ARIMA |
| POST | `/ml/prophet/predict/{symbol}` | Dự đoán Prophet |
| POST | `/ml/lstm/predict/{symbol}` | Dự đoán LSTM |
| POST | `/ml/gru/predict/{symbol}` | Dự đoán GRU |
| POST | `/ml/ensemble/predict/{symbol}` | Dự đoán Ensemble |
| GET | `/ml/compare/{symbol}` | So sánh các mô hình |
| POST | `/ml/backtest/{symbol}` | Backtesting |
| GET | `/news/{symbol}` | Tin tức cổ phiếu |
| GET | `/finbert/sentiment/{symbol}` | Phân tích FinBERT |
| GET | `/indicators/{symbol}` | Chỉ báo kỹ thuật |

Xem chi tiết tại: [API Documentation](docs/API_DOCUMENTATION.md)

---

## 🤖 CÁC MÔ HÌNH ML

### 1. ARIMA (AutoRegressive Integrated Moving Average)
- **Mô tả:** Mô hình thống kê cho time series
- **Parameters:** (p, d, q) tự động tối ưu
- **Ưu điểm:** Đơn giản, hiệu quả với dữ liệu stationary

### 2. Prophet (Facebook)
- **Mô tả:** Mô hình additive với trend + seasonality
- **Ưu điểm:** Xử lý tốt missing data, outliers, holiday effects

### 3. LSTM (Long Short-Term Memory)
- **Mô tả:** Deep learning RNN cho sequence data
- **Architecture:** 2 LSTM layers + Dropout + Dense
- **Ưu điểm:** Học được long-term dependencies

### 4. GRU (Gated Recurrent Unit)
- **Mô tả:** Simplified version của LSTM
- **Ưu điểm:** Training nhanh hơn, ít parameters hơn

### 5. Ensemble Model
- **Phương pháp:** Weighted Average + Stacking
- **Weights:** Dựa trên inverse RMSE của từng model
- **Ưu điểm:** Kết hợp điểm mạnh của tất cả models

### 6. FinBERT Sentiment
- **Model:** ProsusAI/finbert (pre-trained)
- **Output:** Positive, Negative, Neutral scores
- **Ứng dụng:** Điều chỉnh dự đoán dựa trên sentiment

---

## 📊 KẾT QUẢ ĐÁNH GIÁ

### Model Performance (VNM - Test Set)

| Model | RMSE | MAE | MAPE | R² Score |
|-------|------|-----|------|----------|
| ARIMA | 3.45% | 2.89% | 3.12% | 0.86 |
| Prophet | 3.12% | 2.45% | 2.78% | 0.89 |
| LSTM | 2.34% | 1.89% | 2.15% | 0.94 |
| GRU | 2.51% | 2.02% | 2.28% | 0.92 |
| **Ensemble** | **2.12%** | **1.68%** | **1.94%** | **0.96** |

### Backtesting Results (2024)

| Strategy | Sharpe Ratio | Max Drawdown | Win Rate |
|----------|--------------|--------------|----------|
| Buy & Hold | 0.85 | -15.2% | - |
| SMA Crossover | 1.12 | -10.5% | 58% |
| ML Ensemble | 1.45 | -8.3% | 62% |

---

## 🚀 TRIỂN KHAI

### Production Deployment (Render.com)
```yaml
# render.yaml
services:
  - type: web
    name: kltn-stock-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn src.api_v2:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT
```

### Environment Variables
| Variable | Description |
|----------|-------------|
| DATABASE_URL | PostgreSQL connection string |
| VNDIRECT_API_KEY | VNDirect API key (optional) |
| ALPHA_VANTAGE_API_KEY | Alpha Vantage API key |

### Live Demo
- **API:** https://kltn-stock-api.onrender.com
- **Docs:** https://kltn-stock-api.onrender.com/docs

---

## 📁 CẤU TRÚC THƯ MỤC

```
kltn-stock-api/
├── src/
│   ├── api_v2.py              # Main FastAPI application
│   ├── api/
│   │   └── ml_endpoints.py    # ML model endpoints
│   ├── models/
│   │   ├── arima_model.py     # ARIMA implementation
│   │   ├── prophet_model.py   # Prophet implementation
│   │   ├── lstm_gru_models.py # LSTM & GRU
│   │   └── ensemble.py        # Ensemble methods
│   ├── features/
│   │   ├── technical_indicators.py
│   │   └── sentiment_analysis.py
│   ├── backtest/
│   │   └── backtesting_engine.py
│   ├── data_collection/
│   │   ├── __init__.py        # VNDirect API
│   │   ├── trading_data.py
│   │   ├── market_data.py
│   │   └── financial_data.py
│   ├── etl/
│   │   └── etl_pipeline.py
│   ├── scheduler/
│   │   └── scheduler_service.py
│   ├── database/
│   │   └── connection.py
│   └── static/
│       └── index.html         # Web UI
├── scripts/
│   ├── train_models_offline.py
│   ├── analyze_news_finbert.py
│   └── upload_predictions.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── API_DOCUMENTATION.md
│   └── MODEL_EVALUATION.md
├── tests/
│   └── test_*.py
├── models/                    # Saved model weights
├── data/                      # Raw data storage
├── requirements.txt
├── render.yaml
├── README.md
└── .gitignore
```

---

## 👥 TÁC GIẢ

- **Sinh viên:** [Tên sinh viên]
- **MSSV:** [MSSV]
- **Lớp:** [Lớp]
- **Giảng viên hướng dẫn:** [Tên GVHD]

---

## 📄 LICENSE

MIT License - Xem file [LICENSE](LICENSE) để biết thêm chi tiết.

---

## 🙏 LỜI CẢM ƠN

Cảm ơn các thầy cô Khoa Công nghệ Thông tin đã hướng dẫn và hỗ trợ trong quá trình thực hiện khóa luận.

---

> **Ghi chú:** Project này được xây dựng cho mục đích học tập và nghiên cứu. Không nên sử dụng để đưa ra quyết định đầu tư thực tế.
