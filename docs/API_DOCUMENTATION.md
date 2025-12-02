# 📡 API DOCUMENTATION

## Tổng quan

Hệ thống cung cấp RESTful API để truy cập các chức năng dự đoán giá cổ phiếu, phân tích kỹ thuật, và phân tích cảm tính.

**Base URL:**
```
https://kltn-stock-api.onrender.com/api
```

**Documentation:**
- Swagger UI: https://kltn-stock-api.onrender.com/docs
- ReDoc: https://kltn-stock-api.onrender.com/redoc

---

## 🔐 Authentication

Hiện tại API không yêu cầu authentication. Trong tương lai sẽ hỗ trợ:
- API Key
- JWT Token

---

## 📊 Response Format

### Success Response
```json
{
    "status": "success",
    "data": { ... },
    "message": "Optional message"
}
```

### Error Response
```json
{
    "detail": "Error message",
    "status_code": 400
}
```

---

## 📈 Stock Endpoints

### GET /api/stocks
Lấy danh sách tất cả cổ phiếu

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| limit | int | No | Số lượng tối đa (default: 100) |
| active_only | bool | No | Chỉ lấy mã đang active (default: true) |

**Response:**
```json
[
    {
        "id": 1,
        "symbol": "VNM",
        "name": "Công ty Cổ phần Sữa Việt Nam",
        "sector": "Consumer Goods",
        "exchange": "HOSE",
        "is_active": true
    }
]
```

**Example:**
```bash
curl https://kltn-stock-api.onrender.com/api/stocks?limit=10
```

---

### GET /api/stocks/{symbol}
Lấy thông tin chi tiết một cổ phiếu

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu (VNM, FPT, ...) |

**Response:**
```json
{
    "id": 1,
    "symbol": "VNM",
    "name": "Công ty Cổ phần Sữa Việt Nam",
    "sector": "Consumer Goods",
    "exchange": "HOSE",
    "is_active": true,
    "created_at": "2024-01-01T00:00:00"
}
```

---

## 💰 Price Endpoints

### GET /api/prices/{symbol}
Lấy lịch sử giá của cổ phiếu

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| limit | int | No | Số ngày (default: 30) |
| start_date | string | No | Ngày bắt đầu (YYYY-MM-DD) |
| end_date | string | No | Ngày kết thúc (YYYY-MM-DD) |

**Response:**
```json
[
    {
        "date": "2024-11-28",
        "open": 75.5,
        "high": 76.2,
        "low": 75.0,
        "close": 76.0,
        "volume": 1234567
    }
]
```

**Example:**
```bash
curl "https://kltn-stock-api.onrender.com/api/prices/VNM?limit=7"
```

---

### GET /api/prices/{symbol}/latest
Lấy giá mới nhất

**Response:**
```json
{
    "symbol": "VNM",
    "date": "2024-11-28",
    "open": 75.5,
    "high": 76.2,
    "low": 75.0,
    "close": 76.0,
    "volume": 1234567,
    "change": 0.5,
    "change_percent": 0.66
}
```

---

## 🤖 ML Prediction Endpoints

### POST /api/ml/arima/predict/{symbol}
Dự đoán giá sử dụng ARIMA

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày dự đoán (default: 7) |

**Response:**
```json
{
    "symbol": "VNM",
    "model": "ARIMA",
    "predictions": [
        {"date": "2024-11-29", "price": 76.5, "confidence": 0.85},
        {"date": "2024-11-30", "price": 76.8, "confidence": 0.82}
    ],
    "metrics": {
        "rmse": 3.45,
        "mae": 2.89
    }
}
```

---

### POST /api/ml/prophet/predict/{symbol}
Dự đoán giá sử dụng Prophet

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày dự đoán (default: 7) |

**Response:**
```json
{
    "symbol": "VNM",
    "model": "Prophet",
    "predictions": [
        {
            "date": "2024-11-29",
            "price": 76.3,
            "yhat_lower": 74.5,
            "yhat_upper": 78.1
        }
    ],
    "components": {
        "trend": "up",
        "weekly": 0.5,
        "yearly": -0.2
    }
}
```

---

### POST /api/ml/lstm/predict/{symbol}
Dự đoán giá sử dụng LSTM

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày dự đoán (default: 7) |

---

### POST /api/ml/gru/predict/{symbol}
Dự đoán giá sử dụng GRU

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày dự đoán (default: 7) |

---

### POST /api/ml/ensemble/predict/{symbol}
Dự đoán giá sử dụng Ensemble (kết hợp tất cả models)

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày dự đoán (default: 7) |
| include_sentiment | bool | No | Kết hợp sentiment (default: true) |

**Response:**
```json
{
    "symbol": "VNM",
    "model": "Ensemble",
    "predictions": [
        {"date": "2024-11-29", "price": 76.4, "confidence": 0.89}
    ],
    "model_weights": {
        "arima": 0.18,
        "prophet": 0.20,
        "lstm": 0.32,
        "gru": 0.30
    },
    "sentiment_adjustment": 0.02,
    "recommendation": "BUY",
    "reasoning": [
        "Technical indicators show uptrend",
        "Positive news sentiment"
    ]
}
```

---

### GET /api/ml/compare/{symbol}
So sánh kết quả dự đoán của tất cả models

**Response:**
```json
{
    "symbol": "VNM",
    "comparison": {
        "arima": {"prediction": 76.5, "rmse": 3.45},
        "prophet": {"prediction": 76.3, "rmse": 3.12},
        "lstm": {"prediction": 76.8, "rmse": 2.34},
        "gru": {"prediction": 76.6, "rmse": 2.51},
        "ensemble": {"prediction": 76.4, "rmse": 2.12}
    },
    "best_model": "ensemble"
}
```

---

### GET /api/ml/status
Kiểm tra trạng thái các models

**Response:**
```json
{
    "status": "healthy",
    "models": {
        "arima": {"available": true, "last_trained": "2024-11-28"},
        "prophet": {"available": true, "last_trained": "2024-11-28"},
        "lstm": {"available": true, "weights_loaded": true},
        "gru": {"available": true, "weights_loaded": true},
        "finbert": {"available": true, "model": "ProsusAI/finbert"}
    }
}
```

---

## 📉 Backtesting Endpoints

### POST /api/ml/backtest/{symbol}
Chạy backtest với ML models

**Request Body:**
```json
{
    "symbol": "VNM",
    "start_date": "2024-01-01",
    "end_date": "2024-11-28",
    "initial_capital": 100000000,
    "model": "ensemble",
    "strategy": "ml_signal"
}
```

**Response:**
```json
{
    "symbol": "VNM",
    "period": {
        "start": "2024-01-01",
        "end": "2024-11-28",
        "trading_days": 230
    },
    "metrics": {
        "total_return": 0.245,
        "sharpe_ratio": 1.45,
        "sortino_ratio": 1.78,
        "max_drawdown": -0.083,
        "win_rate": 0.62,
        "profit_factor": 1.82,
        "total_trades": 38
    },
    "trades": [
        {
            "date": "2024-01-15",
            "action": "BUY",
            "price": 72.5,
            "shares": 1000,
            "profit": null
        }
    ],
    "equity_curve": [100000000, 101500000, ...]
}
```

---

### POST /api/backtest/advanced
Backtest nâng cao với nhiều chiến lược

**Request Body:**
```json
{
    "symbol": "VNM",
    "start_date": "2024-01-01",
    "end_date": "2024-11-28",
    "initial_capital": 100000000,
    "strategy": "sma_crossover",
    "stop_loss_pct": 0.05,
    "take_profit_pct": 0.10
}
```

---

## 📰 News & Sentiment Endpoints

### GET /api/news
Lấy tin tức thị trường

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| limit | int | No | Số tin (default: 20) |

---

### GET /api/news/{symbol}
Lấy tin tức của một cổ phiếu

**Response:**
```json
{
    "status": "success",
    "news": [
        {
            "title": "VNM công bố lợi nhuận quý 3",
            "summary": "Vinamilk báo cáo...",
            "source": "CafeF",
            "published_at": "2024-11-28 10:30",
            "url": "https://...",
            "sentiment": "positive",
            "sentiment_score": 0.75,
            "impact": "Tin tức tích cực có thể hỗ trợ giá cổ phiếu"
        }
    ],
    "sentiment_summary": {
        "overall": "positive",
        "positive_count": 5,
        "negative_count": 1,
        "neutral_count": 3,
        "avg_score": 0.45,
        "recommendation": "Xu hướng tin tức tích cực"
    }
}
```

---

### GET /api/finbert/sentiment/{symbol}
Phân tích sentiment bằng FinBERT

**Response:**
```json
{
    "status": "ok",
    "symbol": "VNM",
    "sentiment_summary": {
        "positive_count": 8,
        "negative_count": 2,
        "neutral_count": 5,
        "avg_score": 0.35,
        "overall": "positive",
        "recommendation": "Sentiment tích cực, có thể cân nhắc mua"
    },
    "recent_news": [
        {
            "title": "...",
            "finbert_sentiment": "positive",
            "finbert_score": 0.89
        }
    ]
}
```

---

## 📊 Technical Indicators Endpoints

### GET /api/indicators/{symbol}
Lấy các chỉ báo kỹ thuật

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| limit | int | No | Số ngày (default: 30) |

**Response:**
```json
[
    {
        "date": "2024-11-28",
        "sma_10": 75.5,
        "sma_20": 74.8,
        "sma_50": 73.2,
        "rsi": 58.5,
        "macd": 0.45,
        "macd_signal": 0.32,
        "macd_histogram": 0.13,
        "bollinger_upper": 78.2,
        "bollinger_middle": 75.0,
        "bollinger_lower": 71.8
    }
]
```

---

### GET /api/indicators/{symbol}/latest
Lấy chỉ báo kỹ thuật mới nhất

---

## 🔄 Data Collection Endpoints

### POST /api/data/fetch/{symbol}
Thu thập dữ liệu cho một cổ phiếu

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| symbol | string | Yes | Mã cổ phiếu |
| days | int | No | Số ngày (default: 365) |
| from_date | string | No | Từ ngày (YYYY-MM-DD) |
| to_date | string | No | Đến ngày (YYYY-MM-DD) |

**Response:**
```json
{
    "status": "success",
    "symbol": "VNM",
    "records_added": 250,
    "records_updated": 5,
    "date_range": {
        "from": "2024-01-01",
        "to": "2024-11-28"
    }
}
```

---

### POST /api/data/fetch-all
Thu thập dữ liệu cho tất cả cổ phiếu

---

### POST /api/data/sync-daily
Đồng bộ dữ liệu mới nhất

---

### GET /api/data/status
Kiểm tra trạng thái dữ liệu

**Response:**
```json
{
    "status": "ok",
    "summary": {
        "total_stocks": 30,
        "needs_sync": 2,
        "up_to_date": 28
    },
    "stocks": [
        {
            "symbol": "VNM",
            "last_date": "2024-11-28",
            "total_records": 365,
            "needs_sync": false
        }
    ]
}
```

---

## 🛠 Admin Endpoints

### POST /api/admin/init-db
Khởi tạo database với VN30 stocks

### GET /api/admin/db-status
Kiểm tra trạng thái database

### DELETE /api/admin/reset-db
Reset database (⚠️ Cẩn thận!)

---

## 📊 Statistics Endpoints

### GET /api/stats/overview
Thống kê tổng quan

**Response:**
```json
{
    "stocks": {
        "total": 30,
        "active": 30
    },
    "price_records": 10950,
    "predictions": 1500,
    "latest_update": "2024-11-28T15:30:00"
}
```

---

## ⚙️ Rate Limiting

| Tier | Requests/Minute | Requests/Day |
|------|-----------------|--------------|
| Free | 100 | 10,000 |

---

## 🔗 WebSocket (Future)

```
ws://kltn-stock-api.onrender.com/ws/prices/{symbol}
```

---

## 📝 Error Codes

| Code | Description |
|------|-------------|
| 400 | Bad Request - Invalid parameters |
| 404 | Not Found - Resource doesn't exist |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |

---

## 💡 Examples

### Python
```python
import requests

# Get stock prices
response = requests.get(
    "https://kltn-stock-api.onrender.com/api/prices/VNM",
    params={"limit": 30}
)
prices = response.json()

# Make prediction
response = requests.post(
    "https://kltn-stock-api.onrender.com/api/ml/ensemble/predict/VNM",
    params={"days": 7}
)
prediction = response.json()
```

### JavaScript
```javascript
// Get stock prices
fetch('https://kltn-stock-api.onrender.com/api/prices/VNM?limit=30')
    .then(res => res.json())
    .then(data => console.log(data));

// Make prediction
fetch('https://kltn-stock-api.onrender.com/api/ml/ensemble/predict/VNM?days=7', {
    method: 'POST'
})
    .then(res => res.json())
    .then(data => console.log(data));
```

### cURL
```bash
# Get stocks
curl https://kltn-stock-api.onrender.com/api/stocks

# Get prices
curl "https://kltn-stock-api.onrender.com/api/prices/VNM?limit=30"

# Make prediction
curl -X POST "https://kltn-stock-api.onrender.com/api/ml/ensemble/predict/VNM?days=7"

# Run backtest
curl -X POST "https://kltn-stock-api.onrender.com/api/ml/backtest/VNM" \
    -H "Content-Type: application/json" \
    -d '{"start_date": "2024-01-01", "end_date": "2024-11-28"}'
```

---

## 📚 SDKs (Future)

- Python SDK: `pip install kltn-stock-api`
- JavaScript SDK: `npm install kltn-stock-api`

---

*API Version: 2.0.0 | Last Updated: December 2025*
