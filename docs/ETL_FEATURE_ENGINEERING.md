# ETL Pipeline & Feature Engineering

## 📊 Tổng quan Kiến trúc

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DATA PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   EXTRACT    │───▶│  TRANSFORM   │───▶│     LOAD     │───▶│  FEATURE  │ │
│  │   (Trích     │    │  (Biến đổi)  │    │   (Tải lên)  │    │ ENGINEER  │ │
│  │    xuất)     │    │              │    │              │    │           │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│        │                   │                   │                   │        │
│        ▼                   ▼                   ▼                   ▼        │
│  ╔════════════╗      ╔════════════╗      ╔════════════╗      ╔══════════╗ │
│  ║ VNDirect   ║      ║ Validate   ║      ║ SQLite DB  ║      ║Technical ║ │
│  ║ API        ║      ║ Clean      ║      ║ CSV Files  ║      ║Indicators║ │
│  ║ Fireant    ║      ║ Normalize  ║      ║ Cloud S3   ║      ║FinBERT   ║ │
│  ║ Web Scrape ║      ║ Deduplicate║      ║            ║      ║Sentiment ║ │
│  ╚════════════╝      ╚════════════╝      ╚════════════╝      ╚══════════╝ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. EXTRACT - Trích xuất dữ liệu

### 1.1 Module kết nối API (VNDirect, Fireant)

**File:** `src/data_collection/vndirect_api.py`

```python
# Kết nối VNDirect API lấy dữ liệu OHLCV
class VNDirectAPI:
    BASE_URL = "https://finfo-api.vndirect.com.vn"
    
    def get_stock_price(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giá lịch sử từ VNDirect
        
        Returns:
            DataFrame với các cột: date, Open, High, Low, Close, Volume
        """
```

**Endpoints được sử dụng:**
| API | Endpoint | Mục đích |
|-----|----------|----------|
| VNDirect | `/v4/stock_prices` | Dữ liệu giá OHLCV |
| VNDirect | `/v4/stocks` | Thông tin cổ phiếu |
| VNDirect | `/v4/industry_classification` | Phân ngành |

### 1.2 Web Scraping Module (BeautifulSoup)

**File:** `src/news_service.py`

```python
# Scraping tin tức từ các nguồn
class NewsCollector:
    SOURCES = [
        "cafef.vn",
        "vnexpress.net/kinh-doanh",
        "fireant.vn"
    ]
    
    def scrape_news(self, symbol: str) -> List[Dict]:
        """
        Quét tin tức liên quan đến mã cổ phiếu
        
        Returns:
            List of {title, content, url, published_at, source}
        """
```

---

## 2. TRANSFORM - Biến đổi dữ liệu

### 2.1 Data Validation

**File:** `src/etl/etl_pipeline.py` - Class `DataValidator`

| Check | Mô tả | Ngưỡng |
|-------|-------|--------|
| Missing Values | % dữ liệu bị thiếu | ≤ 5% |
| Duplicates | Trùng lặp theo date | 0 |
| OHLC Relationship | High ≥ Low, etc. | Valid |
| Price Range | Giá trong khoảng hợp lệ | 100 - 1,000,000 VND |
| Daily Change | Thay đổi giá trong ngày | ≤ 30% |
| Negative Values | Giá âm không được phép | 0 |

### 2.2 Data Cleaning

**File:** `src/etl/etl_pipeline.py` - Class `DataTransformer`

```python
class DataTransformer:
    """Transform pipeline bao gồm:"""
    
    def standardize_columns(df):
        """Chuẩn hóa tên cột: DATE→date, CLOSE→Close, etc."""
    
    def clean_missing_values(df, method='ffill'):
        """Xử lý missing: forward fill, interpolate, hoặc drop"""
    
    def remove_duplicates(df, keep='last'):
        """Loại bỏ dữ liệu trùng lặp"""
    
    def fix_ohlc_relationship(df):
        """Sửa High/Low không hợp lệ"""
    
    def convert_date(df):
        """Chuyển đổi date sang datetime"""
```

### 2.3 Data Normalization

```python
# Chuẩn hóa giá để training ML models
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# MinMaxScaler cho LSTM/GRU
scaler = MinMaxScaler(feature_range=(0, 1))
normalized_prices = scaler.fit_transform(prices)

# StandardScaler cho các models khác
standard_scaler = StandardScaler()
standardized_features = standard_scaler.fit_transform(features)
```

---

## 3. LOAD - Tải dữ liệu

### 3.1 Database Storage (SQLite)

**File:** `src/database/models.py`

```python
class StockPrice(Base):
    __tablename__ = 'stock_prices'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(BigInteger)
    
    # Technical Indicators
    sma_20 = Column(Float)
    rsi = Column(Float)
    macd = Column(Float)
```

### 3.2 Raw Data Storage (CSV/Cloud)

```python
class CSVLoader(Loader):
    """Lưu dữ liệu thô vào file CSV"""
    
    def load(self, df: pd.DataFrame, symbol: str) -> int:
        filename = f"./data/raw/{symbol}_{date}.csv"
        df.to_csv(filename, index=False)
```

**Cấu trúc thư mục lưu trữ:**
```
data/
├── raw/                    # Dữ liệu thô chưa xử lý
│   ├── VNM_20241202.csv
│   ├── VIC_20241202.csv
│   └── ...
├── processed/              # Dữ liệu đã transform
│   ├── VNM_features.csv
│   └── ...
├── models/                 # Saved models
│   ├── lstm_VNM.h5
│   └── ...
└── predictions/            # Kết quả dự đoán
    └── predictions_latest.json
```

---

## 4. FEATURE ENGINEERING - Xây dựng Đặc trưng

### 4.1 Technical Indicators Module

**File:** `src/features/technical_indicators.py`

| Indicator | Công thức | Ý nghĩa |
|-----------|-----------|---------|
| **SMA** | $SMA_n = \frac{1}{n}\sum_{i=1}^{n} P_i$ | Xu hướng trung bình |
| **EMA** | $EMA_t = \alpha \cdot P_t + (1-\alpha) \cdot EMA_{t-1}$ | Xu hướng có trọng số |
| **RSI** | $RSI = 100 - \frac{100}{1 + RS}$ | Quá mua/quá bán |
| **MACD** | $MACD = EMA_{12} - EMA_{26}$ | Động lượng |
| **Bollinger** | $BB = SMA \pm 2\sigma$ | Volatility bands |
| **ATR** | $ATR = \frac{1}{n}\sum TR$ | Biến động |
| **Stochastic** | $\%K = \frac{C - L_{14}}{H_{14} - L_{14}} \times 100$ | Momentum |

**Code tính toán:**
```python
class TechnicalIndicators:
    def add_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Thêm 20+ technical indicators vào DataFrame"""
        
        # Moving Averages
        df['sma_5'] = df['close'].rolling(5).mean()
        df['sma_20'] = df['close'].rolling(20).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        df['rsi'] = 100 - (100 / (1 + gain/loss))
        
        # MACD
        df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        df['bb_upper'] = df['bb_middle'] + 2 * df['close'].rolling(20).std()
        df['bb_lower'] = df['bb_middle'] - 2 * df['close'].rolling(20).std()
        
        return df
```

### 4.2 Sentiment Analysis Module (FinBERT)

**File:** `src/features/sentiment_analysis.py`

```python
class FinBERTSentimentAnalyzer:
    """
    Phân tích cảm tính tin tức tài chính sử dụng FinBERT
    Model: ProsusAI/finbert (fine-tuned BERT cho financial domain)
    """
    
    def predict_sentiment(self, text: str) -> Dict:
        """
        Returns:
            {
                'positive': 0.85,   # Xác suất tích cực
                'negative': 0.05,   # Xác suất tiêu cực
                'neutral': 0.10,    # Xác suất trung lập
                'sentiment': 'positive',
                'score': 0.85
            }
        """
```

**Aggregation theo ngày:**
```python
class SentimentAggregator:
    def aggregate_by_date(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Tổng hợp sentiment score theo ngày cho từng mã cổ phiếu
        
        Output columns:
        - daily_sentiment: (positive - negative) score
        - sentiment_ma_3: Moving average 3 ngày
        - sentiment_ma_7: Moving average 7 ngày
        - sentiment_momentum: Thay đổi sentiment
        - news_count: Số tin tức trong ngày
        """
```

---

## 5. Complete ETL Pipeline

### 5.1 Full Pipeline Flow

```python
from src.etl.etl_pipeline import ETLPipeline, run_etl_for_symbol

# Khởi tạo pipeline
pipeline = ETLPipeline(
    extractor=VNDirectExtractor(),
    loader=DatabaseLoader(db_session),
    validator=DataValidator(max_missing_pct=0.05),
    transformer=DataTransformer()
)

# Chạy ETL cho một mã
result = pipeline.run(
    symbol='VNM',
    start_date='2024-01-01',
    end_date='2024-12-01',
    validate=True,
    add_features=True
)

print(result.to_dict())
# {
#     'success': True,
#     'symbol': 'VNM',
#     'records_extracted': 250,
#     'records_transformed': 248,
#     'records_loaded': 248,
#     'records_skipped': 2,
#     'duration_seconds': 3.45,
#     'validation': {
#         'is_valid': True,
#         'status': 'valid',
#         'stats': {'missing_pct': 0.5, 'duplicates': 0}
#     }
# }
```

### 5.2 Batch Processing

```python
# Chạy ETL cho tất cả cổ phiếu
symbols = ['VNM', 'VIC', 'VHM', 'HPG', 'FPT', ...]

results = pipeline.run_batch(
    symbols=symbols,
    start_date='2024-01-01',
    end_date='2024-12-01'
)

# Summary
summary = pipeline.get_summary()
# {
#     'total_runs': 30,
#     'successful': 28,
#     'failed': 2,
#     'total_extracted': 7500,
#     'total_loaded': 7420,
#     'avg_duration_seconds': 2.8
# }
```

### 5.3 Incremental ETL (Daily Update)

```python
from src.etl.etl_pipeline import IncrementalETL

# Chỉ load dữ liệu mới từ ngày cuối cùng
incremental = IncrementalETL(pipeline, db_session)
result = incremental.run_incremental('VNM')

# Tự động detect ngày cuối cùng trong DB và chỉ fetch dữ liệu mới
```

---

## 6. API Endpoints cho ETL

| Endpoint | Method | Mô tả |
|----------|--------|-------|
| `/api/data/fetch/{symbol}` | POST | Fetch dữ liệu cho 1 mã |
| `/api/data/fetch-all` | POST | Fetch tất cả cổ phiếu |
| `/api/etl/run/{symbol}` | POST | Chạy full ETL pipeline |
| `/api/etl/status` | GET | Xem trạng thái ETL |
| `/api/features/{symbol}` | GET | Lấy features đã tính |

---

## 7. Scheduled Jobs (Cron)

**File:** `src/scheduler/jobs.py`

```python
# Chạy tự động hàng ngày lúc 18:00 (sau giờ đóng cửa)
schedule.every().day.at("18:00").do(run_daily_etl)

# Chạy mỗi tuần để re-calculate indicators
schedule.every().monday.at("07:00").do(recalculate_features)

# Chạy mỗi giờ để cập nhật tin tức
schedule.every().hour.do(fetch_latest_news)
```

---

## 8. Data Quality Metrics

| Metric | Mục tiêu | Thực tế |
|--------|----------|---------|
| Missing Rate | < 5% | ~0.5% |
| Duplicate Rate | 0% | 0% |
| OHLC Validity | 100% | 99.8% |
| Data Freshness | T+1 | T+1 |
| API Success Rate | > 95% | 97.3% |

---

## 9. Files & Structure

```
src/
├── etl/
│   ├── __init__.py
│   └── etl_pipeline.py          # Main ETL classes
├── features/
│   ├── __init__.py
│   ├── technical_indicators.py  # MACD, RSI, BB, etc.
│   └── sentiment_analysis.py    # FinBERT sentiment
├── data_collection/
│   ├── __init__.py
│   └── vndirect_api.py          # API connectors
├── database/
│   ├── __init__.py
│   ├── models.py                # SQLAlchemy models
│   └── connection.py            # DB connection
└── scheduler/
    ├── __init__.py
    └── jobs.py                  # Scheduled tasks
```

---

## 10. Usage Examples

### Example 1: Full Pipeline

```python
# 1. Extract từ VNDirect
df_raw = extractor.extract('VNM', '2024-01-01', '2024-12-01')

# 2. Validate
validation = validator.validate(df_raw)
print(validation.to_dict())

# 3. Transform
df_clean = transformer.transform(df_raw, add_features=True)

# 4. Add Technical Indicators
ti = TechnicalIndicators()
df_features = ti.add_all_indicators(df_clean)

# 5. Load to Database
loader.load(df_features, 'VNM')
```

### Example 2: With Sentiment

```python
# 1. Collect news
news_df = news_collector.get_news('VNM')

# 2. Analyze sentiment
analyzer = FinBERTSentimentAnalyzer()
news_with_sentiment = analyzer.analyze_news_dataframe(news_df)

# 3. Aggregate by date
aggregator = SentimentAggregator()
daily_sentiment = aggregator.aggregate_by_date(news_with_sentiment)

# 4. Merge với price data
df_final = pd.merge(df_features, daily_sentiment, on=['date', 'symbol'])
```

---

*Document này mô tả đầy đủ ETL Pipeline và Feature Engineering modules trong dự án KLTN Stock Prediction.*
