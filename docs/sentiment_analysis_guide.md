# 📊 Module Phân tích Cảm tính (Sentiment Analysis) - Hướng dẫn Chi tiết

## 🎯 Tổng quan

Module Phân tích Cảm tính sử dụng **PhoBERT** (Vietnamese BERT) để phân tích cảm xúc từ tin tức tiếng Việt về cổ phiếu, sau đó tổng hợp điểm cảm tính theo ngày cho từng mã.

---

## 🏗️ Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENTIMENT ANALYSIS PIPELINE                   │
└─────────────────────────────────────────────────────────────────┘

   📰 TIN TỨC             🤖 PHÂN TÍCH           💾 LƯU TRỮ
   ────────────           ────────────           ─────────
                                                             
   news_articles     ┌──────────────────┐     analyzed_news
   ┌──────────┐     │   PhoBERT Model   │     ┌──────────┐
   │ symbol   │────▶│   wonrax/phobert  │────▶│ sentiment│
   │ title    │     │   vietnamese-     │     │ scores   │
   │ summary  │     │   sentiment       │     │ positive │
   │ content  │     └──────────────────┘     │ negative │
   │published │            │                  │ neutral  │
   └──────────┘            │                  └──────────┘
                           │                       │
                           ▼                       │
                  ┌────────────────┐              │
                  │ Sentiment      │              │
                  │ Classification │              │
                  │ • Positive     │              │
                  │ • Negative     │              │
                  │ • Neutral      │              │
                  └────────────────┘              │
                                                  │
                                                  ▼
                                    ┌──────────────────────┐
                                    │  sentiment_summary   │
                                    │  (Tổng hợp theo ngày)│
                                    ├──────────────────────┤
                                    │ symbol | date        │
                                    │ positive_count       │
                                    │ negative_count       │
                                    │ neutral_count        │
                                    │ avg_score            │
                                    │ overall_sentiment    │
                                    └──────────────────────┘
```

---

## 📁 Cấu trúc Files

```
D:\KLTN\
├── scripts/
│   └── analyze_news_finbert.py        # 🚀 Script chính chạy phân tích
│
├── src/
│   ├── sentiment/
│   │   ├── __init__.py
│   │   └── finbert_analyzer.py        # 🧠 Module FinBERT analyzer
│   │
│   ├── database/
│   │   └── models.py                  # 📊 Database models
│   │                                     - AnalyzedNews
│   │                                     - SentimentSummary
│   │
│   └── api_v2.py                      # 🌐 API endpoints
│                                         GET /api/finbert/sentiment/{symbol}
│                                         GET /api/finbert/summary
│                                         GET /api/finbert/status
│
└── docs/
    └── sentiment_analysis_guide.md    # 📖 Tài liệu này
```

---

## 🔧 Cách Hoạt động Chi tiết

### **BƯỚC 1: Thu thập Tin tức** 📰

**File**: `scripts/analyze_news_finbert.py` - Class `NewsFetcher`

```python
class NewsFetcher:
    def fetch_all(self, symbols=None):
        # Lấy tin tức từ RSS feeds:
        # 1. CafeF RSS
        # 2. VnExpress RSS
        # 3. Database (news_articles table)
        
        # Filter theo:
        # - symbols (mã cổ phiếu)
        # - days (số ngày gần đây)
        
        return news_list  # List[Dict]
```

**Output**: Danh sách tin tức
```python
[
    {
        'symbol': 'VNM',
        'title': 'Vinamilk báo cáo lợi nhuận quý 3 tăng 25%',
        'summary': 'Công ty CP Sữa Việt Nam Vinamilk công bố...',
        'url': 'https://...',
        'source': 'CafeF',
        'published_at': datetime(2024, 11, 15, 10, 30)
    },
    ...
]
```

---

### **BƯỚC 2: Phân tích Cảm tính bằng PhoBERT** 🤖

**File**: `scripts/analyze_news_finbert.py` - Class `VietnameseSentimentAnalyzer`

**Model sử dụng**: `wonrax/phobert-base-vietnamese-sentiment`
- Fine-tuned PhoBERT cho sentiment analysis tiếng Việt
- Output: 3 classes (Positive, Negative, Neutral)

**Quy trình phân tích**:

```python
class VietnameseSentimentAnalyzer:
    def analyze(self, text: str) -> Dict:
        # 1. Tokenize văn bản tiếng Việt
        inputs = self.tokenizer(
            text[:256],  # PhoBERT max 256 tokens
            return_tensors="pt",
            truncation=True
        )
        
        # 2. Đưa vào PhoBERT model
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)[0]
        
        # 3. Lấy xác suất cho mỗi class
        neg_score = probs[0].item()  # Negative
        pos_score = probs[1].item()  # Positive
        neu_score = probs[2].item()  # Neutral
        
        # 4. Tính sentiment tổng thể (-1 đến +1)
        overall_score = pos_score - neg_score
        
        # 5. Xác định sentiment chính
        sentiment = max({'negative': neg_score, 
                        'positive': pos_score, 
                        'neutral': neu_score}, 
                       key=scores.get)
        
        return {
            'sentiment': sentiment,      # 'positive'/'negative'/'neutral'
            'score': overall_score,      # -1.0 đến +1.0
            'positive': pos_score,       # 0.0 đến 1.0
            'negative': neg_score,       # 0.0 đến 1.0
            'neutral': neu_score         # 0.0 đến 1.0
        }
```

**Ví dụ Input/Output**:

| Input Text | Sentiment | Score | Positive | Negative | Neutral |
|------------|-----------|-------|----------|----------|---------|
| "Vinamilk báo lãi tăng 25%" | positive | +0.78 | 0.89 | 0.05 | 0.06 |
| "HPG sụt giảm doanh thu" | negative | -0.65 | 0.10 | 0.82 | 0.08 |
| "VNIndex đi ngang" | neutral | +0.05 | 0.35 | 0.30 | 0.35 |

---

### **BƯỚC 3: Lưu vào Database** 💾

**File**: `scripts/analyze_news_finbert.py` - Class `DatabaseManager`

**Table 1: `analyzed_news`** (Lưu từng tin tức đã phân tích)

```sql
CREATE TABLE analyzed_news (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),              -- Mã cổ phiếu (VNM, HPG, ...)
    title TEXT NOT NULL,             -- Tiêu đề tin tức
    summary TEXT,                    -- Tóm tắt
    url TEXT UNIQUE,                 -- Link tin tức
    source VARCHAR(100),             -- Nguồn (CafeF, VnExpress, ...)
    published_at TIMESTAMP,          -- Ngày xuất bản
    
    -- Kết quả phân tích sentiment
    sentiment VARCHAR(20),           -- 'positive', 'negative', 'neutral'
    sentiment_score FLOAT,           -- -1.0 đến +1.0
    positive_score FLOAT,            -- Xác suất positive (0-1)
    negative_score FLOAT,            -- Xác suất negative (0-1)
    neutral_score FLOAT,             -- Xác suất neutral (0-1)
    
    -- Metadata
    analyzed_at TIMESTAMP DEFAULT NOW(),
    model_version VARCHAR(50) DEFAULT 'phobert-v1'
);

CREATE INDEX idx_symbol ON analyzed_news(symbol);
CREATE INDEX idx_published ON analyzed_news(published_at);
```

**Insert dữ liệu**:
```python
db.save_news({
    'symbol': 'VNM',
    'title': 'Vinamilk báo lãi tăng 25%',
    'sentiment': 'positive',
    'sentiment_score': 0.78,
    'positive_score': 0.89,
    'negative_score': 0.05,
    'neutral_score': 0.06,
    'published_at': datetime.now()
})
```

---

### **BƯỚC 4: Tổng hợp Sentiment theo Ngày** 📊

**File**: `scripts/analyze_news_finbert.py` - Method `update_summary()`

**Table 2: `sentiment_summary`** (Tổng hợp theo ngày cho từng mã)

```sql
CREATE TABLE sentiment_summary (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    
    -- Đếm số lượng tin tức theo sentiment
    positive_count INT DEFAULT 0,    -- Số tin positive
    negative_count INT DEFAULT 0,    -- Số tin negative
    neutral_count INT DEFAULT 0,     -- Số tin neutral
    
    -- Điểm trung bình
    avg_score FLOAT DEFAULT 0,       -- Sentiment score trung bình (-1 đến +1)
    overall_sentiment VARCHAR(20),   -- Sentiment tổng thể ngày đó
    news_count INT DEFAULT 0,        -- Tổng số tin trong ngày
    
    updated_at TIMESTAMP DEFAULT NOW(),
    
    UNIQUE(symbol, date)
);

CREATE INDEX idx_summary_symbol_date ON sentiment_summary(symbol, date);
```

**Logic tổng hợp**:
```python
def update_summary(symbol: str, date: date):
    # 1. Đếm số lượng tin theo sentiment
    SELECT 
        COUNT(*) FILTER (WHERE sentiment = 'positive') as positive_count,
        COUNT(*) FILTER (WHERE sentiment = 'negative') as negative_count,
        COUNT(*) FILTER (WHERE sentiment = 'neutral') as neutral_count,
        AVG(sentiment_score) as avg_score,
        COUNT(*) as total_news
    FROM analyzed_news
    WHERE symbol = 'VNM' 
      AND DATE(published_at) = '2024-11-15'
    
    # 2. Xác định overall sentiment
    if positive_count > negative_count:
        overall_sentiment = 'positive'
    elif negative_count > positive_count:
        overall_sentiment = 'negative'
    else:
        overall_sentiment = 'neutral'
    
    # 3. Insert hoặc Update vào sentiment_summary
    INSERT INTO sentiment_summary (symbol, date, positive_count, ...)
    VALUES ('VNM', '2024-11-15', 5, 1, 2, 0.45, 'positive', 8)
    ON CONFLICT (symbol, date) DO UPDATE SET ...
```

**Ví dụ dữ liệu tổng hợp**:

| symbol | date | positive | negative | neutral | avg_score | overall | news_count |
|--------|------|----------|----------|---------|-----------|---------|------------|
| VNM | 2024-11-15 | 5 | 1 | 2 | +0.45 | positive | 8 |
| HPG | 2024-11-15 | 2 | 4 | 1 | -0.32 | negative | 7 |
| VCB | 2024-11-15 | 3 | 3 | 2 | +0.05 | neutral | 8 |

---

## 🚀 Cách Chạy

### **Option 1: Chạy script offline (Khuyến nghị)**

```bash
# Activate virtual environment
& D:\KLTN\venv\Scripts\Activate.ps1

# Phân tích tất cả tin tức mới (7 ngày gần đây)
python scripts/analyze_news_finbert.py

# Phân tích cho mã cụ thể
python scripts/analyze_news_finbert.py --symbols VNM HPG VCB

# Phân tích 30 ngày gần đây, tối đa 200 tin
python scripts/analyze_news_finbert.py --days 30 --limit 200
```

**Output**:
```
============================================================
🤖 FinBERT News Sentiment Analyzer
============================================================
✅ Database connected
📊 Current: 150 news, 25 symbols
📰 Fetching news...
✅ Found 45 articles
📰 Will analyze 45 articles
🔄 Loading Vietnamese Sentiment model...
✅ PhoBERT Vietnamese Sentiment loaded on GPU (CUDA)

[1/45] 📈 VNM: positive (0.78)
[2/45] 📉 HPG: negative (-0.42)
[3/45] ➡️ VCB: neutral (0.05)
...
📊 Updating daily summaries...

============================================================
📊 ANALYSIS COMPLETE
============================================================
✅ Analyzed: 45 news
💾 Saved: 45 news
📈 Positive: 20 (44%)
📉 Negative: 12 (27%)
➡️ Neutral: 13 (29%)
📊 Symbols: VNM, HPG, VCB, FPT, ...
⏱️  Duration: 23.5s
============================================================
```

---

### **Option 2: Gọi qua API**

**⚠️ Lưu ý**: API chỉ **ĐỌC** dữ liệu đã phân tích, KHÔNG phân tích realtime.

**Endpoint 1: Lấy sentiment cho 1 mã**

```bash
GET http://localhost:8000/api/finbert/sentiment/VNM?days=7
```

**Response**:
```json
{
    "status": "ok",
    "symbol": "VNM",
    "sentiment_summary": {
        "avg_score": 0.45,
        "overall_sentiment": "positive",
        "positive_count": 15,
        "negative_count": 3,
        "neutral_count": 5,
        "total_news": 23,
        "recommendation": "📈 BUY SIGNAL - Sentiment tích cực"
    },
    "daily_summary": [
        {
            "date": "2024-11-15",
            "positive": 5,
            "negative": 1,
            "neutral": 2,
            "avg_score": 0.62,
            "overall": "positive",
            "news_count": 8
        },
        ...
    ],
    "recent_news": [
        {
            "title": "Vinamilk báo lãi tăng 25%",
            "sentiment": "positive",
            "sentiment_score": 0.78,
            "scores": {
                "positive": 0.89,
                "negative": 0.05,
                "neutral": 0.06
            }
        },
        ...
    ]
}
```

**Endpoint 2: Tổng hợp toàn thị trường**

```bash
GET http://localhost:8000/api/finbert/summary
```

**Response**:
```json
{
    "status": "ok",
    "market_summary": [
        {
            "symbol": "VNM",
            "date": "2024-11-15",
            "avg_score": 0.45,
            "overall": "positive",
            "positive": 5,
            "negative": 1,
            "neutral": 2,
            "total": 8
        },
        ...
    ],
    "statistics": {
        "total_symbols": 25,
        "total_news": 156,
        "positive_symbols": 12,
        "negative_symbols": 7,
        "neutral_symbols": 6
    }
}
```

**Endpoint 3: Kiểm tra trạng thái**

```bash
GET http://localhost:8000/api/finbert/status
```

**Response**:
```json
{
    "status": "ready",
    "model": "PhoBERT Vietnamese Sentiment",
    "statistics": {
        "total_news_analyzed": 156,
        "symbols_covered": 25,
        "latest_analysis": "2024-11-15 16:30:00",
        "latest_summary": "2024-11-15 16:31:00"
    },
    "symbols": ["VNM", "HPG", "VCB", "FPT", ...]
}
```

---

## 🔍 Các Trường hợp Sử dụng

### **1. Trading Signal từ Sentiment**

```python
# Logic: Nếu avg_score > 0.3 → BUY
#        Nếu avg_score < -0.3 → SELL
#        Nếu -0.3 < avg_score < 0.3 → HOLD

if avg_score > 0.3:
    signal = "📈 BUY - Sentiment tích cực mạnh"
elif avg_score < -0.3:
    signal = "📉 SELL - Sentiment tiêu cực"
else:
    signal = "➡️ HOLD - Sentiment trung lập"
```

### **2. Cảnh báo Sentiment đột biến**

```python
# Nếu sentiment thay đổi mạnh trong 1 ngày
if today_score - yesterday_score > 0.5:
    alert = "⚠️ Sentiment tăng đột biến - Có tin tốt?"
elif today_score - yesterday_score < -0.5:
    alert = "🚨 Sentiment giảm mạnh - Có tin xấu?"
```

### **3. Kết hợp với Phân tích Kỹ thuật**

```python
# Kết hợp Technical Indicators + Sentiment
if RSI < 30 and sentiment == 'positive' and avg_score > 0.4:
    signal = "🔥 STRONG BUY - Oversold + Tin tốt"
elif RSI > 70 and sentiment == 'negative' and avg_score < -0.4:
    signal = "⛔ STRONG SELL - Overbought + Tin xấu"
```

---

## 📊 Database Schema

```sql
-- Table 1: analyzed_news (Chi tiết từng tin)
CREATE TABLE analyzed_news (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20),
    title TEXT,
    sentiment VARCHAR(20),        -- positive/negative/neutral
    sentiment_score FLOAT,        -- -1.0 đến +1.0
    positive_score FLOAT,         -- 0.0 đến 1.0
    negative_score FLOAT,         -- 0.0 đến 1.0
    neutral_score FLOAT,          -- 0.0 đến 1.0
    published_at TIMESTAMP,
    analyzed_at TIMESTAMP DEFAULT NOW()
);

-- Table 2: sentiment_summary (Tổng hợp theo ngày)
CREATE TABLE sentiment_summary (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    date DATE NOT NULL,
    positive_count INT,           -- Số tin positive
    negative_count INT,           -- Số tin negative
    neutral_count INT,            -- Số tin neutral
    avg_score FLOAT,              -- Điểm trung bình
    overall_sentiment VARCHAR(20), -- Sentiment tổng thể
    news_count INT,               -- Tổng số tin
    updated_at TIMESTAMP,
    UNIQUE(symbol, date)
);
```

---

## 🎓 Công thức Tính toán

### **1. Sentiment Score**
```
sentiment_score = positive_score - negative_score
Range: -1.0 (rất tiêu cực) đến +1.0 (rất tích cực)

Ví dụ:
- Positive: 0.89, Negative: 0.05 → Score = 0.89 - 0.05 = +0.84
- Positive: 0.15, Negative: 0.75 → Score = 0.15 - 0.75 = -0.60
```

### **2. Average Daily Score**
```
avg_score = Σ(sentiment_score) / news_count

Ví dụ ngày 15/11/2024 cho VNM:
- 5 tin positive: [+0.78, +0.65, +0.82, +0.55, +0.70]
- 1 tin negative: [-0.42]
- 2 tin neutral: [+0.05, -0.10]

avg_score = (0.78 + 0.65 + 0.82 + 0.55 + 0.70 - 0.42 + 0.05 - 0.10) / 8
          = 3.03 / 8 = +0.38
```

### **3. Overall Sentiment**
```
if positive_count > negative_count:
    overall = 'positive'
elif negative_count > positive_count:
    overall = 'negative'
else:
    overall = 'neutral'
```

---

## ⚙️ Configuration

**File**: `src/config/etl_config.yaml`

```yaml
sentiment:
  model:
    name: "wonrax/phobert-base-vietnamese-sentiment"
    device: "cuda"  # hoặc "cpu"
    max_length: 256
    batch_size: 16
  
  scoring:
    positive_threshold: 0.3    # > 0.3 → BUY signal
    negative_threshold: -0.3   # < -0.3 → SELL signal
    
  aggregation:
    lookback_days: 7           # Tổng hợp 7 ngày gần đây
    min_news_count: 3          # Tối thiểu 3 tin để đưa ra signal
```

---

## 🐛 Troubleshooting

### **Lỗi: Model không tải được**
```bash
# Solution: Cài đặt dependencies
pip install transformers torch
pip install sentencepiece  # Cho PhoBERT tokenizer
```

### **Lỗi: Database connection failed**
```bash
# Solution: Kiểm tra DATABASE_URL trong .env
DATABASE_URL=postgresql://user:password@host:port/database
```

### **Lỗi: "analyzed_news table does not exist"**
```bash
# Solution: Chạy script để tạo tables
python scripts/analyze_news_finbert.py
# Tables sẽ được tự động tạo lần đầu chạy
```

### **Warning: CUDA not available**
```
# Không phải lỗi nghiêm trọng, model sẽ chạy trên CPU (chậm hơn)
# Nếu muốn dùng GPU, cài PyTorch với CUDA:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

---

## 📈 Performance

**Benchmark trên máy Windows 10, Python 3.13**:

| Environment | Speed | GPU Memory |
|-------------|-------|------------|
| CPU (Intel i7) | ~2 tin/giây | - |
| GPU (NVIDIA RTX 3060) | ~10 tin/giây | ~2GB VRAM |

**Khuyến nghị**:
- Với < 50 tin: CPU đủ nhanh
- Với > 200 tin: Nên dùng GPU

---

## 🎯 Tóm tắt Workflow

```
1. FETCH NEWS (NewsFetcher)
   ↓
   📰 Lấy tin từ RSS + Database
   
2. ANALYZE (VietnameseSentimentAnalyzer)
   ↓
   🤖 PhoBERT phân tích → sentiment + scores
   
3. SAVE (DatabaseManager.save_news)
   ↓
   💾 Lưu vào analyzed_news table
   
4. AGGREGATE (DatabaseManager.update_summary)
   ↓
   📊 Tổng hợp → sentiment_summary table
   
5. API ACCESS (FastAPI endpoints)
   ↓
   🌐 Truy vấn qua /api/finbert/*
```

---

## 📚 Tài liệu Tham khảo

1. **PhoBERT Paper**: https://arxiv.org/abs/2003.00744
2. **FinBERT Paper**: https://arxiv.org/abs/1908.10063
3. **Model Hub**: https://huggingface.co/wonrax/phobert-base-vietnamese-sentiment
4. **Transformers Docs**: https://huggingface.co/docs/transformers

---

## 💡 Tips & Best Practices

1. **Chạy phân tích mỗi tối sau giờ đóng cửa** (18:00)
2. **Lưu model_version** để tracking khi model thay đổi
3. **Backup analyzed_news table** trước khi re-analyze
4. **Kết hợp với Technical Indicators** để tăng độ chính xác
5. **Monitor avg_score trends** - xu hướng quan trọng hơn điểm tuyệt đối

---

**Tác giả**: KLTN Stock Prediction System  
**Version**: 1.0  
**Last Updated**: December 2, 2024
