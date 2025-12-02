# 📚 Documentation Index

## Tài liệu hướng dẫn hệ thống

### 🎯 Sentiment Analysis Pipeline

#### 1. [Sentiment Pipeline Guide](sentiment_pipeline_guide.md) - **Chi tiết đầy đủ**
   - **Mô tả**: Hướng dẫn chi tiết 7 bước sentiment analysis với FinBERT
   - **Nội dung**:
     - Kiến trúc hệ thống (Pure FinBERT vs Hybrid)
     - Chi tiết từng bước (1-7) với code examples
     - Performance benchmarks
     - API reference
     - Use cases thực tế
     - Troubleshooting
   - **Đối tượng**: Developers cần hiểu sâu về pipeline
   - **Thời gian đọc**: 30-45 phút

#### 2. [Quick Start Guide](quick_start_sentiment.md) - **Cài đặt nhanh**
   - **Mô tả**: Hướng dẫn cài đặt và sử dụng nhanh trong 5 phút
   - **Nội dung**:
     - Cài đặt dependencies
     - 3 use cases chính
     - Configuration options
     - Common tasks
     - FAQ
   - **Đối tượng**: Developers muốn bắt đầu ngay
   - **Thời gian đọc**: 5-10 phút

### 📰 News Relevance Model

#### 3. [News Relevance Guide](news_relevance_guide.md) - **Mô hình độ liên quan**
   - **Mô tả**: Hướng dẫn về mô hình đánh giá mức độ liên quan tin tức với cổ phiếu
   - **Nội dung**:
     - 5 features: exact_match, company_name, aliases, keywords, industry
     - Công thức tính điểm (weighted scoring)
     - Company profiles (18+ major stocks)
     - API endpoints
     - UI integration
   - **Đối tượng**: Developers làm việc với tin tức
   - **Thời gian đọc**: 20-30 phút

### 📊 Dataset Description

#### 4. [Dataset Description](dataset_description.md) - **Mô tả tập dữ liệu**
   - **Mô tả**: Tài liệu chi tiết về tập dữ liệu sử dụng trong thực nghiệm
   - **Nội dung**:
     - 30 mã VN30 stocks với phân nhóm ngành
     - Cấu trúc OHLCV data (~37,500 records)
     - 20+ technical indicators với công thức
     - News data và sentiment analysis (~50,000 articles)
     - Data sources: VNDirect API, SSI API, RSS feeds
     - Data preprocessing & quality metrics
     - Database schema và storage structure
   - **Đối tượng**: Researchers, data scientists, reviewers
   - **Thời gian đọc**: 30-40 phút

---

## 🗂️ Cấu trúc Documentation

```
docs/
├── README.md                          # This file
├── sentiment_pipeline_guide.md        # 📘 Full pipeline guide (1400 lines)
├── quick_start_sentiment.md           # 🚀 Quick start (300 lines)
├── news_relevance_guide.md            # 📰 Relevance model (590 lines)
└── dataset_description.md             # 📊 Dataset documentation (1200 lines)
```

---

## 📖 Reading Path

### Path 1: Tôi muốn bắt đầu nhanh
```
1. quick_start_sentiment.md (5 phút)
   → Test pipeline ngay: python scripts/run_sentiment_pipeline.py VNM --days 7
   
2. sentiment_pipeline_guide.md - Section "Sử dụng Pipeline" (10 phút)
   → Hiểu rõ options và outputs

3. news_relevance_guide.md - Section "API Endpoints" (5 phút)
   → Integrate vào code của bạn
```

**Total**: 20 phút → Ready to use

### Path 2: Tôi muốn hiểu sâu hệ thống
```
1. sentiment_pipeline_guide.md - Full read (45 phút)
   → Hiểu toàn bộ 7 bước chi tiết
   
2. news_relevance_guide.md - Full read (30 phút)
   → Hiểu công thức relevance scoring

3. dataset_description.md - Full read (40 phút)
   → Hiểu cấu trúc dữ liệu và nguồn thu thập

4. Đọc source code:
   - src/hybrid_sentiment.py (400 lines)
   - src/sentiment_pipeline.py (600 lines)
   - src/news_relevance.py (400 lines)
```

**Total**: 3-4 giờ → Expert level

### Path 3: Tôi cần giải quyết vấn đề cụ thể

#### Problem: Sentiment trả về toàn neutral
→ `quick_start_sentiment.md` - Section "FAQ" - Q1

#### Problem: Pipeline chạy chậm
→ `sentiment_pipeline_guide.md` - Section "Performance Metrics"
→ `quick_start_sentiment.md` - Section "Performance Tips"

#### Problem: Muốn thêm keywords mới
→ `quick_start_sentiment.md` - Section "FAQ" - Q2
→ `sentiment_pipeline_guide.md` - Section "Bước 5" - Keyword-based logic

#### Problem: Muốn filter tin tức theo mức độ liên quan
→ `news_relevance_guide.md` - Section "Sử dụng Relevance Model"

#### Problem: Hiểu cấu trúc dữ liệu training
→ `dataset_description.md` - Section "Loại Dữ liệu"
→ `dataset_description.md` - Section "Thống kê Mô tả"

#### Problem: Cần biết nguồn dữ liệu
→ `dataset_description.md` - Section "Nguồn Thu thập Dữ liệu"

---

## 🎓 Concepts Index

### Sentiment Analysis Concepts

| Concept | Location | Description |
|---------|----------|-------------|
| **FinBERT** | `sentiment_pipeline_guide.md` - Bước 4 | Pre-trained BERT for financial text |
| **Tokenization** | `sentiment_pipeline_guide.md` - Bước 3 | Convert text to BERT tokens |
| **Embedding** | `sentiment_pipeline_guide.md` - Bước 4 | 768-dim vector representation |
| **Hybrid Approach** | `sentiment_pipeline_guide.md` - Bước 5 | Keyword + FinBERT combined |
| **Sentiment Score** | `sentiment_pipeline_guide.md` - Bước 6 | Numerical conversion [-1, 1] |
| **Daily Aggregation** | `sentiment_pipeline_guide.md` - Bước 6 | Group by date statistics |

### Relevance Model Concepts

| Concept | Location | Description |
|---------|----------|-------------|
| **TF-IDF Scoring** | `news_relevance_guide.md` - Section "Công thức" | Weighted feature scoring |
| **Exact Match** | `news_relevance_guide.md` - Feature 1 | Direct stock symbol match (40%) |
| **Company Name** | `news_relevance_guide.md` - Feature 2 | Company name mention (30%) |
| **Aliases** | `news_relevance_guide.md` - Feature 3 | Alternative names (20%) |
| **Keywords** | `news_relevance_guide.md` - Feature 4 | Related terms (15%) |
| **Industry** | `news_relevance_guide.md` - Feature 5 | Industry context (10%) |

---

## 💻 Code Examples

### Example 1: Basic Sentiment Analysis

```python
from src.hybrid_sentiment import HybridSentimentAnalyzer

analyzer = HybridSentimentAnalyzer(use_finbert=False)
result = analyzer.analyze("Vinamilk lợi nhuận tăng 25%")

print(result['sentiment'])       # 'neutral'
print(result['sentiment_score']) # 0.0
```

**Explained in**: `quick_start_sentiment.md` - Use Case 1

### Example 2: Batch Processing

```python
from scripts.run_sentiment_pipeline import run_pipeline_for_symbol

result = run_pipeline_for_symbol('VNM', days=30, save_csv=True)
print(f"Analyzed {result['total_news']} news articles")
```

**Explained in**: `sentiment_pipeline_guide.md` - Section "Sử dụng Pipeline"

### Example 3: Relevance Scoring

```python
from src.news_relevance import NewsRelevanceModel

model = NewsRelevanceModel()
score = model.calculate_relevance_score(
    text="Vinamilk công bố kết quả kinh doanh",
    symbol='VNM'
)

print(f"Relevance: {score['relevance_score']:.2f}")
```

**Explained in**: `news_relevance_guide.md` - Section "Sử dụng"

### Example 4: API Integration

```python
import requests

response = requests.post(
    'http://localhost:8000/api/ml/sentiment',
    json={'texts': ['Thị trường tăng điểm']}
)

print(response.json()['results'][0]['sentiment'])
```

**Explained in**: `quick_start_sentiment.md` - Use Case 3

---

## 📊 Data Flow Diagrams

### Sentiment Pipeline Flow

```
Raw News → Collector → Cleaner → Tokenizer → Embedder → Predictor → Converter → Integrator
           (Step 1)   (Step 2)   (Step 3)     (Step 4)   (Step 5)    (Step 6)    (Step 7)
                                                                                      ↓
                                                                               ML Model Features
```

**Detailed in**: `sentiment_pipeline_guide.md` - Section "Kiến trúc"

### Relevance Scoring Flow

```
News Text → Extract Features → Calculate Weights → Combine Scores → Final Score (0-1)
                  ↓                    ↓                 ↓
            [exact_match]        [40% weight]      [weighted_sum]
            [company_name]       [30% weight]
            [aliases]            [20% weight]
            [keywords]           [15% weight]
            [industry]           [10% weight]
```

**Detailed in**: `news_relevance_guide.md` - Section "Công thức"

---

## 🔧 Configuration Files

### Python Dependencies

```
# requirements.txt
transformers>=4.36.0    # For FinBERT
torch>=2.1.0           # Deep learning
pandas>=2.3.3          # Data processing
numpy>=2.3.5           # Numerical operations
sqlalchemy>=2.0.0      # Database integration
```

**Location**: `/requirements.txt`

### Model Configuration

```python
# Hybrid Analyzer Config
FINBERT_MODEL = 'ProsusAI/finbert'
BATCH_SIZE = 16
MAX_LENGTH = 512
USE_GPU = True  # Auto-detect

# Keyword-based Config
POSITIVE_KEYWORDS = [...]  # 80+ keywords
NEGATIVE_KEYWORDS = [...]  # 70+ keywords
CONFIDENCE_THRESHOLD = 0.6
```

**Location**: `src/hybrid_sentiment.py` - Lines 20-50

### Relevance Model Config

```python
# Feature Weights
EXACT_MATCH_WEIGHT = 0.40
COMPANY_NAME_WEIGHT = 0.30
ALIASES_WEIGHT = 0.20
KEYWORDS_WEIGHT = 0.15
INDUSTRY_WEIGHT = 0.10

# Thresholds
HIGH_RELEVANCE = 0.60
MEDIUM_RELEVANCE = 0.30
LOW_RELEVANCE = 0.15
```

**Location**: `src/news_relevance.py` - Lines 15-30

---

## 🧪 Testing

### Test Files

```
scripts/
└── run_sentiment_pipeline.py   # Integration test

tests/ (to be created)
├── test_sentiment.py           # Unit tests for sentiment
├── test_relevance.py           # Unit tests for relevance
└── test_integration.py         # Full pipeline tests
```

### Run Tests

```bash
# Test sentiment pipeline
python scripts/run_sentiment_pipeline.py VNM --days 7

# Test hybrid analyzer
python -m src.hybrid_sentiment

# Test relevance model
python -m src.news_relevance
```

**Results documented in**: `sentiment_pipeline_guide.md` - Section "Kết quả thực tế"

---

## 📈 Performance Metrics

### Benchmarks Summary

| Metric | Keyword-based | FinBERT (CPU) | FinBERT (GPU) |
|--------|---------------|---------------|---------------|
| **Speed** | 2ms/text | 50ms/text | 10ms/text |
| **Memory** | 50MB | 500MB | 500MB + 200MB VRAM |
| **Accuracy (Vi)** | 85% | 45% | 45% |
| **Accuracy (En)** | 70% | 90% | 90% |

**Full details**: `sentiment_pipeline_guide.md` - Section "Performance Metrics"

---

## 🐛 Known Issues & Solutions

### Issue 1: FinBERT all neutral for Vietnamese
- **Cause**: Model trained on English only
- **Solution**: Use Hybrid with `use_finbert=False`
- **Documented**: `quick_start_sentiment.md` - FAQ Q1

### Issue 2: Out of memory
- **Cause**: FinBERT model too large
- **Solution**: Reduce batch_size or use keyword-based
- **Documented**: `sentiment_pipeline_guide.md` - Troubleshooting

### Issue 3: Slow processing
- **Cause**: CPU inference
- **Solution**: Use GPU or keyword-based for Vietnamese
- **Documented**: `quick_start_sentiment.md` - Performance Tips

---

## 📞 Getting Help

### Documentation Navigation

1. **Start here**: `quick_start_sentiment.md`
2. **Need details**: `sentiment_pipeline_guide.md`
3. **Working with news**: `news_relevance_guide.md`

### Search Documentation

```bash
# Search for specific topic
grep -r "keyword-based" docs/
grep -r "FinBERT" docs/
grep -r "relevance score" docs/
```

### Common Questions

| Question | Answer Location |
|----------|----------------|
| How to install? | `quick_start_sentiment.md` - Section 1 |
| What is FinBERT? | `sentiment_pipeline_guide.md` - Bước 4 |
| How to add keywords? | `quick_start_sentiment.md` - FAQ Q2 |
| How to deploy? | `quick_start_sentiment.md` - FAQ Q4 |
| What is relevance score? | `news_relevance_guide.md` - Section "Công thức" |

---

## 🎯 Quick Links

### Documentation Files
- [📘 Full Pipeline Guide](sentiment_pipeline_guide.md)
- [🚀 Quick Start](quick_start_sentiment.md)
- [📰 Relevance Model](news_relevance_guide.md)
- [📊 Dataset Description](dataset_description.md)

### Code Files
- [src/hybrid_sentiment.py](../src/hybrid_sentiment.py) - Hybrid analyzer
- [src/sentiment_pipeline.py](../src/sentiment_pipeline.py) - Pure FinBERT
- [src/news_relevance.py](../src/news_relevance.py) - Relevance model
- [scripts/run_sentiment_pipeline.py](../scripts/run_sentiment_pipeline.py) - CLI tool

### API Endpoints
- `POST /api/ml/sentiment` - Sentiment analysis
- `GET /api/news/{symbol}` - News with relevance
- `GET /api/news/features/sentiment` - Keyword list
- `GET /api/news/features/relevance/{symbol}` - Company profile

### Database Documentation
- See `dataset_description.md` for full schema
- `stock_prices`: OHLCV data (~37,500 records)
- `technical_indicators`: 20+ indicators
- `news`: ~50,000 articles with sentiment

---

## ✅ Checklist for New Users

- [ ] Read `quick_start_sentiment.md` (5 minutes)
- [ ] Install dependencies: `pip install transformers torch pandas`
- [ ] Test pipeline: `python scripts/run_sentiment_pipeline.py VNM --days 7`
- [ ] Check output: `data/sentiment_analysis/VNM_*.csv`
- [ ] Read full guide: `sentiment_pipeline_guide.md` (30 minutes)
- [ ] Integrate into your code (see Use Cases)
- [ ] Deploy to production (see FAQ Q4)

---

## 📝 Version History

### v1.0 (2024-12-03)
- ✅ Complete 7-step sentiment pipeline
- ✅ Hybrid approach (keyword + FinBERT)
- ✅ News relevance model (5 features)
- ✅ CLI tool for batch processing
- ✅ API integration
- ✅ Comprehensive documentation (3 guides)

---

## 👨‍💻 Contributing

### Add New Keywords

1. Edit `src/hybrid_sentiment.py`
2. Add to `POSITIVE_KEYWORDS` or `NEGATIVE_KEYWORDS`
3. Test: `python -m src.hybrid_sentiment`
4. Document in `sentiment_pipeline_guide.md`

### Add New Company Profile

1. Edit `src/news_relevance.py`
2. Add to `COMPANY_PROFILES` dict
3. Test: `python -m src.news_relevance`
4. Update `news_relevance_guide.md`

### Report Issues

- GitHub Issues: https://github.com/leminhman135/KLTN
- Include: error message, code snippet, expected behavior

---

**Last Updated**: 2024-12-03  
**Author**: Le Minh Man  
**Project**: KLTN Stock Prediction System
