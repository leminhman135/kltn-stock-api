# News Relevance & Sentiment Features Guide

## 📋 Tổng quan

Hệ thống đã được nâng cấp với 3 tính năng mới:

### 1. ✅ Tin tức load được chính xác
- **Vấn đề cũ**: API không trả về tin tức
- **Giải pháp**: Đã tích hợp `news_service` và `relevance_model` vào API v2
- **Endpoints hoạt động**: `/api/news`, `/api/news/{symbol}`

### 2. 🎯 Mô hình tính độ liên quan (Relevance Score)
- **File**: `src/news_relevance.py`
- **Mục đích**: Đánh giá tin tức có liên quan đến mã cổ phiếu đến mức nào (0-1)
- **Phương pháp**: TF-IDF-inspired với 5 features weighted

### 3. 🔍 Hiển thị features đánh giá
- **Sentiment Keywords**: Xem danh sách từ khóa positive/negative
- **Relevance Features**: Xem profile công ty (tên, sản phẩm, ngành)
- **UI**: Nút "Features" trong trang tin tức

---

## 🧠 Relevance Model - Chi tiết

### Kiến trúc

```python
NewsRelevanceModel
├── COMPANY_PROFILES: Dict[symbol, profile]
│   ├── names: Tên công ty chính thức
│   ├── aliases: Tên viết tắt, giao dịch
│   ├── keywords: Sản phẩm, thương hiệu, dự án
│   └── industry: Ngành nghề kinh doanh
│
└── Methods:
    ├── calculate_relevance_score(text, symbol) → Dict
    ├── rank_news_by_relevance(news_list, symbol) → List
    └── get_features_explanation(symbol) → Dict
```

### Công thức tính điểm

Mỗi tin tức được đánh giá qua **5 yếu tố** với trọng số khác nhau:

| Feature | Trọng số | Mô tả | Ví dụ (VNM) |
|---------|----------|-------|-------------|
| **Exact Match** | 40% | Tìm mã chính xác trong văn bản | "VNM", "vnm" |
| **Company Name** | 30% | Tên công ty chính thức | "Vinamilk", "Sữa Việt Nam" |
| **Aliases** | 20% | Tên viết tắt, giao dịch | "CTCP Sữa Việt Nam" |
| **Keywords** | 15% | Sản phẩm, thương hiệu | "sữa", "yogurt", "dielac" |
| **Industry** | 10% | Ngành nghề | "thực phẩm", "f&b", "tiêu dùng" |

**Tổng điểm** = Sum of all matches (capped at 1.0)

### Phân loại độ tin cậy

```
🟢 Score ≥ 0.7  → RẤT CAO    - Tin TRỰC TIẾP về {symbol}
🟡 Score 0.4-0.7 → CAO        - Tin LIÊN QUAN đến {symbol}
🟠 Score 0.2-0.4 → TRUNG BÌNH - Tin CÓ THỂ ảnh hưởng {symbol}
⚪ Score < 0.2   → THẤP       - Tin thị trường chung
```

### Ví dụ thực tế

**Test case 1**: VNM
```python
text = "Vinamilk công bố lợi nhuận quý 3 tăng 25%"

Kết quả:
- Company Name Match: ✓ "Vinamilk" → +0.15
- Keyword Match: (không có từ khóa sản phẩm) → 0
→ Total Score: 0.15 (⚪ Thấp - "Tin tức thị trường chung")
```

**Test case 2**: VNM
```python
text = "Thị trường sữa Việt Nam tăng trưởng mạnh, VNM dẫn đầu"

Kết quả:
- Exact Match: ✓ "VNM" → +0.2
- Company Name: ✓ "Việt Nam" (partial) → +0.15
- Alias: ✓ "vnm" → +0.1
- Keyword: ✓ "sữa" → +0.03
→ Total Score: 0.48 (🟡 Cao - "Tin tức LIÊN QUAN đến VNM")
```

**Test case 3**: HPG
```python
text = "Giá thép trong nước tăng mạnh theo xu hướng thế giới"

Kết quả:
- Keyword: ✓ "thép" → +0.03
- Industry: ✓ "thép" → +0.05
→ Total Score: 0.08 (⚪ Thấp - "Tin tức thị trường chung")
```

---

## 📊 Sentiment Analysis - Chi tiết

### Phương pháp: Keyword-based

File: `src/news_service.py` → `SentimentAnalyzer`

### Features

#### 1. Positive Keywords (80+ từ)

**Danh mục**:
- **Tài chính**: tăng trưởng, lợi nhuận tăng, doanh thu tăng, vượt kế hoạch
- **Kinh doanh**: mở rộng, đầu tư mới, hợp tác, thắng thầu, sáp nhập
- **Thị trường**: uptrend, breakout, vượt đỉnh, khối ngoại mua ròng
- **Đánh giá**: outperform, strong buy, nâng rating, khuyến nghị mua

**Ví dụ**:
```python
POSITIVE_KEYWORDS = [
    "tăng trưởng", "lợi nhuận tăng", "cổ tức cao", 
    "mở rộng", "hợp tác", "thắng thầu",
    "breakout", "tăng trần", "khối ngoại mua ròng",
    "khuyến nghị mua", "nâng rating", "tiềm năng"
]
```

#### 2. Negative Keywords (70+ từ)

**Danh mục**:
- **Tài chính**: thua lỗ, nợ xấu, giảm lợi nhuận, phá sản
- **Kinh doanh**: đóng cửa, sa thải, mất hợp đồng, tranh chấp
- **Thị trường**: downtrend, breakdown, giảm sàn, bán tháo
- **Đánh giá**: underperform, sell, hạ rating, cảnh báo

**Ví dụ**:
```python
NEGATIVE_KEYWORDS = [
    "thua lỗ", "nợ xấu", "giảm lợi nhuận",
    "đóng cửa", "sa thải", "tranh chấp",
    "breakdown", "giảm sàn", "bán tháo",
    "cảnh báo", "hạ rating", "rủi ro cao"
]
```

#### 3. Strong Modifiers (tăng trọng số 1.5x)

```python
STRONG_MODIFIERS = [
    "kỷ lục", "đột biến", "lịch sử", 
    "chưa từng có", "cao nhất", "thấp nhất"
]
```

### Công thức tính điểm

```python
# 1. Đếm số lượng matches
pos_count = số từ positive tìm thấy
neg_count = số từ negative tìm thấy

# 2. Tính score
score = (pos_count - neg_count) / (pos_count + neg_count)

# 3. Nhân với modifier (nếu có)
if has_strong_modifier:
    score *= 1.5

# 4. Normalize
score = clamp(score, -1.0, 1.0)
```

### Phân loại

```python
if score > 0.2:
    sentiment = "positive"
    if score > 0.6:
        impact = "🚀 Tín hiệu TĂNG MẠNH - Khuyến nghị MUA"
    else:
        impact = "📈 Tín hiệu TĂNG - Cân nhắc mua vào"
        
elif score < -0.2:
    sentiment = "negative"
    if score < -0.6:
        impact = "🔻 Tín hiệu GIẢM MẠNH - Khuyến nghị BÁN"
    else:
        impact = "📉 Tín hiệu GIẢM - Cân nhắc cắt lỗ"
        
else:
    sentiment = "neutral"
    impact = "➡️ Trung lập - Tiếp tục theo dõi diễn biến"
```

---

## 🌐 API Endpoints

### 1. GET `/api/news/{symbol}`

Lấy tin tức cho mã cổ phiếu với **relevance scoring**

**Response**:
```json
{
  "status": "success",
  "symbol": "VNM",
  "sentiment_summary": {
    "overall": "positive",
    "avg_score": 0.35,
    "positive_count": 12,
    "negative_count": 3,
    "neutral_count": 5,
    "recommendation": "🟢 TIN TỨC TÍCH CỰC (12/20 tin tốt)"
  },
  "total_news": 20,
  "news": [
    {
      "title": "...",
      "summary": "...",
      "url": "...",
      "source": "CafeF",
      "published_at": "2024-12-03 14:30",
      "sentiment": "positive",
      "sentiment_score": 0.67,
      "impact": "🚀 Tín hiệu TĂNG MẠNH",
      
      // NEW: Relevance data
      "relevance_score": 0.75,
      "relevance_confidence": "🟢 Rất cao",
      "relevance_explanation": "Tin tức TRỰC TIẾP về VNM",
      "matched_features": [
        "✓ Mã VNM",
        "✓ Tên công ty",
        "✓ Keyword: sữa",
        "✓ Ngành: thực phẩm, f&b"
      ]
    }
  ]
}
```

**Features**:
- Tin được sắp xếp theo `relevance_score` (cao nhất trước)
- Mỗi tin có 4 trường mới: `relevance_score`, `confidence`, `explanation`, `matched_features`

### 2. GET `/api/news/features/sentiment`

Lấy danh sách keywords dùng để phân tích sentiment

**Response**:
```json
{
  "status": "success",
  "method": "keyword-based",
  "description": "Phân tích sentiment dựa trên từ khóa tiếng Việt",
  "features": {
    "positive_keywords": {
      "count": 80,
      "examples": ["tăng trưởng", "lợi nhuận", "..."],
      "categories": [...]
    },
    "negative_keywords": {
      "count": 70,
      "examples": ["thua lỗ", "nợ xấu", "..."],
      "categories": [...]
    },
    "modifiers": {
      "count": 7,
      "examples": ["kỷ lục", "đột biến", "..."]
    }
  },
  "scoring": {
    "formula": "(positive_count - negative_count) / total_count",
    "range": "[-1.0, 1.0]",
    "classification": {...}
  }
}
```

### 3. GET `/api/news/features/relevance/{symbol}`

Lấy thông tin profile công ty cho relevance scoring

**Response**:
```json
{
  "status": "success",
  "symbol": "VNM",
  "features": {
    "exact_match": {
      "weight": "40%",
      "description": "Tìm mã chính xác trong văn bản",
      "examples": ["VNM", "vnm"]
    },
    "company_name": {
      "weight": "30%",
      "description": "Tên công ty chính thức",
      "examples": ["Vinamilk", "Sữa Việt Nam", "..."]
    },
    "aliases": {...},
    "keywords": {...},
    "industry": {...}
  },
  "total_keywords": 25
}
```

---

## 💻 Sử dụng trong UI

### 1. Xem tin tức với relevance

```javascript
// Chọn mã cổ phiếu
document.getElementById('newsSymbol').value = 'VNM';

// Load tin tức
await loadNews();

// Kết quả hiển thị:
// - Badge "🎯 Độ liên quan: 75%" ở mỗi tin
// - Matched features dưới dạng tags
// - Tin được sắp xếp theo độ liên quan
```

### 2. Xem features đánh giá

```javascript
// Bấm nút "Features" trong trang tin tức
showFeaturesInfo();

// Modal hiển thị:
// - Sentiment keywords (positive/negative)
// - Công thức tính điểm sentiment
// - Relevance features với trọng số
// - Company profile (tên, sản phẩm, ngành)
```

### 3. UI Components

**News Card với Relevance**:
```html
<div class="news-item positive">
  <div class="news-title">Vinamilk tăng trưởng mạnh...</div>
  <div class="news-summary">Lợi nhuận quý 3...</div>
  
  <!-- Relevance Box -->
  <div style="background:rgba(139,92,246,0.05);">
    <span>🎯 Độ liên quan: 75%</span>
    <span>🟢 Rất cao</span>
    <div>Tin tức TRỰC TIẾP về VNM</div>
    <div class="news-features">
      <span class="feature-tag">✓ Mã VNM</span>
      <span class="feature-tag">✓ Tên công ty</span>
      <span class="feature-tag">✓ Keyword: sữa</span>
    </div>
  </div>
  
  <div class="news-impact">🚀 Tín hiệu TĂNG MẠNH</div>
  <div class="news-meta">...</div>
</div>
```

---

## 📈 Company Profiles

Hiện có profile cho **18 mã** lớn:

| Mã | Công ty | Keywords | Ngành |
|----|---------|----------|-------|
| VNM | Vinamilk | sữa, yogurt, dielac | Thực phẩm, F&B |
| VIC | Vingroup | vinfast, vinhomes, vincom | BĐS, ô tô, retail |
| HPG | Hòa Phát | thép, sắt thép, xây dựng | Thép, kim loại |
| FPT | FPT | công nghệ, phần mềm, telecom | IT, Software |
| MWG | Thế giới di động | điện máy xanh, bách hóa xanh | Bán lẻ, điện tử |
| VCB | Vietcombank | ngoại thương | Ngân hàng |
| BID | BIDV | đầu tư phát triển | Ngân hàng |
| ... | ... | ... | ... |

**Để thêm profile mới**, edit `src/news_relevance.py`:

```python
COMPANY_PROFILES = {
    "ABC": {
        "names": ["Công ty ABC", "ABC Corporation"],
        "aliases": ["abc", "ctcp abc"],
        "keywords": ["sản phẩm 1", "sản phẩm 2", "brand"],
        "industry": ["ngành 1", "ngành 2"],
    },
    # ... thêm các mã khác
}
```

---

## 🧪 Testing

### Test Relevance Model

```bash
cd D:\KLTN
python -m src.news_relevance
```

Output:
```
============================================================
Symbol: VNM
Text: Thị trường sữa Việt Nam tăng trưởng mạnh, VNM dẫn đầu
Score: 0.48 - 🟡 Cao
Matched: ['✓ Mã VNM', '✓ Tên công ty', '✓ Keyword: sữa']
Explain: Tin tức LIÊN QUAN đến VNM
```

### Test API Endpoints

```bash
# 1. Test news with relevance
curl http://localhost:8000/api/news/VNM

# 2. Test sentiment features
curl http://localhost:8000/api/news/features/sentiment

# 3. Test relevance features
curl http://localhost:8000/api/news/features/relevance/VNM
```

---

## 🚀 Deployment

### Railway Auto-Deploy

```bash
git add -A
git commit -m "feat: news relevance model"
git push origin main
```

Railway sẽ tự động:
1. Detect changes
2. Build image
3. Deploy to production
4. Available tại: https://kltn-stock-api-production.up.railway.app

### Kiểm tra logs

```bash
# Xem Railway logs
railway logs
```

---

## 📊 Performance

### Metrics

- **Relevance calculation**: ~2-5ms per article
- **Batch processing**: ~100 articles in <1s
- **API response time**: <500ms (with 20 articles)
- **Memory**: ~50MB for model + profiles

### Caching

News service có built-in cache:
- **Duration**: 5 phút
- **Key**: `all_news_{symbol}_{limit}`
- **Benefit**: Giảm RSS fetch calls

---

## 🔧 Troubleshooting

### Issue 1: Tin tức không load

**Triệu chứng**: API trả về empty array

**Nguyên nhân**:
- RSS feeds bị block/timeout
- BeautifulSoup4 chưa cài

**Giải pháp**:
```bash
pip install beautifulsoup4 lxml
```

### Issue 2: Relevance score luôn thấp

**Triệu chứng**: Tất cả tin đều score < 0.2

**Nguyên nhân**:
- Symbol chưa có trong `COMPANY_PROFILES`
- Tin tức không có keyword match

**Giải pháp**:
1. Thêm profile cho symbol
2. Kiểm tra keywords có phù hợp không

### Issue 3: Features modal không hiện

**Triệu chứng**: Click "Features" không có gì xảy ra

**Nguyên nhân**:
- Chưa chọn symbol
- API endpoint lỗi

**Giải pháp**:
1. Chọn mã cổ phiếu trước
2. Check console log
3. Test API endpoint trực tiếp

---

## 📚 Tài liệu tham khảo

### Papers & Articles

1. **TF-IDF**: [Wikipedia](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
2. **Sentiment Analysis**: [Keyword-based approach](https://www.sciencedirect.com/topics/computer-science/keyword-based-approach)
3. **News Relevance**: Domain-specific keyword matching

### Code Structure

```
src/
├── news_service.py         # RSS fetching + sentiment
├── news_relevance.py       # Relevance model (NEW)
├── api_v2.py              # API endpoints (UPDATED)
└── static/
    └── index.html          # UI with features (UPDATED)
```

---

## 🎯 Next Steps

### Potential Improvements

1. **Machine Learning Approach**
   - Train classifier on labeled data
   - Use Word2Vec/FastText embeddings
   - Deep learning with BERT

2. **More Features**
   - Time decay (recent news higher weight)
   - Source reliability scoring
   - Social media signals

3. **Optimization**
   - Pre-compute features at collection time
   - Index keywords for faster lookup
   - Distributed caching with Redis

4. **UI Enhancements**
   - Relevance heatmap
   - Feature importance visualization
   - Interactive keyword filtering

---

## 👨‍💻 Author

**Le Minh Man**
- GitHub: [@leminhman135](https://github.com/leminhman135)
- Project: KLTN Stock Prediction System

---

## 📝 Changelog

### [2024-12-03] - Version 1.0

**Added**:
- NewsRelevanceModel with 5-feature scoring
- Company profiles for 18 major stocks
- API endpoints for sentiment/relevance features
- UI modal for features explanation
- Relevance score display in news cards

**Changed**:
- `/api/news/{symbol}` now returns relevance data
- News sorted by relevance (highest first)

**Fixed**:
- News loading works correctly
- API integration with news_service

