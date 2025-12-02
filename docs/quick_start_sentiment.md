# Quick Start Guide - Sentiment Analysis Pipeline

## 🚀 Cài đặt nhanh (5 phút)

### 1. Cài dependencies

```bash
pip install transformers torch pandas numpy sqlalchemy
```

### 2. Test pipeline ngay

```bash
# Analyze VNM last 7 days
python scripts/run_sentiment_pipeline.py VNM --days 7
```

**Expected output**:
```
✓ Thu thập 15 tin tức
✓ Phân tích sentiment: 60% neutral, 27% positive, 13% negative
✓ Average score: 0.178 (🟡 neutral trend)
✓ Saved: data/sentiment_analysis/VNM_news_*.csv
```

---

## 📋 3 Use Cases chính

### Use Case 1: Phân tích 1 mã cổ phiếu

```python
from src.hybrid_sentiment import EnhancedSentimentPipeline

pipeline = EnhancedSentimentPipeline(use_finbert=False)

# Get news for VNM
from src.news_service import get_recent_news
news_df = get_recent_news(symbol='VNM', days=30)

# Analyze
news_analyzed, daily_sentiment = pipeline.process_news_dataframe(
    news_df, 
    text_col='summary'
)

# View results
print(f"Average sentiment: {daily_sentiment['daily_sentiment_mean'].mean():.2f}")
print(f"Total news: {len(news_analyzed)}")
```

**Output**:
```
Average sentiment: 0.18
Total news: 45
```

### Use Case 2: So sánh nhiều mã

```python
symbols = ['VNM', 'VIC', 'HPG', 'FPT', 'MWG']
results = {}

for symbol in symbols:
    news = get_recent_news(symbol=symbol, days=7)
    _, daily = pipeline.process_news_dataframe(news, text_col='summary')
    
    results[symbol] = {
        'avg_sentiment': daily['daily_sentiment_mean'].mean(),
        'news_count': len(news)
    }

# Sort by sentiment
import pandas as pd
df_results = pd.DataFrame(results).T.sort_values('avg_sentiment', ascending=False)
print(df_results)
```

**Output**:
```
     avg_sentiment  news_count
VIC          0.45          12
FPT          0.32          18
VNM          0.18          15
HPG         -0.05          22
MWG         -0.23           9
```

### Use Case 3: API Integration

```python
# POST request
import requests

response = requests.post(
    'http://localhost:8000/api/ml/sentiment',
    json={
        'texts': [
            'Vinamilk công bố lợi nhuận tăng 25%',
            'Thị trường chứng khoán sụt giảm mạnh'
        ]
    }
)

results = response.json()['results']
for i, r in enumerate(results):
    print(f"Text {i+1}: {r['sentiment']} (score: {r['sentiment_score']})")
```

**Output**:
```
Text 1: neutral (score: 0.0)
Text 2: negative (score: -1.0)
```

---

## ⚙️ Configuration

### Config 1: Keyword-based only (Fast)

```python
from src.hybrid_sentiment import HybridSentimentAnalyzer

analyzer = HybridSentimentAnalyzer(
    use_finbert=False  # Không load FinBERT
)

result = analyzer.analyze("Cổ phiếu tăng mạnh", method='keyword')
```

**Performance**: 2ms/text, 50MB RAM

### Config 2: Hybrid (Recommended)

```python
analyzer = HybridSentimentAnalyzer(
    use_finbert=True,  # Load FinBERT for English
    finbert_model='ProsusAI/finbert'
)

result = analyzer.analyze("Stock surges", method='auto')
```

**Performance**: 
- Vietnamese: 2ms/text (keyword)
- English: 50ms/text (FinBERT)
- RAM: 500MB

### Config 3: GPU acceleration

```python
import torch

# Check GPU
print(f"GPU available: {torch.cuda.is_available()}")

# Auto use GPU if available
analyzer = HybridSentimentAnalyzer(use_finbert=True)
# Will automatically use GPU
```

**Performance**: 10ms/text (GPU), 200MB VRAM

---

## 📊 Outputs

### Output 1: News-level analysis

**File**: `VNM_news_20241203_021846.csv`

```csv
date,symbol,title,summary,sentiment,sentiment_score,positive,negative,neutral,confidence,method
2024-12-01,VNM,"Vinamilk lợi nhuận...","Lợi nhuận sau thuế...",neutral,0.0,0.0,0.0,1.0,0.0,keyword-based
2024-12-01,VNM,"Thị trường sữa...","Sữa tươi tăng...",positive,1.0,1.0,0.0,0.0,1.0,keyword-based
```

**Columns**:
- `sentiment`: positive / neutral / negative
- `sentiment_score`: [-1, 1]
- `confidence`: [0, 1]
- `method`: keyword-based / finbert

### Output 2: Daily aggregation

**File**: `VNM_daily_20241203_021846.csv`

```csv
date,symbol,daily_sentiment_mean,daily_sentiment_std,positive_count,negative_count,neutral_count,news_count
2024-12-01,VNM,0.25,0.45,2,1,2,5
2024-12-02,VNM,-0.15,0.30,1,2,0,3
2024-12-03,VNM,0.40,0.25,4,0,3,7
```

**Columns**:
- `daily_sentiment_mean`: Average sentiment per day
- `daily_sentiment_std`: Volatility
- `news_count`: Number of news

### Output 3: API response

```json
{
  "sentiment": "negative",
  "sentiment_score": -1.0,
  "positive": 0.0,
  "negative": 1.0,
  "neutral": 0.0,
  "confidence": 1.0,
  "method": "keyword-based",
  "explanation": "🔻 Tín hiệu GIẢM MẠNH - Khuyến nghị BÁN"
}
```

---

## 🛠️ Common Tasks

### Task 1: Analyze all major stocks

```bash
python scripts/run_sentiment_pipeline.py --all --days 30
```

Will analyze: VNM, VIC, HPG, FPT, MWG, VCB, BID, CTG, VHM, MSN, VRE, PLX, GAS, POW, TCB, MBB, ACB, SSI

### Task 2: Daily automated analysis

```python
# schedule_sentiment.py
import schedule
import time
from scripts.run_sentiment_pipeline import run_pipeline_for_symbol

def daily_analysis():
    symbols = ['VNM', 'VIC', 'HPG']
    for symbol in symbols:
        run_pipeline_for_symbol(symbol, days=1, save_to_db=True)
        print(f"✓ Analyzed {symbol}")

# Run every day at 9 PM
schedule.every().day.at("21:00").do(daily_analysis)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Task 3: Export to database

```python
from src.database import save_sentiment_to_db

# After analysis
save_sentiment_to_db(news_analyzed, daily_sentiment)
```

**Schema**:
```sql
CREATE TABLE analyzed_news (
    id SERIAL PRIMARY KEY,
    date DATE,
    symbol VARCHAR(10),
    title TEXT,
    sentiment VARCHAR(20),
    sentiment_score FLOAT,
    confidence FLOAT
);

CREATE TABLE daily_sentiment (
    id SERIAL PRIMARY KEY,
    date DATE,
    symbol VARCHAR(10),
    daily_sentiment_mean FLOAT,
    news_count INT
);
```

---

## 🔍 Debugging

### Check 1: Model loaded?

```python
from src.hybrid_sentiment import HybridSentimentAnalyzer

analyzer = HybridSentimentAnalyzer(use_finbert=True)
print(f"FinBERT loaded: {analyzer.finbert_analyzer is not None}")
```

### Check 2: GPU working?

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Check 3: Keywords working?

```python
analyzer = HybridSentimentAnalyzer(use_finbert=False)

test_cases = [
    "tăng trưởng mạnh",  # Should be positive
    "sụt giảm",          # Should be negative
    "ổn định"            # Should be neutral
]

for text in test_cases:
    result = analyzer.analyze(text, method='keyword')
    print(f"{text}: {result['sentiment']} ({result['sentiment_score']})")
```

**Expected**:
```
tăng trưởng mạnh: positive (1.0)
sụt giảm: negative (-1.0)
ổn định: neutral (0.0)
```

---

## 📈 Performance Tips

### Tip 1: Batch processing

```python
# SLOW: Process one by one
for text in texts:
    result = analyzer.analyze(text)

# FAST: Batch processing
results = analyzer.analyze_batch(texts, batch_size=16)
```

**Speed improvement**: 10x faster

### Tip 2: Use keyword-based for Vietnamese

```python
# SLOW: FinBERT for Vietnamese (all neutral)
analyzer = HybridSentimentAnalyzer(use_finbert=True)
result = analyzer.analyze("tăng trưởng", method='finbert')

# FAST: Keyword-based (accurate)
result = analyzer.analyze("tăng trưởng", method='keyword')
```

**Speed improvement**: 25x faster

### Tip 3: Cache results

```python
import pickle

# Save results
with open('sentiment_cache.pkl', 'wb') as f:
    pickle.dump(news_analyzed, f)

# Load later
with open('sentiment_cache.pkl', 'rb') as f:
    news_analyzed = pickle.load(f)
```

---

## ❓ FAQ

### Q1: Tại sao FinBERT trả về toàn neutral cho tiếng Việt?

**A**: FinBERT được train trên dữ liệu tiếng Anh. Dùng keyword-based cho tiếng Việt:

```python
analyzer = HybridSentimentAnalyzer(use_finbert=False)
```

### Q2: Làm sao thêm keywords mới?

**A**: Edit `src/hybrid_sentiment.py`:

```python
POSITIVE_KEYWORDS = [
    # ... existing keywords ...
    'phát triển bền vững',  # Add new
    'đột phá công nghệ',    # Add new
]
```

### Q3: Pipeline chạy chậm, làm sao?

**A**: 3 solutions:
1. Dùng keyword-based: `use_finbert=False`
2. Giảm batch size: `batch_size=4`
3. Dùng GPU: Install CUDA + cuDNN

### Q4: Làm sao deploy lên production?

**A**: Railway deployment:

```bash
# 1. Commit code
git add -A
git commit -m "Deploy sentiment pipeline"
git push origin main

# 2. Railway auto-deploy
# Check: https://your-app.railway.app/api/ml/sentiment

# 3. Test endpoint
curl -X POST https://your-app.railway.app/api/ml/sentiment \
  -H "Content-Type: application/json" \
  -d '{"texts": ["test"]}'
```

---

## 📞 Support

- **Documentation**: `docs/sentiment_pipeline_guide.md`
- **Code**: `src/hybrid_sentiment.py`, `scripts/run_sentiment_pipeline.py`
- **Issues**: GitHub Issues
- **Author**: Le Minh Man

---

## ✅ Next Steps

1. **Test with your data**:
   ```bash
   python scripts/run_sentiment_pipeline.py YOUR_SYMBOL --days 30
   ```

2. **Integrate with ML model**:
   - Read `docs/sentiment_pipeline_guide.md` → Section "Bước 7"
   - Use sentiment features in your price prediction

3. **Monitor performance**:
   - Check daily sentiment scores
   - Alert if strong negative sentiment
   - Backtest trading strategy

---

**Good luck! 🚀**
