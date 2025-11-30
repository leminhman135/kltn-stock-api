# Quick Start Guide - Stock Prediction System 🚀

## Bước 1: Cài đặt môi trường

### Windows
```powershell
# Tạo virtual environment
python -m venv venv

# Kích hoạt
.\venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

### Linux/Mac
```bash
# Tạo virtual environment
python3 -m venv venv

# Kích hoạt
source venv/bin/activate

# Install packages
pip install -r requirements.txt
```

## Bước 2: Chạy hệ thống

### Option 1: Web Interface (Đơn giản nhất) 🌐

```bash
# Start Streamlit app
streamlit run src/web_app.py
```

Mở trình duyệt: `http://localhost:8501`

**Features:**
- ✅ Giao diện trực quan
- ✅ Không cần code
- ✅ Visualizations đẹp
- ✅ Phù hợp cho demo

### Option 2: API Backend 🔌

```bash
# Start FastAPI server
python src/api.py
```

API Docs: `http://localhost:8000/docs`

**Endpoints:**
- `POST /api/data/collect` - Thu thập dữ liệu
- `POST /api/predict` - Dự đoán
- `POST /api/backtest` - Backtest
- `POST /api/train` - Train models

### Option 3: Command Line 💻

```bash
# Full pipeline
python main.py --symbol VNM.VN --mode full

# Chỉ predict
python main.py --symbol AAPL --mode predict

# Backtest
python main.py --symbol HPG.VN --mode backtest
```

## Bước 3: Examples

### Example 1: Dự đoán giá Apple (AAPL)

**Web UI:**
1. Mở Streamlit app
2. Sidebar: Nhập `AAPL`
3. Chọn date range
4. Vào page "Price Prediction"
5. Click "Predict"

**Command Line:**
```bash
python main.py --symbol AAPL --start-date 2022-01-01 --mode full
```

### Example 2: Backtest chiến lược VNM

**Python Code:**
```python
from main import StockPredictionSystem

system = StockPredictionSystem({
    'symbols': ['VNM.VN'],
    'start_date': '2022-01-01',
    'models_to_train': ['arima', 'prophet'],
    'initial_capital': 100000
})

results = system.run_full_pipeline('VNM.VN')
print(results['backtest_results'])
```

### Example 3: So sánh nhiều models

```python
from models.arima_model import ARIMAModel
from models.prophet_model import ProphetModel

# Load data
from data_collection import YahooFinanceAPI
api = YahooFinanceAPI()
df = api.get_stock_data('AAPL', '2022-01-01', '2024-01-01')

# Train ARIMA
arima = ARIMAModel()
arima.fit(df['close'], auto_order=True)

# Train Prophet
prophet = ProphetModel()
prophet.fit(df['close'])

# Compare
arima_metrics = arima.evaluate(df['close'].tail(30))
prophet_metrics = prophet.evaluate(df['close'].tail(30))

print(f"ARIMA MAE: {arima_metrics['mae']:.2f}")
print(f"Prophet MAE: {prophet_metrics['mae']:.2f}")
```

## Bước 4: Tùy chỉnh

### Thay đổi models

Edit `main.py`:
```python
config = {
    'models_to_train': ['arima', 'prophet', 'lstm', 'gru'],  # Chọn models
    'ensemble_type': 'stacking',  # hoặc 'weighted', 'average'
}
```

### Thêm symbols

```python
config = {
    'symbols': ['AAPL', 'GOOGL', 'MSFT', 'VNM.VN', 'FPT.VN'],
}
```

### Điều chỉnh risk management

```python
config = {
    'initial_capital': 100000,
    'commission': 0.001,  # 0.1%
    'stop_loss': 0.05,    # 5%
    'take_profit': 0.10   # 10%
}
```

## Troubleshooting 🔧

### Lỗi: Module not found

```bash
# Đảm bảo đang ở đúng directory
cd D:\KLTN

# Activate venv
.\venv\Scripts\Activate.ps1

# Reinstall
pip install -r requirements.txt
```

### Lỗi: Data not found

**Giải pháp 1**: Kiểm tra internet connection

**Giải pháp 2**: Sử dụng VPN nếu bị chặn

**Giải pháp 3**: Thử symbol khác:
- US stocks: AAPL, GOOGL, MSFT
- VN stocks: VNM.VN, VIC.VN, HPG.VN

### Lỗi: FinBERT slow/error

FinBERT model lớn (~400MB) và cần download lần đầu.

**Giải pháp**: Chạy một lần để download model:
```python
from features.sentiment_analysis import FinBERTSentimentAnalyzer

# Sẽ download model lần đầu (có thể mất 5-10 phút)
analyzer = FinBERTSentimentAnalyzer()
```

Hoặc skip sentiment analysis:
```python
config = {
    # Không train sentiment
}
```

### Lỗi: LSTM/GRU training slow

Deep learning models cần nhiều thời gian.

**Giải pháp 1**: Giảm epochs
```python
model.fit(train_data, epochs=20, verbose=1)  # Thay vì 100
```

**Giải pháp 2**: Sử dụng CPU-only models (ARIMA, Prophet)
```python
config = {
    'models_to_train': ['arima', 'prophet'],  # Skip LSTM/GRU
}
```

**Giải pháp 3**: Sử dụng pre-trained models

## Performance Tips ⚡

### Tăng tốc độ

1. **Giảm data size**:
```python
config = {
    'start_date': '2023-01-01',  # Thay vì '2020-01-01'
}
```

2. **Chọn ít models**:
```python
config = {
    'models_to_train': ['arima', 'prophet'],  # Bỏ LSTM/GRU
}
```

3. **Sử dụng cache**:
```python
# Data sẽ được cache trong session
if 'data' in st.session_state:
    df = st.session_state['data']
```

### Cải thiện accuracy

1. **Thêm nhiều features**:
```python
from features.technical_indicators import TechnicalIndicators
ti = TechnicalIndicators()
df = ti.add_all_indicators(df)
```

2. **Sử dụng ensemble**:
```python
config = {
    'ensemble_type': 'stacking',  # Best performance
    'meta_model_type': 'ridge'
}
```

3. **Tune hyperparameters**:
```python
# LSTM
model = LSTMModel(
    lookback=90,      # Thử 60, 90, 120
    units=[100, 50],  # Thử [50,50], [100,50], [100,100,50]
    dropout=0.3       # Thử 0.2, 0.3, 0.4
)
```

## Next Steps 📚

### 1. Học về models
- ARIMA: `src/models/arima_model.py`
- Prophet: `src/models/prophet_model.py`
- LSTM/GRU: `src/models/lstm_gru_models.py`
- Ensemble: `src/models/ensemble.py`

### 2. Thử các strategies
- Long only
- Long-short
- Threshold-based
- Custom strategy

### 3. Tích hợp sentiment
```python
from features.sentiment_analysis import SentimentAnalysisPipeline

pipeline = SentimentAnalysisPipeline()
news_sentiment, daily_sentiment = pipeline.process_news(news_df)
```

### 4. Deploy production
- Containerize với Docker
- Deploy lên cloud (AWS, GCP, Azure)
- Setup monitoring và alerts
- Implement real-time updates

## Resources 📖

### Documentation
- Streamlit: https://docs.streamlit.io
- FastAPI: https://fastapi.tiangolo.com
- Prophet: https://facebook.github.io/prophet
- TensorFlow: https://www.tensorflow.org

### Papers
- ARIMA: Box & Jenkins (1976)
- Prophet: Taylor & Letham (2018)
- LSTM: Hochreiter & Schmidhuber (1997)
- FinBERT: Araci (2019)

### Tutorials
- Time Series: https://www.kaggle.com/learn/time-series
- Deep Learning: https://www.tensorflow.org/tutorials
- Financial ML: "Advances in Financial Machine Learning" by Marcos López de Prado

## Support 💬

Nếu gặp vấn đề:
1. Check README.md
2. Check code comments
3. Google error message
4. Ask ChatGPT/Copilot
5. Open GitHub issue

Happy Trading! 📈💰
