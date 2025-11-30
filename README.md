# Stock Price Prediction System 📈

Hệ thống dự đoán giá cổ phiếu toàn diện sử dụng AI/ML, kết hợp nhiều mô hình time-series, sentiment analysis, và ensemble learning.

## ✨ Tính năng chính

### 1. 📊 Thu thập dữ liệu đa nguồn
- **API Integration**: Yahoo Finance, Alpha Vantage, VNDirect
- **Web Scraping**: BeautifulSoup và Scrapy cho tin tức tài chính
- **Real-time data**: Cập nhật dữ liệu liên tục

### 2. 🔄 ETL Pipeline
- Extract: Trích xuất từ nhiều nguồn
- Transform: Làm sạch, chuẩn hóa dữ liệu
- Load: Lưu trữ vào database/file

### 3. ⚙️ Feature Engineering
- **Technical Indicators**: 20+ chỉ báo kỹ thuật
  - Moving Averages (SMA, EMA)
  - RSI, MACD, Bollinger Bands
  - Stochastic Oscillator, ATR, ADX
  - CCI, Williams %R, OBV
- **Sentiment Features**: Điểm cảm tính từ tin tức

### 4. 💭 Sentiment Analysis
- **FinBERT Model**: Fine-tuned BERT cho tài chính
- Phân tích tin tức: Positive, Negative, Neutral
- Tổng hợp sentiment theo ngày
- Tích hợp vào mô hình dự đoán

### 5. 🤖 Multiple Time-Series Models

#### ARIMA (AutoRegressive Integrated Moving Average)
- **Ưu điểm**: Đơn giản, dễ giải thích, không cần nhiều dữ liệu
- **Nhược điểm**: Chỉ phù hợp với dữ liệu tuyến tính, stationary
- **Use case**: Dự đoán ngắn hạn, dữ liệu ổn định

#### Prophet (Facebook)
- **Ưu điểm**: Xử lý tốt seasonality, missing data, outliers
- **Nhược điểm**: Chậm hơn ARIMA, cần nhiều dữ liệu
- **Use case**: Dự đoán trung-dài hạn, có seasonality

#### LSTM (Long Short-Term Memory)
- **Ưu điểm**: Học được long-term dependencies, phức tạp
- **Nhược điểm**: Cần nhiều dữ liệu, training lâu, dễ overfit
- **Use case**: Dữ liệu phi tuyến, nhiều features

#### GRU (Gated Recurrent Unit)
- **Ưu điểm**: Nhanh hơn LSTM, ít parameters
- **Nhược điểm**: Có thể kém hơn LSTM với long sequences
- **Use case**: Alternative cho LSTM khi cần tốc độ

### 6. 🎯 Ensemble Learning với Meta-Learning

#### Simple Average Ensemble
- Trung bình của tất cả models
- Không cần training

#### Weighted Average Ensemble
- Weights dựa trên validation performance
- Tự động tối ưu weights

#### Stacking (Meta-Learning) ⭐
- Level 0: Base models (ARIMA, Prophet, LSTM, GRU)
- Level 1: Meta-model (Ridge, Random Forest, MLP)
- Học cách kết hợp tối ưu các models
- **Thường cho kết quả tốt nhất**

### 7. 🔄 Backtesting Engine

- **Chiến lược giao dịch**:
  - Long Only
  - Long-Short
  - Threshold-based
  
- **Risk Management**:
  - Stop Loss
  - Take Profit
  - Position sizing

- **Metrics**:
  - Total Return
  - Sharpe Ratio
  - Max Drawdown
  - Win Rate
  - Average P&L

### 8. 🌐 Web Application & API

#### FastAPI Backend
- RESTful API endpoints
- Data collection
- Predictions
- Backtesting
- Model training

#### Streamlit Frontend
- Interactive dashboard
- Real-time predictions
- Backtesting visualization
- Model comparison
- Sentiment analysis dashboard

## 🚀 Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd KLTN
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure (Optional)
Create `.env` file:
```
ALPHA_VANTAGE_API_KEY=your_key_here
```

## 📖 Usage

### 1. Command Line Interface

#### Full Pipeline
```bash
python main.py --symbol VNM.VN --mode full
```

#### Prediction Only
```bash
python main.py --symbol AAPL --mode predict
```

#### Backtesting
```bash
python main.py --symbol VIC.VN --mode backtest --start-date 2023-01-01
```

### 2. Web Application

#### Start Streamlit UI
```bash
streamlit run src/web_app.py
```
Navigate to: `http://localhost:8501`

#### Start FastAPI Backend
```bash
python src/api.py
```
API Documentation: `http://localhost:8000/docs`

### 3. Python API

```python
from main import StockPredictionSystem

# Initialize
system = StockPredictionSystem({
    'symbols': ['AAPL'],
    'start_date': '2022-01-01',
    'models_to_train': ['arima', 'prophet', 'lstm'],
    'ensemble_type': 'stacking'
})

# Run full pipeline
results = system.run_full_pipeline('AAPL')

# Access results
print(f"Models: {results['models'].keys()}")
print(f"Backtest: {results['backtest_results']}")
print(f"Predictions: {results['predictions']}")
```

## 📂 Project Structure

```
KLTN/
├── data/
│   ├── raw_data.csv
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── data_collection.py      # API & Web Scraping
│   ├── data_processing.py      # ETL Pipeline
│   ├── analysis.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical_indicators.py
│   │   └── sentiment_analysis.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── arima_model.py
│   │   ├── prophet_model.py
│   │   ├── lstm_gru_models.py
│   │   └── ensemble.py
│   ├── backtesting.py
│   ├── api.py                   # FastAPI Backend
│   └── web_app.py              # Streamlit Frontend
├── main.py                      # Main Orchestration
├── requirements.txt
└── README.md
```

## 🔬 Models Comparison

| Model | Training Time | Prediction Speed | Accuracy | Complexity |
|-------|--------------|------------------|----------|------------|
| ARIMA | Fast ⚡ | Fast ⚡ | Medium | Low |
| Prophet | Medium ⏱️ | Medium ⏱️ | Good | Medium |
| LSTM | Slow 🐌 | Fast ⚡ | Very Good | High |
| GRU | Medium ⏱️ | Fast ⚡ | Good | High |
| **Ensemble** | Slow 🐌 | Medium ⏱️ | **Best** ⭐ | High |

## 📊 Performance Metrics

### Evaluation Metrics
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Trading Metrics
- **Total Return %**
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown %**: Largest peak-to-trough decline
- **Win Rate %**: Percentage of profitable trades

## 🎓 Research & Theory

### ARIMA
- **Paper**: Box, G. E., & Jenkins, G. M. (1976)
- **Theory**: AR(p) + I(d) + MA(q)
- **Stationarity**: Dickey-Fuller test

### Prophet
- **Paper**: Taylor & Letham (2018)
- **Components**: Trend + Seasonality + Holidays + Error

### LSTM/GRU
- **LSTM**: Hochreiter & Schmidhuber (1997)
- **GRU**: Cho et al. (2014)
- **Gates**: Forget, Input, Output (LSTM); Reset, Update (GRU)

### FinBERT
- **Base**: BERT (Devlin et al., 2019)
- **Fine-tuned**: Financial news corpus
- **Output**: Positive, Negative, Neutral

### Meta-Learning (Stacking)
- **Wolpert (1992)**: Stacked Generalization
- **Level 0**: Diverse base learners
- **Level 1**: Meta-learner combines predictions

## ⚠️ Important Notes

### Data Requirements
- **Minimum**: 2 years historical data
- **Recommended**: 5+ years for LSTM/GRU
- **Frequency**: Daily data works best

### Model Selection
- **Short-term (1-7 days)**: ARIMA, Ensemble
- **Medium-term (1-3 months)**: Prophet, LSTM, Ensemble
- **Long-term (3+ months)**: Prophet, Ensemble

### Computational Requirements
- **CPU**: Multi-core recommended for training
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but speeds up LSTM/GRU training
- **Storage**: ~1GB for models and data

## 🔧 Configuration

Edit `main.py` or create custom config:

```python
config = {
    'symbols': ['AAPL', 'GOOGL', 'MSFT'],
    'start_date': '2020-01-01',
    'end_date': '2024-01-01',
    'train_split': 0.8,
    'models_to_train': ['arima', 'prophet', 'lstm', 'gru'],
    'ensemble_type': 'stacking',  # 'average', 'weighted', 'stacking'
    'backtest_strategy': 'long_only',
    'initial_capital': 100000,
}
```

## 📝 TODO / Future Improvements

- [ ] Add more data sources (Bloomberg, Reuters)
- [ ] Implement reinforcement learning for trading
- [ ] Add cryptocurrency support
- [ ] Real-time prediction updates
- [ ] Portfolio optimization
- [ ] Risk management advanced features
- [ ] Mobile app
- [ ] Cloud deployment

## 📚 References

1. Box, G. E., & Jenkins, G. M. (1976). Time series analysis: forecasting and control
2. Taylor, S. J., & Letham, B. (2018). Forecasting at scale
3. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory
4. Araci, D. (2019). FinBERT: Financial Sentiment Analysis with Pre-trained Language Models
5. Wolpert, D. H. (1992). Stacked generalization

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 👨‍💻 Author

Khóa luận tốt nghiệp - KLTN 2024

## 📞 Contact

For questions or support, please open an issue or contact via email.

---

**⚠️ Disclaimer**: This system is for educational and research purposes only. Do not use for actual trading without proper validation and risk management. Past performance does not guarantee future results.
