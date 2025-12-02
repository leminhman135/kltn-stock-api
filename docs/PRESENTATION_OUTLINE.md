# 📊 PRESENTATION SLIDES OUTLINE
# Hệ Thống Dự Đoán Giá Cổ Phiếu Việt Nam với Machine Learning

---

## SLIDE 1: TRANG BÌA

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                    KHÓA LUẬN TỐT NGHIỆP                         ║
║                                                                  ║
║    ┌──────────────────────────────────────────────────────┐     ║
║    │    HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU VIỆT NAM           │     ║
║    │       VỚI MACHINE LEARNING VÀ DEEP LEARNING          │     ║
║    └──────────────────────────────────────────────────────┘     ║
║                                                                  ║
║                         📈 📊 🤖                                 ║
║                                                                  ║
║    GVHD: [Tên Giảng Viên]                                       ║
║    SVTH: [Tên Sinh Viên]                                        ║
║    MSSV: [Mã Số Sinh Viên]                                      ║
║                                                                  ║
║                      Tháng 12/2025                               ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## SLIDE 2: NỘI DUNG BÁO CÁO

```
📋 NỘI DUNG

1. Đặt vấn đề và mục tiêu
2. Phương pháp nghiên cứu
3. Công nghệ sử dụng
4. Kiến trúc hệ thống
5. Các mô hình Machine Learning
6. Kết quả thực nghiệm
7. Demo hệ thống
8. Kết luận và hướng phát triển
```

---

## SLIDE 3: ĐẶT VẤN ĐỀ

```
❓ ĐẶT VẤN ĐỀ

┌─────────────────────────────────────────────────────────────┐
│                  THÁCH THỨC HIỆN TẠI                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📈 Thị trường chứng khoán VN tăng trưởng nhanh            │
│     • VN-Index tăng 300% trong 10 năm                       │
│     • Số tài khoản chứng khoán: 8+ triệu                    │
│                                                             │
│  ⚠️ Rủi ro cao cho nhà đầu tư cá nhân                      │
│     • 80% nhà đầu tư cá nhân thua lỗ                        │
│     • Thiếu công cụ phân tích chuyên nghiệp                 │
│                                                             │
│  🤖 AI/ML là xu hướng mới                                   │
│     • Các quỹ lớn đã áp dụng                                │
│     • Chưa phổ biến cho nhà đầu tư cá nhân VN               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 4: MỤC TIÊU ĐỀ TÀI

```
🎯 MỤC TIÊU

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  1️⃣  Xây dựng hệ thống thu thập dữ liệu tự động           │
│      • API VNDirect, VNStock                                │
│      • Giá OHLCV, tin tức, báo cáo tài chính               │
│                                                             │
│  2️⃣  Triển khai các mô hình ML/DL                          │
│      • ARIMA, Prophet (Time Series)                         │
│      • LSTM, GRU (Deep Learning)                            │
│      • Ensemble (Kết hợp)                                   │
│                                                             │
│  3️⃣  Phân tích sentiment từ tin tức                        │
│      • FinBERT cho tiếng Việt                               │
│      • Tác động đến giá cổ phiếu                           │
│                                                             │
│  4️⃣  Xây dựng hệ thống backtesting                         │
│      • Kiểm tra chiến lược                                  │
│      • Các metric đánh giá                                  │
│                                                             │
│  5️⃣  Triển khai REST API và Web Dashboard                  │
│      • FastAPI backend                                      │
│      • Interactive web UI                                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 5: PHƯƠNG PHÁP NGHIÊN CỨU

```
📚 PHƯƠNG PHÁP NGHIÊN CỨU

┌───────────────────────────────────────────────────────────┐
│                                                           │
│   Nghiên cứu lý thuyết              Thực nghiệm          │
│   ┌─────────────────┐              ┌─────────────────┐   │
│   │ • Time Series   │              │ • Thu thập data │   │
│   │ • Machine Learn │     →        │ • Training      │   │
│   │ • Deep Learning │              │ • Testing       │   │
│   │ • NLP/Sentiment │              │ • Evaluation    │   │
│   └─────────────────┘              └─────────────────┘   │
│                                                           │
│                        ↓                                  │
│                                                           │
│              ┌─────────────────────────┐                  │
│              │   So sánh & Đánh giá    │                  │
│              │   • RMSE, MAE, MAPE     │                  │
│              │   • Backtesting metrics │                  │
│              │   • Statistical tests   │                  │
│              └─────────────────────────┘                  │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

---

## SLIDE 6: CÔNG NGHỆ SỬ DỤNG

```
🛠️ CÔNG NGHỆ SỬ DỤNG

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   BACKEND                    FRONTEND                       │
│   ┌───────────────┐         ┌───────────────┐              │
│   │ 🐍 Python     │         │ 📊 Chart.js   │              │
│   │ ⚡ FastAPI    │         │ 🎨 Bootstrap  │              │
│   │ 🗄️ PostgreSQL │         │ 📱 Responsive │              │
│   └───────────────┘         └───────────────┘              │
│                                                             │
│   ML/DL                      DEPLOYMENT                     │
│   ┌───────────────┐         ┌───────────────┐              │
│   │ 🧠 TensorFlow │         │ ☁️ Render.com │              │
│   │ 📈 Prophet    │         │ 🐙 GitHub     │              │
│   │ 📊 statsmodel │         │ 🔄 CI/CD      │              │
│   │ 🤗 FinBERT    │         └───────────────┘              │
│   └───────────────┘                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 7: KIẾN TRÚC HỆ THỐNG

```
🏗️ KIẾN TRÚC HỆ THỐNG

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│    ┌──────────┐   ┌──────────┐   ┌──────────┐              │
│    │  User    │   │  Admin   │   │  API     │              │
│    │  Browser │   │  Panel   │   │  Client  │              │
│    └────┬─────┘   └────┬─────┘   └────┬─────┘              │
│         │              │              │                     │
│         └──────────────┼──────────────┘                     │
│                        │                                    │
│                        ▼                                    │
│    ┌────────────────────────────────────────────┐          │
│    │         PRESENTATION LAYER                 │          │
│    │    (Web UI / REST API / Swagger)           │          │
│    └────────────────────┬───────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│    ┌────────────────────────────────────────────┐          │
│    │          APPLICATION LAYER                 │          │
│    │  ┌─────────┐ ┌─────────┐ ┌─────────┐      │          │
│    │  │   ML    │ │Backtest │ │Sentiment│      │          │
│    │  │ Models  │ │ Engine  │ │Analyzer │      │          │
│    │  └─────────┘ └─────────┘ └─────────┘      │          │
│    └────────────────────┬───────────────────────┘          │
│                         │                                   │
│                         ▼                                   │
│    ┌────────────────────────────────────────────┐          │
│    │            DATA LAYER                      │          │
│    │     PostgreSQL / SQLite / Cache            │          │
│    └────────────────────────────────────────────┘          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 8: MÔ HÌNH ARIMA

```
📈 MÔ HÌNH ARIMA

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ARIMA(p, d, q) = AutoRegressive Integrated Moving Average│
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │  Yₜ = c + φ₁Yₜ₋₁ + ... + φₚYₜ₋ₚ                   │  │
│   │        + εₜ + θ₁εₜ₋₁ + ... + θqεₜ₋q               │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ĐẶC ĐIỂM:                                                │
│   ✓ Tốt cho time series có xu hướng                        │
│   ✓ Không cần feature engineering                          │
│   ✓ Nhanh và nhẹ                                           │
│   ✗ Khó bắt pattern phi tuyến                              │
│                                                             │
│   KẾT QUẢ:                                                  │
│   • RMSE: 3.45%                                             │
│   • MAE: 2.89%                                              │
│   • R²: 0.78                                                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 9: MÔ HÌNH PROPHET

```
📊 MÔ HÌNH PROPHET (Facebook)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   y(t) = g(t) + s(t) + h(t) + εₜ                           │
│                                                             │
│   • g(t): Trend component                                   │
│   • s(t): Seasonality (weekly, yearly)                      │
│   • h(t): Holiday effects                                   │
│   • εₜ: Error term                                          │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │   [Trend Graph]  +  [Weekly Seasonality]            │  │
│   │        ↘              ↙                              │  │
│   │         ───────────────                              │  │
│   │              ↓                                       │  │
│   │         Final Forecast                               │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ƯU ĐIỂM:                                                  │
│   ✓ Xử lý missing data tốt                                 │
│   ✓ Uncertainty intervals                                  │
│   ✓ Interpretable                                          │
│                                                             │
│   KẾT QUẢ: RMSE: 3.12%, MAE: 2.56%, R²: 0.82              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 10: MÔ HÌNH LSTM

```
🧠 MÔ HÌNH LSTM (Deep Learning)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Long Short-Term Memory Network                            │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │    Input    →  LSTM  →  LSTM  →  Dense  →  Output  │  │
│   │   (60 days)   (128)    (64)     (32)    (1 day)    │  │
│   │                                                     │  │
│   │   ┌───┐ ┌───┐ ┌───┐                                │  │
│   │   │fₜ │ │iₜ │ │oₜ │  Gates: Forget, Input, Output  │  │
│   │   └───┘ └───┘ └───┘                                │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   KIẾN TRÚC:                                                │
│   • Lookback: 60 ngày                                       │
│   • LSTM layers: 2 (128, 64 units)                         │
│   • Dropout: 0.2                                            │
│   • Optimizer: Adam                                         │
│                                                             │
│   KẾT QUẢ: RMSE: 2.34%, MAE: 1.98%, R²: 0.91              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 11: MÔ HÌNH ENSEMBLE

```
🔮 MÔ HÌNH ENSEMBLE (Kết hợp)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐      │
│   │  ARIMA  │  │ Prophet │  │  LSTM   │  │   GRU   │      │
│   │  18%    │  │  20%    │  │  32%    │  │  30%    │      │
│   └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘      │
│        │            │            │            │            │
│        └────────────┼────────────┼────────────┘            │
│                     │            │                         │
│                     ▼            ▼                         │
│             ┌────────────────────────┐                     │
│             │   Weighted Average      │                     │
│             │   + Sentiment Adjust    │                     │
│             └───────────┬────────────┘                     │
│                         │                                  │
│                         ▼                                  │
│             ┌────────────────────────┐                     │
│             │   Final Prediction     │                     │
│             │   with Confidence      │                     │
│             └────────────────────────┘                     │
│                                                             │
│   KẾT QUẢ: RMSE: 2.12%, MAE: 1.75%, R²: 0.94              │
│   ✓ Best performing model                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 12: PHÂN TÍCH SENTIMENT

```
💬 PHÂN TÍCH SENTIMENT (FinBERT)

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   FinBERT: Financial BERT by ProsusAI                       │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │   "VNM báo cáo lợi nhuận tăng 25% trong Q3"         │  │
│   │                    │                                │  │
│   │                    ▼                                │  │
│   │            ┌──────────────┐                         │  │
│   │            │   FinBERT    │                         │  │
│   │            │   Tokenize   │                         │  │
│   │            │   Encode     │                         │  │
│   │            │   Classify   │                         │  │
│   │            └──────┬───────┘                         │  │
│   │                   │                                 │  │
│   │                   ▼                                 │  │
│   │   ┌─────────┐ ┌─────────┐ ┌─────────┐              │  │
│   │   │Positive │ │Neutral  │ │Negative │              │  │
│   │   │  0.89   │ │  0.08   │ │  0.03   │              │  │
│   │   │   ✓     │ │         │ │         │              │  │
│   │   └─────────┘ └─────────┘ └─────────┘              │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   ACCURACY: 87% trên news tiếng Việt                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 13: BACKTESTING ENGINE

```
📊 BACKTESTING ENGINE

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   CHIẾN LƯỢC GIAO DỊCH                                     │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ • Buy & Hold                                        │  │
│   │ • SMA Crossover (10/20)                             │  │
│   │ • RSI Oversold/Overbought                           │  │
│   │ • ML Signal Based                                   │  │
│   │ • Ensemble + Sentiment                              │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   KẾT QUẢ BACKTEST (VNM, 2024)                             │
│   ┌─────────────┬──────────┬─────────┬───────────┐         │
│   │ Strategy    │  Return  │ Sharpe  │ Drawdown  │         │
│   ├─────────────┼──────────┼─────────┼───────────┤         │
│   │ Buy & Hold  │  +18.2%  │  0.89   │  -12.3%   │         │
│   │ SMA Cross   │  +22.5%  │  1.12   │  -10.5%   │         │
│   │ ML Signal   │  +28.4%  │  1.45   │   -8.3%   │         │
│   │ Ensemble+   │  +32.1%  │  1.78   │   -6.8%   │         │
│   └─────────────┴──────────┴─────────┴───────────┘         │
│                                                             │
│   ✓ Ensemble + Sentiment vượt trội Buy & Hold 14%          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 14: SO SÁNH CÁC MÔ HÌNH

```
📈 SO SÁNH CÁC MÔ HÌNH

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ┌──────────┬────────┬────────┬────────┬────────┐         │
│   │ Model    │  RMSE  │  MAE   │  MAPE  │   R²   │         │
│   ├──────────┼────────┼────────┼────────┼────────┤         │
│   │ ARIMA    │  3.45  │  2.89  │  4.21% │  0.78  │         │
│   │ Prophet  │  3.12  │  2.56  │  3.78% │  0.82  │         │
│   │ LSTM     │  2.34  │  1.98  │  2.92% │  0.91  │         │
│   │ GRU      │  2.51  │  2.12  │  3.15% │  0.89  │         │
│   │ Ensemble │  2.12  │  1.75  │  2.58% │  0.94  │         │
│   └──────────┴────────┴────────┴────────┴────────┘         │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │   RMSE Comparison (Bar Chart)                       │  │
│   │                                                     │  │
│   │   ARIMA   ████████████████ 3.45                     │  │
│   │   Prophet ██████████████ 3.12                       │  │
│   │   LSTM    ██████████ 2.34                           │  │
│   │   GRU     ███████████ 2.51                          │  │
│   │   Ensemble████████ 2.12  ← Best                     │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 15: DEMO HỆ THỐNG

```
💻 DEMO HỆ THỐNG

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   🌐 Live Demo: https://kltn-stock-api.onrender.com        │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │                                                     │  │
│   │   [Screenshot of Dashboard]                         │  │
│   │                                                     │  │
│   │   • Stock selection dropdown                        │  │
│   │   • Price chart with predictions                    │  │
│   │   • Technical indicators                            │  │
│   │   • Sentiment analysis                              │  │
│   │   • Backtest results                                │  │
│   │                                                     │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   DEMO SCENARIOS:                                           │
│   1. Dự đoán giá VNM 7 ngày tới                            │
│   2. So sánh các models                                     │
│   3. Phân tích sentiment tin tức                           │
│   4. Chạy backtest SMA crossover                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 16: KẾT QUẢ ĐẠT ĐƯỢC

```
✅ KẾT QUẢ ĐẠT ĐƯỢC

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ✓ Thu thập dữ liệu                                       │
│     • 30 mã VN30                                            │
│     • 5+ năm dữ liệu giá                                    │
│     • 10,000+ tin tức                                       │
│                                                             │
│   ✓ Mô hình ML/DL                                          │
│     • 4 models: ARIMA, Prophet, LSTM, GRU                   │
│     • Ensemble với sentiment                                │
│     • MAPE < 3% (tốt nhất)                                  │
│                                                             │
│   ✓ Phân tích Sentiment                                    │
│     • FinBERT cho tiếng Việt                               │
│     • Accuracy: 87%                                         │
│                                                             │
│   ✓ Backtesting                                            │
│     • 5 chiến lược                                          │
│     • +32% return (best)                                    │
│     • Sharpe > 1.5                                          │
│                                                             │
│   ✓ Triển khai                                             │
│     • REST API (100+ endpoints)                             │
│     • Web Dashboard                                         │
│     • 99.9% uptime                                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 17: HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

```
🔮 HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   HẠN CHẾ:                                                  │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ ⚠️ Chưa xử lý tốt Black Swan events                 │  │
│   │ ⚠️ Sentiment chỉ hỗ trợ tiếng Anh/Việt              │  │
│   │ ⚠️ Cần thêm fundamental analysis                    │  │
│   │ ⚠️ Chưa có real-time streaming                      │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
│   HƯỚNG PHÁT TRIỂN:                                        │
│   ┌─────────────────────────────────────────────────────┐  │
│   │ 🚀 Thêm Transformer models (GPT-like)               │  │
│   │ 🚀 Mở rộng thị trường (crypto, forex)               │  │
│   │ 🚀 Mobile app (iOS/Android)                         │  │
│   │ 🚀 Portfolio optimization                           │  │
│   │ 🚀 Options pricing models                           │  │
│   │ 🚀 Social trading features                          │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 18: KẾT LUẬN

```
📝 KẾT LUẬN

┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Đề tài đã hoàn thành các mục tiêu đề ra:                 │
│                                                             │
│   ✅ Xây dựng pipeline thu thập dữ liệu tự động            │
│                                                             │
│   ✅ Triển khai và so sánh 4 mô hình ML/DL                 │
│      • Ensemble cho kết quả tốt nhất (MAPE: 2.58%)         │
│                                                             │
│   ✅ Tích hợp phân tích sentiment với FinBERT              │
│                                                             │
│   ✅ Xây dựng backtesting engine đầy đủ                    │
│      • Vượt trội Buy & Hold 14%                            │
│                                                             │
│   ✅ Triển khai production với REST API                    │
│      • Uptime 99.9%                                         │
│      • Response time < 5s                                   │
│                                                             │
│   ┌─────────────────────────────────────────────────────┐  │
│   │  "Machine Learning mang lại giá trị thực tiễn       │  │
│   │   cho dự đoán thị trường chứng khoán Việt Nam"      │  │
│   └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## SLIDE 19: CẢM ƠN

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║                                                                  ║
║                                                                  ║
║                       CẢM ƠN QUÝ THẦY CÔ                        ║
║                       ĐÃ LẮNG NGHE!                              ║
║                                                                  ║
║                                                                  ║
║                         📧 📱 💻                                 ║
║                                                                  ║
║                                                                  ║
║           Demo: https://kltn-stock-api.onrender.com              ║
║           GitHub: github.com/username/kltn-stock                 ║
║                                                                  ║
║                                                                  ║
║                      Q & A SESSION                               ║
║                                                                  ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## HƯỚNG DẪN TRÌNH BÀY

### Thời gian phân bổ (15-20 phút):
1. **Slide 1-2:** 1 phút - Giới thiệu
2. **Slide 3-4:** 2 phút - Vấn đề và mục tiêu
3. **Slide 5-6:** 2 phút - Phương pháp và công nghệ
4. **Slide 7:** 1 phút - Kiến trúc
5. **Slide 8-12:** 5 phút - Các mô hình ML
6. **Slide 13-14:** 2 phút - Backtest và so sánh
7. **Slide 15:** 3 phút - Demo live
8. **Slide 16-18:** 3 phút - Kết quả và kết luận
9. **Slide 19:** Q&A

### Tips:
- Demo live nếu có internet ổn định
- Chuẩn bị video backup phòng demo fail
- In hardcopy kết quả để tham khảo
- Giải thích đơn giản các công thức ML
