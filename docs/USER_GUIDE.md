# 📋 HƯỚNG DẪN SỬ DỤNG (User Guide)

## Giới Thiệu

Chào mừng bạn đến với **Hệ thống Dự đoán Giá Cổ phiếu Việt Nam**! 

Đây là hệ thống sử dụng Machine Learning và phân tích kỹ thuật để dự đoán xu hướng giá cổ phiếu trên thị trường chứng khoán Việt Nam.

---

## 1. Truy Cập Hệ Thống

### 1.1 Dashboard Web

**URL:** https://kltn-stock-api.onrender.com

![Dashboard Overview](images/dashboard.png)

### 1.2 API Documentation

- **Swagger UI:** https://kltn-stock-api.onrender.com/docs
- **ReDoc:** https://kltn-stock-api.onrender.com/redoc

---

## 2. Các Tính Năng Chính

### 2.1 Xem Thông Tin Cổ Phiếu

1. Truy cập Dashboard
2. Chọn mã cổ phiếu từ dropdown (VD: VNM, FPT, VIC...)
3. Xem thông tin:
   - Giá hiện tại
   - Biến động trong ngày
   - Khối lượng giao dịch
   - Biểu đồ giá lịch sử

### 2.2 Xem Chỉ Báo Kỹ Thuật

Hệ thống cung cấp các chỉ báo:

| Chỉ báo | Ý nghĩa |
|---------|---------|
| **SMA** (Simple Moving Average) | Xu hướng trung bình |
| **EMA** (Exponential Moving Average) | Xu hướng có trọng số |
| **RSI** (Relative Strength Index) | Quá mua/quá bán |
| **MACD** | Xu hướng và momentum |
| **Bollinger Bands** | Biến động giá |

**Cách đọc:**
- RSI > 70: Cổ phiếu đang bị quá mua
- RSI < 30: Cổ phiếu đang bị quá bán
- MACD cắt lên Signal: Tín hiệu mua
- MACD cắt xuống Signal: Tín hiệu bán

### 2.3 Dự Đoán Giá với ML

#### Bước 1: Chọn cổ phiếu
Nhập hoặc chọn mã cổ phiếu (VD: VNM)

#### Bước 2: Chọn model dự đoán
- **ARIMA**: Tốt cho dữ liệu có xu hướng rõ ràng
- **Prophet**: Xử lý tốt seasonality và holiday
- **LSTM**: Deep learning cho pattern phức tạp
- **GRU**: Nhanh hơn LSTM, hiệu quả tương đương
- **Ensemble**: Kết hợp tất cả (recommended)

#### Bước 3: Chọn số ngày dự đoán
Từ 1-30 ngày

#### Bước 4: Xem kết quả
- Biểu đồ dự đoán
- Độ tin cậy (confidence)
- Khuyến nghị mua/bán

### 2.4 Phân Tích Sentiment

Hệ thống phân tích tin tức tự động:

1. Thu thập tin tức từ nhiều nguồn
2. Phân tích bằng FinBERT (AI)
3. Đánh giá: Tích cực / Tiêu cực / Trung lập
4. Tác động đến giá cổ phiếu

**Ví dụ kết quả:**
```
📰 Tin tức VNM (5 tin gần nhất)
├── ✅ "Lợi nhuận Q3 tăng 15%" - Tích cực (0.89)
├── ✅ "Mở rộng thị trường xuất khẩu" - Tích cực (0.72)
├── ⚪ "Họp ĐHCĐ thường niên" - Trung lập (0.12)
├── ❌ "Chi phí nguyên liệu tăng" - Tiêu cực (-0.45)
└── ✅ "Hợp tác chiến lược mới" - Tích cực (0.81)

📊 Tổng quan: Tích cực (Điểm: 0.42)
💡 Khuyến nghị: Tin tức hỗ trợ xu hướng tăng giá
```

### 2.5 Backtesting

Kiểm tra chiến lược giao dịch với dữ liệu quá khứ:

1. Chọn cổ phiếu
2. Chọn khoảng thời gian (VD: 01/01/2024 - 30/11/2024)
3. Chọn chiến lược:
   - **Buy & Hold**: Mua và giữ
   - **SMA Crossover**: Giao cắt SMA
   - **ML Signal**: Theo tín hiệu ML
4. Nhập vốn ban đầu
5. Xem kết quả:
   - Tổng lợi nhuận
   - Sharpe Ratio
   - Max Drawdown
   - Tỷ lệ thắng

---

## 3. Sử Dụng API

### 3.1 Lấy Danh Sách Cổ Phiếu

```bash
curl https://kltn-stock-api.onrender.com/api/stocks
```

### 3.2 Lấy Giá Lịch Sử

```bash
curl "https://kltn-stock-api.onrender.com/api/prices/VNM?limit=30"
```

### 3.3 Dự Đoán Giá

```bash
# Dự đoán bằng Ensemble (khuyến nghị)
curl -X POST "https://kltn-stock-api.onrender.com/api/ml/ensemble/predict/VNM?days=7"
```

### 3.4 Phân Tích Sentiment

```bash
curl "https://kltn-stock-api.onrender.com/api/finbert/sentiment/VNM"
```

### 3.5 Python Example

```python
import requests

# Dự đoán giá VNM 7 ngày tới
response = requests.post(
    "https://kltn-stock-api.onrender.com/api/ml/ensemble/predict/VNM",
    params={"days": 7}
)

data = response.json()
print(f"Model: {data['model']}")
for pred in data['predictions']:
    print(f"  {pred['date']}: {pred['price']:,.0f} VND (±{(1-pred['confidence'])*100:.1f}%)")
```

---

## 4. Giải Thích Kết Quả

### 4.1 Các Metric Dự Đoán

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **RMSE** | Sai số trung bình | Càng nhỏ càng tốt |
| **MAE** | Sai số tuyệt đối | Càng nhỏ càng tốt |
| **MAPE** | Sai số phần trăm | < 5% là tốt |
| **R²** | Độ phù hợp | Gần 1 là tốt |

### 4.2 Các Metric Backtest

| Metric | Ý nghĩa | Tốt khi |
|--------|---------|---------|
| **Total Return** | Tổng lợi nhuận | > 0 |
| **Sharpe Ratio** | Lợi nhuận/rủi ro | > 1 |
| **Sortino Ratio** | Sharpe chỉ tính downside | > 1.5 |
| **Max Drawdown** | Giảm tối đa | > -20% |
| **Win Rate** | Tỷ lệ lệnh thắng | > 55% |

### 4.3 Khuyến Nghị

Hệ thống đưa ra 5 mức khuyến nghị:

| Mức | Điều kiện |
|-----|-----------|
| 🟢 **STRONG BUY** | ML + TA + Sentiment đều tích cực |
| 🟢 **BUY** | ML tích cực, TA hỗ trợ |
| ⚪ **HOLD** | Tín hiệu không rõ ràng |
| 🔴 **SELL** | ML tiêu cực, TA cảnh báo |
| 🔴 **STRONG SELL** | Tất cả tín hiệu đều tiêu cực |

---

## 5. Lưu Ý Quan Trọng

### ⚠️ Tuyên Bố Miễn Trừ Trách Nhiệm

> **Đây chỉ là công cụ hỗ trợ nghiên cứu và học tập.**
> 
> Các dự đoán và khuyến nghị không phải là lời khuyên đầu tư. 
> Người dùng tự chịu trách nhiệm với các quyết định đầu tư của mình.
> 
> Thị trường chứng khoán có rủi ro cao. Quá khứ không đảm bảo tương lai.

### 📌 Best Practices

1. **Không dựa hoàn toàn vào dự đoán ML** - Kết hợp phân tích fundamental
2. **Quản lý rủi ro** - Không đầu tư quá 10% vào một cổ phiếu
3. **Theo dõi thường xuyên** - Thị trường thay đổi liên tục
4. **Backtest trước khi trade thật** - Kiểm tra chiến lược với dữ liệu quá khứ
5. **Đa dạng hóa danh mục** - Không bỏ tất cả trứng vào một giỏ

---

## 6. Câu Hỏi Thường Gặp (FAQ)

### Q: Model nào chính xác nhất?
**A:** Ensemble thường cho kết quả tốt nhất vì kết hợp nhiều models. Tuy nhiên, tùy từng cổ phiếu có thể khác nhau.

### Q: Dự đoán bao xa là hợp lý?
**A:** Dự đoán 3-7 ngày thường chính xác hơn. Dự đoán > 14 ngày độ tin cậy giảm đáng kể.

### Q: Tại sao dự đoán khác với thực tế?
**A:** Thị trường bị ảnh hưởng bởi nhiều yếu tố bất ngờ: tin tức, sự kiện chính trị, biến động kinh tế toàn cầu...

### Q: Hệ thống cập nhật dữ liệu khi nào?
**A:** 
- Dữ liệu giá: Sau 15:00 mỗi ngày giao dịch
- Tin tức: Real-time
- Model retrain: Hàng tuần

### Q: Làm sao để truy cập API?
**A:** API hiện tại miễn phí với rate limit 100 requests/phút. Xem documentation tại `/docs`.

---

## 7. Hỗ Trợ

Nếu gặp vấn đề, vui lòng liên hệ:

- **Email:** support@kltn-stock-api.com
- **GitHub Issues:** [github.com/username/kltn-stock/issues](https://github.com)

---

*Phiên bản hướng dẫn: 1.0 | Cập nhật: Tháng 12/2025*
