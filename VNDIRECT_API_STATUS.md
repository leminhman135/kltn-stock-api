# VNDirect API - Trạng Thái & Hướng Dẫn Sử Dụng

## ✅ API Hoạt Động Tốt

### 1. **dchart-api (Historical Price Data)**
- **Endpoint**: `https://dchart-api.vndirect.com.vn/dchart/history`
- **Mục đích**: Lấy dữ liệu giá lịch sử (OHLCV)
- **Trạng thái**: ✅ **HOẠT ĐỘNG TỐT**
- **Dữ liệu trả về**: 
  - Giá Open, High, Low, Close
  - Volume (khối lượng giao dịch)
  - Timestamps (Unix format)
- **Resolutions hỗ trợ**:
  - `D` - Daily (ngày)
  - `1` - 1 minute
  - `5` - 5 minutes
  - `15` - 15 minutes
  - `30` - 30 minutes
  - `60` - 1 hour

**Ví dụ sử dụng:**
```python
from src.data_collection import VNDirectAPI

api = VNDirectAPI()
df = api.get_stock_price('VNM.VN', '2024-01-01', '2024-12-31')
print(f"Đã lấy {len(df)} bản ghi")
print(df.head())
```

**Kết quả test thực tế:**
- ✅ VNM: 248 records (1 năm)
- ✅ VIC: 127 records (6 tháng)
- ✅ HPG: 127 records (6 tháng)
- ✅ Dữ liệu chính xác, đầy đủ

## ❌ API Không Hoạt Động / DNS Error

### 2. **finfo-api (Company Info & Fundamentals)**
- **Endpoint**: `https://finfo-api.vndirect.com.vn/v4/`
- **Mục đích**: Thông tin công ty, chỉ số tài chính
- **Trạng thái**: ❌ **DNS RESOLUTION ERROR**
- **Lỗi**: `Failed to resolve 'finfo-api.vndirect.com.vn'`

### 3. **fwtapi2 (Market Data)**
- **Endpoint**: `https://fwtapi2.vndirect.com.vn/`
- **Mục đích**: Tổng quan thị trường
- **Trạng thái**: ❌ **DNS RESOLUTION ERROR**

## 📊 Dữ liệu Có Sẵn vs Không Có Sẵn

### ✅ Có Sẵn (từ dchart API):
1. **Giá lịch sử** (Historical Prices)
   - Open, High, Low, Close
   - Volume
   - Multiple timeframes (1min đến Daily)

2. **Dữ liệu intraday** (Trong ngày)
   - 1-minute, 5-minute intervals
   - Real-time or near real-time

### ❌ Không Có Sẵn (finfo API không hoạt động):
1. **Thông tin công ty**
   - Tên công ty, ngành nghề
   - Market cap, số lượng cổ phiếu
   
2. **Chỉ số tài chính**
   - P/E, P/B, EPS
   - ROE, ROA, ROI
   
3. **Báo cáo tài chính**
   - Cân đối kế toán
   - Kết quả kinh doanh
   - Lưu chuyển tiền tệ
   
4. **Thông tin sở hữu**
   - Cổ đông lớn
   - Room nước ngoài
   
5. **Cổ tức & Sự kiện**
   - Lịch sử chi trả cổ tức
   - Sự kiện doanh nghiệp
   
6. **Top stocks**
   - Top tăng/giảm
   - Top khối lượng/giá trị

## 🔄 Giải Pháp Thay Thế

### Dùng Yahoo Finance cho dữ liệu bổ sung:
```python
from src.data_collection import YahooFinanceAPI

yahoo = YahooFinanceAPI()
df = yahoo.get_stock_data('VNM.VN', '2024-01-01', '2024-12-31')
```

**Ưu điểm Yahoo Finance:**
- ✅ Có thông tin công ty cơ bản
- ✅ Có một số chỉ số tài chính
- ✅ API stable, không bị DNS error
- ❌ Nhưng dữ liệu VN ít hơn VNDirect

### Scraping từ Website (Backup plan):
```python
from src.data_collection import NewsScraperBS4

scraper = NewsScraperBS4()
news = scraper.scrape_cafef('VNM', pages=5)
```

## 📈 So Sánh Chất Lượng Dữ Liệu

### VNDirect dchart vs Yahoo Finance (Test với VNM):

| Tiêu chí | VNDirect | Yahoo Finance |
|----------|----------|---------------|
| **Số records (6 tháng)** | 127 | 127 |
| **Giá trị trùng khớp** | ~95-99% | ~95-99% |
| **Độ trễ** | Real-time | 15-20 phút |
| **Độ tin cậy** | Cao (nguồn VN) | Trung bình |
| **API stability** | Tốt (dchart) | Rất tốt |

**Kết luận:** 
- Dùng **VNDirect dchart** làm nguồn chính cho giá cổ phiếu VN
- Dùng **Yahoo Finance** làm backup và lấy thêm metadata
- Tương quan giữa 2 nguồn: > 0.95 (rất tốt)

## 💡 Khuyến Nghị

### Chiến lược thu thập dữ liệu hiện tại:

1. **Giá cổ phiếu**: VNDirect dchart API ✅
   - Đầy đủ, chính xác
   - Nhiều timeframes
   
2. **Thông tin công ty**: Yahoo Finance API ✅
   - Thông tin cơ bản đầy đủ
   - Chỉ số P/E, Market Cap
   
3. **Tin tức**: Web Scraping ⚠️
   - CafeF, VnExpress
   - Cần cẩn thận với rate limiting

4. **Phân tích kỹ thuật**: Tự tính toán ✅
   - RSI, MACD, Bollinger Bands
   - Moving Averages
   - Đã implement trong `technical_indicators.py`

### Code example hoàn chỉnh:

```python
from src.data_collection import VNDirectAPI, YahooFinanceAPI
from src.features.technical_indicators import TechnicalIndicators

# 1. Lấy giá từ VNDirect
vnd_api = VNDirectAPI()
df_vnd = vnd_api.get_stock_price('VNM.VN', '2024-01-01', '2024-12-31')

# 2. Lấy thông tin từ Yahoo (fallback)
yahoo_api = YahooFinanceAPI()
df_yahoo = yahoo_api.get_stock_data('VNM.VN', '2024-01-01', '2024-12-31')

# 3. Tính chỉ số kỹ thuật
ti = TechnicalIndicators()
df_with_indicators = ti.calculate_all_indicators(df_vnd)

# 4. So sánh 2 nguồn
print(f"VNDirect: {len(df_vnd)} records")
print(f"Yahoo: {len(df_yahoo)} records")
print(f"Indicators: {df_with_indicators.columns.tolist()}")
```

## 🔍 Kiểm Tra Dữ Liệu

Sử dụng trang **🔍 Kiểm tra dữ liệu** trong web app:

1. Chọn mã cổ phiếu
2. Click "🔄 So Sánh Dữ Liệu"
3. Xem:
   - Số lượng records từ mỗi nguồn
   - Biểu đồ chồng lấp
   - Hệ số tương quan
   - Chênh lệch %

**Chỉ số chất lượng:**
- Correlation > 0.95: Xuất sắc ✅
- Correlation > 0.85: Tốt ✅
- Correlation > 0.70: Chấp nhận được ⚠️
- Correlation < 0.70: Cần kiểm tra ❌

## 📞 Hỗ Trợ

Nếu cần thêm dữ liệu hoặc API không hoạt động:
1. Kiểm tra network/firewall
2. Test bằng script `test_vndirect_apis.py`
3. Xem log chi tiết trong terminal
4. Dùng Yahoo Finance làm fallback

---
**Cập nhật**: 30/11/2025
**Trạng thái**: VNDirect dchart API hoạt động tốt, finfo API không khả dụng
