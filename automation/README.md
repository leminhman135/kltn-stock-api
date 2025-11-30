# Automation Module - Hướng Dẫn Sử Dụng

## 📋 Tổng Quan

Module automation cung cấp hệ thống tự động hóa hoàn chỉnh cho:
- ✅ Thu thập dữ liệu từ API (VNDirect, Yahoo Finance)
- ✅ Web scraping tin tức (CafeF, VietStock, NDH)
- ✅ ETL Pipeline (Extract, Transform, Load)
- ✅ Tính toán technical indicators
- ✅ Phân tích sentiment
- ✅ Training models định kỳ
- ✅ Backup và cleanup tự động

---

## 🚀 Cách Sử Dụng

### 1. Cài Đặt Dependencies

```powershell
pip install schedule beautifulsoup4 scrapy requests
```

### 2. Chạy Scheduler (Background Process)

**Option A: Chạy scheduler liên tục**
```powershell
python automation/scheduler.py
```

Scheduler sẽ chạy các tasks theo lịch:
- 📊 **18:00** - Thu thập dữ liệu
- 🔄 **18:30** - Xử lý và làm sạch
- 📈 **19:00** - Tính technical indicators
- 💭 **20:00** - Phân tích sentiment
- 🤖 **Chủ Nhật 02:00** - Training models
- 💾 **Chủ Nhật 03:00** - Backup data
- 🗑️  **Chủ Nhật 04:00** - Cleanup old files

**Option B: Chạy manual (test)**
```powershell
# Chạy tất cả tasks ngay
python automation/scheduler.py --run-now

# Chỉ thu thập data
python automation/scheduler.py --collect-only

# Chỉ xử lý data
python automation/scheduler.py --process-only

# Chỉ train models
python automation/scheduler.py --train-only
```

### 3. Web Scraping Tin Tức

**Scrape tin tức cho 1 mã**
```python
from automation.web_scraper import NewsScraper

scraper = NewsScraper()
articles_df = scraper.scrape_all('VNM')
print(f"Found {len(articles_df)} articles")
```

**Scrape cho nhiều mã**
```python
from automation.web_scraper import scrape_news_for_stocks

symbols = ['VNM', 'VIC', 'HPG', 'VCB', 'FPT']
scrape_news_for_stocks(symbols, output_dir='data/news')
```

**Output**: CSV files trong `data/news/` với columns:
- symbol, source, title, summary, link, date_str, scraped_at

### 4. Chạy Như Windows Service (Background)

**PowerShell Script:**
```powershell
# Tạo file run_scheduler.ps1
@"
`$pythonPath = "D:\KLTN\venv\Scripts\python.exe"
`$scriptPath = "D:\KLTN\automation\scheduler.py"

while (`$true) {
    & `$pythonPath `$scriptPath
    Start-Sleep -Seconds 60
}
"@ | Out-File -FilePath "run_scheduler.ps1"

# Chạy trong background
Start-Process powershell -ArgumentList "-File run_scheduler.ps1" -WindowStyle Hidden
```

**Hoặc dùng Task Scheduler:**
1. Mở **Task Scheduler**
2. **Create Basic Task**
3. Trigger: **When computer starts**
4. Action: **Start a program**
   - Program: `D:\KLTN\venv\Scripts\python.exe`
   - Arguments: `D:\KLTN\automation\scheduler.py`
5. ✅ Done

---

## 📁 Cấu Trúc Thư Mục

```
data/
├── raw/                          # Dữ liệu thô từ API
│   ├── VNM_raw_20251130.csv
│   ├── VIC_raw_20251130.csv
│   └── ...
├── processed/                    # Dữ liệu đã xử lý
│   ├── VNM_processed.csv
│   ├── VNM_with_indicators.csv
│   └── ...
├── news/                         # Tin tức scraped
│   ├── VNM_news_20251130.csv
│   └── ...
├── backups/                      # Backup hàng tuần
│   ├── 20251201/
│   └── ...
└── models/                       # Trained models
    ├── VNM_arima.pkl
    └── ...

automation/
├── scheduler.py                  # Main scheduler
├── web_scraper.py               # Web scraping module
├── logs/                         # Logs
│   └── scheduler.log
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Thay Đổi Stocks

Edit `automation/scheduler.py`:
```python
STOCKS = ['VNM', 'VIC', 'HPG', 'VCB', 'FPT', 'VHM', 'MSN', 'CTG', 'TCB', 'BID']
```

### Thay Đổi Lịch Trình

```python
# Ví dụ: Thu thập data lúc 17:00 thay vì 18:00
schedule.every().day.at("17:00").do(collect_stock_data)

# Chạy training mỗi ngày thay vì mỗi tuần
schedule.every().day.at("23:00").do(train_models_weekly)
```

### Thêm Nguồn Tin Mới

Edit `automation/web_scraper.py`, thêm method:
```python
def scrape_new_source(self, symbol: str) -> List[Dict]:
    """Scrape from new source"""
    # Your scraping code here
    pass
```

---

## 📊 Logs & Monitoring

### Xem Logs
```powershell
# Xem real-time
Get-Content automation/logs/scheduler.log -Wait

# Xem 50 dòng cuối
Get-Content automation/logs/scheduler.log -Tail 50
```

### Log Format
```
2025-11-30 18:00:01 - automation.scheduler - INFO - 🚀 STARTING DATA COLLECTION
2025-11-30 18:00:05 - automation.scheduler - INFO - ✅ Saved 248 records to data/raw/VNM_raw_20251130.csv
```

---

## 🔧 Troubleshooting

### ❌ Error: Module not found
```powershell
pip install schedule beautifulsoup4 requests pandas
```

### ❌ Error: Permission denied
- Chạy PowerShell as Administrator
- Hoặc thay đổi quyền folder: `icacls "D:\KLTN\data" /grant Users:F`

### ❌ Web scraping bị block
- Thêm delay: `time.sleep(2)` giữa các requests
- Sử dụng proxy
- Rotate User-Agent

### ❌ Scheduler không chạy đúng giờ
- Kiểm tra timezone: `import datetime; print(datetime.datetime.now())`
- Schedule dùng local time, đảm bảo máy đúng giờ

---

## 🚀 Production Deployment

### Option 1: Windows Service (NSSM)
```powershell
# Download NSSM: https://nssm.cc/download
nssm install StockScheduler "D:\KLTN\venv\Scripts\python.exe" "D:\KLTN\automation\scheduler.py"
nssm start StockScheduler
```

### Option 2: Docker Container
```dockerfile
FROM python:3.11
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["python", "automation/scheduler.py"]
```

### Option 3: Cloud (AWS/Azure)
- Deploy trên EC2/Azure VM
- Chạy scheduler 24/7
- Setup monitoring với CloudWatch/Azure Monitor

---

## 📈 Performance Tips

1. **Parallel Processing**: Sử dụng `multiprocessing` cho multiple stocks
2. **Caching**: Cache API responses để tránh duplicate requests
3. **Database**: Lưu vào PostgreSQL thay vì CSV (nhanh hơn)
4. **Queue**: Dùng Celery + Redis cho task queue
5. **Monitoring**: Setup Prometheus + Grafana

---

## 🔜 Planned Features

- [ ] Support thêm nguồn tin: Bloomberg, Reuters
- [ ] Telegram/Email notifications khi có lỗi
- [ ] Dashboard monitoring real-time
- [ ] Machine learning cho auto-tuning scheduler
- [ ] API endpoints để trigger tasks manually

---

## 📞 Support

Nếu gặp vấn đề:
1. Check logs: `automation/logs/scheduler.log`
2. Test manual: `python automation/scheduler.py --run-now`
3. Verify data: Check `data/raw/` và `data/processed/`

---

## ✅ Quick Start

```powershell
# 1. Install dependencies
pip install schedule beautifulsoup4 requests pandas

# 2. Test thu thập data
python automation/scheduler.py --collect-only

# 3. Test xử lý data
python automation/scheduler.py --process-only

# 4. Chạy scheduler
python automation/scheduler.py

# 5. Xem logs
Get-Content automation/logs/scheduler.log -Wait
```

**Done! Automation system đang chạy 🚀**
