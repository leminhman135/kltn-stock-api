# 📋 KẾ HOẠCH DỰ ÁN VÀ PHÂN CÔNG CÔNG VIỆC

## Thông tin chung

- **Tên đề tài:** Xây dựng hệ thống dự đoán giá cổ phiếu Việt Nam sử dụng Machine Learning
- **Thời gian thực hiện:** 16 tuần (1 học kỳ)
- **Sinh viên thực hiện:** [Tên sinh viên]
- **Giảng viên hướng dẫn:** [Tên GVHD]

---

## 📅 Gantt Chart

```
Tuần    1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16
═══════════════════════════════════════════════════════════════════════════
Phase 1: Khảo sát & Phân tích (Tuần 1-4)
───────────────────────────────────────────────────────────────────────────
1.1 Nghiên cứu tài liệu          ████████
1.2 Khảo sát nguồn dữ liệu           ████████
1.3 Phân tích yêu cầu                    ████████
1.4 Thiết kế kiến trúc                       ████████

Phase 2: Xây dựng Module Thu thập (Tuần 4-6)
───────────────────────────────────────────────────────────────────────────
2.1 Kết nối VNDirect API                     ████████
2.2 Xây dựng ETL Pipeline                        ████████
2.3 Thiết kế Database Schema                         ████

Phase 3: Xây dựng Module Đặc trưng (Tuần 6-8)
───────────────────────────────────────────────────────────────────────────
3.1 Technical Indicators                             ████████
3.2 FinBERT Sentiment                                    ████████
3.3 Feature Engineering                                      ████

Phase 4: Xây dựng Mô hình ML (Tuần 8-12)
───────────────────────────────────────────────────────────────────────────
4.1 ARIMA Model                                              ████████
4.2 Prophet Model                                                ████████
4.3 LSTM/GRU Models                                                  ████████████
4.4 Ensemble Model                                                           ████████

Phase 5: Xây dựng API & UI (Tuần 11-14)
───────────────────────────────────────────────────────────────────────────
5.1 FastAPI Backend                                                      ████████████
5.2 Web Dashboard UI                                                             ████████
5.3 API Documentation                                                                ████

Phase 6: Testing & Deployment (Tuần 14-16)
───────────────────────────────────────────────────────────────────────────
6.1 Unit Testing                                                                     ████████
6.2 Integration Testing                                                                  ████████
6.3 Deploy to Render                                                                         ████
6.4 Viết báo cáo                                                                             ████████

Milestone
───────────────────────────────────────────────────────────────────────────
◆ M1: Hoàn thành khảo sát       ◆
◆ M2: Hoàn thành thu thập             ◆
◆ M3: Hoàn thành ML models                                                   ◆
◆ M4: Hoàn thành API + UI                                                            ◆
◆ M5: Bảo vệ khóa luận                                                                       ◆
```

---

## 📊 Chi tiết các Phase

### Phase 1: Khảo sát & Phân tích (Tuần 1-4)

| Task ID | Task Name | Duration | Start | End | Output |
|---------|-----------|----------|-------|-----|--------|
| 1.1 | Nghiên cứu tài liệu về ML trong finance | 2 tuần | Tuần 1 | Tuần 2 | Literature review |
| 1.2 | Khảo sát nguồn dữ liệu VN stock | 2 tuần | Tuần 2 | Tuần 3 | Data sources list |
| 1.3 | Phân tích yêu cầu hệ thống | 2 tuần | Tuần 3 | Tuần 4 | SRS document |
| 1.4 | Thiết kế kiến trúc hệ thống | 2 tuần | Tuần 4 | Tuần 5 | Architecture doc |

**Deliverables:**
- ✅ Báo cáo khảo sát nguồn dữ liệu
- ✅ Tài liệu phân tích yêu cầu
- ✅ Sơ đồ kiến trúc hệ thống

---

### Phase 2: Xây dựng Module Thu thập (Tuần 4-6)

| Task ID | Task Name | Duration | Dependencies | Output |
|---------|-----------|----------|--------------|--------|
| 2.1 | Kết nối VNDirect API | 2 tuần | 1.2 | data_collection.py |
| 2.2 | Xây dựng ETL Pipeline | 2 tuần | 2.1 | etl_pipeline.py |
| 2.3 | Thiết kế Database Schema | 1 tuần | 2.1 | database/models.py |

**Deliverables:**
- ✅ Module kết nối VNDirect API
- ✅ ETL pipeline hoàn chỉnh
- ✅ Database với VN30 stocks

---

### Phase 3: Xây dựng Module Đặc trưng (Tuần 6-8)

| Task ID | Task Name | Duration | Dependencies | Output |
|---------|-----------|----------|--------------|--------|
| 3.1 | Technical Indicators | 2 tuần | 2.2 | technical_indicators.py |
| 3.2 | FinBERT Sentiment | 2 tuần | 2.2 | sentiment_analysis.py |
| 3.3 | Feature Engineering | 1 tuần | 3.1, 3.2 | Combined features |

**Deliverables:**
- ✅ Module tính chỉ báo kỹ thuật (MACD, RSI, Bollinger, SMA)
- ✅ Module phân tích sentiment FinBERT
- ✅ Feature dataset

---

### Phase 4: Xây dựng Mô hình ML (Tuần 8-12)

| Task ID | Task Name | Duration | Dependencies | Output |
|---------|-----------|----------|--------------|--------|
| 4.1 | ARIMA Model | 2 tuần | 3.3 | arima_model.py |
| 4.2 | Prophet Model | 2 tuần | 3.3 | prophet_model.py |
| 4.3 | LSTM/GRU Models | 3 tuần | 3.3 | lstm_gru_models.py |
| 4.4 | Ensemble Model | 2 tuần | 4.1, 4.2, 4.3 | ensemble.py |

**Deliverables:**
- ✅ ARIMA model với auto parameter selection
- ✅ Prophet model với seasonality
- ✅ LSTM và GRU models
- ✅ Ensemble model (Weighted Average + Stacking)

---

### Phase 5: Xây dựng API & UI (Tuần 11-14)

| Task ID | Task Name | Duration | Dependencies | Output |
|---------|-----------|----------|--------------|--------|
| 5.1 | FastAPI Backend | 3 tuần | 4.4 | api_v2.py, ml_endpoints.py |
| 5.2 | Web Dashboard | 2 tuần | 5.1 | static/index.html |
| 5.3 | API Documentation | 1 tuần | 5.1 | Swagger docs |

**Deliverables:**
- ✅ REST API với 100+ endpoints
- ✅ Web Dashboard responsive
- ✅ API documentation (Swagger/OpenAPI)

---

### Phase 6: Testing & Deployment (Tuần 14-16)

| Task ID | Task Name | Duration | Dependencies | Output |
|---------|-----------|----------|--------------|--------|
| 6.1 | Unit Testing | 2 tuần | 5.1 | tests/ |
| 6.2 | Integration Testing | 2 tuần | 6.1 | Test results |
| 6.3 | Deploy to Render | 1 tuần | 6.2 | Live API |
| 6.4 | Viết báo cáo | 2 tuần | All | Final report |

**Deliverables:**
- ✅ Test coverage > 80%
- ✅ Live API: https://kltn-stock-api.onrender.com
- ✅ Báo cáo khóa luận

---

## 📈 Tiến độ thực hiện

### Weekly Status Report

| Tuần | Tasks Completed | Issues | Next Week |
|------|-----------------|--------|-----------|
| 1 | Literature review (50%) | None | Complete review |
| 2 | Literature review (100%), Data sources survey | None | Analyze VNDirect API |
| 3 | VNDirect API analysis, SRS draft | None | Complete SRS |
| 4 | SRS complete, Architecture design | None | Start development |
| 5 | data_collection.py, Basic ETL | API rate limits | Implement retry logic |
| 6 | ETL pipeline complete, Database schema | None | Technical indicators |
| 7 | technical_indicators.py | ta-lib installation | Use ta package |
| 8 | FinBERT integration, Feature engineering | GPU not available | Use CPU inference |
| 9 | ARIMA model | Auto-parameter selection slow | Cache best params |
| 10 | Prophet model | Holiday data for VN | Create custom holidays |
| 11 | LSTM/GRU models | Training time | Early stopping |
| 12 | Ensemble model, Initial API | None | Complete API |
| 13 | API endpoints, Dashboard UI | CORS issues | Configure CORS |
| 14 | Dashboard complete, Testing | None | Deploy to Render |
| 15 | Deployment, Bug fixes | Cold start on Render | Keep-alive setup |
| 16 | Final report, Presentation | None | Defense preparation |

### Completion Status

```
Overall Progress: ████████████████████████████████████████████████ 100%

Phase 1: ████████████████████ 100% ✅
Phase 2: ████████████████████ 100% ✅
Phase 3: ████████████████████ 100% ✅
Phase 4: ████████████████████ 100% ✅
Phase 5: ████████████████████ 100% ✅
Phase 6: ████████████████████ 100% ✅
```

---

## 📝 Work Breakdown Structure (WBS)

```
1. HỆ THỐNG DỰ ĐOÁN GIÁ CỔ PHIẾU
├── 1.1 KHẢO SÁT VÀ PHÂN TÍCH
│   ├── 1.1.1 Nghiên cứu tài liệu
│   │   ├── Machine Learning for Finance
│   │   ├── Time Series Analysis
│   │   └── Sentiment Analysis
│   ├── 1.1.2 Khảo sát dữ liệu
│   │   ├── VNDirect API
│   │   ├── CafeF News
│   │   └── Financial statements
│   └── 1.1.3 Phân tích yêu cầu
│       ├── Functional requirements
│       └── Non-functional requirements
│
├── 1.2 THU THẬP DỮ LIỆU
│   ├── 1.2.1 VNDirect API Integration
│   │   ├── Price data (OHLCV)
│   │   ├── Trading data
│   │   └── Financial data
│   ├── 1.2.2 News Scraping
│   │   ├── CafeF headlines
│   │   └── VnExpress finance
│   └── 1.2.3 ETL Pipeline
│       ├── Extract
│       ├── Transform
│       └── Load
│
├── 1.3 XÂY DỰNG ĐẶC TRƯNG
│   ├── 1.3.1 Technical Indicators
│   │   ├── SMA, EMA
│   │   ├── RSI
│   │   ├── MACD
│   │   └── Bollinger Bands
│   └── 1.3.2 Sentiment Analysis
│       ├── FinBERT integration
│       └── Score aggregation
│
├── 1.4 MÔ HÌNH MACHINE LEARNING
│   ├── 1.4.1 Statistical Models
│   │   ├── ARIMA
│   │   └── Prophet
│   ├── 1.4.2 Deep Learning
│   │   ├── LSTM
│   │   └── GRU
│   └── 1.4.3 Ensemble
│       ├── Weighted Average
│       └── Stacking
│
├── 1.5 BACKTESTING
│   ├── 1.5.1 Signal Generation
│   ├── 1.5.2 Trade Simulation
│   └── 1.5.3 Performance Metrics
│       ├── Sharpe Ratio
│       ├── Max Drawdown
│       └── Win Rate
│
├── 1.6 API DEVELOPMENT
│   ├── 1.6.1 Core Endpoints
│   │   ├── Stocks
│   │   ├── Prices
│   │   └── Predictions
│   ├── 1.6.2 ML Endpoints
│   │   ├── Model predictions
│   │   ├── Model comparison
│   │   └── Backtesting
│   └── 1.6.3 Admin Endpoints
│       ├── Database management
│       └── Data synchronization
│
├── 1.7 WEB UI
│   ├── 1.7.1 Dashboard
│   │   ├── Stock grid
│   │   ├── Price chart
│   │   └── Statistics
│   ├── 1.7.2 Market Board
│   ├── 1.7.3 Prediction Page
│   └── 1.7.4 News & Sentiment
│
└── 1.8 DEPLOYMENT
    ├── 1.8.1 Render.com setup
    ├── 1.8.2 Database migration
    └── 1.8.3 Monitoring
```

---

## 🎯 Milestones

| ID | Milestone | Target Date | Status | Deliverables |
|----|-----------|-------------|--------|--------------|
| M1 | Hoàn thành khảo sát | Tuần 4 | ✅ Done | SRS, Architecture |
| M2 | Hoàn thành thu thập dữ liệu | Tuần 6 | ✅ Done | ETL, Database |
| M3 | Hoàn thành ML models | Tuần 12 | ✅ Done | All models |
| M4 | Hoàn thành API + UI | Tuần 14 | ✅ Done | Working application |
| M5 | Bảo vệ khóa luận | Tuần 16 | ⏳ Pending | Final presentation |

---

## 🔧 Tools & Resources

### Development Tools
| Tool | Purpose |
|------|---------|
| VS Code | IDE |
| Git/GitHub | Version control |
| Postman | API testing |
| DBeaver | Database management |

### Python Libraries
| Library | Version | Purpose |
|---------|---------|---------|
| FastAPI | 0.100+ | Web framework |
| SQLAlchemy | 2.0+ | ORM |
| TensorFlow | 2.x | Deep learning |
| Prophet | 1.1+ | Time series |
| Transformers | 4.x | FinBERT |

### Cloud Services
| Service | Purpose |
|---------|---------|
| Render.com | Web hosting |
| PostgreSQL (Render) | Database |
| GitHub Actions | CI/CD |
| UptimeRobot | Monitoring |

---

## 📊 Risk Management

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| API rate limits | High | Medium | Implement caching, rate limiting |
| Model overfitting | Medium | High | Cross-validation, regularization |
| Deployment issues | Medium | Medium | Docker containerization |
| Data quality | Low | High | Data validation in ETL |
| Time constraints | Medium | High | Priority-based development |

---

## 📞 Communication Plan

### Weekly Meetings
- **When:** Thứ 3, 14:00 - 15:00
- **Where:** Online (Google Meet)
- **Participants:** Sinh viên + GVHD

### Reporting
- Weekly status email
- GitHub issues for technical discussions
- Final presentation slides

---

## ✅ Quality Checklist

### Code Quality
- [x] Follow PEP 8 style guide
- [x] Type hints for functions
- [x] Docstrings for modules
- [x] Error handling

### Testing
- [x] Unit tests (coverage > 70%)
- [x] Integration tests
- [x] API endpoint tests
- [x] Model performance tests

### Documentation
- [x] README.md
- [x] ARCHITECTURE.md
- [x] API_DOCUMENTATION.md
- [x] MODEL_EVALUATION.md
- [x] PROJECT_PLAN.md
- [ ] Final thesis report
- [ ] Presentation slides

---

*Last updated: December 2025*
