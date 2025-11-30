# 🚀 Hướng Dẫn Deploy Lên Render.com

## ✨ Ưu Điểm Render.com
- ✅ Hoàn toàn **MIỄN PHÍ** (Web Service + PostgreSQL)
- ✅ **Không cần GitHub** - Deploy trực tiếp từ folder
- ✅ PostgreSQL miễn phí 90 ngày (tự động gia hạn)
- ✅ HTTPS tự động
- ✅ Ổn định hơn Railway

---

## 📋 Các Bước Deploy

### Bước 1: Đăng Ký Tài Khoản Render
1. Truy cập: https://render.com
2. Click **"Get Started"**
3. Đăng ký bằng email (không cần GitHub)
4. Xác nhận email

### Bước 2: Tạo PostgreSQL Database
1. Vào Dashboard → Click **"New +"** → **"PostgreSQL"**
2. Điền thông tin:
   - **Name**: `kltn-postgres`
   - **Database**: `kltn_db`
   - **User**: `kltn_user`
   - **Region**: Singapore (gần VN nhất)
   - **Plan**: **Free**
3. Click **"Create Database"**
4. Đợi 2-3 phút để database khởi tạo
5. **Sao chép Internal Database URL** (dạng: `postgresql://...`)

### Bước 3: Upload Code Lên Render
**Cách 1: Dùng Render CLI (Khuyên dùng)**
```bash
# Cài đặt Render CLI
npm install -g @render/cli

# Login
render login

# Deploy
cd d:\KLTN
render deploy
```

**Cách 2: Upload ZIP thủ công**
1. Nén toàn bộ folder `d:\KLTN` thành `kltn.zip`
2. Vào Render Dashboard → **"New +"** → **"Web Service"**
3. Chọn **"Deploy an existing image or upload files"**
4. Upload file `kltn.zip`

### Bước 4: Cấu Hình Web Service
1. Sau khi upload, điền thông tin:
   - **Name**: `kltn-stock-api`
   - **Region**: Singapore
   - **Branch**: main (nếu dùng Git) hoặc bỏ qua
   - **Runtime**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `./start.sh`

2. **Environment Variables** (rất quan trọng):
   Click **"Add Environment Variable"**, thêm:
   
   | Key | Value |
   |-----|-------|
   | `DATABASE_URL` | (Paste Internal Database URL từ bước 2) |
   | `PORT` | `10000` |
   | `PYTHON_VERSION` | `3.11.0` |

3. Click **"Create Web Service"**

### Bước 5: Đợi Deploy
1. Render sẽ tự động build (3-5 phút)
2. Xem log trong tab **"Logs"**
3. Khi thấy: `✅ Build completed successfully!` → Thành công

### Bước 6: Kiểm Tra API
1. URL của bạn: `https://kltn-stock-api.onrender.com`
2. Test các endpoint:
   - Health: https://kltn-stock-api.onrender.com/api/health
   - Docs: https://kltn-stock-api.onrender.com/docs
   - Root: https://kltn-stock-api.onrender.com/

---

## 🔧 Xử Lý Lỗi Thường Gặp

### Lỗi 1: "Build failed"
**Nguyên nhân**: Thiếu dependencies trong `requirements.txt`
**Giải pháp**:
```bash
# Kiểm tra lại requirements.txt có đầy đủ không
pip freeze > requirements.txt
```

### Lỗi 2: "Application failed to respond"
**Nguyên nhân**: Port không đúng hoặc uvicorn không chạy
**Giải pháp**: Kiểm tra `start.sh`:
- Đảm bảo có `--port ${PORT:-10000}`
- Kiểm tra `src.api_v2:app` đúng path

### Lỗi 3: "Database connection failed"
**Nguyên nhân**: `DATABASE_URL` không đúng
**Giải pháp**:
1. Vào PostgreSQL dashboard
2. Copy lại **Internal Database URL**
3. Update lại Environment Variable `DATABASE_URL`
4. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Lỗi 4: "Permission denied: ./build.sh"
**Nguyên nhân**: File script không có quyền execute
**Giải pháp**: Thêm vào `build.sh` đầu file:
```bash
chmod +x build.sh
chmod +x start.sh
```

---

## 🎯 Các Endpoint Quan Trọng

Sau khi deploy thành công, test các endpoint:

### 1. Health Check
```bash
curl https://kltn-stock-api.onrender.com/api/health
```
**Kết quả mong đợi**:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-30T...",
  "database": "connected"
}
```

### 2. API Documentation
Truy cập: https://kltn-stock-api.onrender.com/docs
Sẽ thấy Swagger UI với tất cả 25+ endpoints

### 3. Lấy Danh Sách Cổ Phiếu
```bash
curl https://kltn-stock-api.onrender.com/api/stocks
```

---

## 📊 Giới Hạn Free Tier

| Tính năng | Giới hạn |
|-----------|----------|
| Web Service | 750 giờ/tháng |
| RAM | 512 MB |
| Database | 1 GB storage |
| Bandwidth | 100 GB/tháng |
| Auto-sleep | Sau 15 phút không dùng |

**Lưu ý**: API sẽ sleep sau 15 phút không có request. Request đầu tiên sau sleep sẽ mất 30-60 giây để wake up.

**Giải pháp**: Dùng cron job ping mỗi 10 phút:
```bash
# Tạo cron job trên UptimeRobot.com (free)
# Ping: https://kltn-stock-api.onrender.com/api/health
# Interval: 5 phút
```

---

## 🔐 Bảo Mật

### Thêm API Key (Tùy chọn)
Nếu muốn bảo vệ API, thêm vào Environment Variables:
```
API_KEY=your-secret-key-here
```

Sau đó sửa code để check API key trong headers.

---

## 📈 Nâng Cấp Lên Paid Plan

Nếu cần:
- **Starter Plan**: $7/tháng
  - Không sleep
  - 1GB RAM
  - Custom domain

- **Pro Plan**: $25/tháng
  - 4GB RAM
  - Priority support
  - More resources

---

## 🆘 Liên Hệ Support

- Render Docs: https://render.com/docs
- Community: https://community.render.com
- Support: support@render.com

---

## ✅ Checklist Deploy

- [ ] Đã tạo tài khoản Render.com
- [ ] Đã tạo PostgreSQL database
- [ ] Đã sao chép Database URL
- [ ] Đã upload code (CLI hoặc ZIP)
- [ ] Đã cấu hình Environment Variables
- [ ] Build thành công (xem logs)
- [ ] `/api/health` trả về 200 OK
- [ ] `/docs` hiển thị Swagger UI
- [ ] Database kết nối thành công

**Hoàn thành hết checklist = Deploy thành công! 🎉**
