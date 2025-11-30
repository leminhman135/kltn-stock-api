# 📋 HƯỚNG DẪN DEPLOY NHANH - RENDER.COM

## ✅ File ZIP đã sẵn sàng: `kltn-render.zip` (0.29 MB)

---

## 🚀 CÁC BƯỚC THỰC HIỆN (10 phút)

### **BƯỚC 1: Đăng ký Render.com** (2 phút)
1. Mở trình duyệt: https://render.com
2. Click **"Get Started"** (góc phải trên)
3. Chọn **"Sign Up"**
4. Nhập email + password (KHÔNG cần GitHub)
5. Xác nhận email trong hộp thư

---

### **BƯỚC 2: Tạo Database** (3 phút)
1. Sau khi đăng nhập, click **"New +"** (góc phải)
2. Chọn **"PostgreSQL"**
3. Điền thông tin:
   ```
   Name: kltn-postgres
   Database: kltn_db
   User: kltn_user
   Region: Singapore
   Instance Type: Free
   ```
4. Click **"Create Database"**
5. Đợi 1-2 phút cho database khởi tạo
6. **QUAN TRỌNG**: Vào tab **"Info"**, copy **"Internal Database URL"**
   - Dạng: `postgresql://kltn_user:xxxxx@xxxxx.oregon-postgres.render.com/kltn_db`
   - Lưu vào Notepad

---

### **BƯỚC 3: Upload Code** (2 phút)
1. Quay lại Dashboard, click **"New +"** → **"Web Service"**
2. Chọn **"Deploy from Git or Docker Image"** → Click **"Next"**
3. Ở phần **"Public Git repository"**, bỏ qua
4. Kéo xuống phần **"Deploy from a file"**
5. Click **"Upload File"**
6. Chọn file `d:\KLTN\kltn-render.zip`
7. Đợi upload xong

---

### **BƯỚC 4: Cấu hình Service** (3 phút)
Sau khi upload, điền các thông tin:

#### **Basic Info:**
```
Name: kltn-stock-api
Region: Singapore
Branch: (bỏ qua)
Runtime: Python 3
```

#### **Build & Deploy:**
```
Build Command: ./build.sh
Start Command: ./start.sh
```

#### **Environment Variables** (Quan trọng nhất!)
Click **"Add Environment Variable"**, thêm 3 biến:

**Biến 1:**
```
Key: DATABASE_URL
Value: [Paste Internal Database URL từ Bước 2]
```

**Biến 2:**
```
Key: PORT
Value: 10000
```

**Biến 3:**
```
Key: PYTHON_VERSION
Value: 3.11.0
```

#### **Instance Type:**
```
Free
```

---

### **BƯỚC 5: Deploy!**
1. Click **"Create Web Service"** (nút xanh ở dưới)
2. Render sẽ bắt đầu build (3-5 phút)
3. Xem log trong tab **"Logs"**:
   - Nếu thấy `✅ Build completed successfully!` → OK
   - Nếu thấy `==> Your service is live 🎉` → HOÀN THÀNH!

---

## 🎯 KIỂM TRA API

Sau khi deploy xong, URL của bạn sẽ là:
```
https://kltn-stock-api.onrender.com
```

### Test các endpoint:

**1. Health Check:**
```
https://kltn-stock-api.onrender.com/api/health
```
→ Phải trả về: `{"status": "healthy"}`

**2. Swagger Docs:**
```
https://kltn-stock-api.onrender.com/docs
```
→ Phải hiển thị trang Swagger UI

**3. API Root:**
```
https://kltn-stock-api.onrender.com/
```
→ Phải trả về thông tin API

---

## ⚠️ XỬ LÝ LỖI

### Lỗi: "Build failed"
**Nguyên nhân**: Thiếu dependencies
**Giải pháp**: 
1. Vào tab **"Logs"**
2. Tìm dòng lỗi (thường là `ModuleNotFoundError`)
3. Báo lại cho tôi, tôi sẽ fix

### Lỗi: "Permission denied: ./build.sh"
**Nguyên nhân**: File không có quyền execute
**Giải pháp**:
1. Vào **"Settings"** → **"Build & Deploy"**
2. Đổi Build Command thành:
   ```
   chmod +x build.sh && ./build.sh
   ```
3. Click **"Save Changes"**
4. Click **"Manual Deploy"** → **"Deploy latest commit"**

### Lỗi: "Application failed to respond"
**Nguyên nhân**: Environment Variables sai
**Giải pháp**:
1. Vào **"Environment"** tab
2. Kiểm tra lại 3 biến: `DATABASE_URL`, `PORT`, `PYTHON_VERSION`
3. Đảm bảo `DATABASE_URL` đúng (copy từ PostgreSQL dashboard)

---

## 📞 CẦN HỖ TRỢ?

Nếu gặp lỗi:
1. Chụp màn hình tab **"Logs"**
2. Gửi cho tôi, tôi sẽ fix ngay

---

## ✅ CHECKLIST

- [ ] Đã đăng ký Render.com
- [ ] Đã tạo PostgreSQL database
- [ ] Đã copy Internal Database URL
- [ ] Đã upload file kltn-render.zip
- [ ] Đã điền đủ 3 Environment Variables
- [ ] Build thành công (xem Logs)
- [ ] Truy cập được /docs
- [ ] /api/health trả về 200 OK

**HẾT! 🎉**
