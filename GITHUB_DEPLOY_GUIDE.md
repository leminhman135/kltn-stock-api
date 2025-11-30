# 🚀 HƯỚNG DẪN TẠO GITHUB REPO & DEPLOY RENDER

## Bước 1: Cài Git (2 phút)

### Tải Git cho Windows:
1. Truy cập: https://git-scm.com/download/win
2. Download "64-bit Git for Windows Setup"
3. Chạy file .exe và click Next, Next, Install
4. Sau khi cài xong, **KHỞI ĐỘNG LẠI VS Code**

---

## Bước 2: Tạo GitHub Repository (3 phút)

### A. Đăng ký/Đăng nhập GitHub
1. Truy cập: https://github.com
2. Click **"Sign up"** nếu chưa có tài khoản
3. Hoặc **"Sign in"** nếu đã có

### B. Tạo Repository Mới
1. Click nút **"+"** (góc phải trên) → **"New repository"**
2. Điền thông tin:
   ```
   Repository name: kltn-stock-api
   Description: Stock Prediction API using FastAPI
   Visibility: Public (hoặc Private nếu muốn)
   ☐ Add a README file (BỎ QUA - không tick)
   ☐ Add .gitignore (BỎ QUA)
   ☐ Choose a license (BỎ QUA)
   ```
3. Click **"Create repository"**

### C. Lấy URL Repository
Sau khi tạo, bạn sẽ thấy trang hướng dẫn. **Copy URL** này:
```
https://github.com/YOUR_USERNAME/kltn-stock-api.git
```

---

## Bước 3: Push Code Lên GitHub (2 phút)

### Sau khi cài Git và khởi động lại VS Code:

Chạy các lệnh này trong Terminal:

```bash
# Bước 1: Khởi tạo Git
cd d:\KLTN
git init

# Bước 2: Cấu hình Git (lần đầu tiên)
git config --global user.email "your_email@example.com"
git config --global user.name "Your Name"

# Bước 3: Add tất cả files
git add .

# Bước 4: Commit
git commit -m "Initial commit: KLTN Stock Prediction API"

# Bước 5: Kết nối với GitHub repo (thay YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/kltn-stock-api.git

# Bước 6: Push lên GitHub
git branch -M main
git push -u origin main
```

**Lưu ý**: Lần đầu push, GitHub sẽ yêu cầu đăng nhập:
- Username: GitHub username của bạn
- Password: **Personal Access Token** (không phải password)
  * Tạo token tại: https://github.com/settings/tokens
  * Click "Generate new token (classic)"
  * Chọn quyền: `repo` (full control)
  * Copy token và dùng làm password

---

## Bước 4: Deploy Từ GitHub Lên Render (3 phút)

### A. Kết Nối GitHub Với Render
1. Vào https://render.com → Dashboard
2. Click **"New +"** → **"Web Service"**
3. Click **"Connect GitHub"** (nếu chưa kết nối)
4. Cho phép Render truy cập GitHub

### B. Chọn Repository
1. Tìm repo: `kltn-stock-api`
2. Click **"Connect"**

### C. Cấu Hình Deploy
Điền thông tin:

```
Name: kltn-stock-api
Region: Singapore
Branch: main
Runtime: Python 3

Build Command: pip install -r requirements.txt
Start Command: uvicorn src.api_v2:app --host 0.0.0.0 --port $PORT
```

### D. Environment Variables
Thêm 2 biến:

**Biến 1:**
```
Key: DATABASE_URL
Value: [Paste PostgreSQL URL từ database đã tạo]
```

**Biến 2:**
```
Key: PYTHON_VERSION
Value: 3.11.0
```

### E. Deploy!
1. Click **"Create Web Service"**
2. Đợi 5-7 phút để build
3. Xem logs để theo dõi

---

## ✅ CHECKLIST

**Trước khi deploy:**
- [ ] Đã cài Git và khởi động lại VS Code
- [ ] Đã tạo GitHub repository
- [ ] Đã copy repository URL
- [ ] Đã push code lên GitHub thành công
- [ ] Thấy code trên GitHub repo

**Deploy trên Render:**
- [ ] Đã tạo PostgreSQL database
- [ ] Đã connect GitHub với Render
- [ ] Đã chọn đúng repository
- [ ] Đã điền đủ Environment Variables
- [ ] Build thành công

**Kiểm tra:**
- [ ] https://kltn-stock-api.onrender.com/docs hoạt động
- [ ] /api/health trả về 200 OK

---

## ⚠️ XỬ LÝ LỖI

### Git không được nhận dạng
**Giải pháp**: 
1. Cài Git: https://git-scm.com/download/win
2. **Khởi động lại VS Code**
3. Thử lại lệnh git

### GitHub yêu cầu authentication
**Giải pháp**:
1. Tạo Personal Access Token: https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Chọn quyền `repo`
4. Copy token
5. Dùng token thay vì password khi push

### Render không thấy repository
**Giải pháp**:
1. Đảm bảo repo là Public
2. Hoặc authorize Render truy cập Private repos
3. Refresh trang Render

---

## 🎯 TÓM TẮT NHANH

1. **Cài Git** → https://git-scm.com/download/win
2. **Tạo GitHub repo** → https://github.com/new
3. **Push code**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <URL>
   git push -u origin main
   ```
4. **Deploy Render** → Connect GitHub → Chọn repo → Deploy

**XONG! 🎉**
