# Railway Multi-Service Setup Guide

## 🏗️ Architecture

```
Railway Project "kltn-stock"
│
├── 🗄️ PostgreSQL (Database Plugin)
│   └── Shared by all services via DATABASE_URL
│
├── 🔧 api (Main Service)
│   ├── FastAPI with static frontend
│   ├── Port: 8080
│   ├── Memory: ~512MB
│   └── Start: uvicorn src.api_v2:app
│
├── ⏰ cron (Cron Worker)
│   ├── Scheduler for daily updates
│   ├── No public domain needed
│   ├── Memory: ~256MB
│   └── Start: python -m src.cron_worker
│
└── 🤖 ml-worker (Optional - ML Service)
    ├── Heavy ML predictions
    ├── Port: 8081 (internal)
    ├── Memory: ~2GB
    └── Start: uvicorn src.ml_worker:app
```

## 📋 Setup Steps

### Step 1: Create Services in Railway

1. Go to Railway Dashboard
2. In your project, click **"+ New"** → **"Empty Service"**
3. Create these services:
   - `api` - Main API
   - `cron` - Cron worker
   - `ml-worker` (optional) - ML service

### Step 2: Configure Each Service

#### API Service (`api`)
```
Settings → Deploy:
- Root Directory: /
- Start Command: uvicorn src.api_v2:app --host 0.0.0.0 --port 8080

Settings → Networking:
- Add public domain: kltn.up.railway.app
- Port: 8080

Variables:
- DATABASE_URL: (reference from PostgreSQL)
- PYTHONUNBUFFERED: 1
```

#### Cron Service (`cron`)
```
Settings → Deploy:
- Root Directory: /
- Start Command: python -m src.cron_worker

Variables:
- DATABASE_URL: (reference from PostgreSQL)
- PYTHONUNBUFFERED: 1
- RUN_ON_STARTUP: false
```

#### ML Worker Service (`ml-worker`) - Optional
```
Settings → Deploy:
- Root Directory: /
- Start Command: uvicorn src.ml_worker:app --host 0.0.0.0 --port 8081

Settings → Networking:
- Private networking only (no public domain)
- Port: 8081

Variables:
- DATABASE_URL: (reference from PostgreSQL)
- PYTHONUNBUFFERED: 1
- ML_WORKER_URL: (internal URL for API to call)
```

### Step 3: Share Database

All services should reference the same PostgreSQL:
1. In each service → Variables
2. Click **"Add Reference"**
3. Select PostgreSQL → DATABASE_URL

### Step 4: Internal Communication

For API to call ML Worker internally:
```python
# In api_v2.py
ML_WORKER_URL = os.getenv("ML_WORKER_URL", "http://ml-worker.railway.internal:8081")

async def call_ml_worker(symbol: str, days: int):
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{ML_WORKER_URL}/predict/{symbol}",
            params={"days": days}
        )
        return response.json()
```

## 💡 Benefits of Multi-Service

| Aspect | Single Service | Multi-Service |
|--------|---------------|---------------|
| Startup | ~60s (load all) | ~10s (lite API) |
| Memory | ~2GB | 512MB + 256MB + 2GB |
| Scaling | All or nothing | Scale each independently |
| Failure | All down | Only affected service |
| Cost | 1 instance always | Cron can sleep |

## 🚀 Quick Deploy

Option 1: Keep current single service (simpler)
Option 2: Split services (faster, more scalable)

For KLTN demo, single service is fine. Split when needed.
