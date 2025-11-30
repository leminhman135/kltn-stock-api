# n8n Workflows for KLTN Stock Prediction System

This document provides complete n8n workflow configurations to replace `automation/scheduler.py` with visual workflows.

## 📋 Overview

**7 n8n Workflows:**
1. ✅ **Data Collection** (Daily 18:00) - Thu thập dữ liệu từ VNDirect API
2. ✅ **Data Processing** (Daily 18:30) - Làm sạch và xử lý dữ liệu
3. ✅ **Technical Indicators** (Daily 19:00) - Tính toán các chỉ báo kỹ thuật
4. ✅ **Sentiment Analysis** (Daily 20:00) - Phân tích cảm xúc từ tin tức
5. ✅ **Model Training** (Weekly Sunday 02:00) - Huấn luyện mô hình
6. ✅ **Data Backup** (Weekly Sunday 03:00) - Sao lưu dữ liệu
7. ✅ **Cleanup** (Weekly Sunday 04:00) - Xóa dữ liệu cũ

---

## 🔧 Prerequisites

### n8n Installation Options:

**Option 1: n8n Cloud (Recommended for KLTN)**
- Free tier: 5000 executions/month
- No setup needed
- URL: https://n8n.io/cloud

**Option 2: Self-hosted on Railway**
```bash
# Railway will auto-deploy n8n from Docker image
railway up n8nio/n8n
```

**Option 3: Local Docker**
```bash
docker run -it --rm \
  --name n8n \
  -p 5678:5678 \
  -v ~/.n8n:/home/node/.n8n \
  n8nio/n8n
```

### Required Credentials in n8n:

1. **PostgreSQL**
   - Type: Postgres
   - Host: Your Railway/database host
   - Database: kltn_stocks
   - User: postgres
   - Password: [from Railway]
   - SSL: Enabled

2. **HTTP Request Authentication** (Optional)
   - For VNDirect API if needed

---

## Workflow 1: Data Collection

**Trigger:** Cron - Daily at 18:00 (Vietnam time)

### Workflow Structure:

```
[Cron Trigger: 0 18 * * *]
    ↓
[Function: Get Stock List]
    ↓
[Loop: For Each Stock]
    ↓
[HTTP Request: VNDirect API]
    ↓
[Code: Transform Data]
    ↓
[PostgreSQL: Insert/Update stock_prices]
    ↓
[PostgreSQL: Update stock info]
    ↓
[End Loop]
    ↓
[Slack/Email: Notification]
```

### n8n JSON Workflow:

```json
{
  "name": "1. KLTN - Stock Data Collection",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "cronExpression",
              "expression": "0 18 * * *"
            }
          ],
          "timezone": "Asia/Ho_Chi_Minh"
        }
      },
      "name": "Daily 18:00",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1,
      "position": [250, 300]
    },
    {
      "parameters": {
        "functionCode": "// Stock list to collect\nconst stocks = ['VNM', 'VIC', 'HPG', 'VCB', 'FPT', 'VHM', 'MSN', 'CTG', 'TCB', 'BID'];\n\nconst today = new Date().toISOString().split('T')[0];\nconst oneYearAgo = new Date();\noneYearAgo.setFullYear(oneYearAgo.getFullYear() - 1);\nconst startDate = oneYearAgo.toISOString().split('T')[0];\n\nreturn stocks.map(symbol => ({\n  json: {\n    symbol: symbol,\n    start_date: startDate,\n    end_date: today,\n    resolution: '1D'\n  }\n}));"
      },
      "name": "Get Stock List",
      "type": "n8n-nodes-base.function",
      "typeVersion": 1,
      "position": [450, 300]
    },
    {
      "parameters": {
        "url": "=https://dchart-api.vndirect.com.vn/dchart/history?symbol={{$json.symbol}}&resolution={{$json.resolution}}&from={{Date.parse($json.start_date)/1000}}&to={{Date.parse($json.end_date)/1000}}",
        "options": {
          "timeout": 30000
        }
      },
      "name": "VNDirect API",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 2,
      "position": [650, 300]
    },
    {
      "parameters": {
        "functionCode": "const symbol = $input.first().json.symbol;\nconst data = $input.first().json;\n\nif (!data.t || !Array.isArray(data.t)) {\n  return [];\n}\n\nconst records = [];\nfor (let i = 0; i < data.t.length; i++) {\n  const date = new Date(data.t[i] * 1000).toISOString().split('T')[0];\n  \n  records.push({\n    json: {\n      symbol: symbol,\n      date: date,\n      open: data.o[i],\n      high: data.h[i],\n      low: data.l[i],\n      close: data.c[i],\n      volume: data.v[i],\n      source: 'vndirect'\n    }\n  });\n}\n\nreturn records;"
      },
      "name": "Transform Data",
      "type": "n8n-nodes-base.function",
      "typeVersion": 1,
      "position": [850, 300]
    },
    {
      "parameters": {
        "operation": "executeQuery",
        "query": "=INSERT INTO stock_prices (stock_id, date, open, high, low, close, volume, source)\nSELECT \n  (SELECT id FROM stocks WHERE symbol = '{{$json.symbol}}'),\n  '{{$json.date}}'::date,\n  {{$json.open}},\n  {{$json.high}},\n  {{$json.low}},\n  {{$json.close}},\n  {{$json.volume}},\n  '{{$json.source}}'\nON CONFLICT (stock_id, date) DO UPDATE SET\n  open = EXCLUDED.open,\n  high = EXCLUDED.high,\n  low = EXCLUDED.low,\n  close = EXCLUDED.close,\n  volume = EXCLUDED.volume,\n  source = EXCLUDED.source;"
      },
      "name": "Save to PostgreSQL",
      "type": "n8n-nodes-base.postgres",
      "typeVersion": 1,
      "position": [1050, 300],
      "credentials": {
        "postgres": {
          "name": "Railway PostgreSQL"
        }
      }
    }
  ],
  "connections": {
    "Daily 18:00": {
      "main": [[{"node": "Get Stock List", "type": "main", "index": 0}]]
    },
    "Get Stock List": {
      "main": [[{"node": "VNDirect API", "type": "main", "index": 0}]]
    },
    "VNDirect API": {
      "main": [[{"node": "Transform Data", "type": "main", "index": 0}]]
    },
    "Transform Data": {
      "main": [[{"node": "Save to PostgreSQL", "type": "main", "index": 0}]]
    }
  }
}
```

**How to import:**
1. Copy JSON above
2. n8n → Workflows → Import from JSON
3. Update PostgreSQL credentials
4. Activate workflow

---

## Workflow 2: Data Processing & Cleaning

**Trigger:** Cron - Daily at 18:30

### Workflow Structure:

```
[Cron Trigger: 0 18 30 * * *]
    ↓
[PostgreSQL: Get stocks with new data]
    ↓
[Loop: For Each Stock]
    ↓
[HTTP Request: Call FastAPI /process endpoint]
    ↓
[Code: Validation]
    ↓
[End Loop]
    ↓
[Notification]
```

### Simplified n8n Configuration:

**Nodes:**
1. **Schedule Trigger** - `30 18 * * *`
2. **PostgreSQL Query** - `SELECT DISTINCT symbol FROM stocks WHERE is_active = true`
3. **HTTP Request** - `POST http://your-api.railway.app/api/process/{symbol}`
4. **Webhook** (optional) - Send success/failure notification

---

## Workflow 3: Technical Indicators Calculation

**Trigger:** Cron - Daily at 19:00

### Workflow Structure:

```
[Cron Trigger: 0 19 * * *]
    ↓
[PostgreSQL: Get stocks]
    ↓
[Loop: For Each Stock]
    ↓
[PostgreSQL: Get last 200 prices]
    ↓
[Code: Calculate RSI, MACD, BB, etc.]
    ↓
[PostgreSQL: Insert into technical_indicators]
    ↓
[End Loop]
```

### Code Node for Indicators:

```javascript
// Calculate RSI (14 period)
function calculateRSI(prices, period = 14) {
  const changes = [];
  for (let i = 1; i < prices.length; i++) {
    changes.push(prices[i] - prices[i-1]);
  }
  
  const gains = changes.map(c => c > 0 ? c : 0);
  const losses = changes.map(c => c < 0 ? -c : 0);
  
  const avgGain = gains.slice(-period).reduce((a,b) => a+b, 0) / period;
  const avgLoss = losses.slice(-period).reduce((a,b) => a+b, 0) / period;
  
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - (100 / (1 + rs));
}

// Calculate SMA
function calculateSMA(prices, period) {
  const slice = prices.slice(-period);
  return slice.reduce((a,b) => a+b, 0) / period;
}

// Main calculation
const prices = $input.all().map(item => item.json.close);
const dates = $input.all().map(item => item.json.date);
const latestDate = dates[dates.length - 1];

return [{
  json: {
    symbol: $input.first().json.symbol,
    date: latestDate,
    rsi_14: calculateRSI(prices, 14),
    sma_20: calculateSMA(prices, 20),
    sma_50: calculateSMA(prices, 50),
    sma_200: calculateSMA(prices, 200)
  }
}];
```

---

## Workflow 4: Sentiment Analysis

**Trigger:** Cron - Daily at 20:00

### Workflow Structure:

```
[Cron Trigger: 0 20 * * *]
    ↓
[PostgreSQL: Get stocks]
    ↓
[Loop: For Each Stock]
    ↓
[HTTP Request: Scrape news (CafeF, VietStock)]
    ↓
[Code: Parse HTML]
    ↓
[HTTP Request: FinBERT API / HuggingFace]
    ↓
[Code: Aggregate sentiment]
    ↓
[PostgreSQL: Insert sentiment_analysis]
    ↓
[End Loop]
```

### Web Scraping Node:

```javascript
// Scrape CafeF news
const symbol = $json.symbol;
const url = `https://cafef.vn/timeline/${symbol}.chn`;

// Use HTTP Request node with response type HTML
// Then parse with Code node:

const $ = cheerio.load($input.first().binary.data.toString());
const articles = [];

$('.tlitem').each((i, elem) => {
  articles.push({
    title: $(elem).find('.title').text().trim(),
    date: $(elem).find('.time').text().trim(),
    link: $(elem).find('a').attr('href')
  });
});

return articles.map(a => ({ json: a }));
```

---

## Workflow 5: Model Training

**Trigger:** Cron - Weekly Sunday at 02:00

### Workflow Structure:

```
[Cron Trigger: 0 2 * * 0]
    ↓
[PostgreSQL: Get top 3 stocks by volume]
    ↓
[Loop: For Each Stock]
    ↓
[HTTP Request: POST /api/train]
    ↓
[Wait for completion]
    ↓
[PostgreSQL: Save model metrics]
    ↓
[End Loop]
    ↓
[Email: Training report]
```

### HTTP Request to FastAPI:

```json
{
  "method": "POST",
  "url": "http://your-api.railway.app/api/train",
  "body": {
    "symbol": "{{$json.symbol}}",
    "model_types": ["ARIMA", "Prophet", "LSTM", "GRU"],
    "train_test_split": 0.8
  },
  "timeout": 600000
}
```

---

## Workflow 6: Data Backup

**Trigger:** Cron - Weekly Sunday at 03:00

### Workflow Structure:

```
[Cron Trigger: 0 3 * * 0]
    ↓
[PostgreSQL: pg_dump command]
    ↓
[Code: Compress backup]
    ↓
[Google Drive: Upload]
    OR
    [AWS S3: Upload]
    ↓
[Notification: Success/Failure]
```

### PostgreSQL Backup Query:

```sql
COPY (SELECT * FROM stock_prices WHERE date >= CURRENT_DATE - INTERVAL '90 days')
TO '/tmp/backup_prices.csv' WITH CSV HEADER;

COPY (SELECT * FROM technical_indicators WHERE date >= CURRENT_DATE - INTERVAL '90 days')
TO '/tmp/backup_indicators.csv' WITH CSV HEADER;
```

---

## Workflow 7: Cleanup Old Data

**Trigger:** Cron - Weekly Sunday at 04:00

### Workflow Structure:

```
[Cron Trigger: 0 4 * * 0]
    ↓
[PostgreSQL: Delete old records]
    ↓
[PostgreSQL: VACUUM ANALYZE]
    ↓
[Notification]
```

### Cleanup Queries:

```sql
-- Delete raw data older than 2 years
DELETE FROM stock_prices 
WHERE date < CURRENT_DATE - INTERVAL '730 days';

-- Delete old predictions
DELETE FROM predictions 
WHERE target_date < CURRENT_DATE - INTERVAL '90 days';

-- Delete old news
DELETE FROM news_articles 
WHERE scraped_at < CURRENT_DATE - INTERVAL '90 days';

-- Optimize tables
VACUUM ANALYZE;
```

---

## 🔗 Integration with Railway

### Environment Variables in n8n:

```bash
# Set in n8n Settings → Environment Variables
DATABASE_URL=postgresql://...  # From Railway
API_URL=https://your-app.railway.app
WEBHOOK_URL=https://n8n.railway.app/webhook/kltn
```

### Webhook Trigger Example:

```json
{
  "nodes": [
    {
      "name": "Webhook",
      "type": "n8n-nodes-base.webhook",
      "parameters": {
        "path": "kltn-trigger",
        "responseMode": "lastNode",
        "options": {}
      }
    }
  ]
}
```

Trigger URL: `https://your-n8n.railway.app/webhook/kltn-trigger`

---

## 📊 Monitoring & Notifications

### Add to end of each workflow:

**Success Notification:**
```
[Slack/Email Node]
Message: "✅ {workflow_name} completed successfully
- Records processed: {{$json.count}}
- Duration: {{$workflow.duration}}ms"
```

**Error Notification:**
```
[Error Trigger]
    ↓
[Slack/Email Node]
Message: "❌ {workflow_name} failed
- Error: {{$json.error}}
- Time: {{$json.timestamp}}"
```

---

## 🚀 Quick Start Guide

### 1. Import All Workflows:
```bash
# Download workflow JSON files
curl -O https://your-repo/n8n-workflows.zip
unzip n8n-workflows.zip

# Import in n8n:
# Workflows → Import from File → Select each JSON
```

### 2. Configure Credentials:
- PostgreSQL: Railway connection string
- HTTP Request: FastAPI endpoint URL

### 3. Test Workflows:
```
1. Click "Execute Workflow" button
2. Check execution log
3. Verify database records
```

### 4. Activate All:
```
✅ Workflow 1: Data Collection
✅ Workflow 2: Processing
✅ Workflow 3: Indicators
✅ Workflow 4: Sentiment
✅ Workflow 5: Training (weekly)
✅ Workflow 6: Backup (weekly)
✅ Workflow 7: Cleanup (weekly)
```

---

## 📚 Resources

- **n8n Docs**: https://docs.n8n.io
- **PostgreSQL Node**: https://docs.n8n.io/integrations/builtin/app-nodes/n8n-nodes-base.postgres/
- **Cron Expression**: https://crontab.guru
- **Railway Deploy**: See `docs/RAILWAY_DEPLOYMENT.md`

---

## ⚠️ Important Notes

1. **Rate Limiting**: VNDirect API có giới hạn requests. Thêm delay giữa các calls.
2. **Error Handling**: Mỗi workflow nên có error handling node.
3. **Monitoring**: Setup alerts cho failed executions.
4. **Backup**: n8n workflows cũng nên được backup (export JSON định kỳ).
5. **Security**: Không hardcode credentials trong workflows.

---

## 🔄 Migration từ Python Scheduler

Để chuyển từ `automation/scheduler.py` sang n8n:

1. **Stop Python scheduler**:
   ```bash
   # Tắt process đang chạy
   pkill -f "python automation/scheduler.py"
   ```

2. **Import n8n workflows** (7 workflows ở trên)

3. **Activate workflows** trong n8n

4. **Verify**:
   - Kiểm tra database có dữ liệu mới
   - Kiểm tra logs trong n8n
   - Kiểm tra notifications

5. **Remove old code** (optional):
   ```bash
   # Archive scheduler.py
   mv automation/scheduler.py automation/scheduler.py.backup
   ```

---

**✅ Setup Complete! Your KLTN system is now running on n8n + Railway + PostgreSQL.**
