"""Test model predictions"""
import os
os.environ['DATABASE_URL'] = 'postgresql://postgres:ZMuToaDyJBZDyghpCnpvQRrxYjCSxsEs@yamanote.proxy.rlwy.net:47879/railway'

import psycopg2
import pandas as pd
from src.model import StockMLModel

conn = psycopg2.connect(os.environ['DATABASE_URL'])
cur = conn.cursor()
cur.execute('SELECT id, symbol FROM stocks LIMIT 5')
stocks = cur.fetchall()

print("="*70)
print("TEST MODEL PREDICTIONS - REALISTIC CHECK")
print("="*70)

for stock_id, symbol in stocks:
    df = pd.read_sql(
        f'SELECT date, open, high, low, close, volume FROM stock_prices WHERE stock_id = {stock_id} ORDER BY date', 
        conn
    )
    if len(df) < 50:
        continue
    
    model = StockMLModel(symbol)
    model.train(df)
    pred = model.predict(df, steps=5)
    
    lp = pred.get('last_price', 0)
    preds = pred.get('predictions', [])
    
    print(f"\n{symbol}:")
    print(f"  Last price: {lp:,.0f}")
    print(f"  Model useful: {pred.get('model_useful')}")
    print(f"  Accuracy: {pred.get('model_accuracy', 'N/A')}%")
    print(f"  Daily volatility: {pred.get('daily_volatility', 'N/A')}%")
    print(f"  Trend: {pred.get('trend')}")
    print(f"  Predictions:")
    for d, p in zip(pred.get('dates', []), preds):
        change = (p - lp) / lp * 100
        print(f"    {d}: {p:,.0f} ({change:+.2f}%)")
    
    if pred.get('warning'):
        print(f"  ⚠️ {pred.get('warning')}")

conn.close()
print("\n" + "="*70)
print("Test completed!")
