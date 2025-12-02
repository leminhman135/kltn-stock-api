"""Quick script to check dataset statistics"""
from src.database.models import Stock, StockPrice
from src.database.connection import SessionLocal
import pandas as pd

db = SessionLocal()

# Get statistics
stocks = db.query(Stock).all()
prices_count = db.query(StockPrice).count()

print(f"Total stocks: {len(stocks)}")
print(f"Total price records: {prices_count}")
print(f"\nStocks list:")
for stock in stocks[:10]:
    count = db.query(StockPrice).filter(StockPrice.stock_id == stock.id).count()
    print(f"  {stock.symbol}: {count} records")

# Sample data
sample = db.query(StockPrice).first()
if sample:
    print(f"\nSample record:")
    print(f"  Date: {sample.date}")
    print(f"  Open: {sample.open}, High: {sample.high}, Low: {sample.low}")
    print(f"  Close: {sample.close}, Volume: {sample.volume}")

db.close()
