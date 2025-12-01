
from .trading_data import TradingDataCollector, ForeignInvestorData, ProprietaryTradingData
from .market_data import MarketDataCollector, MarketIndex, OwnershipData
from .financial_data import FinancialDataCollector, ValuationData, FundamentalData
from .industry_data import IndustryDataCollector, SectorStatistics

# Import VNDirectAPI from the standalone data_collection.py module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.data_collection import VNDirectAPI
except ImportError:
    # Fallback: define a simple VNDirectAPI using TradingDataCollector
    import requests
    import pandas as pd
    from datetime import datetime
    
    class VNDirectAPI:
        """VNDirect API wrapper using dchart API"""
        
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.base_url = "https://dchart-api.vndirect.com.vn"
            self.session = requests.Session()
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Referer': 'https://dstock.vndirect.com.vn/',
                'Accept': 'application/json, text/plain, */*',
                'Origin': 'https://dstock.vndirect.com.vn'
            })
        
        def get_stock_price(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
            """Lấy dữ liệu giá từ VNDirect dchart API"""
            try:
                clean_symbol = symbol.replace('.VN', '').upper()
                
                start_dt = datetime.strptime(from_date.split()[0], '%Y-%m-%d')
                end_dt = datetime.strptime(to_date.split()[0], '%Y-%m-%d')
                
                from_ts = int(start_dt.timestamp())
                to_ts = int(end_dt.timestamp())
                
                url = f"{self.base_url}/dchart/history"
                params = {
                    'resolution': 'D',
                    'symbol': clean_symbol,
                    'from': from_ts,
                    'to': to_ts
                }
                
                response = self.session.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if data.get('s') == 'ok' and 't' in data:
                        records = []
                        timestamps = data.get('t', [])
                        opens = data.get('o', [])
                        highs = data.get('h', [])
                        lows = data.get('l', [])
                        closes = data.get('c', [])
                        volumes = data.get('v', [])
                        
                        for i in range(len(timestamps)):
                            dt = datetime.fromtimestamp(timestamps[i])
                            records.append({
                                'date': dt,
                                'Open': opens[i] * 1000,
                                'High': highs[i] * 1000,
                                'Low': lows[i] * 1000,
                                'Close': closes[i] * 1000,
                                'Volume': volumes[i],
                                'symbol': clean_symbol
                            })
                        
                        return pd.DataFrame(records)
                
                return pd.DataFrame()
            except Exception as e:
                print(f"Error fetching data: {e}")
                return pd.DataFrame()
        
        def get_stock_info(self, symbol: str):
            """Get basic stock info"""
            return {"symbol": symbol.upper()}

__all__ = [
    # Trading Data
    'TradingDataCollector',
    'ForeignInvestorData',
    'ProprietaryTradingData',
    
    # Market Data
    'MarketDataCollector',
    'MarketIndex',
    'OwnershipData',
    
    # Financial Data
    'FinancialDataCollector',
    'ValuationData',
    'FundamentalData',
    
    # Industry Data
    'IndustryDataCollector',
    'SectorStatistics',
    
    # VNDirect API
    'VNDirectAPI',
]
