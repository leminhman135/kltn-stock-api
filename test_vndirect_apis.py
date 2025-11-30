"""
Script test các API endpoints từ VNDirect
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from src.data_collection import VNDirectAPI
import pandas as pd

def test_all_apis():
    """Test tất cả các API endpoints"""
    
    api = VNDirectAPI()
    symbol = "VNM.VN"
    
    print("=" * 80)
    print("TESTING VNDIRECT API ENDPOINTS")
    print("=" * 80)
    
    # 1. Test Stock Price (Historical)
    print("\n📊 1. HISTORICAL PRICE DATA")
    print("-" * 80)
    df_price = api.get_stock_price(symbol, "2024-11-01", "2024-11-30")
    if not df_price.empty:
        print(f"✅ Records: {len(df_price)}")
        print(df_price.head())
    else:
        print("❌ No data")
    
    # 2. Test Stock Info
    print("\n📋 2. STOCK INFO")
    print("-" * 80)
    info = api.get_stock_info(symbol)
    if info:
        print(f"✅ Company: {info.get('companyName', 'N/A')}")
        print(f"   Exchange: {info.get('exchange', 'N/A')}")
        print(f"   Industry: {info.get('industryName', 'N/A')}")
        print(f"   Market Cap: {info.get('marketCap', 'N/A'):,}")
    else:
        print("❌ No info")
    
    # 3. Test Intraday Data
    print("\n⏱️ 3. INTRADAY DATA (5 min)")
    print("-" * 80)
    df_intraday = api.get_intraday_data(symbol, resolution='5')
    if not df_intraday.empty:
        print(f"✅ Records: {len(df_intraday)}")
        print(df_intraday.head())
    else:
        print("❌ No intraday data")
    
    # 4. Test Market Overview
    print("\n🏛️ 4. MARKET OVERVIEW")
    print("-" * 80)
    market = api.get_market_overview()
    if market:
        print(f"✅ Market data retrieved")
        print(f"   Keys: {list(market.keys())[:5]}")
    else:
        print("❌ No market data")
    
    # 5. Test Top Stocks
    print("\n🔝 5. TOP STOCKS BY VALUE (HOSE)")
    print("-" * 80)
    df_top = api.get_top_stocks(market='HOSE', criteria='value')
    if not df_top.empty:
        print(f"✅ Top stocks: {len(df_top)}")
        if 'code' in df_top.columns and 'totalTradingValue' in df_top.columns:
            print(df_top[['code', 'lastPrice', 'totalTradingValue']].head(10))
    else:
        print("❌ No top stocks data")
    
    # 6. Test Financial Ratios
    print("\n💰 6. FINANCIAL RATIOS")
    print("-" * 80)
    ratios = api.get_financial_ratios(symbol)
    if ratios:
        print(f"✅ Ratios available")
        pe = ratios.get('pe', 'N/A')
        pb = ratios.get('pb', 'N/A')
        roe = ratios.get('roe', 'N/A')
        roa = ratios.get('roa', 'N/A')
        print(f"   P/E: {pe}")
        print(f"   P/B: {pb}")
        print(f"   ROE: {roe}")
        print(f"   ROA: {roa}")
    else:
        print("❌ No ratios data")
    
    # 7. Test Financial Statements
    print("\n📑 7. FINANCIAL STATEMENTS (Balance Sheet)")
    print("-" * 80)
    df_balance = api.get_financial_statements(symbol, report_type='BALANCE_SHEET', period='YEAR')
    if not df_balance.empty:
        print(f"✅ Records: {len(df_balance)}")
        if 'year' in df_balance.columns:
            print(f"   Years: {df_balance['year'].tolist()[:5]}")
    else:
        print("❌ No financial statements")
    
    # 8. Test Ownership Data
    print("\n👥 8. OWNERSHIP DATA")
    print("-" * 80)
    ownership = api.get_ownership_data(symbol)
    if ownership:
        print(f"✅ Ownership data available")
        foreign = ownership.get('foreignCurrentRoom', 'N/A')
        foreign_pct = ownership.get('foreignPercent', 'N/A')
        print(f"   Foreign room: {foreign}")
        print(f"   Foreign %: {foreign_pct}")
    else:
        print("❌ No ownership data")
    
    # 9. Test Dividends
    print("\n💵 9. DIVIDEND HISTORY")
    print("-" * 80)
    df_div = api.get_dividends(symbol)
    if not df_div.empty:
        print(f"✅ Dividend records: {len(df_div)}")
        if 'issueDate' in df_div.columns and 'cashDividend' in df_div.columns:
            print(df_div[['issueDate', 'cashDividend']].head())
    else:
        print("❌ No dividend data")
    
    # 10. Test Corporate Actions
    print("\n🏢 10. CORPORATE ACTIONS")
    print("-" * 80)
    df_actions = api.get_corporate_actions(symbol)
    if not df_actions.empty:
        print(f"✅ Corporate actions: {len(df_actions)}")
        if 'eventName' in df_actions.columns:
            print(df_actions['eventName'].head())
    else:
        print("❌ No corporate actions")
    
    # 11. Test Search
    print("\n🔍 11. SEARCH STOCKS")
    print("-" * 80)
    df_search = api.search_stocks("vinamilk")
    if not df_search.empty:
        print(f"✅ Found: {len(df_search)} stocks")
        if 'code' in df_search.columns and 'companyName' in df_search.columns:
            print(df_search[['code', 'companyName']].head())
    else:
        print("❌ No search results")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    test_all_apis()
