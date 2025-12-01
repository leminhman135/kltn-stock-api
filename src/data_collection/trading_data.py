

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ForeignInvestorData:
    """Dữ liệu giao dịch nhà đầu tư nước ngoài"""
    symbol: str
    date: str
    
    # Khối lượng
    foreign_buy_volume: float  # Khối lượng mua
    foreign_sell_volume: float  # Khối lượng bán
    foreign_net_volume: float  # KL mua ròng (mua - bán)
    
    # Giá trị
    foreign_buy_value: float  # Giá trị mua (VND)
    foreign_sell_value: float  # Giá trị bán (VND)
    foreign_net_value: float  # GT mua ròng (VND)
    
    # Room nước ngoài
    foreign_room: Optional[float] = None  # Room còn lại cho NDTNN
    foreign_ownership_percent: Optional[float] = None  # % sở hữu hiện tại
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ProprietaryTradingData:
    """Dữ liệu giao dịch tự doanh (Proprietary Trading)"""
    symbol: str
    date: str
    
    # Khối lượng tự doanh
    prop_buy_volume: float  # Khối lượng mua tự doanh
    prop_sell_volume: float  # Khối lượng bán tự doanh
    prop_net_volume: float  # KL mua ròng tự doanh
    
    # Giá trị tự doanh
    prop_buy_value: float  # Giá trị mua tự doanh (VND)
    prop_sell_value: float  # Giá trị bán tự doanh (VND)
    prop_net_value: float  # GT mua ròng tự doanh (VND)
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DetailedTradingData:
    """Dữ liệu giao dịch chi tiết"""
    symbol: str
    date: str
    
    # Giá OHLC
    open: float
    high: float
    low: float
    close: float
    
    # Khối lượng
    volume: float  # Tổng khối lượng
    deal_volume: float  # KL khớp lệnh
    put_through_volume: float  # KL thỏa thuận
    
    # Giá trị giao dịch
    total_value: float  # Tổng GTGD (VND)
    deal_value: float  # GT khớp lệnh (VND)
    put_through_value: float  # GT thỏa thuận (VND)
    
    # Thay đổi giá
    price_change: float  # Thay đổi giá
    price_change_percent: float  # % thay đổi
    
    # Thống kê lệnh
    total_buy_orders: Optional[int] = None  # Tổng số lệnh mua
    total_sell_orders: Optional[int] = None  # Tổng số lệnh bán
    
    # Giá tham chiếu
    reference_price: Optional[float] = None  # Giá tham chiếu
    ceiling_price: Optional[float] = None  # Giá trần
    floor_price: Optional[float] = None  # Giá sàn
    
    def to_dict(self) -> Dict:
        return asdict(self)


class TradingDataCollector:
    """Thu thập dữ liệu giao dịch từ nhiều nguồn"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Origin': 'https://dstock.vndirect.com.vn',
            'Referer': 'https://dstock.vndirect.com.vn/',
        })
        
        # API endpoints
        self.dchart_api = "https://dchart-api.vndirect.com.vn"  # Primary - works best
        self.vndirect_api = "https://finfo-api.vndirect.com.vn"
        self.ssi_api = "https://iboard.ssi.com.vn"
        self.cafef_api = "https://s.cafef.vn"
        self.vps_api = "https://bgapidatafeed.vps.com.vn"
    
    def get_detailed_trading_data(self, symbol: str, from_date: str, 
                                   to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giao dịch chi tiết
        
        Args:
            symbol: Mã cổ phiếu (VD: 'VNM', 'FPT')
            from_date: Ngày bắt đầu (YYYY-MM-DD)
            to_date: Ngày kết thúc (YYYY-MM-DD)
        
        Returns:
            DataFrame với dữ liệu giao dịch chi tiết
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # Primary: Use dchart API (most reliable)
            df = self._get_from_dchart_api(clean_symbol, from_date, to_date)
            if not df.empty:
                return df
            
            # Fallback 1: Try finfo API
            df = self._get_from_finfo_api(clean_symbol, from_date, to_date)
            if not df.empty:
                return df
            
            # Fallback 2: Try SSI API
            return self._get_trading_from_ssi(clean_symbol, from_date, to_date)
        
        except Exception as e:
            logger.error(f"❌ Error fetching trading data: {str(e)}")
            return pd.DataFrame()
    
    def _get_from_dchart_api(self, symbol: str, from_date: str, 
                              to_date: str) -> pd.DataFrame:
        """Lấy dữ liệu từ VNDirect dchart API (Primary source)"""
        try:
            # Convert dates to timestamps
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            
            from_ts = int(start_dt.timestamp())
            to_ts = int(end_dt.timestamp())
            
            url = f"{self.dchart_api}/dchart/history"
            params = {
                'resolution': 'D',
                'symbol': symbol,
                'from': from_ts,
                'to': to_ts
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('s') == 'ok' and 't' in data:
                    # Parse response
                    records = []
                    timestamps = data.get('t', [])
                    opens = data.get('o', [])
                    highs = data.get('h', [])
                    lows = data.get('l', [])
                    closes = data.get('c', [])
                    volumes = data.get('v', [])
                    
                    for i in range(len(timestamps)):
                        dt = datetime.fromtimestamp(timestamps[i])
                        
                        # Calculate change
                        prev_close = closes[i-1] if i > 0 else opens[i]
                        price_change = closes[i] - prev_close
                        price_change_pct = (price_change / prev_close * 100) if prev_close > 0 else 0
                        
                        records.append({
                            'symbol': symbol,
                            'date': dt.strftime('%Y-%m-%d'),
                            'open': round(opens[i] * 1000, 0),  # Convert to VND
                            'high': round(highs[i] * 1000, 0),
                            'low': round(lows[i] * 1000, 0),
                            'close': round(closes[i] * 1000, 0),
                            'volume': volumes[i],
                            'deal_volume': volumes[i],
                            'put_through_volume': 0,
                            'total_value': 0,
                            'deal_value': 0,
                            'put_through_value': 0,
                            'price_change': round(price_change * 1000, 0),
                            'price_change_percent': round(price_change_pct, 2),
                            'reference_price': None,
                            'ceiling_price': None,
                            'floor_price': None,
                        })
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ [dchart] Fetched {len(df)} records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ dchart API error: {str(e)}")
            return pd.DataFrame()
    
    def _get_from_finfo_api(self, symbol: str, from_date: str, 
                             to_date: str) -> pd.DataFrame:
        """Lấy dữ liệu từ VNDirect finfo API (Fallback)"""
        try:
            url = f"{self.vndirect_api}/v4/stock_prices"
            params = {
                'q': f'code:{symbol}~date:gte:{from_date}~date:lte:{to_date}',
                'sort': 'date',
                'size': 1000
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        records.append({
                            'symbol': symbol,
                            'date': item.get('date', ''),
                            'open': float(item.get('open', 0)) * 1000,
                            'high': float(item.get('high', 0)) * 1000,
                            'low': float(item.get('low', 0)) * 1000,
                            'close': float(item.get('close', 0)) * 1000,
                            'volume': float(item.get('nmTotalTradedQty', 0)),
                            'deal_volume': float(item.get('nmTotalTradedQty', 0)),
                            'put_through_volume': float(item.get('ptTotalTradedQty', 0)),
                            'total_value': float(item.get('nmTotalTradedValue', 0)) + 
                                          float(item.get('ptTotalTradedValue', 0)),
                            'deal_value': float(item.get('nmTotalTradedValue', 0)),
                            'put_through_value': float(item.get('ptTotalTradedValue', 0)),
                            'price_change': float(item.get('change', 0)) * 1000,
                            'price_change_percent': float(item.get('pctChange', 0)),
                            'reference_price': float(item.get('basicPrice', 0)) * 1000,
                            'ceiling_price': float(item.get('ceilingPrice', 0)) * 1000,
                            'floor_price': float(item.get('floorPrice', 0)) * 1000,
                        })
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ [finfo] Fetched {len(df)} records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ finfo API error: {str(e)}")
            return pd.DataFrame()
    
    def _get_trading_from_ssi(self, symbol: str, from_date: str, 
                              to_date: str) -> pd.DataFrame:
        """Fallback: Lấy từ SSI API"""
        try:
            url = f"{self.ssi_api}/stock/group/price-history"
            params = {
                'symbol': symbol,
                'fromDate': from_date,
                'toDate': to_date
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data:
                    return pd.DataFrame(data['data'])
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"SSI API error: {str(e)}")
            return pd.DataFrame()
    
    def get_foreign_trading_data(self, symbol: str, from_date: str, 
                                  to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giao dịch NDTNN
        
        Args:
            symbol: Mã cổ phiếu
            from_date: Ngày bắt đầu
            to_date: Ngày kết thúc
        
        Returns:
            DataFrame với dữ liệu giao dịch NDTNN
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # VNDirect API for foreign trading
            url = f"{self.vndirect_api}/v4/foreignTrades"
            params = {
                'q': f'code:{clean_symbol}~tradingDate:gte:{from_date}~tradingDate:lte:{to_date}',
                'sort': 'tradingDate',
                'size': 1000
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        record = ForeignInvestorData(
                            symbol=clean_symbol,
                            date=item.get('tradingDate', ''),
                            foreign_buy_volume=float(item.get('buyQtty', 0)),
                            foreign_sell_volume=float(item.get('sellQtty', 0)),
                            foreign_net_volume=float(item.get('buyQtty', 0)) - 
                                              float(item.get('sellQtty', 0)),
                            foreign_buy_value=float(item.get('buyValue', 0)),
                            foreign_sell_value=float(item.get('sellValue', 0)),
                            foreign_net_value=float(item.get('buyValue', 0)) - 
                                             float(item.get('sellValue', 0)),
                            foreign_room=float(item.get('currentRoom', 0)) if item.get('currentRoom') else None,
                            foreign_ownership_percent=float(item.get('foreignPercent', 0)) if item.get('foreignPercent') else None,
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} foreign trading records for {symbol}")
                    return df
            
            # Fallback: CafeF API
            return self._get_foreign_from_cafef(clean_symbol, from_date, to_date)
        
        except Exception as e:
            logger.error(f"❌ Error fetching foreign trading data: {str(e)}")
            return pd.DataFrame()
    
    def _get_foreign_from_cafef(self, symbol: str, from_date: str, 
                                 to_date: str) -> pd.DataFrame:
        """Fallback: Lấy dữ liệu NDTNN từ CafeF"""
        try:
            url = f"{self.cafef_api}/Ajax/PageNew/DataHistory/GDNuocNgoai.ashx"
            params = {
                'symbol': symbol,
                'startDate': from_date,
                'endDate': to_date
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list) and len(data) > 0:
                    records = []
                    for item in data:
                        records.append({
                            'symbol': symbol,
                            'date': item.get('Ngay', ''),
                            'foreign_buy_volume': float(item.get('KLMua', 0)),
                            'foreign_sell_volume': float(item.get('KLBan', 0)),
                            'foreign_net_volume': float(item.get('KLMuaRong', 0)),
                            'foreign_buy_value': float(item.get('GTMua', 0)),
                            'foreign_sell_value': float(item.get('GTBan', 0)),
                            'foreign_net_value': float(item.get('GTMuaRong', 0)),
                        })
                    return pd.DataFrame(records)
            
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"CafeF API error: {str(e)}")
            return pd.DataFrame()
    
    def get_proprietary_trading_data(self, symbol: str, from_date: str, 
                                      to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giao dịch tự doanh
        
        Args:
            symbol: Mã cổ phiếu
            from_date: Ngày bắt đầu
            to_date: Ngày kết thúc
        
        Returns:
            DataFrame với dữ liệu giao dịch tự doanh
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # VNDirect API for proprietary trading
            url = f"{self.vndirect_api}/v4/proprietaryTrades"
            params = {
                'q': f'code:{clean_symbol}~tradingDate:gte:{from_date}~tradingDate:lte:{to_date}',
                'sort': 'tradingDate',
                'size': 1000
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        record = ProprietaryTradingData(
                            symbol=clean_symbol,
                            date=item.get('tradingDate', ''),
                            prop_buy_volume=float(item.get('buyQtty', 0)),
                            prop_sell_volume=float(item.get('sellQtty', 0)),
                            prop_net_volume=float(item.get('buyQtty', 0)) - 
                                           float(item.get('sellQtty', 0)),
                            prop_buy_value=float(item.get('buyValue', 0)),
                            prop_sell_value=float(item.get('sellValue', 0)),
                            prop_net_value=float(item.get('buyValue', 0)) - 
                                          float(item.get('sellValue', 0)),
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} proprietary trading records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching proprietary trading data: {str(e)}")
            return pd.DataFrame()
    
    def get_market_foreign_trading(self, exchange: str = 'HOSE', 
                                    date: Optional[str] = None) -> pd.DataFrame:
        """
        Lấy tổng hợp giao dịch NDTNN toàn thị trường
        
        Args:
            exchange: 'HOSE', 'HNX', 'UPCOM'
            date: Ngày cần lấy (mặc định: hôm nay)
        
        Returns:
            DataFrame với dữ liệu NDTNN toàn thị trường
        """
        try:
            if date is None:
                date = datetime.now().strftime('%Y-%m-%d')
            
            url = f"{self.vndirect_api}/v4/foreignTrades"
            params = {
                'q': f'floor:{exchange}~tradingDate:{date}',
                'sort': 'buyValue:DESC',
                'size': 500
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    
                    # Tính tổng hợp
                    if not df.empty:
                        summary = {
                            'exchange': exchange,
                            'date': date,
                            'total_buy_volume': df['buyQtty'].sum() if 'buyQtty' in df else 0,
                            'total_sell_volume': df['sellQtty'].sum() if 'sellQtty' in df else 0,
                            'total_buy_value': df['buyValue'].sum() if 'buyValue' in df else 0,
                            'total_sell_value': df['sellValue'].sum() if 'sellValue' in df else 0,
                            'stock_count': len(df),
                        }
                        summary['net_volume'] = summary['total_buy_volume'] - summary['total_sell_volume']
                        summary['net_value'] = summary['total_buy_value'] - summary['total_sell_value']
                        
                        logger.info(f"✅ Market foreign trading: Net value = {summary['net_value']:,.0f} VND")
                        return df, summary
            
            return pd.DataFrame(), {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching market foreign trading: {str(e)}")
            return pd.DataFrame(), {}
    
    def get_intraday_trading(self, symbol: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giao dịch trong ngày (real-time)
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame với dữ liệu giao dịch trong ngày
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # SSI API for intraday
            url = f"{self.ssi_api}/stock/group/trade-history"
            params = {'symbol': clean_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and 'items' in data['data']:
                    records = []
                    for item in data['data']['items']:
                        records.append({
                            'symbol': clean_symbol,
                            'time': item.get('t', ''),
                            'price': float(item.get('p', 0)),
                            'volume': float(item.get('v', 0)),
                            'match_type': item.get('type', ''),  # ATO, ATC, Continuous
                            'side': item.get('side', ''),  # Buy/Sell
                        })
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} intraday trades for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching intraday trading: {str(e)}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str) -> Dict:
        """
        Lấy sổ lệnh (Order Book) - 3 bước giá tốt nhất
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict với order book data
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # SSI API for order book
            url = f"{self.ssi_api}/stock/quotes"
            params = {'symbol': clean_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    quote = data['data']
                    order_book = {
                        'symbol': clean_symbol,
                        'timestamp': datetime.now().isoformat(),
                        
                        # Giá khớp
                        'last_price': float(quote.get('lastPrice', 0)),
                        'last_volume': float(quote.get('lastVol', 0)),
                        
                        # 3 bước giá mua tốt nhất
                        'bid': [
                            {'price': float(quote.get('bid1Price', 0)), 'volume': float(quote.get('bid1Vol', 0))},
                            {'price': float(quote.get('bid2Price', 0)), 'volume': float(quote.get('bid2Vol', 0))},
                            {'price': float(quote.get('bid3Price', 0)), 'volume': float(quote.get('bid3Vol', 0))},
                        ],
                        
                        # 3 bước giá bán tốt nhất
                        'ask': [
                            {'price': float(quote.get('ask1Price', 0)), 'volume': float(quote.get('ask1Vol', 0))},
                            {'price': float(quote.get('ask2Price', 0)), 'volume': float(quote.get('ask2Vol', 0))},
                            {'price': float(quote.get('ask3Price', 0)), 'volume': float(quote.get('ask3Vol', 0))},
                        ],
                        
                        # Tổng hợp
                        'total_bid_volume': float(quote.get('totalBidVol', 0)),
                        'total_ask_volume': float(quote.get('totalAskVol', 0)),
                    }
                    
                    logger.info(f"✅ Fetched order book for {symbol}")
                    return order_book
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching order book: {str(e)}")
            return {}
    
    def get_trading_summary(self, symbol: str, days: int = 30) -> Dict:
        """
        Lấy tổng hợp dữ liệu giao dịch
        
        Args:
            symbol: Mã cổ phiếu
            days: Số ngày lấy dữ liệu
        
        Returns:
            Dict với tổng hợp giao dịch
        """
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            from_str = from_date.strftime('%Y-%m-%d')
            to_str = to_date.strftime('%Y-%m-%d')
            
            # Lấy dữ liệu
            trading_df = self.get_detailed_trading_data(symbol, from_str, to_str)
            foreign_df = self.get_foreign_trading_data(symbol, from_str, to_str)
            prop_df = self.get_proprietary_trading_data(symbol, from_str, to_str)
            
            summary = {
                'symbol': symbol.replace('.VN', '').upper(),
                'period': f'{from_str} - {to_str}',
                'days': days,
                
                # Trading summary
                'trading': {
                    'avg_volume': trading_df['volume'].mean() if not trading_df.empty else 0,
                    'avg_value': trading_df['total_value'].mean() if not trading_df.empty else 0,
                    'max_volume': trading_df['volume'].max() if not trading_df.empty else 0,
                    'min_volume': trading_df['volume'].min() if not trading_df.empty else 0,
                    'total_volume': trading_df['volume'].sum() if not trading_df.empty else 0,
                    'total_value': trading_df['total_value'].sum() if not trading_df.empty else 0,
                },
                
                # Foreign investor summary
                'foreign': {
                    'total_buy_volume': foreign_df['foreign_buy_volume'].sum() if not foreign_df.empty else 0,
                    'total_sell_volume': foreign_df['foreign_sell_volume'].sum() if not foreign_df.empty else 0,
                    'net_volume': foreign_df['foreign_net_volume'].sum() if not foreign_df.empty else 0,
                    'total_buy_value': foreign_df['foreign_buy_value'].sum() if not foreign_df.empty else 0,
                    'total_sell_value': foreign_df['foreign_sell_value'].sum() if not foreign_df.empty else 0,
                    'net_value': foreign_df['foreign_net_value'].sum() if not foreign_df.empty else 0,
                    'avg_daily_net': foreign_df['foreign_net_value'].mean() if not foreign_df.empty else 0,
                },
                
                # Proprietary trading summary
                'proprietary': {
                    'total_buy_volume': prop_df['prop_buy_volume'].sum() if not prop_df.empty else 0,
                    'total_sell_volume': prop_df['prop_sell_volume'].sum() if not prop_df.empty else 0,
                    'net_volume': prop_df['prop_net_volume'].sum() if not prop_df.empty else 0,
                    'total_buy_value': prop_df['prop_buy_value'].sum() if not prop_df.empty else 0,
                    'total_sell_value': prop_df['prop_sell_value'].sum() if not prop_df.empty else 0,
                    'net_value': prop_df['prop_net_value'].sum() if not prop_df.empty else 0,
                },
            }
            
            logger.info(f"✅ Generated trading summary for {symbol}")
            return summary
        
        except Exception as e:
            logger.error(f"❌ Error generating trading summary: {str(e)}")
            return {}


# Convenience functions
def get_foreign_net_buying(symbols: List[str], days: int = 5) -> pd.DataFrame:
    """Lấy danh sách cổ phiếu được NDTNN mua ròng"""
    collector = TradingDataCollector()
    
    to_date = datetime.now()
    from_date = to_date - timedelta(days=days)
    
    results = []
    for symbol in symbols:
        df = collector.get_foreign_trading_data(
            symbol, 
            from_date.strftime('%Y-%m-%d'),
            to_date.strftime('%Y-%m-%d')
        )
        
        if not df.empty:
            net_value = df['foreign_net_value'].sum()
            net_volume = df['foreign_net_volume'].sum()
            
            results.append({
                'symbol': symbol,
                'net_value': net_value,
                'net_volume': net_volume,
                'days': days,
            })
    
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values('net_value', ascending=False)
    
    return result_df


if __name__ == "__main__":
    # Test
    collector = TradingDataCollector()
    
    symbol = "VNM"
    from_date = "2024-11-01"
    to_date = "2024-11-30"
    
    # Test detailed trading data
    print("\n=== Detailed Trading Data ===")
    df = collector.get_detailed_trading_data(symbol, from_date, to_date)
    print(df.head())
    
    # Test foreign trading data
    print("\n=== Foreign Trading Data ===")
    df = collector.get_foreign_trading_data(symbol, from_date, to_date)
    print(df.head())
    
    # Test trading summary
    print("\n=== Trading Summary ===")
    summary = collector.get_trading_summary(symbol, days=30)
    print(json.dumps(summary, indent=2, default=str))
