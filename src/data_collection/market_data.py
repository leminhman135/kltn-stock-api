"""
Market Data Collection Module

Thu thập dữ liệu thị trường:
- Chỉ số thị trường: VN-Index, HNX-Index, VN30-Index, etc.
- Tỷ lệ Freefloat
- Tỷ lệ sở hữu nước ngoài
- Thông tin room nước ngoài
"""

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
class MarketIndex:
    """Dữ liệu chỉ số thị trường"""
    index_code: str  # VNINDEX, HNXINDEX, VN30, etc.
    date: str
    
    # Giá trị chỉ số
    open: float
    high: float
    low: float
    close: float
    
    # Thay đổi
    change: float  # Điểm thay đổi
    change_percent: float  # % thay đổi
    
    # Khối lượng và giá trị
    volume: float  # Tổng KLGD
    value: float  # Tổng GTGD (tỷ VND)
    
    # Thống kê cổ phiếu
    advances: int  # Số CP tăng
    declines: int  # Số CP giảm
    unchanged: int  # Số CP không đổi
    
    # Trần/sàn
    ceiling_count: int  # Số CP tăng trần
    floor_count: int  # Số CP giảm sàn
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class OwnershipData:
    """Dữ liệu sở hữu cổ phiếu"""
    symbol: str
    date: str
    
    # Sở hữu nước ngoài
    foreign_percent: float  # % sở hữu NDTNN hiện tại
    foreign_limit: float  # Room nước ngoài tối đa (%)
    foreign_room: float  # Room còn lại (số CP)
    foreign_room_percent: float  # Room còn lại (%)
    
    # Freefloat
    freefloat_ratio: float  # Tỷ lệ freefloat (%)
    freefloat_shares: float  # Số CP tự do chuyển nhượng
    
    # Cổ đông
    total_shares: float  # Tổng số CP
    listed_shares: float  # Số CP niêm yết
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class IndexComponent:
    """Thông tin thành phần chỉ số"""
    index_code: str
    symbol: str
    weight: float  # Tỷ trọng (%)
    market_cap: float  # Vốn hóa
    freefloat_cap: float  # Vốn hóa freefloat
    shares_in_index: float  # Số CP trong rổ
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MarketDataCollector:
    """Thu thập dữ liệu thị trường"""
    
    # Danh sách chỉ số thị trường Việt Nam
    MARKET_INDICES = {
        'VNINDEX': 'VN-Index - Chỉ số chính HOSE',
        'VN30': 'VN30 - Top 30 HOSE',
        'VN100': 'VN100 - Top 100 HOSE',
        'VNMIDCAP': 'VN Mid Cap - Vốn hóa trung bình',
        'VNSMALLCAP': 'VN Small Cap - Vốn hóa nhỏ',
        'VNALLSHARE': 'VN All Share - Toàn bộ HOSE',
        'VNDIAMOND': 'VN Diamond - CP tài chính ưu tú',
        'VNFINLEAD': 'VN Fin Lead - Dẫn đầu tài chính',
        'VNFIN_SELECT': 'VN Fin Select - Chọn lọc tài chính',
        'VNREAL': 'VNReal - Bất động sản',
        'HNXINDEX': 'HNX Index - Chỉ số HNX',
        'HNX30': 'HNX30 - Top 30 HNX',
        'HNXLCAP': 'HNX LCap - Vốn hóa lớn HNX',
        'HNXSMCAP': 'HNX Small Cap - Vốn hóa nhỏ HNX',
        'HNXFIN': 'HNX Fin - Tài chính HNX',
        'HNXMAN': 'HNX Man - Công nghiệp HNX',
        'HNXCON': 'HNX Con - Xây dựng HNX',
        'UPCOM': 'UPCOM Index - Chỉ số UPCOM',
    }
    
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
        self.vndirect_api = "https://finfo-api.vndirect.com.vn"
        self.dchart_api = "https://dchart-api.vndirect.com.vn"
        self.ssi_api = "https://iboard.ssi.com.vn"
        self.hose_api = "https://www.hsx.vn"
    
    def get_index_data(self, index_code: str, from_date: str, 
                       to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu lịch sử chỉ số
        
        Args:
            index_code: Mã chỉ số (VNINDEX, HNX30, etc.)
            from_date: Ngày bắt đầu (YYYY-MM-DD)
            to_date: Ngày kết thúc (YYYY-MM-DD)
        
        Returns:
            DataFrame với dữ liệu chỉ số
        """
        try:
            # Convert dates to timestamps
            start_dt = datetime.strptime(from_date, '%Y-%m-%d')
            end_dt = datetime.strptime(to_date, '%Y-%m-%d')
            
            from_ts = int(start_dt.timestamp())
            to_ts = int(end_dt.timestamp())
            
            # VNDirect dchart API
            url = f"{self.dchart_api}/dchart/history"
            params = {
                'resolution': 'D',
                'symbol': index_code,
                'from': from_ts,
                'to': to_ts
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('s') == 'ok' and all(k in data for k in ['t', 'o', 'h', 'l', 'c', 'v']):
                    df = pd.DataFrame({
                        'date': pd.to_datetime(data['t'], unit='s'),
                        'index_code': index_code,
                        'open': data['o'],
                        'high': data['h'],
                        'low': data['l'],
                        'close': data['c'],
                        'volume': data['v'],
                    })
                    
                    # Tính thay đổi
                    df['change'] = df['close'].diff()
                    df['change_percent'] = df['close'].pct_change() * 100
                    
                    df = df.sort_values('date').reset_index(drop=True)
                    logger.info(f"✅ Fetched {len(df)} index records for {index_code}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching index data: {str(e)}")
            return pd.DataFrame()
    
    def get_realtime_index(self, index_code: str = 'VNINDEX') -> Dict:
        """
        Lấy dữ liệu chỉ số realtime
        
        Args:
            index_code: Mã chỉ số
        
        Returns:
            Dict với dữ liệu realtime
        """
        try:
            # SSI API for realtime
            url = f"{self.ssi_api}/stock/quotes"
            params = {'symbol': index_code}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    idx = data['data']
                    result = MarketIndex(
                        index_code=index_code,
                        date=datetime.now().strftime('%Y-%m-%d'),
                        open=float(idx.get('openPrice', 0)),
                        high=float(idx.get('highPrice', 0)),
                        low=float(idx.get('lowPrice', 0)),
                        close=float(idx.get('lastPrice', 0)),
                        change=float(idx.get('change', 0)),
                        change_percent=float(idx.get('changePercent', 0)),
                        volume=float(idx.get('totalVol', 0)),
                        value=float(idx.get('totalVal', 0)) / 1e9,  # Convert to tỷ VND
                        advances=int(idx.get('advances', 0)),
                        declines=int(idx.get('declines', 0)),
                        unchanged=int(idx.get('unchanged', 0)),
                        ceiling_count=int(idx.get('ceilingCount', 0)),
                        floor_count=int(idx.get('floorCount', 0)),
                    )
                    
                    logger.info(f"✅ Fetched realtime {index_code}: {result.close} ({result.change_percent:+.2f}%)")
                    return result.to_dict()
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching realtime index: {str(e)}")
            return {}
    
    def get_all_indices_realtime(self) -> pd.DataFrame:
        """Lấy dữ liệu realtime tất cả chỉ số"""
        results = []
        
        for index_code in ['VNINDEX', 'VN30', 'HNX', 'HNXINDEX', 'HNX30', 'UPCOM']:
            data = self.get_realtime_index(index_code)
            if data:
                results.append(data)
        
        return pd.DataFrame(results)
    
    def get_freefloat_data(self, symbol: str) -> Dict:
        """
        Lấy thông tin freefloat
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict với thông tin freefloat
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # VNDirect API
            url = f"{self.vndirect_api}/v4/stocks"
            params = {'q': f'code:{clean_symbol}'}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    stock = data['data'][0]
                    
                    total_shares = float(stock.get('listedQty', 0) or stock.get('outstandingShare', 0))
                    freefloat = float(stock.get('freeFloat', 0))
                    
                    result = {
                        'symbol': clean_symbol,
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'total_shares': total_shares,
                        'listed_shares': float(stock.get('listedQty', 0)),
                        'outstanding_shares': float(stock.get('outstandingShare', 0)),
                        'freefloat_ratio': freefloat,
                        'freefloat_shares': total_shares * freefloat / 100 if freefloat else 0,
                        
                        # Thông tin bổ sung
                        'company_name': stock.get('companyName', ''),
                        'exchange': stock.get('exchange', ''),
                        'industry': stock.get('industryName', ''),
                    }
                    
                    logger.info(f"✅ Freefloat {symbol}: {freefloat:.1f}%")
                    return result
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching freefloat data: {str(e)}")
            return {}
    
    def get_foreign_ownership(self, symbol: str) -> Dict:
        """
        Lấy thông tin sở hữu nước ngoài
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict với thông tin sở hữu NDTNN
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # VNDirect API
            url = f"{self.vndirect_api}/v4/ownership"
            params = {'q': f'code:{clean_symbol}'}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    ownership = data['data'][0]
                    
                    result = OwnershipData(
                        symbol=clean_symbol,
                        date=datetime.now().strftime('%Y-%m-%d'),
                        foreign_percent=float(ownership.get('foreignPercent', 0)),
                        foreign_limit=float(ownership.get('foreignMaxPercent', 49)),
                        foreign_room=float(ownership.get('currentRoom', 0)),
                        foreign_room_percent=float(ownership.get('roomPercent', 0)),
                        freefloat_ratio=float(ownership.get('freeFloat', 0)),
                        freefloat_shares=float(ownership.get('freeFloatShare', 0)),
                        total_shares=float(ownership.get('outstandingShare', 0)),
                        listed_shares=float(ownership.get('listedQty', 0)),
                    )
                    
                    logger.info(f"✅ Foreign ownership {symbol}: {result.foreign_percent:.1f}% / {result.foreign_limit:.1f}%")
                    return result.to_dict()
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching foreign ownership: {str(e)}")
            return {}
    
    def get_index_components(self, index_code: str = 'VN30') -> pd.DataFrame:
        """
        Lấy danh sách thành phần của chỉ số
        
        Args:
            index_code: Mã chỉ số (VN30, HNX30, etc.)
        
        Returns:
            DataFrame với danh sách CP thành phần
        """
        try:
            # VNDirect API for index components
            url = f"{self.vndirect_api}/v4/index_components"
            params = {
                'q': f'indexCode:{index_code}',
                'size': 100
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        record = IndexComponent(
                            index_code=index_code,
                            symbol=item.get('code', ''),
                            weight=float(item.get('weight', 0)),
                            market_cap=float(item.get('marketCap', 0)),
                            freefloat_cap=float(item.get('freeFloatCap', 0)),
                            shares_in_index=float(item.get('shareInIndex', 0)),
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    df = df.sort_values('weight', ascending=False)
                    
                    logger.info(f"✅ Fetched {len(df)} components for {index_code}")
                    return df
            
            # Fallback: hardcoded VN30 list
            if index_code == 'VN30':
                vn30_stocks = [
                    'ACB', 'BCM', 'BID', 'BVH', 'CTG', 'FPT', 'GAS', 'GVR', 'HDB', 'HPG',
                    'MBB', 'MSN', 'MWG', 'NVL', 'PDR', 'PLX', 'POW', 'SAB', 'SHB', 'SSI',
                    'STB', 'TCB', 'TPB', 'VCB', 'VHM', 'VIB', 'VIC', 'VJC', 'VNM', 'VPB'
                ]
                return pd.DataFrame({'symbol': vn30_stocks, 'index_code': 'VN30'})
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching index components: {str(e)}")
            return pd.DataFrame()
    
    def get_market_summary(self, exchange: str = 'HOSE') -> Dict:
        """
        Lấy tổng hợp thị trường
        
        Args:
            exchange: 'HOSE', 'HNX', 'UPCOM'
        
        Returns:
            Dict với tổng hợp thị trường
        """
        try:
            # Lấy chỉ số chính
            if exchange == 'HOSE':
                main_index = self.get_realtime_index('VNINDEX')
                vn30_index = self.get_realtime_index('VN30')
            elif exchange == 'HNX':
                main_index = self.get_realtime_index('HNXINDEX')
                vn30_index = self.get_realtime_index('HNX30')
            else:
                main_index = self.get_realtime_index('UPCOM')
                vn30_index = {}
            
            summary = {
                'exchange': exchange,
                'timestamp': datetime.now().isoformat(),
                'main_index': main_index,
                'sub_index': vn30_index,
                
                # Market breadth từ main_index
                'advances': main_index.get('advances', 0),
                'declines': main_index.get('declines', 0),
                'unchanged': main_index.get('unchanged', 0),
                'ceiling': main_index.get('ceiling_count', 0),
                'floor': main_index.get('floor_count', 0),
                
                # Volume và Value
                'total_volume': main_index.get('volume', 0),
                'total_value': main_index.get('value', 0),
            }
            
            # Tính market breadth ratio
            total_stocks = summary['advances'] + summary['declines'] + summary['unchanged']
            if total_stocks > 0:
                summary['advance_decline_ratio'] = summary['advances'] / max(summary['declines'], 1)
                summary['advance_percent'] = (summary['advances'] / total_stocks) * 100
                summary['decline_percent'] = (summary['declines'] / total_stocks) * 100
            
            logger.info(f"✅ Market summary for {exchange}")
            return summary
        
        except Exception as e:
            logger.error(f"❌ Error fetching market summary: {str(e)}")
            return {}
    
    def get_stocks_by_foreign_ownership(self, exchange: str = 'HOSE', 
                                        min_percent: float = 40) -> pd.DataFrame:
        """
        Lấy danh sách CP có tỷ lệ sở hữu nước ngoài cao
        
        Args:
            exchange: Sàn giao dịch
            min_percent: Ngưỡng % sở hữu tối thiểu
        
        Returns:
            DataFrame danh sách CP
        """
        try:
            url = f"{self.vndirect_api}/v4/ownership"
            params = {
                'q': f'floor:{exchange}~foreignPercent:gte:{min_percent}',
                'sort': 'foreignPercent:DESC',
                'size': 100
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    
                    columns_map = {
                        'code': 'symbol',
                        'foreignPercent': 'foreign_percent',
                        'foreignMaxPercent': 'foreign_limit',
                        'currentRoom': 'foreign_room',
                        'freeFloat': 'freefloat_ratio',
                    }
                    
                    df = df.rename(columns=columns_map)
                    
                    logger.info(f"✅ Found {len(df)} stocks with foreign ownership >= {min_percent}%")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching stocks by foreign ownership: {str(e)}")
            return pd.DataFrame()
    
    def get_stocks_near_foreign_limit(self, threshold: float = 5) -> pd.DataFrame:
        """
        Lấy danh sách CP sắp hết room nước ngoài
        
        Args:
            threshold: Ngưỡng room còn lại (%)
        
        Returns:
            DataFrame danh sách CP
        """
        try:
            url = f"{self.vndirect_api}/v4/ownership"
            params = {
                'q': f'roomPercent:lte:{threshold}~roomPercent:gt:0',
                'sort': 'roomPercent:ASC',
                'size': 50
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    
                    logger.info(f"✅ Found {len(df)} stocks near foreign limit")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching stocks near foreign limit: {str(e)}")
            return pd.DataFrame()
    
    def compare_indices(self, from_date: str, to_date: str) -> pd.DataFrame:
        """
        So sánh hiệu suất các chỉ số
        
        Args:
            from_date: Ngày bắt đầu
            to_date: Ngày kết thúc
        
        Returns:
            DataFrame so sánh các chỉ số
        """
        indices = ['VNINDEX', 'VN30', 'HNXINDEX', 'HNX30', 'UPCOM']
        
        results = []
        for idx in indices:
            df = self.get_index_data(idx, from_date, to_date)
            
            if not df.empty and len(df) >= 2:
                start_value = df.iloc[0]['close']
                end_value = df.iloc[-1]['close']
                
                results.append({
                    'index': idx,
                    'start_value': start_value,
                    'end_value': end_value,
                    'change': end_value - start_value,
                    'change_percent': ((end_value / start_value) - 1) * 100,
                    'high': df['high'].max(),
                    'low': df['low'].min(),
                    'volatility': df['change_percent'].std(),
                })
        
        return pd.DataFrame(results)


if __name__ == "__main__":
    # Test
    collector = MarketDataCollector()
    
    # Test index data
    print("\n=== VN-Index Data ===")
    df = collector.get_index_data('VNINDEX', '2024-11-01', '2024-11-30')
    print(df.head())
    
    # Test realtime index
    print("\n=== Realtime VN-Index ===")
    data = collector.get_realtime_index('VNINDEX')
    print(json.dumps(data, indent=2, default=str))
    
    # Test freefloat
    print("\n=== Freefloat Data ===")
    data = collector.get_freefloat_data('VNM')
    print(json.dumps(data, indent=2, default=str))
    
    # Test foreign ownership
    print("\n=== Foreign Ownership ===")
    data = collector.get_foreign_ownership('VNM')
    print(json.dumps(data, indent=2, default=str))
    
    # Test VN30 components
    print("\n=== VN30 Components ===")
    df = collector.get_index_components('VN30')
    print(df.head(10))
