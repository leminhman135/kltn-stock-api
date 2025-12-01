"""
Industry Data Collection Module

Thu thập dữ liệu thống kê ngành:
- Thống kê theo ngành và nhóm ngành
- Cung cầu thị trường
- So sánh hiệu suất ngành
- Phân tích sector rotation
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
class SectorStatistics:
    """Thống kê ngành"""
    sector_code: str
    sector_name: str
    date: str
    
    # Số lượng CP
    stock_count: int
    
    # Giá trị thị trường
    total_market_cap: float  # Tổng vốn hóa (tỷ VND)
    avg_market_cap: float  # Vốn hóa trung bình
    
    # Hiệu suất
    avg_change_percent: float  # Thay đổi giá TB (%)
    advances: int  # Số CP tăng
    declines: int  # Số CP giảm
    unchanged: int  # Số CP không đổi
    
    # Giao dịch
    total_volume: float  # Tổng KLGD
    total_value: float  # Tổng GTGD (tỷ VND)
    avg_volume: float  # KL trung bình
    
    # Định giá trung bình ngành
    avg_pe: Optional[float] = None
    avg_pb: Optional[float] = None
    avg_roe: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class MarketBreadth:
    """Độ rộng thị trường"""
    date: str
    exchange: str
    
    # Thống kê cơ bản
    advances: int
    declines: int
    unchanged: int
    
    # Cực đoan
    ceiling_hits: int  # Tăng trần
    floor_hits: int  # Giảm sàn
    
    # Tỷ lệ
    advance_decline_ratio: float
    advance_percent: float
    decline_percent: float
    
    # New High/Low
    new_highs_52w: int  # Số CP đạt đỉnh 52 tuần
    new_lows_52w: int  # Số CP đạt đáy 52 tuần
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class SupplyDemandData:
    """Dữ liệu cung cầu thị trường"""
    date: str
    exchange: str
    
    # Tổng khớp lệnh
    total_buy_volume: float
    total_sell_volume: float
    matched_volume: float
    
    # Giá trị
    total_buy_value: float
    total_sell_value: float
    matched_value: float
    
    # Dư mua/bán
    excess_buy_volume: float  # KL dư mua
    excess_sell_volume: float  # KL dư bán
    
    # Áp lực
    buy_pressure: float  # 0-100%
    sell_pressure: float  # 0-100%
    
    def to_dict(self) -> Dict:
        return asdict(self)


class IndustryDataCollector:
    """Thu thập dữ liệu thống kê ngành"""
    
    # Phân loại ngành theo ICB (Industry Classification Benchmark)
    SECTORS = {
        '0500': 'Dầu khí',
        '1000': 'Vật liệu cơ bản',
        '1300': 'Hóa chất',
        '1700': 'Tài nguyên cơ bản',
        '2000': 'Công nghiệp',
        '2300': 'Xây dựng & Vật liệu',
        '2700': 'Hàng & Dịch vụ công nghiệp',
        '3000': 'Hàng tiêu dùng',
        '3300': 'Ô tô & Phụ tùng',
        '3500': 'Thực phẩm & Đồ uống',
        '3700': 'Hàng cá nhân & Gia dụng',
        '4000': 'Y tế',
        '4500': 'Chăm sóc sức khỏe',
        '5000': 'Dịch vụ tiêu dùng',
        '5300': 'Bán lẻ',
        '5500': 'Truyền thông',
        '5700': 'Du lịch & Giải trí',
        '6000': 'Viễn thông',
        '6500': 'Tiện ích',
        '7000': 'Tài chính',
        '8300': 'Ngân hàng',
        '8500': 'Bảo hiểm',
        '8700': 'Bất động sản',
        '8770': 'Bất động sản & Quỹ',
        '8900': 'Dịch vụ tài chính',
        '9000': 'Công nghệ',
        '9500': 'Công nghệ thông tin',
    }
    
    # Sub-sectors phổ biến ở Việt Nam
    VN_SECTORS = {
        'Ngân hàng': ['ACB', 'BID', 'CTG', 'EIB', 'HDB', 'LPB', 'MBB', 'MSB', 'OCB', 
                     'SHB', 'SSB', 'STB', 'TCB', 'TPB', 'VCB', 'VIB', 'VPB'],
        'Bất động sản': ['VHM', 'VIC', 'NVL', 'KDH', 'DXG', 'NLG', 'HDG', 'DIG', 'CEO', 
                        'PDR', 'KBC', 'IJC', 'LDG', 'NBB', 'SCR'],
        'Chứng khoán': ['SSI', 'VCI', 'HCM', 'VND', 'SHS', 'MBS', 'VIX', 'AGR', 'BSI', 
                       'FTS', 'TVS', 'CTS', 'APS', 'EVS'],
        'Thép': ['HPG', 'HSG', 'NKG', 'TLH', 'POM', 'TVN', 'DTL', 'VGS', 'SMC'],
        'Bán lẻ': ['MWG', 'FRT', 'DGW', 'PNJ', 'VRE'],
        'Công nghệ': ['FPT', 'CMG', 'VGI', 'SAM', 'ONE', 'CTR', 'ELC'],
        'Dầu khí': ['GAS', 'PLX', 'PVS', 'PVD', 'BSR', 'OIL', 'PVT', 'PVB'],
        'Điện': ['POW', 'GEG', 'REE', 'PC1', 'PPC', 'NT2', 'QTP', 'HND', 'TTA'],
        'Hàng không': ['VJC', 'HVN', 'ACV', 'AST', 'NCT', 'SGN'],
        'Thực phẩm & Đồ uống': ['VNM', 'SAB', 'MSN', 'QNS', 'KDF', 'MCH', 'SBT', 'HAG'],
        'Cao su': ['GVR', 'DPR', 'PHR', 'TRC', 'BRR', 'HRC'],
        'Xây dựng': ['CTD', 'HBC', 'VCG', 'ROS', 'FCN', 'VC3', 'TV2', 'LCG'],
        'Vận tải': ['GMD', 'VOS', 'VIP', 'VTO', 'HAH', 'STG', 'TCL', 'TMS'],
        'Bảo hiểm': ['BVH', 'BMI', 'MIG', 'PTI', 'PVI', 'BIC', 'VNR'],
        'Phân bón & Hóa chất': ['DPM', 'DCM', 'DGC', 'LAS', 'CSV', 'BFC'],
    }
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
        })
        
        # API endpoints
        self.vndirect_api = "https://finfo-api.vndirect.com.vn"
        self.ssi_api = "https://iboard.ssi.com.vn"
    
    def get_sector_statistics(self, sector_code: str = None, 
                              exchange: str = 'HOSE') -> pd.DataFrame:
        """
        Lấy thống kê theo ngành
        
        Args:
            sector_code: Mã ngành ICB (None = tất cả)
            exchange: Sàn giao dịch
        
        Returns:
            DataFrame với thống kê ngành
        """
        try:
            # Build query
            query = f'floor:{exchange}'
            if sector_code:
                query += f'~industryCode:{sector_code}'
            
            url = f"{self.vndirect_api}/v4/ratios"
            params = {
                'q': query + '~reportType:QUARTER',
                'size': 500
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    df = pd.DataFrame(data['data'])
                    
                    # Group by industry
                    if 'industryCode' in df.columns:
                        industry_stats = df.groupby('industryCode').agg({
                            'code': 'count',
                            'marketCap': ['sum', 'mean'],
                            'pe': 'mean',
                            'pb': 'mean',
                            'roe': 'mean',
                        }).reset_index()
                        
                        industry_stats.columns = ['sector_code', 'stock_count', 
                                                  'total_market_cap', 'avg_market_cap',
                                                  'avg_pe', 'avg_pb', 'avg_roe']
                        
                        logger.info(f"✅ Fetched statistics for {len(industry_stats)} sectors")
                        return industry_stats
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching sector statistics: {str(e)}")
            return pd.DataFrame()
    
    def get_vn_sector_performance(self, sector_name: str, 
                                   from_date: str, to_date: str) -> pd.DataFrame:
        """
        Lấy hiệu suất ngành Việt Nam
        
        Args:
            sector_name: Tên ngành (từ VN_SECTORS)
            from_date: Ngày bắt đầu
            to_date: Ngày kết thúc
        
        Returns:
            DataFrame với hiệu suất từng CP trong ngành
        """
        try:
            if sector_name not in self.VN_SECTORS:
                logger.warning(f"Unknown sector: {sector_name}")
                return pd.DataFrame()
            
            symbols = self.VN_SECTORS[sector_name]
            results = []
            
            for symbol in symbols:
                try:
                    url = f"{self.vndirect_api}/v4/stock_prices"
                    params = {
                        'q': f'code:{symbol}~date:gte:{from_date}~date:lte:{to_date}',
                        'sort': 'date',
                        'size': 100
                    }
                    
                    response = self.session.get(url, params=params, timeout=10)
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        if 'data' in data and len(data['data']) >= 2:
                            prices = data['data']
                            start_price = float(prices[0].get('close', 0))
                            end_price = float(prices[-1].get('close', 0))
                            
                            if start_price > 0:
                                change_percent = ((end_price / start_price) - 1) * 100
                                
                                results.append({
                                    'symbol': symbol,
                                    'sector': sector_name,
                                    'start_price': start_price * 1000,
                                    'end_price': end_price * 1000,
                                    'change_percent': change_percent,
                                    'from_date': from_date,
                                    'to_date': to_date,
                                })
                except Exception as e:
                    continue
            
            df = pd.DataFrame(results)
            
            if not df.empty:
                df = df.sort_values('change_percent', ascending=False)
                logger.info(f"✅ Fetched performance for {len(df)} stocks in {sector_name}")
            
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching sector performance: {str(e)}")
            return pd.DataFrame()
    
    def get_all_sectors_performance(self, from_date: str, 
                                     to_date: str) -> pd.DataFrame:
        """
        Lấy hiệu suất tất cả các ngành
        
        Args:
            from_date: Ngày bắt đầu
            to_date: Ngày kết thúc
        
        Returns:
            DataFrame với hiệu suất các ngành
        """
        results = []
        
        for sector_name, symbols in self.VN_SECTORS.items():
            df = self.get_vn_sector_performance(sector_name, from_date, to_date)
            
            if not df.empty:
                sector_stats = {
                    'sector': sector_name,
                    'stock_count': len(df),
                    'avg_return': df['change_percent'].mean(),
                    'median_return': df['change_percent'].median(),
                    'best_performer': df.iloc[0]['symbol'] if not df.empty else None,
                    'best_return': df.iloc[0]['change_percent'] if not df.empty else None,
                    'worst_performer': df.iloc[-1]['symbol'] if not df.empty else None,
                    'worst_return': df.iloc[-1]['change_percent'] if not df.empty else None,
                    'positive_count': len(df[df['change_percent'] > 0]),
                    'negative_count': len(df[df['change_percent'] < 0]),
                }
                results.append(sector_stats)
        
        df = pd.DataFrame(results)
        df = df.sort_values('avg_return', ascending=False)
        
        logger.info(f"✅ Calculated performance for {len(df)} sectors")
        return df
    
    def get_market_breadth(self, exchange: str = 'HOSE') -> Dict:
        """
        Lấy độ rộng thị trường
        
        Args:
            exchange: Sàn giao dịch
        
        Returns:
            Dict với dữ liệu breadth
        """
        try:
            url = f"{self.vndirect_api}/v4/stocks"
            params = {
                'q': f'floor:{exchange}',
                'size': 500
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    stocks = data['data']
                    
                    advances = 0
                    declines = 0
                    unchanged = 0
                    ceiling = 0
                    floor = 0
                    
                    for stock in stocks:
                        change = float(stock.get('change', 0) or 0)
                        
                        if change > 0:
                            advances += 1
                        elif change < 0:
                            declines += 1
                        else:
                            unchanged += 1
                        
                        # Check ceiling/floor
                        close = float(stock.get('close', 0) or 0)
                        ceiling_price = float(stock.get('ceiling', 0) or 0)
                        floor_price = float(stock.get('floor', 0) or 0)
                        
                        if ceiling_price > 0 and abs(close - ceiling_price) < 0.01:
                            ceiling += 1
                        if floor_price > 0 and abs(close - floor_price) < 0.01:
                            floor += 1
                    
                    total = advances + declines + unchanged
                    
                    result = MarketBreadth(
                        date=datetime.now().strftime('%Y-%m-%d'),
                        exchange=exchange,
                        advances=advances,
                        declines=declines,
                        unchanged=unchanged,
                        ceiling_hits=ceiling,
                        floor_hits=floor,
                        advance_decline_ratio=advances / max(declines, 1),
                        advance_percent=(advances / total) * 100 if total > 0 else 0,
                        decline_percent=(declines / total) * 100 if total > 0 else 0,
                        new_highs_52w=0,  # Need separate calculation
                        new_lows_52w=0,
                    )
                    
                    logger.info(f"✅ Market breadth {exchange}: {advances} up, {declines} down")
                    return result.to_dict()
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching market breadth: {str(e)}")
            return {}
    
    def get_supply_demand(self, exchange: str = 'HOSE') -> Dict:
        """
        Lấy dữ liệu cung cầu thị trường
        
        Args:
            exchange: Sàn giao dịch
        
        Returns:
            Dict với dữ liệu cung cầu
        """
        try:
            # SSI API for supply/demand
            url = f"{self.ssi_api}/stock/group/top-stocks"
            params = {'exchange': exchange}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    stocks = data['data']
                    
                    total_buy_vol = sum(float(s.get('totalBidVol', 0) or 0) for s in stocks)
                    total_sell_vol = sum(float(s.get('totalAskVol', 0) or 0) for s in stocks)
                    matched_vol = sum(float(s.get('totalVol', 0) or 0) for s in stocks)
                    
                    total = total_buy_vol + total_sell_vol
                    
                    result = SupplyDemandData(
                        date=datetime.now().strftime('%Y-%m-%d'),
                        exchange=exchange,
                        total_buy_volume=total_buy_vol,
                        total_sell_volume=total_sell_vol,
                        matched_volume=matched_vol,
                        total_buy_value=0,  # Need price data
                        total_sell_value=0,
                        matched_value=0,
                        excess_buy_volume=max(total_buy_vol - total_sell_vol, 0),
                        excess_sell_volume=max(total_sell_vol - total_buy_vol, 0),
                        buy_pressure=(total_buy_vol / total) * 100 if total > 0 else 50,
                        sell_pressure=(total_sell_vol / total) * 100 if total > 0 else 50,
                    )
                    
                    logger.info(f"✅ Supply/Demand {exchange}: Buy pressure {result.buy_pressure:.1f}%")
                    return result.to_dict()
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching supply/demand: {str(e)}")
            return {}
    
    def get_sector_rotation_analysis(self, periods: int = 4) -> Dict:
        """
        Phân tích sector rotation (dòng tiền luân chuyển)
        
        Args:
            periods: Số kỳ phân tích (tuần)
        
        Returns:
            Dict với phân tích rotation
        """
        try:
            results = []
            
            for i in range(periods):
                # Calculate date ranges
                end_date = datetime.now() - timedelta(weeks=i)
                start_date = end_date - timedelta(weeks=1)
                
                from_str = start_date.strftime('%Y-%m-%d')
                to_str = end_date.strftime('%Y-%m-%d')
                
                # Get sector performance
                sector_perf = self.get_all_sectors_performance(from_str, to_str)
                
                if not sector_perf.empty:
                    top_sectors = sector_perf.head(3)['sector'].tolist()
                    bottom_sectors = sector_perf.tail(3)['sector'].tolist()
                    
                    results.append({
                        'period': i + 1,
                        'from_date': from_str,
                        'to_date': to_str,
                        'top_sectors': top_sectors,
                        'bottom_sectors': bottom_sectors,
                        'top_returns': sector_perf.head(3)['avg_return'].tolist(),
                        'bottom_returns': sector_perf.tail(3)['avg_return'].tolist(),
                    })
            
            # Analyze rotation pattern
            rotation_analysis = {
                'periods': results,
                'rotation_pattern': self._identify_rotation_pattern(results),
                'recommendation': self._generate_sector_recommendation(results),
            }
            
            logger.info(f"✅ Sector rotation analysis for {periods} periods")
            return rotation_analysis
        
        except Exception as e:
            logger.error(f"❌ Error in sector rotation analysis: {str(e)}")
            return {}
    
    def _identify_rotation_pattern(self, results: List[Dict]) -> str:
        """Xác định pattern rotation"""
        if len(results) < 2:
            return "Insufficient data"
        
        # Check if same sectors keep appearing on top
        all_top = []
        for r in results:
            all_top.extend(r.get('top_sectors', []))
        
        from collections import Counter
        top_counts = Counter(all_top)
        
        if top_counts.most_common(1)[0][1] >= len(results):
            return "Stable leadership - same sectors leading"
        elif len(set(all_top)) == len(all_top):
            return "High rotation - different sectors each period"
        else:
            return "Moderate rotation - some consistency"
    
    def _generate_sector_recommendation(self, results: List[Dict]) -> Dict:
        """Tạo khuyến nghị ngành"""
        if not results:
            return {}
        
        # Get most recent top sectors
        latest = results[0] if results else {}
        
        return {
            'overweight': latest.get('top_sectors', [])[:2],
            'underweight': latest.get('bottom_sectors', [])[:2],
            'note': 'Based on recent performance, consider overweighting top sectors'
        }
    
    def get_industry_comparison(self, symbols: List[str]) -> pd.DataFrame:
        """
        So sánh các công ty trong cùng ngành
        
        Args:
            symbols: Danh sách mã CP cần so sánh
        
        Returns:
            DataFrame với so sánh
        """
        try:
            results = []
            
            for symbol in symbols:
                url = f"{self.vndirect_api}/v4/ratios"
                params = {
                    'q': f'code:{symbol}~reportType:QUARTER',
                    'size': 1
                }
                
                response = self.session.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    if 'data' in data and len(data['data']) > 0:
                        ratio = data['data'][0]
                        results.append({
                            'symbol': symbol,
                            'market_cap': float(ratio.get('marketCap', 0) or 0),
                            'pe': float(ratio.get('pe', 0) or 0),
                            'pb': float(ratio.get('pb', 0) or 0),
                            'roe': float(ratio.get('roe', 0) or 0),
                            'roa': float(ratio.get('roa', 0) or 0),
                            'gross_margin': float(ratio.get('grossMargin', 0) or 0),
                            'net_margin': float(ratio.get('netMargin', 0) or 0),
                            'debt_equity': float(ratio.get('debtEquityRatio', 0) or 0),
                            'revenue_growth': float(ratio.get('revenueGrowth', 0) or 0),
                        })
            
            df = pd.DataFrame(results)
            
            if not df.empty:
                # Add ranking
                for col in ['roe', 'roa', 'gross_margin', 'net_margin', 'revenue_growth']:
                    if col in df.columns:
                        df[f'{col}_rank'] = df[col].rank(ascending=False)
                
                for col in ['pe', 'pb', 'debt_equity']:
                    if col in df.columns:
                        df[f'{col}_rank'] = df[col].rank(ascending=True)
            
            logger.info(f"✅ Industry comparison for {len(df)} stocks")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error in industry comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_stocks_by_sector(self, sector_name: str) -> List[str]:
        """Lấy danh sách CP theo ngành"""
        return self.VN_SECTORS.get(sector_name, [])
    
    def get_all_sector_names(self) -> List[str]:
        """Lấy danh sách tên các ngành"""
        return list(self.VN_SECTORS.keys())


if __name__ == "__main__":
    # Test
    collector = IndustryDataCollector()
    
    # Test sector statistics
    print("\n=== Sector Statistics ===")
    df = collector.get_sector_statistics(exchange='HOSE')
    print(df.head())
    
    # Test sector performance
    print("\n=== Banking Sector Performance ===")
    df = collector.get_vn_sector_performance('Ngân hàng', '2024-11-01', '2024-11-30')
    print(df)
    
    # Test all sectors performance
    print("\n=== All Sectors Performance ===")
    df = collector.get_all_sectors_performance('2024-11-01', '2024-11-30')
    print(df)
    
    # Test market breadth
    print("\n=== Market Breadth ===")
    data = collector.get_market_breadth('HOSE')
    print(json.dumps(data, indent=2, default=str))
    
    # Test industry comparison
    print("\n=== Banking Comparison ===")
    banks = ['VCB', 'BID', 'CTG', 'TCB', 'MBB']
    df = collector.get_industry_comparison(banks)
    print(df[['symbol', 'market_cap', 'pe', 'pb', 'roe']].to_string())
