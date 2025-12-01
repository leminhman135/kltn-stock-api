"""
Financial Data Collection Module

Thu thập dữ liệu tài chính và định giá:
- Dữ liệu tài chính doanh nghiệp (BCTC, chỉ số tài chính)
- Dữ liệu định giá (P/E, P/B, EV/EBITDA, DCF inputs)
- Chỉ số sinh lời, đòn bẩy, thanh khoản
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import logging
import json

logger = logging.getLogger(__name__)


@dataclass
class ValuationData:
    """Dữ liệu định giá cổ phiếu"""
    symbol: str
    date: str
    
    # Định giá cơ bản
    pe_ratio: Optional[float] = None  # P/E - Giá/EPS
    pb_ratio: Optional[float] = None  # P/B - Giá/Book Value
    ps_ratio: Optional[float] = None  # P/S - Giá/Doanh thu
    
    # EPS
    eps_ttm: Optional[float] = None  # EPS 4 quý gần nhất
    eps_growth: Optional[float] = None  # Tăng trưởng EPS (%)
    
    # Book Value
    book_value: Optional[float] = None  # Giá trị sổ sách/CP
    bvps_growth: Optional[float] = None  # Tăng trưởng BVPS (%)
    
    # Enterprise Value metrics
    ev: Optional[float] = None  # Enterprise Value
    ev_ebitda: Optional[float] = None  # EV/EBITDA
    ev_sales: Optional[float] = None  # EV/Sales
    
    # Market Cap
    market_cap: Optional[float] = None  # Vốn hóa thị trường
    
    # Dividend
    dividend_yield: Optional[float] = None  # Tỷ suất cổ tức (%)
    payout_ratio: Optional[float] = None  # Tỷ lệ chi trả cổ tức (%)
    
    # PEG
    peg_ratio: Optional[float] = None  # P/E / Growth rate
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FundamentalData:
    """Dữ liệu tài chính cơ bản"""
    symbol: str
    period: str  # Q1/2024, FY2024
    period_type: str  # quarter, year
    
    # Kết quả kinh doanh
    revenue: Optional[float] = None  # Doanh thu thuần
    gross_profit: Optional[float] = None  # Lợi nhuận gộp
    operating_profit: Optional[float] = None  # Lợi nhuận từ HĐKD
    net_income: Optional[float] = None  # Lợi nhuận sau thuế
    
    # Tăng trưởng (YoY)
    revenue_growth: Optional[float] = None
    gross_profit_growth: Optional[float] = None
    operating_profit_growth: Optional[float] = None
    net_income_growth: Optional[float] = None
    
    # Biên lợi nhuận
    gross_margin: Optional[float] = None  # Biên lợi nhuận gộp
    operating_margin: Optional[float] = None  # Biên lợi nhuận HĐKD
    net_margin: Optional[float] = None  # Biên lợi nhuận ròng
    
    # Chỉ số sinh lời
    roe: Optional[float] = None  # Return on Equity
    roa: Optional[float] = None  # Return on Assets
    roic: Optional[float] = None  # Return on Invested Capital
    
    # Đòn bẩy tài chính
    debt_to_equity: Optional[float] = None  # Nợ/Vốn CSH
    debt_to_assets: Optional[float] = None  # Nợ/Tổng tài sản
    current_ratio: Optional[float] = None  # Tỷ số thanh toán hiện hành
    quick_ratio: Optional[float] = None  # Tỷ số thanh toán nhanh
    
    # Hiệu suất hoạt động
    asset_turnover: Optional[float] = None  # Vòng quay tài sản
    inventory_turnover: Optional[float] = None  # Vòng quay hàng tồn kho
    receivables_turnover: Optional[float] = None  # Vòng quay khoản phải thu
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BalanceSheetData:
    """Dữ liệu Bảng cân đối kế toán"""
    symbol: str
    period: str
    
    # Tài sản
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    non_current_assets: Optional[float] = None
    cash: Optional[float] = None
    inventory: Optional[float] = None
    receivables: Optional[float] = None
    fixed_assets: Optional[float] = None
    
    # Nợ phải trả
    total_liabilities: Optional[float] = None
    current_liabilities: Optional[float] = None
    non_current_liabilities: Optional[float] = None
    short_term_debt: Optional[float] = None
    long_term_debt: Optional[float] = None
    
    # Vốn chủ sở hữu
    total_equity: Optional[float] = None
    paid_in_capital: Optional[float] = None
    retained_earnings: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class CashFlowData:
    """Dữ liệu Lưu chuyển tiền tệ"""
    symbol: str
    period: str
    
    # Dòng tiền từ HĐKD
    operating_cash_flow: Optional[float] = None
    
    # Dòng tiền từ HĐĐT
    investing_cash_flow: Optional[float] = None
    capex: Optional[float] = None  # Chi đầu tư TSCĐ
    
    # Dòng tiền từ HĐTC
    financing_cash_flow: Optional[float] = None
    dividend_paid: Optional[float] = None
    debt_repayment: Optional[float] = None
    
    # Dòng tiền tự do
    free_cash_flow: Optional[float] = None  # FCF = OCF - CapEx
    
    def to_dict(self) -> Dict:
        return asdict(self)


class FinancialDataCollector:
    """Thu thập dữ liệu tài chính và định giá"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
        })
        
        # API endpoints
        self.vndirect_api = "https://finfo-api.vndirect.com.vn"
        self.cafef_api = "https://s.cafef.vn"
        self.vietstock_api = "https://api.vietstock.vn"
    
    def get_valuation_data(self, symbol: str) -> Dict:
        """
        Lấy dữ liệu định giá cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict với dữ liệu định giá
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # VNDirect ratios API
            url = f"{self.vndirect_api}/v4/ratios"
            params = {'q': f'code:{clean_symbol}~reportType:QUARTER', 'size': 1}
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    ratio = data['data'][0]
                    
                    # Lấy thêm thông tin giá
                    stock_info = self._get_stock_info(clean_symbol)
                    market_cap = stock_info.get('marketCap', 0)
                    price = stock_info.get('lastPrice', 0)
                    
                    result = ValuationData(
                        symbol=clean_symbol,
                        date=datetime.now().strftime('%Y-%m-%d'),
                        pe_ratio=float(ratio.get('pe', 0)) if ratio.get('pe') else None,
                        pb_ratio=float(ratio.get('pb', 0)) if ratio.get('pb') else None,
                        ps_ratio=float(ratio.get('ps', 0)) if ratio.get('ps') else None,
                        eps_ttm=float(ratio.get('epsLastYear', 0)) if ratio.get('epsLastYear') else None,
                        eps_growth=float(ratio.get('epsGrowth', 0)) if ratio.get('epsGrowth') else None,
                        book_value=float(ratio.get('bookValue', 0)) if ratio.get('bookValue') else None,
                        ev=float(ratio.get('ev', 0)) if ratio.get('ev') else None,
                        ev_ebitda=float(ratio.get('evEbitda', 0)) if ratio.get('evEbitda') else None,
                        market_cap=market_cap,
                        dividend_yield=float(ratio.get('dividendYield', 0)) if ratio.get('dividendYield') else None,
                    )
                    
                    # Tính PEG
                    if result.pe_ratio and result.eps_growth and result.eps_growth > 0:
                        result.peg_ratio = result.pe_ratio / result.eps_growth
                    
                    logger.info(f"✅ Valuation {symbol}: P/E={result.pe_ratio}, P/B={result.pb_ratio}")
                    return result.to_dict()
            
            return {}
        
        except Exception as e:
            logger.error(f"❌ Error fetching valuation data: {str(e)}")
            return {}
    
    def _get_stock_info(self, symbol: str) -> Dict:
        """Lấy thông tin giá cổ phiếu"""
        try:
            url = f"{self.vndirect_api}/v4/stocks"
            params = {'q': f'code:{symbol}'}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and len(data['data']) > 0:
                    return data['data'][0]
            return {}
        except:
            return {}
    
    def get_fundamental_data(self, symbol: str, period_type: str = 'quarter', 
                             periods: int = 8) -> pd.DataFrame:
        """
        Lấy dữ liệu tài chính cơ bản
        
        Args:
            symbol: Mã cổ phiếu
            period_type: 'quarter' hoặc 'year'
            periods: Số kỳ cần lấy
        
        Returns:
            DataFrame với dữ liệu tài chính
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            report_type = 'QUARTER' if period_type == 'quarter' else 'YEAR'
            
            url = f"{self.vndirect_api}/v4/ratios"
            params = {
                'q': f'code:{clean_symbol}~reportType:{report_type}',
                'sort': 'reportDate:DESC',
                'size': periods
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        record = FundamentalData(
                            symbol=clean_symbol,
                            period=item.get('reportDate', ''),
                            period_type=period_type,
                            revenue=float(item.get('revenue', 0)) if item.get('revenue') else None,
                            gross_profit=float(item.get('grossProfit', 0)) if item.get('grossProfit') else None,
                            operating_profit=float(item.get('operatingProfit', 0)) if item.get('operatingProfit') else None,
                            net_income=float(item.get('netProfit', 0)) if item.get('netProfit') else None,
                            revenue_growth=float(item.get('revenueGrowth', 0)) if item.get('revenueGrowth') else None,
                            net_income_growth=float(item.get('netProfitGrowth', 0)) if item.get('netProfitGrowth') else None,
                            gross_margin=float(item.get('grossMargin', 0)) if item.get('grossMargin') else None,
                            operating_margin=float(item.get('operatingMargin', 0)) if item.get('operatingMargin') else None,
                            net_margin=float(item.get('netMargin', 0)) if item.get('netMargin') else None,
                            roe=float(item.get('roe', 0)) if item.get('roe') else None,
                            roa=float(item.get('roa', 0)) if item.get('roa') else None,
                            roic=float(item.get('roic', 0)) if item.get('roic') else None,
                            debt_to_equity=float(item.get('debtEquityRatio', 0)) if item.get('debtEquityRatio') else None,
                            current_ratio=float(item.get('currentRatio', 0)) if item.get('currentRatio') else None,
                            quick_ratio=float(item.get('quickRatio', 0)) if item.get('quickRatio') else None,
                            asset_turnover=float(item.get('assetTurnover', 0)) if item.get('assetTurnover') else None,
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} {period_type}ly fundamentals for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching fundamental data: {str(e)}")
            return pd.DataFrame()
    
    def get_balance_sheet(self, symbol: str, periods: int = 8) -> pd.DataFrame:
        """
        Lấy bảng cân đối kế toán
        
        Args:
            symbol: Mã cổ phiếu
            periods: Số kỳ cần lấy
        
        Returns:
            DataFrame với dữ liệu BCĐKT
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            url = f"{self.vndirect_api}/v4/financial_statements"
            params = {
                'q': f'code:{clean_symbol}~reportType:BALANCE_SHEET',
                'sort': 'reportDate:DESC',
                'size': periods
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        record = BalanceSheetData(
                            symbol=clean_symbol,
                            period=item.get('reportDate', ''),
                            total_assets=float(item.get('totalAssets', 0)) if item.get('totalAssets') else None,
                            current_assets=float(item.get('currentAssets', 0)) if item.get('currentAssets') else None,
                            non_current_assets=float(item.get('nonCurrentAssets', 0)) if item.get('nonCurrentAssets') else None,
                            cash=float(item.get('cash', 0)) if item.get('cash') else None,
                            inventory=float(item.get('inventory', 0)) if item.get('inventory') else None,
                            receivables=float(item.get('receivables', 0)) if item.get('receivables') else None,
                            fixed_assets=float(item.get('fixedAssets', 0)) if item.get('fixedAssets') else None,
                            total_liabilities=float(item.get('totalLiabilities', 0)) if item.get('totalLiabilities') else None,
                            current_liabilities=float(item.get('currentLiabilities', 0)) if item.get('currentLiabilities') else None,
                            non_current_liabilities=float(item.get('nonCurrentLiabilities', 0)) if item.get('nonCurrentLiabilities') else None,
                            short_term_debt=float(item.get('shortTermDebt', 0)) if item.get('shortTermDebt') else None,
                            long_term_debt=float(item.get('longTermDebt', 0)) if item.get('longTermDebt') else None,
                            total_equity=float(item.get('equity', 0)) if item.get('equity') else None,
                            paid_in_capital=float(item.get('charterCapital', 0)) if item.get('charterCapital') else None,
                            retained_earnings=float(item.get('retainedEarnings', 0)) if item.get('retainedEarnings') else None,
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} balance sheet records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching balance sheet: {str(e)}")
            return pd.DataFrame()
    
    def get_income_statement(self, symbol: str, periods: int = 8) -> pd.DataFrame:
        """
        Lấy báo cáo kết quả kinh doanh
        
        Args:
            symbol: Mã cổ phiếu
            periods: Số kỳ cần lấy
        
        Returns:
            DataFrame với BCKQKD
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            url = f"{self.vndirect_api}/v4/financial_statements"
            params = {
                'q': f'code:{clean_symbol}~reportType:INCOME_STATEMENT',
                'sort': 'reportDate:DESC',
                'size': periods
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Fetched {len(df)} income statement records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching income statement: {str(e)}")
            return pd.DataFrame()
    
    def get_cash_flow(self, symbol: str, periods: int = 8) -> pd.DataFrame:
        """
        Lấy báo cáo lưu chuyển tiền tệ
        
        Args:
            symbol: Mã cổ phiếu
            periods: Số kỳ cần lấy
        
        Returns:
            DataFrame với BCLCTT
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            url = f"{self.vndirect_api}/v4/financial_statements"
            params = {
                'q': f'code:{clean_symbol}~reportType:CASH_FLOW',
                'sort': 'reportDate:DESC',
                'size': periods
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    records = []
                    for item in data['data']:
                        ocf = float(item.get('operatingCashFlow', 0)) if item.get('operatingCashFlow') else 0
                        capex = float(item.get('capex', 0)) if item.get('capex') else 0
                        
                        record = CashFlowData(
                            symbol=clean_symbol,
                            period=item.get('reportDate', ''),
                            operating_cash_flow=ocf,
                            investing_cash_flow=float(item.get('investingCashFlow', 0)) if item.get('investingCashFlow') else None,
                            capex=capex,
                            financing_cash_flow=float(item.get('financingCashFlow', 0)) if item.get('financingCashFlow') else None,
                            dividend_paid=float(item.get('dividendPaid', 0)) if item.get('dividendPaid') else None,
                            free_cash_flow=ocf - abs(capex) if ocf and capex else None,
                        )
                        records.append(record.to_dict())
                    
                    df = pd.DataFrame(records)
                    logger.info(f"✅ Fetched {len(df)} cash flow records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching cash flow: {str(e)}")
            return pd.DataFrame()
    
    def get_dividend_history(self, symbol: str) -> pd.DataFrame:
        """
        Lấy lịch sử cổ tức
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame với lịch sử cổ tức
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            url = f"{self.vndirect_api}/v4/dividends"
            params = {
                'q': f'code:{clean_symbol}',
                'sort': 'exDate:DESC',
                'size': 50
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Fetched {len(df)} dividend records for {symbol}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching dividend history: {str(e)}")
            return pd.DataFrame()
    
    def get_peer_comparison(self, symbol: str) -> pd.DataFrame:
        """
        So sánh với các công ty cùng ngành
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame với so sánh các công ty
        """
        try:
            clean_symbol = symbol.replace('.VN', '').upper()
            
            # Lấy thông tin ngành
            stock_info = self._get_stock_info(clean_symbol)
            industry = stock_info.get('industryCode', '')
            
            if not industry:
                logger.warning(f"No industry info for {symbol}")
                return pd.DataFrame()
            
            # Lấy các công ty cùng ngành
            url = f"{self.vndirect_api}/v4/ratios"
            params = {
                'q': f'industryCode:{industry}~reportType:QUARTER',
                'sort': 'marketCap:DESC',
                'size': 20
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    
                    # Chọn các cột quan trọng
                    key_columns = ['code', 'marketCap', 'pe', 'pb', 'roe', 'roa', 
                                   'grossMargin', 'netMargin', 'debtEquityRatio']
                    
                    available_cols = [c for c in key_columns if c in df.columns]
                    df = df[available_cols]
                    
                    logger.info(f"✅ Peer comparison: {len(df)} companies in {industry}")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error fetching peer comparison: {str(e)}")
            return pd.DataFrame()
    
    def get_financial_summary(self, symbol: str) -> Dict:
        """
        Lấy tổng hợp dữ liệu tài chính
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict với tổng hợp tài chính
        """
        try:
            valuation = self.get_valuation_data(symbol)
            fundamentals = self.get_fundamental_data(symbol, 'quarter', 4)
            balance_sheet = self.get_balance_sheet(symbol, 4)
            cash_flow = self.get_cash_flow(symbol, 4)
            
            latest_fundamental = fundamentals.iloc[0].to_dict() if not fundamentals.empty else {}
            latest_balance = balance_sheet.iloc[0].to_dict() if not balance_sheet.empty else {}
            latest_cashflow = cash_flow.iloc[0].to_dict() if not cash_flow.empty else {}
            
            summary = {
                'symbol': symbol.replace('.VN', '').upper(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                
                # Định giá
                'valuation': valuation,
                
                # Chỉ số tài chính gần nhất
                'latest_fundamentals': latest_fundamental,
                
                # Bảng cân đối gần nhất
                'latest_balance_sheet': latest_balance,
                
                # Dòng tiền gần nhất
                'latest_cash_flow': latest_cashflow,
                
                # Tăng trưởng (so với cùng kỳ năm trước)
                'growth': {
                    'revenue_growth': latest_fundamental.get('revenue_growth'),
                    'net_income_growth': latest_fundamental.get('net_income_growth'),
                    'eps_growth': valuation.get('eps_growth'),
                },
                
                # Chất lượng tài chính
                'quality_metrics': {
                    'roe': latest_fundamental.get('roe'),
                    'roa': latest_fundamental.get('roa'),
                    'roic': latest_fundamental.get('roic'),
                    'gross_margin': latest_fundamental.get('gross_margin'),
                    'net_margin': latest_fundamental.get('net_margin'),
                    'current_ratio': latest_fundamental.get('current_ratio'),
                    'debt_to_equity': latest_fundamental.get('debt_to_equity'),
                    'free_cash_flow': latest_cashflow.get('free_cash_flow'),
                },
            }
            
            logger.info(f"✅ Generated financial summary for {symbol}")
            return summary
        
        except Exception as e:
            logger.error(f"❌ Error generating financial summary: {str(e)}")
            return {}
    
    def screen_stocks(self, criteria: Dict) -> pd.DataFrame:
        """
        Lọc cổ phiếu theo tiêu chí tài chính
        
        Args:
            criteria: Dict với các tiêu chí lọc
                - pe_max: P/E tối đa
                - pb_max: P/B tối đa
                - roe_min: ROE tối thiểu
                - market_cap_min: Vốn hóa tối thiểu
                - exchange: Sàn giao dịch
        
        Returns:
            DataFrame với danh sách CP thỏa mãn
        """
        try:
            # Build query
            query_parts = []
            
            if 'exchange' in criteria:
                query_parts.append(f"floor:{criteria['exchange']}")
            
            if 'pe_max' in criteria:
                query_parts.append(f"pe:lte:{criteria['pe_max']}")
            
            if 'pb_max' in criteria:
                query_parts.append(f"pb:lte:{criteria['pb_max']}")
            
            if 'roe_min' in criteria:
                query_parts.append(f"roe:gte:{criteria['roe_min']}")
            
            if 'market_cap_min' in criteria:
                query_parts.append(f"marketCap:gte:{criteria['market_cap_min']}")
            
            query = '~'.join(query_parts) if query_parts else ''
            
            url = f"{self.vndirect_api}/v4/ratios"
            params = {
                'q': query + '~reportType:QUARTER' if query else 'reportType:QUARTER',
                'sort': 'marketCap:DESC',
                'size': 100
            }
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Stock screening: Found {len(df)} stocks matching criteria")
                    return df
            
            return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"❌ Error screening stocks: {str(e)}")
            return pd.DataFrame()


# DCF Valuation Helper
class DCFValuation:
    """Định giá DCF (Discounted Cash Flow)"""
    
    @staticmethod
    def calculate_intrinsic_value(
        fcf: float,  # Free Cash Flow hiện tại
        growth_rate: float,  # Tỷ lệ tăng trưởng FCF (%)
        discount_rate: float,  # Tỷ lệ chiết khấu/WACC (%)
        terminal_growth: float = 3,  # Tỷ lệ tăng trưởng vĩnh viễn (%)
        years: int = 10,  # Số năm dự báo
        shares_outstanding: float = 1  # Số cổ phiếu
    ) -> Dict:
        """
        Tính giá trị nội tại theo DCF
        
        Returns:
            Dict với kết quả DCF
        """
        # Convert to decimals
        g = growth_rate / 100
        r = discount_rate / 100
        tg = terminal_growth / 100
        
        # Project FCF
        projected_fcf = []
        current_fcf = fcf
        
        for year in range(1, years + 1):
            current_fcf = current_fcf * (1 + g)
            discount_factor = 1 / (1 + r) ** year
            pv = current_fcf * discount_factor
            
            projected_fcf.append({
                'year': year,
                'fcf': current_fcf,
                'discount_factor': discount_factor,
                'present_value': pv
            })
        
        # Sum of projected FCF PV
        sum_pv = sum(p['present_value'] for p in projected_fcf)
        
        # Terminal Value
        terminal_fcf = projected_fcf[-1]['fcf'] * (1 + tg)
        terminal_value = terminal_fcf / (r - tg)
        terminal_pv = terminal_value / (1 + r) ** years
        
        # Enterprise Value
        enterprise_value = sum_pv + terminal_pv
        
        # Equity Value (assume no debt adjustment for simplicity)
        equity_value = enterprise_value
        
        # Per share value
        intrinsic_value_per_share = equity_value / shares_outstanding
        
        return {
            'fcf_projections': projected_fcf,
            'sum_fcf_pv': sum_pv,
            'terminal_value': terminal_value,
            'terminal_pv': terminal_pv,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'intrinsic_value_per_share': intrinsic_value_per_share,
            
            # Inputs used
            'inputs': {
                'fcf': fcf,
                'growth_rate': growth_rate,
                'discount_rate': discount_rate,
                'terminal_growth': terminal_growth,
                'years': years,
                'shares_outstanding': shares_outstanding
            }
        }


if __name__ == "__main__":
    # Test
    collector = FinancialDataCollector()
    
    symbol = "VNM"
    
    # Test valuation
    print("\n=== Valuation Data ===")
    data = collector.get_valuation_data(symbol)
    print(json.dumps(data, indent=2, default=str))
    
    # Test fundamentals
    print("\n=== Fundamental Data ===")
    df = collector.get_fundamental_data(symbol, 'quarter', 4)
    print(df[['period', 'revenue', 'net_income', 'roe', 'roa']].head())
    
    # Test balance sheet
    print("\n=== Balance Sheet ===")
    df = collector.get_balance_sheet(symbol, 4)
    print(df[['period', 'total_assets', 'total_liabilities', 'total_equity']].head())
    
    # Test cash flow
    print("\n=== Cash Flow ===")
    df = collector.get_cash_flow(symbol, 4)
    print(df[['period', 'operating_cash_flow', 'free_cash_flow']].head())
    
    # Test DCF
    print("\n=== DCF Valuation ===")
    dcf_result = DCFValuation.calculate_intrinsic_value(
        fcf=5000,  # 5000 tỷ VND
        growth_rate=10,  # 10% growth
        discount_rate=12,  # 12% WACC
        terminal_growth=3,  # 3% perpetuity
        years=10,
        shares_outstanding=2000  # 2000 triệu CP
    )
    print(f"Intrinsic Value: {dcf_result['intrinsic_value_per_share']:,.0f} VND/share")
