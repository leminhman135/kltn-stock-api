"""
Extended Database Models for KLTN Stock Prediction System

Thêm các bảng mới để lưu trữ:
- Dữ liệu giao dịch NDTNN và tự doanh
- Dữ liệu thị trường và chỉ số
- Dữ liệu tài chính và định giá
- Thống kê ngành
"""

from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, 
    Text, Boolean, JSON, ForeignKey, Index, UniqueConstraint
)
from sqlalchemy.orm import relationship
from .connection import Base


# ========================
# TRADING DATA MODELS
# ========================

class ForeignTrading(Base):
    """Dữ liệu giao dịch nhà đầu tư nước ngoài"""
    __tablename__ = "foreign_trading"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Khối lượng
    buy_volume = Column(Float, default=0)  # KL mua
    sell_volume = Column(Float, default=0)  # KL bán
    net_volume = Column(Float, default=0)  # KL mua ròng
    
    # Giá trị (VND)
    buy_value = Column(Float, default=0)  # GT mua
    sell_value = Column(Float, default=0)  # GT bán
    net_value = Column(Float, default=0)  # GT mua ròng
    
    # Room nước ngoài
    foreign_room = Column(Float)  # Room còn lại (số CP)
    ownership_percent = Column(Float)  # % sở hữu hiện tại
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_foreign_trading_symbol_date'),
        Index('ix_foreign_trading_symbol_date', 'symbol', 'date'),
    )


class ProprietaryTrading(Base):
    """Dữ liệu giao dịch tự doanh"""
    __tablename__ = "proprietary_trading"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Khối lượng
    buy_volume = Column(Float, default=0)
    sell_volume = Column(Float, default=0)
    net_volume = Column(Float, default=0)
    
    # Giá trị
    buy_value = Column(Float, default=0)
    sell_value = Column(Float, default=0)
    net_value = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_prop_trading_symbol_date'),
        Index('ix_prop_trading_symbol_date', 'symbol', 'date'),
    )


class DetailedTrading(Base):
    """Dữ liệu giao dịch chi tiết"""
    __tablename__ = "detailed_trading"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Giá OHLC
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    
    # Khối lượng chi tiết
    volume = Column(Float, default=0)  # Tổng KL
    deal_volume = Column(Float, default=0)  # KL khớp lệnh
    put_through_volume = Column(Float, default=0)  # KL thỏa thuận
    
    # Giá trị chi tiết
    total_value = Column(Float, default=0)
    deal_value = Column(Float, default=0)
    put_through_value = Column(Float, default=0)
    
    # Thay đổi giá
    price_change = Column(Float, default=0)
    price_change_percent = Column(Float, default=0)
    
    # Giá tham chiếu
    reference_price = Column(Float)
    ceiling_price = Column(Float)
    floor_price = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_detailed_trading_symbol_date'),
        Index('ix_detailed_trading_symbol_date', 'symbol', 'date'),
    )


# ========================
# MARKET DATA MODELS
# ========================

class MarketIndex(Base):
    """Dữ liệu chỉ số thị trường"""
    __tablename__ = "market_indices"
    
    id = Column(Integer, primary_key=True, index=True)
    index_code = Column(String(20), nullable=False, index=True)  # VNINDEX, VN30, etc.
    date = Column(Date, nullable=False, index=True)
    
    # Giá trị OHLC
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    
    # Thay đổi
    change = Column(Float, default=0)
    change_percent = Column(Float, default=0)
    
    # Khối lượng và giá trị
    volume = Column(Float, default=0)
    value = Column(Float, default=0)  # Tỷ VND
    
    # Thống kê
    advances = Column(Integer, default=0)
    declines = Column(Integer, default=0)
    unchanged = Column(Integer, default=0)
    ceiling_count = Column(Integer, default=0)
    floor_count = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('index_code', 'date', name='uq_market_index_code_date'),
        Index('ix_market_index_code_date', 'index_code', 'date'),
    )


class StockOwnership(Base):
    """Dữ liệu sở hữu cổ phiếu"""
    __tablename__ = "stock_ownership"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Sở hữu nước ngoài
    foreign_percent = Column(Float, default=0)  # % sở hữu NDTNN
    foreign_limit = Column(Float, default=49)  # Room tối đa (%)
    foreign_room = Column(Float, default=0)  # Room còn lại (số CP)
    foreign_room_percent = Column(Float, default=0)  # Room còn lại (%)
    
    # Freefloat
    freefloat_ratio = Column(Float, default=0)  # Tỷ lệ freefloat (%)
    freefloat_shares = Column(Float, default=0)  # Số CP freefloat
    
    # Tổng số CP
    total_shares = Column(Float, default=0)
    listed_shares = Column(Float, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_stock_ownership_symbol_date'),
        Index('ix_stock_ownership_symbol_date', 'symbol', 'date'),
    )


class IndexComponent(Base):
    """Thành phần chỉ số"""
    __tablename__ = "index_components"
    
    id = Column(Integer, primary_key=True, index=True)
    index_code = Column(String(20), nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    
    weight = Column(Float, default=0)  # Tỷ trọng (%)
    market_cap = Column(Float, default=0)  # Vốn hóa
    freefloat_cap = Column(Float, default=0)  # Vốn hóa freefloat
    shares_in_index = Column(Float, default=0)  # Số CP trong rổ
    
    effective_date = Column(Date)  # Ngày có hiệu lực
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('index_code', 'symbol', name='uq_index_component'),
        Index('ix_index_components', 'index_code', 'symbol'),
    )


# ========================
# FINANCIAL DATA MODELS
# ========================

class Valuation(Base):
    """Dữ liệu định giá cổ phiếu"""
    __tablename__ = "valuations"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Định giá cơ bản
    pe_ratio = Column(Float)  # P/E
    pb_ratio = Column(Float)  # P/B
    ps_ratio = Column(Float)  # P/S
    
    # EPS
    eps_ttm = Column(Float)  # EPS 4 quý gần nhất
    eps_growth = Column(Float)  # Tăng trưởng EPS (%)
    
    # Book Value
    book_value = Column(Float)  # BVPS
    
    # Enterprise Value
    ev = Column(Float)  # Enterprise Value
    ev_ebitda = Column(Float)  # EV/EBITDA
    ev_sales = Column(Float)  # EV/Sales
    
    # Market Cap
    market_cap = Column(Float)  # Vốn hóa
    
    # Dividend
    dividend_yield = Column(Float)  # Tỷ suất cổ tức (%)
    payout_ratio = Column(Float)  # Tỷ lệ chi trả (%)
    
    # PEG
    peg_ratio = Column(Float)  # P/E / Growth
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'date', name='uq_valuation_symbol_date'),
        Index('ix_valuation_symbol_date', 'symbol', 'date'),
    )


class FinancialStatement(Base):
    """Báo cáo tài chính"""
    __tablename__ = "financial_statements"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    period = Column(String(20), nullable=False)  # Q1/2024, FY2024
    period_type = Column(String(10), nullable=False)  # quarter, year
    report_type = Column(String(30), nullable=False)  # income, balance, cashflow
    
    # JSON data để lưu linh hoạt
    data = Column(JSON)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'period', 'report_type', name='uq_financial_statement'),
        Index('ix_financial_statement', 'symbol', 'period', 'report_type'),
    )


class FinancialRatio(Base):
    """Chỉ số tài chính"""
    __tablename__ = "financial_ratios"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    period = Column(String(20), nullable=False)
    period_type = Column(String(10), nullable=False)
    
    # Kết quả kinh doanh
    revenue = Column(Float)
    gross_profit = Column(Float)
    operating_profit = Column(Float)
    net_income = Column(Float)
    
    # Tăng trưởng
    revenue_growth = Column(Float)
    net_income_growth = Column(Float)
    
    # Biên lợi nhuận
    gross_margin = Column(Float)
    operating_margin = Column(Float)
    net_margin = Column(Float)
    
    # Chỉ số sinh lời
    roe = Column(Float)
    roa = Column(Float)
    roic = Column(Float)
    
    # Đòn bẩy
    debt_to_equity = Column(Float)
    debt_to_assets = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    
    # Hiệu suất
    asset_turnover = Column(Float)
    inventory_turnover = Column(Float)
    receivables_turnover = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('symbol', 'period', name='uq_financial_ratio_symbol_period'),
        Index('ix_financial_ratio_symbol_period', 'symbol', 'period'),
    )


# ========================
# INDUSTRY DATA MODELS
# ========================

class SectorStatistics(Base):
    """Thống kê ngành"""
    __tablename__ = "sector_statistics"
    
    id = Column(Integer, primary_key=True, index=True)
    sector_code = Column(String(20), nullable=False, index=True)
    sector_name = Column(String(100))
    date = Column(Date, nullable=False, index=True)
    
    # Thống kê cơ bản
    stock_count = Column(Integer, default=0)
    total_market_cap = Column(Float, default=0)  # Tỷ VND
    avg_market_cap = Column(Float, default=0)
    
    # Hiệu suất
    avg_change_percent = Column(Float, default=0)
    advances = Column(Integer, default=0)
    declines = Column(Integer, default=0)
    unchanged = Column(Integer, default=0)
    
    # Giao dịch
    total_volume = Column(Float, default=0)
    total_value = Column(Float, default=0)  # Tỷ VND
    avg_volume = Column(Float, default=0)
    
    # Định giá TB ngành
    avg_pe = Column(Float)
    avg_pb = Column(Float)
    avg_roe = Column(Float)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('sector_code', 'date', name='uq_sector_stats_code_date'),
        Index('ix_sector_stats_code_date', 'sector_code', 'date'),
    )


class MarketBreadth(Base):
    """Độ rộng thị trường"""
    __tablename__ = "market_breadth"
    
    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String(10), nullable=False, index=True)  # HOSE, HNX, UPCOM
    date = Column(Date, nullable=False, index=True)
    
    # Thống kê
    advances = Column(Integer, default=0)
    declines = Column(Integer, default=0)
    unchanged = Column(Integer, default=0)
    
    # Cực đoan
    ceiling_hits = Column(Integer, default=0)
    floor_hits = Column(Integer, default=0)
    
    # Tỷ lệ
    advance_decline_ratio = Column(Float, default=0)
    advance_percent = Column(Float, default=0)
    decline_percent = Column(Float, default=0)
    
    # New High/Low
    new_highs_52w = Column(Integer, default=0)
    new_lows_52w = Column(Integer, default=0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('exchange', 'date', name='uq_market_breadth_exchange_date'),
        Index('ix_market_breadth_exchange_date', 'exchange', 'date'),
    )


class SupplyDemand(Base):
    """Cung cầu thị trường"""
    __tablename__ = "supply_demand"
    
    id = Column(Integer, primary_key=True, index=True)
    exchange = Column(String(10), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    
    # Khối lượng
    total_buy_volume = Column(Float, default=0)
    total_sell_volume = Column(Float, default=0)
    matched_volume = Column(Float, default=0)
    
    # Giá trị
    total_buy_value = Column(Float, default=0)
    total_sell_value = Column(Float, default=0)
    matched_value = Column(Float, default=0)
    
    # Dư mua/bán
    excess_buy_volume = Column(Float, default=0)
    excess_sell_volume = Column(Float, default=0)
    
    # Áp lực
    buy_pressure = Column(Float, default=50)  # 0-100%
    sell_pressure = Column(Float, default=50)  # 0-100%
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        UniqueConstraint('exchange', 'date', name='uq_supply_demand_exchange_date'),
        Index('ix_supply_demand_exchange_date', 'exchange', 'date'),
    )


# Export all models
__all__ = [
    'ForeignTrading',
    'ProprietaryTrading',
    'DetailedTrading',
    'MarketIndex',
    'StockOwnership',
    'IndexComponent',
    'Valuation',
    'FinancialStatement',
    'FinancialRatio',
    'SectorStatistics',
    'MarketBreadth',
    'SupplyDemand',
]
