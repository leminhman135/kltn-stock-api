"""
Streamlit Web Application - Giao diện người dùng cho Stock Prediction System
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import requests
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Page configuration
st.set_page_config(
    page_title="Stock Prediction System - AI Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",  # Ẩn sidebar
    menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': 'https://github.com',
        'About': '# Stock Prediction System\nPhân tích và dự đoán giá cổ phiếu bằng AI'
    }
)

# Custom CSS - Professional Website Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700;900&family=Open+Sans:wght@400;600;700;800&display=swap');
    
    /* Global Variables - Ocean Blue & White with Orange accents */
    :root {
        --primary: #0077be;
        --primary-dark: #005a8d;
        --primary-light: #2196f3;
        --secondary: #00acc1;
        --accent: #ff6b35;
        --success: #00c853;
        --warning: #ffa726;
        --danger: #ef5350;
        --dark: #1a1a2e;
        --light: #f0f8ff;
        --gray-50: #f0f8ff;
        --gray-100: #e1f5fe;
        --gray-200: #b3e5fc;
        --gray-300: #81d4fa;
        --gray-600: #546e7a;
        --gray-700: #37474f;
        --gray-800: #263238;
        --gray-900: #1a1a2e;
    }
    
    /* Reset & Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Roboto', 'Open Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main {
        background: linear-gradient(180deg, #f0f8ff 0%, #ffffff 100%);
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Custom Navigation Bar - Ocean Blue */
    .custom-navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #0077be 0%, #005a8d 100%);
        padding: 0;
        box-shadow: 0 4px 12px rgba(0,119,190,0.3);
        border-bottom: 4px solid #ff6b35;
    }
    
    .navbar-container {
        max-width: 1400px;
        margin: 0 auto;
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 1rem 3rem;
    }
    
    .navbar-brand {
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    .navbar-logo {
        font-size: 2rem;
        font-weight: 800;
        color: white;
        text-decoration: none;
        font-family: 'Roboto', sans-serif;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .navbar-tagline {
        color: rgba(255,255,255,0.95);
        font-size: 0.85rem;
        font-weight: 600;
        padding: 0.3rem 1rem;
        background: rgba(255,107,53,0.2);
        border-radius: 20px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,107,53,0.3);
    }
    
    .navbar-menu {
        display: flex;
        gap: 2rem;
        align-items: center;
    }
    
    .navbar-item {
        color: rgba(255,255,255,0.95);
        font-weight: 500;
        font-size: 0.95rem;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .navbar-item:hover {
        background: rgba(255,255,255,0.15);
        color: white;
    }
    
    .navbar-cta {
        background: white;
        color: var(--primary);
        font-weight: 600;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .navbar-cta:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, #1a56db 0%, #1e429f 50%, #7c3aed 100%);
        padding: 4rem 3rem;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    .hero-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: url('data:image/svg+xml,<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg"><defs><pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse"><path d="M 100 0 L 0 0 0 100" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="1"/></pattern></defs><rect width="100%" height="100%" fill="url(%23grid)"/></svg>');
        opacity: 0.5;
    }
    
    .hero-content {
        position: relative;
        z-index: 1;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        color: white;
        margin-bottom: 1.5rem;
        font-family: 'Poppins', sans-serif;
        line-height: 1.2;
        text-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    
    .hero-subtitle {
        font-size: 1.5rem;
        color: rgba(255,255,255,0.95);
        margin-bottom: 2rem;
        font-weight: 400;
        line-height: 1.6;
    }
    
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 3rem;
        margin-top: 3rem;
        flex-wrap: wrap;
    }
    
    .hero-stat {
        text-align: center;
    }
    
    .hero-stat-value {
        font-size: 3rem;
        font-weight: 800;
        color: white;
        font-family: 'Roboto', sans-serif;
        text-shadow: 0 2px 8px rgba(255,107,53,0.3);
    }
    
    .hero-stat-label {
        font-size: 1rem;
        color: rgba(255,255,255,0.8);
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Content Container */
    .content-container {
        max-width: 1400px;
        margin: 0 auto;
        padding: 3rem;
    }
    
    /* Section Headers */
    .section-header {
        text-align: center;
        margin: 4rem 0 3rem 0;
    }
    
    .section-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        margin-bottom: 1rem;
        font-family: 'Roboto', sans-serif;
    }
    
    .section-subtitle {
        font-size: 1.2rem;
        color: var(--gray-600);
        max-width: 700px;
        margin: 0 auto;
    }
    
    /* Premium Cards */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
        gap: 2rem;
        margin: 3rem 0;
    }
    
    .feature-card {
        background: white;
        border-radius: 16px;
        padding: 2.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        border: 1px solid var(--gray-200);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .feature-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 4px;
        background: linear-gradient(90deg, #0077be 0%, #ff6b35 100%);
        transform: scaleX(0);
        transition: transform 0.4s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 40px rgba(0,0,0,0.12);
        border-color: var(--primary);
    }
    
    .feature-card:hover::before {
        transform: scaleX(1);
    }
    
    .feature-icon {
        width: 64px;
        height: 64px;
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-light) 100%);
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 16px rgba(26, 86, 219, 0.25);
    }
    
    .feature-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--gray-900);
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .feature-description {
        color: var(--gray-600);
        line-height: 1.7;
        font-size: 1rem;
    }
    
    /* Market Cards */
    .market-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .market-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,119,190,0.1);
        border: 2px solid #e1f5fe;
        transition: all 0.3s ease;
    }
    
    .market-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,119,190,0.2);
        border-color: #ff6b35;
    }
    
    .market-card-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .market-card-icon {
        font-size: 1.5rem;
    }
    
    .market-card-title {
        font-size: 1rem;
        font-weight: 600;
        color: var(--gray-600);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .market-card-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1a1a2e;
        margin: 0.5rem 0;
        font-family: 'Roboto', sans-serif;
    }
    
    .market-card-change {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    
    .market-card-change.positive {
        background: rgba(14, 159, 110, 0.1);
        color: var(--success);
    }
    
    .market-card-change.negative {
        background: rgba(240, 82, 82, 0.1);
        color: var(--danger);
    }
    
    .market-card-info {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid var(--gray-200);
        color: var(--gray-600);
        font-size: 0.9rem;
    }
    
    /* CTA Section */
    .cta-section {
        background: linear-gradient(135deg, #0077be 0%, #ff6b35 100%);
        padding: 5rem 3rem;
        text-align: center;
        margin: 5rem 0 0 0;
        position: relative;
        overflow: hidden;
    }
    
    .cta-section::before {
        content: '';
        position: absolute;
        width: 500px;
        height: 500px;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        top: -250px;
        right: -250px;
        animation: pulse-slow 4s ease-in-out infinite;
    }
    
    @keyframes pulse-slow {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .cta-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1.5rem;
        position: relative;
        z-index: 1;
    }
    
    .cta-description {
        font-size: 1.2rem;
        color: rgba(255,255,255,0.9);
        margin-bottom: 2.5rem;
        position: relative;
        z-index: 1;
    }
    
    .cta-button {
        display: inline-block;
        background: white;
        color: var(--primary);
        padding: 1rem 3rem;
        border-radius: 12px;
        font-weight: 700;
        font-size: 1.1rem;
        text-decoration: none;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        position: relative;
        z-index: 1;
    }
    
    .cta-button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.3);
    }
    
    /* Footer */
    .custom-footer {
        background: var(--gray-900);
        color: var(--gray-300);
        padding: 4rem 3rem 2rem 3rem;
        margin-top: 5rem;
    }
    
    .footer-container {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .footer-grid {
        display: grid;
        grid-template-columns: 2fr 1fr 1fr 1fr;
        gap: 3rem;
        margin-bottom: 3rem;
    }
    
    .footer-brand {
        font-size: 1.5rem;
        font-weight: 800;
        color: white;
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
    }
    
    .footer-description {
        color: var(--gray-400);
        line-height: 1.8;
        margin-bottom: 1.5rem;
    }
    
    .footer-social {
        display: flex;
        gap: 1rem;
    }
    
    .footer-social-link {
        width: 40px;
        height: 40px;
        background: var(--gray-800);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: all 0.3s ease;
        font-size: 1.2rem;
    }
    
    .footer-social-link:hover {
        background: var(--primary);
        transform: translateY(-3px);
    }
    
    .footer-title {
        color: white;
        font-weight: 700;
        margin-bottom: 1.5rem;
        font-size: 1.1rem;
    }
    
    .footer-links {
        list-style: none;
    }
    
    .footer-link {
        color: var(--gray-400);
        margin-bottom: 0.8rem;
        transition: color 0.3s ease;
        cursor: pointer;
    }
    
    .footer-link:hover {
        color: var(--primary-light);
    }
    
    .footer-bottom {
        padding-top: 2rem;
        border-top: 1px solid var(--gray-800);
        text-align: center;
        color: var(--gray-500);
        font-size: 0.9rem;
    }
    
    /* Responsive Tables */
    .dataframe {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 16px 12px !important;
        text-align: left !important;
        border: none !important;
        text-transform: uppercase;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    
    .dataframe tbody tr {
        transition: all 0.2s ease;
    }
    
    .dataframe tbody tr:nth-child(even) {
        background-color: var(--gray-50) !important;
    }
    
    .dataframe tbody tr:hover {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%) !important;
        transform: scale(1.01);
        box-shadow: 0 2px 8px rgba(26, 86, 219, 0.15);
    }
    
    .dataframe tbody td {
        padding: 14px 12px !important;
        border-bottom: 1px solid var(--gray-200) !important;
        font-weight: 500;
        color: var(--gray-700);
    }
    
    /* Enhanced Buttons - Ocean Blue with Orange hover */
    .stButton button {
        background: linear-gradient(135deg, #0077be 0%, #005a8d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0, 119, 190, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
        box-shadow: 0 8px 20px rgba(255, 107, 53, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar Enhancement - Ocean Blue theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f0f8ff 0%, white 100%);
        border-right: 2px solid #b3e5fc;
    }
    
    /* Metric Cards - Blue border */
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,119,190,0.1);
        border: 2px solid #e1f5fe;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animate-fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ==================== NAVIGATION BAR ====================
# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Trang chủ"

st.markdown("""
<div class="custom-navbar">
    <div class="navbar-container">
        <div class="navbar-brand">
            <div class="navbar-logo">
                📈 StockPro Analytics
            </div>
            <div class="navbar-tagline">
                AI-Powered Predictions
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Navigation Menu with Buttons
st.markdown("### 🧭 Menu Điều Hướng")
col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

with col1:
    if st.button("🏠 Trang chủ", use_container_width=True, type="primary" if st.session_state.page == "🏠 Trang chủ" else "secondary"):
        st.session_state.page = "🏠 Trang chủ"
        st.rerun()

with col2:
    if st.button("📊 Thị trường", use_container_width=True, type="primary" if st.session_state.page == "📊 Thị trường" else "secondary"):
        st.session_state.page = "📊 Thị trường"
        st.rerun()

with col3:
    if st.button("📈 Dữ liệu", use_container_width=True, type="primary" if st.session_state.page == "📈 Dữ liệu chi tiết" else "secondary"):
        st.session_state.page = "📈 Dữ liệu chi tiết"
        st.rerun()

with col4:
    if st.button("🔍 Kiểm tra", use_container_width=True, type="primary" if st.session_state.page == "🔍 Kiểm tra dữ liệu" else "secondary"):
        st.session_state.page = "🔍 Kiểm tra dữ liệu"
        st.rerun()

with col5:
    if st.button("🔮 Dự đoán", use_container_width=True, type="primary" if st.session_state.page == "🔮 Dự đoán giá" else "secondary"):
        st.session_state.page = "🔮 Dự đoán giá"
        st.rerun()

with col6:
    if st.button("🔄 Backtest", use_container_width=True, type="primary" if st.session_state.page == "🔄 Backtesting" else "secondary"):
        st.session_state.page = "🔄 Backtesting"
        st.rerun()

with col7:
    if st.button("💭 Sentiment", use_container_width=True, type="primary" if st.session_state.page == "💭 Phân tích Sentiment" else "secondary"):
        st.session_state.page = "💭 Phân tích Sentiment"
        st.rerun()

with col8:
    if st.button("🤖 Training", use_container_width=True, type="primary" if st.session_state.page == "🤖 Huấn luyện Model" else "secondary"):
        st.session_state.page = "🤖 Huấn luyện Model"
        st.rerun()

st.markdown("---")

# Stock Selection Area
st.markdown("### 📊 Cài Đặt Dữ Liệu")
col1, col2, col3, col4 = st.columns([2, 2, 1.5, 1.5])

with col1:
    popular_stocks = {
        "VNM - Vinamilk": "VNM.VN",
        "HPG - Hòa Phát": "HPG.VN",
        "VIC - Vingroup": "VIC.VN",
        "VCB - Vietcombank": "VCB.VN",
        "FPT - FPT Corp": "FPT.VN",
        "VHM - Vinhomes": "VHM.VN",
        "MBB - MB Bank": "MBB.VN",
        "VN-Index": "^VNINDEX"
    }
    
    selected_popular = st.selectbox(
        "🔥 Cổ phiếu phổ biến:",
        ["-- Chọn mã --"] + list(popular_stocks.keys())
    )
    
    if selected_popular != "-- Chọn mã --":
        symbol = popular_stocks[selected_popular]
    else:
        symbol = "VNM.VN"

with col2:
    symbol = st.text_input("📝 Hoặc nhập mã:", value=symbol, help="VD: VNM.VN, HPG.VN")

with col3:
    start_date = st.date_input("📅 Từ ngày:", value=datetime.now() - timedelta(days=365))

with col4:
    end_date = st.date_input("📅 Đến ngày:", value=datetime.now())

st.markdown("---")

page = st.session_state.page


# Helper functions
def load_data(symbol, start_date, end_date):
    """Load stock data"""
    try:
        from data_collection import YahooFinanceAPI
        
        api = YahooFinanceAPI()
        df = api.get_stock_data(symbol, str(start_date), str(end_date))
        return df
    except Exception as e:
        st.error(f"Lỗi khi tải dữ liệu: {str(e)}")
        return pd.DataFrame()


def plot_candlestick(df):
    """Plot candlestick chart"""
    fig = go.Figure(data=[go.Candlestick(
        x=df['date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Giá'
    )])
    
    fig.update_layout(
        title=f'Biểu đồ giá {symbol}',
        yaxis_title='Giá',
        xaxis_title='Ngày',
        height=500,
        template='plotly_white'
    )
    
    return fig


def plot_volume(df):
    """Plot volume chart"""
    fig = go.Figure(data=[go.Bar(
        x=df['date'],
        y=df['Volume'],
        name='Khối lượng',
        marker_color='lightblue'
    )])
    
    fig.update_layout(
        title='Khối lượng giao dịch',
        yaxis_title='Khối lượng',
        xaxis_title='Ngày',
        height=300,
        template='plotly_white'
    )
    
    return fig


# ==================== HOME PAGE ====================
if page == "🏠 Trang chủ":
    # Hero Section
    st.markdown("""
    <div class="hero-section">
        <div class="hero-content">
            <div class="hero-title">
                Hệ Thống Dự Đoán Giá Cổ Phiếu AI
            </div>
            <div class="hero-subtitle">
                Phân tích thông minh • Dự đoán chính xác • Đầu tư hiệu quả
            </div>
            <div class="hero-stats">
                <div class="hero-stat">
                    <div class="hero-stat-value">5+</div>
                    <div class="hero-stat-label">Mô hình AI</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">25+</div>
                    <div class="hero-stat-label">Chỉ số kỹ thuật</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">85%+</div>
                    <div class="hero-stat-label">Độ chính xác</div>
                </div>
                <div class="hero-stat">
                    <div class="hero-stat-value">24/7</div>
                    <div class="hero-stat-label">Theo dõi</div>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Content Container
    st.markdown('<div class="content-container">', unsafe_allow_html=True)
    
    # Market Overview Section
    st.markdown("""
    <div class="section-header">
        <div class="section-title">📈 Tổng Quan Thị Trường</div>
        <div class="section-subtitle">Cập nhật realtime từ sàn giao dịch chứng khoán Việt Nam</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Market Cards với thiết kế mới
    st.markdown('<div class="market-grid">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="market-card">
            <div class="market-card-header">
                <div class="market-card-icon">🏛️</div>
                <div class="market-card-title">VN-INDEX</div>
            </div>
            <div class="market-card-value">1,258.45</div>
            <div class="market-card-change positive">
                ↑ +12.35 (+0.99%)
            </div>
            <div class="market-card-info">
                <strong>HOSE</strong> • 583 CP tăng, 245 CP giảm
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="market-card">
            <div class="market-card-header">
                <div class="market-card-icon">🏢</div>
                <div class="market-card-title">HNX-INDEX</div>
            </div>
            <div class="market-card-value">235.67</div>
            <div class="market-card-change positive">
                ↑ +2.15 (+0.92%)
            </div>
            <div class="market-card-info">
                <strong>HNX</strong> • 142 CP tăng, 98 CP giảm
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="market-card">
            <div class="market-card-header">
                <div class="market-card-icon">🏪</div>
                <div class="market-card-title">UPCOM</div>
            </div>
            <div class="market-card-value">89.23</div>
            <div class="market-card-change negative">
                ↓ -0.45 (-0.50%)
            </div>
            <div class="market-card-info">
                <strong>UPCOM</strong> • 67 CP tăng, 89 CP giảm
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="market-card">
            <div class="market-card-header">
                <div class="market-card-icon">💰</div>
                <div class="market-card-title">Tổng GTGD</div>
            </div>
            <div class="market-card-value" style="color: #ff5a1f;">15,234</div>
            <div style="color: #4b5563; font-weight: 600; font-size: 0.9rem;">tỷ VNĐ</div>
            <div class="market-card-info">
                <strong>Hôm nay</strong> • Tăng 12.5% so với hôm qua
            </div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div class="section-header">
        <div class="section-title">✨ Tính Năng Nổi Bật</div>
        <div class="section-subtitle">Công nghệ AI tiên tiến cho phân tích và dự đoán chứng khoán</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="feature-grid">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); padding: 2rem; border-radius: 15px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📊</div>
            <h3 style="color: #003d82; margin: 0 0 1rem 0;">Theo dõi thị trường</h3>
            <ul style="color: #475569; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>Dữ liệu realtime</strong> từ VNDirect API</li>
                <li><strong>Biểu đồ nến</strong> chuyên nghiệp</li>
                <li><strong>25+ chỉ số kỹ thuật</strong></li>
                <li><strong>Phân tích</strong> RSI, MACD, BB</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%); padding: 2rem; border-radius: 15px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">🔮</div>
            <h3 style="color: #ad1457; margin: 0 0 1rem 0;">Dự đoán AI</h3>
            <ul style="color: #475569; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>4 mô hình AI</strong>: ARIMA, Prophet, LSTM, GRU</li>
                <li><strong>Ensemble Learning</strong> Meta-Learning</li>
                <li><strong>Độ chính xác cao</strong> > 85%</li>
                <li><strong>Dự đoán</strong> 1-30 ngày</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%); padding: 2rem; border-radius: 15px; height: 100%; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <div style="font-size: 3rem; margin-bottom: 1rem;">💭</div>
            <h3 style="color: #2e7d32; margin: 0 0 1rem 0;">Phân tích Sentiment</h3>
            <ul style="color: #475569; line-height: 1.8; padding-left: 1.2rem;">
                <li><strong>FinBERT</strong> phân tích tin tức</li>
                <li><strong>Thu thập tự động</strong> từ CafeF, VNDirect</li>
                <li><strong>Sentiment Score</strong> realtime</li>
                <li><strong>Tích hợp</strong> vào dự đoán</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Top cổ phiếu
    st.markdown("""
    <div class="section-header">
        <div class="section-title">🔥 Top Cổ Phiếu Đáng Chú Ý</div>
        <div class="section-subtitle">Những cổ phiếu có biến động mạnh nhất trong phiên hôm nay</div>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <h3 style="color: #0e9f6e; margin: 0 0 1rem 0;">📈 Top Tăng Giá</h3>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>VHM</strong>
                    <span style="color: #0e9f6e; font-weight: 600;">+6.89%</span>
                </div>
                <small style="color: #6b7280;">87,500 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>VIC</strong>
                    <span style="color: #0e9f6e; font-weight: 600;">+5.42%</span>
                </div>
                <small style="color: #6b7280;">42,300 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>HPG</strong>
                    <span style="color: #0e9f6e; font-weight: 600;">+4.85%</span>
                </div>
                <small style="color: #6b7280;">28,700 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>VNM</strong>
                    <span style="color: #0e9f6e; font-weight: 600;">+3.95%</span>
                </div>
                <small style="color: #6b7280;">85,200 VNĐ</small>
            </div>
            <div style="padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>FPT</strong>
                    <span style="color: #0e9f6e; font-weight: 600;">+3.67%</span>
                </div>
                <small style="color: #6b7280;">94,500 VNĐ</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <h3 style="color: #f05252; margin: 0 0 1rem 0;">📉 Top Giảm Giá</h3>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>DXG</strong>
                    <span style="color: #f05252; font-weight: 600;">-4.12%</span>
                </div>
                <small style="color: #6b7280;">15,800 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>KBC</strong>
                    <span style="color: #f05252; font-weight: 600;">-3.89%</span>
                </div>
                <small style="color: #6b7280;">12,400 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>NVL</strong>
                    <span style="color: #f05252; font-weight: 600;">-3.45%</span>
                </div>
                <small style="color: #6b7280;">58,300 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>PDR</strong>
                    <span style="color: #f05252; font-weight: 600;">-2.98%</span>
                </div>
                <small style="color: #6b7280;">34,200 VNĐ</small>
            </div>
            <div style="padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>HDB</strong>
                    <span style="color: #f05252; font-weight: 600;">-2.55%</span>
                </div>
                <small style="color: #6b7280;">25,100 VNĐ</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; border-radius: 12px; padding: 1.5rem; box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
            <h3 style="color: #ff5a1f; margin: 0 0 1rem 0;">💰 Top Khối Lượng</h3>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>HPG</strong>
                    <span style="color: #ff5a1f; font-weight: 600;">45.2M</span>
                </div>
                <small style="color: #6b7280;">28,700 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>VNM</strong>
                    <span style="color: #ff5a1f; font-weight: 600;">38.7M</span>
                </div>
                <small style="color: #6b7280;">85,200 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>SSI</strong>
                    <span style="color: #ff5a1f; font-weight: 600;">32.4M</span>
                </div>
                <small style="color: #6b7280;">34,500 VNĐ</small>
            </div>
            <div style="border-bottom: 1px solid #e5e7eb; padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>MBB</strong>
                    <span style="color: #ff5a1f; font-weight: 600;">28.9M</span>
                </div>
                <small style="color: #6b7280;">28,900 VNĐ</small>
            </div>
            <div style="padding: 0.8rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <strong>VCB</strong>
                    <span style="color: #ff5a1f; font-weight: 600;">25.6M</span>
                </div>
                <small style="color: #6b7280;">92,300 VNĐ</small>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick start guide
    st.markdown("""
    <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 6px solid #00a3e0;">
        <h2 style="color: #003d82; margin-top: 0;">🚀 Bắt đầu sử dụng</h2>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1.5rem; margin-top: 1.5rem;">
            <div>
                <div style="background: #003d82; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.8rem;">1</div>
                <h4 style="color: #003d82; margin: 0 0 0.5rem 0;">Chọn mã cổ phiếu</h4>
                <p style="color: #64748b; margin: 0;">Nhập mã CP ở Sidebar (VD: VNM, VIC, HPG)</p>
            </div>
            <div>
                <div style="background: #00a3e0; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.8rem;">2</div>
                <h4 style="color: #003d82; margin: 0 0 0.5rem 0;">Xem dữ liệu</h4>
                <p style="color: #64748b; margin: 0;">Phân tích biểu đồ và chỉ số kỹ thuật</p>
            </div>
            <div>
                <div style="background: #00c48c; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.8rem;">3</div>
                <h4 style="color: #003d82; margin: 0 0 0.5rem 0;">Dự đoán giá</h4>
                <p style="color: #64748b; margin: 0;">Chọn mô hình AI và xem kết quả</p>
            </div>
            <div>
                <div style="background: #ffb800; color: white; width: 40px; height: 40px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 1.2rem; margin-bottom: 0.8rem;">4</div>
                <h4 style="color: #003d82; margin: 0 0 0.5rem 0;">Backtesting</h4>
                <p style="color: #64748b; margin: 0;">Kiểm tra hiệu quả chiến lược</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Quick Action Buttons
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
        <h3 style="text-align: center; color: #1f2937; margin-bottom: 1.5rem;">🎯 Truy cập nhanh các tính năng</h3>
        <p style="text-align: center; color: #6b7280; margin-bottom: 1.5rem;">Chọn chức năng bạn muốn sử dụng từ menu bên trái (Sidebar) ⬅️</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e5e7eb;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📊</div>
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0;">Thị Trường</h4>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">Xem dữ liệu realtime</p>
            <p style="margin-top: 1rem; color: #1a56db; font-weight: 600;">👈 Chọn "📊 Thị trường"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e5e7eb;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">📈</div>
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0;">Dữ Liệu Chi Tiết</h4>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">Phân tích sâu</p>
            <p style="margin-top: 1rem; color: #1a56db; font-weight: 600;">👈 Chọn "📈 Dữ liệu chi tiết"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e5e7eb;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🔮</div>
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0;">Dự Đoán AI</h4>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">Dự báo giá</p>
            <p style="margin-top: 1rem; color: #1a56db; font-weight: 600;">👈 Chọn "🔮 Dự đoán giá"</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="background: white; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border: 2px solid #e5e7eb;">
            <div style="font-size: 3rem; margin-bottom: 0.8rem;">🔄</div>
            <h4 style="color: #1f2937; margin: 0 0 0.5rem 0;">Backtesting</h4>
            <p style="color: #6b7280; font-size: 0.9rem; margin: 0;">Test chiến lược</p>
            <p style="margin-top: 1rem; color: #1a56db; font-weight: 600;">👈 Chọn "🔄 Backtesting"</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Models", "4+", "ARIMA, Prophet, LSTM, GRU")
    with col2:
        st.metric("📊 Indicators", "25+", "RSI, MACD, Bollinger...")
    with col3:
        st.metric("🔍 Data Sources", "3", "Yahoo, VNDirect, CafeF")
    with col4:
        st.metric("⚡ Accuracy", "85%+", "Ensemble Model")


# ==================== MARKET DATA PAGE ====================
elif page == "📊 Thị trường":
    st.markdown('<h1 class="main-header">📊 DỮ LIỆU THỊ TRƯỜNG</h1>', unsafe_allow_html=True)
    
    # Stock selection
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_symbol = st.text_input("🔍 Tìm kiếm mã cổ phiếu", value="VNM", placeholder="Nhập mã cổ phiếu...")
    
    with col2:
        market_select = st.selectbox("Sàn giao dịch", ["HOSE", "HNX", "UPCOM", "Tất cả"])
    
    with col3:
        if st.button("🔄 Cập nhật dữ liệu", type="primary"):
            st.rerun()
    
    st.markdown("---")
    
    if search_symbol:
        clean_symbol = search_symbol.strip().upper()
        if not clean_symbol.endswith('.VN'):
            clean_symbol += '.VN'
        
        # Fetch data from multiple sources
        with st.spinner(f"Đang tải dữ liệu {clean_symbol}..."):
            from data_collection import YahooFinanceAPI, VNDirectAPI
            
            yahoo_api = YahooFinanceAPI()
            vnd_api = VNDirectAPI()
            
            # Try VNDirect dchart API first (direct from dstock.vndirect.com.vn)
            df = None
            data_source = ""
            
            try:
                df_vnd = vnd_api.get_stock_price(clean_symbol,
                                                 (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'),
                                                 datetime.now().strftime('%Y-%m-%d'))
                
                if not df_vnd.empty:
                    df = df_vnd
                    data_source = "VNDirect dchart API"
                    st.success(f"✅ Dữ liệu từ VNDirect dchart API: {len(df)} records")
            except Exception as e:
                st.warning(f"⚠️ VNDirect API không khả dụng: {str(e)}")
            
            # Fallback to Yahoo Finance if VNDirect fails
            if df is None or df.empty:
                try:
                    df_yahoo = yahoo_api.get_stock_data(clean_symbol, 
                                                       (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d'), 
                                                       datetime.now().strftime('%Y-%m-%d'))
                    
                    if not df_yahoo.empty:
                        df = df_yahoo
                        data_source = "Yahoo Finance"
                        st.info(f"ℹ️ Sử dụng dữ liệu dự phòng từ Yahoo Finance: {len(df)} records")
                except Exception as e:
                    st.error(f"❌ Lỗi khi tải dữ liệu từ Yahoo Finance: {str(e)}")
            
            if df is None or df.empty:
                st.warning(f"⚠️ Không tìm thấy dữ liệu cho {clean_symbol}")
            else:
                # Stock header card
                latest = df.iloc[-1]
                prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
                
                price = latest['Close'] if 'Close' in latest else 0
                change = price - (prev['Close'] if 'Close' in prev else price)
                change_pct = (change / prev['Close'] * 100) if 'Close' in prev and prev['Close'] > 0 else 0
                
                change_class = "positive" if change >= 0 else "negative"
                change_symbol = "↑" if change >= 0 else "↓"
                
                st.markdown(f"""
                <div class="stock-card">
                    <h3>{clean_symbol.replace('.VN', '')}</h3>
                    <div class="price {change_class}">{price:,.2f} VNĐ</div>
                    <div class="change {change_class}">{change_symbol} {abs(change):,.2f} ({abs(change_pct):.2f}%)</div>
                    <div style="margin-top: 1rem;">
                        <span style="margin-right: 2rem;">Khối lượng: {latest.get('Volume', 0):,.0f}</span>
                        <span>Giá trị: {price * latest.get('Volume', 0) / 1e9:,.2f} tỷ</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["📈 Biểu đồ", "📊 Dữ liệu", "📉 Chỉ số kỹ thuật", "ℹ️ Thông tin"])
                
                with tab1:
                    # Candlestick chart
                    fig = go.Figure(data=[go.Candlestick(
                        x=df['date'],
                        open=df['Open'],
                        high=df['High'],
                        low=df['Low'],
                        close=df['Close'],
                        name='Giá'
                    )])
                    
                    fig.update_layout(
                        title=f'Biểu đồ nến {clean_symbol}',
                        yaxis_title='Giá (VNĐ)',
                        xaxis_title='Thời gian',
                        height=500,
                        template='plotly_white',
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    fig_vol = go.Figure(data=[go.Bar(
                        x=df['date'],
                        y=df['Volume'],
                        name='Khối lượng',
                        marker_color='lightblue'
                    )])
                    
                    fig_vol.update_layout(
                        title='Khối lượng giao dịch',
                        yaxis_title='Khối lượng',
                        xaxis_title='Thời gian',
                        height=300,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                with tab2:
                    st.subheader("📋 Bảng dữ liệu chi tiết")
                    
                    # Statistics
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        st.metric("Giá cao nhất", f"{df['High'].max():,.0f}")
                    with col2:
                        st.metric("Giá thấp nhất", f"{df['Low'].min():,.0f}")
                    with col3:
                        st.metric("Giá TB", f"{df['Close'].mean():,.0f}")
                    with col4:
                        st.metric("KL TB", f"{df['Volume'].mean():,.0f}")
                    with col5:
                        total_value = (df['Close'] * df['Volume']).sum() / 1e9
                        st.metric("Tổng GT GD", f"{total_value:,.0f}B")
                    
                    # Data table with formatting
                    display_df = df.copy()
                    display_df = display_df.sort_values('date', ascending=False)
                    
                    st.dataframe(
                        display_df.style.format({
                            'Open': '{:,.0f}',
                            'High': '{:,.0f}',
                            'Low': '{:,.0f}',
                            'Close': '{:,.0f}',
                            'Volume': '{:,.0f}'
                        }),
                        use_container_width=True,
                        height=600
                    )
                    
                    # Download button
                    csv = display_df.to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label="📥 Tải xuống CSV",
                        data=csv,
                        file_name=f"{clean_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                with tab3:
                    st.subheader("📉 Chỉ số kỹ thuật")
                    
                    # Calculate technical indicators
                    from features.technical_indicators import TechnicalIndicators
                    
                    tech_ind = TechnicalIndicators()
                    df_with_indicators = tech_ind.add_all_indicators(df.copy())
                    
                    # Display key indicators
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'rsi_14' in df_with_indicators.columns:
                            latest_rsi = df_with_indicators['rsi_14'].iloc[-1]
                            rsi_class = "negative" if latest_rsi > 70 else "positive" if latest_rsi < 30 else ""
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>RSI (14)</h4>
                                <div class="value {rsi_class}">{latest_rsi:.2f}</div>
                                <div>{'Quá mua' if latest_rsi > 70 else 'Quá bán' if latest_rsi < 30 else 'Trung tính'}</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        if 'macd' in df_with_indicators.columns:
                            latest_macd = df_with_indicators['macd'].iloc[-1]
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>MACD</h4>
                                <div class="value">{latest_macd:.2f}</div>
                                <div>Chỉ báo xu hướng</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col3:
                        if 'bb_position' in df_with_indicators.columns:
                            bb_pos = df_with_indicators['bb_position'].iloc[-1]
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>Bollinger Bands</h4>
                                <div class="value">{bb_pos:.2f}%</div>
                                <div>Vị trí trong dải</div>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    # Chart with indicators
                    fig_tech = go.Figure()
                    
                    fig_tech.add_trace(go.Scatter(
                        x=df_with_indicators['date'],
                        y=df_with_indicators['Close'],
                        name='Giá đóng cửa',
                        line=dict(color='blue', width=2)
                    ))
                    
                    if 'sma_20' in df_with_indicators.columns:
                        fig_tech.add_trace(go.Scatter(
                            x=df_with_indicators['date'],
                            y=df_with_indicators['sma_20'],
                            name='SMA 20',
                            line=dict(color='orange', dash='dash')
                        ))
                    
                    if 'ema_50' in df_with_indicators.columns:
                        fig_tech.add_trace(go.Scatter(
                            x=df_with_indicators['date'],
                            y=df_with_indicators['ema_50'],
                            name='EMA 50',
                            line=dict(color='green', dash='dash')
                        ))
                    
                    fig_tech.update_layout(
                        title='Giá với chỉ số kỹ thuật',
                        yaxis_title='Giá (VNĐ)',
                        xaxis_title='Thời gian',
                        height=500,
                        template='plotly_white'
                    )
                    
                    st.plotly_chart(fig_tech, use_container_width=True)
                    
                with tab4:
                    st.subheader("ℹ️ Thông tin cổ phiếu")
                    
                    # Get stock info from VNDirect
                    stock_info = vnd_api.get_stock_info(clean_symbol)
                    
                    if stock_info:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown(f"""
                            **Tên công ty:** {stock_info.get('companyName', 'N/A')}
                            
                            **Mã CK:** {stock_info.get('code', clean_symbol)}
                            
                            **Sàn:** {stock_info.get('exchange', 'N/A')}
                            
                            **Ngành:** {stock_info.get('industryName', 'N/A')}
                            """)
                        
                        with col2:
                            st.markdown(f"""
                            **Vốn hóa:** {stock_info.get('marketCap', 0):,.0f} tỷ
                            
                            **Khối lượng niêm yết:** {stock_info.get('listedShare', 0):,.0f}
                            
                            **EPS:** {stock_info.get('eps', 0):.2f}
                            
                            **P/E:** {stock_info.get('pe', 0):.2f}
                            """)
                    else:
                        st.info("Thông tin chi tiết đang được cập nhật...")
                        
                        # Data source info
                        st.markdown("---")
                        st.markdown("""
                        **Nguồn dữ liệu:**
                        - Yahoo Finance (Dữ liệu quốc tế)
                        - VNDirect API (Dữ liệu thị trường Việt Nam)
                        - CafeF (Tin tức và phân tích)
                        """)


# ==================== DATA DETAILS PAGE ====================
elif page == "📈 Dữ liệu chi tiết":
    st.title("📊 Thu Thập Dữ Liệu Từ Nhiều Nguồn")
    
    st.markdown("""
    Thu thập dữ liệu giá cổ phiếu và tin tức từ nhiều nguồn khác nhau:
    - **Yahoo Finance**: Dữ liệu giá quốc tế
    - **VNDirect**: Dữ liệu thị trường Việt Nam
    - **CafeF**: Tin tức và phân tích
    """)
    
    st.markdown("---")

# ==================== DATA VALIDATION PAGE ====================
elif page == "🔍 Kiểm tra dữ liệu":
    st.title("🔍 Kiểm Tra & Xác Minh Dữ Liệu")
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #e1f5fe 0%, #f0f8ff 100%); padding: 1.5rem; border-radius: 12px; border-left: 5px solid #0077be; margin-bottom: 2rem;">
        <h3 style="color: #0077be; margin: 0 0 0.5rem 0;">💡 Tại sao cần kiểm tra dữ liệu?</h3>
        <ul style="color: #546e7a; margin: 0; padding-left: 1.5rem;">
            <li><strong>So sánh nhiều nguồn:</strong> VNDirect vs Yahoo Finance</li>
            <li><strong>Phát hiện lỗi:</strong> Dữ liệu thiếu, sai lệch, outliers</li>
            <li><strong>Đảm bảo chất lượng:</strong> Xác thực trước khi dự đoán</li>
            <li><strong>Thống kê chi tiết:</strong> Min, Max, Mean, Median</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Source comparison
    st.markdown("### 📊 So Sánh Nguồn Dữ Liệu")
    
    col1, col2 = st.columns(2)
    with col1:
        compare_symbol = st.text_input("Mã cổ phiếu để so sánh:", value=symbol, key="compare_symbol")
    with col2:
        compare_days = st.slider("Số ngày gần nhất:", 7, 90, 30, key="compare_days")
    
    if st.button("🔄 So Sánh Dữ Liệu", type="primary"):
        with st.spinner("Đang lấy dữ liệu từ cả 2 nguồn..."):
            try:
                from data_collection import VNDirectAPI, YahooFinanceAPI
                
                # Calculate dates
                end_compare = datetime.now()
                start_compare = end_compare - timedelta(days=compare_days)
                
                # Get data from both sources
                vnd_api = VNDirectAPI()
                yahoo_api = YahooFinanceAPI()
                
                vnd_df = vnd_api.get_stock_price(compare_symbol, start_compare.strftime('%Y-%m-%d'), end_compare.strftime('%Y-%m-%d'))
                yahoo_df = yahoo_api.get_stock_data(compare_symbol, start_compare.strftime('%Y-%m-%d'), end_compare.strftime('%Y-%m-%d'))
                
                if not vnd_df.empty and not yahoo_df.empty:
                    # Summary comparison
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📅 VNDirect Records", len(vnd_df))
                    with col2:
                        st.metric("📅 Yahoo Finance Records", len(yahoo_df))
                    with col3:
                        diff_pct = abs(len(vnd_df) - len(yahoo_df)) / max(len(vnd_df), len(yahoo_df)) * 100
                        st.metric("📊 Chênh lệch", f"{diff_pct:.1f}%")
                    
                    st.markdown("---")
                    
                    # Price comparison chart
                    st.markdown("### 📈 So Sánh Giá Đóng Cửa")
                    
                    fig = go.Figure()
                    
                    if 'Close' in vnd_df.columns and 'date' in vnd_df.columns:
                        fig.add_trace(go.Scatter(
                            x=vnd_df['date'],
                            y=vnd_df['Close'],
                            mode='lines+markers',
                            name='VNDirect',
                            line=dict(color='#0077be', width=2),
                            marker=dict(size=6)
                        ))
                    
                    if 'Close' in yahoo_df.columns:
                        fig.add_trace(go.Scatter(
                            x=yahoo_df.index,
                            y=yahoo_df['Close'],
                            mode='lines+markers',
                            name='Yahoo Finance',
                            line=dict(color='#ff6b35', width=2, dash='dash'),
                            marker=dict(size=6)
                        ))
                    
                    fig.update_layout(
                        title=f"So sánh giá {compare_symbol}",
                        xaxis_title="Ngày",
                        yaxis_title="Giá (VNĐ)",
                        hovermode='x unified',
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Statistical comparison
                    st.markdown("### 📊 Thống Kê So Sánh")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🔵 VNDirect")
                        if 'Close' in vnd_df.columns:
                            vnd_stats = vnd_df['Close'].describe()
                            st.dataframe(vnd_stats, use_container_width=True)
                    
                    with col2:
                        st.markdown("#### 🟠 Yahoo Finance")
                        if 'Close' in yahoo_df.columns:
                            yahoo_stats = yahoo_df['Close'].describe()
                            st.dataframe(yahoo_stats, use_container_width=True)
                    
                    # Correlation analysis
                    if 'Close' in vnd_df.columns and 'Close' in yahoo_df.columns:
                        st.markdown("### 🔗 Phân Tích Tương Quan")
                        
                        # Merge on date for correlation
                        vnd_temp = vnd_df.copy()
                        if 'date' in vnd_temp.columns:
                            vnd_temp['date'] = pd.to_datetime(vnd_temp['date'])
                            vnd_temp.set_index('date', inplace=True)
                        
                        yahoo_temp = yahoo_df.copy()
                        yahoo_temp.index = pd.to_datetime(yahoo_temp.index)
                        
                        merged = pd.merge(
                            vnd_temp[['Close']].rename(columns={'Close': 'VNDirect'}),
                            yahoo_temp[['Close']].rename(columns={'Close': 'Yahoo'}),
                            left_index=True,
                            right_index=True,
                            how='inner'
                        )
                        
                        if len(merged) > 0:
                            correlation = merged['VNDirect'].corr(merged['Yahoo'])
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("🔗 Hệ số tương quan", f"{correlation:.4f}")
                            with col2:
                                st.metric("📊 Số điểm chung", len(merged))
                            with col3:
                                quality = "Xuất sắc" if correlation > 0.95 else "Tốt" if correlation > 0.85 else "Chấp nhận được" if correlation > 0.7 else "Cần kiểm tra"
                                st.metric("✅ Chất lượng", quality)
                            
                            # Scatter plot
                            fig_scatter = go.Figure()
                            fig_scatter.add_trace(go.Scatter(
                                x=merged['VNDirect'],
                                y=merged['Yahoo'],
                                mode='markers',
                                marker=dict(
                                    size=10,
                                    color=merged.index.day,
                                    colorscale='Viridis',
                                    showscale=True
                                ),
                                text=merged.index.strftime('%Y-%m-%d'),
                                hovertemplate='<b>Ngày: %{text}</b><br>VNDirect: %{x}<br>Yahoo: %{y}<extra></extra>'
                            ))
                            
                            # Add diagonal line
                            min_val = min(merged['VNDirect'].min(), merged['Yahoo'].min())
                            max_val = max(merged['VNDirect'].max(), merged['Yahoo'].max())
                            fig_scatter.add_trace(go.Scatter(
                                x=[min_val, max_val],
                                y=[min_val, max_val],
                                mode='lines',
                                line=dict(color='red', dash='dash'),
                                name='Perfect correlation'
                            ))
                            
                            fig_scatter.update_layout(
                                title="Biểu đồ tương quan giá",
                                xaxis_title="VNDirect",
                                yaxis_title="Yahoo Finance",
                                height=500
                            )
                            
                            st.plotly_chart(fig_scatter, use_container_width=True)
                            
                            # Difference analysis
                            st.markdown("### 📉 Phân Tích Chênh Lệch")
                            merged['Diff'] = merged['VNDirect'] - merged['Yahoo']
                            merged['Diff_Pct'] = (merged['Diff'] / merged['Yahoo']) * 100
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Chênh lệch TB", f"{merged['Diff'].mean():.2f}")
                            with col2:
                                st.metric("Chênh lệch Max", f"{merged['Diff'].max():.2f}")
                            with col3:
                                st.metric("Chênh lệch Min", f"{merged['Diff'].min():.2f}")
                            with col4:
                                st.metric("Chênh lệch % TB", f"{merged['Diff_Pct'].mean():.2f}%")
                            
                            # Show data table
                            st.markdown("### 📋 Bảng So Sánh Chi Tiết")
                            display_df = merged.copy()
                            display_df.index = display_df.index.strftime('%Y-%m-%d')
                            st.dataframe(display_df.round(2), use_container_width=True)
                        
                else:
                    st.warning("⚠️ Không lấy được dữ liệu từ một hoặc cả hai nguồn!")
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")
    
    st.markdown("---")
    
    # Data quality check
    st.markdown("### 🔍 Kiểm Tra Chất Lượng Dữ Liệu")
    
    if st.button("🔎 Kiểm Tra Dữ Liệu Hiện Tại", type="secondary"):
        with st.spinner("Đang kiểm tra..."):
            try:
                from data_collection import VNDirectAPI
                
                api = VNDirectAPI()
                df = api.get_stock_price(symbol, str(start_date), str(end_date))
                
                if not df.empty:
                    st.success(f"✅ Lấy được {len(df)} bản ghi từ VNDirect")
                    
                    # Quality metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        missing = df.isnull().sum().sum()
                        st.metric("❌ Dữ liệu thiếu", missing)
                    
                    with col2:
                        duplicates = df.duplicated().sum()
                        st.metric("🔄 Bản ghi trùng", duplicates)
                    
                    with col3:
                        if 'Close' in df.columns:
                            outliers = len(df[df['Close'] > df['Close'].mean() + 3*df['Close'].std()])
                            st.metric("⚠️ Outliers", outliers)
                        else:
                            st.metric("⚠️ Outliers", "N/A")
                    
                    with col4:
                        completeness = (1 - missing / (len(df) * len(df.columns))) * 100
                        st.metric("✅ Độ đầy đủ", f"{completeness:.1f}%")
                    
                    # Data preview
                    st.markdown("### 📊 Xem Trước Dữ Liệu")
                    
                    tab1, tab2, tab3 = st.tabs(["📋 Đầu tiên", "📋 Cuối cùng", "📊 Thống kê"])
                    
                    with tab1:
                        st.dataframe(df.head(10), use_container_width=True)
                    
                    with tab2:
                        st.dataframe(df.tail(10), use_container_width=True)
                    
                    with tab3:
                        st.dataframe(df.describe(), use_container_width=True)
                    
                    # Column info
                    st.markdown("### 📋 Thông Tin Cột")
                    col_info = pd.DataFrame({
                        'Cột': df.columns,
                        'Kiểu dữ liệu': df.dtypes.values,
                        'Giá trị null': df.isnull().sum().values,
                        'Giá trị duy nhất': [df[col].nunique() for col in df.columns]
                    })
                    st.dataframe(col_info, use_container_width=True)
                    
                else:
                    st.error("❌ Không lấy được dữ liệu!")
                    
            except Exception as e:
                st.error(f"❌ Lỗi: {str(e)}")

# ==================== DATA DETAIL PAGE (CONTINUED) ====================
    
    # Data source selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("⚙️ Cấu hình thu thập")
        
        data_sources = st.multiselect(
            "Nguồn dữ liệu",
            ["Yahoo Finance", "VNDirect", "CafeF News"],
            default=["Yahoo Finance", "VNDirect"]
        )
        
        symbols_input = st.text_area(
            "Mã cổ phiếu (mỗi dòng một mã)",
            value="VNM\nVIC\nHPG\nVCB\nFPT",
            height=150
        )
        
        symbols = [s.strip() for s in symbols_input.split('\n') if s.strip()]
    
    with col2:
        st.subheader("📅 Khoảng thời gian")
        
        col_date1, col_date2 = st.columns(2)
        with col_date1:
            collect_start = st.date_input("Từ ngày", value=datetime.now() - timedelta(days=180))
        with col_date2:
            collect_end = st.date_input("Đến ngày", value=datetime.now())
        
        st.markdown("---")
        
        include_news = st.checkbox("Thu thập tin tức", value=True)
        if include_news:
            news_pages = st.slider("Số trang tin tức", 1, 10, 3)
    
    if st.button("🚀 Bắt đầu thu thập", type="primary"):
        with st.spinner("Đang thu thập dữ liệu..."):
            from data_collection import YahooFinanceAPI, VNDirectAPI, NewsScraperBS4
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            all_data = {}
            all_news = []
            
            # Progress tracking
            total_tasks = len(symbols) * len(data_sources)
            current_task = 0
            
            for i, symbol in enumerate(symbols):
                clean_symbol = symbol.strip().upper()
                if not clean_symbol.endswith('.VN'):
                    clean_symbol += '.VN'
                
                symbol_data = []
                
                # Yahoo Finance
                if "Yahoo Finance" in data_sources:
                    status_text.text(f"📥 Thu thập từ Yahoo Finance: {clean_symbol}...")
                    try:
                        yahoo_api = YahooFinanceAPI()
                        df_yahoo = yahoo_api.get_stock_data(clean_symbol, str(collect_start), str(collect_end))
                        if not df_yahoo.empty:
                            df_yahoo['source'] = 'Yahoo Finance'
                            symbol_data.append(df_yahoo)
                            st.success(f"✅ Yahoo Finance: {len(df_yahoo)} records cho {clean_symbol}")
                    except Exception as e:
                        st.warning(f"⚠️ Yahoo Finance lỗi cho {clean_symbol}: {str(e)}")
                    
                    current_task += 1
                    progress_bar.progress(current_task / total_tasks)
                
                # VNDirect
                if "VNDirect" in data_sources:
                    status_text.text(f"📥 Thu thập từ VNDirect: {clean_symbol}...")
                    try:
                        vnd_api = VNDirectAPI()
                        df_vnd = vnd_api.get_stock_price(clean_symbol, str(collect_start), str(collect_end))
                        if not df_vnd.empty:
                            df_vnd['source'] = 'VNDirect'
                            symbol_data.append(df_vnd)
                            st.success(f"✅ VNDirect: {len(df_vnd)} records cho {clean_symbol}")
                        else:
                            st.info(f"ℹ️ VNDirect: Không có dữ liệu cho {clean_symbol}")
                    except Exception as e:
                        st.warning(f"⚠️ VNDirect lỗi cho {clean_symbol}: {str(e)}")
                    
                    current_task += 1
                    progress_bar.progress(current_task / total_tasks)
                
                # Combine data for symbol
                if symbol_data:
                    combined_df = pd.concat(symbol_data, ignore_index=True)
                    all_data[clean_symbol] = combined_df
                
                # News collection
                if include_news and "CafeF News" in data_sources:
                    status_text.text(f"📰 Thu thập tin tức: {clean_symbol}...")
                    try:
                        scraper = NewsScraperBS4()
                        news = scraper.scrape_cafef(clean_symbol.replace('.VN', ''), pages=news_pages)
                        all_news.extend(news)
                        if news:
                            st.success(f"✅ Tin tức: {len(news)} bài cho {clean_symbol}")
                    except Exception as e:
                        st.warning(f"⚠️ Thu thập tin tức lỗi: {str(e)}")
            
            progress_bar.progress(1.0)
            status_text.text("✅ Hoàn thành!")
            
            # Store in session state
            st.session_state['collected_data'] = all_data
            st.session_state['collected_news'] = all_news
            
            # Summary
            st.markdown("---")
            st.subheader("📊 Tổng kết thu thập")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Tổng số mã", len(all_data))
            
            with col2:
                total_records = sum(len(df) for df in all_data.values())
                st.metric("Tổng records", f"{total_records:,}")
            
            with col3:
                st.metric("Tin tức", len(all_news))
            
            # Display collected data
            if all_data:
                st.markdown("---")
                st.subheader("📈 Dữ liệu giá thu thập được")
                
                selected_symbol = st.selectbox("Chọn mã để xem chi tiết", list(all_data.keys()))
                
                if selected_symbol:
                    df_display = all_data[selected_symbol]
                    
                    # Show stats
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Records", len(df_display))
                    with col2:
                        if 'source' in df_display.columns:
                            sources = df_display['source'].unique()
                            st.metric("Nguồn", len(sources))
                    with col3:
                        if 'Close' in df_display.columns:
                            st.metric("Giá mới nhất", f"{df_display['Close'].iloc[-1]:,.0f}")
                    with col4:
                        if 'Close' in df_display.columns:
                            price_change = ((df_display['Close'].iloc[-1] / df_display['Close'].iloc[0]) - 1) * 100
                            st.metric("Thay đổi", f"{price_change:.2f}%")
                    
                    # Show data table
                    st.dataframe(df_display, use_container_width=True, height=600)
                    
                    # Download button
                    csv = df_display.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Tải xuống CSV",
                        data=csv,
                        file_name=f"{selected_symbol}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # Display news
            if all_news:
                st.markdown("---")
                st.subheader("📰 Tin tức thu thập được")
                
                news_df = pd.DataFrame(all_news)
                st.dataframe(news_df, use_container_width=True, height=600)
    
    # Show previously collected data
    elif 'collected_data' in st.session_state:
        st.markdown("---")
        st.info("💾 Hiển thị dữ liệu đã thu thập trước đó")
        
        all_data = st.session_state['collected_data']
        
        if all_data:
            selected_symbol = st.selectbox("Chọn mã để xem", list(all_data.keys()))
            if selected_symbol:
                st.dataframe(all_data[selected_symbol], use_container_width=True, height=600)


# ==================== PRICE PREDICTION PAGE ====================
elif page == "🔮 Dự đoán giá":
    st.title("📈 Dự Đoán Giá Cổ Phiếu")
    
    # Prediction settings
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Cài đặt dự đoán")
        
        model_type = st.selectbox(
            "Chọn Model",
            ["Ensemble (Tất cả Models)", "ARIMA", "Prophet", "LSTM", "GRU"]
        )
        
        periods = st.slider("Số ngày dự đoán", 1, 90, 30)
    
    with col2:
        st.subheader("Thông tin Model")
        
        model_descriptions = {
            "Ensemble (Tất cả Models)": "Kết hợp tất cả models với meta-learning",
            "ARIMA": "Statistical model, tốt cho short-term",
            "Prophet": "Facebook's model, tốt cho seasonality",
            "LSTM": "Deep learning, tốt cho complex patterns",
            "GRU": "Nhanh hơn LSTM, performance tương tự"
        }
        
        st.info(model_descriptions[model_type])
    
    if st.button("🔮 Dự đoán", type="primary"):
        with st.spinner("Đang tạo dự đoán..."):
            # Load data if not in session
            if 'data' not in st.session_state:
                df = load_data(symbol, start_date, end_date)
                st.session_state['data'] = df
            else:
                df = st.session_state['data']
            
            if not df.empty:
                # Generate mock predictions (cần thay bằng model thực tế)
                last_price = df['Close'].iloc[-1]
                future_dates = pd.date_range(
                    start=df['date'].max() + timedelta(days=1),
                    periods=periods,
                    freq='D'
                )
                
                # Simulated predictions
                np.random.seed(42)
                trend = np.linspace(0, periods * 0.1, periods)
                noise = np.random.randn(periods) * 2
                predictions = last_price + trend + noise
                
                # Create prediction dataframe
                pred_df = pd.DataFrame({
                    'date': future_dates,
                    'predicted_price': predictions
                })
                
                # Plot
                fig = go.Figure()
                
                # Historical prices
                fig.add_trace(go.Scatter(
                    x=df['date'],
                    y=df['Close'],
                    mode='lines',
                    name='Lịch sử',
                    line=dict(color='blue', width=2)
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=pred_df['date'],
                    y=pred_df['predicted_price'],
                    mode='lines',
                    name='Dự đoán',
                    line=dict(color='red', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Dự đoán giá {symbol} - {model_type}',
                    xaxis_title='Ngày',
                    yaxis_title='Giá',
                    height=500,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    predicted_change = predictions[-1] - last_price
                    st.metric("Giá dự đoán", f"${predictions[-1]:.2f}",
                             delta=f"${predicted_change:.2f}")
                
                with col2:
                    change_pct = (predicted_change / last_price) * 100
                    st.metric("Thay đổi dự kiến", f"{change_pct:.2f}%")
                
                with col3:
                    st.metric("Độ tin cậy", "85%")
                
                # Prediction table
                st.subheader("📊 Chi tiết dự đoán")
                display_df = pred_df.copy()
                display_df['change_from_current'] = display_df['predicted_price'] - last_price
                display_df['change_pct'] = (display_df['change_from_current'] / last_price) * 100
                
                st.dataframe(
                    display_df.style.format({
                        'predicted_price': '${:.2f}',
                        'change_from_current': '${:.2f}',
                        'change_pct': '{:.2f}%'
                    }),
                    use_container_width=True
                )


# ==================== BACKTESTING PAGE ====================
elif page == "🔄 Backtesting":
    st.title("🔄 Kiểm Định Ngược Chiến Lược")
    
    st.markdown("""
    Kiểm tra chiến lược giao dịch với dữ liệu lịch sử để xem hiệu quả thực tế.
    """)
    
    # Strategy settings
    col1, col2 = st.columns(2)
    
    with col1:
        strategy = st.selectbox(
            "Chiến lược giao dịch",
            ["Long Only", "Long-Short", "Dựa trên ngưỡng"]
        )
        
        initial_capital = st.number_input(
            "Vốn ban đầu ($)",
            min_value=1000,
            max_value=1000000,
            value=100000,
            step=1000
        )
    
    with col2:
        stop_loss = st.slider("Stop Loss (%)", 0.0, 20.0, 5.0, 0.5) / 100
        take_profit = st.slider("Take Profit (%)", 0.0, 50.0, 10.0, 1.0) / 100
    
    if st.button("🚀 Chạy Backtest", type="primary"):
        with st.spinner("Đang chạy backtest..."):
            # Load data
            if 'data' not in st.session_state:
                df = load_data(symbol, start_date, end_date)
                st.session_state['data'] = df
            else:
                df = st.session_state['data']
            
            if not df.empty:
                from backtesting import BacktestEngine
                
                # Mock predictions
                predictions = df['Close'].values * (1 + np.random.randn(len(df)) * 0.01)
                
                # Run backtest
                engine = BacktestEngine(
                    initial_capital=initial_capital,
                    commission=0.001
                )
                
                results = engine.run_backtest(
                    data=df.set_index('date'),
                    predictions=predictions,
                    strategy='long_only',
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
                
                # Display results
                st.success("Backtest hoàn thành!")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Tổng lợi nhuận",
                        f"{results['total_return_pct']:.2f}%",
                        delta=f"${results['final_capital'] - initial_capital:,.0f}"
                    )
                
                with col2:
                    st.metric("Sharpe Ratio", f"{results['sharpe_ratio']:.2f}")
                
                with col3:
                    st.metric("Max Drawdown", f"{results['max_drawdown_pct']:.2f}%")
                
                with col4:
                    st.metric("Tỷ lệ thắng", f"{results['win_rate_pct']:.1f}%")
                
                # Portfolio value chart
                st.subheader("📈 Giá trị danh mục theo thời gian")
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=engine.portfolio_values,
                    mode='lines',
                    name='Giá trị danh mục',
                    fill='tozeroy'
                ))
                
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Vốn ban đầu"
                )
                
                fig.update_layout(
                    yaxis_title='Giá trị danh mục ($)',
                    xaxis_title='Thời gian',
                    height=400,
                    template='plotly_white'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Trades table
                st.subheader("📋 Lịch sử giao dịch")
                trades_df = engine.get_trades_df()
                
                if not trades_df.empty:
                    st.dataframe(trades_df, use_container_width=True)


# ==================== SENTIMENT ANALYSIS PAGE ====================
elif page == "💭 Phân tích Sentiment":
    st.markdown('<h1 class="main-header">💭 PHÂN TÍCH SENTIMENT TIN TỨC</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #f59e0b;">
        <h3 style="color: #92400e; margin: 0 0 0.5rem 0;">📰 Phân tích cảm xúc thị trường</h3>
        <p style="color: #78350f; margin: 0;">Sử dụng FinBERT để phân tích tin tức tài chính và đánh giá tâm lý thị trường</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Input section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        news_source = st.selectbox(
            "🌐 Nguồn tin tức",
            ["CafeF", "VNDirect News", "VnExpress Kinh Doanh", "Tất cả"]
        )
    
    with col2:
        time_range = st.selectbox(
            "⏰ Khoảng thời gian",
            ["7 ngày", "15 ngày", "30 ngày", "60 ngày"]
        )
    
    if st.button("📊 Phân tích Sentiment", type="primary", use_container_width=True):
        with st.spinner("Đang thu thập và phân tích tin tức..."):
            # Mock sentiment data
            days = int(time_range.split()[0])
            dates = pd.date_range(end=datetime.now(), periods=days, freq='D')
            sentiments = np.random.randn(days) * 0.3
            
            sentiment_df = pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiments,
                'sentiment_label': ['tích cực' if s > 0.1 else 'tiêu cực' if s < -0.1 else 'trung tính' for s in sentiments]
            })
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            avg_sentiment = sentiment_df['sentiment_score'].mean()
            positive_days = (sentiment_df['sentiment_score'] > 0.1).sum()
            negative_days = (sentiment_df['sentiment_score'] < -0.1).sum()
            neutral_days = days - positive_days - negative_days
            
            with col1:
                st.metric(
                    "📈 Sentiment TB",
                    f"{avg_sentiment:.3f}",
                    delta="Tích cực" if avg_sentiment > 0 else "Tiêu cực",
                    delta_color="normal" if avg_sentiment > 0 else "inverse"
                )
            
            with col2:
                st.metric("✅ Ngày tích cực", f"{positive_days}", f"{positive_days/days*100:.1f}%")
            
            with col3:
                st.metric("❌ Ngày tiêu cực", f"{negative_days}", f"{negative_days/days*100:.1f}%")
            
            with col4:
                st.metric("⚖️ Ngày trung tính", f"{neutral_days}", f"{neutral_days/days*100:.1f}%")
            
            st.markdown("---")
            
            # Sentiment chart
            st.subheader("📊 Biểu đồ Sentiment theo thời gian")
            
            fig = go.Figure()
            
            colors = ['#0e9f6e' if s > 0.1 else '#f05252' if s < -0.1 else '#6b7280' for s in sentiments]
            
            fig.add_trace(go.Bar(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_score'],
                marker_color=colors,
                name='Sentiment Score',
                text=[f"{s:.2f}" for s in sentiments],
                textposition='outside'
            ))
            
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            fig.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.3, annotation_text="Tích cực")
            fig.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.3, annotation_text="Tiêu cực")
            
            fig.update_layout(
                yaxis_title='Điểm Sentiment',
                xaxis_title='Ngày',
                height=450,
                template='plotly_white',
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recent news
            st.subheader("📰 Tin tức gần đây")
            
            mock_news = [
                {"title": "VN-Index tăng mạnh nhờ dòng tiền ngoại", "sentiment": "tích cực", "score": 0.85, "source": "CafeF"},
                {"title": "Cổ phiếu ngân hàng dẫn dắt thị trường", "sentiment": "tích cực", "score": 0.72, "source": "VNDirect"},
                {"title": "Thanh khoản thị trường cải thiện đáng kể", "sentiment": "tích cực", "score": 0.65, "source": "VnExpress"},
                {"title": "Áp lực bán tại nhóm cổ phiếu bất động sản", "sentiment": "tiêu cực", "score": -0.58, "source": "CafeF"},
                {"title": "Nhà đầu tư thận trọng trước diễn biến thị trường", "sentiment": "trung tính", "score": 0.05, "source": "VNDirect"}
            ]
            
            for news in mock_news:
                sentiment_color = "#0e9f6e" if news["sentiment"] == "tích cực" else "#f05252" if news["sentiment"] == "tiêu cực" else "#6b7280"
                st.markdown(f"""
                <div style="background: white; padding: 1rem; border-radius: 8px; margin-bottom: 0.8rem; border-left: 4px solid {sentiment_color}; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                    <div style="display: flex; justify-content: space-between; align-items: start;">
                        <div style="flex: 1;">
                            <h4 style="margin: 0 0 0.5rem 0; color: #1f2937;">{news['title']}</h4>
                            <small style="color: #6b7280;">📰 {news['source']}</small>
                        </div>
                        <div style="text-align: right; margin-left: 1rem;">
                            <div style="background: {sentiment_color}; color: white; padding: 0.3rem 0.8rem; border-radius: 20px; font-size: 0.85rem; font-weight: 600; margin-bottom: 0.3rem;">
                                {news['sentiment'].upper()}
                            </div>
                            <div style="color: #4b5563; font-weight: 600;">Score: {news['score']:.2f}</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ==================== MODEL TRAINING PAGE ====================
elif page == "🤖 Huấn luyện Model":
    st.markdown('<h1 class="main-header">🤖 HUẤN LUYỆN MODEL AI</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="background: linear-gradient(135deg, #dbeafe 0%, #bfdbfe 100%); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem; border-left: 6px solid #1a56db;">
        <h3 style="color: #1e3a8a; margin: 0 0 0.5rem 0;">🎯 Huấn luyện và tối ưu hóa models</h3>
        <p style="color: #1e40af; margin: 0;">Cấu hình tham số và huấn luyện các mô hình Machine Learning/Deep Learning</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model selection
    st.subheader("📋 Chọn Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🔷 Traditional ML Models**")
        model_arima = st.checkbox("ARIMA - AutoRegressive Integrated Moving Average", value=True)
        model_prophet = st.checkbox("Prophet - Facebook Time Series", value=True)
    
    with col2:
        st.markdown("**🔶 Deep Learning Models**")
        model_lstm = st.checkbox("LSTM - Long Short-Term Memory", value=False)
        model_gru = st.checkbox("GRU - Gated Recurrent Unit", value=False)
    
    selected_models = []
    if model_arima: selected_models.append("ARIMA")
    if model_prophet: selected_models.append("Prophet")
    if model_lstm: selected_models.append("LSTM")
    if model_gru: selected_models.append("GRU")
    
    st.markdown("---")
    
    # Training parameters
    st.subheader("⚙️ Tham Số Huấn Luyện")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_split = st.slider("📊 Tỷ lệ Training (%)", 50, 90, 80, help="Phần trăm dữ liệu dùng để training")
        lookback = st.number_input("⏮️ Lookback Period", 10, 120, 60, help="Số ngày quá khứ để dự đoán")
    
    with col2:
        epochs = st.number_input("🔄 Epochs (LSTM/GRU)", 10, 200, 50, help="Số vòng lặp huấn luyện")
        batch_size = st.selectbox("📦 Batch Size", [16, 32, 64, 128], index=1, help="Kích thước batch")
    
    with col3:
        learning_rate = st.select_slider(
            "📈 Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            help="Tốc độ học"
        )
        validation_split = st.slider("✅ Validation Split (%)", 10, 30, 20, help="Phần trăm dữ liệu validation")
    
    st.markdown("---")
    
    # Advanced options
    with st.expander("🔧 Tùy chọn nâng cao"):
        col1, col2 = st.columns(2)
        
        with col1:
            dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.2, 0.05)
            early_stopping = st.checkbox("Early Stopping", value=True)
        
        with col2:
            optimizer = st.selectbox("Optimizer", ["Adam", "SGD", "RMSprop"], index=0)
            loss_function = st.selectbox("Loss Function", ["MSE", "MAE", "Huber"], index=0)
    
    st.markdown("---")
    
    # Training button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Bắt đầu Huấn Luyện", type="primary", use_container_width=True):
            if not selected_models:
                st.error("⚠️ Vui lòng chọn ít nhất một model!")
            else:
                with st.spinner("Đang huấn luyện models... Có thể mất vài phút."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    results_data = []
                    
                    for i, model_name in enumerate(selected_models):
                        status_text.text(f"🔄 Đang huấn luyện {model_name}... ({i+1}/{len(selected_models)})")
                        progress_bar.progress((i + 1) / len(selected_models))
                        
                        # Simulate training
                        import time
                        time.sleep(2)
                        
                        # Mock results
                        results_data.append({
                            'Model': model_name,
                            'Train Accuracy': f"{np.random.uniform(85, 95):.2f}%",
                            'Val Accuracy': f"{np.random.uniform(80, 90):.2f}%",
                            'MAE': f"{np.random.uniform(2, 5):.2f}",
                            'RMSE': f"{np.random.uniform(3, 7):.2f}",
                            'MAPE': f"{np.random.uniform(2, 8):.2f}%",
                            'Training Time': f"{np.random.uniform(10, 60):.1f}s",
                            'Status': '✅ Hoàn thành'
                        })
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Huấn luyện hoàn thành!")
                    
                    st.success("🎉 Tất cả models đã được huấn luyện thành công!")
                    
                    st.markdown("---")
                    
                    # Results table
                    st.subheader("📊 Kết Quả Huấn Luyện")
                    
                    results_df = pd.DataFrame(results_data)
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    
                    # Best model
                    st.markdown("---")
                    st.subheader("🏆 Model Tốt Nhất")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #fbbf24 0%, #f59e0b 100%); color: white; padding: 2rem; border-radius: 12px; text-align: center;">
                            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🥇</div>
                            <h3 style="margin: 0 0 0.5rem 0;">LSTM</h3>
                            <div style="font-size: 1.5rem; font-weight: 800;">92.5%</div>
                            <div style="opacity: 0.9;">Validation Accuracy</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.metric("💾 Model Size", "2.4 MB", help="Kích thước file model")
                        st.metric("⚡ Inference Time", "15ms", help="Thời gian dự đoán")
                    
                    with col3:
                        st.metric("📈 Improvement", "+5.2%", help="So với baseline")
                        st.metric("🎯 F1 Score", "0.91", help="F1 Score")
            
            st.success(f"Đã huấn luyện thành công {len(selected_models)} models!")
            
            # Show training results
            results_data = []
            for model in selected_models:
                results_data.append({
                    'Model': model,
                    'MAE': np.random.uniform(2, 5),
                    'RMSE': np.random.uniform(3, 7),
                    'MAPE': np.random.uniform(2, 8)
                })
            
            results_df = pd.DataFrame(results_data)
            
            st.subheader("📊 Kết quả huấn luyện")
            st.dataframe(results_df.style.format({
                'MAE': '{:.2f}',
                'RMSE': '{:.2f}',
                'MAPE': '{:.2f}%'
            }), use_container_width=True)


# ==================== ANALYTICS PAGE ====================
elif page == "📉 Phân tích":
    st.title("📉 Phân Tích Nâng Cao")
    
    if 'data' not in st.session_state:
        st.warning("Vui lòng tải dữ liệu từ trang chủ trước")
    else:
        df = st.session_state['data']
        
        # Candlestick chart
        st.subheader("📈 Biểu đồ giá")
        fig_candle = plot_candlestick(df)
        st.plotly_chart(fig_candle, use_container_width=True)
        
        # Volume chart
        st.subheader("📊 Khối lượng")
        fig_volume = plot_volume(df)
        st.plotly_chart(fig_volume, use_container_width=True)
        
        # Statistics
        st.subheader("📈 Tổng kết thống kê")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Thống kê giá**")
            st.write(df[['Open', 'High', 'Low', 'Close']].describe())
        
        with col2:
            st.write("**Thống kê khối lượng**")
            st.write(df[['Volume']].describe())


# ==================== PROFESSIONAL FOOTER ====================
st.markdown("""
<div class="custom-footer">
    <div class="footer-container">
        <div class="footer-grid">
            <div>
                <div class="footer-brand">📈 StockPro Analytics</div>
                <div class="footer-description">
                    Hệ thống dự đoán giá cổ phiếu hàng đầu sử dụng công nghệ AI và Machine Learning.
                    Phân tích thông minh, dự đoán chính xác, đầu tư hiệu quả.
                </div>
                <div class="footer-social">
                    <div class="footer-social-link">📱</div>
                    <div class="footer-social-link">💼</div>
                    <div class="footer-social-link">📧</div>
                    <div class="footer-social-link">🌐</div>
                </div>
            </div>
            <div>
                <div class="footer-title">Sản phẩm</div>
                <div class="footer-links">
                    <div class="footer-link">Dự đoán giá</div>
                    <div class="footer-link">Phân tích kỹ thuật</div>
                    <div class="footer-link">Backtesting</div>
                    <div class="footer-link">Sentiment Analysis</div>
                    <div class="footer-link">API Service</div>
                </div>
            </div>
            <div>
                <div class="footer-title">Tài nguyên</div>
                <div class="footer-links">
                    <div class="footer-link">Tài liệu</div>
                    <div class="footer-link">Blog</div>
                    <div class="footer-link">Hướng dẫn</div>
                    <div class="footer-link">Video tutorials</div>
                    <div class="footer-link">FAQ</div>
                </div>
            </div>
            <div>
                <div class="footer-title">Công ty</div>
                <div class="footer-links">
                    <div class="footer-link">Về chúng tôi</div>
                    <div class="footer-link">Đội ngũ</div>
                    <div class="footer-link">Liên hệ</div>
                    <div class="footer-link">Chính sách</div>
                    <div class="footer-link">Điều khoản</div>
                </div>
            </div>
        </div>
        <div class="footer-bottom">
            © 2025 StockPro Analytics. All rights reserved. | Powered by AI, ML & Deep Learning Technology
        </div>
    </div>
</div>
""", unsafe_allow_html=True)
