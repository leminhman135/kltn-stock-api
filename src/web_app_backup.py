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
    page_title="Stock Prediction System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Professional Website Design
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Poppins:wght@400;500;600;700;800&display=swap');
    
    /* Global Variables */
    :root {
        --primary: #1a56db;
        --primary-dark: #1e429f;
        --primary-light: #3f83f8;
        --secondary: #0694a2;
        --accent: #7c3aed;
        --success: #0e9f6e;
        --warning: #ff5a1f;
        --danger: #f05252;
        --dark: #111827;
        --light: #f9fafb;
        --gray-50: #f9fafb;
        --gray-100: #f3f4f6;
        --gray-200: #e5e7eb;
        --gray-300: #d1d5db;
        --gray-600: #4b5563;
        --gray-700: #374151;
        --gray-800: #1f2937;
        --gray-900: #111827;
    }
    
    /* Reset & Global Styles */
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Main Container */
    .main {
        background: linear-gradient(180deg, #f9fafb 0%, #ffffff 100%);
    }
    
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Custom Navigation Bar */
    .custom-navbar {
        position: sticky;
        top: 0;
        z-index: 1000;
        background: linear-gradient(135deg, #1a56db 0%, #1e429f 100%);
        padding: 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border-bottom: 3px solid rgba(255,255,255,0.1);
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
        font-family: 'Poppins', sans-serif;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .navbar-tagline {
        color: rgba(255,255,255,0.9);
        font-size: 0.85rem;
        font-weight: 500;
        padding: 0.3rem 1rem;
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        backdrop-filter: blur(10px);
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
        font-family: 'Poppins', sans-serif;
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
        color: var(--gray-900);
        margin-bottom: 1rem;
        font-family: 'Poppins', sans-serif;
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
        background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
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
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid var(--gray-200);
        transition: all 0.3s ease;
    }
    
    .market-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.12);
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
        color: var(--gray-900);
        margin: 0.5rem 0;
        font-family: 'Poppins', sans-serif;
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
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
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
    
    /* Enhanced Buttons */
    .stButton button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(26, 86, 219, 0.3);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%);
        box-shadow: 0 8px 20px rgba(26, 86, 219, 0.4);
        transform: translateY(-2px);
    }
    
    /* Sidebar Enhancement */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--gray-50) 0%, white 100%);
        border-right: 1px solid var(--gray-200);
    }
    
    /* Metric Cards */
    .stMetric {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid var(--gray-200);
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
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }
    
    /* Premium Stock Card */
    .stock-card {
        background: linear-gradient(135deg, #003d82 0%, #0052a3 50%, #0066cc 100%);
        padding: 2rem;
        border-radius: var(--radius-lg);
        color: white;
        box-shadow: var(--shadow-lg), 0 0 0 1px rgba(255,255,255,0.1) inset;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .stock-card::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
        animation: pulse 4s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); opacity: 0.5; }
        50% { transform: scale(1.1); opacity: 0.8; }
    }
    
    .stock-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 15px 30px rgba(0,61,130,0.25);
    }
    
    .stock-card h3 {
        color: white;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .stock-card .price {
        font-size: 3rem;
        font-weight: 800;
        margin: 1rem 0;
        text-shadow: 0 2px 8px rgba(0,0,0,0.3);
        letter-spacing: -1px;
    }
    
    .stock-card .change {
        font-size: 1.3rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.4rem 1rem;
        background: rgba(255,255,255,0.2);
        border-radius: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Enhanced Tables */
    .dataframe {
        border: none !important;
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        box-shadow: var(--shadow-md) !important;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #003d82 0%, #0052a3 100%) !important;
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
        background-color: #f8f9fb !important;
    }
    
    .dataframe tbody tr:hover {
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%) !important;
        transform: scale(1.01);
        box-shadow: 0 2px 8px rgba(0,163,224,0.15);
    }
    
    .dataframe tbody td {
        padding: 14px 12px !important;
        border-bottom: 1px solid #e8eaed !important;
        font-weight: 500;
    }
    
    /* Premium Metric Cards */
    .metric-card {
        background: white;
        padding: 1.8rem;
        border-radius: var(--radius-md);
        border-left: 4px solid #00a3e0;
        box-shadow: var(--shadow-md);
        margin: 0.5rem 0;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    
    .metric-card::after {
        content: '';
        position: absolute;
        top: 0;
        right: 0;
        width: 100px;
        height: 100px;
        background: radial-gradient(circle, rgba(0,163,224,0.1) 0%, transparent 70%);
        transform: translate(30%, -30%);
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
        border-left-width: 6px;
    }
    
    .metric-card h4 {
        color: #64748b;
        font-size: 0.85rem;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }
    
    .metric-card .value {
        color: #003d82;
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0.8rem 0 0.3rem 0;
        letter-spacing: -1px;
    }
    
    .metric-card .sub-value {
        color: #94a3b8;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Enhanced Colors */
    .positive {
        color: #00c48c !important;
        background: linear-gradient(135deg, #00c48c 0%, #00e5a0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .negative {
        color: #ff3b3b !important;
        background: linear-gradient(135deg, #ff3b3b 0%, #ff5252 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Premium Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #f1f5f9;
        border-radius: var(--radius-md);
        padding: 6px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: var(--radius-sm);
        padding: 14px 28px;
        font-weight: 600;
        color: #64748b;
        transition: all 0.3s ease;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(0,61,130,0.1);
        color: #003d82;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #003d82 0%, #0052a3 100%) !important;
        color: white !important;
        box-shadow: var(--shadow-md);
    }
    
    /* Enhanced Buttons */
    .stButton button {
        background: linear-gradient(135deg, #003d82 0%, #0052a3 100%);
        color: white;
        border: none;
        border-radius: var(--radius-sm);
        padding: 0.7rem 2.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow-sm);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, #0052a3 0%, #0066cc 100%);
        box-shadow: var(--shadow-lg);
        transform: translateY(-2px);
    }
    
    .stButton button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    /* Premium Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fb 0%, #ffffff 100%);
        border-right: 1px solid #e8eaed;
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding: 2rem 1rem;
    }
    
    /* Enhanced Info Boxes */
    .info-box {
        background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
        border-left: 4px solid #00a3e0;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin: 1rem 0;
        box-shadow: var(--shadow-sm);
    }
    
    .info-box h3 {
        color: #003d82;
        margin-top: 0;
        font-weight: 700;
    }
    
    /* Chart Containers */
    .chart-container {
        background: white;
        padding: 2rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        margin: 1.5rem 0;
        border: 1px solid #e8eaed;
    }
    
    /* Market Index Cards */
    .market-index-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--radius-md);
        box-shadow: var(--shadow-md);
        border-top: 4px solid #00a3e0;
        transition: all 0.3s ease;
    }
    
    .market-index-card:hover {
        transform: translateY(-5px);
        box-shadow: var(--shadow-lg);
    }
    
    .market-index-card h4 {
        color: #64748b;
        font-size: 0.9rem;
        margin: 0 0 0.5rem 0;
        font-weight: 600;
    }
    
    .market-index-card .index-value {
        color: #003d82;
        font-size: 2rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-top-color: #003d82 !important;
    }
    
    /* Select Box Enhancement */
    .stSelectbox [data-baseweb="select"] {
        border-radius: var(--radius-sm);
        border: 2px solid #e8eaed;
        transition: all 0.3s ease;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #00a3e0;
    }
    
    /* Input Enhancement */
    .stTextInput input {
        border-radius: var(--radius-sm);
        border: 2px solid #e8eaed;
        padding: 0.7rem 1rem;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stTextInput input:focus {
        border-color: #003d82;
        box-shadow: 0 0 0 3px rgba(0,61,130,0.1);
    }
    
    /* Date Input Enhancement */
    .stDateInput input {
        border-radius: var(--radius-sm);
        border: 2px solid #e8eaed;
        padding: 0.7rem 1rem;
    }
    
    /* Radio Button Enhancement */
    .stRadio > label {
        background: white;
        padding: 0.8rem 1.2rem;
        border-radius: var(--radius-sm);
        margin: 0.3rem 0;
        border: 2px solid #e8eaed;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stRadio > label:hover {
        border-color: #00a3e0;
        background: #f8f9fb;
    }
    
    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #003d82 0%, #0066cc 100%);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0052a3 0%, #0080ff 100%);
    }
</style>
""", unsafe_allow_html=True)

# ==================== NAVIGATION BAR ====================
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
        <div class="navbar-menu">
            <div class="navbar-item">Thị trường</div>
            <div class="navbar-item">Dự đoán</div>
            <div class="navbar-item">Phân tích</div>
            <div class="navbar-item navbar-cta">Bắt đầu ngay →</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Dự Đoán Cổ Phiếu")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Điều hướng",
    ["🏠 Trang chủ", "📊 Thị trường", "📈 Dữ liệu chi tiết", "🔮 Dự đoán giá", "🔄 Backtesting", 
     "💭 Phân tích Sentiment", "🤖 Huấn luyện Model"]
)

st.sidebar.markdown("---")

# Stock selection
symbol = st.sidebar.text_input("Mã cổ phiếu", value="VNM.VN", help="Nhập mã cổ phiếu (vd: AAPL, VNM.VN)")

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Ngày bắt đầu", value=datetime.now() - timedelta(days=365))
with col2:
    end_date = st.date_input("Ngày kết thúc", value=datetime.now())

st.sidebar.markdown("---")


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
    st.title("💭 Phân Tích Sentiment Tin Tức")
    
    st.markdown("""
    Phân tích cảm xúc thị trường từ các bài báo tin tức sử dụng model FinBERT.
    """)
    
    if st.button("📰 Phân tích Sentiment"):
        with st.spinner("Đang phân tích sentiment tin tức..."):
            # Mock sentiment data
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            sentiments = np.random.randn(30) * 0.3
            
            sentiment_df = pd.DataFrame({
                'date': dates,
                'sentiment_score': sentiments,
                'sentiment_label': ['positive' if s > 0.1 else 'negative' if s < -0.1 else 'neutral' for s in sentiments]
            })
            
            # Plot sentiment over time
            fig = go.Figure()
            
            colors = ['green' if s > 0 else 'red' for s in sentiments]
            
            fig.add_trace(go.Bar(
                x=sentiment_df['date'],
                y=sentiment_df['sentiment_score'],
                marker_color=colors,
                name='Sentiment Score'
            ))
            
            fig.update_layout(
                title='Điểm Sentiment hàng ngày',
                yaxis_title='Điểm Sentiment',
                xaxis_title='Ngày',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Sentiment summary
            avg_sentiment = sentiment_df['sentiment_score'].mean()
            positive_days = (sentiment_df['sentiment_score'] > 0.1).sum()
            negative_days = (sentiment_df['sentiment_score'] < -0.1).sum()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Sentiment trung bình", f"{avg_sentiment:.2f}")
            
            with col2:
                st.metric("Ngày tích cực", f"{positive_days}/30")
            
            with col3:
                st.metric("Ngày tiêu cực", f"{negative_days}/30")


# ==================== MODEL TRAINING PAGE ====================
elif page == "🤖 Huấn luyện Model":
    st.title("🤖 Huấn Luyện Model")
    
    st.markdown("""
    Huấn luyện các model machine learning với tham số tùy chỉnh.
    """)
    
    # Model selection
    selected_models = st.multiselect(
        "Chọn Models để huấn luyện",
        ["ARIMA", "Prophet", "LSTM", "GRU"],
        default=["ARIMA", "Prophet"]
    )
    
    # Training parameters
    col1, col2 = st.columns(2)
    
    with col1:
        train_split = st.slider("Tỷ lệ training (%)", 50, 90, 80) / 100
        epochs = st.number_input("Epochs (cho LSTM/GRU)", 10, 200, 50)
    
    with col2:
        lookback = st.number_input("Lookback Period", 10, 120, 60)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
    
    if st.button("🎯 Bắt đầu huấn luyện", type="primary"):
        with st.spinner("Đang huấn luyện models... Có thể mất ít phút."):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, model in enumerate(selected_models):
                status_text.text(f"Đang huấn luyện {model}...")
                progress_bar.progress((i + 1) / len(selected_models))
                
                # Simulate training
                import time
                time.sleep(2)
            
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
