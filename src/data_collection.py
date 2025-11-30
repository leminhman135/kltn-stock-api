"""
Module thu thập dữ liệu từ nhiều nguồn khác nhau:
- API: Yahoo Finance, Alpha Vantage, VNDirect
- Web Scraping: BeautifulSoup, Scrapy
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import time
import logging
import json

# Optional imports - only load if available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    yf = None
    YFINANCE_AVAILABLE = False
    
try:
    from scrapy import Spider, Request
    from scrapy.crawler import CrawlerProcess
    SCRAPY_AVAILABLE = True
except ImportError:
    Spider = None
    Request = None
    CrawlerProcess = None
    SCRAPY_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YahooFinanceAPI:
    """Thu thập dữ liệu giá cổ phiếu từ Yahoo Finance"""
    
    def __init__(self):
        if not YFINANCE_AVAILABLE:
            logger.warning("yfinance not installed. Yahoo Finance API will not work.")
        self.session = requests.Session()
    
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, 
                       interval: str = '1d') -> pd.DataFrame:
        """
        Lấy dữ liệu giá cổ phiếu
        
        Args:
            symbol: Mã cổ phiếu (vd: 'AAPL', 'VNM.VN')
            start_date: Ngày bắt đầu (YYYY-MM-DD)
            end_date: Ngày kết thúc (YYYY-MM-DD)
            interval: Khoảng thời gian (1d, 1h, 1m, etc.)
        
        Returns:
            DataFrame với các cột: Open, High, Low, Close, Volume
        """
        if not YFINANCE_AVAILABLE:
            logger.error("yfinance not installed. Cannot fetch data from Yahoo Finance.")
            return pd.DataFrame()
            
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            
            if df.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Chuẩn hóa tên cột
            df.reset_index(inplace=True)
            df.rename(columns={'Date': 'date'}, inplace=True)
            df['symbol'] = symbol
            
            logger.info(f"Successfully fetched {len(df)} records for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(self, symbols: List[str], start_date: str, 
                           end_date: str) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu cho nhiều mã cổ phiếu"""
        results = {}
        for symbol in symbols:
            df = self.get_stock_data(symbol, start_date, end_date)
            if not df.empty:
                results[symbol] = df
            time.sleep(0.5)  # Tránh rate limiting
        return results


class AlphaVantageAPI:
    """Thu thập dữ liệu từ Alpha Vantage API"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
    
    def get_daily_data(self, symbol: str, outputsize: str = 'full') -> pd.DataFrame:
        """Lấy dữ liệu giá hàng ngày"""
        try:
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': self.api_key,
                'outputsize': outputsize
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'Time Series (Daily)' not in data:
                logger.error(f"Error: {data.get('Note', 'Unknown error')}")
                return pd.DataFrame()
            
            df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume'
            })
            df = df.astype(float)
            df['symbol'] = symbol
            
            return df
        
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
            return pd.DataFrame()
    
    def get_news_sentiment(self, tickers: str) -> pd.DataFrame:
        """Lấy tin tức và phân tích cảm tính"""
        try:
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': tickers,
                'apikey': self.api_key
            }
            
            response = requests.get(self.base_url, params=params)
            data = response.json()
            
            if 'feed' not in data:
                return pd.DataFrame()
            
            news_data = []
            for item in data['feed']:
                news_data.append({
                    'title': item.get('title'),
                    'url': item.get('url'),
                    'time_published': item.get('time_published'),
                    'summary': item.get('summary'),
                    'sentiment_score': item.get('overall_sentiment_score'),
                    'sentiment_label': item.get('overall_sentiment_label')
                })
            
            return pd.DataFrame(news_data)
        
        except Exception as e:
            logger.error(f"Error fetching news sentiment: {str(e)}")
            return pd.DataFrame()


class NewsScraperBS4:
    """Web scraper sử dụng BeautifulSoup để thu thập tin tức tài chính"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_cafef(self, symbol: str, pages: int = 5) -> List[Dict]:
        """Scrape tin tức từ CafeF.vn"""
        news_list = []
        base_url = f"https://cafef.vn/timeline/{symbol.lower()}.chn"
        
        try:
            for page in range(1, pages + 1):
                url = f"{base_url}?pageindex={page}"
                response = requests.get(url, headers=self.headers)
                
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                articles = soup.find_all('div', class_='timeline-head')
                
                for article in articles:
                    try:
                        title_elem = article.find('h3')
                        link_elem = article.find('a')
                        time_elem = article.find('span', class_='time')
                        
                        if title_elem and link_elem:
                            news_list.append({
                                'symbol': symbol,
                                'title': title_elem.text.strip(),
                                'url': link_elem.get('href'),
                                'date': time_elem.text.strip() if time_elem else '',
                                'source': 'CafeF'
                            })
                    except Exception as e:
                        continue
                
                time.sleep(1)  # Tránh spam requests
            
            logger.info(f"Scraped {len(news_list)} articles for {symbol} from CafeF")
            return news_list
        
        except Exception as e:
            logger.error(f"Error scraping CafeF: {str(e)}")
            return news_list
    
    def scrape_vndirect_news(self, symbol: str) -> List[Dict]:
        """Scrape tin tức từ VNDirect"""
        news_list = []
        url = f"https://dchart.vndirect.com.vn/events/{symbol}"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                # Parse structure tùy theo website cụ thể
                # Đây là template, cần điều chỉnh theo HTML thực tế
                
            return news_list
        
        except Exception as e:
            logger.error(f"Error scraping VNDirect: {str(e)}")
            return news_list
    
    def get_article_content(self, url: str) -> str:
        """Lấy nội dung chi tiết của bài viết"""
        try:
            response = requests.get(url, headers=self.headers)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Tìm nội dung chính (cần điều chỉnh selector theo từng website)
            content_div = soup.find('div', class_='article-content')
            
            if content_div:
                paragraphs = content_div.find_all('p')
                content = ' '.join([p.text.strip() for p in paragraphs])
                return content
            
            return ""
        
        except Exception as e:
            logger.error(f"Error getting article content: {str(e)}")
            return ""


# Only define Scrapy spider if scrapy is available
if SCRAPY_AVAILABLE:
    class NewsScraperScrapy(Spider):
        """Scrapy spider để thu thập tin tức quy mô lớn"""
        
        name = "financial_news_spider"
        
        def __init__(self, symbols: List[str] = None, *args, **kwargs):
            super(NewsScraperScrapy, self).__init__(*args, **kwargs)
            self.symbols = symbols or []
            self.results = []
        
        def start_requests(self):
            """Tạo các requests ban đầu"""
            for symbol in self.symbols:
                url = f"https://cafef.vn/timeline/{symbol.lower()}.chn"
                yield Request(url, callback=self.parse, meta={'symbol': symbol})
        
        def parse(self, response):
            """Parse trang tin tức"""
            symbol = response.meta['symbol']
            
            articles = response.css('div.timeline-head')
            
            for article in articles:
                item = {
                    'symbol': symbol,
                    'title': article.css('h3::text').get(),
                    'url': article.css('a::attr(href)').get(),
                    'date': article.css('span.time::text').get(),
                    'source': 'CafeF'
                }
                
                self.results.append(item)
                yield item
else:
    # Dummy class if scrapy not available
    class NewsScraperScrapy:
        def __init__(self, *args, **kwargs):
            logger.warning("Scrapy not installed. NewsScraperScrapy will not work.")


class VNDirectAPI:
    """Kết nối API VNDirect - sử dụng đầy đủ các endpoints"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://dchart-api.vndirect.com.vn"
        self.finfo_url = "https://finfo-api.vndirect.com.vn"
        self.market_url = "https://board-api.vndirect.com.vn"  # Changed endpoint
        self.quote_url = "https://dchart-api.vndirect.com.vn"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://dstock.vndirect.com.vn/',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9,vi;q=0.8',
            'Origin': 'https://dstock.vndirect.com.vn'
        })
    
    def get_stock_price(self, symbol: str, from_date: str, to_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu giá cổ phiếu VN từ VNDirect dchart API (giống n8n HTTP request)
        
        Args:
            symbol: Mã cổ phiếu (vd: 'VNM.VN' hoặc 'VNM')
            from_date: Ngày bắt đầu (YYYY-MM-DD hoặc datetime string)
            to_date: Ngày kết thúc (YYYY-MM-DD hoặc datetime string)
        
        Returns:
            DataFrame với cột: date, Open, High, Low, Close, Volume, symbol
        """
        try:
            # Remove .VN suffix for VNDirect API
            clean_symbol = symbol.replace('.VN', '')
            
            # Convert dates to Unix timestamp - handle both date and datetime strings
            try:
                # Try parsing as date only first
                start_dt = datetime.strptime(from_date.split()[0], '%Y-%m-%d')
                end_dt = datetime.strptime(to_date.split()[0], '%Y-%m-%d')
            except:
                # Fallback to pandas for flexible parsing
                start_dt = pd.to_datetime(from_date)
                end_dt = pd.to_datetime(to_date)
            
            from_timestamp = int(start_dt.timestamp())
            to_timestamp = int(end_dt.timestamp())
            
            # API endpoint giống như n8n HTTP request
            url = f"{self.base_url}/dchart/history"
            params = {
                'resolution': 'D',  # D=daily, 1=1min, 5=5min, 15=15min, 30=30min, 60=1hour
                'symbol': clean_symbol,
                'from': from_timestamp,
                'to': to_timestamp
            }
            
            logger.info(f"Fetching from VNDirect dchart API: {clean_symbol} ({start_dt.date()} to {end_dt.date()})")
            
            response = self.session.get(url, params=params, timeout=15)
            
            if response.status_code != 200:
                logger.error(f"VNDirect API returned status {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Check if data is valid
            if data.get('s') != 'ok':
                logger.error(f"VNDirect API error: {data.get('s', 'unknown')}")
                return pd.DataFrame()
            
            # Parse response data (format giống TradingView)
            # t: timestamps (Unix), o: open, h: high, l: low, c: close, v: volume
            if not all(key in data for key in ['t', 'o', 'h', 'l', 'c', 'v']):
                logger.error("Missing required fields in VNDirect response")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame({
                'date': pd.to_datetime(data['t'], unit='s'),
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Close': data['c'],
                'Volume': data['v']
            })
            
            df['symbol'] = symbol
            
            # Sort by date
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"✅ Successfully fetched {len(df)} records from VNDirect dchart API for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"❌ Error fetching VNDirect dchart data: {str(e)}")
            return pd.DataFrame()
    
    def get_stock_info(self, symbol: str) -> Dict:
        """
        Lấy thông tin cơ bản về cổ phiếu từ VNDirect API
        
        Args:
            symbol: Mã cổ phiếu (vd: 'VNM.VN' hoặc 'VNM')
        
        Returns:
            Dict chứa thông tin công ty
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            # Try dchart/quotes endpoint first (more reliable)
            url = f"{self.quote_url}/dchart/quotes"
            params = {'symbols': clean_symbol}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data and len(data) > 0:
                    stock_data = data[0]
                    logger.info(f"✅ Successfully fetched stock info for {symbol} from quotes API")
                    return stock_data
            
            # Fallback: Try finfo API
            url2 = f"{self.finfo_url}/v4/stocks"
            params2 = {'q': f'code:{clean_symbol}'}
            
            try:
                response2 = self.session.get(url2, params=params2, timeout=10)
                if response2.status_code == 200:
                    data2 = response2.json()
                    if 'data' in data2 and len(data2['data']) > 0:
                        logger.info(f"✅ Successfully fetched stock info for {symbol} from finfo API")
                        return data2['data'][0]
            except:
                pass
            
            logger.warning(f"No stock info found for {symbol}")
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching stock info: {str(e)}")
            return {}
    
    def get_intraday_data(self, symbol: str, resolution: str = '1') -> pd.DataFrame:
        """
        Lấy dữ liệu trong ngày (intraday)
        
        Args:
            symbol: Mã cổ phiếu
            resolution: '1'=1min, '5'=5min, '15'=15min, '30'=30min, '60'=1hour
        
        Returns:
            DataFrame với dữ liệu intraday
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            # Get today's data
            end_dt = datetime.now()
            start_dt = end_dt.replace(hour=0, minute=0, second=0)
            
            url = f"{self.base_url}/dchart/history"
            params = {
                'resolution': resolution,
                'symbol': clean_symbol,
                'from': int(start_dt.timestamp()),
                'to': int(end_dt.timestamp())
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('s') == 'ok' and all(k in data for k in ['t', 'o', 'h', 'l', 'c', 'v']):
                    df = pd.DataFrame({
                        'time': pd.to_datetime(data['t'], unit='s'),
                        'Open': data['o'],
                        'High': data['h'],
                        'Low': data['l'],
                        'Close': data['c'],
                        'Volume': data['v']
                    })
                    
                    logger.info(f"✅ Fetched {len(df)} intraday records for {symbol}")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching intraday data: {str(e)}")
            return pd.DataFrame()
    
    def get_market_overview(self) -> Dict:
        """
        Lấy tổng quan thị trường (VN-INDEX, HNX-INDEX, UPCOM)
        
        Returns:
            Dict chứa thông tin chỉ số thị trường
        """
        try:
            url = f"{self.market_url}/market-info"
            
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                logger.info("✅ Fetched market overview")
                return data
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching market overview: {str(e)}")
            return {}
    
    def get_top_stocks(self, market: str = 'HOSE', criteria: str = 'value') -> pd.DataFrame:
        """
        Lấy danh sách cổ phiếu top (theo giá trị giao dịch, khối lượng, thay đổi giá)
        
        Args:
            market: 'HOSE', 'HNX', 'UPCOM'
            criteria: 'value' (giá trị GD), 'volume' (khối lượng), 'gainers' (tăng giá), 'losers' (giảm giá)
        
        Returns:
            DataFrame với danh sách cổ phiếu top
        """
        try:
            url = f"{self.finfo_url}/v4/stocks"
            params = {
                'q': f'exchange:{market}',
                'size': 50
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    
                    # Sort based on criteria
                    if criteria == 'value' and 'totalTradingValue' in df.columns:
                        df = df.sort_values('totalTradingValue', ascending=False)
                    elif criteria == 'volume' and 'totalTradingQtty' in df.columns:
                        df = df.sort_values('totalTradingQtty', ascending=False)
                    elif criteria == 'gainers' and 'perChange' in df.columns:
                        df = df[df['perChange'] > 0].sort_values('perChange', ascending=False)
                    elif criteria == 'losers' and 'perChange' in df.columns:
                        df = df[df['perChange'] < 0].sort_values('perChange', ascending=True)
                    
                    logger.info(f"✅ Fetched top {criteria} stocks for {market}")
                    return df.head(20)
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching top stocks: {str(e)}")
            return pd.DataFrame()
    
    def get_financial_ratios(self, symbol: str) -> Dict:
        """
        Lấy các chỉ số tài chính (P/E, P/B, EPS, ROE, ROA, etc.)
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict chứa các chỉ số tài chính
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            url = f"{self.finfo_url}/v4/ratios"
            params = {'q': f'code:{clean_symbol}'}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    logger.info(f"✅ Fetched financial ratios for {symbol}")
                    return data['data'][0]
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching financial ratios: {str(e)}")
            return {}
    
    def get_financial_statements(self, symbol: str, report_type: str = 'BALANCE_SHEET', 
                                 period: str = 'YEAR') -> pd.DataFrame:
        """
        Lấy báo cáo tài chính
        
        Args:
            symbol: Mã cổ phiếu
            report_type: 'BALANCE_SHEET' (Cân đối kế toán), 'INCOME_STATEMENT' (Kết quả KD), 
                        'CASH_FLOW' (Lưu chuyển tiền tệ)
            period: 'YEAR' (năm), 'QUARTER' (quý)
        
        Returns:
            DataFrame với báo cáo tài chính
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            url = f"{self.finfo_url}/v4/financial_statements"
            params = {
                'q': f'code:{clean_symbol}~reportType:{report_type}~period:{period}',
                'size': 20
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Fetched {report_type} for {symbol}")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching financial statements: {str(e)}")
            return pd.DataFrame()
    
    def get_ownership_data(self, symbol: str) -> Dict:
        """
        Lấy thông tin sở hữu (Cổ đông lớn, sở hữu nước ngoài, etc.)
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dict chứa thông tin sở hữu
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            url = f"{self.finfo_url}/v4/ownership"
            params = {'q': f'code:{clean_symbol}'}
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data and len(data['data']) > 0:
                    logger.info(f"✅ Fetched ownership data for {symbol}")
                    return data['data'][0]
            
            return {}
            
        except Exception as e:
            logger.error(f"❌ Error fetching ownership data: {str(e)}")
            return {}
    
    def get_dividends(self, symbol: str) -> pd.DataFrame:
        """
        Lấy lịch sử chi trả cổ tức
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame với lịch sử cổ tức
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            url = f"{self.finfo_url}/v4/dividends"
            params = {
                'q': f'code:{clean_symbol}',
                'size': 50
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Fetched dividend history for {symbol}")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching dividends: {str(e)}")
            return pd.DataFrame()
    
    def get_corporate_actions(self, symbol: str) -> pd.DataFrame:
        """
        Lấy lịch sử sự kiện doanh nghiệp (tăng vốn, phát hành, etc.)
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            DataFrame với lịch sử sự kiện
        """
        try:
            clean_symbol = symbol.replace('.VN', '')
            
            url = f"{self.finfo_url}/v4/corporate_actions"
            params = {
                'q': f'code:{clean_symbol}',
                'size': 50
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Fetched corporate actions for {symbol}")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error fetching corporate actions: {str(e)}")
            return pd.DataFrame()
    
    def search_stocks(self, keyword: str) -> pd.DataFrame:
        """
        Tìm kiếm cổ phiếu theo từ khóa
        
        Args:
            keyword: Từ khóa tìm kiếm (mã, tên công ty)
        
        Returns:
            DataFrame với kết quả tìm kiếm
        """
        try:
            url = f"{self.finfo_url}/v4/stocks"
            params = {
                'q': f'code,companyName~{keyword}',
                'size': 20
            }
            
            response = self.session.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'data' in data:
                    df = pd.DataFrame(data['data'])
                    logger.info(f"✅ Found {len(df)} stocks matching '{keyword}'")
                    return df
            
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"❌ Error searching stocks: {str(e)}")
            return pd.DataFrame()


class DataCollectionPipeline:
    """Pipeline tổng hợp để thu thập dữ liệu từ nhiều nguồn"""
    
    def __init__(self, alpha_vantage_key: Optional[str] = None):
        self.yahoo = YahooFinanceAPI()
        self.alpha_vantage = AlphaVantageAPI(alpha_vantage_key) if alpha_vantage_key else None
        self.news_scraper = NewsScraperBS4()
        self.vndirect = VNDirectAPI()
    
    def collect_all_data(self, symbols: List[str], start_date: str, 
                        end_date: str) -> Tuple[Dict, pd.DataFrame]:
        """
        Thu thập dữ liệu giá và tin tức cho danh sách cổ phiếu
        
        Returns:
            Tuple[Dict, DataFrame]: (price_data, news_data)
        """
        # Thu thập dữ liệu giá
        logger.info("Collecting price data...")
        price_data = self.yahoo.get_multiple_stocks(symbols, start_date, end_date)
        
        # Thu thập tin tức
        logger.info("Collecting news data...")
        all_news = []
        for symbol in symbols:
            news = self.news_scraper.scrape_cafef(symbol, pages=3)
            all_news.extend(news)
            time.sleep(1)
        
        news_df = pd.DataFrame(all_news)
        
        logger.info(f"Data collection complete. Price data for {len(price_data)} symbols, "
                   f"{len(news_df)} news articles")
        
        return price_data, news_df


if __name__ == "__main__":
    # Test data collection
    pipeline = DataCollectionPipeline()
    
    # Test với một số mã cổ phiếu Việt Nam
    symbols = ['VNM.VN', 'VIC.VN', 'HPG.VN']
    start_date = '2023-01-01'
    end_date = '2024-01-01'
    
    price_data, news_data = pipeline.collect_all_data(symbols, start_date, end_date)
    
    print(f"\nCollected data for {len(price_data)} symbols")
    for symbol, df in price_data.items():
        print(f"{symbol}: {len(df)} records")
