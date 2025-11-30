"""
Web Scraping Module - Thu thập tin tức và dữ liệu văn bản
Sử dụng BeautifulSoup và Scrapy
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import logging
import time
from typing import List, Dict
import re

logger = logging.getLogger(__name__)


class NewsScr aper:
    """
    Scraper cho các trang tin tức tài chính Việt Nam
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        self.session = requests.Session()
    
    def scrape_cafef(self, symbol: str, max_pages: int = 3) -> List[Dict]:
        """
        Scrape tin tức từ CafeF.vn
        """
        logger.info(f"🌐 Scraping CafeF for {symbol}...")
        articles = []
        
        try:
            for page in range(1, max_pages + 1):
                url = f"https://cafef.vn/timeline-tag/{symbol.lower()}.chn?trang={page}"
                
                response = self.session.get(url, headers=self.headers, timeout=10)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Parse articles
                items = soup.find_all('div', class_='tlitem')
                
                for item in items:
                    try:
                        title_tag = item.find('h3')
                        if not title_tag:
                            continue
                        
                        title = title_tag.get_text(strip=True)
                        link = title_tag.find('a')['href'] if title_tag.find('a') else ''
                        
                        # Get date
                        date_tag = item.find('span', class_='time')
                        date_str = date_tag.get_text(strip=True) if date_tag else ''
                        
                        # Get summary
                        summary_tag = item.find('p', class_='sapo')
                        summary = summary_tag.get_text(strip=True) if summary_tag else ''
                        
                        articles.append({
                            'symbol': symbol,
                            'source': 'CafeF',
                            'title': title,
                            'summary': summary,
                            'link': link,
                            'date_str': date_str,
                            'scraped_at': datetime.now()
                        })
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Error parsing article: {e}")
                
                time.sleep(1)  # Be nice to server
            
            logger.info(f"✅ Scraped {len(articles)} articles from CafeF")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error scraping CafeF: {e}")
            return articles
    
    def scrape_vietstock(self, symbol: str) -> List[Dict]:
        """
        Scrape tin tức từ VietStock.vn
        """
        logger.info(f"🌐 Scraping VietStock for {symbol}...")
        articles = []
        
        try:
            url = f"https://vietstock.vn/{symbol.upper()}/tin-tuc.htm"
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse news items
            news_items = soup.find_all('div', class_='news-item')
            
            for item in news_items[:20]:  # Top 20
                try:
                    title_tag = item.find('h3') or item.find('h4')
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    link_tag = title_tag.find('a')
                    link = link_tag['href'] if link_tag else ''
                    
                    # Get date
                    date_tag = item.find('span', class_='date')
                    date_str = date_tag.get_text(strip=True) if date_tag else ''
                    
                    articles.append({
                        'symbol': symbol,
                        'source': 'VietStock',
                        'title': title,
                        'summary': '',
                        'link': link if link.startswith('http') else f"https://vietstock.vn{link}",
                        'date_str': date_str,
                        'scraped_at': datetime.now()
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing article: {e}")
            
            logger.info(f"✅ Scraped {len(articles)} articles from VietStock")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error scraping VietStock: {e}")
            return articles
    
    def scrape_ndh(self, symbol: str) -> List[Dict]:
        """
        Scrape tin tức từ Nhịp Đập Đầu Tư
        """
        logger.info(f"🌐 Scraping NDH for {symbol}...")
        articles = []
        
        try:
            url = f"https://ndh.vn/tag/{symbol.lower()}"
            
            response = self.session.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Parse articles
            article_items = soup.find_all('article')
            
            for item in article_items[:15]:
                try:
                    title_tag = item.find('h2') or item.find('h3')
                    if not title_tag:
                        continue
                    
                    title = title_tag.get_text(strip=True)
                    link_tag = title_tag.find('a')
                    link = link_tag['href'] if link_tag else ''
                    
                    # Get summary
                    summary_tag = item.find('div', class_='excerpt') or item.find('p')
                    summary = summary_tag.get_text(strip=True) if summary_tag else ''
                    
                    articles.append({
                        'symbol': symbol,
                        'source': 'NDH',
                        'title': title,
                        'summary': summary,
                        'link': link if link.startswith('http') else f"https://ndh.vn{link}",
                        'date_str': '',
                        'scraped_at': datetime.now()
                    })
                    
                except Exception as e:
                    logger.warning(f"⚠️ Error parsing article: {e}")
            
            logger.info(f"✅ Scraped {len(articles)} articles from NDH")
            return articles
            
        except Exception as e:
            logger.error(f"❌ Error scraping NDH: {e}")
            return articles
    
    def scrape_all(self, symbol: str) -> pd.DataFrame:
        """
        Scrape from all sources
        """
        logger.info(f"🌐 Scraping all sources for {symbol}...")
        
        all_articles = []
        
        # CafeF
        all_articles.extend(self.scrape_cafef(symbol))
        
        # VietStock
        all_articles.extend(self.scrape_vietstock(symbol))
        
        # NDH
        all_articles.extend(self.scrape_ndh(symbol))
        
        df = pd.DataFrame(all_articles)
        
        if not df.empty:
            # Remove duplicates
            df = df.drop_duplicates(subset=['title', 'source'])
            logger.info(f"✅ Total {len(df)} unique articles scraped for {symbol}")
        else:
            logger.warning(f"⚠️ No articles found for {symbol}")
        
        return df


class CompanyInfoScraper:
    """
    Scraper cho thông tin công ty từ các nguồn
    """
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
    
    def scrape_vietstock_company(self, symbol: str) -> Dict:
        """
        Scrape thông tin công ty từ VietStock
        """
        logger.info(f"🏢 Scraping company info for {symbol} from VietStock...")
        
        try:
            url = f"https://vietstock.vn/{symbol.upper()}/company-profile.htm"
            
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            company_info = {
                'symbol': symbol,
                'source': 'VietStock'
            }
            
            # Parse company name
            name_tag = soup.find('h1', class_='company-name')
            if name_tag:
                company_info['company_name'] = name_tag.get_text(strip=True)
            
            # Parse industry
            industry_tag = soup.find('div', class_='industry')
            if industry_tag:
                company_info['industry'] = industry_tag.get_text(strip=True)
            
            # Parse info table
            info_rows = soup.find_all('tr')
            for row in info_rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    key = cols[0].get_text(strip=True).lower()
                    value = cols[1].get_text(strip=True)
                    company_info[key] = value
            
            logger.info(f"✅ Scraped company info for {symbol}")
            return company_info
            
        except Exception as e:
            logger.error(f"❌ Error scraping company info: {e}")
            return {'symbol': symbol, 'source': 'VietStock', 'error': str(e)}


# ==================== SCRAPY SPIDER (Advanced) ====================
try:
    import scrapy
    from scrapy.crawler import CrawlerProcess
    
    class FinanceNewsSpider(scrapy.Spider):
        """
        Scrapy Spider cho tin tức tài chính (advanced scraping)
        """
        name = 'finance_news'
        
        def __init__(self, symbol='VNM', *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.symbol = symbol
            self.start_urls = [
                f'https://cafef.vn/timeline-tag/{symbol.lower()}.chn',
                f'https://vietstock.vn/{symbol.upper()}/tin-tuc.htm'
            ]
        
        def parse(self, response):
            """Parse news articles"""
            # CafeF parsing
            if 'cafef' in response.url:
                for article in response.css('div.tlitem'):
                    yield {
                        'symbol': self.symbol,
                        'source': 'CafeF',
                        'title': article.css('h3::text').get(),
                        'link': article.css('h3 a::attr(href)').get(),
                        'date': article.css('span.time::text').get(),
                        'summary': article.css('p.sapo::text').get()
                    }
            
            # VietStock parsing
            elif 'vietstock' in response.url:
                for article in response.css('div.news-item'):
                    yield {
                        'symbol': self.symbol,
                        'source': 'VietStock',
                        'title': article.css('h3::text, h4::text').get(),
                        'link': article.css('a::attr(href)').get(),
                        'date': article.css('span.date::text').get()
                    }
    
except ImportError:
    logger.warning("⚠️ Scrapy not installed. Using BeautifulSoup only.")
    FinanceNewsSpider = None


# ==================== MAIN FUNCTION ====================
def scrape_news_for_stocks(symbols: List[str], output_dir: str = 'data/news'):
    """
    Main function để scrape news cho nhiều stocks
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = NewsScraper()
    
    for symbol in symbols:
        logger.info(f"📰 Scraping news for {symbol}...")
        
        try:
            df = scraper.scrape_all(symbol)
            
            if not df.empty:
                # Save to CSV
                output_file = os.path.join(output_dir, f'{symbol}_news_{datetime.now().strftime("%Y%m%d")}.csv')
                df.to_csv(output_file, index=False, encoding='utf-8-sig')
                logger.info(f"✅ Saved {len(df)} articles to {output_file}")
            
            time.sleep(2)  # Be nice to servers
            
        except Exception as e:
            logger.error(f"❌ Error scraping {symbol}: {e}")


if __name__ == '__main__':
    # Test scraping
    logging.basicConfig(level=logging.INFO)
    
    test_symbols = ['VNM', 'VIC', 'HPG']
    scrape_news_for_stocks(test_symbols)
