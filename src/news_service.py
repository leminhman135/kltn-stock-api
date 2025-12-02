"""
News Service - Thu thập và phân tích tin tức chứng khoán THẬT
Lấy dữ liệu từ RSS feeds và web scraping từ các nguồn uy tín
"""

import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import hashlib

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False


class SentimentType(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    source: str
    published_at: str
    symbol: Optional[str] = None
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.0
    impact_prediction: str = ""


class SentimentAnalyzer:
    """Phân tích sentiment tin tức chứng khoán Việt Nam với từ điển mở rộng"""
    
    POSITIVE_KEYWORDS = [
        # Tài chính - Lợi nhuận
        "tăng trưởng", "lợi nhuận tăng", "doanh thu tăng", "vượt kế hoạch",
        "cổ tức cao", "chia cổ tức", "tăng vốn", "phát hành thêm",
        "lãi ròng", "lãi kỷ lục", "tăng mạnh", "bứt phá", "đột phá",
        "triển vọng tốt", "khuyến nghị mua", "mục tiêu tăng", "kỳ vọng",
        "lạc quan", "tích cực", "khả quan", "thuận lợi", "hoàn thành",
        "vượt mức", "cao hơn kỳ vọng", "tốt hơn dự báo",
        # Kinh doanh
        "mở rộng", "đầu tư mới", "hợp tác", "ký kết", "thắng thầu",
        "thâu tóm", "sáp nhập", "dự án mới", "ra mắt", "xuất khẩu tăng", 
        "thị phần tăng", "khách hàng mới", "đơn hàng mới", "hợp đồng lớn",
        "mở thêm", "khai trương", "chiến lược", "đổi mới",
        # Thị trường
        "uptrend", "breakout", "vượt đỉnh", "thanh khoản cao", "tăng điểm",
        "khối ngoại mua ròng", "dòng tiền vào", "tăng trần", "bật tăng",
        "hồi phục", "phục hồi", "đảo chiều tăng", "xanh", "sáng",
        "tăng giá", "nâng hạng", "thu hút vốn",
        # Đánh giá
        "outperform", "overweight", "strong buy", "nâng rating", "nâng mục tiêu",
        "vượt kỳ vọng", "khuyến nghị", "tiềm năng", "cơ hội",
    ]
    
    NEGATIVE_KEYWORDS = [
        # Tài chính - Thua lỗ
        "thua lỗ", "lỗ ròng", "giảm lợi nhuận", "doanh thu giảm",
        "nợ xấu", "nợ tăng", "phá sản", "giải thể", "mất vốn",
        "cắt cổ tức", "không chia cổ tức", "hủy niêm yết", "tăng vốn ảo",
        "bị phạt", "vi phạm", "gian lận", "điều tra", "thanh tra",
        "tăng trưởng âm", "sụt giảm", "thất thu", "thất bại",
        "không hoàn thành", "thấp hơn kỳ vọng", "kém dự báo",
        # Kinh doanh
        "thu hẹp", "đóng cửa", "cắt giảm", "sa thải", "ngừng hoạt động",
        "mất hợp đồng", "kiện tụng", "tranh chấp", "đình công",
        "tồn kho tăng", "khách hàng rời bỏ", "mất thị phần",
        "tái cơ cấu", "cắt giảm nhân sự",
        # Thị trường
        "downtrend", "breakdown", "thủng đáy", "mất đáy", "thanh khoản thấp",
        "khối ngoại bán ròng", "dòng tiền ra", "giảm sàn", "rơi tự do",
        "bán tháo", "cắt lỗ", "panic sell", "tháo chạy", "lao dốc",
        "giảm mạnh", "giảm sâu", "sụp đổ", "đỏ", "rung lắc",
        "giảm giá", "hạ hạng", "rút vốn",
        # Đánh giá
        "underperform", "underweight", "sell", "hạ rating", "hạ mục tiêu",
        "cảnh báo", "rủi ro cao", "kém kỳ vọng", "thất vọng", "lo ngại",
    ]
    
    STRONG_MODIFIERS = ["kỷ lục", "đột biến", "lịch sử", "chưa từng có", "mạnh nhất", "lớn nhất", "cao nhất", "thấp nhất"]
    
    def analyze(self, text: str) -> tuple:
        text_lower = text.lower()
        
        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)
        
        has_strong = any(m in text_lower for m in self.STRONG_MODIFIERS)
        multiplier = 1.5 if has_strong else 1.0
        
        total = pos_count + neg_count
        if total == 0:
            return SentimentType.NEUTRAL, 0.0, "Không có tín hiệu rõ ràng từ tin tức"
        
        score = ((pos_count - neg_count) / total) * multiplier
        score = max(-1.0, min(1.0, score))
        
        if score > 0.2:
            sentiment = SentimentType.POSITIVE
            impact = "🚀 Tín hiệu TĂNG MẠNH - Khuyến nghị MUA" if score > 0.6 else "📈 Tín hiệu TĂNG - Cân nhắc mua vào"
        elif score < -0.2:
            sentiment = SentimentType.NEGATIVE
            impact = "🔻 Tín hiệu GIẢM MẠNH - Khuyến nghị BÁN" if score < -0.6 else "📉 Tín hiệu GIẢM - Cân nhắc cắt lỗ"
        else:
            sentiment = SentimentType.NEUTRAL
            impact = "➡️ Trung lập - Tiếp tục theo dõi diễn biến"
        
        return sentiment, round(score, 2), impact


class NewsService:
    """Service thu thập tin tức THẬT từ nhiều nguồn RSS và Web"""
    
    # RSS Feeds chứng khoán Việt Nam - Nguồn uy tín (Updated 2024-12-03)
    # Chỉ giữ lại các feeds đang hoạt động ổn định
    RSS_FEEDS = {
        # Báo tài chính chuyên ngành - WORKING
        "VnEconomy_ChungKhoan": "https://vneconomy.vn/chung-khoan.rss",
        "VnEconomy_DoanhNghiep": "https://vneconomy.vn/doanh-nghiep.rss",
        "VnEconomy_TaiChinh": "https://vneconomy.vn/tai-chinh-ngan-hang.rss",
        
        # Báo chính thống - WORKING
        "TuoiTre_KinhDoanh": "https://tuoitre.vn/rss/kinh-doanh.rss",
        "ThanhNien_KinhTe": "https://thanhnien.vn/rss/kinh-te.rss",
        "VietnamNet_KinhDoanh": "https://vietnamnet.vn/rss/kinh-doanh.rss",
        "VnExpress_KinhDoanh": "https://vnexpress.net/rss/kinh-doanh.rss",
        "DanTri_KinhDoanh": "https://dantri.com.vn/kinh-doanh.rss",
        
        # Các feeds đã tắt/lỗi (để tham khảo):
        # "CafeF": "https://cafef.vn/rss/chung-khoan.rss",  # 404 Error - đổi cấu trúc
        # "VTV": "https://vtv.vn/kinh-te.rss",  # 404 Error
        # "VietStock": "https://vietstock.vn/api/rss/cate/2",  # XML parse error
        # "NDH": "https://ndh.vn/rss/tai-chinh.rss",  # Connection timeout
    }
    
    # Mapping mã CK -> từ khóa
    STOCK_KEYWORDS = {
        "VNM": ["vinamilk", "sữa việt nam", "vnm", "sữa vinamilk"],
        "VIC": ["vingroup", "tập đoàn vin", "vic", "vinfast", "vin group"],
        "VHM": ["vinhomes", "vhm", "vin homes"],
        "VCB": ["vietcombank", "ngân hàng ngoại thương", "vcb"],
        "BID": ["bidv", "ngân hàng đầu tư", "bid"],
        "CTG": ["vietinbank", "ngân hàng công thương", "ctg"],
        "TCB": ["techcombank", "tcb", "techcom"],
        "MBB": ["mb bank", "mbbank", "quân đội", "mbb", "mb"],
        "HPG": ["hòa phát", "hoa phat", "thép hòa phát", "hpg"],
        "MSN": ["masan", "msn", "tập đoàn masan"],
        "FPT": ["fpt", "fpt corporation", "tập đoàn fpt"],
        "MWG": ["thế giới di động", "điện máy xanh", "mwg", "bách hóa xanh", "mobile world"],
        "VPB": ["vpbank", "vp bank", "vpb"],
        "GAS": ["pvgas", "pv gas", "khí việt nam", "gas"],
        "SAB": ["sabeco", "bia sài gòn", "sab"],
        "PLX": ["petrolimex", "xăng dầu", "plx"],
        "VJC": ["vietjet", "vjc", "vietjet air"],
        "SSI": ["ssi", "chứng khoán ssi"],
        "VRE": ["vincom retail", "vre", "vincom"],
        "POW": ["pv power", "pow", "điện lực dầu khí"],
        "NVL": ["novaland", "nvl", "nova"],
        "ACB": ["acb", "á châu", "ngân hàng á châu"],
        "STB": ["sacombank", "stb"],
        "TPB": ["tpbank", "tiên phong", "tpb"],
        "HDB": ["hdbank", "hdb", "phát triển tphcm"],
        "VND": ["vndirect", "vnd", "chứng khoán vndirect"],
        "GVR": ["cao su việt nam", "gvr", "rubber"],
        "BCM": ["becamex", "bcm"],
        "PDR": ["phát đạt", "pdr"],
        "SHB": ["shb", "sài gòn hà nội"],
    }
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8",
        }
        self._cache = {}
        self._cache_time = {}
        self._cache_duration = 300  # 5 phút
    
    def _get_cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_cache_valid(self, key: str) -> bool:
        if key not in self._cache_time:
            return False
        return (datetime.now() - self._cache_time[key]).seconds < self._cache_duration
    
    def _clean_html(self, text: str) -> str:
        """Loại bỏ HTML tags"""
        if not text:
            return ""
        if HAS_BS4:
            return BeautifulSoup(text, 'html.parser').get_text()
        # Fallback nếu không có BeautifulSoup
        clean = re.sub(r'<[^>]+>', '', text)
        clean = re.sub(r'&[a-zA-Z]+;', ' ', clean)
        return clean.strip()
    
    def _parse_date(self, date_str: str) -> str:
        """Parse nhiều định dạng ngày"""
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d %H:%M")
        
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT",
            "%a, %d %b %Y %H:%M:%S +0700",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%d/%m/%Y %H:%M",
            "%Y-%m-%d %H:%M:%S",
            "%d-%m-%Y %H:%M",
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.strftime("%Y-%m-%d %H:%M")
            except:
                continue
        
        # Thử parse ngày tiếng Việt
        try:
            # "2 giờ trước", "30 phút trước"
            if "giờ trước" in date_str.lower():
                hours = int(re.search(r'\d+', date_str).group())
                return (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M")
            if "phút trước" in date_str.lower():
                mins = int(re.search(r'\d+', date_str).group())
                return (datetime.now() - timedelta(minutes=mins)).strftime("%Y-%m-%d %H:%M")
            if "ngày trước" in date_str.lower():
                days = int(re.search(r'\d+', date_str).group())
                return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d %H:%M")
        except:
            pass
        
        return datetime.now().strftime("%Y-%m-%d %H:%M")
    
    def fetch_rss(self, feed_name: str, feed_url: str, limit: int = 15) -> List[NewsArticle]:
        """Lấy tin từ RSS feed"""
        cache_key = self._get_cache_key(f"rss_{feed_url}")
        
        if self._is_cache_valid(cache_key) and cache_key in self._cache:
            return self._cache[cache_key][:limit]
        
        news = []
        try:
            response = requests.get(feed_url, headers=self.headers, timeout=10)
            response.encoding = 'utf-8'
            
            if response.status_code == 200:
                # Parse XML
                try:
                    root = ET.fromstring(response.content)
                except ET.ParseError:
                    # Thử clean content trước khi parse
                    content = response.content.decode('utf-8', errors='ignore')
                    content = re.sub(r'&(?!amp;|lt;|gt;|quot;|apos;)', '&amp;', content)
                    root = ET.fromstring(content.encode('utf-8'))
                
                # Tìm items trong RSS
                items = root.findall('.//item')
                if not items:
                    items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
                
                for item in items[:limit * 2]:
                    try:
                        # Lấy title
                        title_el = item.find('title')
                        if title_el is None:
                            title_el = item.find('{http://www.w3.org/2005/Atom}title')
                        title = title_el.text if title_el is not None and title_el.text else ""
                        title = self._clean_html(title)
                        
                        if not title or len(title) < 10:
                            continue
                        
                        # Lấy description/summary
                        desc_el = item.find('description')
                        if desc_el is None:
                            desc_el = item.find('summary')
                        if desc_el is None:
                            desc_el = item.find('{http://www.w3.org/2005/Atom}summary')
                        
                        summary = ""
                        if desc_el is not None and desc_el.text:
                            summary = self._clean_html(desc_el.text)
                            summary = summary[:400] + "..." if len(summary) > 400 else summary
                        else:
                            summary = title[:200]
                        
                        # Lấy link
                        link_el = item.find('link')
                        if link_el is None:
                            link_el = item.find('{http://www.w3.org/2005/Atom}link')
                        
                        url = ""
                        if link_el is not None:
                            url = link_el.text if link_el.text else link_el.get('href', '')
                        
                        if not url:
                            continue
                        
                        # Lấy ngày
                        pub_el = item.find('pubDate')
                        if pub_el is None:
                            pub_el = item.find('published')
                        if pub_el is None:
                            pub_el = item.find('{http://www.w3.org/2005/Atom}published')
                        
                        date_str = self._parse_date(pub_el.text if pub_el is not None else "")
                        
                        # Phân tích sentiment
                        sentiment, score, impact = self.analyzer.analyze(title + " " + summary)
                        
                        news.append(NewsArticle(
                            title=title.strip(),
                            summary=summary.strip(),
                            url=url.strip(),
                            source=feed_name.replace("_", " "),
                            published_at=date_str,
                            sentiment=sentiment,
                            sentiment_score=score,
                            impact_prediction=impact
                        ))
                    except Exception as e:
                        continue
                
                # Cache kết quả
                if news:
                    self._cache[cache_key] = news
                    self._cache_time[cache_key] = datetime.now()
                
        except Exception as e:
            print(f"Error fetching RSS {feed_name}: {e}")
        
        return news[:limit]
    
    def scrape_cafef_web(self, limit: int = 10) -> List[NewsArticle]:
        """Scrape tin từ CafeF website (backup khi RSS không hoạt động)"""
        news = []
        if not HAS_BS4:
            return news
            
        try:
            # Scrape trang chứng khoán CafeF
            url = "https://cafef.vn/chung-khoan.chn"
            response = requests.get(url, headers=self.headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # CafeF structure: tìm các article items
                articles = soup.select('.tlitem, .item-news, .box-category-item')
                
                for article in articles[:limit]:
                    try:
                        # Tìm title và link
                        title_link = article.select_one('h3 a, .title a, a[data-type="headline"]')
                        if not title_link:
                            continue
                        
                        title = title_link.get('title') or title_link.get_text(strip=True)
                        link = title_link.get('href', '')
                        
                        if not title or len(title) < 15:
                            continue
                        
                        # Fix relative URLs
                        if link and not link.startswith('http'):
                            link = 'https://cafef.vn' + link
                        
                        # Tìm summary
                        summary_el = article.select_one('.sapo, .summary, .box-category-sapo, p')
                        summary = summary_el.get_text(strip=True) if summary_el else title[:200]
                        
                        # Sentiment analysis
                        sentiment, score, impact = self.analyzer.analyze(title + " " + summary)
                        
                        news.append(NewsArticle(
                            title=title.strip(),
                            summary=summary[:350],
                            url=link,
                            source="CafeF",
                            published_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
                            sentiment=sentiment,
                            sentiment_score=score,
                            impact_prediction=impact
                        ))
                    except Exception as e:
                        continue
                        
        except Exception as e:
            print(f"CafeF scraping error: {e}")
        
        return news
    
    def filter_by_symbol(self, news: List[NewsArticle], symbol: str) -> List[NewsArticle]:
        """Lọc tin theo mã cổ phiếu"""
        if not symbol:
            return news
        
        symbol = symbol.upper()
        keywords = self.STOCK_KEYWORDS.get(symbol, [])
        keywords.append(symbol.lower())
        
        filtered = []
        for article in news:
            text = (article.title + " " + article.summary).lower()
            if any(kw.lower() in text for kw in keywords):
                article.symbol = symbol
                filtered.append(article)
        
        return filtered
    
    def get_all_news(self, symbol: str = None, limit: int = 100) -> List[NewsArticle]:
        """Lấy tin từ TẤT CẢ nguồn RSS và web scraping"""
        cache_key = self._get_cache_key(f"all_news_{symbol or 'general'}_{limit}")
        
        if self._is_cache_valid(cache_key) and cache_key in self._cache:
            return self._cache[cache_key]
        
        all_news = []
        
        # 1. Lấy từ tất cả RSS feeds (tăng limit để có nhiều tin hơn)
        for feed_name, feed_url in self.RSS_FEEDS.items():
            try:
                news = self.fetch_rss(feed_name, feed_url, limit=20)  # Increased from 12 to 20
                all_news.extend(news)
            except Exception as e:
                print(f"Error with {feed_name}: {e}")
                continue
        
        # 2. Thử scrape thêm từ CafeF Web (vì RSS đã tắt)
        try:
            cafef_news = self.scrape_cafef_web(limit=15)  # Increased from 10 to 15
            all_news.extend(cafef_news)
        except Exception as e:
            print(f"CafeF scraping failed: {e}")
        
        # 3. Loại bỏ trùng lặp theo title
        seen_titles = set()
        unique_news = []
        for article in all_news:
            # Normalize title để so sánh
            title_key = re.sub(r'[^\w\s]', '', article.title.lower())[:50]
            if title_key not in seen_titles and len(title_key) > 10:
                seen_titles.add(title_key)
                unique_news.append(article)
        
        # 4. Lọc theo symbol nếu có
        if symbol:
            symbol_news = self.filter_by_symbol(unique_news, symbol)
            
            # Nếu có ít tin riêng, thêm tin thị trường chung
            if len(symbol_news) < 10:
                general_news = [n for n in unique_news if n not in symbol_news]
                # Đánh dấu tin chung
                for n in general_news:
                    if not n.symbol:
                        n.symbol = "MARKET"
                symbol_news.extend(general_news[:15 - len(symbol_news)])
            
            unique_news = symbol_news
        
        # 5. Sắp xếp theo thời gian (mới nhất trước)
        unique_news.sort(key=lambda x: x.published_at, reverse=True)
        
        result = unique_news[:limit]
        
        # Cache kết quả
        self._cache[cache_key] = result
        self._cache_time[cache_key] = datetime.now()
        
        return result
    
    def get_sentiment_summary(self, symbol: str) -> Dict:
        """Tổng hợp sentiment cho một mã cổ phiếu"""
        news = self.get_all_news(symbol, limit=30)
        
        if not news:
            return {
                "symbol": symbol,
                "overall": "neutral",
                "avg_score": 0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "total_news": 0,
                "recommendation": "Không có đủ tin tức để phân tích"
            }
        
        # Tính số lượng từng loại
        pos = sum(1 for n in news if n.sentiment == SentimentType.POSITIVE)
        neg = sum(1 for n in news if n.sentiment == SentimentType.NEGATIVE)
        neu = sum(1 for n in news if n.sentiment == SentimentType.NEUTRAL)
        
        # Tính điểm trung bình
        avg_score = sum(n.sentiment_score for n in news) / len(news)
        
        # Xác định xu hướng tổng thể
        if avg_score > 0.25:
            overall = "positive"
            rec = f"🟢 TIN TỨC TÍCH CỰC ({pos}/{len(news)} tin tốt) - Xu hướng thuận lợi, cân nhắc MUA vào"
        elif avg_score < -0.25:
            overall = "negative"
            rec = f"🔴 TIN TỨC TIÊU CỰC ({neg}/{len(news)} tin xấu) - Cân nhắc BÁN hoặc chờ đợi thêm"
        else:
            overall = "neutral"
            rec = f"🟡 TIN TỨC TRUNG LẬP - Thị trường đang chờ tín hiệu, theo dõi thêm"
        
        return {
            "symbol": symbol,
            "overall": overall,
            "avg_score": round(avg_score, 2),
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "total_news": len(news),
            "recommendation": rec
        }


# Singleton instance
news_service = NewsService()
