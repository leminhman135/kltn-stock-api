"""
News Service - Thu thập và phân tích tin tức chứng khoán
Sử dụng Sentiment Analysis để dự đoán ảnh hưởng đến giá cổ phiếu
"""

import re
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum


class SentimentType(str, Enum):
    POSITIVE = "positive"      # Tích cực - có thể tăng giá
    NEGATIVE = "negative"      # Tiêu cực - có thể giảm giá
    NEUTRAL = "neutral"        # Trung lập


@dataclass
class NewsArticle:
    title: str
    summary: str
    url: str
    source: str
    published_at: str
    symbol: Optional[str] = None
    sentiment: SentimentType = SentimentType.NEUTRAL
    sentiment_score: float = 0.0  # -1 đến 1
    impact_prediction: str = ""   # Dự đoán ảnh hưởng


class SentimentAnalyzer:
    """
    Phân tích sentiment tin tức chứng khoán Việt Nam
    Sử dụng từ điển từ khóa và rules
    """
    
    # Từ khóa tích cực - có thể làm tăng giá
    POSITIVE_KEYWORDS = [
        # Tài chính
        "tăng trưởng", "lợi nhuận tăng", "doanh thu tăng", "vượt kế hoạch",
        "cổ tức cao", "chia cổ tức", "tăng vốn", "phát hành thêm",
        "lãi ròng", "lãi kỷ lục", "tăng mạnh", "bứt phá",
        "triển vọng tốt", "khuyến nghị mua", "mục tiêu tăng",
        # Kinh doanh
        "mở rộng", "đầu tư mới", "hợp tác", "ký kết",
        "thâu tóm", "sáp nhập", "dự án mới", "thắng thầu",
        "xuất khẩu tăng", "thị phần tăng", "khách hàng mới",
        # Thị trường
        "uptrend", "breakout", "vượt đỉnh", "thanh khoản cao",
        "khối ngoại mua ròng", "dòng tiền vào", "tăng trần",
        # Đánh giá
        "outperform", "overweight", "strong buy", "nâng rating",
    ]
    
    # Từ khóa tiêu cực - có thể làm giảm giá
    NEGATIVE_KEYWORDS = [
        # Tài chính
        "thua lỗ", "lỗ ròng", "giảm lợi nhuận", "doanh thu giảm",
        "nợ xấu", "nợ tăng", "phá sản", "giải thể",
        "cắt cổ tức", "không chia cổ tức", "hủy niêm yết",
        "bị phạt", "vi phạm", "gian lận", "điều tra",
        # Kinh doanh  
        "thu hẹp", "đóng cửa", "cắt giảm", "sa thải",
        "mất hợp đồng", "kiện tụng", "tranh chấp",
        "tồn kho tăng", "khách hàng rời bỏ",
        # Thị trường
        "downtrend", "breakdown", "mất đáy", "thanh khoản thấp",
        "khối ngoại bán ròng", "dòng tiền ra", "giảm sàn",
        "bán tháo", "cắt lỗ", "panic sell",
        # Đánh giá
        "underperform", "underweight", "sell", "hạ rating",
        "cảnh báo", "rủi ro cao",
    ]
    
    # Từ khóa mạnh (tăng trọng số)
    STRONG_MODIFIERS = ["kỷ lục", "đột biến", "lịch sử", "chưa từng có", "mạnh nhất", "lớn nhất"]
    WEAK_MODIFIERS = ["nhẹ", "nhỏ", "tạm thời", "ngắn hạn"]
    
    def analyze(self, text: str) -> tuple[SentimentType, float, str]:
        """
        Phân tích sentiment của văn bản
        Returns: (sentiment_type, score, impact_prediction)
        """
        text_lower = text.lower()
        
        # Đếm từ khóa
        pos_count = sum(1 for kw in self.POSITIVE_KEYWORDS if kw in text_lower)
        neg_count = sum(1 for kw in self.NEGATIVE_KEYWORDS if kw in text_lower)
        
        # Điều chỉnh theo modifier
        has_strong = any(m in text_lower for m in self.STRONG_MODIFIERS)
        has_weak = any(m in text_lower for m in self.WEAK_MODIFIERS)
        
        multiplier = 1.5 if has_strong else (0.5 if has_weak else 1.0)
        
        # Tính điểm
        total = pos_count + neg_count
        if total == 0:
            return SentimentType.NEUTRAL, 0.0, "Không có tín hiệu rõ ràng"
        
        score = ((pos_count - neg_count) / total) * multiplier
        score = max(-1.0, min(1.0, score))  # Clamp to [-1, 1]
        
        # Xác định sentiment
        if score > 0.2:
            sentiment = SentimentType.POSITIVE
            if score > 0.6:
                impact = "🚀 Tín hiệu TĂNG MẠNH - Khuyến nghị MUA"
            else:
                impact = "📈 Tín hiệu TĂNG - Có thể cân nhắc mua"
        elif score < -0.2:
            sentiment = SentimentType.NEGATIVE
            if score < -0.6:
                impact = "🔻 Tín hiệu GIẢM MẠNH - Khuyến nghị BÁN"
            else:
                impact = "📉 Tín hiệu GIẢM - Cân nhắc cắt lỗ"
        else:
            sentiment = SentimentType.NEUTRAL
            impact = "➡️ Trung lập - Theo dõi thêm"
        
        return sentiment, score, impact


class NewsService:
    """
    Service thu thập tin tức từ nhiều nguồn
    """
    
    def __init__(self):
        self.analyzer = SentimentAnalyzer()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        # Mapping mã CK -> từ khóa tìm kiếm
        self.stock_keywords = {
            "VNM": ["vinamilk", "VNM", "sữa vinamilk"],
            "VIC": ["vingroup", "VIC", "tập đoàn vin"],
            "VHM": ["vinhomes", "VHM", "bất động sản vin"],
            "VCB": ["vietcombank", "VCB", "ngân hàng ngoại thương"],
            "BID": ["bidv", "BID", "ngân hàng đầu tư"],
            "CTG": ["vietinbank", "CTG", "ngân hàng công thương"],
            "TCB": ["techcombank", "TCB"],
            "MBB": ["mb bank", "MBB", "quân đội"],
            "HPG": ["hòa phát", "HPG", "thép hòa phát"],
            "MSN": ["masan", "MSN", "tập đoàn masan"],
            "FPT": ["fpt", "FPT"],
            "MWG": ["thế giới di động", "MWG", "điện máy xanh"],
            "VPB": ["vpbank", "VPB"],
            "GAS": ["pvgas", "GAS", "khí việt nam"],
            "SAB": ["sabeco", "SAB", "bia sài gòn"],
            "PLX": ["petrolimex", "PLX", "xăng dầu"],
            "VJC": ["vietjet", "VJC"],
            "SSI": ["ssi", "SSI", "chứng khoán ssi"],
            "VRE": ["vincom retail", "VRE"],
            "POW": ["pv power", "POW"],
        }
    
    def get_news_cafef(self, symbol: str = None, limit: int = 10) -> List[NewsArticle]:
        """Thu thập tin từ CafeF"""
        news = []
        try:
            # CafeF RSS hoặc API
            if symbol:
                url = f"https://cafef.vn/du-lieu/Ajax/Search.aspx?keyword={symbol}&type=1"
            else:
                url = "https://cafef.vn/chung-khoan.chn"
            
            # Giả lập dữ liệu (thực tế cần scrape hoặc dùng API)
            sample_news = self._get_sample_news(symbol)
            for item in sample_news[:limit]:
                sentiment, score, impact = self.analyzer.analyze(item["title"] + " " + item["summary"])
                news.append(NewsArticle(
                    title=item["title"],
                    summary=item["summary"],
                    url=item["url"],
                    source="CafeF",
                    published_at=item["date"],
                    symbol=symbol,
                    sentiment=sentiment,
                    sentiment_score=score,
                    impact_prediction=impact
                ))
        except Exception as e:
            print(f"Error fetching CafeF: {e}")
        return news
    
    def get_news_vndirect(self, symbol: str = None, limit: int = 10) -> List[NewsArticle]:
        """Thu thập tin từ VNDirect"""
        news = []
        try:
            if symbol:
                url = f"https://www.vndirect.com.vn/portal/tin-tuc/{symbol}.shtml"
            
            sample_news = self._get_sample_news(symbol, source="VNDirect")
            for item in sample_news[:limit]:
                sentiment, score, impact = self.analyzer.analyze(item["title"] + " " + item["summary"])
                news.append(NewsArticle(
                    title=item["title"],
                    summary=item["summary"],
                    url=item["url"],
                    source="VNDirect",
                    published_at=item["date"],
                    symbol=symbol,
                    sentiment=sentiment,
                    sentiment_score=score,
                    impact_prediction=impact
                ))
        except Exception as e:
            print(f"Error fetching VNDirect: {e}")
        return news
    
    def get_all_news(self, symbol: str = None, limit: int = 20) -> List[NewsArticle]:
        """Lấy tin từ tất cả nguồn"""
        all_news = []
        all_news.extend(self.get_news_cafef(symbol, limit // 2))
        all_news.extend(self.get_news_vndirect(symbol, limit // 2))
        
        # Sắp xếp theo thời gian
        all_news.sort(key=lambda x: x.published_at, reverse=True)
        return all_news[:limit]
    
    def get_sentiment_summary(self, symbol: str) -> Dict:
        """Tổng hợp sentiment cho một mã"""
        news = self.get_all_news(symbol, limit=20)
        
        if not news:
            return {
                "symbol": symbol,
                "total_news": 0,
                "sentiment": "neutral",
                "avg_score": 0,
                "positive": 0,
                "negative": 0,
                "neutral": 0,
                "recommendation": "Không có tin tức",
                "news": []
            }
        
        pos = sum(1 for n in news if n.sentiment == SentimentType.POSITIVE)
        neg = sum(1 for n in news if n.sentiment == SentimentType.NEGATIVE)
        neu = sum(1 for n in news if n.sentiment == SentimentType.NEUTRAL)
        avg_score = sum(n.sentiment_score for n in news) / len(news)
        
        # Xác định xu hướng
        if avg_score > 0.3:
            overall = "positive"
            rec = "🟢 XU HƯỚNG TÍCH CỰC - Cân nhắc MUA"
        elif avg_score < -0.3:
            overall = "negative"
            rec = "🔴 XU HƯỚNG TIÊU CỰC - Cân nhắc BÁN"
        else:
            overall = "neutral"
            rec = "🟡 TRUNG LẬP - Tiếp tục theo dõi"
        
        return {
            "symbol": symbol,
            "total_news": len(news),
            "sentiment": overall,
            "avg_score": round(avg_score, 2),
            "positive": pos,
            "negative": neg,
            "neutral": neu,
            "recommendation": rec,
            "news": [
                {
                    "title": n.title,
                    "summary": n.summary,
                    "url": n.url,
                    "source": n.source,
                    "published_at": n.published_at,
                    "sentiment": n.sentiment.value,
                    "score": round(n.sentiment_score, 2),
                    "impact": n.impact_prediction
                }
                for n in news[:10]
            ]
        }
    
    def _get_sample_news(self, symbol: str = None, source: str = "CafeF") -> List[Dict]:
        """Dữ liệu mẫu (thực tế sẽ scrape từ web)"""
        today = datetime.now()
        
        # Tin chung thị trường
        general_news = [
            {
                "title": "VN-Index tăng mạnh, thanh khoản đạt kỷ lục 25.000 tỷ đồng",
                "summary": "Thị trường chứng khoán phiên hôm nay chứng kiến đà tăng mạnh với dòng tiền đổ vào các cổ phiếu bluechip. Khối ngoại mua ròng hơn 500 tỷ đồng.",
                "url": f"https://{source.lower()}.vn/news/1",
                "date": (today - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M")
            },
            {
                "title": "Fed giữ nguyên lãi suất, chứng khoán châu Á tăng điểm",
                "summary": "Quyết định giữ nguyên lãi suất của Fed tạo tâm lý tích cực cho thị trường châu Á. Triển vọng dòng vốn ngoại vào Việt Nam được đánh giá khả quan.",
                "url": f"https://{source.lower()}.vn/news/2", 
                "date": (today - timedelta(hours=5)).strftime("%Y-%m-%d %H:%M")
            },
            {
                "title": "Cảnh báo rủi ro margin, nhiều nhà đầu tư bị call margin",
                "summary": "Thị trường giảm sâu khiến nhiều nhà đầu tư sử dụng đòn bẩy cao phải bán tháo cổ phiếu. Các công ty chứng khoán siết chặt tỷ lệ margin.",
                "url": f"https://{source.lower()}.vn/news/3",
                "date": (today - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
            },
        ]
        
        # Tin theo mã cổ phiếu
        stock_specific_news = {
            "VNM": [
                {
                    "title": "Vinamilk báo lãi kỷ lục quý 3, vượt 15% kế hoạch năm",
                    "summary": "CTCP Sữa Việt Nam (VNM) công bố lợi nhuận sau thuế quý 3 đạt 3.200 tỷ đồng, tăng trưởng 18% so với cùng kỳ. Doanh thu xuất khẩu tăng mạnh 25%.",
                    "url": f"https://{source.lower()}.vn/vnm/1",
                    "date": (today - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M")
                },
                {
                    "title": "VNM sẽ chia cổ tức tiền mặt 20%, tỷ lệ cao nhất ngành",
                    "summary": "HĐQT Vinamilk thông qua phương án chia cổ tức năm 2025 bằng tiền mặt với tỷ lệ 20%. Ngày chốt quyền dự kiến 15/12.",
                    "url": f"https://{source.lower()}.vn/vnm/2",
                    "date": (today - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
                },
            ],
            "VIC": [
                {
                    "title": "Vingroup đẩy mạnh đầu tư công nghệ, rót thêm 500 triệu USD cho VinFast",
                    "summary": "Tập đoàn Vingroup công bố kế hoạch tăng vốn đầu tư cho VinFast nhằm mở rộng thị trường xe điện tại Mỹ và châu Âu.",
                    "url": f"https://{source.lower()}.vn/vic/1",
                    "date": (today - timedelta(hours=4)).strftime("%Y-%m-%d %H:%M")
                },
            ],
            "HPG": [
                {
                    "title": "Hòa Phát: Giá thép phục hồi, lợi nhuận Q4 dự kiến tăng 30%",
                    "summary": "Giá thép trong nước và xuất khẩu tăng mạnh giúp biên lợi nhuận của Hòa Phát cải thiện đáng kể. Các CTCK nâng khuyến nghị từ Hold lên Buy.",
                    "url": f"https://{source.lower()}.vn/hpg/1",
                    "date": (today - timedelta(hours=6)).strftime("%Y-%m-%d %H:%M")
                },
                {
                    "title": "HPG bị điều tra bán phá giá tại thị trường EU",
                    "summary": "Ủy ban Châu Âu khởi xướng điều tra chống bán phá giá đối với thép cuộn cán nóng từ Việt Nam, trong đó có sản phẩm của Hòa Phát.",
                    "url": f"https://{source.lower()}.vn/hpg/2",
                    "date": (today - timedelta(days=2)).strftime("%Y-%m-%d %H:%M")
                },
            ],
            "VCB": [
                {
                    "title": "Vietcombank lãi kỷ lục hơn 40.000 tỷ đồng năm 2025",
                    "summary": "Ngân hàng TMCP Ngoại thương Việt Nam dự kiến đạt lợi nhuận trước thuế hơn 40.000 tỷ đồng, tăng 15% so với năm trước và là mức cao nhất ngành.",
                    "url": f"https://{source.lower()}.vn/vcb/1",
                    "date": (today - timedelta(hours=8)).strftime("%Y-%m-%d %H:%M")
                },
            ],
            "FPT": [
                {
                    "title": "FPT ký hợp đồng AI trị giá 200 triệu USD với đối tác Nhật Bản",
                    "summary": "Tập đoàn FPT công bố ký kết thỏa thuận hợp tác chiến lược về trí tuệ nhân tạo với tập đoàn công nghệ hàng đầu Nhật Bản, trị giá 200 triệu USD trong 5 năm.",
                    "url": f"https://{source.lower()}.vn/fpt/1",
                    "date": (today - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M")
                },
            ],
            "MWG": [
                {
                    "title": "Thế Giới Di Động đóng cửa thêm 100 cửa hàng, tái cơ cấu mạnh",
                    "summary": "MWG thông báo đóng cửa thêm 100 cửa hàng điện máy không hiệu quả trong Q4/2025. Công ty tập trung vào mảng Bách Hóa Xanh và online.",
                    "url": f"https://{source.lower()}.vn/mwg/1",
                    "date": (today - timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
                },
            ],
        }
        
        result = general_news.copy()
        
        if symbol and symbol.upper() in stock_specific_news:
            result = stock_specific_news[symbol.upper()] + result
        
        return result


# Singleton instance
news_service = NewsService()
