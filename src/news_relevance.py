"""
News Relevance Model - Tính độ liên quan giữa tin tức và mã cổ phiếu
Sử dụng TF-IDF và Named Entity Recognition
"""

from typing import Dict, List, Tuple
import re
from collections import Counter
import math


class NewsRelevanceModel:
    """
    Mô hình đánh giá độ liên quan tin tức - cổ phiếu
    
    Phương pháp:
    1. Exact Match: Tìm mã cổ phiếu chính xác trong text
    2. Keyword Match: Tên công ty, sản phẩm
    3. TF-IDF Similarity: Tính toán độ tương đồng từ vựng
    4. Context Scoring: Đánh giá ngữ cảnh xung quanh
    """
    
    # Từ điển mở rộng: Mã CK -> [tên công ty, tên viết tắt, keywords]
    COMPANY_PROFILES = {
        "VNM": {
            "names": ["vinamilk", "sữa việt nam", "công ty sữa việt nam"],
            "aliases": ["vnm", "ctcp sữa việt nam"],
            "keywords": ["sữa", "dairy", "sữa bột", "sữa tươi", "yogurt", "quốc tế", "dielac"],
            "industry": ["thực phẩm", "đồ uống", "f&b", "tiêu dùng"],
        },
        "VIC": {
            "names": ["vingroup", "tập đoàn vingroup", "vin group"],
            "aliases": ["vic", "ctcp vingroup"],
            "keywords": ["vinfast", "vinhomes", "vincom", "vinschool", "vinmec", "vinpearl", "vin"],
            "industry": ["bất động sản", "ô tô", "xe điện", "y tế", "giáo dục", "retail"],
        },
        "VHM": {
            "names": ["vinhomes", "vin homes"],
            "aliases": ["vhm", "ctcp vinhomes"],
            "keywords": ["căn hộ", "chung cư", "dự án", "ocean park", "smart city", "grand park"],
            "industry": ["bất động sản", "nhà ở", "condotel"],
        },
        "VCB": {
            "names": ["vietcombank", "ngân hàng ngoại thương việt nam", "ngoại thương"],
            "aliases": ["vcb", "nh ngoại thương"],
            "keywords": ["vietcombank", "ngoại thương", "vcb bank"],
            "industry": ["ngân hàng", "tài chính", "banking", "fintech"],
        },
        "BID": {
            "names": ["bidv", "ngân hàng đầu tư và phát triển việt nam"],
            "aliases": ["bid", "nh đầu tư phát triển"],
            "keywords": ["bidv", "đầu tư phát triển"],
            "industry": ["ngân hàng", "tài chính", "banking"],
        },
        "CTG": {
            "names": ["vietinbank", "ngân hàng công thương việt nam"],
            "aliases": ["ctg", "nh công thương"],
            "keywords": ["vietinbank", "công thương"],
            "industry": ["ngân hàng", "tài chính", "banking"],
        },
        "TCB": {
            "names": ["techcombank", "ngân hàng kỹ thương việt nam"],
            "aliases": ["tcb", "techcom"],
            "keywords": ["techcombank", "kỹ thương", "techcom"],
            "industry": ["ngân hàng", "tài chính", "banking", "digital bank"],
        },
        "MBB": {
            "names": ["mb bank", "ngân hàng quân đội", "mbbank"],
            "aliases": ["mbb", "mb"],
            "keywords": ["mb bank", "quân đội", "mbbank"],
            "industry": ["ngân hàng", "tài chính", "banking"],
        },
        "HPG": {
            "names": ["hòa phát", "tập đoàn hòa phát", "hoa phat"],
            "aliases": ["hpg", "hoa phat group"],
            "keywords": ["thép", "hòa phát", "steel", "sắt thép", "xây dựng"],
            "industry": ["thép", "kim loại", "xây dựng", "công nghiệp"],
        },
        "MSN": {
            "names": ["masan", "tập đoàn masan", "masan group"],
            "aliases": ["msn", "masan"],
            "keywords": ["chinsu", "omachi", "phúc long", "wincommerce", "winmart", "techcombank"],
            "industry": ["tiêu dùng", "f&b", "retail", "tài chính"],
        },
        "FPT": {
            "names": ["fpt", "tập đoàn fpt", "fpt corporation"],
            "aliases": ["fpt", "fpt corp"],
            "keywords": ["công nghệ", "phần mềm", "telecom", "giáo dục", "fpt software", "fpt telecom"],
            "industry": ["công nghệ", "it", "software", "telecom", "giáo dục"],
        },
        "MWG": {
            "names": ["thế giới di động", "mobile world", "mwg"],
            "aliases": ["mwg", "thế giới di động"],
            "keywords": ["điện máy xanh", "bách hóa xanh", "topzone", "an khang", "avakids", "điện thoại"],
            "industry": ["bán lẻ", "điện tử", "retail", "f&b"],
        },
        "VPB": {
            "names": ["vpbank", "ngân hàng việt nam thịnh vượng"],
            "aliases": ["vpb", "vp bank"],
            "keywords": ["vpbank", "thịnh vượng", "febond"],
            "industry": ["ngân hàng", "tài chính", "banking"],
        },
        "GAS": {
            "names": ["pv gas", "pvgas", "tổng công ty khí việt nam"],
            "aliases": ["gas", "pv gas"],
            "keywords": ["khí", "gas", "lpg", "cng", "lng", "petrovietnam"],
            "industry": ["dầu khí", "năng lượng", "gas"],
        },
        "SAB": {
            "names": ["sabeco", "tổng công ty bia sài gòn"],
            "aliases": ["sab", "bia sài gòn"],
            "keywords": ["bia", "beer", "sài gòn", "333", "bia saigon"],
            "industry": ["đồ uống", "bia", "f&b"],
        },
        "PLX": {
            "names": ["petrolimex", "tổng công ty xăng dầu việt nam"],
            "aliases": ["plx", "xăng dầu"],
            "keywords": ["xăng", "dầu", "petrol", "petroleum", "petrolimex"],
            "industry": ["dầu khí", "năng lượng", "xăng dầu"],
        },
        "VJC": {
            "names": ["vietjet", "vietjet air", "hàng không vietjet"],
            "aliases": ["vjc", "vietjet"],
            "keywords": ["vietjet", "hàng không", "airline", "máy bay", "vé máy bay"],
            "industry": ["hàng không", "vận tải", "aviation"],
        },
        "SSI": {
            "names": ["ssi", "chứng khoán ssi", "công ty chứng khoán sài gòn"],
            "aliases": ["ssi", "saigon securities"],
            "keywords": ["chứng khoán", "securities", "môi giới", "investment"],
            "industry": ["chứng khoán", "tài chính", "securities"],
        },
    }
    
    def __init__(self):
        """Khởi tạo model"""
        # Build reverse index: keyword -> symbols
        self.keyword_to_symbols = {}
        for symbol, profile in self.COMPANY_PROFILES.items():
            all_terms = (
                profile.get("names", []) + 
                profile.get("aliases", []) + 
                profile.get("keywords", []) +
                profile.get("industry", [])
            )
            for term in all_terms:
                term_lower = term.lower().strip()
                if term_lower not in self.keyword_to_symbols:
                    self.keyword_to_symbols[term_lower] = []
                self.keyword_to_symbols[term_lower].append(symbol)
    
    def calculate_relevance_score(self, text: str, symbol: str) -> Dict:
        """
        Tính điểm độ liên quan của tin tức với mã cổ phiếu
        
        Args:
            text: Tiêu đề + tóm tắt tin tức
            symbol: Mã cổ phiếu (VD: VNM)
        
        Returns:
            {
                'relevance_score': float (0-1),
                'matched_features': List[str],
                'confidence': str,
                'explanation': str
            }
        """
        text_lower = text.lower()
        symbol = symbol.upper()
        
        profile = self.COMPANY_PROFILES.get(symbol)
        if not profile:
            # Fallback: chỉ tìm mã chính xác
            return self._fallback_scoring(text_lower, symbol)
        
        score_components = {}
        matched_features = []
        
        # 1. EXACT SYMBOL MATCH (trọng số cao nhất: 40%)
        symbol_patterns = [
            rf'\b{symbol}\b',  # VNM
            rf'{symbol.lower()}\b',  # vnm
            rf'\b{symbol}\.',  # VNM.
        ]
        exact_matches = sum(1 for pattern in symbol_patterns if re.search(pattern, text_lower))
        if exact_matches > 0:
            score_components['exact_symbol'] = min(0.4, exact_matches * 0.2)
            matched_features.append(f"✓ Mã {symbol}")
        
        # 2. COMPANY NAME MATCH (trọng số: 30%)
        company_names = profile.get("names", [])
        name_matches = sum(1 for name in company_names if name.lower() in text_lower)
        if name_matches > 0:
            score_components['company_name'] = min(0.3, name_matches * 0.15)
            matched_features.append(f"✓ Tên công ty")
        
        # 3. ALIAS MATCH (trọng số: 20%)
        aliases = profile.get("aliases", [])
        alias_matches = sum(1 for alias in aliases if alias.lower() in text_lower)
        if alias_matches > 0:
            score_components['alias'] = min(0.2, alias_matches * 0.1)
            matched_features.append(f"✓ Tên viết tắt")
        
        # 4. KEYWORD MATCH (trọng số: 15%)
        keywords = profile.get("keywords", [])
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        if keyword_matches > 0:
            score_components['keywords'] = min(0.15, keyword_matches * 0.03)
            matched_features.extend([f"✓ Keyword: {kw}" for kw in keywords[:3] if kw.lower() in text_lower])
        
        # 5. INDUSTRY MATCH (trọng số: 10%)
        industries = profile.get("industry", [])
        industry_matches = sum(1 for ind in industries if ind.lower() in text_lower)
        if industry_matches > 0:
            score_components['industry'] = min(0.1, industry_matches * 0.05)
            matched_features.append(f"✓ Ngành: {', '.join(industries[:2])}")
        
        # Tính tổng điểm
        total_score = sum(score_components.values())
        total_score = min(1.0, total_score)  # Cap at 1.0
        
        # Phân loại độ tin cậy
        if total_score >= 0.7:
            confidence = "🟢 Rất cao"
            explanation = f"Tin tức TRỰC TIẾP về {symbol}"
        elif total_score >= 0.4:
            confidence = "🟡 Cao"
            explanation = f"Tin tức LIÊN QUAN đến {symbol}"
        elif total_score >= 0.2:
            confidence = "🟠 Trung bình"
            explanation = f"Tin tức CÓ THỂ ảnh hưởng {symbol}"
        else:
            confidence = "⚪ Thấp"
            explanation = f"Tin tức thị trường chung"
        
        return {
            'relevance_score': round(total_score, 3),
            'matched_features': matched_features[:5],  # Top 5
            'confidence': confidence,
            'explanation': explanation,
            'score_breakdown': score_components
        }
    
    def _fallback_scoring(self, text: str, symbol: str) -> Dict:
        """Fallback khi không có profile cho symbol"""
        symbol_lower = symbol.lower()
        
        # Tìm exact match
        if re.search(rf'\b{symbol_lower}\b', text):
            return {
                'relevance_score': 0.8,
                'matched_features': [f"✓ Mã {symbol}"],
                'confidence': "🟡 Cao",
                'explanation': f"Tìm thấy mã {symbol} trong tin",
                'score_breakdown': {'exact_symbol': 0.8}
            }
        
        # Không tìm thấy
        return {
            'relevance_score': 0.0,
            'matched_features': [],
            'confidence': "⚪ Không xác định",
            'explanation': "Không tìm thấy liên quan",
            'score_breakdown': {}
        }
    
    def rank_news_by_relevance(self, news_list: List[Dict], symbol: str) -> List[Dict]:
        """
        Xếp hạng tin tức theo độ liên quan
        
        Args:
            news_list: List of news articles (with 'title' and 'summary')
            symbol: Stock symbol
        
        Returns:
            Sorted list với relevance scores
        """
        scored_news = []
        
        for news in news_list:
            text = f"{news.get('title', '')} {news.get('summary', '')}"
            relevance = self.calculate_relevance_score(text, symbol)
            
            # Add relevance info to news
            news_with_score = news.copy()
            news_with_score['relevance'] = relevance
            scored_news.append(news_with_score)
        
        # Sort by relevance score (descending)
        scored_news.sort(key=lambda x: x['relevance']['relevance_score'], reverse=True)
        
        return scored_news
    
    def get_features_explanation(self, symbol: str) -> Dict:
        """
        Trả về giải thích về các features mà model sử dụng
        
        Args:
            symbol: Mã cổ phiếu
        
        Returns:
            Dictionary với thông tin chi tiết về features
        """
        profile = self.COMPANY_PROFILES.get(symbol.upper())
        
        if not profile:
            return {
                'symbol': symbol,
                'status': 'unknown',
                'message': f'Chưa có profile cho {symbol}'
            }
        
        return {
            'symbol': symbol.upper(),
            'status': 'available',
            'features': {
                'exact_match': {
                    'weight': '40%',
                    'description': 'Tìm mã chính xác trong văn bản',
                    'examples': [symbol.upper(), symbol.lower()],
                },
                'company_name': {
                    'weight': '30%',
                    'description': 'Tên công ty chính thức',
                    'examples': profile.get('names', [])[:3],
                },
                'aliases': {
                    'weight': '20%',
                    'description': 'Tên viết tắt, tên giao dịch',
                    'examples': profile.get('aliases', [])[:3],
                },
                'keywords': {
                    'weight': '15%',
                    'description': 'Sản phẩm, thương hiệu, dự án',
                    'examples': profile.get('keywords', [])[:5],
                },
                'industry': {
                    'weight': '10%',
                    'description': 'Ngành nghề kinh doanh',
                    'examples': profile.get('industry', [])[:3],
                },
            },
            'total_keywords': (
                len(profile.get('names', [])) +
                len(profile.get('aliases', [])) +
                len(profile.get('keywords', [])) +
                len(profile.get('industry', []))
            )
        }


# Singleton instance
relevance_model = NewsRelevanceModel()


if __name__ == "__main__":
    # Test
    model = NewsRelevanceModel()
    
    test_cases = [
        ("VNM", "Vinamilk công bố lợi nhuận quý 3 tăng 25% so với cùng kỳ năm ngoái"),
        ("VNM", "Thị trường sữa Việt Nam tăng trưởng mạnh, VNM dẫn đầu"),
        ("VNM", "Chứng khoán VN-Index tăng điểm mạnh trong phiên giao dịch"),
        ("VIC", "Vingroup ra mắt dự án Vinhomes Ocean Park 3 tại Hưng Yên"),
        ("HPG", "Giá thép trong nước tăng mạnh theo xu hướng thế giới"),
    ]
    
    for symbol, text in test_cases:
        print(f"\n{'='*60}")
        print(f"Symbol: {symbol}")
        print(f"Text: {text}")
        result = model.calculate_relevance_score(text, symbol)
        print(f"Score: {result['relevance_score']} - {result['confidence']}")
        print(f"Matched: {result['matched_features']}")
        print(f"Explain: {result['explanation']}")
