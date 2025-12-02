"""Debug RSS feeds"""
import requests
import xml.etree.ElementTree as ET

feeds = {
    "CafeF": "https://cafef.vn/rss/chung-khoan.rss",
    "VnEconomy": "https://vneconomy.vn/chung-khoan.rss",
    "VTV": "https://vtv.vn/kinh-te.rss",
    "TuoiTre": "https://tuoitre.vn/rss/kinh-doanh.rss",
}

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}

for name, url in feeds.items():
    print(f"\n{'='*70}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Status: {response.status_code}")
        print(f"Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"Length: {len(response.content)} bytes")
        
        if response.status_code == 200:
            try:
                root = ET.fromstring(response.content)
                items = root.findall('.//item')
                print(f"✓ Found {len(items)} items")
                
                if items:
                    first = items[0]
                    title = first.find('title')
                    if title is not None:
                        print(f"First title: {title.text[:70]}...")
            except ET.ParseError as e:
                print(f"✗ XML Parse Error: {e}")
                print(f"First 500 chars: {response.text[:500]}")
        else:
            print(f"✗ HTTP Error")
            
    except Exception as e:
        print(f"✗ Error: {e}")
