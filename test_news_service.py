"""Test news service"""
from src.news_service import NewsService

print("Testing NewsService...")
ns = NewsService()

# Test 1: Get all news
print("\n1. Testing get_all_news()...")
try:
    news = ns.get_all_news(symbol=None, limit=5)
    print(f"✓ Found {len(news)} news articles")
    for i, n in enumerate(news[:3], 1):
        print(f"  {i}. [{n.source}] {n.title[:70]}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: Get news for VNM
print("\n2. Testing get_all_news(symbol='VNM')...")
try:
    news_vnm = ns.get_all_news(symbol='VNM', limit=5)
    print(f"✓ Found {len(news_vnm)} news for VNM")
    for i, n in enumerate(news_vnm[:3], 1):
        print(f"  {i}. {n.title[:70]}")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Check RSS feeds individually
print("\n3. Testing individual RSS feeds...")
for feed_name, feed_url in list(ns.RSS_FEEDS.items())[:3]:
    try:
        news = ns.fetch_rss(feed_name, feed_url, limit=2)
        print(f"✓ {feed_name}: {len(news)} articles")
    except Exception as e:
        print(f"✗ {feed_name}: {e}")

print("\n" + "="*70)
print("Test complete!")
