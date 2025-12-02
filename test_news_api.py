"""Test API endpoints"""
import requests
import json

BASE_URL = "http://localhost:8000"

print("Testing News API Endpoints")
print("="*70)

# Test 1: Get market news
print("\n1. GET /api/news (market news)")
try:
    response = requests.get(f"{BASE_URL}/api/news?limit=10", timeout=30)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {data.get('status')}")
        print(f"✓ Total: {data.get('total')} news")
        if data.get('news'):
            for i, news in enumerate(data['news'][:3], 1):
                print(f"  {i}. [{news['source']}] {news['title'][:70]}")
                print(f"     Sentiment: {news['sentiment']} (score: {news['sentiment_score']})")
    else:
        print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 2: Get stock-specific news
print("\n2. GET /api/news/VNM (VNM news)")
try:
    response = requests.get(f"{BASE_URL}/api/news/VNM?limit=10", timeout=30)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {data.get('status')}")
        print(f"✓ Total news: {data.get('total_news')}")
        if data.get('sentiment_summary'):
            summary = data['sentiment_summary']
            print(f"✓ Sentiment: {summary.get('overall')} (avg: {summary.get('avg_score', 0):.2f})")
            print(f"  Positive: {summary.get('positive_count')}")
            print(f"  Negative: {summary.get('negative_count')}")
            print(f"  Neutral: {summary.get('neutral_count')}")
        if data.get('news'):
            print(f"\nTop 3 news:")
            for i, news in enumerate(data['news'][:3], 1):
                print(f"  {i}. {news['title'][:70]}")
                print(f"     Relevance: {news.get('relevance_score', 0):.2f} | Sentiment: {news['sentiment']}")
    else:
        print(f"✗ HTTP {response.status_code}: {response.text[:200]}")
except Exception as e:
    print(f"✗ Error: {e}")

# Test 3: Sentiment features
print("\n3. GET /api/news/features/sentiment")
try:
    response = requests.get(f"{BASE_URL}/api/news/features/sentiment", timeout=10)
    if response.status_code == 200:
        data = response.json()
        print(f"✓ Status: {data.get('status')}")
        print(f"✓ Method: {data.get('method')}")
        if data.get('features'):
            pos = data['features'].get('positive_keywords', {})
            neg = data['features'].get('negative_keywords', {})
            print(f"  Positive keywords: {pos.get('count', 0)}")
            print(f"  Negative keywords: {neg.get('count', 0)}")
    else:
        print(f"✗ HTTP {response.status_code}")
except Exception as e:
    print(f"✗ Error: {e}")

print("\n" + "="*70)
print("Test complete!")
