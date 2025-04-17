#!/usr/bin/env python3
from flask import Flask, jsonify
import requests
import json
import os
from datetime import datetime

app = Flask(__name__)

# Import configuration with environment variables
from config import NEWS_API_KEY

# News API settings
NEWS_API_URL = "https://newsapi.org/v2/everything"
NEWS_CACHE_FILE = os.path.join(os.path.dirname(__file__), 'news/news_cache.json')
NEWS_CACHE_DURATION = 3600  # Cache news for 1 hour

def get_cached_news():
    """Get cached news if available and not expired"""
    if os.path.exists(NEWS_CACHE_FILE):
        try:
            with open(NEWS_CACHE_FILE, 'r') as f:
                cache = json.load(f)
            
            # Check if cache is expired
            if cache.get('last_updated'):
                last_updated = datetime.fromisoformat(cache['last_updated'])
                now = datetime.now()
                time_diff = (now - last_updated).total_seconds()
                
                if time_diff < NEWS_CACHE_DURATION:
                    print(f"Returning news from cache, age: {time_diff/60:.1f} minutes")
                    return cache.get('articles', [])
                
            print("News cache expired or invalid")
        except Exception as e:
            print(f"Error reading news cache: {e}")
    
    return None

def save_news_cache(articles):
    """Save news to cache with timestamp"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(NEWS_CACHE_FILE), exist_ok=True)
        
        cache = {
            'last_updated': datetime.now().isoformat(),
            'articles': articles
        }
        
        with open(NEWS_CACHE_FILE, 'w') as f:
            json.dump(cache, f)
            
        print(f"Saved {len(articles)} news articles to cache")
    except Exception as e:
        print(f"Error saving news cache: {e}")

def fetch_ufc_news():
    """Fetch UFC news from NewsAPI"""
    # First check if we have valid cached news
    cached_news = get_cached_news()
    if cached_news:
        return cached_news
    
    try:
        # Fetch fresh news
        params = {
            'q': 'UFC OR "Ultimate Fighting Championship" OR MMA',
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 12,  # Get more articles to filter for quality
            'apiKey': NEWS_API_KEY
        }
        
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('status') == 'ok':
            articles = data.get('articles', [])
            
            # Process and filter articles to ensure they're UFC-related
            processed_articles = []
            for article in articles:
                # Skip articles without an image
                if not article.get('urlToImage'):
                    continue
                    
                # Extract relevant fields and add a unique ID
                processed_article = {
                    'id': hash(article.get('url', '') + article.get('publishedAt', '')),
                    'title': article.get('title', ''),
                    'description': article.get('description', ''),
                    'url': article.get('url', ''),
                    'imageUrl': article.get('urlToImage', ''),
                    'source': article.get('source', {}).get('name', ''),
                    'publishedAt': article.get('publishedAt', '')
                }
                
                # Only add good quality news about UFC
                if (
                    'ufc' in processed_article['title'].lower() or 
                    'mma' in processed_article['title'].lower() or
                    'ultimate fighting championship' in processed_article['title'].lower()
                ):
                    processed_articles.append(processed_article)
            
            # Take up to 8 articles
            processed_articles = processed_articles[:8]
            
            # Save to cache
            save_news_cache(processed_articles)
            
            return processed_articles
        else:
            print(f"Error fetching UFC news: {data.get('message', 'Unknown error')}")
            return []
    
    except Exception as e:
        print(f"Exception fetching UFC news: {e}")
        return []

@app.route('/')
def home():
    return "UFC News API is running! Try /news"

@app.route('/news')
def get_news():
    news = fetch_ufc_news()
    return jsonify({
        "news": news,
        "count": len(news),
        "timestamp": datetime.now().isoformat()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)