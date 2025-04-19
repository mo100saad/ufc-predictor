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
    """Fetch UFC news from NewsAPI with focused UFC/MMA-related keywords"""
    # First check if we have valid cached news
    cached_news = get_cached_news()
    if cached_news:
        return cached_news
    
    try:
        # Use more specific search terms to ensure relevance over recency
        params = {
            # Use more specific search terms focused on MMA/UFC content
            'q': '"UFC" OR "Ultimate Fighting Championship" OR "MMA" OR "Dana White" OR "octagon" OR "UFC fighter" OR "UFC champion" OR "Wrestling" OR "BJJ" OR "Muay Thai" OR "Bellator"',
            'sortBy': 'relevancy',  # Switch to relevancy instead of publishing date for better results
            'language': 'en',
            'pageSize': 30,  # Get more articles to have better filtering options
            'apiKey': NEWS_API_KEY
        }
        
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('status') == 'ok':
            articles = data.get('articles', [])
            
            # Process and filter articles to ensure they're UFC/MMA-related
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
                
                # Use organized keyword matching to ensure relevance
                # Primary keywords are MMA-specific terms
                primary_keywords = ['ufc', 'mma', 'dana white', 'octagon', 'bellator']
                
                # Secondary keywords include combat sports disciplines
                secondary_keywords = ['wrestling', 'bjj', 'brazilian jiu-jitsu', 'muay thai', 'submission', 'knockout']
                
                # Tertiary keywords are more general fight terms
                tertiary_keywords = ['champion', 'title fight', 'fighter', 'bout', 'pay-per-view']
                
                title_lower = processed_article['title'].lower()
                desc_lower = processed_article['description'].lower() if processed_article['description'] else ""
                source_lower = processed_article['source'].lower()
                
                # Scoring system for article relevance:
                # 1. Primary keywords score highest
                # 2. Secondary keywords (combat sports) are next most valuable
                # 3. Tertiary keywords provide additional matching
                
                primary_matches = sum(1 for keyword in primary_keywords if keyword in title_lower or keyword in desc_lower)
                secondary_matches = sum(1 for keyword in secondary_keywords if keyword in title_lower or keyword in desc_lower)
                tertiary_matches = sum(1 for keyword in tertiary_keywords if keyword in title_lower or keyword in desc_lower)
                
                # Check if the source is an MMA-focused outlet
                mma_sources = ['ufc', 'mma', 'sherdog', 'tapology', 'bjpenn', 'bloodyelbow', 'mmafighting', 'mmajunkie']
                is_mma_source = any(source in source_lower for source in mma_sources)
                
                # Calculate relevance score
                relevance_score = (primary_matches * 3) + (secondary_matches * 2) + tertiary_matches + (5 if is_mma_source else 0)
                
                # Include articles with primary or secondary keywords or from MMA sources
                if primary_matches > 0 or secondary_matches > 0 or is_mma_source:
                    processed_article['relevance_score'] = relevance_score
                    processed_articles.append(processed_article)
            
            # Sort by relevance score, then by date
            processed_articles.sort(key=lambda x: (-(x.get('relevance_score', 0)), 
                                                 -datetime.fromisoformat(x['publishedAt'].replace('Z', '+00:00')).timestamp()))
            
            # Take up to 10 most relevant articles
            processed_articles = processed_articles[:10]
            
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