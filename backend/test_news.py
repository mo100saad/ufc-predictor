#!/usr/bin/env python3

import requests
import json
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

# News API settings 
NEWS_API_KEY = os.environ.get("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_ufc_news():
    """Fetch UFC news from NewsAPI directly"""
    try:
        # Fetch fresh news
        params = {
            'q': 'UFC OR "Ultimate Fighting Championship" OR MMA',
            'sortBy': 'publishedAt',
            'language': 'en',
            'pageSize': 5,  # Get just a few articles for testing
            'apiKey': NEWS_API_KEY
        }
        
        print(f"Sending request to {NEWS_API_URL} with params: {params}")
        response = requests.get(NEWS_API_URL, params=params)
        data = response.json()
        
        if response.status_code == 200 and data.get('status') == 'ok':
            articles = data.get('articles', [])
            print(f"Successfully fetched {len(articles)} articles")
            
            # Just print the titles for testing
            for i, article in enumerate(articles):
                print(f"{i+1}. {article.get('title')}")
                print(f"   Source: {article.get('source', {}).get('name', '')}")
                print(f"   Published: {article.get('publishedAt', '')}")
                print()
            
            return articles
        else:
            print(f"Error fetching UFC news: {data.get('message', 'Unknown error')}")
            print(f"Status code: {response.status_code}")
            return []
    
    except Exception as e:
        print(f"Exception fetching UFC news: {e}")
        return []

if __name__ == "__main__":
    print("Testing NewsAPI direct connection...")
    articles = fetch_ufc_news()
    print(f"\nTotal articles: {len(articles)}")