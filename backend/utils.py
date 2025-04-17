import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sqlite3
import json
import re
import requests
from bs4 import BeautifulSoup
import logging
from config import DATABASE_PATH

# Configure logging for image scraping
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
image_logger = logging.getLogger('ufc_images')

# File path for image URL cache
IMAGE_CACHE_PATH = os.path.join(os.path.dirname(DATABASE_PATH), 'fighter_images_cache.json')

def slugify_name(name):
    """
    Convert a fighter name to a slug for UFC.com URLs
    
    Args:
        name (str): Fighter name
        
    Returns:
        str: Slugified name for URL
    """
    if not name:
        return ""
    
    # Convert to lowercase
    slug = name.lower()
    
    # Remove apostrophes and other punctuation (except hyphens)
    slug = re.sub(r"['\"\(\)\.,:;]", "", slug)
    
    # Replace spaces with hyphens
    slug = re.sub(r"\s+", "-", slug)
    
    # Remove any remaining non-alphanumeric characters except hyphens
    slug = re.sub(r"[^a-z0-9\-]", "", slug)
    
    return slug

def load_image_cache():
    """
    Load the fighter image URL cache from disk
    
    Returns:
        dict: Mapping of fighter slugs to image URLs
    """
    if os.path.exists(IMAGE_CACHE_PATH):
        try:
            with open(IMAGE_CACHE_PATH, 'r') as f:
                return json.load(f)
        except Exception as e:
            image_logger.error(f"Error loading image cache: {e}")
    
    # Create a new cache if it doesn't exist or couldn't be loaded
    return {}

def save_image_cache(cache):
    """
    Save the fighter image URL cache to disk
    
    Args:
        cache (dict): Mapping of fighter slugs to image URLs
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(IMAGE_CACHE_PATH), exist_ok=True)
        
        with open(IMAGE_CACHE_PATH, 'w') as f:
            json.dump(cache, f)
    except Exception as e:
        image_logger.error(f"Error saving image cache: {e}")

def get_fighter_image_url(name, fetch_if_missing=False, source=None):
    """
    Get the image URL for a fighter with enhanced multi-source support
    
    Args:
        name (str): Fighter name
        fetch_if_missing (bool): Whether to fetch the image if it's not in the cache
        source (str, optional): Specific source to try ('ufc', 'sherdog', 'wikipedia'). Default is None (try all sources).
        
    Returns:
        str: URL to the fighter's image, or fallback image URL if not found
    """
    if not name:
        return "/static/placeholder.png"
    
    # Create slug for URL
    slug = slugify_name(name)
    
    # Check cache first (if no specific source is requested)
    if not source:
        cache = load_image_cache()
        if slug in cache:
            # Return cached URL (or fallback if "not_found" is stored)
            if cache[slug] == "not_found":
                return "/static/placeholder.png"
            return cache[slug]
    
    # Return placeholder if we're not supposed to fetch
    if not fetch_if_missing:
        return "/static/placeholder.png"
    
    # Prepare scraping
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    cache = load_image_cache()
    
    # Define sources to try
    sources_to_try = []
    if source:
        # If a specific source is requested, only try that one
        sources_to_try = [source]
    else:
        # Try all sources in sequence
        sources_to_try = ['ufc', 'sherdog', 'wikipedia']
    
    # Try each source
    for src in sources_to_try:
        try:
            image_url = None
            
            # UFC.com
            if src == 'ufc':
                image_logger.info(f"Trying UFC.com for {name}")
                ufc_url = f"https://ufc.com/athlete/{slug}"
                response = requests.get(ufc_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    img_tag = soup.find('img', class_='hero-profile__image')
                    
                    if img_tag and 'src' in img_tag.attrs:
                        image_url = img_tag['src']
                        image_logger.info(f"Found image for {name} on UFC.com")
            
            # Sherdog.com
            elif src == 'sherdog':
                image_logger.info(f"Trying Sherdog for {name}")
                
                # Convert name for Sherdog URL format
                sherdog_name = name.lower().replace(' ', '-')
                
                # Try direct name-based URL first
                sherdog_url = f"https://www.sherdog.com/image_crop/200/300/_images/fighter/{sherdog_name}.jpg"
                response = requests.head(sherdog_url, headers=headers, timeout=5)
                
                if response.status_code == 200:
                    image_url = sherdog_url
                    image_logger.info(f"Found image for {name} on Sherdog")
                else:
                    # Try a search on Sherdog as fallback
                    search_url = f"https://www.sherdog.com/stats/fightfinder?SearchTxt={name.replace(' ', '+')}"
                    response = requests.get(search_url, headers=headers, timeout=10)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        fighter_link = soup.select_one('table.fightfinder_result td a')
                        
                        if fighter_link and 'href' in fighter_link.attrs:
                            fighter_page = requests.get(fighter_link['href'], headers=headers, timeout=10)
                            if fighter_page.status_code == 200:
                                soup = BeautifulSoup(fighter_page.content, 'html.parser')
                                img = soup.select_one('img.profile_image')
                                if img and 'src' in img.attrs:
                                    image_url = "https://www.sherdog.com" + img['src']
                                    image_logger.info(f"Found image for {name} on Sherdog via search")
            
            # Wikipedia
            elif src == 'wikipedia':
                image_logger.info(f"Trying Wikipedia for {name}")
                wiki_search_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={name}+UFC+fighter&format=json"
                response = requests.get(wiki_search_url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    search_data = response.json()
                    if 'query' in search_data and 'search' in search_data['query'] and search_data['query']['search']:
                        # Get first result
                        page_id = search_data['query']['search'][0]['pageid']
                        
                        # Get page content
                        wiki_content_url = f"https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&pithumbsize=500&pageids={page_id}"
                        content_response = requests.get(wiki_content_url, headers=headers, timeout=10)
                        
                        if content_response.status_code == 200:
                            content_data = content_response.json()
                            if ('query' in content_data and 'pages' in content_data['query'] and 
                                str(page_id) in content_data['query']['pages'] and 
                                'thumbnail' in content_data['query']['pages'][str(page_id)]):
                                image_url = content_data['query']['pages'][str(page_id)]['thumbnail']['source']
                                image_logger.info(f"Found image for {name} on Wikipedia")
            
            # If we found an image from this source, save and return it
            if image_url:
                # Save to cache (if not source-specific request)
                if not source:
                    cache[slug] = image_url
                    save_image_cache(cache)
                
                return image_url
                
        except Exception as e:
            image_logger.error(f"Error fetching image for {name} from {src}: {e}")
    
    # If we get here, we tried all sources and couldn't find an image
    if not source:
        image_logger.warning(f"Could not find image for {name} from any source")
        cache[slug] = "not_found"
        save_image_cache(cache)
    
    return "/static/placeholder.png"

def validate_dataset(df):
    essential_columns = ['R_fighter', 'B_fighter', 'Winner']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing essential columns: {missing_columns}")
        
    # Check that Winner column has expected values
    valid_winners = ['Red', 'Blue', 'Draw']
    invalid_winners = set(df['Winner'].unique()) - set(valid_winners)
    
    if invalid_winners:
        raise ValueError(f"Invalid values in Winner column: {invalid_winners}")
        
    return True

def feature_engineering(df):
    """
    Create additional features from the raw fight data
    
    Parameters:
    df (pandas.DataFrame): DataFrame with fight data
    
    Returns:
    pandas.DataFrame: DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    df_features = df.copy()
    
    # Calculate striking accuracy
    if all(col in df_features.columns for col in ['fighter1_sig_strikes_landed', 'fighter1_sig_strikes_attempted']):
        df_features['fighter1_striking_accuracy'] = df_features['fighter1_sig_strikes_landed'] / df_features['fighter1_sig_strikes_attempted'].replace(0, 1)
    
    if all(col in df_features.columns for col in ['fighter2_sig_strikes_landed', 'fighter2_sig_strikes_attempted']):
        df_features['fighter2_striking_accuracy'] = df_features['fighter2_sig_strikes_landed'] / df_features['fighter2_sig_strikes_attempted'].replace(0, 1)
    
    # Calculate takedown accuracy
    if all(col in df_features.columns for col in ['fighter1_takedowns_landed', 'fighter1_takedowns_attempted']):
        df_features['fighter1_takedown_accuracy'] = df_features['fighter1_takedowns_landed'] / df_features['fighter1_takedowns_attempted'].replace(0, 1)
    
    if all(col in df_features.columns for col in ['fighter2_takedowns_landed', 'fighter2_takedowns_attempted']):
        df_features['fighter2_takedown_accuracy'] = df_features['fighter2_takedowns_landed'] / df_features['fighter2_takedowns_attempted'].replace(0, 1)
    
    # Calculate physical advantages
    if all(col in df_features.columns for col in ['fighter1_height', 'fighter2_height']):
        df_features['height_difference'] = df_features['fighter1_height'] - df_features['fighter2_height']
    
    if all(col in df_features.columns for col in ['fighter1_weight', 'fighter2_weight']):
        df_features['weight_difference'] = df_features['fighter1_weight'] - df_features['fighter2_weight']
    
    if all(col in df_features.columns for col in ['fighter1_reach', 'fighter2_reach']):
        df_features['reach_difference'] = df_features['fighter1_reach'] - df_features['fighter2_reach']
    
    # Calculate win rate features
    if all(col in df_features.columns for col in ['fighter1_wins', 'fighter1_losses']):
        df_features['fighter1_win_rate'] = df_features['fighter1_wins'] / (df_features['fighter1_wins'] + df_features['fighter1_losses'] + 1e-5)
    
    if all(col in df_features.columns for col in ['fighter2_wins', 'fighter2_losses']):
        df_features['fighter2_win_rate'] = df_features['fighter2_wins'] / (df_features['fighter2_wins'] + df_features['fighter2_losses'] + 1e-5)
    
    # Calculate age-related features
    if all(col in df_features.columns for col in ['fighter1_age', 'fighter2_age']):
        df_features['age_difference'] = df_features['fighter1_age'] - df_features['fighter2_age']
    
    return df_features

def get_fighter_stats(fighter_name):
    """
    Get the latest stats for a fighter from the database
    
    Parameters:
    fighter_name (str): Name of the fighter
    
    Returns:
    dict: Dictionary containing the fighter's stats
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get fighter basic info
        cursor.execute('''
        SELECT * FROM fighters WHERE name = ?
        ''', (fighter_name,))
        
        fighter = cursor.fetchone()
        
        if not fighter:
            return None
        
        # Get fighter's latest fight stats
        cursor.execute('''
        SELECT fs.* 
        FROM fight_stats fs
        JOIN fighters f ON fs.fighter_id = f.id
        JOIN fights ft ON fs.fight_id = ft.id
        WHERE f.name = ?
        ORDER BY ft.date DESC
        LIMIT 1
        ''', (fighter_name,))
        
        stats = cursor.fetchone()
        
        # Combine data
        fighter_dict = dict(fighter)
        
        if stats:
            stats_dict = dict(stats)
            fighter_dict.update(stats_dict)
        
        conn.close()
        return fighter_dict
    
    except Exception as e:
        print(f"Error getting fighter stats: {str(e)}")
        return None

def calculate_elo_ratings(fight_data, initial_elo=1500, k_factor=32):
    """
    Calculate Elo ratings for all fighters based on their fight history
    
    Parameters:
    fight_data (pandas.DataFrame): DataFrame with fight data
    initial_elo (int): Initial Elo rating for new fighters
    k_factor (int): K-factor for Elo calculation
    
    Returns:
    dict: Dictionary of fighter names to their current Elo rating
    """
    # Sort fights by date
    if 'date' in fight_data.columns:
        fight_data = fight_data.sort_values('date')
    
    # Initialize Elo ratings
    elo_ratings = {}
    
    for _, fight in fight_data.iterrows():
        fighter1 = fight['fighter1_name']
        fighter2 = fight['fighter2_name']
        winner = fight['winner_name']
        
        # Add fighters if they're not in the ratings yet
        if fighter1 not in elo_ratings:
            elo_ratings[fighter1] = initial_elo
        
        if fighter2 not in elo_ratings:
            elo_ratings[fighter2] = initial_elo
        
        # Calculate expected outcome
        r1 = elo_ratings[fighter1]
        r2 = elo_ratings[fighter2]
        
        e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        e2 = 1 / (1 + 10 ** ((r1 - r2) / 400))
        
        # Determine actual outcome
        if winner == fighter1:
            s1, s2 = 1, 0
        elif winner == fighter2:
            s1, s2 = 0, 1
        else:
            s1, s2 = 0.5, 0.5  # Draw
        
        # Update ratings
        elo_ratings[fighter1] = r1 + k_factor * (s1 - e1)
        elo_ratings[fighter2] = r2 + k_factor * (s2 - e2)
    
    return elo_ratings

def export_prediction_data(prediction_data, export_path='data/predictions.csv'):
    """
    Export prediction data to CSV
    
    Parameters:
    prediction_data (list): List of prediction dictionaries
    export_path (str): Path to export the CSV file
    
    Returns:
    str: Path to the exported file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(export_path), exist_ok=True)
    
    # Create DataFrame from prediction data
    df = pd.DataFrame(prediction_data)
    
    # Export to CSV
    df.to_csv(export_path, index=False)
    
    return export_path