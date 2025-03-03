import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sqlite3
from config import DATABASE_PATH

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