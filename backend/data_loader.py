import pandas as pd
import os
import requests
from zipfile import ZipFile
from io import BytesIO
import numpy as np
from config import CSV_FILE_PATH

def download_ufc_dataset(save_path=CSV_FILE_PATH):
    """
    Download a UFC dataset from a sample URL or Kaggle
    Note: For Kaggle, you would need to use the Kaggle API with credentials
    """
    # Sample URL - replace with actual dataset URL if available
    url = "https://raw.githubusercontent.com/sample/ufc-dataset/main/ufc_data.csv"
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Try to download the file
        print(f"Downloading UFC dataset to {save_path}...")
        response = requests.get(url)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            f.write(response.content)
        
        print("Dataset downloaded successfully!")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        print("Creating a sample dataset instead...")
        create_sample_dataset(save_path)
        return False

def create_sample_dataset(save_path=CSV_FILE_PATH):
    """
    Create a sample UFC dataset if download fails
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate fighter names
    first_names = ['Alex', 'Jon', 'Conor', 'Khabib', 'Francis', 'Amanda', 'Cris', 'Valentina', 'Israel', 'Max']
    last_names = ['Smith', 'Jones', 'McGregor', 'Nurmagomedov', 'Ngannou', 'Nunes', 'Cyborg', 'Shevchenko', 'Adesanya', 'Holloway']
    
    fighter_names = [f"{fn} {ln}" for fn in first_names for ln in last_names][:50]  # Generate 50 unique fighters
    
    # Generate fighter stats
    fighters = []
    for name in fighter_names:
        fighters.append({
            'name': name,
            'height': np.random.uniform(160, 195),  # cm
            'weight': np.random.uniform(55, 120),   # kg
            'reach': np.random.uniform(160, 210),   # cm
            'stance': np.random.choice(['Orthodox', 'Southpaw', 'Switch']),
            'wins': np.random.randint(5, 30),
            'losses': np.random.randint(0, 10),
            'draws': np.random.randint(0, 3),
            'sig_strikes_per_min': np.random.uniform(1.5, 7.0),
            'takedown_avg': np.random.uniform(0, 5.0),
            'sub_avg': np.random.uniform(0, 2.0),
            'win_streak': np.random.randint(0, 8)
        })
    
    # Generate fights
    fights = []
    weight_classes = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 
                      'Middleweight', 'Light Heavyweight', 'Heavyweight']
    methods = ['KO/TKO', 'Submission', 'Decision - Unanimous', 'Decision - Split', 'Decision - Majority']
    
    for i in range(500):  # Generate 500 fights
        # Select two different fighters
        idx1, idx2 = np.random.choice(len(fighters), 2, replace=False)
        fighter1 = fighters[idx1]
        fighter2 = fighters[idx2]
        
        # Determine winner based on random probability influenced by stats
        fighter1_score = (
            fighter1['sig_strikes_per_min'] * 0.3 + 
            fighter1['takedown_avg'] * 0.2 + 
            fighter1['win_streak'] * 0.2 + 
            fighter1['wins'] / (fighter1['wins'] + fighter1['losses'] + 0.1) * 0.3
        )
        
        fighter2_score = (
            fighter2['sig_strikes_per_min'] * 0.3 + 
            fighter2['takedown_avg'] * 0.2 + 
            fighter2['win_streak'] * 0.2 + 
            fighter2['wins'] / (fighter2['wins'] + fighter2['losses'] + 0.1) * 0.3
        )
        
        win_prob = fighter1_score / (fighter1_score + fighter2_score)
        winner = fighter1 if np.random.random() < win_prob else fighter2
        
        # Create fight data
        fight_date = pd.Timestamp('2018-01-01') + pd.Timedelta(days=np.random.randint(0, 365*4))
        fight = {
            'fighter1_name': fighter1['name'],
            'fighter2_name': fighter2['name'],
            'fighter1_height': fighter1['height'],
            'fighter2_height': fighter2['height'],
            'fighter1_weight': fighter1['weight'],
            'fighter2_weight': fighter2['weight'],
            'fighter1_reach': fighter1['reach'],
            'fighter2_reach': fighter2['reach'],
            'fighter1_stance': fighter1['stance'],
            'fighter2_stance': fighter2['stance'],
            'fighter1_wins': fighter1['wins'],
            'fighter2_wins': fighter2['wins'],
            'fighter1_losses': fighter1['losses'],
            'fighter2_losses': fighter2['losses'],
            'fighter1_draws': fighter1['draws'],
            'fighter2_draws': fighter2['draws'],
            'fighter1_age': np.random.randint(21, 40),
            'fighter2_age': np.random.randint(21, 40),
            'fighter1_sig_strikes_per_min': fighter1['sig_strikes_per_min'],
            'fighter2_sig_strikes_per_min': fighter2['sig_strikes_per_min'],
            'fighter1_takedown_avg': fighter1['takedown_avg'],
            'fighter2_takedown_avg': fighter2['takedown_avg'],
            'fighter1_sub_avg': fighter1['sub_avg'],
            'fighter2_sub_avg': fighter2['sub_avg'],
            'fighter1_win_streak': fighter1['win_streak'],
            'fighter2_win_streak': fighter2['win_streak'],
            'weight_class': np.random.choice(weight_classes),
            'method': np.random.choice(methods),
            'rounds': np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.1, 0.5, 0.05, 0.15]),
            'time': f"{np.random.randint(0, 5)}:{np.random.randint(0, 60):02d}",
            'date': fight_date.strftime('%Y-%m-%d'),
            'winner_name': winner['name'],
            'fighter1_sig_strikes_landed': np.random.randint(10, 150),
            'fighter1_sig_strikes_attempted': np.random.randint(50, 250),
            'fighter2_sig_strikes_landed': np.random.randint(10, 150),
            'fighter2_sig_strikes_attempted': np.random.randint(50, 250),
            'fighter1_takedowns_landed': np.random.randint(0, 10),
            'fighter1_takedowns_attempted': np.random.randint(0, 15),
            'fighter2_takedowns_landed': np.random.randint(0, 10),
            'fighter2_takedowns_attempted': np.random.randint(0, 15)
        }
        
        fights.append(fight)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(fights)
    df.to_csv(save_path, index=False)
    print(f"Sample UFC dataset created and saved to {save_path}")
    return df

def preprocess_dataset(csv_path=CSV_FILE_PATH):
    """
    Preprocess the dataset to ensure it has all required columns and handle missing values
    """
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Creating sample dataset...")
        create_sample_dataset(csv_path)
        return pd.read_csv(csv_path)
    
    df = pd.read_csv(csv_path)
    
    # Check if the dataset has the basic required columns
    required_columns = [
        'fighter1_name', 'fighter2_name', 'winner_name',
        'fighter1_height', 'fighter2_height',
        'fighter1_weight', 'fighter2_weight',
        'fighter1_reach', 'fighter2_reach'
    ]
    
    # If missing any essential columns, create a new sample dataset
    if not all(col in df.columns for col in required_columns):
        print("Dataset is missing required columns. Creating a new sample dataset...")
        create_sample_dataset(csv_path)
        return pd.read_csv(csv_path)
    
    # Fill missing values with appropriate defaults
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].mean())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])
    
    print("Dataset preprocessing completed.")
    return df

if __name__ == "__main__":
    # Test the data loader
    preprocess_dataset()