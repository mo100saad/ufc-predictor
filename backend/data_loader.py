import pandas as pd
import os
import requests
import numpy as np
from config import CSV_FILE_PATH
import kaggle

def load_dataset(save_path=CSV_FILE_PATH):
    """
    Load the dataset from a local file, GitHub, or Kaggle.
    If none are available, create a sample dataset.
    """
    # Try to load the dataset locally
    if os.path.exists(save_path):
        print(f"Loading dataset from {save_path}...")
        return pd.read_csv(save_path)
    
    # If local file not found, try downloading from GitHub
    print(f"Dataset not found locally. Attempting to download from GitHub...")
    github_url = "https://raw.githubusercontent.com/mo100saad/ufc-predictor/main/backend/data/fighter_stats.csv"
    try:
        response = requests.get(github_url)
        response.raise_for_status()  # Raise an error for bad status codes
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Dataset downloaded from GitHub and saved to {save_path}.")
        return pd.read_csv(save_path)
    except Exception as e:
        print(f"Failed to download from GitHub: {str(e)}")
    
    # If GitHub fails, try downloading from Kaggle
    print("Attempting to download from Kaggle...")
    try:
        kaggle.api.authenticate()  # Ensure Kaggle API is authenticated
        kaggle.api.dataset_download_files(
            'aaronfriasr/ufc-fighters-statistics-2024',
            path=os.path.dirname(save_path),
            unzip=True
        )
        print(f"Dataset downloaded from Kaggle and saved to {save_path}.")
        return pd.read_csv(save_path)
    except Exception as e:
        print(f"Failed to download from Kaggle: {str(e)}")
    
    # If all else fails, create a sample dataset
    print("All download attempts failed. Creating a sample dataset...")
    return create_sample_dataset(save_path)

def create_sample_dataset(save_path=CSV_FILE_PATH):
    """
    Create a sample UFC dataset if download fails - using the actual UFC dataset structure
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Generate fighter names
    first_names = ['Alex', 'Jon', 'Conor', 'Khabib', 'Francis', 'Amanda', 'Cris', 'Valentina', 'Israel', 'Max']
    last_names = ['Smith', 'Jones', 'McGregor', 'Nurmagomedov', 'Ngannou', 'Nunes', 'Cyborg', 'Shevchenko', 'Adesanya', 'Holloway']
    
    fighter_names = [f"{fn} {ln}" for fn in first_names for ln in last_names][:50]  # Generate 50 unique fighters
    
    # Generate fighter stats - using column names from the actual dataset
    fighters = []
    for name in fighter_names:
        fighters.append({
            'name': name,
            'height_cms': np.random.uniform(160, 195),  # cm
            'weight_lbs': np.random.uniform(125, 265),   # lbs (UFC weight classes)
            'reach_cms': np.random.uniform(160, 210),   # cm
            'stance': np.random.choice(['Orthodox', 'Southpaw', 'Switch']),
            'age': np.random.randint(21, 40),
            'avg_SIG_STR_landed': np.random.uniform(1.5, 7.0),
            'avg_SIG_STR_pct': np.random.uniform(30, 70),
            'avg_TD_landed': np.random.uniform(0, 5.0),
            'avg_TD_pct': np.random.uniform(0, 100),
            'avg_SUB_ATT': np.random.uniform(0, 2.0),
            'avg_KD': np.random.uniform(0, 2.0),
            'avg_REV': np.random.uniform(0, 1.0),
            'win_by_KO/TKO': np.random.randint(0, 10),
            'win_by_Submission': np.random.randint(0, 8),
            'win_by_Decision_Unanimous': np.random.randint(0, 10),
            'win_by_Decision_Split': np.random.randint(0, 5)
        })
    
    # Generate fights
    fights = []
    weight_classes = ['Flyweight', 'Bantamweight', 'Featherweight', 'Lightweight', 'Welterweight', 
                      'Middleweight', 'Light Heavyweight', 'Heavyweight']
    methods = ['KO/TKO', 'Submission', 'Decision - Unanimous', 'Decision - Split', 'Decision - Majority']
    
    for i in range(200):  # Generate 200 fights
        # Select two different fighters
        idx1, idx2 = np.random.choice(len(fighters), 2, replace=False)
        fighter_red = fighters[idx1]
        fighter_blue = fighters[idx2]
        
        # Determine winner based on random probability influenced by stats
        red_score = (
            fighter_red['avg_SIG_STR_landed'] * 0.3 + 
            fighter_red['avg_TD_landed'] * 0.2 + 
            fighter_red['avg_KD'] * 0.3 +
            fighter_red['avg_SUB_ATT'] * 0.2
        )
        
        blue_score = (
            fighter_blue['avg_SIG_STR_landed'] * 0.3 + 
            fighter_blue['avg_TD_landed'] * 0.2 + 
            fighter_blue['avg_KD'] * 0.3 +
            fighter_blue['avg_SUB_ATT'] * 0.2
        )
        
        win_prob = red_score / (red_score + blue_score)
        winner = "Red" if np.random.random() < win_prob else "Blue"
        
        # Create fight data using the actual column structure from the UFC dataset
        fight_date = pd.Timestamp('2018-01-01') + pd.Timedelta(days=np.random.randint(0, 365*4))
        
        # Basic fight info
        fight = {
            'R_fighter': fighter_red['name'],
            'B_fighter': fighter_blue['name'],
            'date': fight_date.strftime('%Y-%m-%d'),
            'location': f"UFC {np.random.randint(200, 300)}, Las Vegas, Nevada, USA",
            'weight_class': np.random.choice(weight_classes),
            'title_bout': np.random.choice([True, False], p=[0.1, 0.9]),
            'Winner': winner,
            'method': np.random.choice(methods),
            'rounds': np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.1, 0.5, 0.05, 0.15]),
            'time': f"{np.random.randint(0, 5)}:{np.random.randint(0, 60):02d}"
        }
        
        # Fighter physical attributes
        fight.update({
            'R_Height_cms': fighter_red['height_cms'],
            'B_Height_cms': fighter_blue['height_cms'],
            'R_Reach_cms': fighter_red['reach_cms'],
            'B_Reach_cms': fighter_blue['reach_cms'],
            'R_Weight_lbs': fighter_red['weight_lbs'],
            'B_Weight_lbs': fighter_blue['weight_lbs'],
            'R_Stance': fighter_red['stance'],
            'B_Stance': fighter_blue['stance'],
            'R_age': fighter_red['age'],
            'B_age': fighter_blue['age']
        })
        
        # Fighter stats
        fight.update({
            'R_avg_SIG_STR_pct': fighter_red['avg_SIG_STR_pct'],
            'B_avg_SIG_STR_pct': fighter_blue['avg_SIG_STR_pct'],
            'R_avg_SIG_STR_landed': fighter_red['avg_SIG_STR_landed'],
            'B_avg_SIG_STR_landed': fighter_blue['avg_SIG_STR_landed'],
            'R_avg_SIG_STR_attempted': fighter_red['avg_SIG_STR_landed'] * (100 / fighter_red['avg_SIG_STR_pct']),
            'B_avg_SIG_STR_attempted': fighter_blue['avg_SIG_STR_landed'] * (100 / fighter_blue['avg_SIG_STR_pct']),
            'R_avg_TD_pct': fighter_red['avg_TD_pct'],
            'B_avg_TD_pct': fighter_blue['avg_TD_pct'],
            'R_avg_TD_landed': fighter_red['avg_TD_landed'],
            'B_avg_TD_landed': fighter_blue['avg_TD_landed'],
            'R_avg_SUB_ATT': fighter_red['avg_SUB_ATT'],
            'B_avg_SUB_ATT': fighter_blue['avg_SUB_ATT'],
            'R_avg_KD': fighter_red['avg_KD'],
            'B_avg_KD': fighter_blue['avg_KD'],
            'R_avg_REV': fighter_red['avg_REV'],
            'B_avg_REV': fighter_blue['avg_REV']
        })
        
        # Fight outcome stats
        fight.update({
            'R_win_by_KO/TKO': fighter_red['win_by_KO/TKO'],
            'B_win_by_KO/TKO': fighter_blue['win_by_KO/TKO'],
            'R_win_by_Submission': fighter_red['win_by_Submission'],
            'B_win_by_Submission': fighter_blue['win_by_Submission'],
            'R_win_by_Decision_Unanimous': fighter_red['win_by_Decision_Unanimous'],
            'B_win_by_Decision_Unanimous': fighter_blue['win_by_Decision_Unanimous'],
            'R_win_by_Decision_Split': fighter_red['win_by_Decision_Split'],
            'B_win_by_Decision_Split': fighter_blue['win_by_Decision_Split']
        })
        
        fights.append(fight)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(fights)
    df.to_csv(save_path, index=False)
    print(f"Sample UFC dataset created and saved to {save_path} with {len(fights)} fights")
    return df

def preprocess_dataset(csv_path=CSV_FILE_PATH):
    """
    Preprocess the dataset to ensure it has all required columns and handle missing values
    """
    if not os.path.exists(csv_path):
        print(f"Dataset not found at {csv_path}. Creating sample dataset...")
        create_sample_dataset(csv_path)
        return pd.read_csv(csv_path)
    
    df = load_dataset(csv_path)
    
    # Check if the dataset has the basic required columns based on actual UFC data structure
    required_columns = [
        'R_fighter', 'B_fighter', 'Winner',
        'R_Height_cms', 'B_Height_cms',
        'R_Weight_lbs', 'B_Weight_lbs',
        'R_Reach_cms', 'B_Reach_cms',
        'date', 'weight_class', 'method'
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
    
    # Map the dataset to our expected structure for database import
    df_mapped = df.copy()
    
    # Rename columns to match our database structure if needed
    column_mapping = {
        'R_fighter': 'fighter1_name',
        'B_fighter': 'fighter2_name',
        'R_Height_cms': 'fighter1_height',
        'B_Height_cms': 'fighter2_height',
        'R_Weight_lbs': 'fighter1_weight',
        'B_Weight_lbs': 'fighter2_weight',
        'R_Reach_cms': 'fighter1_reach',
        'B_Reach_cms': 'fighter2_reach',
        'R_Stance': 'fighter1_stance',
        'B_Stance': 'fighter2_stance',
        'R_age': 'fighter1_age',
        'B_age': 'fighter2_age',
        'R_avg_SIG_STR_landed': 'fighter1_sig_strikes_per_min',
        'B_avg_SIG_STR_landed': 'fighter2_sig_strikes_per_min',
        'R_avg_TD_landed': 'fighter1_takedown_avg',
        'B_avg_TD_landed': 'fighter2_takedown_avg',
        'R_avg_SUB_ATT': 'fighter1_sub_avg',
        'B_avg_SUB_ATT': 'fighter2_sub_avg'
    }
    
    # Apply column mapping
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns:
            df_mapped[new_col] = df[old_col]
    
    # Add winner_name based on Winner column (Red or Blue)
    df_mapped['winner_name'] = df.apply(
        lambda row: row['R_fighter'] if row['Winner'] == 'Red' else row['B_fighter'] if row['Winner'] == 'Blue' else None, 
        axis=1
    )
    
    # Calculate win streak (this would normally be more complex)
    # This is a simplification - in a real app you'd compute this from fight history
    df_mapped['fighter1_win_streak'] = np.random.randint(0, 5, size=len(df))
    df_mapped['fighter2_win_streak'] = np.random.randint(0, 5, size=len(df))
    
    # Save preprocessed data
    df_mapped.to_csv(csv_path, index=False)
    print("Dataset preprocessing completed and saved.")
    return df_mapped

if __name__ == "__main__":
    # Test the data loader
    preprocess_dataset()