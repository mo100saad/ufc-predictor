import os
import pandas as pd
import numpy as np
import logging
from config import CSV_FILE_PATH, TRAINING_CSV_PATH, DATA_DIR

logger = logging.getLogger('ufc_model')

def sync_csv_files():
    """
    Synchronize the two CSV files:
    - fighter_stats.csv (reference database of fighter statistics)
    - ufc_dataset.csv (actual fight data for training)
    
    This function makes sure the training data is properly prepared.
    """
    logger.info("Synchronizing CSV files")
    
    # Check if needed directories exist
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Define paths
    fighter_stats_path = CSV_FILE_PATH  # Reference database
    training_data_path = TRAINING_CSV_PATH  # Training dataset
    
    # Check if main UFC dataset exists for training
    ufc_dataset_path = os.path.join(DATA_DIR, 'ufc_dataset.csv')
    if not os.path.exists(ufc_dataset_path) and not os.path.exists(training_data_path):
        logger.warning(f"UFC dataset not found at {ufc_dataset_path}. Using sample data.")
        create_sample_dataset(ufc_dataset_path)
    
    # If fighter stats don't exist, create minimal version
    if not os.path.exists(fighter_stats_path):
        logger.warning(f"Fighter stats not found at {fighter_stats_path}. Creating minimal version.")
        # Extract fighter stats from the UFC dataset if possible
        if os.path.exists(ufc_dataset_path):
            extract_fighter_stats(ufc_dataset_path, fighter_stats_path)
        else:
            create_minimal_fighter_stats(fighter_stats_path)
    
    # If training data doesn't exist, process the UFC dataset
    if not os.path.exists(training_data_path) and os.path.exists(ufc_dataset_path):
        logger.info(f"Processing UFC dataset for training")
        process_data_for_training(ufc_dataset_path, training_data_path)
    
    # Verify files exist
    files_exist = os.path.exists(fighter_stats_path) and os.path.exists(training_data_path)
    
    if files_exist:
        logger.info("CSV files successfully synchronized:")
        logger.info(f"Fighter stats: {fighter_stats_path}")
        logger.info(f"Training data: {training_data_path}")
        print(f"CSV files successfully synchronized:")
        print(f"Fighter stats: {fighter_stats_path}")
        print(f"Training data: {training_data_path}")
    else:
        logger.error("Failed to synchronize CSV files")
        print("Failed to synchronize CSV files")
    
    return files_exist

def verify_csv_consistency():
    """
    Verify that both CSV files exist and are suitable for training.
    Returns True if files are consistent, False otherwise.
    """
    # Define paths
    fighter_stats_path = CSV_FILE_PATH  # Reference database
    training_data_path = TRAINING_CSV_PATH  # Training dataset
    
    # Check if both files exist
    fighter_stats_exists = os.path.exists(fighter_stats_path)
    training_data_exists = os.path.exists(training_data_path)
    
    if not fighter_stats_exists:
        print(f"Fighter stats file not found at {fighter_stats_path}")
        return False
    
    if not training_data_exists:
        print(f"Training data file not found at {training_data_path}")
        return False
    
    # Check if training data has required columns
    try:
        # Load the training data
        training_df = pd.read_csv(training_data_path)
        
        # Check for target column (winner or fighter1_won)
        has_target = 'winner' in training_df.columns or 'Winner' in training_df.columns or 'fighter1_won' in training_df.columns
        
        if not has_target:
            print("Training data is missing required target column (fighter1_won or Winner)")
            return False
            
        # Check for physical attribute differences
        has_diffs = any(col.endswith('_diff') for col in training_df.columns)
        if not has_diffs:
            print("Warning: Training data doesn't contain difference columns")
        
        # Load fighter stats to verify it's a valid reference
        fighter_df = pd.read_csv(fighter_stats_path)
        
        # Basic fighter stats check
        if len(fighter_df) < 10:  # Arbitrary minimum number
            print("Warning: Fighter stats database seems too small")
        
        return True
    except Exception as e:
        print(f"Error checking CSV consistency: {str(e)}")
        return False

def extract_fighter_stats(ufc_dataset_path, fighter_stats_path):
    """
    Extract fighter statistics from UFC dataset to create a fighter stats database
    """
    try:
        # Load the UFC dataset
        ufc_df = pd.read_csv(ufc_dataset_path)
        
        # Get unique fighters and their most recent stats
        fighters = set()
        
        # Extract from red corner
        if 'r_fighter' in ufc_df.columns or 'R_fighter' in ufc_df.columns:
            red_col = 'r_fighter' if 'r_fighter' in ufc_df.columns else 'R_fighter'
            fighters.update(ufc_df[red_col].unique())
        
        # Extract from blue corner
        if 'b_fighter' in ufc_df.columns or 'B_fighter' in ufc_df.columns:
            blue_col = 'b_fighter' if 'b_fighter' in ufc_df.columns else 'B_fighter'
            fighters.update(ufc_df[blue_col].unique())
        
        # Create a DataFrame with fighters
        fighter_stats = []
        
        # Find the most recent stats for each fighter
        for fighter in fighters:
            # Look for the most recent fight as red fighter
            red_col = 'r_fighter' if 'r_fighter' in ufc_df.columns else 'R_fighter'
            blue_col = 'b_fighter' if 'b_fighter' in ufc_df.columns else 'B_fighter'
            
            # Try to get the most recent fight (assuming data is chronologically ordered)
            red_fights = ufc_df[ufc_df[red_col] == fighter]
            blue_fights = ufc_df[ufc_df[blue_col] == fighter]
            
            if len(red_fights) > 0:
                latest_fight = red_fights.iloc[-1]
                prefix = 'r_' if 'r_fighter' in ufc_df.columns else 'R_'
                
                # Get stats using the appropriate prefix
                stats = {
                    'name': fighter,
                    'height': latest_fight.get(f'{prefix}height', latest_fight.get(f'{prefix}Height_cms')),
                    'weight': latest_fight.get(f'{prefix}weight', latest_fight.get(f'{prefix}Weight_lbs')),
                    'reach': latest_fight.get(f'{prefix}reach', latest_fight.get(f'{prefix}Reach_cms')),
                    'stance': latest_fight.get(f'{prefix}stance', latest_fight.get(f'{prefix}Stance')),
                    'SLpM': latest_fight.get(f'{prefix}SLpM_total', latest_fight.get(f'{prefix}avg_SIG_STR_landed')),
                    'SApM': latest_fight.get(f'{prefix}SApM_total', 0),
                    'td_avg': latest_fight.get(f'{prefix}td_avg', 0),
                    'sub_avg': latest_fight.get(f'{prefix}sub_avg', 0)
                }
                fighter_stats.append(stats)
            elif len(blue_fights) > 0:
                latest_fight = blue_fights.iloc[-1]
                prefix = 'b_' if 'b_fighter' in ufc_df.columns else 'B_'
                
                # Get stats using the appropriate prefix
                stats = {
                    'name': fighter,
                    'height': latest_fight.get(f'{prefix}height', latest_fight.get(f'{prefix}Height_cms')),
                    'weight': latest_fight.get(f'{prefix}weight', latest_fight.get(f'{prefix}Weight_lbs')),
                    'reach': latest_fight.get(f'{prefix}reach', latest_fight.get(f'{prefix}Reach_cms')),
                    'stance': latest_fight.get(f'{prefix}stance', latest_fight.get(f'{prefix}Stance')),
                    'SLpM': latest_fight.get(f'{prefix}SLpM_total', latest_fight.get(f'{prefix}avg_SIG_STR_landed')),
                    'SApM': latest_fight.get(f'{prefix}SApM_total', 0),
                    'td_avg': latest_fight.get(f'{prefix}td_avg', 0),
                    'sub_avg': latest_fight.get(f'{prefix}sub_avg', 0)
                }
                fighter_stats.append(stats)
        
        # Create DataFrame and save
        fighter_df = pd.DataFrame(fighter_stats)
        fighter_df.to_csv(fighter_stats_path, index=False)
        logger.info(f"Extracted stats for {len(fighter_df)} fighters to {fighter_stats_path}")
    
    except Exception as e:
        logger.error(f"Error extracting fighter stats: {e}")
        create_minimal_fighter_stats(fighter_stats_path)

def create_minimal_fighter_stats(fighter_stats_path):
    """Create a minimal fighter stats database with dummy data"""
    # Create some dummy fighter data
    fighters = [
        {"name": "Conor McGregor", "height": 175, "weight": 155, "reach": 188, "stance": "Southpaw", 
         "SLpM": 5.32, "SApM": 4.09, "td_avg": 0.7, "sub_avg": 0.0},
        {"name": "Khabib Nurmagomedov", "height": 178, "weight": 155, "reach": 178, "stance": "Orthodox", 
         "SLpM": 4.10, "SApM": 2.23, "td_avg": 5.35, "sub_avg": 0.6},
        {"name": "Jon Jones", "height": 193, "weight": 205, "reach": 215, "stance": "Orthodox", 
         "SLpM": 4.31, "SApM": 2.04, "td_avg": 1.85, "sub_avg": 0.5},
        {"name": "Amanda Nunes", "height": 173, "weight": 135, "reach": 178, "stance": "Orthodox", 
         "SLpM": 4.57, "SApM": 2.65, "td_avg": 2.15, "sub_avg": 0.6},
        {"name": "Israel Adesanya", "height": 193, "weight": 185, "reach": 203, "stance": "Switch", 
         "SLpM": 3.98, "SApM": 2.53, "td_avg": 0.0, "sub_avg": 0.0}
    ]
    
    # Create DataFrame and save
    fighter_df = pd.DataFrame(fighters)
    fighter_df.to_csv(fighter_stats_path, index=False)
    logger.info(f"Created minimal fighter stats dataset with {len(fighter_df)} fighters")

def process_data_for_training(input_csv_path, output_csv_path):
    """
    Process the UFC dataset into a format suitable for training
    
    Args:
        input_csv_path (str): Path to the UFC dataset
        output_csv_path (str): Path to save the processed training data
    """
    logger.info(f"Processing data from {input_csv_path} for training")
    
    try:
        # Load the UFC dataset
        df = pd.read_csv(input_csv_path)
        
        # Make a copy for processing
        processed_df = df.copy()
        
        # Check if we have the outcome column
        if 'winner' in processed_df.columns:
            # Standardize to capitalized version
            processed_df['Winner'] = processed_df['winner']
        elif 'Winner' not in processed_df.columns:
            logger.error("No winner column found in dataset")
            raise ValueError("The dataset must contain a 'winner' or 'Winner' column")
        
        # Create fighter1_won column (1 if Red won, 0 if Blue won)
        if 'fighter1_won' not in processed_df.columns:
            processed_df['fighter1_won'] = (processed_df['Winner'] == 'Red').astype(int)
        
        # Handle missing values in numeric columns
        for col in processed_df.columns:
            if processed_df[col].dtype in ['float64', 'int64'] and processed_df[col].isna().any():
                # For percentage columns, fill with median
                if 'pct' in col or 'acc' in col or 'def' in col:
                    processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                # For count columns, fill with 0
                else:
                    processed_df[col] = processed_df[col].fillna(0)
        
        # Handle categorical columns
        for col in processed_df.columns:
            if processed_df[col].dtype == 'object' and col not in ['Winner', 'r_fighter', 'b_fighter', 'R_fighter', 'B_fighter']:
                processed_df[col] = processed_df[col].fillna('Unknown')
        
        # Ensure the physical diff columns exist (if they don't already)
        # These are critical features for reducing corner bias
        if 'height_diff' not in processed_df.columns:
            # Check for both naming conventions
            if 'r_height' in processed_df.columns and 'b_height' in processed_df.columns:
                processed_df['height_diff'] = processed_df['r_height'] - processed_df['b_height']
            elif 'R_Height_cms' in processed_df.columns and 'B_Height_cms' in processed_df.columns:
                processed_df['height_diff'] = processed_df['R_Height_cms'] - processed_df['B_Height_cms']
        
        if 'reach_diff' not in processed_df.columns:
            if 'r_reach' in processed_df.columns and 'b_reach' in processed_df.columns:
                processed_df['reach_diff'] = processed_df['r_reach'] - processed_df['b_reach']
            elif 'R_Reach_cms' in processed_df.columns and 'B_Reach_cms' in processed_df.columns:
                processed_df['reach_diff'] = processed_df['R_Reach_cms'] - processed_df['B_Reach_cms']
        
        if 'weight_diff' not in processed_df.columns:
            if 'r_weight' in processed_df.columns and 'b_weight' in processed_df.columns:
                processed_df['weight_diff'] = processed_df['r_weight'] - processed_df['b_weight']
            elif 'R_Weight_lbs' in processed_df.columns and 'B_Weight_lbs' in processed_df.columns:
                processed_df['weight_diff'] = processed_df['R_Weight_lbs'] - processed_df['B_Weight_lbs']
        
        # Check that we have the key features
        minimum_cols = ['fighter1_won']
        if not all(col in processed_df.columns for col in minimum_cols):
            missing = [col for col in minimum_cols if col not in processed_df.columns]
            logger.error(f"Missing required columns after processing: {missing}")
            raise ValueError(f"Missing required columns after processing: {missing}")
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        
        # Save the processed data
        processed_df.to_csv(output_csv_path, index=False)
        
        logger.info(f"Processed data saved to {output_csv_path}: {processed_df.shape}")
        return processed_df
        
    except Exception as e:
        logger.error(f"Error processing data for training: {e}")
        # Create a minimal training dataset for testing
        create_minimal_training_dataset(output_csv_path)
        return pd.read_csv(output_csv_path)

def create_minimal_training_dataset(output_path):
    """Create a minimal training dataset for testing"""
    # Create simple fight data with key columns
    data = []
    for i in range(100):
        # Create fights with varying attributes
        height_diff = np.random.uniform(-15, 15)
        reach_diff = np.random.uniform(-20, 20)
        weight_diff = np.random.uniform(-10, 10)
        
        # Probability of winning increases with positive differences
        prob_red_wins = 0.5 + (0.02 * height_diff) + (0.01 * reach_diff) + (0.03 * weight_diff)
        # Clip probability to 0.05-0.95 range
        prob_red_wins = max(0.05, min(0.95, prob_red_wins))
        
        # Determine winner
        red_wins = np.random.random() < prob_red_wins
        
        # Create a fight record
        fight = {
            'r_fighter': f"Fighter_R_{i}",
            'b_fighter': f"Fighter_B_{i}",
            'Winner': 'Red' if red_wins else 'Blue',
            'fighter1_won': 1 if red_wins else 0,
            'height_diff': height_diff,
            'reach_diff': reach_diff,
            'weight_diff': weight_diff,
            'sig_str_diff': np.random.uniform(-50, 50),
            'sig_str_acc_diff': np.random.uniform(-0.3, 0.3),
            'td_diff': np.random.uniform(-5, 5),
            'sub_att_diff': np.random.uniform(-3, 3)
        }
        data.append(fight)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created minimal training dataset with {len(df)} fights at {output_path}")
    return df

def create_sample_dataset(save_path):
    """Create a sample UFC dataset for testing"""
    # This creates a sample dataset with actual UFC-like structure
    # for running and testing the model
    
    # Generate fighter names
    first_names = ['Alex', 'Jon', 'Conor', 'Khabib', 'Francis', 'Amanda', 'Cris', 'Valentina', 'Israel', 'Max']
    last_names = ['Smith', 'Jones', 'McGregor', 'Nurmagomedov', 'Ngannou', 'Nunes', 'Cyborg', 'Shevchenko', 'Adesanya', 'Holloway']
    
    fighter_names = [f"{fn} {ln}" for fn in first_names for ln in last_names][:50]  # 50 unique fighters
    
    # Generate fighters with realistic attributes
    fighters = []
    for name in fighter_names:
        height = np.random.uniform(160, 195)  # cm
        weight = np.random.uniform(125, 265)  # lbs
        reach = height * 1.02 + np.random.uniform(-5, 15)  # Realistic reach
        
        fighters.append({
            'name': name,
            'height': height,
            'weight': weight,
            'reach': reach,
            'stance': np.random.choice(['Orthodox', 'Southpaw', 'Switch'], p=[0.7, 0.25, 0.05]),
            'SLpM': np.random.uniform(2.0, 6.0),
            'SApM': np.random.uniform(2.0, 5.0),
            'str_acc': np.random.uniform(0.3, 0.7),
            'str_def': np.random.uniform(0.4, 0.7),
            'td_avg': np.random.uniform(0, 5.0),
            'td_acc': np.random.uniform(0.2, 0.8),
            'td_def': np.random.uniform(0.3, 0.9),
            'sub_avg': np.random.uniform(0, 2.0)
        })
    
    # Generate fights
    fights = []
    for i in range(200):
        # Select two different fighters
        idx1, idx2 = np.random.choice(len(fighters), 2, replace=False)
        fighter1 = fighters[idx1]
        fighter2 = fighters[idx2]
        
        # Calculate advantage factors
        height_diff = fighter1['height'] - fighter2['height']
        reach_diff = fighter1['reach'] - fighter2['reach']
        weight_diff = fighter1['weight'] - fighter2['weight']
        
        # Create a probability model based on physical and skill attributes
        # Higher value = more likely fighter1 (red) wins
        win_factor = (
            0.5 +  # Base probability
            0.02 * height_diff +  # Height advantage
            0.01 * reach_diff +   # Reach advantage
            0.03 * (fighter1['SLpM'] - fighter2['SLpM']) +  # Striking advantage
            0.02 * (fighter2['SApM'] - fighter1['SApM']) +  # Defense advantage
            0.02 * (fighter1['td_avg'] - fighter2['td_avg']) +  # Takedown advantage
            0.01 * (fighter1['sub_avg'] - fighter2['sub_avg'])   # Submission advantage
        )
        
        # Clip to reasonable range and add some randomness
        win_prob = max(0.05, min(0.95, win_factor)) + np.random.uniform(-0.2, 0.2)
        win_prob = max(0.05, min(0.95, win_prob))  # Re-clip after adding randomness
        
        # Determine winner
        red_wins = np.random.random() < win_prob
        
        # Create fight record
        fight = {
            'r_fighter': fighter1['name'],
            'b_fighter': fighter2['name'],
            'winner': 'Red' if red_wins else 'Blue',
            'r_height': fighter1['height'],
            'b_height': fighter2['height'],
            'height_diff': height_diff,
            'r_reach': fighter1['reach'],
            'b_reach': fighter2['reach'],
            'reach_diff': reach_diff,
            'r_weight': fighter1['weight'],
            'b_weight': fighter2['weight'],
            'weight_diff': weight_diff,
            'r_SLpM': fighter1['SLpM'],
            'b_SLpM': fighter2['SLpM'],
            'SLpM_diff': fighter1['SLpM'] - fighter2['SLpM'],
            'r_SApM': fighter1['SApM'],
            'b_SApM': fighter2['SApM'],
            'SApM_diff': fighter1['SApM'] - fighter2['SApM'],
            'r_str_acc': fighter1['str_acc'],
            'b_str_acc': fighter2['str_acc'],
            'str_acc_diff': fighter1['str_acc'] - fighter2['str_acc'],
            'r_td_avg': fighter1['td_avg'],
            'b_td_avg': fighter2['td_avg'],
            'td_avg_diff': fighter1['td_avg'] - fighter2['td_avg'],
            'r_sub_avg': fighter1['sub_avg'],
            'b_sub_avg': fighter2['sub_avg'],
            'sub_avg_diff': fighter1['sub_avg'] - fighter2['sub_avg']
        }
        fights.append(fight)
    
    # Create DataFrame and save
    df = pd.DataFrame(fights)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    
    logger.info(f"Created sample UFC dataset with {len(fights)} fights at {save_path}")
    return df