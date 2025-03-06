import sqlite3
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("Starting complete model training process...")

# Ensure directories exist
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Step 1: Extract data from database
try:
    # Connect to your database
    conn = sqlite3.connect('data/ufc_fighters.db')
    
    # Get all fighters data
    fighters_df = pd.read_sql_query("SELECT * FROM fighters", conn)
    conn.close()
    
    print(f"Loaded {len(fighters_df)} fighters from database")
except Exception as e:
    print(f"Error loading data from database: {e}")
    exit(1)

# Step 2: Create fight pairs for training
try:
    fights = []
    fighters = fighters_df.to_dict('records')
    
    # Create training pairs - we'll generate synthetic matchups
    for _ in range(10000):  # Create 10,000 matchups
        fighter1 = random.choice(fighters)
        fighter2 = random.choice(fighters)
        
        if fighter1['id'] != fighter2['id']:  # Don't match a fighter against themselves
            # Calculate win probabilities based on stats
            # Convert None values to 0 or appropriate defaults
            f1_wins = fighter1['wins'] if fighter1['wins'] is not None else 0
            f1_losses = fighter1['losses'] if fighter1['losses'] is not None else 0
            f1_td_avg = fighter1['td_avg'] if fighter1['td_avg'] is not None else 0
            f1_slpm = fighter1['SLpM'] if fighter1['SLpM'] is not None else 0
            f1_str_def = fighter1['str_def'] if fighter1['str_def'] is not None else 0.5
            f1_td_def = fighter1['td_def'] if fighter1['td_def'] is not None else 0.5
            
            f2_wins = fighter2['wins'] if fighter2['wins'] is not None else 0
            f2_losses = fighter2['losses'] if fighter2['losses'] is not None else 0
            f2_td_avg = fighter2['td_avg'] if fighter2['td_avg'] is not None else 0
            f2_slpm = fighter2['SLpM'] if fighter2['SLpM'] is not None else 0
            f2_str_def = fighter2['str_def'] if fighter2['str_def'] is not None else 0.5
            f2_td_def = fighter2['td_def'] if fighter2['td_def'] is not None else 0.5
            
            f1_strength = (
                (f1_wins / (f1_wins + f1_losses + 1)) * 0.3 +
                (f1_td_avg / 10) * 0.2 + 
                (f1_slpm / 6) * 0.2 +
                f1_str_def * 0.15 +
                f1_td_def * 0.15
            )
            
            f2_strength = (
                (f2_wins / (f2_wins + f2_losses + 1)) * 0.3 +
                (f2_td_avg / 10) * 0.2 + 
                (f2_slpm / 6) * 0.2 +
                f2_str_def * 0.15 +
                f2_td_def * 0.15
            )
            
            # Add some randomness
            win_prob = 0.5 + (f1_strength - f2_strength) 
            fighter1_won = random.random() < win_prob
            
            # Create fight data
            fight = {}
            
            # For numerical features
            numerical_cols = ['wins', 'losses', 'height', 'weight', 'reach', 'age', 
                             'SLpM', 'sig_str_acc', 'SApM', 'str_def', 
                             'td_avg', 'td_acc', 'td_def', 'sub_avg']
            
            # Handle numerical features 
            for col in numerical_cols:
                # Set None values to 0
                fight[f'fighter1_{col}'] = fighter1[col] if fighter1[col] is not None else 0
                fight[f'fighter2_{col}'] = fighter2[col] if fighter2[col] is not None else 0
            
            # Calculate difference features (only for numerical)
            fight['height_diff'] = (fighter1['height'] or 0) - (fighter2['height'] or 0)
            fight['weight_diff'] = (fighter1['weight'] or 0) - (fighter2['weight'] or 0)
            fight['reach_diff'] = (fighter1['reach'] or 0) - (fighter2['reach'] or 0)
            
            # Safe win rate calculation
            fighter1_total = f1_wins + f1_losses
            fighter2_total = f2_wins + f2_losses
            
            fighter1_win_rate = f1_wins / fighter1_total if fighter1_total > 0 else 0
            fighter2_win_rate = f2_wins / fighter2_total if fighter2_total > 0 else 0
            
            fight['win_rate_diff'] = fighter1_win_rate - fighter2_win_rate
            
            # More diffs for key stats
            fight['td_avg_diff'] = (fighter1['td_avg'] or 0) - (fighter2['td_avg'] or 0)
            fight['SLpM_diff'] = (fighter1['SLpM'] or 0) - (fighter2['SLpM'] or 0)
            fight['sub_avg_diff'] = (fighter1['sub_avg'] or 0) - (fighter2['sub_avg'] or 0)
            
            # Handle stances - convert to binary features
            # Convert None to 'Unknown'
            fighter1_stance = fighter1['stance'] if fighter1['stance'] is not None else 'Unknown'
            fighter2_stance = fighter2['stance'] if fighter2['stance'] is not None else 'Unknown'
            
            # Target variable
            fight['fighter1_won'] = 1 if fighter1_won else 0
            
            # Add fighter names (for reference, not for training)
            fight['fighter1_name'] = fighter1['name']
            fight['fighter2_name'] = fighter2['name']
            
            # Add stances (we'll handle these with one-hot encoding)
            fight['fighter1_stance'] = fighter1_stance
            fight['fighter2_stance'] = fighter2_stance
            
            fights.append(fight)
    
    # Convert to DataFrame
    fights_df = pd.DataFrame(fights)
    
    # Save raw training dataset (before preprocessing)
    raw_training_path = 'data/raw_training_data.csv'
    fights_df.to_csv(raw_training_path, index=False)
    
    print(f"Created raw training dataset with {len(fights_df)} fights")
    print(f"Dataset saved to {raw_training_path}")
except Exception as e:
    print(f"Error creating training data: {e}")
    exit(1)

# Step 3: Preprocess the data and train the model
try:
    # Separate target and names
    y = fights_df['fighter1_won']
    names = fights_df[['fighter1_name', 'fighter2_name']]
    
    # Handle categorical variables
    stances = fights_df[['fighter1_stance', 'fighter2_stance']]
    
    # Get numeric columns
    numeric_cols = [col for col in fights_df.columns 
                   if col not in ['fighter1_won', 'fighter1_name', 'fighter2_name', 
                                 'fighter1_stance', 'fighter2_stance']]
    
    # Create X without categorical or target variables
    X = fights_df[numeric_cols].copy()
    
    # Check for and replace any NaN values with 0
    print(f"X shape before cleaning: {X.shape}")
    print(f"NaN values in X: {X.isna().sum().sum()}")
    
    # Replace NaN with 0
    X = X.fillna(0)
    print(f"NaN values in X after filling: {X.isna().sum().sum()}")
    
    # One-hot encode stances
    stance_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    stance_encoded = stance_encoder.fit_transform(stances)
    
    # Get stance feature names
    stance_feature_names = []
    for i, category in enumerate(stance_encoder.categories_):
        prefix = f"fighter{i+1}_stance"
        stance_feature_names.extend([f"{prefix}_{stance}" for stance in category])
    
    # Convert encoded stances to DataFrame
    stance_df = pd.DataFrame(stance_encoded, columns=stance_feature_names)
    
    # Combine numeric and stance features
    X_combined = pd.concat([X.reset_index(drop=True), stance_df.reset_index(drop=True)], axis=1)
    
    # Final check for NaN values
    if X_combined.isna().sum().sum() > 0:
        print("Warning: Still have NaN values after preprocessing!")
        X_combined = X_combined.fillna(0)
    
    # Save feature columns for prediction
    feature_columns = X_combined.columns.tolist()
    joblib.dump(feature_columns, 'models/feature_columns.joblib')
    print(f"Saved {len(feature_columns)} feature names")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler for prediction
    joblib.dump(scaler, 'models/scaler.joblib')
    print("Saved scaler")
    
    # Save encoder for prediction
    joblib.dump(stance_encoder, 'models/stance_encoder.joblib')
    print("Saved stance encoder")
    
    # Train a random forest model (simple but effective)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate model
    train_accuracy = model.score(X_train_scaled, y_train)
    test_accuracy = model.score(X_test_scaled, y_test)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Save model
    joblib.dump(model, 'models/ensemble_model.joblib')
    print("Saved model to models/ensemble_model.joblib")
    
    # Save model info
    model_info = {
        'is_pytorch': False,
        'features': feature_columns,
        'accuracy': test_accuracy,
        'has_stance_encoding': True
    }
    joblib.dump(model_info, 'models/model_info.joblib')
    print("Saved model info")
    
    print("Training complete!")
except Exception as e:
    print(f"Error during preprocessing and training: {e}")
    import traceback
    traceback.print_exc()