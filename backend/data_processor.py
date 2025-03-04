import pandas as pd
import os
import argparse
from datetime import datetime
import numpy as np

def process_data_for_training(input_file, output_file):
    """
    Process the raw UFC CSV data into a format suitable for training.
    
    Parameters:
    - input_file: Path to the raw CSV database file
    - output_file: Path to save the processed training data
    """
    print(f"Processing {input_file} to create training data...")
    
    # Read the raw data
    df = pd.read_csv(input_file)
    
    # Perform UFC-specific column processing and feature engineering for training
    
    # 1. Handle missing values for UFC data
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('unknown')
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # 2. Feature engineering specific to UFC data
    
    # Calculate height and reach differences
    if all(col in df.columns for col in ['R_Height_cms', 'B_Height_cms']):
        df['height_difference'] = df['R_Height_cms'] - df['B_Height_cms']
    
    if all(col in df.columns for col in ['R_Reach_cms', 'B_Reach_cms']):
        df['reach_difference'] = df['R_Reach_cms'] - df['B_Reach_cms']
    
    if all(col in df.columns for col in ['R_Weight_lbs', 'B_Weight_lbs']):
        df['weight_difference'] = df['R_Weight_lbs'] - df['B_Weight_lbs']
    
    # Calculate striking and grappling efficiency metrics
    if all(col in df.columns for col in ['R_avg_SIG_STR_landed', 'R_avg_SIG_STR_attempted']):
        df['R_striking_efficiency'] = df['R_avg_SIG_STR_landed'] / df['R_avg_SIG_STR_attempted'].replace(0, 1)
    
    if all(col in df.columns for col in ['B_avg_SIG_STR_landed', 'B_avg_SIG_STR_attempted']):
        df['B_striking_efficiency'] = df['B_avg_SIG_STR_landed'] / df['B_avg_SIG_STR_attempted'].replace(0, 1)
    
    if all(col in df.columns for col in ['R_avg_TD_landed', 'R_avg_TD_attempted']):
        df['R_takedown_efficiency'] = df['R_avg_TD_landed'] / df['R_avg_TD_attempted'].replace(0, 1)
    
    if all(col in df.columns for col in ['B_avg_TD_landed', 'B_avg_TD_attempted']):
        df['B_takedown_efficiency'] = df['B_avg_TD_landed'] / df['B_avg_TD_attempted'].replace(0, 1)
    
    # Create win percentage features
    if all(col in df.columns for col in ['R_wins', 'R_losses']):
        df['R_win_percentage'] = df['R_wins'] / (df['R_wins'] + df['R_losses'] + 1e-5)  # Avoid division by zero
    
    if all(col in df.columns for col in ['B_wins', 'B_losses']):
        df['B_win_percentage'] = df['B_wins'] / (df['B_wins'] + df['B_losses'] + 1e-5)
    
    # 3. Normalize numerical columns for ML
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'Winner' and not col.startswith('date')]
    
    for col in numerical_cols:
        if df[col].max() > df[col].min():
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    
    # 4. Encode categorical variables if needed
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col not in ['R_fighter', 'B_fighter', 'Winner', 'date', 'location']]
    
    for col in categorical_cols:
        df_encoded = pd.get_dummies(df[col], prefix=col, drop_first=True)
        df = pd.concat([df, df_encoded], axis=1)
    
    # 5. Convert Winner to binary target (1 if Red wins, 0 if Blue wins)
    if 'Winner' in df.columns:
        df['fighter1_won'] = np.where(df['Winner'] == 'Red', 1, 0)
    
    # 6. Drop unnecessary columns for training
    columns_to_drop = ['location', 'date', 'time']  # Common columns not needed for training
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    # Save the processed data
    df.to_csv(output_file, index=False)
    print(f"UFC training data saved to {output_file}")
    print(f"Original columns: {len(pd.read_csv(input_file).columns)}")
    print(f"Processed columns: {len(df.columns)}")
    print(f"Number of fights in training data: {len(df)}")

def main():
    parser = argparse.ArgumentParser(description="Process UFC CSV data for training")
    parser.add_argument("--input", default="data/ufc_data.csv", help="Path to raw UFC CSV database")
    parser.add_argument("--output", default="data/ufc_training_data.csv", help="Path to save UFC training data")
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Process the data
    process_data_for_training(args.input, args.output)

if __name__ == "__main__":
    main()