import os
import pandas as pd
import numpy as np
from config import CSV_FILE_PATH, TRAINING_CSV_PATH
from data_loader import preprocess_dataset, create_sample_dataset
import argparse

def sync_csv_files():
    """
    Synchronize the two CSV files - generate the training CSV from the main data CSV.
    Use this when the main CSV has been updated and you need to regenerate the training data.
    """
    from data_processor import process_data_for_training
    
    # Check if main CSV exists
    if not os.path.exists(CSV_FILE_PATH):
        print(f"Main CSV file not found at {CSV_FILE_PATH}. Creating sample dataset...")
        create_sample_dataset(CSV_FILE_PATH)
    
    # Make sure data directory exists
    os.makedirs(os.path.dirname(TRAINING_CSV_PATH), exist_ok=True)
    
    # Process the main CSV data into the training CSV
    process_data_for_training(CSV_FILE_PATH, TRAINING_CSV_PATH)
    
    print(f"CSV files successfully synchronized:")
    print(f"Main data: {CSV_FILE_PATH}")
    print(f"Training data: {TRAINING_CSV_PATH}")

def verify_csv_consistency():
    """
    Verify that both CSV files exist and are consistent with each other.
    Returns True if files are consistent, False otherwise.
    """
    # Check if both files exist
    main_exists = os.path.exists(CSV_FILE_PATH)
    training_exists = os.path.exists(TRAINING_CSV_PATH)
    
    if not main_exists and not training_exists:
        print("Neither main CSV nor training CSV exist.")
        return False
    
    if not main_exists:
        print(f"Main CSV file not found at {CSV_FILE_PATH}")
        return False
    
    if not training_exists:
        print(f"Training CSV file not found at {TRAINING_CSV_PATH}")
        return False
    
    # Load both files to check consistency
    try:
        main_df = pd.read_csv(CSV_FILE_PATH)
        training_df = pd.read_csv(TRAINING_CSV_PATH)
        
        # Basic consistency checks
        main_rows = len(main_df)
        training_rows = len(training_df)
        
        # Get fighter names from both datasets (if available)
        main_fighters = set()
        if 'R_fighter' in main_df.columns and 'B_fighter' in main_df.columns:
            main_fighters.update(main_df['R_fighter'].unique())
            main_fighters.update(main_df['B_fighter'].unique())
        
        training_fighters = set()
        if 'R_fighter' in training_df.columns and 'B_fighter' in training_df.columns:
            training_fighters.update(training_df['R_fighter'].unique())
            training_fighters.update(training_df['B_fighter'].unique())
        
        # Check if training data is a subset of main data
        if main_fighters and training_fighters:
            fighters_overlap = len(training_fighters.intersection(main_fighters))
            fighters_pct = (fighters_overlap / len(training_fighters)) * 100 if training_fighters else 0
            
            print(f"Main CSV: {main_rows} rows")
            print(f"Training CSV: {training_rows} rows")
            print(f"Fighter overlap: {fighters_pct:.1f}% of training fighters exist in main data")
            
            # If less than 80% of fighters match, files may be inconsistent
            if fighters_pct < 80 and len(training_fighters) > 10:
                print("WARNING: Training data contains many fighters not in the main dataset!")
                return False
        
        # Check if the training data has the required target column
        if 'fighter1_won' not in training_df.columns and 'Winner' not in training_df.columns:
            print("WARNING: Training data is missing required target column (fighter1_won or Winner)")
            return False
            
        return True
    
    except Exception as e:
        print(f"Error checking CSV consistency: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="UFC CSV Synchronization Utility")
    parser.add_argument("--sync", action="store_true", help="Synchronize main and training CSV files")
    parser.add_argument("--verify", action="store_true", help="Verify CSV consistency")
    
    args = parser.parse_args()
    
    if args.verify:
        is_consistent = verify_csv_consistency()
        print(f"CSV consistency check {'PASSED' if is_consistent else 'FAILED'}")
    
    if args.sync:
        sync_csv_files()
    
    # If no args provided, do both
    if not args.verify and not args.sync:
        is_consistent = verify_csv_consistency()
        print(f"CSV consistency check {'PASSED' if is_consistent else 'FAILED'}")
        
        if not is_consistent:
            user_input = input("Would you like to synchronize the CSV files? (y/n): ")
            if user_input.lower() == 'y':
                sync_csv_files()

if __name__ == "__main__":
    main()