import pandas as pd
import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Part 1: Load and prepare the data files
def load_fight_data():
    """
    Load the main data.csv file as your database reference
    and preprocessed_data.csv for machine learning
    """
    # Load main dataset
    data_df = pd.read_csv('data.csv')
    
    # Load preprocessed dataset
    preprocessed_df = pd.read_csv('preprocessed_data.csv')
    
    print(f"Main dataset: {data_df.shape[0]} rows, {data_df.shape[1]} columns")
    print(f"Preprocessed dataset: {preprocessed_df.shape[0]} rows, {preprocessed_df.shape[1]} columns")
    
    return data_df, preprocessed_df

# Part 2: Create a SQLite database for the reference dataset
def create_ufc_database(data_df):
    """
    Store the reference data.csv in a SQLite database
    """
    # Create a SQLite database
    conn = sqlite3.connect('ufc_fights.db')
    
    # Store the main dataset
    data_df.to_sql('fights', conn, if_exists='replace', index=False)
    
    # Create fighter profiles table derived from the fights table
    fighter_query = """
    CREATE TABLE IF NOT EXISTS fighters AS
    SELECT DISTINCT fighter_name, 
           MAX(Stance) AS Stance,
           MAX(Height_cms) AS Height_cms,
           MAX(Reach_cms) AS Reach_cms,
           MAX(Weight_lbs) AS Weight_lbs
    FROM (
        SELECT R_fighter AS fighter_name, 
               R_Stance AS Stance,
               R_Height_cms AS Height_cms,
               R_Reach_cms AS Reach_cms,
               R_Weight_lbs AS Weight_lbs
        FROM fights
        UNION ALL
        SELECT B_fighter AS fighter_name, 
               B_Stance AS Stance,
               B_Height_cms AS Height_cms,
               B_Reach_cms AS Reach_cms,
               B_Weight_lbs AS Weight_lbs
        FROM fights
    ) AS fighter_data
    GROUP BY fighter_name
    """
    
    # Create views for common analytics
    wins_query = """
    CREATE VIEW fighter_records AS
    SELECT fighter_name,
           SUM(CASE WHEN winner = fighter_name THEN 1 ELSE 0 END) AS wins,
           SUM(CASE WHEN winner != fighter_name AND winner != 'Draw' THEN 1 ELSE 0 END) AS losses,
           SUM(CASE WHEN winner = 'Draw' THEN 1 ELSE 0 END) AS draws
    FROM (
        SELECT R_fighter AS fighter_name, Winner AS winner FROM fights
        UNION ALL
        SELECT B_fighter AS fighter_name, Winner AS winner FROM fights
    ) AS fight_results
    GROUP BY fighter_name
    """
    
    try:
        conn.execute(fighter_query)
        conn.execute(wins_query)
        conn.commit()
        print("Database created successfully with fights table and derived views")
    except Exception as e:
        print(f"Error creating database: {e}")
    finally:
        conn.close()

# Part 3: Prepare the machine learning dataset
def prepare_ml_dataset(preprocessed_df):
    """
    Prepare the preprocessed dataset for machine learning
    """
    # Split features and target
    X = preprocessed_df.drop(['Winner'], axis=1)
    y = preprocessed_df['Winner']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale numerical features
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # Save the processed datasets
    os.makedirs('ml_data', exist_ok=True)
    X_train.to_csv('ml_data/X_train.csv', index=False)
    X_test.to_csv('ml_data/X_test.csv', index=False)
    y_train.to_csv('ml_data/y_train.csv', index=False)
    y_test.to_csv('ml_data/y_test.csv', index=False)
    
    print("Machine learning datasets created and saved to ml_data/")
    
    return X_train, X_test, y_train, y_test

# Main execution function
def main():
    # Load data
    data_df, preprocessed_df = load_fight_data()
    
    # Create database with reference data
    create_ufc_database(data_df)
    
    # Prepare ML datasets
    X_train, X_test, y_train, y_test = prepare_ml_dataset(preprocessed_df)
    
    print("\nData preparation complete!")
    print("You now have:")
    print("1. A SQLite database (ufc_fights.db) with the reference data")
    print("2. ML-ready datasets in the ml_data/ directory")
    
    # Optional: Delete raw files you no longer need
    if input("Do you want to delete the raw_*.csv files that are no longer needed? (y/n): ").lower() == 'y':
        raw_files = [f for f in os.listdir() if f.startswith('raw_') and f.endswith('.csv')]
        for file in raw_files:
            os.remove(file)
            print(f"Deleted: {file}")

if __name__ == "__main__":
    main()