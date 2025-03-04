import sqlite3
import pandas as pd
import os
from config import DATABASE_PATH, CSV_FILE_PATH, TRAINING_CSV_PATH

def get_db_connection():
    """
    Get a connection to the SQLite database
    Returns: sqlite3.Connection object
    """
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initialize the database schema with all required tables
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create fighters table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fighters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        height REAL,
        weight REAL,
        reach REAL,
        stance TEXT,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0
    )
    ''')
    
    # Create fights table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fighter1_id INTEGER,
        fighter2_id INTEGER,
        winner_id INTEGER,
        date TEXT,
        location TEXT,
        weight_class TEXT,
        method TEXT,
        rounds INTEGER,
        time TEXT,
        FOREIGN KEY (fighter1_id) REFERENCES fighters (id),
        FOREIGN KEY (fighter2_id) REFERENCES fighters (id),
        FOREIGN KEY (winner_id) REFERENCES fighters (id)
    )
    ''')
    
    # Create fight_stats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fight_stats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fight_id INTEGER,
        fighter_id INTEGER,
        sig_strikes_landed INTEGER,
        sig_strikes_attempted INTEGER,
        takedowns_landed INTEGER,
        takedowns_attempted INTEGER,
        sig_strikes_per_min REAL,
        takedown_avg REAL,
        sub_avg REAL,
        win_streak INTEGER,
        FOREIGN KEY (fight_id) REFERENCES fights (id),
        FOREIGN KEY (fighter_id) REFERENCES fighters (id)
    )
    ''')
    
    conn.commit()
    conn.close()
    print("Database schema initialized")

def import_csv_to_db():
    """
    Import data from the main CSV file into the database
    """
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found at {CSV_FILE_PATH}")
        return False
    
    df = pd.read_csv(CSV_FILE_PATH)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Track the IDs of fighters for reference
    fighter_ids = {}
    
    # Import fighters (Red corner)
    for _, row in df.iterrows():
        if 'R_fighter' in row and pd.notna(row['R_fighter']):
            fighter_name = row['R_fighter']
            
            # Check if fighter already exists
            cursor.execute('SELECT id FROM fighters WHERE name = ?', (fighter_name,))
            fighter = cursor.fetchone()
            
            if fighter is None:
                # Add new fighter
                cursor.execute('''
                INSERT INTO fighters (name, height, weight, reach, stance)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    fighter_name,
                    row.get('R_Height_cms') if 'R_Height_cms' in row and pd.notna(row['R_Height_cms']) else None,
                    row.get('R_Weight_lbs') if 'R_Weight_lbs' in row and pd.notna(row['R_Weight_lbs']) else None,
                    row.get('R_Reach_cms') if 'R_Reach_cms' in row and pd.notna(row['R_Reach_cms']) else None,
                    row.get('R_Stance') if 'R_Stance' in row and pd.notna(row['R_Stance']) else None
                ))
                fighter_id = cursor.lastrowid
            else:
                fighter_id = fighter['id']
                
                # Update fighter stats if we have new data
                cursor.execute('''
                UPDATE fighters SET 
                    height = COALESCE(?, height),
                    weight = COALESCE(?, weight),
                    reach = COALESCE(?, reach),
                    stance = COALESCE(?, stance)
                WHERE id = ?
                ''', (
                    row.get('R_Height_cms') if 'R_Height_cms' in row and pd.notna(row['R_Height_cms']) else None,
                    row.get('R_Weight_lbs') if 'R_Weight_lbs' in row and pd.notna(row['R_Weight_lbs']) else None,
                    row.get('R_Reach_cms') if 'R_Reach_cms' in row and pd.notna(row['R_Reach_cms']) else None,
                    row.get('R_Stance') if 'R_Stance' in row and pd.notna(row['R_Stance']) else None,
                    fighter_id
                ))
            
            fighter_ids[fighter_name] = fighter_id
    
    # Import fighters (Blue corner)
    for _, row in df.iterrows():
        if 'B_fighter' in row and pd.notna(row['B_fighter']):
            fighter_name = row['B_fighter']
            
            # Check if fighter already exists
            cursor.execute('SELECT id FROM fighters WHERE name = ?', (fighter_name,))
            fighter = cursor.fetchone()
            
            if fighter is None:
                # Add new fighter
                cursor.execute('''
                INSERT INTO fighters (name, height, weight, reach, stance)
                VALUES (?, ?, ?, ?, ?)
                ''', (
                    fighter_name,
                    row.get('B_Height_cms') if 'B_Height_cms' in row and pd.notna(row['B_Height_cms']) else None,
                    row.get('B_Weight_lbs') if 'B_Weight_lbs' in row and pd.notna(row['B_Weight_lbs']) else None,
                    row.get('B_Reach_cms') if 'B_Reach_cms' in row and pd.notna(row['B_Reach_cms']) else None,
                    row.get('B_Stance') if 'B_Stance' in row and pd.notna(row['B_Stance']) else None
                ))
                fighter_id = cursor.lastrowid
            else:
                fighter_id = fighter['id']
                
                # Update fighter stats if we have new data
                cursor.execute('''
                UPDATE fighters SET 
                    height = COALESCE(?, height),
                    weight = COALESCE(?, weight),
                    reach = COALESCE(?, reach),
                    stance = COALESCE(?, stance)
                WHERE id = ?
                ''', (
                    row.get('B_Height_cms') if 'B_Height_cms' in row and pd.notna(row['B_Height_cms']) else None,
                    row.get('B_Weight_lbs') if 'B_Weight_lbs' in row and pd.notna(row['B_Weight_lbs']) else None,
                    row.get('B_Reach_cms') if 'B_Reach_cms' in row and pd.notna(row['B_Reach_cms']) else None,
                    row.get('B_Stance') if 'B_Stance' in row and pd.notna(row['B_Stance']) else None,
                    fighter_id
                ))
            
            fighter_ids[fighter_name] = fighter_id
    
    # Import fights
    for _, row in df.iterrows():
        if 'R_fighter' in row and 'B_fighter' in row and pd.notna(row['R_fighter']) and pd.notna(row['B_fighter']):
            fighter1_name = row['R_fighter']
            fighter2_name = row['B_fighter']
            
            # Determine winner
            winner_id = None
            if 'Winner' in row and pd.notna(row['Winner']):
                if row['Winner'] == 'Red' and fighter1_name in fighter_ids:
                    winner_id = fighter_ids[fighter1_name]
                elif row['Winner'] == 'Blue' and fighter2_name in fighter_ids:
                    winner_id = fighter_ids[fighter2_name]
                elif row['Winner'] == 'Draw':
                    winner_id = None
                elif row['Winner'] in fighter_ids:  # If Winner contains fighter name
                    winner_id = fighter_ids[row['Winner']]
            
            # Add fight
            cursor.execute('''
            INSERT INTO fights (fighter1_id, fighter2_id, winner_id, date, location, weight_class, method, rounds, time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fighter_ids.get(fighter1_name),
                fighter_ids.get(fighter2_name),
                winner_id,
                row.get('date') if 'date' in row and pd.notna(row['date']) else None,
                row.get('location') if 'location' in row and pd.notna(row['location']) else None,
                row.get('weight_class') if 'weight_class' in row and pd.notna(row['weight_class']) else None,
                row.get('method') if 'method' in row and pd.notna(row['method']) else None,
                row.get('rounds') if 'rounds' in row and pd.notna(row['rounds']) else None,
                row.get('time') if 'time' in row and pd.notna(row['time']) else None
            ))
            
            fight_id = cursor.lastrowid
            
            # Add fight stats for fighter1 (Red corner)
            cursor.execute('''
            INSERT INTO fight_stats (
                fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                takedowns_landed, takedowns_attempted, sig_strikes_per_min,
                takedown_avg, sub_avg, win_streak
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fight_id,
                fighter_ids.get(fighter1_name),
                row.get('R_avg_SIG_STR_landed') if 'R_avg_SIG_STR_landed' in row and pd.notna(row['R_avg_SIG_STR_landed']) else None,
                row.get('R_avg_SIG_STR_attempted') if 'R_avg_SIG_STR_attempted' in row and pd.notna(row['R_avg_SIG_STR_attempted']) else None,
                row.get('R_avg_TD_landed') if 'R_avg_TD_landed' in row and pd.notna(row['R_avg_TD_landed']) else None,
                row.get('R_avg_TD_attempted') if 'R_avg_TD_attempted' in row and pd.notna(row['R_avg_TD_attempted']) else None,
                row.get('R_avg_SIG_STR_landed') if 'R_avg_SIG_STR_landed' in row and pd.notna(row['R_avg_SIG_STR_landed']) else None,
                row.get('R_avg_TD_landed') if 'R_avg_TD_landed' in row and pd.notna(row['R_avg_TD_landed']) else None,
                row.get('R_avg_SUB_ATT') if 'R_avg_SUB_ATT' in row and pd.notna(row['R_avg_SUB_ATT']) else None,
                row.get('fighter1_win_streak') if 'fighter1_win_streak' in row and pd.notna(row['fighter1_win_streak']) else None
            ))
            
            # Add fight stats for fighter2 (Blue corner)
            cursor.execute('''
            INSERT INTO fight_stats (
                fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                takedowns_landed, takedowns_attempted, sig_strikes_per_min,
                takedown_avg, sub_avg, win_streak
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fight_id,
                fighter_ids.get(fighter2_name),
                row.get('B_avg_SIG_STR_landed') if 'B_avg_SIG_STR_landed' in row and pd.notna(row['B_avg_SIG_STR_landed']) else None,
                row.get('B_avg_SIG_STR_attempted') if 'B_avg_SIG_STR_attempted' in row and pd.notna(row['B_avg_SIG_STR_attempted']) else None,
                row.get('B_avg_TD_landed') if 'B_avg_TD_landed' in row and pd.notna(row['B_avg_TD_landed']) else None,
                row.get('B_avg_TD_attempted') if 'B_avg_TD_attempted' in row and pd.notna(row['B_avg_TD_attempted']) else None,
                row.get('B_avg_SIG_STR_landed') if 'B_avg_SIG_STR_landed' in row and pd.notna(row['B_avg_SIG_STR_landed']) else None,
                row.get('B_avg_TD_landed') if 'B_avg_TD_landed' in row and pd.notna(row['B_avg_TD_landed']) else None,
                row.get('B_avg_SUB_ATT') if 'B_avg_SUB_ATT' in row and pd.notna(row['B_avg_SUB_ATT']) else None,
                row.get('fighter2_win_streak') if 'fighter2_win_streak' in row and pd.notna(row['fighter2_win_streak']) else None
            ))
    
    # Update fighter win/loss records
    update_fighter_records(conn)
    
    conn.commit()
    conn.close()
    print(f"Imported data from {CSV_FILE_PATH} to database")
    return True

def update_fighter_records(conn):
    """
    Update fighter win/loss/draw records based on fight data
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    # Update wins
    cursor.execute('''
    UPDATE fighters 
    SET wins = (
        SELECT COUNT(*) 
        FROM fights 
        WHERE winner_id = fighters.id
    )
    ''')
    
    # Update losses
    cursor.execute('''
    UPDATE fighters 
    SET losses = (
        SELECT COUNT(*) 
        FROM fights 
        WHERE (fighter1_id = fighters.id OR fighter2_id = fighters.id)
        AND winner_id IS NOT NULL
        AND winner_id != fighters.id
    )
    ''')
    
    # Update draws
    cursor.execute('''
    UPDATE fighters 
    SET draws = (
        SELECT COUNT(*) 
        FROM fights 
        WHERE (fighter1_id = fighters.id OR fighter2_id = fighters.id)
        AND winner_id IS NULL
    )
    ''')

def get_fight_data_for_training():
    """
    Retrieve processed fight data for model training.
    Now uses the dedicated training CSV file instead of querying the database directly.
    
    Returns:
        pd.DataFrame: DataFrame with fight data in format suitable for model training
    """
    from config import TRAINING_CSV_PATH
    import pandas as pd
    import os
    from csv_sync import verify_csv_consistency, sync_csv_files
    
    # Check if the training CSV exists
    if not os.path.exists(TRAINING_CSV_PATH):
        # If it doesn't exist, try to generate it from the main CSV
        print(f"Training CSV not found at {TRAINING_CSV_PATH}, attempting to generate it")
        sync_csv_files()
        
        # If it still doesn't exist, fall back to the database query
        if not os.path.exists(TRAINING_CSV_PATH):
            print("Failed to generate training CSV, falling back to database query")
            return get_fight_data_from_database()
    
    try:
        # Load the pre-processed training data
        df = pd.read_csv(TRAINING_CSV_PATH)
        print(f"Loaded training data from {TRAINING_CSV_PATH}: {len(df)} rows, {len(df.columns)} columns")
        
        # Make sure we have the fighter1_won target column
        if 'fighter1_won' not in df.columns:
            if 'Winner' in df.columns:
                # Convert Winner column to fighter1_won if needed
                df['fighter1_won'] = (df['Winner'] == 'Red').astype(int)
            else:
                raise ValueError("Training data does not contain 'fighter1_won' or 'Winner' column")
        
        return df
    
    except Exception as e:
        print(f"Error loading training data: {e}")
        # Fall back to the original method if anything goes wrong
        return get_fight_data_from_database()

def get_fight_data_from_database():
    """
    Original function to retrieve fight data from the database.
    Used as a fallback when the training CSV is not available.
    
    Returns:
        pd.DataFrame: DataFrame with fight data from database
    """
    import pandas as pd
    
    conn = get_db_connection()
    
    # Query to get fight data with features from both fighters
    query = '''
    SELECT 
        f.id AS fight_id,
        f1.name AS fighter1_name,
        f2.name AS fighter2_name,
        f1.height AS fighter1_height,
        f1.weight AS fighter1_weight,
        f1.reach AS fighter1_reach,
        f1.stance AS fighter1_stance,
        f1.wins AS fighter1_wins,
        f1.losses AS fighter1_losses,
        f1.draws AS fighter1_draws,
        f2.height AS fighter2_height,
        f2.weight AS fighter2_weight,
        f2.reach AS fighter2_reach,
        f2.stance AS fighter2_stance,
        f2.wins AS fighter2_wins,
        f2.losses AS fighter2_losses,
        f2.draws AS fighter2_draws,
        fs1.sig_strikes_landed AS fighter1_sig_strikes_landed,
        fs1.sig_strikes_attempted AS fighter1_sig_strikes_attempted,
        fs1.takedowns_landed AS fighter1_takedowns_landed,
        fs1.takedowns_attempted AS fighter1_takedowns_attempted,
        fs1.sig_strikes_per_min AS fighter1_sig_strikes_per_min,
        fs1.takedown_avg AS fighter1_takedown_avg,
        fs1.sub_avg AS fighter1_sub_avg,
        fs1.win_streak AS fighter1_win_streak,
        fs2.sig_strikes_landed AS fighter2_sig_strikes_landed,
        fs2.sig_strikes_attempted AS fighter2_sig_strikes_attempted,
        fs2.takedowns_landed AS fighter2_takedowns_landed,
        fs2.takedowns_attempted AS fighter2_takedowns_attempted,
        fs2.sig_strikes_per_min AS fighter2_sig_strikes_per_min,
        fs2.takedown_avg AS fighter2_takedown_avg,
        fs2.sub_avg AS fighter2_sub_avg,
        fs2.win_streak AS fighter2_win_streak,
        CASE WHEN f.winner_id = f.fighter1_id THEN 1 ELSE 0 END AS fighter1_won
    FROM 
        fights f
    JOIN 
        fighters f1 ON f.fighter1_id = f1.id
    JOIN 
        fighters f2 ON f.fighter2_id = f2.id
    JOIN 
        fight_stats fs1 ON fs1.fight_id = f.id AND fs1.fighter_id = f.fighter1_id
    JOIN 
        fight_stats fs2 ON fs2.fight_id = f.id AND fs2.fighter_id = f.fighter2_id
    WHERE 
        f.winner_id IS NOT NULL
    '''
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Add additional columns for compatibility with model
    if 'fighter1_won' not in df.columns:
        if 'winner_id' in df.columns and 'fighter1_id' in df.columns:
            df['fighter1_won'] = (df['winner_id'] == df['fighter1_id']).astype(int)
    
    # Handle non-numeric columns - convert stance to one-hot if needed
    if 'fighter1_stance' in df.columns and df['fighter1_stance'].dtype == 'object':
        # Get dummy variables for stance
        stance_dummies = pd.get_dummies(df[['fighter1_stance', 'fighter2_stance']], 
                                       prefix=['fighter1', 'fighter2'],
                                       prefix_sep='_stance_')
        # Drop original stance columns and join with dummies
        df = df.drop(['fighter1_stance', 'fighter2_stance'], axis=1)
        df = pd.concat([df, stance_dummies], axis=1)
    
    return df