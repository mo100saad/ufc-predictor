import sqlite3
import pandas as pd
import os
from config import DATABASE_PATH, CSV_FILE_PATH

def get_db_connection():
    """Create a connection to the SQLite database"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initialize the database with tables"""
    # Make sure the data directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    # Create connection
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
        wins INTEGER,
        losses INTEGER,
        draws INTEGER
    )
    ''')
    
    # Create fights table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fights (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fighter1_id INTEGER,
        fighter2_id INTEGER,
        fighter1_age REAL,
        fighter2_age REAL,
        weight_class TEXT,
        method TEXT,
        rounds INTEGER,
        time TEXT,
        date TEXT,
        winner_id INTEGER,
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

def import_csv_to_db():
    """Import data from CSV file to SQLite database"""
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found at {CSV_FILE_PATH}")
        return
    
    # Read the CSV file
    df = pd.read_csv(CSV_FILE_PATH)
    
    # Process the dataframe and insert into database
    # This part depends on your CSV structure, but here's a general approach
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Dictionary to store fighter IDs
    fighter_ids = {}
    
    # Process each row in the dataframe
    for _, row in df.iterrows():
        # Process fighter 1
        fighter1_name = row.get('fighter1_name')
        if fighter1_name not in fighter_ids:
            # Insert new fighter
            cursor.execute('''
            INSERT INTO fighters (name, height, weight, reach, stance, wins, losses, draws)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fighter1_name,
                row.get('fighter1_height', 0),
                row.get('fighter1_weight', 0),
                row.get('fighter1_reach', 0),
                row.get('fighter1_stance', ''),
                row.get('fighter1_wins', 0),
                row.get('fighter1_losses', 0),
                row.get('fighter1_draws', 0)
            ))
            fighter_ids[fighter1_name] = cursor.lastrowid
        
        # Process fighter 2
        fighter2_name = row.get('fighter2_name')
        if fighter2_name not in fighter_ids:
            # Insert new fighter
            cursor.execute('''
            INSERT INTO fighters (name, height, weight, reach, stance, wins, losses, draws)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                fighter2_name,
                row.get('fighter2_height', 0),
                row.get('fighter2_weight', 0),
                row.get('fighter2_reach', 0),
                row.get('fighter2_stance', ''),
                row.get('fighter2_wins', 0),
                row.get('fighter2_losses', 0),
                row.get('fighter2_draws', 0)
            ))
            fighter_ids[fighter2_name] = cursor.lastrowid
        
        # Get fighter IDs
        fighter1_id = fighter_ids[fighter1_name]
        fighter2_id = fighter_ids[fighter2_name]
        
        # Determine winner
        winner_id = None
        if row.get('winner_name') == fighter1_name:
            winner_id = fighter1_id
        elif row.get('winner_name') == fighter2_name:
            winner_id = fighter2_id
        
        # Insert fight data
        cursor.execute('''
        INSERT INTO fights (fighter1_id, fighter2_id, fighter1_age, fighter2_age, 
                          weight_class, method, rounds, time, date, winner_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fighter1_id,
            fighter2_id,
            row.get('fighter1_age', 0),
            row.get('fighter2_age', 0),
            row.get('weight_class', ''),
            row.get('method', ''),
            row.get('rounds', 0),
            row.get('time', ''),
            row.get('date', ''),
            winner_id
        ))
        
        fight_id = cursor.lastrowid
        
        # Insert fight stats for fighter 1
        cursor.execute('''
        INSERT INTO fight_stats (fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                               takedowns_landed, takedowns_attempted, sig_strikes_per_min,
                               takedown_avg, sub_avg, win_streak)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fight_id,
            fighter1_id,
            row.get('fighter1_sig_strikes_landed', 0),
            row.get('fighter1_sig_strikes_attempted', 0),
            row.get('fighter1_takedowns_landed', 0),
            row.get('fighter1_takedowns_attempted', 0),
            row.get('fighter1_sig_strikes_per_min', 0),
            row.get('fighter1_takedown_avg', 0),
            row.get('fighter1_sub_avg', 0),
            row.get('fighter1_win_streak', 0)
        ))
        
        # Insert fight stats for fighter 2
        cursor.execute('''
        INSERT INTO fight_stats (fight_id, fighter_id, sig_strikes_landed, sig_strikes_attempted,
                               takedowns_landed, takedowns_attempted, sig_strikes_per_min,
                               takedown_avg, sub_avg, win_streak)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            fight_id,
            fighter2_id,
            row.get('fighter2_sig_strikes_landed', 0),
            row.get('fighter2_sig_strikes_attempted', 0),
            row.get('fighter2_takedowns_landed', 0),
            row.get('fighter2_takedowns_attempted', 0),
            row.get('fighter2_sig_strikes_per_min', 0),
            row.get('fighter2_takedown_avg', 0),
            row.get('fighter2_sub_avg', 0),
            row.get('fighter2_win_streak', 0)
        ))
    
    conn.commit()
    conn.close()
    print("CSV data imported successfully!")

def get_fight_data_for_training():
    """Retrieve processed fight data for model training"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Query to get fight data with features from both fighters
    query = '''
    SELECT 
        f.id AS fight_id,
        f1.height AS fighter1_height,
        f1.weight AS fighter1_weight,
        f1.reach AS fighter1_reach,
        f2.height AS fighter2_height,
        f2.weight AS fighter2_weight,
        f2.reach AS fighter2_reach,
        fs1.sig_strikes_per_min AS fighter1_sig_strikes_per_min,
        fs1.takedown_avg AS fighter1_takedown_avg,
        fs1.sub_avg AS fighter1_sub_avg,
        fs1.win_streak AS fighter1_win_streak,
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
    
    return df