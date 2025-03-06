import sqlite3
import pandas as pd
import os
from config import FIGHTER_STATS_PATH, DATA_DIR

# Define database path
DATABASE_PATH = os.path.join(DATA_DIR, 'ufc_fighters.db')

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
    os.makedirs(DATA_DIR, exist_ok=True)
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
        age REAL,              /* Added age column */
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        sig_strikes_per_min REAL,
        sig_strike_accuracy REAL,
        sig_strikes_absorbed_per_min REAL,
        sig_strike_defense REAL,
        takedown_avg REAL,
        takedown_accuracy REAL,
        takedown_defense REAL,
        sub_avg REAL,
        win_by_ko REAL,
        win_by_sub REAL,
        win_by_dec REAL
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


def import_fighter_stats_to_db():
    """
    Import fighter statistics from the FIGHTER_STATS_PATH CSV into the database
    This is the main function to call for populating the fighter database
    """
    if not os.path.exists(FIGHTER_STATS_PATH):
        print(f"Fighter stats file not found at {FIGHTER_STATS_PATH}")
        return False
    
    # Initialize the database if it doesn't exist
    if not os.path.exists(DATABASE_PATH):
        init_db()
    
    # Load fighter stats from CSV
    print(f"Loading fighter stats from {FIGHTER_STATS_PATH}")
    df = pd.read_csv(FIGHTER_STATS_PATH)
    
    # Filter out rows with missing name (required field)
    df = df[df['name'].notna()]
    print(f"Filtered to {len(df)} fighters with valid names")
    
    # Connect to the database
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Drop and recreate the fighters table to match the CSV structure exactly
    cursor.execute("DROP TABLE IF EXISTS fighters")
    conn.commit()
    
    # Create the fighters table with columns that exactly match the CSV
    columns = []
    for col in df.columns:
        if col == 'name':
            columns.append(f"{col} TEXT NOT NULL")
        else:
            columns.append(f"{col} REAL")
    
    create_table_sql = f"""
    CREATE TABLE fighters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {', '.join(columns)}
    )
    """
    
    cursor.execute(create_table_sql)
    conn.commit()
    
    # Print available columns for debugging
    print(f"Created table with columns: {', '.join(df.columns)}")
    
    # Import each fighter
    success_count = 0
    error_count = 0
    
    # Build the INSERT statement dynamically
    placeholders = ', '.join(['?'] * len(df.columns))
    column_names = ', '.join(df.columns)
    insert_sql = f"INSERT INTO fighters ({column_names}) VALUES ({placeholders})"
    
    for idx, row in df.iterrows():
        try:
            # Extract values, handling NaN
            values = []
            for col in df.columns:
                if pd.isna(row[col]):
                    values.append(None)
                else:
                    values.append(row[col])
            
            # Insert into database
            cursor.execute(insert_sql, values)
            success_count += 1
            
        except Exception as e:
            print(f"Error importing fighter at row {idx}: {e}")
            error_count += 1
            continue
    
    # Commit changes and close connection
    conn.commit()
    conn.close()
    
    print(f"Imported {success_count} fighters from {FIGHTER_STATS_PATH} to database")
    if error_count > 0:
        print(f"Encountered {error_count} errors during import")
    return True

def import_csv_to_db():
    """
    Import data from the main UFC dataset CSV file into the database
    """
    if not os.path.exists(CSV_FILE_PATH):
        print(f"CSV file not found at {CSV_FILE_PATH}")
        return False
    
    # Initialize the database if it doesn't exist
    if not os.path.exists(DATABASE_PATH):
        init_db()
    
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
                row.get('R_SLpM_total') if 'R_SLpM_total' in row and pd.notna(row['R_SLpM_total']) else None,
                row.get('R_td_avg') if 'R_td_avg' in row and pd.notna(row['R_td_avg']) else None,
                row.get('R_sub_avg') if 'R_sub_avg' in row and pd.notna(row['R_sub_avg']) else None,
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
                row.get('B_SLpM_total') if 'B_SLpM_total' in row and pd.notna(row['B_SLpM_total']) else None,
                row.get('B_td_avg') if 'B_td_avg' in row and pd.notna(row['B_td_avg']) else None,
                row.get('B_sub_avg') if 'B_sub_avg' in row and pd.notna(row['B_sub_avg']) else None,
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

def get_all_fighters():
    """
    Get a list of all fighters in the database
    
    Returns:
        list: List of fighter dictionaries
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM fighters
    ORDER BY name
    ''')
    
    # Convert row objects to dictionaries
    fighters = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return fighters

def get_fighter_details(fighter_id):
    """
    Get detailed information about a specific fighter
    
    Args:
        fighter_id (int): ID of the fighter
        
    Returns:
        dict: Fighter details
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get basic fighter info
    cursor.execute('''
    SELECT * FROM fighters
    WHERE id = ?
    ''', (fighter_id,))
    fighter = dict(cursor.fetchone() or {})
    
    if fighter:
        # Get fight history
        cursor.execute('''
        SELECT 
            f.id as fight_id,
            f.date,
            f.method,
            f.rounds,
            f.weight_class,
            CASE 
                WHEN f.fighter1_id = ? THEN f2.name 
                ELSE f1.name 
            END as opponent,
            CASE 
                WHEN f.winner_id = ? THEN 'Win'
                WHEN f.winner_id IS NULL THEN 'Draw'
                ELSE 'Loss'
            END as result
        FROM 
            fights f
        JOIN 
            fighters f1 ON f.fighter1_id = f1.id
        JOIN 
            fighters f2 ON f.fighter2_id = f2.id
        WHERE 
            f.fighter1_id = ? OR f.fighter2_id = ?
        ORDER BY 
            f.date DESC
        ''', (fighter_id, fighter_id, fighter_id, fighter_id))
        
        fighter['fight_history'] = [dict(row) for row in cursor.fetchall()]
        
        # Get statistics from fight_stats
        cursor.execute('''
        SELECT 
            AVG(sig_strikes_landed) as avg_sig_strikes_landed,
            AVG(takedowns_landed) as avg_takedowns_landed,
            MAX(win_streak) as max_win_streak
        FROM 
            fight_stats
        WHERE 
            fighter_id = ?
        ''', (fighter_id,))
        
        stats = dict(cursor.fetchone() or {})
        fighter.update(stats)
    
    conn.close()
    return fighter

def search_fighters(query):
    """
    Search for fighters by name
    
    Args:
        query (str): Search query
        
    Returns:
        list: List of matching fighters
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    search_term = f"%{query}%"
    cursor.execute('''
    SELECT * FROM fighters
    WHERE name LIKE ?
    ORDER BY name
    LIMIT 20
    ''', (search_term,))
    
    fighters = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return fighters

# CLI Handler
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UFC Fighter Database Manager")
    parser.add_argument('--init-db', action='store_true', help='Initialize the database schema')
    parser.add_argument('--import-fighters', action='store_true', help='Import fighter stats from CSV')
    parser.add_argument('--import-fights', action='store_true', help='Import fight data from CSV')
    parser.add_argument('--list-fighters', action='store_true', help='List all fighters')
    parser.add_argument('--search', type=str, help='Search for a fighter by name')
    parser.add_argument('--fighter-details', type=int, help='Get details for a fighter by ID')
    
    args = parser.parse_args()
    
    if args.init_db:
        init_db()
        print(f"Database initialized at {DATABASE_PATH}")
        
    if args.import_fighters:
        import_fighter_stats_to_db()
        
    if args.import_fights:
        import_csv_to_db()
        
    if args.list_fighters:
        fighters = get_all_fighters()
        print(f"Found {len(fighters)} fighters:")
        for fighter in fighters[:20]:  # Show first 20
            print(f"{fighter['id']}: {fighter['name']} ({fighter['wins']}-{fighter['losses']}-{fighter['draws']})")
        if len(fighters) > 20:
            print(f"...and {len(fighters) - 20} more")
            
    if args.search:
        fighters = search_fighters(args.search)
        print(f"Found {len(fighters)} matches for '{args.search}':")
        for fighter in fighters:
            print(f"{fighter['id']}: {fighter['name']} ({fighter['wins']}-{fighter['losses']}-{fighter['draws']})")
            
    if args.fighter_details:
        fighter = get_fighter_details(args.fighter_details)
        if fighter:
            print(f"Details for {fighter['name']}:")
            print(f"Record: {fighter['wins']}-{fighter['losses']}-{fighter['draws']}")
            print(f"Height: {fighter['height']}")
            print(f"Weight: {fighter['weight']}")
            print(f"Reach: {fighter['reach']}")
            print(f"Stance: {fighter['stance']}")
            print("\nFight History:")
            for fight in fighter.get('fight_history', [])[:5]:
                print(f"{fight['date']}: {fighter['name']} vs {fight['opponent']} - {fight['result']}")
        else:
            print(f"Fighter with ID {args.fighter_details} not found")
            
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()

'''python database.py --init-db                 # Initialize the database
python database.py --import-fighters         # Import fighter stats from CSV
python database.py --list-fighters           # List all fighters in the database
python database.py --search "McGregor"       # Search for a fighter
python database.py --fighter-details 123     # Get detailed stats for a fighter'''