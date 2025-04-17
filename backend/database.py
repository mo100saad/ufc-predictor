import sqlite3
import pandas as pd
import os
from config import FIGHTER_STATS_PATH, DATA_DIR
import logging
from utils import get_fighter_image_url

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('ufc_database')

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
    Initialize the database schema with the original tables
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Create fighters table with original schema
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS fighters (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        draws INTEGER DEFAULT 0,
        height REAL,
        weight REAL,
        reach REAL,
        stance TEXT,
        SLpM REAL,
        sig_str_acc REAL,
        SApM REAL,
        str_def REAL,
        td_avg REAL,
        td_acc REAL,
        td_def REAL,
        sub_avg REAL
    )
    ''')
    
    # Create fights table (keep this for future use)
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
    
    conn.commit()
    conn.close()
    logger.info("Database schema initialized")

def import_fighter_stats_to_db():
    """
    Import fighter statistics from the CSV into the database
    """
    if not os.path.exists(FIGHTER_STATS_PATH):
        logger.error(f"Fighter stats file not found at {FIGHTER_STATS_PATH}")
        return False
    
    # Initialize the database if it doesn't exist
    if not os.path.exists(DATABASE_PATH):
        init_db()
    
    try:
        # Load fighter stats from CSV
        logger.info(f"Loading fighter stats from {FIGHTER_STATS_PATH}")
        df = pd.read_csv(FIGHTER_STATS_PATH)
        
        # Filter out rows with missing name (required field)
        df = df[df['name'].notna()]
        logger.info(f"Filtered to {len(df)} fighters with valid names")
        
        # Connect to the database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Drop the existing fighters table to start fresh
        cursor.execute("DROP TABLE IF EXISTS fighters")
        conn.commit()
        
        # Create the fighters table with original schema
        cursor.execute('''
        CREATE TABLE fighters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            wins INTEGER DEFAULT 0,
            losses INTEGER DEFAULT 0,
            draws INTEGER DEFAULT 0,
            height REAL,
            weight REAL,
            reach REAL,
            stance TEXT,
            SLpM REAL,
            sig_str_acc REAL,
            SApM REAL,
            str_def REAL,
            td_avg REAL,
            td_acc REAL,
            td_def REAL,
            sub_avg REAL
        )
        ''')
        conn.commit()
        
        # Import each fighter
        success_count = 0
        error_count = 0
        
        insert_sql = '''
        INSERT INTO fighters (
            name, wins, losses, draws, height, weight, reach, stance,
            SLpM, sig_str_acc, SApM, str_def, td_avg, td_acc, td_def, sub_avg
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        '''
        
        for idx, row in df.iterrows():
            try:
                # Handle all columns, extracting values with proper null handling
                name = row['name'] if 'name' in row and pd.notna(row['name']) else None
                wins = int(row['wins']) if 'wins' in row and pd.notna(row['wins']) else 0
                losses = int(row['losses']) if 'losses' in row and pd.notna(row['losses']) else 0
                draws = int(row['draws']) if 'draws' in row and pd.notna(row['draws']) else 0
                height = float(row['height']) if 'height' in row and pd.notna(row['height']) else None
                weight = float(row['weight']) if 'weight' in row and pd.notna(row['weight']) else None
                reach = float(row['reach']) if 'reach' in row and pd.notna(row['reach']) else None
                stance = row['stance'] if 'stance' in row and pd.notna(row['stance']) else None
                SLpM = float(row['SLpM']) if 'SLpM' in row and pd.notna(row['SLpM']) else None
                sig_str_acc = float(row['sig_str_acc']) if 'sig_str_acc' in row and pd.notna(row['sig_str_acc']) else None
                SApM = float(row['SApM']) if 'SApM' in row and pd.notna(row['SApM']) else None
                str_def = float(row['str_def']) if 'str_def' in row and pd.notna(row['str_def']) else None
                td_avg = float(row['td_avg']) if 'td_avg' in row and pd.notna(row['td_avg']) else None
                td_acc = float(row['td_acc']) if 'td_acc' in row and pd.notna(row['td_acc']) else None
                td_def = float(row['td_def']) if 'td_def' in row and pd.notna(row['td_def']) else None
                sub_avg = float(row['sub_avg']) if 'sub_avg' in row and pd.notna(row['sub_avg']) else None
                
                if name is None:
                    continue  # Skip if name is missing
                
                # Insert into database
                cursor.execute(insert_sql, (
                    name, wins, losses, draws, height, weight, reach, stance,
                    SLpM, sig_str_acc, SApM, str_def, td_avg, td_acc, td_def, sub_avg
                ))
                success_count += 1
                
            except Exception as e:
                logger.error(f"Error importing fighter at row {idx}: {e}")
                error_count += 1
                continue
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        logger.info(f"Imported {success_count} fighters from {FIGHTER_STATS_PATH} to database")
        if error_count > 0:
            logger.warning(f"Encountered {error_count} errors during import")
        return True
        
    except Exception as e:
        logger.error(f"Error during fighter import: {e}")
        return False
    

def get_all_fighters():
    """
    Get a list of all fighters in the database
    
    Returns:
        list: List of fighter dictionaries with image URLs
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT * FROM fighters
    ORDER BY name
    ''')
    
    # Convert row objects to dictionaries with proper serialization
    fighters = []
    for row in cursor.fetchall():
        fighter = {}
        for idx, col in enumerate(cursor.description):
            value = row[idx]
            # Handle binary data for JSON serialization
            if isinstance(value, bytes):
                try:
                    fighter[col[0]] = value.decode('utf-8', errors='ignore') or None
                except:
                    fighter[col[0]] = None
            else:
                fighter[col[0]] = value
        
        # Add image URL for the fighter from cache
        if 'name' in fighter and fighter['name']:
            fighter['image_url'] = get_fighter_image_url(fighter['name'])
        
        fighters.append(fighter)
    
    conn.close()
    return fighters

def search_fighters(query):
    """
    Search for fighters by name
    
    Args:
        query (str): Search query
        
    Returns:
        list: List of matching fighters with image URLs
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
    
    # Convert to properly serializable dictionaries
    fighters = []
    for row in cursor.fetchall():
        fighter = {}
        for idx, col in enumerate(cursor.description):
            value = row[idx]
            # Handle binary data
            if isinstance(value, bytes):
                try:
                    fighter[col[0]] = value.decode('utf-8', errors='ignore') or None
                except:
                    fighter[col[0]] = None
            else:
                fighter[col[0]] = value
        
        # Add image URL for the fighter
        if 'name' in fighter and fighter['name']:
            fighter['image_url'] = get_fighter_image_url(fighter['name'])
            
        fighters.append(fighter)
    
    conn.close()
    return fighters

def get_fighter_details(fighter_id):
    """
    Get detailed information about a specific fighter
    
    Args:
        fighter_id (int): ID of the fighter
        
    Returns:
        dict: Fighter details with image URL
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # Get basic fighter info
    cursor.execute('''
    SELECT * FROM fighters
    WHERE id = ?
    ''', (fighter_id,))
    
    fighter_row = cursor.fetchone()
    if not fighter_row:
        conn.close()
        return None
    
    # Convert row to dict with proper JSON serialization
    fighter = {}
    for idx, col in enumerate(cursor.description):
        value = fighter_row[idx]
        # Handle binary data
        if isinstance(value, bytes):
            try:
                fighter[col[0]] = value.decode('utf-8', errors='ignore') or None
            except:
                fighter[col[0]] = None
        else:
            fighter[col[0]] = value
    
    # Add image URL for the fighter
    if 'name' in fighter and fighter['name']:
        fighter['image_url'] = get_fighter_image_url(fighter['name'])
    
    conn.close()
    return fighter

# CLI Handler
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="UFC Fighter Database Manager")
    parser.add_argument('--init-db', action='store_true', help='Initialize the database schema')
    parser.add_argument('--import-fighters', action='store_true', help='Import fighter stats from CSV')
    parser.add_argument('--list-fighters', action='store_true', help='List all fighters')
    parser.add_argument('--search', type=str, help='Search for a fighter by name')
    parser.add_argument('--fighter-details', type=int, help='Get details for a fighter by ID')
    
    args = parser.parse_args()
    
    if args.init_db:
        init_db()
        print(f"Database initialized at {DATABASE_PATH}")
        
    if args.import_fighters:
        import_fighter_stats_to_db()
        
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
            
            # Remove fight history section since we don't have that data anymore
            
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