import os
from database import init_db, import_fighter_stats_to_db, DATABASE_PATH

def setup_database():
    """Initialize database and import fighter data"""
    print("Setting up UFC Predictor database...")
    
    # Check if database exists
    if os.path.exists(DATABASE_PATH):
        print(f"Database already exists at {DATABASE_PATH}. Removing...")
        os.remove(DATABASE_PATH)
    
    # Initialize database
    print("Initializing database schema...")
    init_db()
    
    # Import fighter stats
    print("Importing fighter stats...")
    success = import_fighter_stats_to_db()
    
    if success:
        print("Database setup complete!")
    else:
        print("Error importing fighter stats.")

if __name__ == "__main__":
    setup_database()