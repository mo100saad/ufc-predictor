import os

# Flask application settings
DEBUG = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_development')

# Database settings
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ufc_fights.db')
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ufc_data.csv')  # Main data CSV for lookups
TRAINING_CSV_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ufc_training_data.csv')  # Processed CSV for training
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'data', 'scaler.save')

# CSV file settings
CSV_SYNC_ON_STARTUP = True  # Whether to check CSV consistency on startup

# Model settings - optimized for better accuracy
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'fight_predictor_model.pth')
BATCH_SIZE = 32  # Reduced from 64 for better gradient estimation
LEARNING_RATE = 0.0005  # Reduced for more precise optimization
EPOCHS = 400   # Increased to allow more training iteration EARLY STOPPAGE IMPLEMENTED IN CASE
TEST_SIZE = 0.15  # Slightly reduced to provide more training data
VALIDATION_SIZE = 0.15  # Increased for better validation