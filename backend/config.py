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

# Model settings - optimized for more realistic results
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'fight_predictor_model.pth')
BATCH_SIZE = 32  # Reduced batch size for better gradient estimation
LEARNING_RATE = 0.0005  # Reduced for more precise optimization
EPOCHS = 400  # Higher count with early stopping
TEST_SIZE = 0.15  # Test data portion
VALIDATION_SIZE = 0.15  # Validation data portion

# Bias mitigation and augmentation settings
ENABLE_CROSS_VALIDATION = False  # Set to True only when you want to perform full cross-validation
BIAS_CORRECTION_LEVEL = 'medium'  # Options: 'none', 'light', 'medium', 'aggressive'
POSITION_SWAP_WEIGHT = 1.0  # Weight for position-swapped predictions (0.0-1.0)

# Quick debug mode with minimal augmentation
QUICK_MODE = False  # Set to True for fast debugging

# Augmentation controls
ENABLE_POSITION_SWAP = True  # Enable position swapping augmentation
POSITION_SWAP_FACTOR = 1  # Add 1 swapped version per original sample (2x data)
MAX_AUGMENTATION_FACTOR = 2  # Maximum multiplication of dataset size
ENABLE_DOMINANT_FIGHTER_CORRECTION = True  # Enable dominant fighter bias correction

# Training parameters for quick mode
if QUICK_MODE:
    EPOCHS = 20
    BATCH_SIZE = 64
    ENABLE_POSITION_SWAP = True
    POSITION_SWAP_FACTOR = 1  # Only double the data in quick mode
    ENABLE_DOMINANT_FIGHTER_CORRECTION = False
    
# Feature selection - whether to use reduced feature set
USE_REDUCED_FEATURES = False  # Set to True to use only essential features
REGULARIZATION_STRENGTH = 1e-3  # L2 regularization weight decay
GRADIENT_CLIP_VALUE = 0.5  # Gradient clipping threshold
EARLY_STOPPING_PATIENCE = 15  # Epochs with no improvement before stopping