import os

# Flask application settings
DEBUG = True
SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_for_development')

# Database settings
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ufc_fights.db')
CSV_FILE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'ufc_data.csv')

# Model settings
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'data', 'fight_predictor_model.pth')
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 50
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1