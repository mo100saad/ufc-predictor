"""Configuration module for UFC predictor backend"""

import os
from pathlib import Path
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    logging.warning(f".env file not found at {env_path}. Using environment variables.")

# API keys
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')

# Application settings
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
PORT = int(os.environ.get('PORT', 5000))

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
DATABASE_PATH = os.path.join(DATA_DIR, 'ufc_fighters.db')