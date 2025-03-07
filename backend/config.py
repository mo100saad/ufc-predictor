"""
UFC Fight Predictor Configuration

This file contains all configurable parameters for the UFC Fight Predictor model.
"""

import os

# File and directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Dataset paths
RAW_DATASET_PATH = os.path.join(DATA_DIR, 'ufc_dataset.csv')
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR, 'ufc_processed.csv')
FIGHTER_STATS_PATH = os.path.join(DATA_DIR, 'fighter_stats.csv')
#UFC_DATASET_MASTER = os.path.join(DATA_DIR, 'ufc-datasetmaster.csv')

# Model paths
ENSEMBLE_MODEL_PATH = os.path.join(MODEL_DIR, 'ensemble_model.joblib')
PYTORCH_MODEL_PATH = os.path.join(MODEL_DIR, 'pytorch_model.pth')
FEATURE_IMPORTANCE_PATH = os.path.join(MODEL_DIR, 'feature_importance.csv')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# Training configuration
RANDOM_SEED = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15
USE_ENSEMBLE = True
USE_PYTORCH = True
AUGMENT_DATA = True
FEATURE_REDUCTION = True
POSITION_SWAP_WEIGHT = 1.0  # Weight for position-swapped predictions (0.0-1.0)

# Ensemble model parameters
ENSEMBLE_PARAMS = {
    'xgboost': {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_child_weight': 2,
        'gamma': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'random_state': RANDOM_SEED
    },
    'random_forest': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'bootstrap': True,
        'random_state': RANDOM_SEED
    },
    'gradient_boosting': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'subsample': 0.8,
        'random_state': RANDOM_SEED
    },
    'logistic_regression': {
        'C': 1.0,
        'penalty': 'l2',
        'solver': 'liblinear',
        'max_iter': 1000,
        'random_state': RANDOM_SEED
    }
}

# PyTorch model parameters
PYTORCH_PARAMS = {
    'hidden_size': 128,
    'dropout_rate': 0.3,
    'batch_size': 64,
    'learning_rate': 0.001,
    'weight_decay': 0.0005,
    'epochs': 100,
    'patience': 15,
    'focal_loss': True,
    'focal_alpha': 0.25,
    'focal_gamma': 2.0
}

# Feature engineering parameters
FEATURE_ENGINEERING_PARAMS = {
    'use_physical_advantages': True,
    'use_style_indicators': True,
    'use_efficiency_metrics': True,
    'use_experience_metrics': True,
    'use_interaction_features': True,
    'normalize_features': True
}

# Logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.FileHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'filename': os.path.join(LOG_DIR, 'ufc_predictor.log'),
            'mode': 'a'
        },
    },
    'loggers': {
        '': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': True
        }
    }
}