import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ufc_model')

def load_data(file_path):
    """
    Load UFC data from CSV file
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded data
    """
    logger.info(f"Loading data from {file_path}")
    
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded data with {len(df)} rows and {len(df.columns)} columns")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def validate_data(df):
    """
    Validate that the UFC dataset has the required columns
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        bool: True if valid, False otherwise
    """
    logger.info("Validating data")
    
    # Check for required columns - either R_/B_ or fighter1_/fighter2_ format
    required_patterns = [
        ['R_fighter', 'B_fighter', 'Winner'],
        ['fighter1_name', 'fighter2_name', 'fighter1_won']
    ]
    
    is_valid = False
    
    for pattern in required_patterns:
        if all(any(col in df.columns for col in [p, p.lower()]) for p in pattern):
            is_valid = True
            break
    
    if not is_valid:
        logger.warning("Data validation failed: Missing required columns")
        return False
    
    # Check for sufficient numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) < 10:
        logger.warning("Data validation failed: Insufficient numeric features")
        return False
    
    # Check for target variable
    if 'Winner' in df.columns:
        if not df['Winner'].isin(['Red', 'Blue']).all():
            logger.warning("Data validation failed: 'Winner' column has invalid values")
            return False
    elif 'fighter1_won' in df.columns:
        if not df['fighter1_won'].isin([0, 1]).all():
            logger.warning("Data validation failed: 'fighter1_won' column has invalid values")
            return False
    
    logger.info("Data validation passed")
    return True

def convert_column_format(df):
    """
    Convert between R_/B_ and fighter1_/fighter2_ column formats
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        pd.DataFrame: Data with consistent column naming
    """
    logger.info("Converting column format for consistency")
    
    # Make a copy of the dataframe
    df_out = df.copy()
    
    # Check if we have R_/B_ format
    has_rb_format = any(col.startswith('R_') or col.startswith('B_') for col in df.columns)
    
    # Check if we have fighter1_/fighter2_ format
    has_fighter_format = any(col.startswith('fighter1_') or col.startswith('fighter2_') for col in df.columns)
    
    # Convert R_/B_ to fighter1_/fighter2_ if needed
    if has_rb_format and not has_fighter_format:
        logger.info("Converting R_/B_ format to fighter1_/fighter2_")
        
        # Create mapping for renaming
        rename_mapping = {}
        
        for col in df.columns:
            if col.startswith('R_'):
                rename_mapping[col] = col.replace('R_', 'fighter1_')
            elif col.startswith('B_'):
                rename_mapping[col] = col.replace('B_', 'fighter2_')
        
        # Rename columns
        df_out = df_out.rename(columns=rename_mapping)
        
        # Convert 'Winner' column to 'fighter1_won' if it exists
        if 'Winner' in df_out.columns:
            df_out['fighter1_won'] = (df_out['Winner'] == 'Red').astype(int)
    
    # Convert fighter1_/fighter2_ to R_/B_ if needed (this is less common)
    elif has_fighter_format and not has_rb_format:
        logger.info("Converting fighter1_/fighter2_ format to R_/B_")
        
        # Create mapping for renaming
        rename_mapping = {}
        
        for col in df.columns:
            if col.startswith('fighter1_'):
                rename_mapping[col] = col.replace('fighter1_', 'R_')
            elif col.startswith('fighter2_'):
                rename_mapping[col] = col.replace('fighter2_', 'B_')
        
        # Rename columns
        df_out = df_out.rename(columns=rename_mapping)
        
        # Convert 'fighter1_won' column to 'Winner' if it exists
        if 'fighter1_won' in df_out.columns:
            df_out['Winner'] = df_out['fighter1_won'].map({1: 'Red', 0: 'Blue'})
    
    logger.info("Column format conversion complete")
    return df_out

def create_feature_pairs(df):
    """
    Create a mapping of feature pairs (fighter1/fighter2 or R/B)
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        dict: Dictionary mapping feature pairs
    """
    # Determine column prefix format
    if any(col.startswith('fighter1_') for col in df.columns):
        prefix1, prefix2 = 'fighter1_', 'fighter2_'
    else:
        prefix1, prefix2 = 'R_', 'B_'
    
    # Find all fighter1/R columns
    fighter1_cols = [col for col in df.columns if col.startswith(prefix1)]
    
    # Create mapping to fighter2/B columns
    pairs = {}
    for col1 in fighter1_cols:
        col2 = col1.replace(prefix1, prefix2)
        if col2 in df.columns:
            pairs[col1] = col2
    
    return pairs

def handle_missing_values(df):
    """
    Handle missing values in UFC data
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        pd.DataFrame: Data with missing values handled
    """
    logger.info("Handling missing values")
    
    # Make a copy of the dataframe
    df_out = df.copy()
    
    # Get feature pairs
    feature_pairs = create_feature_pairs(df)
    
    # Handle missing values for each column
    for col in df_out.columns:
        # Skip non-numeric columns except for the target variable
        if df_out[col].dtype == 'object' and col not in ['fighter1_won', 'Winner']:
            continue
            
        # Check if column has missing values
        if df_out[col].isna().any():
            # For fighter attributes, use the median
            if col.endswith('_height') or col.endswith('_weight') or col.endswith('_reach') or col.endswith('_age'):
                df_out[col] = df_out[col].fillna(df_out[col].median())
            
            # For fight records, use 0
            elif 'win' in col.lower() or 'loss' in col.lower() or col.endswith('_draw'):
                df_out[col] = df_out[col].fillna(0)
            
            # For percentages, use the mean
            elif col.endswith('_pct') or '_accuracy' in col:
                df_out[col] = df_out[col].fillna(df_out[col].mean())
            
            # For other numeric columns, use the median
            else:
                df_out[col] = df_out[col].fillna(df_out[col].median())
    
    logger.info("Missing value handling complete")
    return df_out

def create_advantage_features(df):
    """
    Create advantage features comparing fighter1/R with fighter2/B
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        pd.DataFrame: Data with advantage features added
    """
    logger.info("Creating advantage features")
    
    # Make a copy of the dataframe
    df_out = df.copy()
    
    # Get feature pairs
    feature_pairs = create_feature_pairs(df)
    
    # Create advantage features
    for col1, col2 in feature_pairs.items():
        # Skip non-numeric columns
        if df[col1].dtype == 'object' or df[col2].dtype == 'object':
            continue
            
        # Extract the feature name without the prefix
        if '_' in col1:
            feature_name = col1.split('_', 1)[1]
        else:
            continue
            
        # Create advantage feature
        advantage_col = f"{feature_name}_advantage"
        df_out[advantage_col] = df[col1] - df[col2]
    
    # Special case for win percentage
    if 'fighter1_wins' in df_out.columns and 'fighter1_losses' in df_out.columns and \
       'fighter2_wins' in df_out.columns and 'fighter2_losses' in df_out.columns:
        
        # Calculate total fights
        df_out['fighter1_total_fights'] = df_out['fighter1_wins'] + df_out['fighter1_losses']
        df_out['fighter2_total_fights'] = df_out['fighter2_wins'] + df_out['fighter2_losses']
        
        # Calculate win percentage (avoid division by zero)
        df_out['fighter1_win_pct'] = df_out['fighter1_wins'] / df_out['fighter1_total_fights'].replace(0, 1)
        df_out['fighter2_win_pct'] = df_out['fighter2_wins'] / df_out['fighter2_total_fights'].replace(0, 1)
        
        # Calculate win percentage advantage
        df_out['win_pct_advantage'] = df_out['fighter1_win_pct'] - df_out['fighter2_win_pct']
        
        # Calculate experience advantage
        df_out['experience_advantage'] = df_out['fighter1_total_fights'] - df_out['fighter2_total_fights']
    
    logger.info(f"Created {sum('advantage' in col for col in df_out.columns)} advantage features")
    return df_out

def encode_categorical_features(df):
    """
    Encode categorical features in UFC data
    
    Args:
        df (pd.DataFrame): UFC data
        
    Returns:
        pd.DataFrame: Data with categorical features encoded
    """
    logger.info("Encoding categorical features")
    
    # Make a copy of the dataframe
    df_out = df.copy()
    
    # Get feature pairs
    feature_pairs = create_feature_pairs(df)
    
    # Find categorical columns to encode
    categorical_cols = []
    for col in df.columns:
        # Skip target variables and non-object columns
        if col in ['fighter1_won', 'Winner'] or df[col].dtype != 'object':
            continue
            
        # Add to categorical columns
        categorical_cols.append(col)
    
    # Use one-hot encoding for categorical columns
    for col in categorical_cols:
        # Skip if column is all NaN
        if df[col].isna().all():
            continue
            
        # Create dummy variables
        dummies = pd.get_dummies(df[col], prefix=col, dummy_na=True)
        
        # Drop the original column and join with dummies
        df_out = df_out.drop(col, axis=1)
        df_out = pd.concat([df_out, dummies], axis=1)
    
    logger.info("Categorical feature encoding complete")
    return df_out

def normalize_features(df, target_col='fighter1_won'):
    """
    Normalize numeric features in UFC data
    
    Args:
        df (pd.DataFrame): UFC data
        target_col (str): Target column name
        
    Returns:
        pd.DataFrame: Data with normalized features
    """
    logger.info("Normalizing features")
    
    # Make a copy of the dataframe
    df_out = df.copy()
    
    # Identify numeric columns to normalize
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove target column from normalization
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Apply min-max normalization to each numeric column
    for col in numeric_cols:
        # Skip columns that are already normalized (0-1 range)
        if df[col].min() >= 0 and df[col].max() <= 1:
            continue
            
        # Skip columns with all identical values
        if df[col].nunique() <= 1:
            continue
            
        # Apply min-max normalization
        col_min = df[col].min()
        col_max = df[col].max()
        
        if col_max > col_min:
            df_out[col] = (df[col] - col_min) / (col_max - col_min)
    
    logger.info("Feature normalization complete")
    return df_out

def select_features(df, target_col='fighter1_won', max_features=None):
    """
    Select the most important features for UFC prediction
    
    Args:
        df (pd.DataFrame): UFC data
        target_col (str): Target column name
        max_features (int): Maximum number of features to select
        
    Returns:
        pd.DataFrame: Data with selected features
    """
    logger.info("Selecting features")
    
    # Essential features to keep
    essential_features = [
        # Target variable
        target_col,
        
        # Win records and percentages
        'fighter1_wins', 'fighter1_losses', 'fighter2_wins', 'fighter2_losses',
        'fighter1_win_pct', 'fighter2_win_pct', 'win_pct_advantage',
        
        # Physical attributes
        'fighter1_height', 'fighter2_height', 'height_advantage',
        'fighter1_reach', 'fighter2_reach', 'reach_advantage',
        'fighter1_weight', 'fighter2_weight', 'weight_advantage',
        
        # Age
        'fighter1_age', 'fighter2_age',
        
        # Striking stats
        'fighter1_sig_strikes_per_min', 'fighter2_sig_strikes_per_min',
        'fighter1_sig_strike_accuracy', 'fighter2_sig_strike_accuracy',
        'striking_volume_advantage', 'striking_accuracy_advantage',
        
        # Takedown stats
        'fighter1_takedown_avg', 'fighter2_takedown_avg',
        'fighter1_takedown_accuracy', 'fighter2_takedown_accuracy',
        'fighter1_takedown_defense', 'fighter2_takedown_defense',
        'takedown_advantage',
        
        # Submission stats
        'fighter1_sub_avg', 'fighter2_sub_avg', 'submission_advantage',
        
        # Fight outcome stats
        'fighter1_win_by_KO_TKO', 'fighter2_win_by_KO_TKO',
        'fighter1_win_by_SUB', 'fighter2_win_by_SUB',
        'fighter1_win_by_DEC', 'fighter2_win_by_DEC'
    ]
    
    # Add common alternative column names
    alternative_names = []
    for feature in essential_features:
        if feature != target_col:
            # Add R_/B_ version of the feature
            if feature.startswith('fighter1_'):
                alternative_names.append(feature.replace('fighter1_', 'R_'))
            elif feature.startswith('fighter2_'):
                alternative_names.append(feature.replace('fighter2_', 'B_'))
            
            # Add some common variations of feature names
            for old, new in [
                ('sig_strikes_per_min', 'avg_SIG_STR_landed'),
                ('sig_strike_accuracy', 'avg_SIG_STR_pct'),
                ('takedown_avg', 'avg_TD_landed'),
                ('takedown_accuracy', 'avg_TD_pct'),
                ('sub_avg', 'avg_SUB_ATT'),
                ('win_by_KO_TKO', 'win_by_KO/TKO')
            ]:
                if old in feature:
                    alt_feature = feature.replace(old, new)
                    alternative_names.append(alt_feature)
    
    # Add alternative names to essential features
    essential_features.extend(alternative_names)
    
    # Find columns in dataframe that match essential features
    available_features = []
    for feature in essential_features:
        if feature in df.columns:
            available_features.append(feature)
    
    # Make sure target column is included
    if target_col not in available_features and target_col in df.columns:
        available_features.append(target_col)
    
    # If max_features is specified, limit the number of features
    if max_features is not None and len(available_features) > max_features:
        # Always keep the target column
        feature_subset = [target_col]
        
        # Add advantage features first
        advantage_features = [f for f in available_features if 'advantage' in f]
        feature_subset.extend(advantage_features[:max_features//4])
        
        # Add other features
        other_features = [f for f in available_features if f != target_col and 'advantage' not in f]
        feature_subset.extend(other_features[:max_features - len(feature_subset)])
        
        available_features = feature_subset
    
    logger.info(f"Selected {len(available_features)} features")
    return df[available_features]

def prepare_training_data(df, target_col='fighter1_won', test_size=0.2, val_size=0.15, random_seed=42):
    """
    Prepare training, validation, and test datasets
    
    Args:
        df (pd.DataFrame): UFC data
        target_col (str): Target column name
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info("Preparing training data")
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # Split into training+validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Split training+validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),
        random_state=random_seed, 
        stratify=y_train_val
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Create PyTorch DataLoaders for training
    
    Args:
        X_train, X_val, X_test: Feature dataframes
        y_train, y_val, y_test: Target series
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler, feature_columns)
    """
    logger.info(f"Creating dataloaders with batch size {batch_size}")
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
    
    X_val_tensor = torch.FloatTensor(X_val_scaled)
    y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1))
    
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))
    
    # Create datasets
    class UFCDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
    
    train_dataset = UFCDataset(X_train_tensor, y_train_tensor)
    val_dataset = UFCDataset(X_val_tensor, y_val_tensor)
    test_dataset = UFCDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    logger.info("Created dataloaders")
    return train_loader, val_loader, test_loader, scaler, X_train.columns.tolist()

class UFCPredictor(nn.Module):
    """
    UFC Fight Predictor Neural Network
    
    A balanced network with 2 hidden layers and dropout regularization
    """
    def __init__(self, input_size):
        super(UFCPredictor, self).__init__()
        # First hidden layer
        self.layer1 = nn.Linear(input_size, 32)
        # Dropout for regularization
        self.dropout1 = nn.Dropout(p=0.3)
        # Second hidden layer
        self.layer2 = nn.Linear(32, 16)
        # Dropout for regularization
        self.dropout2 = nn.Dropout(p=0.3)
        # Output layer
        self.output = nn.Linear(16, 1)
        
        # Initialize weights with Xavier initialization
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        # First hidden layer with ReLU activation
        x = torch.relu(self.layer1(x))
        # Apply dropout
        x = self.dropout1(x)
        # Second hidden layer with ReLU activation
        x = torch.relu(self.layer2(x))
        # Apply dropout
        x = self.dropout2(x)
        # Output layer with sigmoid activation for probability output
        x = torch.sigmoid(self.output(x))
        return x

def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, 
               weight_decay=1e-4, patience=10, model_path=None):
    """
    Train the UFC prediction model
    
    Args:
        model (nn.Module): Model to train
        train_loader (DataLoader): Training data loader
        val_loader (DataLoader): Validation data loader
        epochs (int): Maximum number of epochs
        learning_rate (float): Learning rate
        weight_decay (float): L2 regularization strength
        patience (int): Early stopping patience
        model_path (str): Path to save the model
        
    Returns:
        tuple: (trained_model, history)
    """
    logger.info(f"Training model for {epochs} epochs (lr={learning_rate}, weight_decay={weight_decay})")
    
    # Determine device (use GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Move model to the device
    model = model.to(device)
    
    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.BCELoss()
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epoch = 0
    
    # Training history
    history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        # Calculate average training loss
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                # Move data to device
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                
                # Save predictions and targets for metrics
                val_preds.extend((outputs > 0.5).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Calculate validation accuracy
        from sklearn.metrics import accuracy_score
        val_accuracy = accuracy_score(val_targets, val_preds)
        
        # Update history
        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)
        
        # Print progress
        logger.info(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epoch = 0
            
            # Save the best model so far
            if model_path:
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                logger.info(f"Saved improved model to {model_path}")
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= patience:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    # Save history
    if model_path:
        history_path = os.path.join(os.path.dirname(model_path), 'training_history.csv')
        pd.DataFrame(history).to_csv(history_path, index=False)
        logger.info(f"Saved training history to {history_path}")
    
    return model, history

def evaluate_model(model, test_loader):
    """
    Evaluate the model on the test set
    
    Args:
        model (nn.Module): Trained model
        test_loader (DataLoader): Test data loader
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model on test set")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to the device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Collect predictions and targets
    test_preds = []
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            # Move data to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get predictions
            outputs = model(inputs)
            
            # Save predictions, probabilities, and targets
            test_preds.extend((outputs > 0.5).cpu().numpy())
            test_probs.extend(outputs.cpu().numpy())
            test_targets.extend(targets.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    test_preds = np.array(test_preds).flatten()
    test_probs = np.array(test_probs).flatten()
    test_targets = np.array(test_targets).flatten()
    
    metrics = {
        'accuracy': accuracy_score(test_targets, test_preds),
        'precision': precision_score(test_targets, test_preds),
        'recall': recall_score(test_targets, test_preds),
        'f1': f1_score(test_targets, test_preds),
        'auc': roc_auc_score(test_targets, test_probs)
    }
    
    # Print metrics
    logger.info(f"Test accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test precision: {metrics['precision']:.4f}")
    logger.info(f"Test recall: {metrics['recall']:.4f}")
    logger.info(f"Test F1 score: {metrics['f1']:.4f}")
    logger.info(f"Test AUC: {metrics['auc']:.4f}")
    
    return metrics

def plot_training_history(history):
    """
    Plot training history
    
    Args:
        history (dict): Training history
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    
    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['epoch'], history['train_loss'], label='Train Loss')
    plt.plot(history['epoch'], history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # Plot validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['epoch'], history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.title('Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    logger.info("Training history plot saved to training_history.png")

def compute_feature_importance(model, X_train, scaler):
    """
    Compute feature importance using permutation approach
    
    Args:
        model (nn.Module): Trained model
        X_train (pd.DataFrame): Training features
        scaler (StandardScaler): Feature scaler
        
    Returns:
        pd.DataFrame: Feature importance
    """
    logger.info("Computing feature importance")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to the device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Scale the features
    X_scaled = scaler.transform(X_train)
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    # Compute baseline predictions
    with torch.no_grad():
        baseline_preds = model(X_tensor).cpu().numpy()
    
    # Compute feature importance
    feature_importance = []
    
    for i, feature_name in enumerate(X_train.columns):
        # Copy the scaled data
        X_permuted = X_scaled.copy()
        
        # Permute the feature
        np.random.shuffle(X_permuted[:, i])
        
        # Convert to tensor and move to device
        X_permuted_tensor = torch.FloatTensor(X_permuted).to(device)
        
        # Compute predictions with permuted feature
        with torch.no_grad():
            permuted_preds = model(X_permuted_tensor).cpu().numpy()
        
        # Compute importance as mean absolute difference
        importance = np.mean(np.abs(baseline_preds - permuted_preds))
        feature_importance.append((feature_name, importance))
    
    # Convert to DataFrame and sort by importance
    importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    
    # Plot feature importance (top 15)
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
    plt.xlabel('Importance')
    plt.title('Top 15 Features by Importance')
    plt.gca().invert_yaxis()  # Display highest importance at the top
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()
    
    logger.info("Feature importance plot saved to feature_importance.png")
    
    return importance_df

def predict_fight(model, fighter1_data, fighter2_data, scaler, feature_columns):
    """
    Predict the outcome of a UFC fight
    
    Args:
        model (nn.Module): Trained model
        fighter1_data (dict): Fighter 1 (Red corner) data
        fighter2_data (dict): Fighter 2 (Blue corner) data
        scaler (StandardScaler): Feature scaler
        feature_columns (list): Feature column names
        
    Returns:
        dict: Prediction results
    """
    logger.info("Predicting fight outcome")
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Move model to the device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    # Prepare features
    features = {}
    
    # Add fighter1 features with correct prefix
    for key, value in fighter1_data.items():
        if not key.startswith('fighter1_'):
            features[f'fighter1_{key}'] = value
        else:
            features[key] = value
    
    # Add fighter2 features with correct prefix
    for key, value in fighter2_data.items():
        if not key.startswith('fighter2_'):
            features[f'fighter2_{key}'] = value
        else:
            features[key] = value
    
    # Calculate advantage features
    if 'fighter1_height' in features and 'fighter2_height' in features:
        features['height_advantage'] = features['fighter1_height'] - features['fighter2_height']
    
    if 'fighter1_reach' in features and 'fighter2_reach' in features:
        features['reach_advantage'] = features['fighter1_reach'] - features['fighter2_reach']
    
    if 'fighter1_weight' in features and 'fighter2_weight' in features:
        features['weight_advantage'] = features['fighter1_weight'] - features['fighter2_weight']
    
    if 'fighter1_wins' in features and 'fighter1_losses' in features and \
       'fighter2_wins' in features and 'fighter2_losses' in features:
        # Calculate total fights
        features['fighter1_total_fights'] = features['fighter1_wins'] + features['fighter1_losses']
        features['fighter2_total_fights'] = features['fighter2_wins'] + features['fighter2_losses']
        
        # Calculate win percentage
        if features['fighter1_total_fights'] > 0 and features['fighter2_total_fights'] > 0:
            features['fighter1_win_pct'] = features['fighter1_wins'] / features['fighter1_total_fights']
            features['fighter2_win_pct'] = features['fighter2_wins'] / features['fighter2_total_fights']
            features['win_pct_advantage'] = features['fighter1_win_pct'] - features['fighter2_win_pct']
        
        # Calculate experience advantage
        features['experience_advantage'] = features['fighter1_total_fights'] - features['fighter2_total_fights']
    
    # Create dataframe with features
    features_df = pd.DataFrame([features])
    
    # Ensure we have all required columns
    for col in feature_columns:
        if col not in features_df.columns:
            features_df[col] = 0
    
    # Keep only the columns in feature_columns
    features_df = features_df[feature_columns]
    
    # Scale the features
    features_scaled = scaler.transform(features_df)
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(features_scaled).to(device)
    
    # Make prediction
    with torch.no_grad():
        win_probability = model(features_tensor).item()
    
    # Create result
    result = {
        'probability_red_wins': float(win_probability),
        'probability_blue_wins': float(1 - win_probability),
        'predicted_winner': 'Red' if win_probability > 0.5 else 'Blue',
        'confidence_level': 'High' if abs(win_probability - 0.5) > 0.25 else 
                           'Medium' if abs(win_probability - 0.5) > 0.1 else 'Low'
    }
    
    logger.info(f"Prediction: {result['predicted_winner']} with {result['confidence_level']} confidence "
                f"({result['probability_red_wins']:.4f} vs {result['probability_blue_wins']:.4f})")
    
    return result