import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ufc_predictor.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ufc_model')

class UFCDataset(Dataset):
    """PyTorch Dataset for UFC fights"""
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class UFCPredictor(nn.Module):
    """
    UFC Fight Predictor Neural Network
    
    A balanced network with 2 hidden layers (32 → 16 → 1) and dropout regularization
    to capture fight dynamics without overfitting.
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

def preprocess_data(raw_data):
    """
    Preprocess UFC fight data for model training
    
    Args:
        raw_data (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Processed dataset for modeling
    """
    logger.info("Starting data preprocessing")
    
    try:
        # Create a copy of the dataframe
        data_df = raw_data.copy()
        
        # Standardize column names (if needed)
        if any(col.startswith('R_') or col.startswith('B_') for col in data_df.columns):
            rename_mapping = {
                col: col.replace('R_', 'fighter1_').replace('B_', 'fighter2_') 
                for col in data_df.columns
            }
            data_df = data_df.rename(columns=rename_mapping)
        
        # Ensure we have the target variable
        if 'Winner' in data_df.columns:
            # Map Red/Blue to 1/0 for fighter1_won
            data_df['fighter1_won'] = (data_df['Winner'] == 'Red').astype(int)
        elif 'winner_id' in data_df.columns and 'fighter1_id' in data_df.columns:
            # If we have IDs instead, use those to determine winner
            data_df['fighter1_won'] = (data_df['winner_id'] == data_df['fighter1_id']).astype(int)
        elif 'fighter1_won' not in data_df.columns:
            # If we don't have a winner column at all, raise an error
            raise ValueError("Dataset must have either 'Winner', 'winner_id', or 'fighter1_won' column")
        
        # Select only numeric columns for modeling
        numeric_data = data_df.select_dtypes(include=[np.number])
        
        # Handle missing values
        for col in numeric_data.columns:
            if numeric_data[col].isna().any():
                if col == 'fighter1_won':
                    # Drop rows with missing target variable
                    numeric_data = numeric_data.dropna(subset=[col])
                else:
                    # Use median imputation for features
                    if col.endswith('_height') or col.endswith('_weight'):
                        numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())
                    else:
                        numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())
        
        logger.info(f"Preprocessed data shape: {numeric_data.shape}")
        return numeric_data
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

def calculate_advantage_features(data_df):
    """
    Calculate fighter advantage features (height, reach, etc.)
    
    Args:
        data_df (pd.DataFrame): Processed dataset
        
    Returns:
        pd.DataFrame: Dataset with advantage features added
    """
    logger.info("Calculating advantage features")
    
    try:
        # Create a copy of the dataframe
        df = data_df.copy()
        
        # Physical advantages (height, reach, weight)
        if all(col in df.columns for col in ['fighter1_height', 'fighter2_height']):
            df['height_advantage'] = df['fighter1_height'] - df['fighter2_height']
        
        if all(col in df.columns for col in ['fighter1_reach', 'fighter2_reach']):
            df['reach_advantage'] = df['fighter1_reach'] - df['fighter2_reach']
        
        if all(col in df.columns for col in ['fighter1_weight', 'fighter2_weight']):
            df['weight_advantage'] = df['fighter1_weight'] - df['fighter2_weight']
        
        # Experience advantage
        if all(col in df.columns for col in ['fighter1_wins', 'fighter1_losses', 'fighter2_wins', 'fighter2_losses']):
            df['fighter1_total_fights'] = df['fighter1_wins'] + df['fighter1_losses']
            df['fighter2_total_fights'] = df['fighter2_wins'] + df['fighter2_losses']
            df['experience_advantage'] = df['fighter1_total_fights'] - df['fighter2_total_fights']
            
            # Win percentage advantage
            df['fighter1_win_pct'] = df['fighter1_wins'] / df['fighter1_total_fights'].replace(0, 1)
            df['fighter2_win_pct'] = df['fighter2_wins'] / df['fighter2_total_fights'].replace(0, 1)
            df['win_pct_advantage'] = df['fighter1_win_pct'] - df['fighter2_win_pct']
        
        # Striking advantages
        if all(col in df.columns for col in ['fighter1_sig_strikes_per_min', 'fighter2_sig_strikes_per_min']):
            df['striking_volume_advantage'] = df['fighter1_sig_strikes_per_min'] - df['fighter2_sig_strikes_per_min']
        
        if all(col in df.columns for col in ['fighter1_sig_strike_accuracy', 'fighter2_sig_strike_accuracy']):
            df['striking_accuracy_advantage'] = df['fighter1_sig_strike_accuracy'] - df['fighter2_sig_strike_accuracy']
        
        # Grappling advantages
        if all(col in df.columns for col in ['fighter1_takedown_avg', 'fighter2_takedown_avg']):
            df['takedown_advantage'] = df['fighter1_takedown_avg'] - df['fighter2_takedown_avg']
        
        if all(col in df.columns for col in ['fighter1_takedown_defense', 'fighter2_takedown_defense']):
            df['takedown_defense_advantage'] = df['fighter1_takedown_defense'] - df['fighter2_takedown_defense']
        
        if all(col in df.columns for col in ['fighter1_sub_avg', 'fighter2_sub_avg']):
            df['submission_advantage'] = df['fighter1_sub_avg'] - df['fighter2_sub_avg']
        
        logger.info(f"Added advantage features. New shape: {df.shape}")
        return df
    
    except Exception as e:
        logger.error(f"Error calculating advantage features: {e}")
        # Return original dataframe if there's an error
        return data_df

def select_key_features(data_df, use_reduced_set=False):
    """
    Select the most important features for the model
    
    Args:
        data_df (pd.DataFrame): Processed dataset
        use_reduced_set (bool): Whether to use a reduced feature set
        
    Returns:
        pd.DataFrame: Dataset with selected features
    """
    logger.info(f"Selecting features (reduced set: {use_reduced_set})")
    
    try:
        # Create a copy of the dataframe
        df = data_df.copy()
        
        # Define key features to keep
        key_features = [
            # Target variable
            'fighter1_won',
            
            # Win rate and experience
            'fighter1_wins', 'fighter1_losses', 'fighter2_wins', 'fighter2_losses',
            'fighter1_win_pct', 'fighter2_win_pct', 'experience_advantage', 'win_pct_advantage',
            
            # Physical attributes
            'fighter1_height', 'fighter2_height', 'height_advantage',
            'fighter1_reach', 'fighter2_reach', 'reach_advantage',
            
            # Age
            'fighter1_age', 'fighter2_age',
            
            # Striking stats
            'fighter1_sig_strikes_per_min', 'fighter2_sig_strikes_per_min', 
            'fighter1_sig_strike_accuracy', 'fighter2_sig_strike_accuracy',
            'striking_volume_advantage', 'striking_accuracy_advantage',
            
            # Defense stats
            'fighter1_sig_strikes_absorbed_per_min', 'fighter2_sig_strikes_absorbed_per_min',
            'fighter1_sig_strike_defense', 'fighter2_sig_strike_defense',
            
            # Takedown stats
            'fighter1_takedown_avg', 'fighter2_takedown_avg',
            'fighter1_takedown_accuracy', 'fighter2_takedown_accuracy',
            'fighter1_takedown_defense', 'fighter2_takedown_defense',
            'takedown_advantage', 'takedown_defense_advantage',
            
            # Submission stats
            'fighter1_sub_avg', 'fighter2_sub_avg', 'submission_advantage',
            
            # Fight method stats
            'fighter1_win_by_KO_TKO', 'fighter2_win_by_KO_TKO',
            'fighter1_win_by_SUB', 'fighter2_win_by_SUB',
            'fighter1_win_by_DEC', 'fighter2_win_by_DEC'
        ]
        
        # For extremely reduced set (if needed)
        if use_reduced_set:
            minimal_features = [
                'fighter1_won', 'win_pct_advantage', 'experience_advantage',
                'height_advantage', 'reach_advantage',
                'striking_volume_advantage', 'striking_accuracy_advantage',
                'takedown_advantage', 'takedown_defense_advantage', 'submission_advantage'
            ]
            key_features = minimal_features
        
        # Keep only available key features
        available_features = [col for col in key_features if col in df.columns]
        
        # Ensure target variable is included
        if 'fighter1_won' not in available_features and 'fighter1_won' in df.columns:
            available_features.append('fighter1_won')
        
        logger.info(f"Selected {len(available_features)} features from {df.shape[1]} columns")
        
        # Return dataframe with selected features
        return df[available_features]
    
    except Exception as e:
        logger.error(f"Error selecting features: {e}")
        # Ensure target variable is included in the returned dataframe
        if 'fighter1_won' in data_df.columns:
            return data_df[['fighter1_won'] + [col for col in data_df.columns if col != 'fighter1_won']]
        return data_df

def train_test_val_split(data_df, test_size=0.15, val_size=0.15):
    """
    Split dataset into train, validation, and test sets
    
    Args:
        data_df (pd.DataFrame): Processed dataset
        test_size (float): Portion of data for testing
        val_size (float): Portion of data for validation
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    logger.info(f"Splitting data (test: {test_size}, val: {val_size})")
    
    try:
        # Separate features and target
        X = data_df.drop('fighter1_won', axis=1)
        y = data_df['fighter1_won']
        
        # Split into train+val and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Split train+val into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_size/(1-test_size),
            random_state=42, 
            stratify=y_train_val
        )
        
        logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    except Exception as e:
        logger.error(f"Error splitting data: {e}")
        raise

def prepare_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=32):
    """
    Prepare PyTorch DataLoaders for training, validation, and testing
    
    Args:
        X_train, X_val, X_test: Features
        y_train, y_val, y_test: Labels
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, scaler)
    """
    logger.info(f"Preparing dataloaders with batch size {batch_size}")
    
    try:
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
        
        X_val_tensor = torch.FloatTensor(X_val_scaled)
        y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1))
        
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))
        
        # Create datasets
        train_dataset = UFCDataset(X_train_tensor, y_train_tensor)
        val_dataset = UFCDataset(X_val_tensor, y_val_tensor)
        test_dataset = UFCDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        logger.info("DataLoaders prepared successfully")
        return train_loader, val_loader, test_loader, scaler, X_train.columns.tolist()
    
    except Exception as e:
        logger.error(f"Error preparing dataloaders: {e}")
        raise

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
    
    try:
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
    
    except Exception as e:
        logger.error(f"Error during training: {e}")
        raise

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
    
    try:
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
        
        # Create confusion matrix
        cm = confusion_matrix(test_targets, test_preds)
        
        # Print metrics
        logger.info(f"Test Metrics:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"Recall: {metrics['recall']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info(f"ROC AUC: {metrics['auc']:.4f}")
        
        # Save confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        from config import MODEL_PATH
        cm_path = os.path.join(os.path.dirname(MODEL_PATH), 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        logger.info(f"Saved confusion matrix to {cm_path}")
        
        return metrics
    
    except Exception as e:
        logger.error(f"Error evaluating model: {e}")
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'auc': 0.0
        }

def compute_feature_importance(model, X_train, scaler):
    """
    Compute feature importance using the permutation method
    
    Args:
        model (nn.Module): Trained model
        X_train (pd.DataFrame): Training features
        scaler (StandardScaler): Fitted scaler
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    logger.info("Computing feature importance")
    
    try:
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
            
            # Compute feature importance (mean absolute difference)
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            feature_importance.append((feature_name, importance))
        
        # Convert to DataFrame and sort by importance
        importance_df = pd.DataFrame(feature_importance, columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        # Plot feature importance
        plt.figure(figsize=(10, 8))
        plt.barh(importance_df['feature'][:15], importance_df['importance'][:15])
        plt.xlabel('Importance')
        plt.title('Top 15 Features by Importance')
        plt.gca().invert_yaxis()  # Display highest importance at the top
        
        from config import MODEL_PATH
        importance_path = os.path.join(os.path.dirname(MODEL_PATH), 'feature_importance.png')
        plt.savefig(importance_path)
        plt.close()
        logger.info(f"Saved feature importance plot to {importance_path}")
        
        # Save to CSV
        csv_path = os.path.join(os.path.dirname(MODEL_PATH), 'feature_importance.csv')
        importance_df.to_csv(csv_path, index=False)
        logger.info(f"Saved feature importance to {csv_path}")
        
        return importance_df
    
    except Exception as e:
        logger.error(f"Error computing feature importance: {e}")
        return pd.DataFrame(columns=['feature', 'importance'])

def predict_fight(model, fighter1_data, fighter2_data, scaler, feature_columns):
    """
    Predict the outcome of a UFC fight
    
    Args:
        model (nn.Module): Trained model
        fighter1_data (dict): Fighter 1 statistics
        fighter2_data (dict): Fighter 2 statistics
        scaler (StandardScaler): Fitted scaler
        feature_columns (list): List of feature column names
        
    Returns:
        dict: Prediction results
    """
    logger.info("Predicting fight outcome")
    
    try:
        # Determine device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to the device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        # Prepare features in the correct format
        features = {}
        
        # Add fighter1 features
        for key, value in fighter1_data.items():
            if not key.startswith('fighter1_'):
                features[f'fighter1_{key}'] = value
            else:
                features[key] = value
        
        # Add fighter2 features
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
            features['fighter1_total_fights'] = features['fighter1_wins'] + features['fighter1_losses']
            features['fighter2_total_fights'] = features['fighter2_wins'] + features['fighter2_losses']
            features['experience_advantage'] = features['fighter1_total_fights'] - features['fighter2_total_fights']
            
            # Win percentage advantage
            if features['fighter1_total_fights'] > 0 and features['fighter2_total_fights'] > 0:
                features['fighter1_win_pct'] = features['fighter1_wins'] / features['fighter1_total_fights']
                features['fighter2_win_pct'] = features['fighter2_wins'] / features['fighter2_total_fights']
                features['win_pct_advantage'] = features['fighter1_win_pct'] - features['fighter2_win_pct']
        
        # Striking advantages
        if 'fighter1_sig_strikes_per_min' in features and 'fighter2_sig_strikes_per_min' in features:
            features['striking_volume_advantage'] = features['fighter1_sig_strikes_per_min'] - features['fighter2_sig_strikes_per_min']
        
        if 'fighter1_sig_strike_accuracy' in features and 'fighter2_sig_strike_accuracy' in features:
            features['striking_accuracy_advantage'] = features['fighter1_sig_strike_accuracy'] - features['fighter2_sig_strike_accuracy']
        
        # Grappling advantages
        if 'fighter1_takedown_avg' in features and 'fighter2_takedown_avg' in features:
            features['takedown_advantage'] = features['fighter1_takedown_avg'] - features['fighter2_takedown_avg']
        
        if 'fighter1_takedown_defense' in features and 'fighter2_takedown_defense' in features:
            features['takedown_defense_advantage'] = features['fighter1_takedown_defense'] - features['fighter2_takedown_defense']
        
        if 'fighter1_sub_avg' in features and 'fighter2_sub_avg' in features:
            features['submission_advantage'] = features['fighter1_sub_avg'] - features['fighter2_sub_avg']
        
        # Create a dataframe with the expected columns
        input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
        
        # Fill in the features we have
        for col in feature_columns:
            if col in features:
                input_df[col] = features[col]
        
        # Scale the features
        scaled_features = scaler.transform(input_df)
        
        # Convert to tensor and move to device
        features_tensor = torch.FloatTensor(scaled_features).to(device)
        
        # Predict
        with torch.no_grad():
            win_probability = model(features_tensor).item()
        
        # Create result dictionary
        result = {
            'probability_red_wins': float(win_probability),
            'probability_blue_wins': float(1 - win_probability),
            'predicted_winner': 'Red' if win_probability > 0.5 else 'Blue',
            'confidence_level': 'High' if abs(win_probability - 0.5) > 0.25 else 
                             'Medium' if abs(win_probability - 0.5) > 0.1 else 'Low'
        }
        
        logger.info(f"Prediction result: {result}")
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return {
            'probability_red_wins': 0.5,
            'probability_blue_wins': 0.5,
            'predicted_winner': 'Unknown',
            'confidence_level': 'Low',
            'error': str(e)
        }

def test_prediction(red_fighter_data=None, blue_fighter_data=None, model_path=None, verbose=True):
    """
    Test UFC fight prediction with given fighter data or sample data
    
    Args:
        red_fighter_data (dict): Red corner fighter stats
        blue_fighter_data (dict): Blue corner fighter stats
        model_path (str): Path to the model file
        verbose (bool): Whether to print detailed output
        
    Returns:
        dict: Prediction results
    """
    # Use sample data if not provided
    if red_fighter_data is None:
        red_fighter_data = {
            'wins': 10, 'losses': 2,
            'height': 180, 'reach': 185, 'weight': 155, 'age': 28,
            'sig_strikes_per_min': 4.2, 'sig_strike_accuracy': 0.48,
            'sig_strikes_absorbed_per_min': 3.1, 'sig_strike_defense': 0.62,
            'takedown_avg': 1.5, 'takedown_accuracy': 0.35, 'takedown_defense': 0.75,
            'sub_avg': 0.8,
            'win_by_KO_TKO': 5, 'win_by_SUB': 3, 'win_by_DEC': 2
        }
    
    if blue_fighter_data is None:
        blue_fighter_data = {
            'wins': 8, 'losses': 4,
            'height': 178, 'reach': 183, 'weight': 155, 'age': 30,
            'sig_strikes_per_min': 3.8, 'sig_strike_accuracy': 0.52,
            'sig_strikes_absorbed_per_min': 2.9, 'sig_strike_defense': 0.58,
            'takedown_avg': 2.2, 'takedown_accuracy': 0.40, 'takedown_defense': 0.70,
            'sub_avg': 0.5,
            'win_by_KO_TKO': 3, 'win_by_SUB': 2, 'win_by_DEC': 3
        }
    
    # Load config
    from config import MODEL_PATH
    
    # Use provided model path or default
    model_path_to_use = model_path or MODEL_PATH
    model_dir = os.path.dirname(model_path_to_use)
    
    try:
        # Load input size
        input_size_path = os.path.join(model_dir, 'input_size.pkl')
        if os.path.exists(input_size_path):
            input_size = joblib.load(input_size_path)
        else:
            # Default to a reasonable value if not found
            logger.warning("Input size file not found, using default value")
            input_size = 45
        
        # Initialize model
        model = UFCPredictor(input_size)
        
        # Load model weights
        model.load_state_dict(torch.load(model_path_to_use, map_location=torch.device('cpu')))
        model.eval()
        
        # Load scaler
        scaler_path = os.path.join(model_dir, 'scaler.save')
        scaler = joblib.load(scaler_path)
        
        # Load feature columns
        feature_columns_path = os.path.join(model_dir, 'feature_columns.pkl')
        if os.path.exists(feature_columns_path):
            feature_columns = joblib.load(feature_columns_path)
        else:
            logger.warning("Feature columns file not found, prediction may be inaccurate")
            # Generate a default list based on the provided fighter data
            feature_columns = []
            for key in red_fighter_data:
                feature_columns.append(f"fighter1_{key}")
            for key in blue_fighter_data:
                feature_columns.append(f"fighter2_{key}")
            # Add typical advantage columns
            feature_columns.extend(['height_advantage', 'reach_advantage', 'weight_advantage',
                                 'experience_advantage', 'win_pct_advantage',
                                 'striking_volume_advantage', 'striking_accuracy_advantage',
                                 'takedown_advantage', 'takedown_defense_advantage',
                                 'submission_advantage'])
        
        # Make prediction
        result = predict_fight(model, red_fighter_data, blue_fighter_data, scaler, feature_columns)
        
        # Display results if verbose
        if verbose:
            print("\n===== UFC FIGHT PREDICTION =====")
            print(f"Red fighter win probability: {result['probability_red_wins']:.4f} ({result['probability_red_wins']*100:.1f}%)")
            print(f"Blue fighter win probability: {result['probability_blue_wins']:.4f} ({result['probability_blue_wins']*100:.1f}%)")
            print(f"Predicted winner: {result['predicted_winner']}")
            print(f"Confidence: {result['confidence_level']}")
            print("===============================")
            
            # Generate a visualization
            plt.figure(figsize=(10, 5))
            labels = ['Red Fighter', 'Blue Fighter']
            probabilities = [result['probability_red_wins'], result['probability_blue_wins']]
            colors = ['#FF6B6B', '#4D96FF']
            
            plt.bar(labels, probabilities, color=colors)
            plt.ylim(0, 1)
            plt.title('UFC Fight Prediction')
            plt.ylabel('Win Probability')
            
            # Add percentage labels
            for i, prob in enumerate(probabilities):
                plt.text(i, prob + 0.02, f"{prob*100:.1f}%", ha='center')
            
            # Save the visualization
            viz_path = os.path.join(model_dir, 'prediction_visualization.png')
            plt.savefig(viz_path)
            plt.close()
            print(f"Visualization saved to {viz_path}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in test prediction: {e}")
        return {
            'probability_red_wins': 0.5,
            'probability_blue_wins': 0.5,
            'predicted_winner': 'Unknown',
            'confidence_level': 'Low',
            'error': str(e)
        }

def compare_fighters(fighter1, fighter2, model_path=None):
    """
    Compare two fighters and show detailed stat comparisons alongside prediction
    
    Args:
        fighter1 (dict): First fighter statistics
        fighter2 (dict): Second fighter statistics
        model_path (str): Path to the model file
        
    Returns:
        dict: Comparison and prediction results
    """
    logger.info("Comparing fighters")
    
    try:
        # Create comparison table
        comparison_data = []
        
        # Common stats to compare
        stats_to_compare = [
            'wins', 'losses', 'height', 'reach', 'weight', 'age',
            'sig_strikes_per_min', 'sig_strike_accuracy',
            'sig_strikes_absorbed_per_min', 'sig_strike_defense',
            'takedown_avg', 'takedown_accuracy', 'takedown_defense',
            'sub_avg', 'win_by_KO_TKO', 'win_by_SUB', 'win_by_DEC'
        ]
        
        # Build comparison table
        for stat in stats_to_compare:
            if stat in fighter1 and stat in fighter2:
                val1 = fighter1[stat]
                val2 = fighter2[stat]
                
                # Determine advantage
                if stat in ['losses', 'sig_strikes_absorbed_per_min']:
                    # Lower is better
                    advantage = 'Fighter 1' if val1 < val2 else 'Fighter 2' if val2 < val1 else 'Even'
                else:
                    # Higher is better
                    advantage = 'Fighter 1' if val1 > val2 else 'Fighter 2' if val2 > val1 else 'Even'
                
                # Format values for percentages
                if stat in ['sig_strike_accuracy', 'takedown_accuracy', 'takedown_defense', 'sig_strike_defense']:
                    val1_display = f"{val1*100:.1f}%" if isinstance(val1, float) else val1
                    val2_display = f"{val2*100:.1f}%" if isinstance(val2, float) else val2
                else:
                    val1_display = val1
                    val2_display = val2
                
                # Add to comparison
                comparison_data.append({
                    'Statistic': stat.replace('_', ' ').title(),
                    'Fighter 1': val1_display,
                    'Fighter 2': val2_display,
                    'Advantage': advantage
                })
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data)
        
        # Make prediction
        result = test_prediction(fighter1, fighter2, model_path, verbose=False)
        
        # Print comparison
        print("\n===== FIGHTER COMPARISON =====")
        print(comparison_df.to_string(index=False))
        print("\n===== PREDICTION =====")
        print(f"Fighter 1 win probability: {result['probability_red_wins']:.4f} ({result['probability_red_wins']*100:.1f}%)")
        print(f"Fighter 2 win probability: {result['probability_blue_wins']:.4f} ({result['probability_blue_wins']*100:.1f}%)")
        print(f"Predicted winner: {'Fighter 1' if result['predicted_winner'] == 'Red' else 'Fighter 2'}")
        print(f"Confidence: {result['confidence_level']}")
        print("===============================")
        
        # Save comparison to CSV
        from config import MODEL_PATH
        csv_path = os.path.join(os.path.dirname(MODEL_PATH), 'fighter_comparison.csv')
        comparison_df.to_csv(csv_path, index=False)
        print(f"Comparison saved to {csv_path}")
        
        return {
            'comparison': comparison_df.to_dict(orient='records'),
            'prediction': result
        }
    
    except Exception as e:
        logger.error(f"Error comparing fighters: {e}")
        return {
            'error': str(e)
        }

def batch_test_predictions(test_cases=None, model_path=None):
    """
    Run batch predictions on multiple test cases to evaluate model performance
    
    Args:
        test_cases (list): List of test case dictionaries with 'red_fighter', 'blue_fighter', and optional 'actual_winner'
        model_path (str): Path to the model file
        
    Returns:
        dict: Summary of prediction results and accuracy
    """
    logger.info("Running batch test predictions")
    
    # Use default test cases if none provided
    if test_cases is None:
        test_cases = [
            {
                'description': 'Experienced fighter vs. Newcomer',
                'red_fighter': {
                    'wins': 15, 'losses': 2, 'height': 180, 'reach': 186, 
                    'sig_strikes_per_min': 4.5, 'sig_strike_accuracy': 0.55,
                    'takedown_avg': 2.3, 'takedown_defense': 0.80
                },
                'blue_fighter': {
                    'wins': 3, 'losses': 0, 'height': 178, 'reach': 182,
                    'sig_strikes_per_min': 3.8, 'sig_strike_accuracy': 0.48,
                    'takedown_avg': 1.1, 'takedown_defense': 0.65
                },
                'actual_winner': 'Red'
            },
            {
                'description': 'Striker vs. Grappler',
                'red_fighter': {
                    'wins': 12, 'losses': 4, 'height': 182, 'reach': 188,
                    'sig_strikes_per_min': 5.2, 'sig_strike_accuracy': 0.58,
                    'takedown_avg': 0.5, 'takedown_defense': 0.75,
                    'win_by_KO_TKO': 8, 'win_by_SUB': 1
                },
                'blue_fighter': {
                    'wins': 14, 'losses': 3, 'height': 179, 'reach': 183,
                    'sig_strikes_per_min': 2.4, 'sig_strike_accuracy': 0.42,
                    'takedown_avg': 4.8, 'takedown_defense': 0.60,
                    'win_by_KO_TKO': 2, 'win_by_SUB': 10
                },
                'actual_winner': 'Blue'
            },
            {
                'description': 'Evenly matched fighters',
                'red_fighter': {
                    'wins': 8, 'losses': 4, 'height': 180, 'reach': 184,
                    'sig_strikes_per_min': 3.9, 'sig_strike_accuracy': 0.49,
                    'takedown_avg': 2.1, 'takedown_defense': 0.70
                },
                'blue_fighter': {
                    'wins': 9, 'losses': 5, 'height': 181, 'reach': 185,
                    'sig_strikes_per_min': 4.1, 'sig_strike_accuracy': 0.51,
                    'takedown_avg': 1.9, 'takedown_defense': 0.68
                },
                'actual_winner': 'Blue'
            }
        ]
    
    results = []
    
    # Run predictions for each test case
    for i, case in enumerate(test_cases):
        try:
            print(f"\nTest case {i+1}/{len(test_cases)}: {case.get('description', 'Unnamed test')}")
            
            # Make prediction
            prediction = test_prediction(
                case['red_fighter'],
                case['blue_fighter'],
                model_path,
                verbose=False
            )
            
            # Create result record
            result = {
                'test_id': i+1,
                'description': case.get('description', 'Unnamed test'),
                'predicted_winner': prediction['predicted_winner'],
                'red_win_probability': prediction['probability_red_wins'],
                'blue_win_probability': prediction['probability_blue_wins'],
                'confidence_level': prediction['confidence_level']
            }
            
            # Add actual winner if provided
            if 'actual_winner' in case:
                result['actual_winner'] = case['actual_winner']
                result['correct'] = (case['actual_winner'] == prediction['predicted_winner'])
                
                print(f"Prediction: {prediction['predicted_winner']} (Confidence: {prediction['confidence_level']})")
                print(f"Actual: {case['actual_winner']} - {'CORRECT' if result['correct'] else 'INCORRECT'}")
            else:
                print(f"Prediction: {prediction['predicted_winner']} (Confidence: {prediction['confidence_level']})")
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Error in test case {i+1}: {e}")
            results.append({
                'test_id': i+1,
                'description': case.get('description', 'Unnamed test'),
                'error': str(e)
            })
    
    # Calculate summary statistics
    summary = {
        'total_tests': len(results),
        'successful_tests': sum(1 for r in results if 'error' not in r)
    }
    
    # Calculate accuracy metrics if actual winners were provided
    correct_predictions = [r for r in results if 'correct' in r and r['correct']]
    if correct_predictions:
        summary['correct_predictions'] = len(correct_predictions)
        summary['accuracy'] = len(correct_predictions) / sum(1 for r in results if 'correct' in r)
        
        # Analyze by confidence level
        high_conf = [r for r in results if 'confidence_level' in r and r['confidence_level'] == 'High' and 'correct' in r]
        med_conf = [r for r in results if 'confidence_level' in r and r['confidence_level'] == 'Medium' and 'correct' in r]
        low_conf = [r for r in results if 'confidence_level' in r and r['confidence_level'] == 'Low' and 'correct' in r]
        
        if high_conf:
            summary['high_confidence_accuracy'] = sum(1 for r in high_conf if r['correct']) / len(high_conf)
            summary['high_confidence_count'] = len(high_conf)
            
        if med_conf:
            summary['medium_confidence_accuracy'] = sum(1 for r in med_conf if r['correct']) / len(med_conf)
            summary['medium_confidence_count'] = len(med_conf)
            
        if low_conf:
            summary['low_confidence_accuracy'] = sum(1 for r in low_conf if r['correct']) / len(low_conf)
            summary['low_confidence_count'] = len(low_conf)
    
    # Print summary
    print("\n===== BATCH TEST SUMMARY =====")
    print(f"Total tests: {summary['total_tests']}")
    print(f"Successful tests: {summary['successful_tests']}")
    
    if 'accuracy' in summary:
        print(f"Overall accuracy: {summary['accuracy']:.2f} ({summary['correct_predictions']}/{sum(1 for r in results if 'correct' in r)})")
        
        if 'high_confidence_accuracy' in summary:
            print(f"High confidence accuracy: {summary['high_confidence_accuracy']:.2f} ({summary['high_confidence_count']} predictions)")
            
        if 'medium_confidence_accuracy' in summary:
            print(f"Medium confidence accuracy: {summary['medium_confidence_accuracy']:.2f} ({summary['medium_confidence_count']} predictions)")
            
        if 'low_confidence_accuracy' in summary:
            print(f"Low confidence accuracy: {summary['low_confidence_accuracy']:.2f} ({summary['low_confidence_count']} predictions)")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    
    from config import MODEL_PATH
    csv_path = os.path.join(os.path.dirname(MODEL_PATH), 'batch_test_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Test results saved to {csv_path}")
    
    return {
        'summary': summary,
        'results': results
    }

def position_bias_test(model, scaler, feature_columns, model_path=None):
    """
    Test the model for position bias (red/blue corner bias)
    
    Args:
        model (nn.Module): Trained model
        scaler (StandardScaler): Fitted scaler
        feature_columns (list): List of feature column names
        model_path (str): Path to the model file
        
    Returns:
        dict: Position bias test results
    """
    logger.info("Testing for position bias")
    
    try:
        # Create test cases with identical fighters in different positions
        test_pairs = []
        for i in range(5):
            # Create a base fighter with slightly varied stats
            base = {
                'wins': 10 + i,
                'losses': 3 + (i % 3),
                'height': 180 - (i % 5),
                'reach': 185 - (i % 7),
                'weight': 155 + (i * 10) % 30,
                'age': 28 + (i % 10),
                'sig_strikes_per_min': 4.0 + (i * 0.2),
                'sig_strike_accuracy': 0.50 + (i * 0.01),
                'takedown_avg': 2.0 + (i * 0.3),
                'takedown_defense': 0.70 + (i * 0.02)
            }
            
            # Create a slightly different fighter
            variation = {
                'wins': 9 + i,
                'losses': 4 + (i % 3),
                'height': 178 + (i % 5),
                'reach': 183 + (i % 7),
                'weight': 155 - (i * 5) % 20,
                'age': 30 - (i % 8),
                'sig_strikes_per_min': 3.8 + (i * 0.15),
                'sig_strike_accuracy': 0.48 + (i * 0.015),
                'takedown_avg': 2.2 - (i * 0.25),
                'takedown_defense': 0.65 + (i * 0.03)
            }
            
            test_pairs.append((base.copy(), variation.copy()))
        
        # Test each pair in both positions
        bias_results = []
        
        for fighter1, fighter2 in test_pairs:
            # Normal prediction (fighter1 = red, fighter2 = blue)
            normal_pred = predict_fight(model, fighter1, fighter2, scaler, feature_columns)
            
            # Swapped prediction (fighter1 = blue, fighter2 = red)
            swapped_pred = predict_fight(model, fighter2, fighter1, scaler, feature_columns)
            
            # Calculate position bias
            normal_prob = normal_pred['probability_red_wins']
            swapped_prob = 1 - swapped_pred['probability_red_wins']  # Invert probability for comparison
            
            bias = abs(normal_prob - swapped_prob)
            
            bias_results.append({
                'normal_prob': normal_prob,
                'swapped_prob': swapped_prob,
                'bias': bias
            })
        
        # Calculate average bias
        avg_bias = sum(result['bias'] for result in bias_results) / len(bias_results)
        
        # Print results
        print("\n===== POSITION BIAS TEST =====")
        print(f"Average position bias: {avg_bias:.4f}")
        for i, result in enumerate(bias_results):
            print(f"Test pair {i+1}: Normal={result['normal_prob']:.4f}, Swapped={result['swapped_prob']:.4f}, Bias={result['bias']:.4f}")
        
        bias_level = "Low" if avg_bias < 0.05 else "Medium" if avg_bias < 0.1 else "High"
        print(f"Position bias level: {bias_level}")
        
        return {
            'average_bias': avg_bias,
            'bias_level': bias_level,
            'bias_results': bias_results
        }
        
    except Exception as e:
        logger.error(f"Error in position bias test: {e}")
        return {
            'error': str(e)
        }

def main():
    """
    Main function to train and test the UFC fight prediction model
    """
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UFC Fight Predictor")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the model')
    parser.add_argument('--predict', action='store_true', help='Make a prediction')
    parser.add_argument('--data-file', type=str, default=None, help='Path to the data file')
    parser.add_argument('--model-file', type=str, default=None, help='Path to the model file')
    
    args = parser.parse_args()
    
    # Load configuration
    from config import CSV_FILE_PATH, MODEL_PATH
    
    # Use command line args or defaults
    data_file = args.data_file or CSV_FILE_PATH
    model_file = args.model_file or MODEL_PATH
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    
    # Train the model
    if args.train:
        print("Loading and preprocessing data...")
        
        # Check if data file exists
        if not os.path.exists(data_file):
            print(f"Data file not found: {data_file}")
            print("Please provide a valid data file or run the data loader first.")
            return
        
        # Load the data
        try:
            from csv_sync import sync_csv_files
            from data_loader import preprocess_dataset
            
            # Preprocess the dataset
            print("Preprocessing dataset...")
            df = preprocess_dataset(data_file)
            
            # Apply additional preprocessing for the model
            print("Preparing data for model...")
            processed_data = preprocess_data(df)
            processed_data = calculate_advantage_features(processed_data)
            processed_data = select_key_features(processed_data)
            
            # Split the data
            X_train, X_val, X_test, y_train, y_val, y_test = train_test_val_split(processed_data)
            
            # Prepare dataloaders
            train_loader, val_loader, test_loader, scaler, feature_columns = prepare_dataloaders(
                X_train, X_val, X_test, y_train, y_val, y_test
            )
            
            # Save feature columns and input size
            joblib.dump(feature_columns, os.path.join(os.path.dirname(model_file), 'feature_columns.pkl'))
            joblib.dump(len(feature_columns), os.path.join(os.path.dirname(model_file), 'input_size.pkl'))
            joblib.dump(scaler, os.path.join(os.path.dirname(model_file), 'scaler.save'))
            
            # Create and train the model
            print("Training model...")
            model = UFCPredictor(input_size=len(feature_columns))
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=100,
                learning_rate=0.001,
                weight_decay=1e-4,
                patience=10,
                model_path=model_file
            )
            
            # Evaluate the model
            print("Evaluating model...")
            metrics = evaluate_model(model, test_loader)
            
            # Compute feature importance
            print("Computing feature importance...")
            importance_df = compute_feature_importance(model, X_train, scaler)
            
            # Test for position bias
            print("Testing for position bias...")
            bias_results = position_bias_test(model, scaler, feature_columns, model_file)
            
            print("\n===== TRAINING COMPLETE =====")
            print(f"Model saved to: {model_file}")
            print(f"Test accuracy: {metrics['accuracy']:.4f}")
            print(f"Test precision: {metrics['precision']:.4f}")
            print(f"Test recall: {metrics['recall']:.4f}")
            print(f"Test F1 score: {metrics['f1']:.4f}")
            print(f"Test AUC: {metrics['auc']:.4f}")
            
            if 'average_bias' in bias_results:
                print(f"Position bias: {bias_results['average_bias']:.4f} ({bias_results['bias_level']})")
            
            print("\nTop 10 features by importance:")
            for i, row in importance_df.head(10).iterrows():
                print(f"{i+1}. {row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
    
    # Test the model
    if args.test:
        print("Testing model...")
        
        # Check if model file exists
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            print("Please train the model first or provide a valid model file.")
            return
        
        try:
            # Run batch tests
            batch_test_predictions(model_path=model_file)
            
        except Exception as e:
            print(f"Error during testing: {e}")
            import traceback
            traceback.print_exc()
    
    # Make a prediction
    if args.predict:
        print("Making a prediction...")
        
        # Check if model file exists
        if not os.path.exists(model_file):
            print(f"Model file not found: {model_file}")
            print("Please train the model first or provide a valid model file.")
            return
        
        try:
            # Sample fighter data
            fighter1 = {
                'name': 'Fighter 1',
                'wins': 12,
                'losses': 3,
                'height': 180,
                'reach': 185,
                'weight': 155,
                'age': 28,
                'sig_strikes_per_min': 4.2,
                'sig_strike_accuracy': 0.52,
                'takedown_avg': 1.8,
                'takedown_defense': 0.75,
                'sub_avg': 0.8,
                'win_by_KO_TKO': 6,
                'win_by_SUB': 4,
                'win_by_DEC': 2
            }
            
            fighter2 = {
                'name': 'Fighter 2',
                'wins': 10,
                'losses': 5,
                'height': 178,
                'reach': 183,
                'weight': 155,
                'age': 30,
                'sig_strikes_per_min': 3.9,
                'sig_strike_accuracy': 0.48,
                'takedown_avg': 2.2,
                'takedown_defense': 0.70,
                'sub_avg': 1.2,
                'win_by_KO_TKO': 3,
                'win_by_SUB': 5,
                'win_by_DEC': 2
            }
            
            # Compare fighters and make prediction
            compare_fighters(fighter1, fighter2, model_path=model_file)
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            import traceback
            traceback.print_exc()
    
    # If no action specified, show help
    if not (args.train or args.test or args.predict):
        parser.print_help()

if __name__ == "__main__":
    main()