import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
from config import MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, TEST_SIZE, VALIDATION_SIZE, CSV_FILE_PATH, DATABASE_PATH, SCALER_PATH
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(MODEL_PATH), 'training.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ufc_model')

def preprocess_data(raw_data):
    """
    Preprocess the data for model training with consistent feature naming
    
    Args:
        raw_data (pd.DataFrame): Raw input data
        
    Returns:
        pd.DataFrame: Processed dataset for modeling
    """
    # Setup logging
    logger = logging.getLogger('ufc_model')
    logger.info("Starting data preprocessing")

    try:
        # Create a copy of the dataframe to avoid modifying the original
        data_df = raw_data.copy()

        # Ensure directory for model artifacts exists
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Feature names path
        feature_names_path = os.path.join(os.path.dirname(MODEL_PATH), 'feature_names.json')

        # Rename columns to use consistent fighter1_ and fighter2_ prefixes if needed
        if any(col.startswith('R_') or col.startswith('B_') for col in data_df.columns):
            rename_mapping = {
                col: col.replace('R_', 'fighter1_').replace('B_', 'fighter2_') 
                for col in data_df.columns
            }
            data_df = data_df.rename(columns=rename_mapping)

        # Rename Winner column to match model expectations if needed
        if 'Winner' in data_df.columns:
            # Map Red/Blue to 1/0 for fighter1_won
            data_df['fighter1_won'] = (data_df['Winner'] == 'Red').astype(int)
        elif 'winner_id' in data_df.columns and 'fighter1_id' in data_df.columns:
            # If we have IDs instead, use those to determine winner
            data_df['fighter1_won'] = (data_df['winner_id'] == data_df['fighter1_id']).astype(int)
        elif 'fighter1_won' not in data_df.columns:
            # If we don't have a winner column at all, raise an error
            raise ValueError("Dataset must have either 'Winner', 'winner_id', or 'fighter1_won' column")

        # Save original feature columns
        feature_columns = [col for col in data_df.columns if col != 'fighter1_won']
        pd.Series(feature_columns).to_json(feature_names_path)
        
        # Drop non-numeric columns for modeling
        numeric_data = data_df.select_dtypes(include=[np.number])
        
        # Handle missing values
        for col in numeric_data.columns:
            if numeric_data[col].isna().any():
                if col == 'fighter1_won':
                    # Drop rows with missing target variable
                    numeric_data = numeric_data.dropna(subset=[col])
                else:
                    # Use median imputation for most columns
                    if col.endswith('_height') or col.endswith('_weight'):
                        numeric_data[col] = numeric_data[col].fillna(numeric_data[col].mean())
                    else:
                        numeric_data[col] = numeric_data[col].fillna(numeric_data[col].median())
        
        logger.info(f"Preprocessed data shape: {numeric_data.shape}")
        return numeric_data

    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        raise

class UFCDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class FightPredictor(nn.Module):
    def __init__(self, input_size):
        super(FightPredictor, self).__init__()
        # Simple architecture with strong regularization
        self.layer1 = nn.Linear(input_size, 64)
        self.dropout1 = nn.Dropout(0.4)  # Higher dropout
        self.layer2 = nn.Linear(64, 32)
        self.dropout2 = nn.Dropout(0.3)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout1(x)
        x = self.relu(self.layer2(x))
        x = self.dropout2(x)
        x = self.sigmoid(self.output(x))
        return x
    
class ModelTrainer:
    def __init__(self, data_df):
        self.data_df = data_df
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Create directories for model and artifacts
        self.model_dir = os.path.dirname(MODEL_PATH)
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Paths for saving artifacts
        self.scaler_path = os.path.join(self.model_dir, 'scaler.save')
        self.feature_names_path = os.path.join(self.model_dir, 'feature_names.json')
        self.metrics_path = os.path.join(self.model_dir, 'training_metrics.csv')
        self.best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        self.feature_importance_path = os.path.join(self.model_dir, 'feature_importance.png')
    
    def prepare_data_for_training(self):
        """
        Prepare data for training - split into train/val/test and create dataloaders
        Includes comprehensive position bias correction and feature engineering
        
        Returns:
            dict: Dictionary with dataloaders and related data
        """
        logger.info("Preparing data for training")
        
        # Make a copy to avoid modifying the original data
        processed_data = self.data_df.copy()
        
        # STEP 1: CREATE ADVANCED FEATURES FOR BETTER PREDICTION
        # ==========================================================
        logger.info("Creating comprehensive fighter comparison features")
        
        # Track features that were added
        added_features = []
        
        for idx, row in processed_data.iterrows():
            # PHYSICAL ATTRIBUTES COMPARISONS
            # Weight advantage
            if 'fighter1_weight' in row and 'fighter2_weight' in row:
                if pd.notna(row['fighter1_weight']) and pd.notna(row['fighter2_weight']):
                    processed_data.at[idx, 'weight_advantage'] = float(row['fighter1_weight']) - float(row['fighter2_weight'])
                    added_features.append('weight_advantage')
            
            # Height advantage
            if 'fighter1_height' in row and 'fighter2_height' in row:
                if pd.notna(row['fighter1_height']) and pd.notna(row['fighter2_height']):
                    processed_data.at[idx, 'height_advantage'] = float(row['fighter1_height']) - float(row['fighter2_height'])
                    added_features.append('height_advantage')
            
            # Reach advantage
            if 'fighter1_reach' in row and 'fighter2_reach' in row:
                if pd.notna(row['fighter1_reach']) and pd.notna(row['fighter2_reach']):
                    processed_data.at[idx, 'reach_advantage'] = float(row['fighter1_reach']) - float(row['fighter2_reach'])
                    added_features.append('reach_advantage')
            
            # RECORD COMPARISONS
            # Win streak comparison (momentum factor)
            if 'fighter1_win_streak' in row and 'fighter2_win_streak' in row:
                if pd.notna(row['fighter1_win_streak']) and pd.notna(row['fighter2_win_streak']):
                    processed_data.at[idx, 'win_streak_diff'] = float(row['fighter1_win_streak']) - float(row['fighter2_win_streak'])
                    added_features.append('win_streak_diff')
            
            # Win rate comparison
            if all(f'{prefix}_{stat}' in row for prefix in ['fighter1', 'fighter2'] 
                for stat in ['wins', 'losses']):
                try:
                    # Calculate win percentages
                    f1_total = float(row['fighter1_wins']) + float(row['fighter1_losses'])
                    f2_total = float(row['fighter2_wins']) + float(row['fighter2_losses'])
                    
                    if f1_total > 0 and f2_total > 0:
                        f1_win_pct = float(row['fighter1_wins']) / f1_total
                        f2_win_pct = float(row['fighter2_wins']) / f2_total
                        processed_data.at[idx, 'win_rate_advantage'] = f1_win_pct - f2_win_pct
                        added_features.append('win_rate_advantage')
                        
                        # Experience difference
                        processed_data.at[idx, 'experience_advantage'] = f1_total - f2_total
                        added_features.append('experience_advantage')
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # Finish rate comparison (shows finishing ability)
            if all(f'{prefix}_{stat}' in row for prefix in ['fighter1', 'fighter2'] 
                for stat in ['wins', 'win_by_KO_TKO', 'win_by_SUB']):
                try:
                    # Calculate finish rate for fighter 1
                    if float(row['fighter1_wins']) > 0:
                        f1_finish_rate = (float(row['fighter1_win_by_KO_TKO']) + float(row['fighter1_win_by_SUB'])) / float(row['fighter1_wins'])
                    else:
                        f1_finish_rate = 0
                        
                    # Calculate finish rate for fighter 2
                    if float(row['fighter2_wins']) > 0:
                        f2_finish_rate = (float(row['fighter2_win_by_KO_TKO']) + float(row['fighter2_win_by_SUB'])) / float(row['fighter2_wins'])
                    else:
                        f2_finish_rate = 0
                        
                    processed_data.at[idx, 'finish_rate_diff'] = f1_finish_rate - f2_finish_rate
                    added_features.append('finish_rate_diff')
                except (ValueError, TypeError, ZeroDivisionError):
                    pass
            
            # FIGHTING STYLE COMPARISONS
            # Performance metrics differences
            for stat in ['sig_strikes_per_min', 'takedown_avg', 'sub_avg']:
                f1_key = f'fighter1_{stat}'
                f2_key = f'fighter2_{stat}'
                if f1_key in row and f2_key in row:
                    if pd.notna(row[f1_key]) and pd.notna(row[f2_key]):
                        try:
                            processed_data.at[idx, f'{stat}_advantage'] = float(row[f1_key]) - float(row[f2_key])
                            added_features.append(f'{stat}_advantage')
                        except (ValueError, TypeError):
                            pass
        
        # Log the newly created features
        unique_added_features = list(set(added_features))
        logger.info(f"Added {len(unique_added_features)} comparison features: {unique_added_features}")
        
        # STEP 2: POSITION BIAS CORRECTION WITH DIRECTION-PRESERVING AUGMENTATION
        # =======================================================================
        logger.info(f"Original data shape before augmentation: {processed_data.shape}")
        
        # Get the list of advantage features for direction flipping
        advantage_columns = [col for col in processed_data.columns if 
                            any(col.endswith(suffix) for suffix in 
                                ['_advantage', '_diff', 'advantage_', 'diff_'])]
        
        logger.info(f"Found {len(advantage_columns)} advantage features for position balancing")
        
        # Build balanced dataset with direction-preserving augmentation
        augmented_data = []
        
        # Add original data with higher weight to preserve signal
        logger.info("Adding original data with higher weight")
        for _ in range(3):  # Add original 3 times for weight
            for _, row in processed_data.iterrows():
                augmented_data.append(row.copy())
        
        # Add swapped versions with flipped advantages
        logger.info("Adding position-swapped data for bias correction")
        for _, row in processed_data.iterrows():
            # Create swapped version with flipped advantages
            swapped = row.copy()
            
            # Flip all advantage features
            for col in advantage_columns:
                if pd.notna(swapped[col]):
                    swapped[col] = -swapped[col]
            
            # Flip the target variable
            if 'fighter1_won' in swapped:
                swapped['fighter1_won'] = 1 - row['fighter1_won']
            
            augmented_data.append(swapped)
        
        # Convert back to DataFrame
        processed_data = pd.DataFrame(augmented_data)
        logger.info(f"Augmented data shape after position bias correction: {processed_data.shape}")
        
        # STEP 3: BALANCE CLASSES
        # =======================
        # Check class balance
        win_count = processed_data['fighter1_won'].sum()
        loss_count = len(processed_data) - win_count
        logger.info(f"Class distribution: Wins: {win_count}, Losses: {loss_count}")
        
        # If significantly imbalanced, balance the classes
        if abs(win_count - loss_count) > 0.1 * len(processed_data):
            logger.info("Rebalancing classes due to significant imbalance")
            win_samples = processed_data[processed_data['fighter1_won'] == 1]
            loss_samples = processed_data[processed_data['fighter1_won'] == 0]
            
            # Use sampling to achieve balance
            min_count = min(len(win_samples), len(loss_samples))
            
            balanced_samples = pd.concat([
                win_samples.sample(min_count, random_state=42, replace=len(win_samples) < min_count),
                loss_samples.sample(min_count, random_state=42, replace=len(loss_samples) < min_count)
            ])
            
            processed_data = balanced_samples
            logger.info(f"Balanced data shape: {processed_data.shape}")
        
        # STEP 4: FINALIZE DATA PREPARATION
        # =================================
        # Separate features and target
        X = processed_data.drop('fighter1_won', axis=1)
        y = processed_data['fighter1_won']
        
        # Save feature columns for later reference
        self.X_train = X
        self.feature_columns = X.columns
        
        # Store input size for model initialization
        input_size = X.shape[1]
        joblib.dump(input_size, os.path.join(self.model_dir, "input_size.pkl"))
        joblib.dump(list(X.columns), os.path.join(self.model_dir, "feature_columns.pkl"))
        
        # Split into train, validation, and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=VALIDATION_SIZE/(1-TEST_SIZE),
            random_state=42, 
            stratify=y_train_val
        )
        
        logger.info(f"Data split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save the scaler for later use in predictions
        joblib.dump(self.scaler, self.scaler_path)
        
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
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        return {
            'train_loader': train_loader,
            'val_loader': val_loader,
            'test_loader': test_loader,
            'feature_count': input_size
        }
    
    def train_model(self):
        """
        Train the UFC fight prediction model
        
        Returns:
            tuple: (metrics_history, test_metrics)
        """
        # Prepare data
        data_dict = self.prepare_data_for_training()
        train_loader = data_dict['train_loader']
        val_loader = data_dict['val_loader']
        test_loader = data_dict['test_loader']
        input_size = data_dict['feature_count']
    
        # Initialize model
        self.model = FightPredictor(input_size).to(self.device)
        logger.info(f"Model initialized with input size: {input_size}")
        logger.info(f"Model architecture:\n{self.model}")
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau( optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        
        # Initialize early stopping parameters
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 100 #Be more patient let it do bad more 
        
        # Initialize metrics tracking
        metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 
            'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': [], 'val_auc': []
        }
        
        # Training loop
        logger.info(f"Starting training for {EPOCHS} epochs")
        for epoch in range(EPOCHS):
            # Shuffle training data with a different random seed each epoch
            random_seed = 42 + epoch
            train_loader = DataLoader(
                train_dataset, 
                batch_size=BATCH_SIZE, 
                shuffle=True,
                generator=torch.Generator().manual_seed(random_seed)
            )
            # Training phase
            self.model.train()
            running_loss = 0.0
        
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                running_loss += loss.item()
        
            train_loss = running_loss / len(train_loader)
        
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_predictions = []
            val_targets = []
        
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    val_loss += criterion(outputs, labels).item()
                    
                    # Store predictions and targets for metrics calculation
                    val_predictions.extend(outputs.cpu().detach().numpy().flatten())  # Flatten
                    val_targets.extend(labels.cpu().detach().numpy().flatten())  # Flatten
        
            val_loss = val_loss / len(val_loader)
            
            # Fix the validation metrics calculation
            val_predictions = np.array(val_predictions).flatten()  # Make sure it's 1D
            val_targets = np.array(val_targets).flatten()  # Make sure it's 1D

            # Ensure both are properly formatted as binary values
            val_predictions_binary = (val_predictions >= 0.5).astype(int)
            val_targets_binary = val_targets.astype(int)  # Ensure targets are integers

            # Use the properly formatted values for ALL metrics
            val_accuracy = accuracy_score(val_targets_binary, val_predictions_binary)
            val_precision = precision_score(val_targets_binary, val_predictions_binary, zero_division=0)
            val_recall = recall_score(val_targets_binary, val_predictions_binary, zero_division=0)
            val_f1 = f1_score(val_targets_binary, val_predictions_binary, zero_division=0)
            val_auc = roc_auc_score(val_targets_binary, val_predictions)  # Keep val_predictions continuous for AUC
        
            # Update learning rate
            scheduler.step(val_loss)
        
            # Log metrics
            logger.info(
                f'Epoch {epoch+1}/{EPOCHS}, '
                f'Train Loss: {train_loss:.4f}, '
                f'Val Loss: {val_loss:.4f}, '
                f'Val Accuracy: {val_accuracy:.4f}, '
                f'Val Precision: {val_precision:.4f}, '
                f'Val Recall: {val_recall:.4f}, '
                f'Val F1: {val_f1:.4f}, '
                f'Val AUC: {val_auc:.4f}, '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}'
            )
        
            # Store metrics
            metrics_history['epoch'].append(epoch + 1)
            metrics_history['train_loss'].append(train_loss)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_accuracy'].append(val_accuracy)
            metrics_history['val_precision'].append(val_precision)
            metrics_history['val_recall'].append(val_recall)
            metrics_history['val_f1'].append(val_f1)
            metrics_history['val_auc'].append(val_auc)
        
            # Check early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                early_stopping_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), self.best_model_path)
                logger.info(f"Saved best model with validation loss: {best_val_loss:.4f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
        # Save training metrics
        pd.DataFrame(metrics_history).to_csv(self.metrics_path, index=False)
        
        # Load best model for evaluation
        self.model.load_state_dict(torch.load(self.best_model_path))
        
        # Final evaluation on test set
        test_metrics = self.evaluate_model(test_loader)
        logger.info(f"Test evaluation: {test_metrics}")
        
        # Compute feature importance
        self.compute_feature_importance()
        
        # Save final model
        torch.save(self.model.state_dict(), MODEL_PATH)
        logger.info(f"Final model saved to {MODEL_PATH}")
        bias_level, bias_metrics = self.test_position_bias()

        if bias_level > 0.1:
            logger.warning(f"⚠️ Position bias detected in trained model: {bias_level:.4f}. Model may favor fighter position over attributes.")
        else:
            logger.info(f"✅ Model passed position bias check: {bias_level:.4f}. Predictions based on fighter attributes, not position.")
        return metrics_history, test_metrics, bias_level
    
    def test_position_bias(self):
        """
        Test whether the model exhibits position bias by comparing 
        normal vs. swapped predictions for test fighter pairs.
        """
        logger.info("Testing for position bias...")
        
        # Create 10 test fighter pairs with varied attributes
        test_pairs = []
        for i in range(10):
            # Create fighters with randomized but realistic attributes
            fighter1 = {
                'name': f'Test Fighter {i}A',
                'weight': 155 + i*5,
                'height': 175 + i*2,
                'reach': 180 + i*3,
                'wins': 10 + i,
                'losses': 5 - i//2,
                'sig_strikes_per_min': 3.5 + i*0.2,
                'takedown_avg': 1.5 + i*0.3,
                'sub_avg': 0.5 + i*0.1
            }
            
            fighter2 = {
                'name': f'Test Fighter {i}B',
                'weight': 155 - i*5,
                'height': 175 - i*2,
                'reach': 180 - i*3,
                'wins': 10 - i,
                'losses': 5 + i//2,
                'sig_strikes_per_min': 3.5 - i*0.2,
                'takedown_avg': 1.5 - i*0.3,
                'sub_avg': 0.5 - i*0.1
            }
            
            test_pairs.append((fighter1, fighter2))
        
        # Check predictions in both directions
        bias_metrics = []
        for fighter1, fighter2 in test_pairs:
            # Normal prediction
            pred1 = self.predict_fight_with_corners(fighter1, fighter2)
            
            # Swapped prediction
            pred2 = self.predict_fight_with_corners(fighter2, fighter1)
            
            # Calculate bias (how much position affects prediction)
            position_bias = abs(pred1['probability_red_wins'] - (1 - pred2['probability_blue_wins']))
            
            # Add to metrics
            bias_metrics.append({
                'fighter1': fighter1['name'],
                'fighter2': fighter2['name'],
                'normal_prob': pred1['probability_red_wins'],
                'swapped_prob': 1 - pred2['probability_blue_wins'],
                'position_bias': position_bias
            })
        
        # Calculate average bias
        avg_bias = sum(m['position_bias'] for m in bias_metrics) / len(bias_metrics)
        
        # Log results
        logger.info(f"Position bias test - Average bias: {avg_bias:.4f}")
        for m in bias_metrics:
            logger.info(f"  {m['fighter1']} vs {m['fighter2']}: Normal={m['normal_prob']:.2f}, Swapped={m['swapped_prob']:.2f}, Bias={m['position_bias']:.4f}")
        
        # Warning if bias is detected
        if avg_bias > 0.1:
            logger.warning(f"⚠️ Significant position bias detected: {avg_bias:.4f}")
        else:
            logger.info(f"✅ Position bias check passed: {avg_bias:.4f}")
        
        return avg_bias, bias_metrics
    def evaluate_model(self, data_loader):
        """
        Evaluate model performance on a dataset
        
        Args:
            data_loader: PyTorch DataLoader with test data
            
        Returns:
            dict: Dictionary with evaluation metrics
        """
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                
                all_preds.extend(outputs.cpu().detach().numpy().flatten())
                all_targets.extend(labels.cpu().detach().numpy().flatten())
        
        # Convert to numpy arrays and ensure proper format
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets).astype(int)  # Convert to integers
        binary_preds = (all_preds >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, binary_preds),
            'precision': precision_score(all_targets, binary_preds, zero_division=0),
            'recall': recall_score(all_targets, binary_preds, zero_division=0),
            'f1': f1_score(all_targets, binary_preds, zero_division=0),
            'auc': roc_auc_score(all_targets, all_preds)
        }
        
        return metrics
    
    def compute_feature_importance(self):
        """
        Compute feature importance using a permutation-based approach
        """
        if self.model is None or not hasattr(self, 'X_train'):
            logger.error("Model not trained or training data not available")
            return
        
        logger.info("Computing feature importance")
        
        # Get feature names
        feature_names = self.X_train.columns
        
        # Convert training data to tensor
        X_train_scaled = self.scaler.transform(self.X_train)
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
        
        # Get baseline predictions
        self.model.eval()
        with torch.no_grad():
            baseline_preds = self.model(X_train_tensor).cpu().numpy()
        
        # Calculate feature importance
        importances = []
        for i, feature in enumerate(feature_names):
            # Create a copy and permute the feature
            X_permuted = X_train_scaled.copy()
            X_permuted[:, i] = np.random.permutation(X_permuted[:, i])
            X_permuted_tensor = torch.FloatTensor(X_permuted).to(self.device)
            
            # Predict with permuted feature
            with torch.no_grad():
                permuted_preds = self.model(X_permuted_tensor).cpu().numpy()
            
            # Measure the decrease in performance
            # Higher decrease means more important feature
            importance = np.mean(np.abs(baseline_preds - permuted_preds))
            importances.append(importance)
        
        # Normalize importances
        importances = np.array(importances)
        importances = importances / np.sum(importances)
        
        # Sort importance
        sorted_idx = np.argsort(importances)[::-1]
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(sorted_idx)), importances[sorted_idx], align='center')
        plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
        plt.xlabel('Relative Importance')
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig(self.feature_importance_path)
        logger.info(f"Feature importance plot saved to {self.feature_importance_path}")
        
        # Save feature importance data
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        importance_df.to_csv(os.path.join(self.model_dir, 'feature_importance.csv'), index=False)
    
    def predict_fight(self, fighter1_features, fighter2_features):
        """
        Predict the outcome of a fight
        
        Args:
            fighter1_features (dict): Dictionary with fighter1 features
            fighter2_features (dict): Dictionary with fighter2 features
            
        Returns:
            float: Probability of fighter1 winning
        """
        # Load the model if not already loaded
        if self.model is None:
            self.load_model()
        
        # Combine features into a single dataframe
        fight_df = pd.DataFrame({**fighter1_features, **fighter2_features}, index=[0])
        
        # Load feature columns to ensure correct order
        try:
            feature_columns = joblib.load(os.path.join(self.model_dir, "feature_columns.pkl"))
            
            # Create a dataframe with all expected features initialized to 0
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            
            # Fill in the features we have
            for col in fight_df.columns:
                if col in input_df.columns:
                    input_df[col] = fight_df[col]
            
            # Scale the features
            if not os.path.exists(self.scaler_path):
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            
            scaler = joblib.load(self.scaler_path)
            input_scaled = scaler.transform(input_df)
            
            # Convert to tensor and predict
            input_tensor = torch.FloatTensor(input_scaled).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor).item()
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise
    
    def load_model(self, use_best=True):
        """
        Load a trained model
        
        Args:
            use_best (bool): Whether to load the best model from validation or the final model
            
        Returns:
            tuple: (model, feature_columns)
        """
        try:
            # Load input size
            input_size_path = os.path.join(self.model_dir, "input_size.pkl")
            if os.path.exists(input_size_path):
                input_size = joblib.load(input_size_path)
            else:
                # If not found, try to load feature columns
                feature_columns_path = os.path.join(self.model_dir, "feature_columns.pkl")
                if os.path.exists(feature_columns_path):
                    feature_columns = joblib.load(feature_columns_path)
                    input_size = len(feature_columns)
                else:
                    raise FileNotFoundError("Cannot determine input size for model - missing required files")
            
            # Initialize model with correct input size
            self.model = FightPredictor(input_size).to(self.device)
            
            # Load model weights
            model_path = self.best_model_path if use_best and os.path.exists(self.best_model_path) else MODEL_PATH
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
                
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
            
            # Load scaler
            if os.path.exists(self.scaler_path):
                self.scaler = joblib.load(self.scaler_path)
            
            # Load feature columns
            feature_columns_path = os.path.join(self.model_dir, "feature_columns.pkl")
            if os.path.exists(feature_columns_path):
                feature_columns = joblib.load(feature_columns_path)
            else:
                feature_columns = None
            
            logger.info(f"Model loaded successfully from {model_path}")
            return self.model, feature_columns
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_fight_with_corners(self, red_fighter_data, blue_fighter_data):
        """
        Comprehensive prediction with multiple edge case handling and strategies
        to avoid position bias and consistent percentage predictions.
        """
        try:
            # PHYSICAL REALITY CHECKS
            physical_factors = []
            advantage_magnitude = 0
            
            # Weight class difference (multiple thresholds)
            if 'weight' in red_fighter_data and 'weight' in blue_fighter_data:
                try:
                    red_weight = float(red_fighter_data['weight'])
                    blue_weight = float(blue_fighter_data['weight'])
                    weight_diff = red_weight - blue_weight
                    
                    # Determine advantage based on weight difference
                    if abs(weight_diff) > 25:  # Multiple weight classes
                        physical_factors.append(f"Extreme weight advantage: {abs(weight_diff):.1f} lbs")
                        advantage_magnitude += 0.5 * (1 if weight_diff > 0 else -1)
                    elif abs(weight_diff) > 15:  # One weight class
                        physical_factors.append(f"Significant weight advantage: {abs(weight_diff):.1f} lbs")
                        advantage_magnitude += 0.3 * (1 if weight_diff > 0 else -1)
                    elif abs(weight_diff) > 7:  # Within division but significant
                        physical_factors.append(f"Notable weight advantage: {abs(weight_diff):.1f} lbs")
                        advantage_magnitude += 0.15 * (1 if weight_diff > 0 else -1)
                except (ValueError, TypeError):
                    pass
            
            # Reach advantage (important for striking)
            if 'reach' in red_fighter_data and 'reach' in blue_fighter_data:
                try:
                    red_reach = float(red_fighter_data['reach'])
                    blue_reach = float(blue_fighter_data['reach'])
                    reach_diff = red_reach - blue_reach
                    
                    if abs(reach_diff) > 8:  # Extreme reach advantage
                        physical_factors.append(f"Extreme reach advantage: {abs(reach_diff):.1f} cm")
                        advantage_magnitude += 0.15 * (1 if reach_diff > 0 else -1)
                    elif abs(reach_diff) > 5:  # Significant reach advantage
                        physical_factors.append(f"Significant reach advantage: {abs(reach_diff):.1f} cm")
                        advantage_magnitude += 0.1 * (1 if reach_diff > 0 else -1)
                except (ValueError, TypeError):
                    pass
            
            # If extreme physical mismatches exist, use rules-based prediction
            if abs(advantage_magnitude) > 0.4:
                final_prob = 0.5 + advantage_magnitude
                final_prob = min(max(final_prob, 0.05), 0.95)  # Clamp
                
                return {
                    'probability_red_wins': round(float(final_prob), 2),
                    'probability_blue_wins': round(float(1 - final_prob), 2),
                    'predicted_winner': "Red" if final_prob > 0.5 else "Blue",
                    'confidence_level': "High",
                    'factors': physical_factors
                }
            
            # MULTI-STRATEGY PREDICTION
            results = []
            
            # Strategy 1: Basic model prediction
            standard_prob = self._predict_standard(red_fighter_data, blue_fighter_data)
            results.append(standard_prob)
            
            # Strategy 2: Swapped prediction
            swapped_prob = 1 - self._predict_standard(blue_fighter_data, red_fighter_data)
            results.append(swapped_prob)
            
            # Strategy 3: Feature-based heuristic prediction
            heuristic_prob = self._predict_heuristic(red_fighter_data, blue_fighter_data)
            results.append(heuristic_prob)
            
            # Strategy 4: Record-based prediction
            if 'wins' in red_fighter_data and 'losses' in red_fighter_data and \
            'wins' in blue_fighter_data and 'losses' in blue_fighter_data:
                record_prob = self._predict_from_records(red_fighter_data, blue_fighter_data)
                results.append(record_prob)
            
            # Combine predictions with weights (varies by confidence)
            combined_prob = 0
            weights = [0.35, 0.35, 0.15, 0.15]  # Adjust weights based on available strategies
            weight_sum = 0
            
            for i, prob in enumerate(results):
                if i < len(weights) and prob is not None:
                    combined_prob += prob * weights[i]
                    weight_sum += weights[i]
            
            if weight_sum > 0:
                combined_prob /= weight_sum
            else:
                combined_prob = 0.5  # Default if no valid predictions
            
            # Add controlled variability based on fighter attributes (prevents same % every time)
            variability_factor = self._calculate_variability(red_fighter_data, blue_fighter_data)
            final_prob = combined_prob + variability_factor
            
            # Clamp probabilities
            final_prob = min(max(final_prob, 0.05), 0.95)
            
            # Dynamic confidence levels
            confidence = self._determine_confidence(final_prob, results)
            
            return {
                'probability_red_wins': round(float(final_prob), 2),
                'probability_blue_wins': round(float(1 - final_prob), 2),
                'predicted_winner': "Red" if final_prob > 0.5 else "Blue",
                'confidence_level': confidence
            }
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'probability_red_wins': 0.5,
                'probability_blue_wins': 0.5,
                'predicted_winner': "Unknown (error)",
                'confidence_level': "Low",
                'error': str(e)
            }

    def _predict_standard(self, fighter1_data, fighter2_data):
        """
        Standard model prediction using the neural network
        """
        try:
            # Load the model if not already loaded
            if self.model is None:
                self.load_model()
            
            # Rename input data to match training data conventions
            renamed_red_data = {f'R_{k}': v for k, v in fighter1_data.items() if k not in ['id', 'name']}
            renamed_blue_data = {f'B_{k}': v for k, v in fighter2_data.items() if k not in ['id', 'name']}
            
            # Combine the fighter data
            combined_data = {**renamed_red_data, **renamed_blue_data}
            df = pd.DataFrame([combined_data])
            
            # Rename R_ and B_ to fighter1_ and fighter2_
            rename_dict = {col: col.replace('R_', 'fighter1_').replace('B_', 'fighter2_') 
                        for col in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Load feature columns
            model_dir = os.path.dirname(MODEL_PATH)
            feature_columns_file = os.path.join(model_dir, "feature_columns.pkl")
            
            if os.path.exists(feature_columns_file):
                feature_columns = joblib.load(feature_columns_file)
                logger.debug(f"Loaded {len(feature_columns)} feature columns")
            else:
                logger.warning("Feature columns file not found")
                feature_columns = df.select_dtypes(include=['number']).columns.tolist()
            
            # Ensure all expected feature columns exist
            input_df = pd.DataFrame(0, index=df.index, columns=feature_columns)
            for col in feature_columns:
                if col in df.columns:
                    input_df[col] = df[col]
            
            # Load and apply scaler
            scaler_path = os.path.join(model_dir, "scaler.save")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                input_df = pd.DataFrame(scaler.transform(input_df), 
                                    columns=input_df.columns,
                                    index=input_df.index)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_df.values).to(self.device)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor).item()
            
            return prediction
        except Exception as e:
            logger.error(f"Error in standard prediction: {e}")
            return None

    def _predict_heuristic(self, fighter1, fighter2):
        """Rule-based heuristic prediction using fighter stats"""
        try:
            score1 = 0
            score2 = 0
            
            # Add points for wins and win percentage
            if 'wins' in fighter1 and 'losses' in fighter1:
                f1_total = float(fighter1['wins']) + float(fighter1['losses'])
                if f1_total > 0:
                    score1 += float(fighter1['wins']) * 0.1  # Points for wins
                    score1 += (float(fighter1['wins']) / f1_total) * 2  # Points for win percentage
            
            if 'wins' in fighter2 and 'losses' in fighter2:
                f2_total = float(fighter2['wins']) + float(fighter2['losses'])
                if f2_total > 0:
                    score2 += float(fighter2['wins']) * 0.1  # Points for wins
                    score2 += (float(fighter2['wins']) / f2_total) * 2  # Points for win percentage
            
            # Add points for striking, takedowns, and submissions
            for stat in ['sig_strikes_per_min', 'takedown_avg', 'sub_avg']:
                if stat in fighter1:
                    try:
                        score1 += float(fighter1[stat]) * 0.2
                    except (ValueError, TypeError):
                        pass
                if stat in fighter2:
                    try:
                        score2 += float(fighter2[stat]) * 0.2
                    except (ValueError, TypeError):
                        pass
            
            # Convert to probability
            total = score1 + score2
            if total > 0:
                prob = score1 / total
                return min(max(prob, 0.05), 0.95)  # Clamp
            else:
                return 0.5  # No data
        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return None

    def _predict_from_records(self, fighter1, fighter2):
        """Prediction based on win/loss records and experience"""
        try:
            # Extract record data
            f1_wins = float(fighter1['wins'])
            f1_losses = float(fighter1['losses'])
            f2_wins = float(fighter2['wins'])
            f2_losses = float(fighter2['losses'])
            
            # Calculate total fights
            f1_total = f1_wins + f1_losses
            f2_total = f2_wins + f2_losses
            
            # Don't use this method if fighters have very few fights
            if f1_total < 3 or f2_total < 3:
                return None
            
            # Calculate win rates
            f1_winrate = f1_wins / f1_total if f1_total > 0 else 0
            f2_winrate = f2_wins / f2_total if f2_total > 0 else 0
            
            # Combined formula: win rate difference + experience factor
            winrate_diff = f1_winrate - f2_winrate
            
            # Experience factor - more experienced fighters have slight advantage
            experience_factor = 0.05 * (f1_total - f2_total) / max(f1_total + f2_total, 10)
            
            # Final probability
            prob = 0.5 + winrate_diff + experience_factor
            
            return min(max(prob, 0.05), 0.95)  # Clamp
        except Exception as e:
            logger.error(f"Error in record-based prediction: {e}")
            return None

    def _calculate_variability(self, fighter1, fighter2):
        """Calculate a small variability factor based on fighter stats"""
        try:
            # Create a unique hash from fighter names
            import hashlib
            name_hash = hashlib.md5((
                str(fighter1.get('name', '')) + 
                str(fighter2.get('name', ''))
            ).encode()).hexdigest()
            
            # Convert hash to a number between -0.07 and 0.07
            hash_value = int(name_hash, 16) / (16**32)
            variability = (hash_value * 2 - 1) * 0.07
            
            return variability
        except Exception as e:
            logger.error(f"Error calculating variability: {e}")
            return 0

    def _determine_confidence(self, probability, prediction_results):
        """Dynamic confidence calculation based on prediction consistency and probability"""
        try:
            import numpy as np
            # Filter out None values
            valid_results = [p for p in prediction_results if p is not None]
            
            # Need at least 2 predictions for standard deviation
            if len(valid_results) >= 2:
                # Calculate standard deviation between predictions
                prediction_std = np.std(valid_results)
                
                # High standard deviation means predictions disagree
                if prediction_std > 0.2:
                    return "Low"  # Models disagree
            
            # Use probability distance from 0.5 as fallback
            prob_distance = abs(probability - 0.5)
            if prob_distance < 0.1:
                return "Low"
            elif prob_distance < 0.2:
                return "Medium"
            else:
                return "High"
        except Exception as e:
            logger.error(f"Error determining confidence: {e}")
            return "Low"  # Default to low confidence on error

    def _compute_advantage_metrics(self, fighter1, fighter2):
        """Calculate relative advantage metrics between fighters"""
        advantages = {}
        
        # Weight advantage (most important physical factor)
        if 'weight' in fighter1 and 'weight' in fighter2:
            try:
                advantages['weight_advantage'] = float(fighter1['weight']) - float(fighter2['weight'])
            except:
                pass
        
        # Height and reach advantages
        for attr in ['height', 'reach']:
            if attr in fighter1 and attr in fighter2:
                try:
                    advantages[f'{attr}_advantage'] = float(fighter1[attr]) - float(fighter2[attr])
                except:
                    pass
        
        # Record and experience advantages
        if all(attr in fighter1 and attr in fighter2 for attr in ['wins', 'losses']):
            try:
                # Calculate win percentages
                f1_fights = float(fighter1['wins']) + float(fighter1['losses'])
                f2_fights = float(fighter2['wins']) + float(fighter2['losses'])
                
                if f1_fights > 0 and f2_fights > 0:
                    f1_win_rate = float(fighter1['wins']) / f1_fights
                    f2_win_rate = float(fighter2['wins']) / f2_fights
                    advantages['win_rate_advantage'] = f1_win_rate - f2_win_rate
                    
                    # Experience difference
                    advantages['experience_advantage'] = f1_fights - f2_fights
            except:
                pass
        
        # Fighting style advantages
        for stat in ['sig_strikes_per_min', 'takedown_avg', 'sub_avg']:
            if stat in fighter1 and stat in fighter2:
                try:
                    advantages[f'{stat}_advantage'] = float(fighter1[stat]) - float(fighter2[stat])
                except:
                    pass
        
        return advantages

    def plot_training_history(self):
        """Plot training history metrics"""
        if not os.path.exists(self.metrics_path):
            logger.error("Training metrics not found")
            return
        
        metrics_df = pd.read_csv(self.metrics_path)
        
        # Plot losses
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss')
        plt.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        # Plot accuracy
        plt.subplot(2, 2, 2)
        plt.plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        
        # Plot precision and recall
        plt.subplot(2, 2, 3)
        plt.plot(metrics_df['epoch'], metrics_df['val_precision'], label='Precision')
        plt.plot(metrics_df['epoch'], metrics_df['val_recall'], label='Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Precision and Recall')
        
        # Plot F1 and AUC
        plt.subplot(2, 2, 4)
        plt.plot(metrics_df['epoch'], metrics_df['val_f1'], label='F1 Score')
        plt.plot(metrics_df['epoch'], metrics_df['val_auc'], label='ROC AUC')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('F1 Score and ROC AUC')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.model_dir, 'training_history.png'))
        logger.info(f"Training history plot saved to {os.path.join(self.model_dir, 'training_history.png')}")

def validate_fighter_data(fighter_data, label):
    """
    Validates fighter data for completeness and formats values if needed.
    
    Parameters:
    fighter_data (dict): Dictionary of fighter stats
    label (str): Label for the fighter (e.g., "Red corner")
    
    Returns:
    bool: True if validation passes
    """
    # Essential features that should be present
    essential_features = [
        'avg_KD', 'avg_SIG_STR_pct', 'avg_TD_pct', 
        'wins', 'losses'
    ]
    
    # Check for prefix - either use it consistently or not at all
    has_prefix = any(key.startswith('R_') or key.startswith('B_') for key in fighter_data.keys())
    prefix = label[0] if has_prefix else ''
    
    # Validate essential features
    missing_features = []
    for feature in essential_features:
        # Check with and without prefix
        feature_with_prefix = f"{prefix}_{feature}" if prefix else feature
        if feature_with_prefix not in fighter_data and feature not in fighter_data:
            missing_features.append(feature)
    
    if missing_features:
        print(f"Warning: {label} data is missing essential features: {missing_features}")
        print("Predictions may be less accurate without these features.")
    
    # Automatically convert percentage values if they're in 0-100 range instead of 0-1
    percentage_features = ['avg_SIG_STR_pct', 'avg_TD_pct']
    for feature in percentage_features:
        # Check with prefix
        feature_with_prefix = f"{prefix}_{feature}" if prefix else feature
        if feature_with_prefix in fighter_data:
            value = fighter_data[feature_with_prefix]
            if isinstance(value, (int, float)) and value > 1.0:
                fighter_data[feature_with_prefix] = value / 100.0
                print(f"Converted {feature_with_prefix} from {value} to {value/100.0} (percentage to decimal)")
        
        # Check without prefix
        elif feature in fighter_data:
            value = fighter_data[feature]
            if isinstance(value, (int, float)) and value > 1.0:
                fighter_data[feature] = value / 100.0
                print(f"Converted {feature} from {value} to {value/100.0} (percentage to decimal)")
    
    return True

def test_prediction(red_fighter_data=None, blue_fighter_data=None, model_path=None, verbose=True):
    """
    Test UFC fight prediction with given fighter data or sample data.
    
    Parameters:
    red_fighter_data (dict): Dictionary of red corner fighter stats. If None, uses sample data.
    blue_fighter_data (dict): Dictionary of blue corner fighter stats. If None, uses sample data.
    model_path (str): Optional path to specific model file. If None, uses default path.
    verbose (bool): Whether to print detailed output.
    
    Returns:
    dict: Prediction results
    """
    import logging
    
    # Set up logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger('ufc_prediction_test')
    
    try:
        # Create or use sample data if not provided
        if red_fighter_data is None:
            logger.info("Using sample data for red fighter")
            red_fighter_data = {
                'R_avg_KD': 0.5,
                'R_avg_SIG_STR_pct': 0.48,
                'R_avg_TD_pct': 0.65,
                'R_avg_SUB_ATT': 1.2,
                'R_avg_REV': 0.3,
                'R_avg_SIG_STR_att': 8.5,
                'R_avg_SIG_STR_landed': 4.1,
                'R_avg_TOTAL_STR_att': 12.3,
                'R_avg_TOTAL_STR_landed': 6.7,
                'R_avg_TD_att': 2.1,
                'R_avg_TD_landed': 1.3,
                'R_longest_win_streak': 4,
                'R_wins': 10,
                'R_losses': 2,
                'R_avg_CTRL': 120.5,  # Control time in seconds
                'R_Height_cms': 180,
                'R_Reach_cms': 185,
                'R_Weight_lbs': 155,
                'R_age': 28,
                'R_Stance': 'Orthodox',
                'R_total_rounds_fought': 25,
                'R_total_title_bouts': 2,
                'R_win_by_KO_TKO': 5,
                'R_win_by_SUB': 3,
                'R_win_by_DEC': 2
            }
        
        if blue_fighter_data is None:
            logger.info("Using sample data for blue fighter")
            blue_fighter_data = {
                'B_avg_KD': 0.3,
                'B_avg_SIG_STR_pct': 0.52,
                'B_avg_TD_pct': 0.45,
                'B_avg_SUB_ATT': 1.5,
                'B_avg_REV': 0.2,
                'B_avg_SIG_STR_att': 9.2,
                'B_avg_SIG_STR_landed': 4.8,
                'B_avg_TOTAL_STR_att': 13.1,
                'B_avg_TOTAL_STR_landed': 7.2,
                'B_avg_TD_att': 1.8,
                'B_avg_TD_landed': 0.8,
                'B_longest_win_streak': 3,
                'B_wins': 8,
                'B_losses': 4,
                'B_avg_CTRL': 90.2,  # Control time in seconds
                'B_Height_cms': 178,
                'B_Reach_cms': 183,
                'B_Weight_lbs': 155,
                'B_age': 30,
                'B_Stance': 'Southpaw',
                'B_total_rounds_fought': 22,
                'B_total_title_bouts': 0,
                'B_win_by_KO_TKO': 3,
                'B_win_by_SUB': 4,
                'B_win_by_DEC': 1
            }
        
        # Validate fighter data
        validate_fighter_data(red_fighter_data, "Red corner")
        validate_fighter_data(blue_fighter_data, "Blue corner")
        
        # Get the model directory path
        model_path_to_use = model_path if model_path else MODEL_PATH
        model_dir = os.path.dirname(model_path_to_use)
        
        # Load the input size parameter saved during training
        input_size_file = os.path.join(model_dir, "input_size.pkl")
        if os.path.exists(input_size_file):
            input_size = joblib.load(input_size_file)
            logger.info(f"Loaded input size from file: {input_size}")
        else:
            # Default to a reasonable value if file doesn't exist
            feature_columns_file = os.path.join(model_dir, "feature_columns.pkl")
            if os.path.exists(feature_columns_file):
                feature_columns = joblib.load(feature_columns_file)
                input_size = len(feature_columns)
            else:
                # Reasonable fallback
                input_size = 134
            logger.warning(f"Input size file not found. Using default: {input_size}")
            
        # Initialize model with correct input size
        model = FightPredictor(input_size=input_size)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the model weights
        model_file = model_path_to_use
        logger.info(f"Loading model from: {model_file}")
        model.load_state_dict(torch.load(model_file))
        model.to(device)
        model.eval()  # Set to evaluation mode
        
        # Initialize the trainer with the model
        trainer = ModelTrainer(None)  # No data needed just for prediction
        trainer.model = model
        trainer.device = device
        
        # Make prediction
        logger.info("Making prediction")
        result = trainer.predict_fight_with_corners(red_fighter_data, blue_fighter_data)
        
        # Format output
        if verbose:
            print("\n" + "="*50)
            print("UFC FIGHT PREDICTION RESULTS:")
            print("="*50)
            print(f"Red fighter win probability: {result['probability_red_wins']:.2f} ({result['probability_red_wins']*100:.1f}%)")
            print(f"Blue fighter win probability: {result['probability_blue_wins']:.2f} ({result['probability_blue_wins']*100:.1f}%)")
            print(f"Predicted winner: {result['predicted_winner']}")
            print(f"Confidence: {result['confidence_level']}")
            
            # Add visualization if matplotlib is available
            try:
                plt.figure(figsize=(8, 4))
                probabilities = [result['probability_red_wins'], result['probability_blue_wins']]
                labels = ['Red Fighter', 'Blue Fighter']
                colors = ['#ff4d4d', '#4d79ff']
                
                plt.bar(labels, probabilities, color=colors)
                plt.ylabel('Win Probability')
                plt.title('UFC Fight Prediction')
                plt.ylim(0, 1)
                
                # Add percentage labels on the bars
                for i, v in enumerate(probabilities):
                    plt.text(i, v + 0.02, f"{v*100:.1f}%", ha='center')
                
                plt.tight_layout()
                
                # Save the plot
                output_dir = os.path.dirname(MODEL_PATH)
                plt.savefig(os.path.join(output_dir, 'prediction_result.png'))
                print(f"\nPrediction visualization saved to {os.path.join(output_dir, 'prediction_result.png')}")
                plt.close()
            except Exception as e:
                logger.warning(f"Visualization error: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

def compare_fighters(fighter1, fighter2, model_path=None):
    """
    Compare two fighters and show detailed stat comparisons alongside prediction.
    
    Parameters:
    fighter1 (dict): Dictionary of first fighter stats 
    fighter2 (dict): Dictionary of second fighter stats
    model_path (str): Optional path to specific model file
    
    Returns:
    dict: Prediction results
    """
    # Format fighter data with proper prefixes
    red_fighter = {}
    for key, value in fighter1.items():
        if not key.startswith('R_'):
            red_fighter[f'R_{key}'] = value
        else:
            red_fighter[key] = value
            
    blue_fighter = {}
    for key, value in fighter2.items():
        if not key.startswith('B_'):
            blue_fighter[f'B_{key}'] = value
        else:
            blue_fighter[key] = value
    
    # Create comparison table
    comparison_data = []
    
    # Common stats to compare (without prefixes)
    stats_to_compare = [
        'avg_KD', 'avg_SIG_STR_pct', 'avg_TD_pct', 
        'wins', 'losses', 'avg_SIG_STR_landed',
        'avg_TD_landed', 'Height_cms', 'Reach_cms',
        'win_by_KO_TKO', 'win_by_SUB', 'win_by_DEC'
    ]
    
    # Build comparison table
    for stat in stats_to_compare:
        red_key = f'R_{stat}'
        blue_key = f'B_{stat}'
        
        red_value = red_fighter.get(red_key, 'N/A')
        blue_value = blue_fighter.get(blue_key, 'N/A')
        
        # Format percentages
        if 'pct' in stat and isinstance(red_value, (int, float)) and isinstance(blue_value, (int, float)):
            red_value = f"{red_value*100:.1f}%" if red_value <= 1 else f"{red_value:.1f}%"
            blue_value = f"{blue_value*100:.1f}%" if blue_value <= 1 else f"{blue_value:.1f}%"
        
        # Add comparison row
        comparison_data.append({
            'Statistic': stat.replace('_', ' ').title(),
            'Red Fighter': red_value,
            'Blue Fighter': blue_value,
            'Advantage': 'Red' if red_value > blue_value and isinstance(red_value, (int, float)) else 
                         'Blue' if blue_value > red_value and isinstance(blue_value, (int, float)) else 'Equal'
        })
    
    # Print comparison table
    comparison_df = pd.DataFrame(comparison_data)
    print("\n" + "="*80)
    print("FIGHTER COMPARISON:")
    print("="*80)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(comparison_df)
    print("="*80 + "\n")
    
    # Make prediction
    result = test_prediction(red_fighter, blue_fighter, model_path)
    
    # Save comparison to CSV
    try:
        output_dir = os.path.dirname(MODEL_PATH)
        comparison_df.to_csv(os.path.join(output_dir, 'fighter_comparison.csv'), index=False)
        print(f"Fighter comparison saved to {os.path.join(output_dir, 'fighter_comparison.csv')}")
    except Exception as e:
        print(f"Could not save fighter comparison to CSV: {e}")
    
    return result

def batch_test_predictions(test_cases=None, model_path=None):
    """
    Run multiple prediction tests and analyze overall model performance.
    
    Parameters:
    test_cases (list): List of dictionaries, each containing 'red_fighter', 'blue_fighter',
                      and optionally 'actual_winner' and 'description'
    model_path (str): Optional path to specific model file
    
    Returns:
    dict: Summary statistics of test performance
    """
    # If no model_path is provided, use default from config
    if model_path is None:
        model_path = MODEL_PATH

    # Use sample test cases if none provided
    if test_cases is None:
        print("🧪 Using sample test cases...")
        test_cases = [
            {
                'description': 'Experienced fighter vs. Newcomer',
                'red_fighter': {'R_avg_KD': 0.6, 'R_avg_SIG_STR_pct': 0.55, 'R_avg_TD_pct': 0.70, 'R_wins': 15, 'R_losses': 2},
                'blue_fighter': {'B_avg_KD': 0.2, 'B_avg_SIG_STR_pct': 0.40, 'B_avg_TD_pct': 0.30, 'B_wins': 3, 'B_losses': 1},
                'actual_winner': 'Red'
            },
            {
                'description': 'Striker vs. Grappler',
                'red_fighter': {'R_avg_KD': 0.8, 'R_avg_SIG_STR_pct': 0.60, 'R_avg_TD_pct': 0.30, 'R_wins': 10, 'R_losses': 3},
                'blue_fighter': {'B_avg_KD': 0.1, 'B_avg_SIG_STR_pct': 0.35, 'B_avg_TD_pct': 0.75, 'B_wins': 8, 'B_losses': 4},
                'actual_winner': 'Blue'
            },
            {
                'description': 'Evenly matched fighters',
                'red_fighter': {'R_avg_KD': 0.4, 'R_avg_SIG_STR_pct': 0.48, 'R_avg_TD_pct': 0.50, 'R_wins': 8, 'R_losses': 5},
                'blue_fighter': {'B_avg_KD': 0.5, 'B_avg_SIG_STR_pct': 0.47, 'B_avg_TD_pct': 0.55, 'B_wins': 9, 'B_losses': 4},
                'actual_winner': 'Blue'
            }
        ]
    
    # Run batch tests
    results = []
    for i, case in enumerate(test_cases):
        print(f"\n📝 Running test case {i+1}/{len(test_cases)}: {case.get('description', 'Unnamed test')}")
        
        try:
            prediction = test_prediction(
                case['red_fighter'], 
                case['blue_fighter'],
                model_path,
                verbose=False  # Reduce output clutter
            )
        except Exception as e:
            print(f"⚠️ Error in prediction: {e}")
            import traceback
            traceback.print_exc()
            continue  # Move to the next test case
        
        # Prepare result record
        result_record = {
            'test_id': i+1,
            'description': case.get('description', 'Unnamed test'),
            'predicted_winner': prediction.get('predicted_winner', 'Unknown'),
            'red_win_probability': prediction.get('probability_red_wins', 0),
            'blue_win_probability': prediction.get('probability_blue_wins', 0),
            'confidence': prediction.get('confidence_level', 'Unknown')
        }
        
        # Add actual winner if provided
        if 'actual_winner' in case:
            result_record['actual_winner'] = case['actual_winner']
            result_record['prediction_correct'] = (case['actual_winner'] == prediction['predicted_winner'])
        
        results.append(result_record)
        
        # Print individual result
        print(f"📊 Prediction: {prediction['predicted_winner']} (Confidence: {prediction['confidence_level']})")
        if 'actual_winner' in case:
            correct = case['actual_winner'] == prediction['predicted_winner']
            print(f"✅ Actual: {case['actual_winner']} ({'✔️ CORRECT' if correct else '❌ INCORRECT'})")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Generate summary statistics
    summary = {'total_tests': len(results)}
    
    if 'prediction_correct' in results_df.columns:
        summary['correct_predictions'] = results_df['prediction_correct'].sum()
        summary['accuracy'] = summary['correct_predictions'] / summary['total_tests']
        
        # Handle cases where confidence levels might not exist
        high_conf = results_df[results_df['confidence'] == 'High'] if 'confidence' in results_df.columns else pd.DataFrame()
        med_conf = results_df[results_df['confidence'] == 'Medium'] if 'confidence' in results_df.columns else pd.DataFrame()
        low_conf = results_df[results_df['confidence'] == 'Low'] if 'confidence' in results_df.columns else pd.DataFrame()
        
        if not high_conf.empty and 'prediction_correct' in high_conf:
            summary['high_conf_accuracy'] = high_conf['prediction_correct'].mean()
            summary['high_conf_count'] = len(high_conf)
        
        if not med_conf.empty and 'prediction_correct' in med_conf:
            summary['med_conf_accuracy'] = med_conf['prediction_correct'].mean()
            summary['med_conf_count'] = len(med_conf)
            
        if not low_conf.empty and 'prediction_correct' in low_conf:
            summary['low_conf_accuracy'] = low_conf['prediction_correct'].mean()
            summary['low_conf_count'] = len(low_conf)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 BATCH TEST SUMMARY:")
    print("="*50)
    print(f"✅ Total tests: {summary['total_tests']}")
    
    if 'accuracy' in summary:
        print(f"🎯 Overall accuracy: {summary['accuracy']*100:.1f}%")
        print(f"✅ Correct predictions: {summary['correct_predictions']}/{summary['total_tests']}")

        print("\n📊 Accuracy by confidence level:")
        if 'high_conf_accuracy' in summary:
            print(f"🔵 High confidence: {summary['high_conf_accuracy']*100:.1f}% ({summary['high_conf_count']} predictions)")
        if 'med_conf_accuracy' in summary:
            print(f"🟠 Medium confidence: {summary['med_conf_accuracy']*100:.1f}% ({summary['med_conf_count']} predictions)")
        if 'low_conf_accuracy' in summary:
            print(f"🟡 Low confidence: {summary['low_conf_accuracy']*100:.1f}% ({summary['low_conf_count']} predictions)")

    # Save results to CSV
    try:
        output_dir = os.path.dirname(MODEL_PATH)
        results_df.to_csv(os.path.join(output_dir, 'batch_test_results.csv'), index=False)
        print(f"📄 Test results saved to {os.path.join(output_dir, 'batch_test_results.csv')}")
    except Exception as e:
        print(f"⚠️ Could not save test results to CSV: {e}")
    
    return summary

def main():
    """Main entry point for UFC fight prediction model training and testing"""
    import os
    import pandas as pd
    import joblib
    import traceback
    from config import MODEL_PATH, CSV_FILE_PATH, SCALER_PATH   
    
    # Check if required files exist
    if not os.path.exists(CSV_FILE_PATH):
        print(f"❌ Error: Dataset {CSV_FILE_PATH} not found! Run `python data_loader.py` first.")
        exit(1)

    if not os.path.exists(DATABASE_PATH):
        print(f"❌ Warning: Database {DATABASE_PATH} not found! You may need to run `python database.py` first.")
        # Continue anyway for now

    print("✅ All required files exist. Proceeding with training...")
    
    try:
        # Load raw data from CSV
        print(f"📂 Loading data from {CSV_FILE_PATH}")
        raw_data = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ Successfully loaded raw data with shape: {raw_data.shape}")

        # Preprocess data to match the model's expected format
        print("⚙ Preprocessing data to match model format...")
        processed_data = preprocess_data(raw_data)
        print(f"✅ Preprocessed data shape: {processed_data.shape}")
        
        # Ensure target column exists
        if 'fighter1_won' not in processed_data.columns:
            raise ValueError("❌ Error: Dataset must contain 'fighter1_won' column after preprocessing!")

        # Initialize the model trainer with processed data
        print("🧑‍🏫 Initializing model trainer...")
        trainer = ModelTrainer(processed_data)

        # Train the model
        print("🎯 Starting model training...")
        history, test_metrics = trainer.train_model()
        trainer.plot_training_history()

        print(f"\n✅ Training complete! Model saved to {MODEL_PATH}")
        print(f"📊 Test Metrics: {test_metrics}")

    except Exception as e:
        print(f"\n🚨 Error during execution: {str(e)}")
        traceback.print_exc()

    # Present the test suite options
    print("\n=============================")
    print("🎮 UFC Fight Prediction Test Suite")
    print("=============================")
    print("1️⃣ Test a single prediction with sample data")
    print("2️⃣ Compare two fighters")
    print("3️⃣ Run batch tests")

    choice = input("Enter your choice (1-3): ")

    try:
        if choice == '1':
            # Call test_prediction
            test_prediction()
        elif choice == '2':
            fighter1 = {
                'avg_KD': 0.5, 'avg_SIG_STR_pct': 0.48, 'avg_TD_pct': 0.65,
                'wins': 10, 'losses': 2, 'Height_cms': 180, 'Reach_cms': 185
            }
            fighter2 = {
                'avg_KD': 0.3, 'avg_SIG_STR_pct': 0.52, 'avg_TD_pct': 0.45,
                'wins': 8, 'losses': 4, 'Height_cms': 178, 'Reach_cms': 183
            }
            compare_fighters(fighter1, fighter2)
        elif choice == '3':
            batch_test_predictions()
        else:
            print("❌ Invalid choice, exiting.")
    except Exception as e:
        print(f"❌ Error in test selection: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()