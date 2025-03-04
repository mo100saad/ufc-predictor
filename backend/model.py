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
from config import MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, TEST_SIZE, VALIDATION_SIZE, CSV_FILE_PATH, MODEL_PATH, DATABASE_PATH, SCALER_PATH
import logging

# Ensure required files exist before training
if not os.path.exists(CSV_FILE_PATH):
    print(f"❌ Error: Dataset {CSV_FILE_PATH} not found! Run `python data_loader.py` first.")
    exit(1)

if not os.path.exists(DATABASE_PATH):
    print(f"❌ Error: Database {DATABASE_PATH} not found! Run `python database.py` first.")
    exit(1)

print("✅ All required files exist. Proceeding with training...")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(MODEL_PATH), 'training.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('ufc_model')

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
        # Expanded architecture with more neurons in hidden layers
        self.layer1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.layer2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.layer3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)  # Increased dropout for better regularization
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.layer3(x)))
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
    
    def preprocess_data(self):
        logger.info("Starting data preprocessing")

        # Save original column names for later validation
        feature_columns = self.data_df.drop('fighter1_won', axis=1).columns.tolist()
        pd.Series(feature_columns).to_json(self.feature_names_path)
        
        # Drop non-numeric columns if any
        self.data_df = self.data_df.select_dtypes(include=[np.number])
        
        # Handle missing values with more sophisticated approach
        num_columns = self.data_df.columns
        for col in num_columns:
            if self.data_df[col].isna().any():
                if col == 'fighter1_won':
                    # Don't impute target variable
                    self.data_df = self.data_df.dropna(subset=[col])
                else:
                    # Impute with median instead of 0 for better representation
                    self.data_df[col] = self.data_df[col].fillna(self.data_df[col].median())
        
        # Separate features and labels
        X = self.data_df.drop('fighter1_won', axis=1)
        y = self.data_df['fighter1_won']
        
        # Split data into train and test sets
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42, stratify=y
        )
        
        # Further split training data into training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=VALIDATION_SIZE/(1-TEST_SIZE),  # Adjust for previous split
            random_state=42, 
            stratify=y_train_val
        )
        
        logger.info(f"Data split: Train: {len(X_train)}, Validation: {len(X_val)}, Test: {len(X_test)}")
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save the scaler
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
        
        # Store original feature dataframes for feature importance later
        self.X_train = X_train
        
        return train_loader, val_loader, test_loader, X.shape[1]
    
    def train_model(self):
        # Preprocess data
        train_loader, val_loader, test_loader, input_size = self.preprocess_data()
    
        # Initialize model
        self.model = FightPredictor(input_size).to(self.device)
        logger.info(f"Model initialized with input size: {input_size}")
        logger.info(f"Model architecture:\n{self.model}")
        
        # Save the input size for future reference
        import joblib
        import os
        from config import MODEL_PATH
        joblib.dump(input_size, os.path.join(os.path.dirname(MODEL_PATH), "input_size.pkl"))
        
        # Save feature columns from dataloader
        if hasattr(self, 'feature_columns'):
            joblib.dump(self.feature_columns, os.path.join(os.path.dirname(MODEL_PATH), "feature_columns.pkl"))
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
        
        # Initialize early stopping parameters
        best_val_loss = float('inf')
        early_stopping_counter = 0
        early_stopping_patience = 10
        
        # Initialize metrics tracking
        metrics_history = {
            'epoch': [], 'train_loss': [], 'val_loss': [], 
            'val_accuracy': [], 'val_precision': [], 
            'val_recall': [], 'val_f1': [], 'val_auc': []
        }
        
        # Training loop
        logger.info(f"Starting training for {EPOCHS} epochs")
        for epoch in range(EPOCHS):
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
        
        return metrics_history, test_metrics
    
    def evaluate_model(self, data_loader):
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
        """Compute feature importance using a permutation-based approach"""
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
    
    def load_model(self, use_best=True):
        """Load the trained model"""
        # Load the scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Load training feature names for validation
        training_features_path = os.path.join(self.model_dir, 'training_features.pkl')
        training_features = joblib.load(training_features_path)
        
        # Get input size from training features
        input_size = len(training_features)
        
        # Initialize model with the correct input size
        self.model = FightPredictor(input_size)
        
        # Load model weights from the best model (if available) or fallback to MODEL_PATH
        model_path = self.best_model_path if use_best and os.path.exists(self.best_model_path) else MODEL_PATH
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path} with input size {input_size}")
        
        return self.model, training_features

    
    def predict_fight_with_corners(self, red_fighter_data, blue_fighter_data):
        """
        Make a prediction using fighter corner data
        
        Parameters:
        red_fighter_data (dict): Red corner fighter stats
        blue_fighter_data (dict): Blue corner fighter stats
        
        Returns:
        dict: Prediction results including probabilities and predicted winner
        """
        import pandas as pd
        import torch
        import joblib
        import os
        from config import MODEL_PATH
        import logging
        
        logger = logging.getLogger('ufc_prediction')
        
        try:
            # Combine the fighter data
            combined_data = {**red_fighter_data, **blue_fighter_data}
            df = pd.DataFrame([combined_data])
            
            # Load feature columns
            model_dir = os.path.dirname(MODEL_PATH)
            feature_columns_file = os.path.join(model_dir, "feature_columns.pkl")
            
            if os.path.exists(feature_columns_file):
                feature_columns = joblib.load(feature_columns_file)
                logger.info(f"Loaded {len(feature_columns)} feature columns")
            else:
                # Fall back to a heuristic method if file doesn't exist
                logger.warning("Feature columns file not found, using model's input_size")
                # Get all possible numeric features from the dataframe
                feature_columns = df.select_dtypes(include=['number']).columns.tolist()
                
                # Check if we have more features than model expects
                input_size_file = os.path.join(model_dir, "input_size.pkl")
                if os.path.exists(input_size_file):
                    input_size = joblib.load(input_size_file)
                    if len(feature_columns) > input_size:
                        logger.warning(f"Found {len(feature_columns)} features but model expects {input_size}")
                        # This is a fallback but not ideal
                        feature_columns = feature_columns[:input_size]
            
            # Rename R_ and B_ to fighter1_ and fighter2_
            rename_dict = {col: col.replace('R_', 'fighter1_').replace('B_', 'fighter2_') 
                        for col in df.columns}
            df = df.rename(columns=rename_dict)
            
            # Handle one-hot encoding for stance (if present)
            for prefix in ['fighter1_Stance', 'fighter2_Stance']:
                stance_cols = [col for col in df.columns if col.startswith(prefix)]
                if stance_cols:
                    stance_dummies = pd.get_dummies(df[stance_cols[0]], prefix=prefix)
                    df = pd.concat([df, stance_dummies], axis=1)
                    df = df.drop(columns=stance_cols)
            
            # Convert all columns to numeric
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_numeric(df[col])
                    except:
                        logger.warning(f"Dropping non-numeric column: {col}")
                        df = df.drop(columns=[col])
            
            # Ensure all expected feature columns exist
            # Create missing columns with zeros
            input_df = pd.DataFrame(0, index=df.index, columns=feature_columns)
            for col in feature_columns:
                if col in df.columns:
                    input_df[col] = df[col]
            
            # If a scaler was used during training, apply it
            scaler_path = os.path.join(model_dir, "scaler.save")
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                input_df = pd.DataFrame(scaler.transform(input_df), 
                                    columns=input_df.columns,
                                    index=input_df.index)
                
            # Convert to tensor
            input_tensor = torch.FloatTensor(input_df.values)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                prediction = self.model(input_tensor).item()
            
            # Get probabilities
            probability_red_wins = prediction
            probability_blue_wins = 1 - prediction
            
            # Determine confidence level
            if abs(probability_red_wins - 0.5) < 0.1:
                confidence = "Low"
            elif abs(probability_red_wins - 0.5) < 0.25:
                confidence = "Medium"
            else:
                confidence = "High"
                
            # Determine winner
            predicted_winner = "Red" if probability_red_wins > 0.5 else "Blue"
            
            return {
                'probability_red_wins': probability_red_wins,
                'probability_blue_wins': probability_blue_wins,
                'predicted_winner': predicted_winner,
                'confidence_level': confidence
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _validate_fighter_features(self, features, fighter_label):
        """Validate fighter features against expected format"""
        # Load expected feature names
        expected_features = pd.read_json(self.feature_names_path, typ='series').tolist()
        
        # Extract prefix from first feature to determine expected prefix
        expected_prefix = expected_features[0].split('_')[0]
        
        # Filter expected features for this fighter
        if fighter_label == "fighter1":
            expected_fighter_features = [f for f in expected_features if f.startswith("fighter1_")]
        else:  # fighter2
            # Need to map fighter2 features to corresponding fighter1 features in the model
            expected_fighter_features = [f for f in expected_features if f.startswith("fighter2_")]
        
        # Check if all required features are present
        for feature in expected_fighter_features:
            # Get the feature name without the fighter prefix
            feature_name = '_'.join(feature.split('_')[1:])
            if feature_name not in features:
                raise ValueError(f"Missing required feature '{feature_name}' for {fighter_label}")
        
        # Check if there are any unexpected features
        for feature in features:
            feature_with_prefix = f"{fighter_label}_{feature}"
            if fighter_label == "fighter1":
                if f"fighter1_{feature}" not in expected_features:
                    logger.warning(f"Unexpected feature '{feature}' for {fighter_label}")
            else:
                # For fighter2, check if corresponding fighter1 feature exists
                fighter1_feature = f"fighter1_{feature}"
                if fighter1_feature not in expected_features:
                    logger.warning(f"Unexpected feature '{feature}' for {fighter_label}")

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
#Testing method 
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
    import pandas as pd
    import os
    import logging
    import joblib
    import torch
    from config import MODEL_PATH
    
    # Set up logging
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING)
    logger = logging.getLogger('ufc_prediction_test')
    
    try:
        # Import required classes
        from model import ModelTrainer, FightPredictor
        
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
        
        # Validate feature completeness
        logger.info("Validating fighter data")
        validate_fighter_data(red_fighter_data, "Red corner")
        validate_fighter_data(blue_fighter_data, "Blue corner")
        
        # Get the model directory path
        model_dir = os.path.dirname(model_path) if model_path else os.path.dirname(MODEL_PATH)
        
        # Load the input size parameter saved during training
        input_size_file = os.path.join(model_dir, "input_size.pkl")
        if os.path.exists(input_size_file):
            input_size = joblib.load(input_size_file)
            logger.info(f"Loaded input size from file: {input_size}")
        else:
            # Default to the value from error message if file doesn't exist
            input_size = 134
            logger.warning(f"Input size file not found. Using default: {input_size}")
            
        # Initialize model with correct input size
        model = FightPredictor(input_size=input_size)
        
        # Load the model weights
        model_file = model_path if model_path else MODEL_PATH
        logger.info(f"Loading model from: {model_file}")
        model.load_state_dict(torch.load(model_file))
        model.eval()  # Set to evaluation mode
        
        # Initialize the trainer but set the model directly
        trainer = ModelTrainer(None)  # No data needed for prediction
        trainer.model = model
        
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
                import matplotlib.pyplot as plt
                
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
            except:
                logger.warning("Matplotlib not available, skipping visualization")
        
        return result
        
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

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
    import pandas as pd
    import os
    
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
    except:
        print("Could not save fighter comparison to CSV")
    
    return result

import pandas as pd
import os
import traceback
from config import MODEL_PATH  # Ensure correct model path is used

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

    # ✅ Use sample test cases if none provided
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
    
    # ✅ Run batch tests
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
            traceback.print_exc()
            continue  # Move to the next test case
        
        # ✅ Prepare result record
        result_record = {
            'test_id': i+1,
            'description': case.get('description', 'Unnamed test'),
            'predicted_winner': prediction.get('predicted_winner', 'Unknown'),
            'red_win_probability': prediction.get('probability_red_wins', 0),
            'blue_win_probability': prediction.get('probability_blue_wins', 0),
            'confidence': prediction.get('confidence_level', 'Unknown')
        }
        
        # ✅ Add actual winner if provided
        if 'actual_winner' in case:
            result_record['actual_winner'] = case['actual_winner']
            result_record['prediction_correct'] = (case['actual_winner'] == prediction['predicted_winner'])
        
        results.append(result_record)
        
        # ✅ Print individual result
        print(f"📊 Prediction: {prediction['predicted_winner']} (Confidence: {prediction['confidence_level']})")
        if 'actual_winner' in case:
            correct = case['actual_winner'] == prediction['predicted_winner']
            print(f"✅ Actual: {case['actual_winner']} ({'✔️ CORRECT' if correct else '❌ INCORRECT'})")
    
    # ✅ Create results dataframe
    results_df = pd.DataFrame(results)
    
    # ✅ Generate summary statistics
    summary = {'total_tests': len(results)}
    
    if 'prediction_correct' in results_df.columns:
        summary['correct_predictions'] = results_df['prediction_correct'].sum()
        summary['accuracy'] = summary['correct_predictions'] / summary['total_tests']
        
        # ✅ Handle cases where confidence levels might not exist
        high_conf = results_df[results_df['confidence'] == 'High'] if 'confidence' in results_df.columns else pd.DataFrame()
        med_conf = results_df[results_df['confidence'] == 'Medium'] if 'confidence' in results_df.columns else pd.DataFrame()
        low_conf = results_df[results_df['confidence'] == 'Low'] if 'confidence' in results_df.columns else pd.DataFrame()
        
        if not high_conf.empty:
            summary['high_conf_accuracy'] = high_conf['prediction_correct'].mean() if 'prediction_correct' in high_conf else 'N/A'
            summary['high_conf_count'] = len(high_conf)
        
        if not med_conf.empty:
            summary['med_conf_accuracy'] = med_conf['prediction_correct'].mean() if 'prediction_correct' in med_conf else 'N/A'
            summary['med_conf_count'] = len(med_conf)
            
        if not low_conf.empty:
            summary['low_conf_accuracy'] = low_conf['prediction_correct'].mean() if 'prediction_correct' in low_conf else 'N/A'
            summary['low_conf_count'] = len(low_conf)
    
    # ✅ Print summary
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

    # ✅ Save results to CSV
    try:
        output_dir = os.path.dirname(MODEL_PATH)
        results_df.to_csv(os.path.join(output_dir, 'batch_test_results.csv'), index=False)
        print(f"📄 Test results saved to {os.path.join(output_dir, 'batch_test_results.csv')}")
    except Exception as e:
        print(f"⚠️ Could not save test results to CSV: {e}")
    
    return summary
def main():
    import os
    import pandas as pd
    import joblib
    import traceback
    from config import MODEL_PATH, CSV_FILE_PATH, SCALER_PATH
    # Import ModelTrainer and FightPredictor from model.py (which also contains test_prediction)
    from utils import preprocess_data

    print("🚀 Starting UFC fight prediction model training...")

    try:
        # Load raw data from CSV
        print(f"📂 Loading data from {CSV_FILE_PATH}")
        raw_data = pd.read_csv(CSV_FILE_PATH)
        print(f"✅ Successfully loaded raw data with shape: {raw_data.shape}")

        # Preprocess data to match the model's expected format
        print("⚙ Preprocessing data to match model format...")
        processed_data, scaler = preprocess_data(raw_data)
        print(f"✅ Preprocessed data shape: {processed_data.shape}")
        print(f"🔍 Columns after preprocessing: {list(processed_data.columns[:5])}...")

        # Ensure target column exists
        if 'fighter1_won' not in processed_data.columns:
            raise ValueError("❌ Error: Dataset must contain 'fighter1_won' column after preprocessing!")

        # Save the scaler for later use during inference
        joblib.dump(scaler, SCALER_PATH)
        print(f"📊 Scaler saved to {SCALER_PATH}")

        # Save training feature columns for consistent inference
        training_features = list(processed_data.columns)
        joblib.dump(training_features, os.path.join(os.path.dirname(MODEL_PATH), "training_features.pkl"))
        print("📊 Training feature columns saved.")

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
            # Call test_prediction directly from model.py
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
