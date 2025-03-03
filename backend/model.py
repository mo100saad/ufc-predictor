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
from config import MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, TEST_SIZE, VALIDATION_SIZE
import logging

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
                    val_predictions.extend(outputs.cpu().numpy())
                    val_targets.extend(labels.cpu().numpy())
            
            val_loss = val_loss / len(val_loader)
            
            # Calculate validation metrics
            val_predictions_binary = (np.array(val_predictions) >= 0.5).astype(int)
            val_targets = np.array(val_targets)
            
            val_accuracy = accuracy_score(val_targets, val_predictions_binary)
            val_precision = precision_score(val_targets, val_predictions_binary, zero_division=0)
            val_recall = recall_score(val_targets, val_predictions_binary, zero_division=0)
            val_f1 = f1_score(val_targets, val_predictions_binary, zero_division=0)
            val_auc = roc_auc_score(val_targets, val_predictions)
            
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
        """Evaluate model on the provided data loader"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for features, labels in data_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_targets = np.array(all_targets)
        
        # Binary predictions
        binary_preds = (all_predictions >= 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_targets, binary_preds),
            'precision': precision_score(all_targets, binary_preds, zero_division=0),
            'recall': recall_score(all_targets, binary_preds, zero_division=0),
            'f1_score': f1_score(all_targets, binary_preds, zero_division=0),
            'roc_auc': roc_auc_score(all_targets, all_predictions)
        }
        
        # Create confusion matrix
        cm = confusion_matrix(all_targets, binary_preds)
        
        # Calculate additional metrics
        true_positives = cm[1, 1]
        false_positives = cm[0, 1]
        true_negatives = cm[0, 0]
        false_negatives = cm[1, 0]
        
        metrics['specificity'] = true_negatives / (true_negatives + false_positives)
        metrics['positive_predictive_value'] = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
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
        
        # Load feature names for validation
        feature_names = pd.read_json(self.feature_names_path, typ='series').tolist()
        
        # Get input size from feature names
        input_size = len(feature_names)
        
        # Initialize model
        self.model = FightPredictor(input_size)
        
        # Load model weights
        model_path = self.best_model_path if use_best and os.path.exists(self.best_model_path) else MODEL_PATH
        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        
        return self.model, feature_names
    
    def predict_fight(self, fighter1_features, fighter2_features):
        """
        Predict the outcome of a fight between two fighters
        
        Parameters:
        fighter1_features: Dict of features for fighter 1
        fighter2_features: Dict of features for fighter 2
        
        Returns:
        dict: Prediction results including probability and confidence metrics
        """
        if self.model is None:
            self.model, expected_features = self.load_model()
        
        logger.info("Predicting fight outcome")
        
        try:
            # Validate input features
            self._validate_fighter_features(fighter1_features, "fighter1")
            self._validate_fighter_features(fighter2_features, "fighter2")
            
            # Combine features
            combined_features = {**fighter1_features, **fighter2_features}
            
            # Convert to DataFrame with same structure as training data
            df = pd.DataFrame([combined_features])
            
            # Scale features
            scaled_features = self.scaler.transform(df)
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(scaled_features).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                probability = self.model(features_tensor).item()
            
            # Determine confidence level
            confidence_level = "High" if abs(probability - 0.5) > 0.3 else \
                              "Medium" if abs(probability - 0.5) > 0.15 else \
                              "Low"
            
            # Create result dictionary
            result = {
                "probability_fighter1_wins": probability,
                "probability_fighter2_wins": 1 - probability,
                "predicted_winner": "Fighter 1" if probability >= 0.5 else "Fighter 2",
                "confidence_level": confidence_level
            }
            
            logger.info(f"Prediction result: {result}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}")
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

def example_usage():
    """Example of how to use the UFC model"""
    # 1. Create a config.py file first with these parameters:
    """
    MODEL_PATH = 'models/ufc_model.pth'
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 100
    TEST_SIZE = 0.2
    VALIDATION_SIZE = 0.1
    """
    
    # 2. Load your data
    data = pd.read_csv('ufc_fight_data.csv')
    
    # 3. Train the model
    trainer = ModelTrainer(data)
    history, test_metrics = trainer.train_model()
    
    # 4. Plot training history
    trainer.plot_training_history()
    
    # 5. Make a prediction
    fighter1 = {
        'age': 30,
        'height': 180,
        'weight': 77,
        'reach': 188,
        'stance': 1,  # Encoded categorical value
        'wins': 15,
        'losses': 2,
        'win_streak': 3,
        # ... other features
    }
    
    fighter2 = {
        'age': 32,
        'height': 178,
        'weight': 77,
        'reach': 182,
        'stance': 0,  # Encoded categorical value
        'wins': 12,
        'losses': 4,
        'win_streak': 1,
        # ... other features
    }
    
    result = trainer.predict_fight(fighter1, fighter2)
    print(f"Prediction: {result}")