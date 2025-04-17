import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.feature_selection import SelectFromModel

# Constants for position bias mitigation
POSITION_SWAP_WEIGHT = 1.0  # Weight for position-swapped predictions (0.0-1.0)

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
    UFC Fight Predictor Neural Network - More robust architecture
    """
    def __init__(self, input_size, hidden_size=128, dropout_rate=0.3):
        super(UFCPredictor, self).__init__()
        
        # Input layer with batch normalization
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Hidden layers with batch normalization
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(hidden_size // 2, hidden_size // 4)
        self.bn3 = nn.BatchNorm1d(hidden_size // 4)
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.output = nn.Linear(hidden_size // 4, 1)
        
        # Initialize weights using Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.output.weight)
    
    def forward(self, x):
        # First hidden layer with ReLU activation, batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Third hidden layer
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer with sigmoid activation
        x = torch.sigmoid(self.output(x))
        
        return x

class FocalLoss(nn.Module):
    """
    Focal Loss implementation to handle class imbalance
    
    Focal Loss increases the contribution of hard-to-classify examples by
    reducing the importance of easy-to-classify examples.
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCELoss(reduction='none')
        
    def forward(self, inputs, targets):
        BCE_loss = self.bce(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()

class EnsemblePredictor:
    """
    Ensemble predictor that combines multiple models
    """
    def __init__(self, models=None, weights=None):
        self.models = models or []
        self.weights = weights or []
        
        # Normalize weights if provided
        if self.weights:
            total = sum(self.weights)
            self.weights = [w / total for w in self.weights]
    
    def add_model(self, model, weight=1.0):
        """Add a model to the ensemble"""
        self.models.append(model)
        self.weights.append(weight)
        
        # Normalize weights
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
    
    def predict(self, X):
        """
        Make a weighted ensemble prediction
        
        Args:
            X: Input features
            
        Returns:
            numpy.ndarray: Weighted prediction probabilities
        """
        if not self.models:
            raise ValueError("No models in ensemble")
        
        # Get predictions from each model
        preds = []
        for i, model in enumerate(self.models):
            if isinstance(model, nn.Module):
                # Neural network model
                device = next(model.parameters()).device
                X_tensor = torch.FloatTensor(X).to(device)
                model.eval()
                with torch.no_grad():
                    pred = model(X_tensor).cpu().numpy()
                preds.append(pred)
            elif hasattr(model, 'predict_proba'):
                # Scikit-learn model with predict_proba
                pred = model.predict_proba(X)[:, 1].reshape(-1, 1)
                preds.append(pred)
            else:
                # Other model types
                pred = model.predict(X).reshape(-1, 1)
                preds.append(pred)
        
        # Weighted average of predictions
        weighted_preds = np.zeros_like(preds[0])
        for i, pred in enumerate(preds):
            weighted_preds += pred * self.weights[i]
        
        return weighted_preds

def enhanced_feature_engineering(df):
    """
    Enhanced feature engineering for UFC fight prediction
    
    This function creates advanced features from the dataset to improve
    model performance and reduce bias
    
    Args:
        df (pd.DataFrame): UFC dataset
        
    Returns:
        pd.DataFrame: Enhanced dataset with additional features
    """
    logger.info("Performing enhanced feature engineering")
    
    # Make a copy of the dataframe
    df_enhanced = df.copy()
    
    # 1. Keep the difference columns as they're crucial for position bias reduction
    diff_columns = [col for col in df_enhanced.columns if '_diff' in col]
    logger.info(f"Using {len(diff_columns)} difference columns for bias reduction")
    
    # 2. Create additional features based on the UFC dataset structure
    
    # Win rate features
    if all(col in df_enhanced.columns for col in ['r_wins_total', 'r_losses_total']):
        # Red corner win rate
        total_fights_r = df_enhanced['r_wins_total'] + df_enhanced['r_losses_total']
        df_enhanced['r_win_rate'] = df_enhanced['r_wins_total'] / total_fights_r.replace(0, 1)
        
        # Blue corner win rate
        total_fights_b = df_enhanced['b_wins_total'] + df_enhanced['b_losses_total']
        df_enhanced['b_win_rate'] = df_enhanced['b_wins_total'] / total_fights_b.replace(0, 1)
        
        # Win rate difference
        df_enhanced['win_rate_diff'] = df_enhanced['r_win_rate'] - df_enhanced['b_win_rate']
    
    # 3. Combat style effectiveness metrics
    
    # Striking efficiency ratio
    if all(col in df_enhanced.columns for col in ['r_sig_str_acc_total', 'b_str_def_total']):
        # For red corner
        df_enhanced['r_striking_efficiency'] = df_enhanced['r_sig_str_acc_total'] * (1 - df_enhanced['b_str_def_total'])
        # For blue corner
        df_enhanced['b_striking_efficiency'] = df_enhanced['b_sig_str_acc_total'] * (1 - df_enhanced['r_str_def_total'])
        # Difference
        df_enhanced['striking_efficiency_diff'] = df_enhanced['r_striking_efficiency'] - df_enhanced['b_striking_efficiency']
    
    # Grappling efficiency ratio
    if all(col in df_enhanced.columns for col in ['r_td_acc_total', 'b_td_def_total']):
        # For red corner
        df_enhanced['r_grappling_efficiency'] = df_enhanced['r_td_acc_total'] * (1 - df_enhanced['b_td_def_total'])
        # For blue corner
        df_enhanced['b_grappling_efficiency'] = df_enhanced['b_td_acc_total'] * (1 - df_enhanced['r_td_def_total'])
        # Difference
        df_enhanced['grappling_efficiency_diff'] = df_enhanced['r_grappling_efficiency'] - df_enhanced['b_grappling_efficiency']
    
    # 4. Fighting style indicators
    
    # Striker vs Grappler indicator
    if all(col in df_enhanced.columns for col in ['r_SLpM_total', 'r_td_avg']):
        df_enhanced['r_striker_score'] = df_enhanced['r_SLpM_total'] / (df_enhanced['r_td_avg'] + 0.1)
        df_enhanced['b_striker_score'] = df_enhanced['b_SLpM_total'] / (df_enhanced['b_td_avg'] + 0.1)
        df_enhanced['style_matchup_diff'] = df_enhanced['r_striker_score'] - df_enhanced['b_striker_score']
    
    # 5. Defense vs Offense metrics
    
    # Defense-to-offense ratio
    if all(col in df_enhanced.columns for col in ['r_str_def_total', 'r_sig_str_acc_total']):
        df_enhanced['r_defense_offense_ratio'] = df_enhanced['r_str_def_total'] / df_enhanced['r_sig_str_acc_total'].replace(0, 0.01)
        df_enhanced['b_defense_offense_ratio'] = df_enhanced['b_str_def_total'] / df_enhanced['b_sig_str_acc_total'].replace(0, 0.01)
        df_enhanced['defense_offense_diff'] = df_enhanced['r_defense_offense_ratio'] - df_enhanced['b_defense_offense_ratio']
    
    # 6. Physical advantage composite metrics
    
    # Create a composite physical advantage metric
    if all(col in df_enhanced.columns for col in ['height_diff', 'reach_diff', 'weight_diff']):
        # Normalize the differences
        height_diff_norm = df_enhanced['height_diff'] / df_enhanced['height_diff'].abs().max()
        reach_diff_norm = df_enhanced['reach_diff'] / df_enhanced['reach_diff'].abs().max()
        weight_diff_norm = df_enhanced['weight_diff'] / df_enhanced['weight_diff'].abs().max()
        
        # Composite physical advantage (weighted sum)
        df_enhanced['physical_advantage'] = (height_diff_norm * 0.3) + (reach_diff_norm * 0.4) + (weight_diff_norm * 0.3)
    
    # 7. Experience features
    
    # Experience difference (total fights)
    if all(col in df_enhanced.columns for col in ['r_wins_total', 'r_losses_total', 'b_wins_total', 'b_losses_total']):
        df_enhanced['r_total_fights'] = df_enhanced['r_wins_total'] + df_enhanced['r_losses_total']
        df_enhanced['b_total_fights'] = df_enhanced['b_wins_total'] + df_enhanced['b_losses_total']
        df_enhanced['experience_diff'] = df_enhanced['r_total_fights'] - df_enhanced['b_total_fights']
    
    # 8. Feature interactions
    
    # Striking x Takedown interaction (helps model learn complex relationships)
    if all(col in df_enhanced.columns for col in ['r_SLpM_total', 'r_td_avg']):
        df_enhanced['r_strike_td_interaction'] = df_enhanced['r_SLpM_total'] * df_enhanced['r_td_avg']
        df_enhanced['b_strike_td_interaction'] = df_enhanced['b_SLpM_total'] * df_enhanced['b_td_avg']
        df_enhanced['strike_td_interaction_diff'] = df_enhanced['r_strike_td_interaction'] - df_enhanced['b_strike_td_interaction']
    
    # 9. Fight effectiveness score - combining striking and grappling
    
    if all(col in df_enhanced.columns for col in ['r_SLpM_total', 'r_td_avg', 'r_sub_avg']):
        # Calculate overall effectiveness score (weighted sum of striking, takedowns, and submissions)
        df_enhanced['r_effectiveness'] = df_enhanced['r_SLpM_total']*0.6 + df_enhanced['r_td_avg']*0.3 + df_enhanced['r_sub_avg']*0.1
        df_enhanced['b_effectiveness'] = df_enhanced['b_SLpM_total']*0.6 + df_enhanced['b_td_avg']*0.3 + df_enhanced['b_sub_avg']*0.1
        df_enhanced['effectiveness_diff'] = df_enhanced['r_effectiveness'] - df_enhanced['b_effectiveness']
    
    # 10. Normalize some of the new features to prevent extreme values
    
    new_features = [col for col in df_enhanced.columns if col not in df.columns]
    for col in new_features:
        if df_enhanced[col].dtype != 'object' and not pd.isna(df_enhanced[col]).all():  # Only normalize numeric columns
            # Skip columns that are already normalized or binary
            if df_enhanced[col].min() >= -1 and df_enhanced[col].max() <= 1:
                continue
                
            # Apply robust normalization with outlier capping
            q1, q3 = df_enhanced[col].quantile(0.01), df_enhanced[col].quantile(0.99)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            capped = df_enhanced[col].clip(lower_bound, upper_bound)
            
            # Standardize to zero mean and unit variance
            mean = capped.mean()
            std = capped.std()
            if std > 0:  # Avoid division by zero
                df_enhanced[col] = (capped - mean) / std
    
    logger.info(f"Enhanced feature engineering complete. Added {len(new_features)} new features")
    return df_enhanced

def select_optimal_features(X, y, threshold=0.005, max_features=None):
    """
    Select the most important features using multiple methods
    
    Args:
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        threshold (float): Importance threshold for feature selection
        max_features (int): Maximum number of features to select
        
    Returns:
        list: Selected feature names
    """
    logger.info("Selecting optimal features")
    
    # Initialize feature importance dictionary
    feature_importance = {col: 0 for col in X.columns}
    
    # Method 1: Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    rf_importance = dict(zip(X.columns, rf.feature_importances_))
    for feature, importance in rf_importance.items():
        feature_importance[feature] += importance
    
    # Method 2: XGBoost importance
    xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb_model.fit(X, y)
    
    xgb_importance = dict(zip(X.columns, xgb_model.feature_importances_))
    for feature, importance in xgb_importance.items():
        feature_importance[feature] += importance
    
    # Method 3: Gradient Boosting importance
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X, y)
    
    gb_importance = dict(zip(X.columns, gb.feature_importances_))
    for feature, importance in gb_importance.items():
        feature_importance[feature] += importance
    
    # Normalize feature importance
    total_importance = sum(feature_importance.values())
    if total_importance > 0:
        feature_importance = {feature: importance / total_importance 
                           for feature, importance in feature_importance.items()}
    
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Select features above threshold
    selected_features = [feature for feature, importance in sorted_features 
                       if importance > threshold]
    
    # Apply max_features limit if specified
    if max_features and len(selected_features) > max_features:
        selected_features = selected_features[:max_features]
    
    logger.info(f"Selected {len(selected_features)} optimal features out of {len(X.columns)}")
    
    # Log top 20 features
    top_features = sorted_features[:20]
    feature_info = "\n".join([f"{i+1}. {feature}: {importance:.4f}" 
                           for i, (feature, importance) in enumerate(top_features)])
    logger.info(f"Top 20 features by importance:\n{feature_info}")
    
    return selected_features

def train_ensemble_model(X_train, y_train, X_val, y_val):
    """
    Train an ensemble of models for UFC fight prediction
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame): Validation features
        y_val (pd.Series): Validation target
        
    Returns:
        EnsemblePredictor: Trained ensemble model
    """
    logger.info("Training ensemble model")
    
    # Initialize ensemble
    ensemble = EnsemblePredictor()
    
    # Train XGBoost model
    logger.info("Training XGBoost model")
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        eval_metric='auc',  # Move eval_metric here as a model parameter
        random_state=42,
        n_jobs=-1
    )
    
    # Train without eval_set
    xgb_model.fit(X_train, y_train)
    
    # Evaluate XGBoost model
    y_pred_xgb = xgb_model.predict(X_val)
    xgb_acc = accuracy_score(y_val, y_pred_xgb)
    xgb_auc = roc_auc_score(y_val, xgb_model.predict_proba(X_val)[:, 1])
    logger.info(f"XGBoost - Val Accuracy: {xgb_acc:.4f}, AUC: {xgb_auc:.4f}")
    
    # Add XGBoost model to ensemble
    ensemble.add_model(xgb_model, weight=2.5)
    
    # Train Random Forest model
    logger.info("Training Random Forest model")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Random Forest model
    y_pred_rf = rf_model.predict(X_val)
    rf_acc = accuracy_score(y_val, y_pred_rf)
    rf_auc = roc_auc_score(y_val, rf_model.predict_proba(X_val)[:, 1])
    logger.info(f"Random Forest - Val Accuracy: {rf_acc:.4f}, AUC: {rf_auc:.4f}")
    
    # Add Random Forest model to ensemble
    ensemble.add_model(rf_model, weight=1.5)
    
    # Train Gradient Boosting model
    logger.info("Training Gradient Boosting model")
    gb_model = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=5,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42
    )
    gb_model.fit(X_train, y_train)
    
    # Evaluate Gradient Boosting model
    y_pred_gb = gb_model.predict(X_val)
    gb_acc = accuracy_score(y_val, y_pred_gb)
    gb_auc = roc_auc_score(y_val, gb_model.predict_proba(X_val)[:, 1])
    logger.info(f"Gradient Boosting - Val Accuracy: {gb_acc:.4f}, AUC: {gb_auc:.4f}")
    
    # Add Gradient Boosting model to ensemble
    ensemble.add_model(gb_model, weight=1.0)
    
    # Train Logistic Regression model
    logger.info("Training Logistic Regression model")
    lr_model = LogisticRegression(
        C=1.0,
        penalty='l2',
        solver='liblinear',
        random_state=42,
        max_iter=1000
    )
    lr_model.fit(X_train, y_train)
    
    # Evaluate Logistic Regression model
    y_pred_lr = lr_model.predict(X_val)
    lr_acc = accuracy_score(y_val, y_pred_lr)
    lr_auc = roc_auc_score(y_val, lr_model.predict_proba(X_val)[:, 1])
    logger.info(f"Logistic Regression - Val Accuracy: {lr_acc:.4f}, AUC: {lr_auc:.4f}")
    
    # Add Logistic Regression model to ensemble
    ensemble.add_model(lr_model, weight=0.5)
    
    # Evaluate ensemble
    ensemble_preds = ensemble.predict(X_val)
    ensemble_preds_binary = (ensemble_preds > 0.5).astype(int)
    ensemble_acc = accuracy_score(y_val, ensemble_preds_binary)
    ensemble_auc = roc_auc_score(y_val, ensemble_preds)
    
    logger.info(f"Ensemble - Val Accuracy: {ensemble_acc:.4f}, AUC: {ensemble_auc:.4f}")
    
    return ensemble

def train_pytorch_model(X_train, y_train, X_val, y_val, params=None):
    """
    Train a PyTorch neural network model
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training target
        X_val (np.ndarray): Validation features
        y_val (np.ndarray): Validation target
        params (dict): Training parameters
        
    Returns:
        tuple: (trained model, training history)
    """
    logger.info("Training PyTorch neural network model")
    
    # Default parameters
    default_params = {
        'hidden_size': 128,
        'dropout_rate': 0.3,
        'batch_size': 64,
        'learning_rate': 0.001,
        'weight_decay': 0.0005,
        'epochs': 100,
        'patience': 15,
        'focal_loss': True,
        'focal_alpha': 0.25,
        'focal_gamma': 2.0,
        'pos_weight': None
    }
    
    # Update with provided parameters
    if params:
        default_params.update(params)
    
    params = default_params
    
    # Create model
    input_size = X_train.shape[1]
    model = UFCPredictor(
        input_size=input_size,
        hidden_size=params['hidden_size'],
        dropout_rate=params['dropout_rate']
    )
    
    # Create dataloaders
    train_tensor_x = torch.FloatTensor(X_train)
    train_tensor_y = torch.FloatTensor(y_train.values.reshape(-1, 1))
    val_tensor_x = torch.FloatTensor(X_val)
    val_tensor_y = torch.FloatTensor(y_val.values.reshape(-1, 1))
    
    train_dataset = torch.utils.data.TensorDataset(train_tensor_x, train_tensor_y)
    val_dataset = torch.utils.data.TensorDataset(val_tensor_x, val_tensor_y)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=params['batch_size'], shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=params['batch_size']
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Define loss function
    if params['focal_loss']:
        criterion = FocalLoss(
            alpha=params['focal_alpha'],
            gamma=params['focal_gamma']
        )
        logger.info("Using Focal Loss")
    elif params['pos_weight'] is not None:
        pos_weight = torch.tensor([params['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"Using weighted BCE loss with pos_weight={params['pos_weight']:.4f}")
    else:
        criterion = nn.BCELoss()
        logger.info("Using BCE Loss")
    
    # Define optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=params['learning_rate'],
        weight_decay=params['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_auc': [],
        'lr': []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_model_state = None
    no_improve_epoch = 0
    
    # Training loop
    for epoch in range(params['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        val_loss /= len(val_loader.dataset)
        
        # Calculate metrics
        val_preds = np.array(val_preds).flatten()
        val_targets = np.array(val_targets).flatten()
        
        val_preds_binary = (val_preds > 0.5).astype(int)
        val_acc = accuracy_score(val_targets, val_preds_binary)
        val_auc = roc_auc_score(val_targets, val_preds)
        
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_auc'].append(val_auc)
        history['lr'].append(current_lr)
        
        # Print progress
        logger.info(f"Epoch {epoch+1}/{params['epochs']} - "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}, "
                   f"Val AUC: {val_auc:.4f}, "
                   f"LR: {current_lr:.6f}")
        
        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= params['patience']:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model state
    model.load_state_dict(best_model_state)
    
    return model, history

def predict_fight(model, fighter1_data, fighter2_data, scaler, feature_columns, is_pytorch=None, position_swap_mitigation=True):
    """
    Predict the outcome of a UFC fight using either PyTorch or scikit-learn model
    
    Args:
        model: Trained model (can be PyTorch neural network or scikit-learn model)
        fighter1_data (dict): Fighter 1 statistics
        fighter2_data (dict): Fighter 2 statistics
        scaler (StandardScaler): Fitted scaler
        feature_columns (list): List of feature column names
        is_pytorch (bool): Whether the model is a PyTorch model (optional)
        position_swap_mitigation (bool): Whether to mitigate position bias by averaging swapped predictions
        
    Returns:
        dict: Prediction results
    """
    try:
        # Log input data for debugging
        logger.info(f"predict_fight called with fighter1: {fighter1_data.get('name', 'Unknown')}")
        logger.info(f"predict_fight called with fighter2: {fighter2_data.get('name', 'Unknown')}")
        
        # Import needed libraries
        import pandas as pd
        import numpy as np
        
        # Better check if model is PyTorch
        if is_pytorch is None:
            try:
                is_pytorch = hasattr(model, 'parameters') and callable(model.parameters)
            except:
                is_pytorch = False
        
        # Put model in eval mode if it's PyTorch
        if is_pytorch:
            try:
                import torch
                device = next(model.parameters()).device
                model.eval()
            except Exception as e:
                logger.warning(f"Error setting PyTorch model to eval mode: {e}")
                is_pytorch = False  # Fall back to non-PyTorch handling
        
        # Define a function to prepare a single prediction
        def prepare_prediction(f1_data, f2_data):
            # Prepare features in the correct format
            features = {}
            
            # Add fighter1 features
            for key, value in f1_data.items():
                if value is None:
                    continue
                    
                if not key.startswith('fighter1_'):
                    features[f'fighter1_{key}'] = value
                else:
                    features[key] = value
            
            # Add fighter2 features
            for key, value in f2_data.items():
                if value is None:
                    continue
                    
                if not key.startswith('fighter2_'):
                    features[f'fighter2_{key}'] = value
                else:
                    features[key] = value
            
            # Calculate advantage features
            if 'fighter1_height' in features and 'fighter2_height' in features:
                try:
                    features['height_diff'] = float(features['fighter1_height']) - float(features['fighter2_height'])
                except (ValueError, TypeError):
                    features['height_diff'] = 0
            else:
                features['height_diff'] = 0
            
            if 'fighter1_reach' in features and 'fighter2_reach' in features:
                try:
                    features['reach_diff'] = float(features['fighter1_reach']) - float(features['fighter2_reach'])
                except (ValueError, TypeError):
                    features['reach_diff'] = 0
            else:
                features['reach_diff'] = 0
            
            if 'fighter1_weight' in features and 'fighter2_weight' in features:
                try:
                    features['weight_diff'] = float(features['fighter1_weight']) - float(features['fighter2_weight'])
                except (ValueError, TypeError):
                    features['weight_diff'] = 0
            else:
                features['weight_diff'] = 0
            
            # Win percentage advantage
            try:
                f1_wins = float(features.get('fighter1_wins', 0))
                f1_losses = float(features.get('fighter1_losses', 0))
                f2_wins = float(features.get('fighter2_wins', 0))
                f2_losses = float(features.get('fighter2_losses', 0))
                
                fighter1_total = f1_wins + f1_losses
                fighter2_total = f2_wins + f2_losses
                
                fighter1_win_rate = f1_wins / fighter1_total if fighter1_total > 0 else 0
                fighter2_win_rate = f2_wins / fighter2_total if fighter2_total > 0 else 0
                
                features['win_rate_diff'] = fighter1_win_rate - fighter2_win_rate
            except:
                features['win_rate_diff'] = 0
            
            # More differences
            try:
                features['td_avg_diff'] = float(features.get('fighter1_td_avg', 0)) - float(features.get('fighter2_td_avg', 0))
            except:
                features['td_avg_diff'] = 0
                
            try:
                features['SLpM_diff'] = float(features.get('fighter1_SLpM', 0)) - float(features.get('fighter2_SLpM', 0))
            except:
                features['SLpM_diff'] = 0
                
            try:
                features['sub_avg_diff'] = float(features.get('fighter1_sub_avg', 0)) - float(features.get('fighter2_sub_avg', 0))
            except:
                features['sub_avg_diff'] = 0
                
            try:
                f1_sApM = float(features.get('fighter1_SApM', 0))
                f2_sApM = float(features.get('fighter2_SApM', 0))
                features['SApM_diff'] = f1_sApM - f2_sApM
            except:
                features['SApM_diff'] = 0
                
            try:
                # Defense effectiveness comparison
                f1_def = float(features.get('fighter1_str_def', 0))
                f2_def = float(features.get('fighter2_str_def', 0))
                features['str_def_diff'] = f1_def - f2_def
            except:
                features['str_def_diff'] = 0
                
            try:
                # Takedown defense comparison
                f1_td_def = float(features.get('fighter1_td_def', 0))
                f2_td_def = float(features.get('fighter2_td_def', 0))
                features['td_def_diff'] = f1_td_def - f2_td_def
            except:
                features['td_def_diff'] = 0
                
            try:
                # Experience difference (total fights)
                f1_wins = float(features.get('fighter1_wins', 0))
                f1_losses = float(features.get('fighter1_losses', 0))
                f2_wins = float(features.get('fighter2_wins', 0))
                f2_losses = float(features.get('fighter2_losses', 0))
                
                f1_fights = f1_wins + f1_losses
                f2_fights = f2_wins + f2_losses
                
                features['experience_diff'] = f1_fights - f2_fights
            except:
                features['experience_diff'] = 0
            
            # Get stances for one-hot encoding if needed
            fighter1_stance = f1_data.get('stance', 'Unknown')
            fighter2_stance = f2_data.get('stance', 'Unknown')
            
            # Try to load the stance encoder if available
            try:
                import joblib
                stance_encoder = joblib.load('models/stance_encoder.joblib')
                
                # Encode stances
                stance_df = pd.DataFrame({
                    'fighter1_stance': [fighter1_stance if fighter1_stance else 'Unknown'],
                    'fighter2_stance': [fighter2_stance if fighter2_stance else 'Unknown']
                })
                
                stance_encoded = stance_encoder.transform(stance_df)
                
                # Get stance feature names
                stance_feature_names = []
                for i, category in enumerate(stance_encoder.categories_):
                    prefix = f"fighter{i+1}_stance"
                    stance_feature_names.extend([f"{prefix}_{stance}" for stance in category])
                
                # Add encoded stances to features
                for i, col in enumerate(stance_feature_names):
                    if i < stance_encoded.shape[1]:  # Safety check
                        features[col] = stance_encoded[0, i]
                
            except Exception as e:
                logger.warning(f"Could not load stance encoder: {e}")
                # Continue without stance encoding
            
            # Create a dataframe with exactly the columns the model expects
            input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
            
            # Fill in the features we have
            for col in feature_columns:
                if col in features:
                    try:
                        input_df[col] = float(features[col])
                    except (ValueError, TypeError):
                        # Keep default of 0 for non-numeric values
                        logger.warning(f"Non-numeric value for {col}: {features[col]}")
            
            # Scale the features
            scaled_features = scaler.transform(input_df)
            
            return scaled_features
        
        # Function to get a single prediction
        def get_win_probability(scaled_features):
            if is_pytorch:
                # PyTorch prediction
                import torch
                features_tensor = torch.FloatTensor(scaled_features).to(device)
                with torch.no_grad():
                    win_probability = model(features_tensor).item()
            else:
                # scikit-learn prediction
                if hasattr(model, 'predict_proba'):
                    win_probability = model.predict_proba(scaled_features)[0, 1]
                else:
                    # If model doesn't have predict_proba, use predict and assume binary output
                    prediction = model.predict(scaled_features)[0]
                    win_probability = float(prediction)
            
            return float(win_probability)
        
        # Prepare normal prediction (fighter1 in red corner)
        normal_features = prepare_prediction(fighter1_data, fighter2_data)
        normal_prob = get_win_probability(normal_features)
        
        # If position_swap_mitigation is True, also predict with positions swapped
        # and use a weighted average to reduce position bias
        final_prob = normal_prob
        
        if position_swap_mitigation:
            try:
                # Prepare swapped prediction (fighter2 in red corner)
                swapped_features = prepare_prediction(fighter2_data, fighter1_data)
                swapped_prob = get_win_probability(swapped_features)
                
                # The probability of fighter1 winning when in the blue corner
                # is 1 minus the probability of fighter2 winning when in the red corner
                adjusted_prob = 1.0 - swapped_prob
                
                # Weight for position-swapped prediction (from constants)
                swap_weight = POSITION_SWAP_WEIGHT
                
                # Calculate weighted average of the two probabilities
                # This helps mitigate position bias in the model
                final_prob = (normal_prob * (1.0 - swap_weight)) + (adjusted_prob * swap_weight)
                
                logger.info(f"Position bias mitigation: normal_prob={normal_prob:.4f}, " +
                            f"adjusted_prob={adjusted_prob:.4f}, final_prob={final_prob:.4f}")
            except Exception as e:
                logger.warning(f"Error in position swap mitigation: {e}, using normal prediction")
                final_prob = normal_prob
        
        # Determine confidence level
        if abs(final_prob - 0.5) > 0.3:
            confidence = "High"
        elif abs(final_prob - 0.5) > 0.15:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Create result dictionary
        result = {
            'probability_fighter1_wins': float(final_prob),
            'probability_fighter2_wins': float(1 - final_prob),
            'predicted_winner': 'fighter1' if final_prob > 0.5 else 'fighter2',
            'confidence_level': confidence
        }
        
        logger.info(f"Prediction result: {result}")
        return result
        
    except Exception as e:
        import traceback
        logger.error(f"Error in predict_fight: {e}")
        logger.error(traceback.format_exc())
        # Return a fallback prediction in case of errors
        return {
            'probability_fighter1_wins': 0.5,
            'probability_fighter2_wins': 0.5,
            'predicted_winner': 'unknown',
            'confidence_level': 'Unknown (Error)',
            'error': str(e)
        }
    
     
def augment_with_position_swap(df):
    """
    Augment the training dataset by swapping red and blue corner positions
    
    This function performs position swapping to prevent red/blue corner bias,
    essentially doubling the dataset with each fight represented from both perspectives.
    
    Args:
        df (pd.DataFrame): Original dataframe with fight data
        
    Returns:
        pd.DataFrame: Augmented dataframe with position-swapped versions
    """
    logger.info(f"Starting position swap augmentation on dataset with {len(df)} rows")
    
    # Make a copy of the original data
    df_augmented = df.copy()
    
    # Create swapped version of the data
    df_swapped = df.copy()
    
    # Find all pairs of columns to swap (R_ with B_ or fighter1_ with fighter2_)
    r_columns = [col for col in df.columns if col.startswith('r_') or col.startswith('fighter1_') or col.startswith('R_')]
    b_columns = [col for col in df.columns if col.startswith('b_') or col.startswith('fighter2_') or col.startswith('B_')]
    
    # Create mapping for column pairs
    r_to_b_map = {}
    for r_col in r_columns:
        # Handle different prefix styles
        if r_col.startswith('r_'):
            b_col = r_col.replace('r_', 'b_')
        elif r_col.startswith('fighter1_'):
            b_col = r_col.replace('fighter1_', 'fighter2_')
        elif r_col.startswith('R_'):
            b_col = r_col.replace('R_', 'B_')
            
        if b_col in b_columns:
            r_to_b_map[r_col] = b_col
    
    # Perform the swap
    for r_col, b_col in r_to_b_map.items():
        df_swapped[r_col], df_swapped[b_col] = df_swapped[b_col].copy(), df_swapped[r_col].copy()
    
    # Swap the target variable (invert)
    if 'fighter1_won' in df_swapped.columns:
        df_swapped['fighter1_won'] = 1 - df_swapped['fighter1_won']
    elif 'winner' in df_swapped.columns:
        # If 'winner' column is present, swap 'Red' and 'Blue'
        df_swapped['winner'] = df_swapped['winner'].map({'Red': 'Blue', 'Blue': 'Red', 'Draw': 'Draw'})
    elif 'Winner' in df_swapped.columns:
        df_swapped['Winner'] = df_swapped['Winner'].map({'Red': 'Blue', 'Blue': 'Red', 'Draw': 'Draw'})
    
    # Combine original and swapped data
    df_combined = pd.concat([df_augmented, df_swapped], ignore_index=True)
    
    # Handle advantage columns - these should be negative in the swapped version
    advantage_cols = [col for col in df_combined.columns if '_diff' in col.lower() or 'advantage' in col.lower()]
    for col in advantage_cols:
        # Negate the values in the second half of the dataframe (the swapped version)
        df_combined.loc[len(df_augmented):, col] = -df_combined.loc[len(df_augmented):, col]
    
    logger.info(f"Completed position swap augmentation. New dataset size: {len(df_combined)} rows")
    return df_combined

def prepare_training_data(df, test_size=0.2, val_size=0.15, random_seed=42):
    """
    Prepare training, validation, and test datasets with proper stratification
    
    Args:
        df (pd.DataFrame): UFC data
        test_size (float): Proportion of data for testing
        val_size (float): Proportion of data for validation
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    logger.info("Preparing training data")
    
    # Separate features and target
    if 'fighter1_won' in df.columns:
        target_col = 'fighter1_won'
    elif 'winner' in df.columns:
        df['fighter1_won'] = (df['winner'] == 'Red').astype(int)
        target_col = 'fighter1_won'
    elif 'Winner' in df.columns:
        df['fighter1_won'] = (df['Winner'] == 'Red').astype(int)
        target_col = 'fighter1_won'
    else:
        raise ValueError("No suitable target column found in the data")
    
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
    
    # Initialize and fit the scaler on training data only
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    
    # Transform validation and test data
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )
    
    logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def evaluate_model(model, X_test, y_test, scaler=None, is_pytorch=False):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained model (PyTorch or scikit-learn)
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        scaler (StandardScaler): Feature scaler (for PyTorch models)
        is_pytorch (bool): Whether the model is a PyTorch model
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model on test data")
    
    if is_pytorch:
        # For PyTorch models
        device = next(model.parameters()).device
        model.eval()
        
        # Convert to tensor
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)
        
        # Get predictions
        with torch.no_grad():
            test_preds_prob = model(X_test_tensor).cpu().numpy().flatten()
            test_preds = (test_preds_prob > 0.5).astype(int)
    else:
        # For scikit-learn models
        if hasattr(model, 'predict_proba'):
            test_preds_prob = model.predict_proba(X_test)[:, 1]
        else:
            test_preds_prob = model.predict(X_test)
        
        test_preds = (test_preds_prob > 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, test_preds),
        'precision': precision_score(y_test, test_preds),
        'recall': recall_score(y_test, test_preds),
        'f1': f1_score(y_test, test_preds),
        'auc': roc_auc_score(y_test, test_preds_prob)
    }
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, test_preds)
    
    # Print metrics
    logger.info(f"Test Metrics:")
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall: {metrics['recall']:.4f}")
    logger.info(f"F1 Score: {metrics['f1']:.4f}")
    logger.info(f"ROC AUC: {metrics['auc']:.4f}")
    logger.info(f"Confusion Matrix:\n{cm}")
    
    return metrics

def save_model(model, scaler, feature_columns, model_path, is_pytorch=False):
    """
    Save the model and associated data
    
    Args:
        model: Trained model (PyTorch or scikit-learn)
        scaler (StandardScaler): Feature scaler
        feature_columns (list): List of feature column names
        model_path (str): Path to save the model
        is_pytorch (bool): Whether the model is a PyTorch model
    """
    logger.info(f"Saving model to {model_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    # Save model
    if is_pytorch:
        torch.save(model.state_dict(), model_path)
    else:
        joblib.dump(model, model_path)
    
    # Save scaler
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
    joblib.dump(scaler, scaler_path)
    
    # Save feature columns
    feature_path = os.path.join(os.path.dirname(model_path), 'feature_columns.joblib')
    joblib.dump(feature_columns, feature_path)
    
    # Save model info
    info = {
        'is_pytorch': is_pytorch,
        'feature_count': len(feature_columns),
        'date_saved': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    info_path = os.path.join(os.path.dirname(model_path), 'model_info.joblib')
    joblib.dump(info, info_path)
    
    logger.info(f"Model and associated data saved successfully")

def load_model(model_path):
    """
    Load the model and associated data
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        tuple: (model, scaler, feature_columns, model_info)
    """
    logger.info(f"Loading model from {model_path}")
    
    # Load model info
    info_path = os.path.join(os.path.dirname(model_path), 'model_info.joblib')
    model_info = joblib.load(info_path)
    
    # Load model
    if model_info['is_pytorch']:
        # For PyTorch models
        input_size = model_info.get('input_size', len(joblib.load(os.path.join(os.path.dirname(model_path), 'feature_columns.joblib'))))
        model = UFCPredictor(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    else:
        # For scikit-learn models
        model = joblib.load(model_path)
    
    # Load scaler
    scaler_path = os.path.join(os.path.dirname(model_path), 'scaler.joblib')
    scaler = joblib.load(scaler_path)
    
    # Load feature columns
    feature_path = os.path.join(os.path.dirname(model_path), 'feature_columns.joblib')
    feature_columns = joblib.load(feature_path)
    
    logger.info(f"Model loaded successfully with {len(feature_columns)} features")
    return model, scaler, feature_columns, model_info