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
    import pandas as pd
    import logging
    
    logger = logging.getLogger('ufc_model')
    logger.info(f"Starting position swap augmentation on dataset with {len(df)} rows")
    
    # Make a copy of the original data
    df_augmented = df.copy()
    
    # Create swapped version of the data
    df_swapped = df.copy()
    
    # Find all pairs of columns to swap (R_ with B_ or fighter1_ with fighter2_)
    r_columns = [col for col in df.columns if col.startswith('R_') or col.startswith('fighter1_')]
    b_columns = [col for col in df.columns if col.startswith('B_') or col.startswith('fighter2_')]
    
    # Create mapping for column pairs
    r_to_b_map = {}
    for r_col in r_columns:
        # Handle both R_/B_ and fighter1_/fighter2_ prefixes
        if r_col.startswith('R_'):
            b_col = r_col.replace('R_', 'B_')
        else:  # fighter1_
            b_col = r_col.replace('fighter1_', 'fighter2_')
            
        if b_col in b_columns:
            r_to_b_map[r_col] = b_col
    
    # Perform the swap
    for r_col, b_col in r_to_b_map.items():
        df_swapped[r_col], df_swapped[b_col] = df_swapped[b_col].copy(), df_swapped[r_col].copy()
    
    # Swap the target variable (invert)
    if 'fighter1_won' in df_swapped.columns:
        df_swapped['fighter1_won'] = 1 - df_swapped['fighter1_won']
    elif 'Winner' in df_swapped.columns:
        # If 'Winner' column is present, swap 'Red' and 'Blue'
        df_swapped['Winner'] = df_swapped['Winner'].map({'Red': 'Blue', 'Blue': 'Red', 'Draw': 'Draw'})
    
    # Combine original and swapped data
    df_combined = pd.concat([df_augmented, df_swapped], ignore_index=True)
    
    # Handle advantage columns - these should be negative in the swapped version
    advantage_cols = [col for col in df_combined.columns if 'advantage' in col or 'difference' in col]
    for col in advantage_cols:
        # Negate the values in the second half of the dataframe (the swapped version)
        df_combined.loc[len(df_augmented):, col] = -df_combined.loc[len(df_augmented):, col]
    
    logger.info(f"Completed position swap augmentation. New dataset size: {len(df_combined)} rows")
    return df_combined

def verify_position_swap_effect(model, scaler, feature_columns):
    """
    Verify the effect of position swapping on model predictions
    
    Args:
        model (nn.Module): Trained model
        scaler (StandardScaler): Fitted scaler
        feature_columns (list): List of feature column names
        
    Returns:
        float: Average position bias (difference in predictions when positions are swapped)
    """
    import torch
    import numpy as np
    import pandas as pd
    import logging
    
    logger = logging.getLogger('ufc_model')
    logger.info("Verifying position swap effect")
    
    # Create some test fighter pairs
    test_pairs = []
    for i in range(10):
        fighter1 = {
            'wins': 10 + i,
            'losses': 5 - (i % 3),
            'height': 180,
            'reach': 185,
            'sig_strikes_per_min': 4.0 + (i * 0.1),
            'takedown_avg': 1.5 + (i * 0.2),
            'sub_avg': 0.5 + (i * 0.1),
        }
        
        fighter2 = {
            'wins': 8 + (i % 5),
            'losses': 4 + (i % 4),
            'height': 178,
            'reach': 182,
            'sig_strikes_per_min': 3.5 + (i * 0.15),
            'takedown_avg': 2.0 - (i * 0.1),
            'sub_avg': 0.8 - (i * 0.05),
        }
        
        test_pairs.append((fighter1, fighter2))
    
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    # Test each pair
    bias_values = []
    
    for fighter1, fighter2 in test_pairs:
        # Create features with fighter1 as red corner
        features_normal = {}
        for key, val in fighter1.items():
            features_normal[f'fighter1_{key}'] = val
        for key, val in fighter2.items():
            features_normal[f'fighter2_{key}'] = val
            
        # Add some advantage features
        if 'height' in fighter1 and 'height' in fighter2:
            features_normal['height_advantage'] = fighter1['height'] - fighter2['height']
        if 'reach' in fighter1 and 'reach' in fighter2:
            features_normal['reach_advantage'] = fighter1['reach'] - fighter2['reach']
        if 'wins' in fighter1 and 'losses' in fighter1 and 'wins' in fighter2 and 'losses' in fighter2:
            f1_total = fighter1['wins'] + fighter1['losses']
            f2_total = fighter2['wins'] + fighter2['losses']
            if f1_total > 0 and f2_total > 0:
                features_normal['fighter1_win_pct'] = fighter1['wins'] / f1_total
                features_normal['fighter2_win_pct'] = fighter2['wins'] / f2_total
                features_normal['win_pct_advantage'] = features_normal['fighter1_win_pct'] - features_normal['fighter2_win_pct']
            features_normal['experience_advantage'] = f1_total - f2_total
        
        # Create features with positions swapped
        features_swapped = {}
        for key, val in fighter2.items():
            features_swapped[f'fighter1_{key}'] = val
        for key, val in fighter1.items():
            features_swapped[f'fighter2_{key}'] = val
            
        # Add advantage features with opposite sign
        if 'height_advantage' in features_normal:
            features_swapped['height_advantage'] = -features_normal['height_advantage']
        if 'reach_advantage' in features_normal:
            features_swapped['reach_advantage'] = -features_normal['reach_advantage']
        if 'win_pct_advantage' in features_normal:
            features_swapped['fighter1_win_pct'] = features_normal['fighter2_win_pct']
            features_swapped['fighter2_win_pct'] = features_normal['fighter1_win_pct']
            features_swapped['win_pct_advantage'] = -features_normal['win_pct_advantage']
        if 'experience_advantage' in features_normal:
            features_swapped['experience_advantage'] = -features_normal['experience_advantage']
        
        # Create dataframes
        df_normal = pd.DataFrame([features_normal])
        df_swapped = pd.DataFrame([features_swapped])
        
        # Fill missing columns with zeros
        for col in feature_columns:
            if col not in df_normal.columns:
                df_normal[col] = 0
            if col not in df_swapped.columns:
                df_swapped[col] = 0
                
        # Keep only the columns in feature_columns
        df_normal = df_normal[feature_columns]
        df_swapped = df_swapped[feature_columns]
        
        # Scale the features
        normal_scaled = scaler.transform(df_normal)
        swapped_scaled = scaler.transform(df_swapped)
        
        # Convert to tensors
        normal_tensor = torch.FloatTensor(normal_scaled).to(device)
        swapped_tensor = torch.FloatTensor(swapped_scaled).to(device)
        
        # Get predictions
        with torch.no_grad():
            normal_pred = model(normal_tensor).item()
            swapped_pred = model(swapped_tensor).item()
        
        # Calculate bias (difference between normal and inverted swapped prediction)
        # If unbiased, normal_pred should equal 1 - swapped_pred
        bias = abs(normal_pred - (1 - swapped_pred))
        bias_values.append(bias)
        
        logger.info(f"Pair {len(bias_values)}: Normal={normal_pred:.4f}, Swapped={1-swapped_pred:.4f}, Bias={bias:.4f}")
    
    # Calculate average bias
    avg_bias = sum(bias_values) / len(bias_values)
    logger.info(f"Average position bias: {avg_bias:.4f}")
    
    return avg_bias

def correct_position_bias(model, dataset, validation_data=None, epochs=5):
    """
    Fine-tune model to correct position bias using position-swapped data
    
    Args:
        model (nn.Module): Model to fine-tune
        dataset (Dataset): Dataset of fight data
        validation_data (Dataset): Validation dataset
        epochs (int): Number of fine-tuning epochs
        
    Returns:
        nn.Module: Fine-tuned model
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import logging
    
    logger = logging.getLogger('ufc_model')
    logger.info("Starting position bias correction fine-tuning")
    
    # Prepare device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Create data loader with different batch size for fine-tuning
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    if validation_data:
        val_loader = DataLoader(validation_data, batch_size=batch_size)
    
    # Define loss function and optimizer (with lower learning rate for fine-tuning)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    
    # Fine-tuning loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in dataloader:
            # Move to device
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        # Calculate average loss
        epoch_loss = running_loss / len(dataset)
        
        # Validation
        if validation_data:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    
                    # Calculate accuracy
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_loss = val_loss / len(validation_data)
            val_acc = correct / total
            
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
        else:
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.4f}")
    
    logger.info("Position bias correction fine-tuning complete")
    return model