import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import joblib
from config import MODEL_PATH, BATCH_SIZE, LEARNING_RATE, EPOCHS, TEST_SIZE

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
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.dropout(x)
        x = self.relu(self.layer2(x))
        x = self.dropout(x)
        x = self.relu(self.layer3(x))
        x = self.sigmoid(self.output(x))
        return x

class ModelTrainer:
    def __init__(self, data_df):
        self.data_df = data_df
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the directory for model if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Path for the scaler
        self.scaler_path = os.path.join(os.path.dirname(MODEL_PATH), 'scaler.save')
    
    def preprocess_data(self):
        # Drop non-numeric columns if any
        self.data_df = self.data_df.select_dtypes(include=[np.number])
        
        # Handle missing values
        self.data_df = self.data_df.fillna(0)
        
        # Separate features and labels
        X = self.data_df.drop('fighter1_won', axis=1)
        y = self.data_df['fighter1_won']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=42
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Save the scaler
        joblib.dump(self.scaler, self.scaler_path)
        
        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled)
        y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))
        
        # Create datasets
        train_dataset = UFCDataset(X_train_tensor, y_train_tensor)
        test_dataset = UFCDataset(X_test_tensor, y_test_tensor)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
        
        return train_loader, test_loader, X.shape[1]
    
    def train_model(self):
        # Preprocess data
        train_loader, test_loader, input_size = self.preprocess_data()
        
        # Initialize model
        self.model = FightPredictor(input_size).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        
        # Training loop
        for epoch in range(EPOCHS):
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
                optimizer.step()
                
                running_loss += loss.item()
            
            # Evaluate on test set
            self.model.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    test_loss += criterion(outputs, labels).item()
                    
                    predicted = (outputs >= 0.5).float()
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {running_loss/len(train_loader):.4f}, '
                  f'Test Loss: {test_loss/len(test_loader):.4f}, Test Accuracy: {accuracy:.2f}%')
        
        # Save the model
        torch.save(self.model.state_dict(), MODEL_PATH)
        print(f"Model saved to {MODEL_PATH}")
    
    def load_model(self):
        # Load the scaler
        self.scaler = joblib.load(self.scaler_path)
        
        # Get input size from data
        input_size = self.data_df.drop('fighter1_won', axis=1).shape[1]
        
        # Initialize and load model
        self.model = FightPredictor(input_size)
        self.model.load_state_dict(torch.load(MODEL_PATH))
        self.model.to(self.device)
        self.model.eval()
        
        return self.model
    
    def predict_fight(self, fighter1_features, fighter2_features):
        """
        Predict the outcome of a fight between two fighters
        
        Parameters:
        fighter1_features: Dict of features for fighter 1
        fighter2_features: Dict of features for fighter 2
        
        Returns:
        probability: Float, the probability of fighter 1 winning
        """
        if self.model is None:
            self.load_model()
        
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
        
        return probability