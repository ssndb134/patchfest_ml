import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os

# --- Dataset ---
class SlidingWindowDataset(Dataset):
    """
    Creates a sliding window dataset for time series forecasting.
    X: Window of past `seq_length` features
    y: Next `horizon` target values
    """
    def __init__(self, data, seq_length=30, horizon=30, target_col='count'):
        self.data = data
        self.seq_length = seq_length
        self.horizon = horizon
        self.target_col = target_col
        
        # Keep only numeric columns for features
        self.features = self.data.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore').columns.tolist()
        
        # If no other features, use target as feature
        if not self.features:
            self.features = [target_col]
        else:
            # Maybe include target in features? Typically yes for AR
            self.features = [target_col] + [f for f in self.features if f != target_col]

        self.feature_data = self.data[self.features].values.astype(np.float32)
        self.target_data = self.data[target_col].values.astype(np.float32)

    def __len__(self):
        # We need seq_length for X and horizon for y
        # Total available start indices = len - seq - horizon + 1
        return max(0, len(self.data) - self.seq_length - self.horizon + 1)

    def __getitem__(self, idx):
        # X: [idx : idx+seq_length]
        # y: [idx+seq_length : idx+seq_length+horizon]
        x_window = self.feature_data[idx : idx + self.seq_length]
        y_window = self.target_data[idx + self.seq_length : idx + self.seq_length + self.horizon]
        
        return torch.tensor(x_window), torch.tensor(y_window)

# --- Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=30, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass LSTM
        # out shape: (batch_size, seq_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

# --- Training ---
def train_model(model, train_loader, val_loader, epochs=100, lr=0.001, patience=10, model_path="models/lstm.pt"):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    print("Starting training...")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        train_loss /= len(train_loader) if len(train_loader) > 0 else 1
        val_loss /= len(val_loader) if len(val_loader) > 0 else 1
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    print(f"Training complete. Best Val Loss: {best_val_loss:.4f}")
    if best_val_loss != float('inf'):
         print(f"Model saved to {model_path}")

# --- Main ---
if __name__ == "__main__":
    # Parameters
    SEQ_LENGTH = 30
    HORIZON = 30
    BATCH_SIZE = 32
    HIDDEN_SIZE = 64
    LAYERS = 2
    
    # Load Data
    data_path = "data/processed/enhanced_forecast_data.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}")
        exit()
        
    df = pd.read_csv(data_path)
    
    # Check if we have enough data
    min_needed = SEQ_LENGTH + HORIZON + 1
    if len(df) < min_needed:
        print(f"Not enough data points ({len(df)}) for window {SEQ_LENGTH} and horizon {HORIZON}.")
        print("Using dummy data for demonstration (Since prompt asked for Code Design/Implementation).")
        # Generate dummy data
        dates = pd.date_range(start='2020-01-01', periods=200)
        df = pd.DataFrame({
            'date': dates, 
            'count': np.sin(np.linspace(0, 20, 200)) * 100 + 150 + np.random.normal(0, 10, 200),
            'lag_1': np.zeros(200), # Dummy features
            'lag_7': np.zeros(200)
        })

    # Prepare data splits (No Leakage: standard time split)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size].reset_index(drop=True)
    val_df = df.iloc[train_size:].reset_index(drop=True)
    
    train_dataset = SlidingWindowDataset(train_df, seq_length=SEQ_LENGTH, horizon=HORIZON)
    val_dataset = SlidingWindowDataset(val_df, seq_length=SEQ_LENGTH, horizon=HORIZON)
    
    # Handle case where validation set is too small for window
    if len(val_dataset) == 0:
        print("Validation set too small for sliding window. Adjusting split or using train for demo.")
        val_dataset = train_dataset # Fallback for demo code correctness
        
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False) # Shuffle False for time series usually better? 
    # Actually, within training set, shuffling windows is fine and helps generalization.
    # But for strict time series, simple implementation often shuffles.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Input size calculation
    sample_x, sample_y = train_dataset[0]
    input_dim = sample_x.shape[1]
    
    model = LSTMModel(input_size=input_dim, hidden_size=HIDDEN_SIZE, num_layers=LAYERS, output_size=HORIZON)
    
    # output dir
    os.makedirs("models", exist_ok=True)
    
    train_model(model, train_loader, val_loader, epochs=50, patience=5, model_path="models/lstm.pt")
