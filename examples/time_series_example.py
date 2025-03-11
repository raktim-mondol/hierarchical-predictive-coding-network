"""
Time Series Example for Hierarchical Predictive Coding Network

This script demonstrates how to use the HPCN model with temporal dynamics
for time series prediction and anomaly detection.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Add parent directory to path to import hpcn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpcn.model import HPCNWithTemporalDynamics
from hpcn.layers import PCLayer
from hpcn.utils import plot_training_curves


def generate_sine_wave(n_samples=1000, seq_length=50, n_features=1, freq=0.1, noise=0.1):
    """
    Generate sine wave data
    
    Args:
        n_samples (int, optional): Number of samples. Defaults to 1000.
        seq_length (int, optional): Sequence length. Defaults to 50.
        n_features (int, optional): Number of features. Defaults to 1.
        freq (float, optional): Frequency of the sine wave. Defaults to 0.1.
        noise (float, optional): Noise level. Defaults to 0.1.
        
    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    # Generate time steps
    t = np.arange(0, (n_samples + seq_length) / freq) * freq
    
    # Generate sine wave
    data = np.sin(t)
    
    # Add noise
    if noise > 0:
        data += np.random.normal(0, noise, size=len(data))
    
    # Create sequences
    X = np.zeros((n_samples, seq_length, n_features))
    y = np.zeros((n_samples, n_features))
    
    for i in range(n_samples):
        X[i, :, 0] = data[i:i+seq_length]
        y[i, 0] = data[i+seq_length]
    
    return X, y


def create_anomalies(X, y, anomaly_ratio=0.05, anomaly_scale=5.0):
    """
    Create anomalies in the data
    
    Args:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        anomaly_ratio (float, optional): Ratio of anomalies. Defaults to 0.05.
        anomaly_scale (float, optional): Scale of anomalies. Defaults to 5.0.
        
    Returns:
        tuple: (X, y, anomaly_indices)
    """
    n_samples = X.shape[0]
    n_anomalies = int(n_samples * anomaly_ratio)
    
    # Randomly select indices for anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    
    # Create anomalies
    for i in anomaly_indices:
        # Add spike to the last time step
        X[i, -1, 0] += anomaly_scale * np.random.choice([-1, 1])
    
    return X, y, anomaly_indices


def prepare_data(X, y, train_ratio=0.8, batch_size=32):
    """
    Prepare data for training
    
    Args:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        train_ratio (float, optional): Ratio of training data. Defaults to 0.8.
        batch_size (int, optional): Batch size. Defaults to 32.
        
    Returns:
        tuple: (train_loader, test_loader, scaler)
    """
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = np.zeros_like(X)
    
    # Scale each feature
    for i in range(X.shape[2]):
        X_scaled[:, :, i] = scaler.fit_transform(X[:, :, i])
    
    y_scaled = scaler.transform(y)
    
    # Split data
    n_train = int(X.shape[0] * train_ratio)
    X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
    y_train, y_test = y_scaled[:n_train], y_scaled[n_train:]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader, scaler


def create_model(seq_length, n_features):
    """
    Create HPCN model with temporal dynamics
    
    Args:
        seq_length (int): Sequence length
        n_features (int): Number of features
        
    Returns:
        HPCNWithTemporalDynamics: HPCN model with temporal dynamics
    """
    # Define input size
    input_size = n_features
    
    # Create layers
    layers = [
        PCLayer(input_size=input_size, hidden_size=32),
        PCLayer(input_size=32, hidden_size=16),
        PCLayer(input_size=16, hidden_size=8)
    ]
    
    # Create model
    model = HPCNWithTemporalDynamics(
        layers, 
        supervised=True, 
        output_size=n_features,
        temporal_window=5
    )
    
    return model


def train_model(model, train_loader, test_loader, epochs=50, learning_rate=0.001, device='cpu'):
    """
    Train the model
    
    Args:
        model (HPCNWithTemporalDynamics): HPCN model with temporal dynamics
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        epochs (int, optional): Number of epochs. Defaults to 50.
        learning_rate (float, optional): Learning rate. Defaults to 0.001.
        device (str, optional): Device to train on. Defaults to 'cpu'.
        
    Returns:
        tuple: (losses, model)
    """
    print("Training model...")
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            
            # Compute loss
            loss = criterion(outputs[:, -1], y_batch)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Testing
        model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                # Forward pass
                outputs = model(X_batch)
                
                # Compute loss
                loss = criterion(outputs[:, -1], y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Test Loss: {test_loss:.6f}")
    
    return train_losses, test_losses, model


def detect_anomalies(model, X, threshold=0.1, device='cpu'):
    """
    Detect anomalies in the data
    
    Args:
        model (HPCNWithTemporalDynamics): Trained HPCN model
        X (np.ndarray): Input sequences
        threshold (float, optional): Anomaly threshold. Defaults to 0.1.
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        np.ndarray: Anomaly scores
    """
    model.to(device)
    model.eval()
    
    # Convert to PyTorch tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # Compute prediction errors
    with torch.no_grad():
        _, _, errors = model(X_tensor, return_all=True, return_errors=True)
    
    # Compute anomaly scores (mean squared error of the last time step)
    anomaly_scores = torch.mean(errors[0][:, -1, :]**2, dim=1).cpu().numpy()
    
    return anomaly_scores


def plot_results(X, y, predictions, anomaly_scores=None, anomaly_indices=None, scaler=None):
    """
    Plot results
    
    Args:
        X (np.ndarray): Input sequences
        y (np.ndarray): Target values
        predictions (np.ndarray): Predicted values
        anomaly_scores (np.ndarray, optional): Anomaly scores. Defaults to None.
        anomaly_indices (np.ndarray, optional): Indices of true anomalies. Defaults to None.
        scaler (MinMaxScaler, optional): Scaler used to normalize the data. Defaults to None.
    """
    # Inverse transform if scaler is provided
    if scaler is not None:
        y = scaler.inverse_transform(y)
        predictions = scaler.inverse_transform(predictions)
    
    # Plot predictions
    plt.figure(figsize=(12, 6))
    plt.plot(y, label='True')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    plt.title('Time Series Prediction')
    plt.savefig('time_series_prediction.png')
    
    # Plot anomaly scores if available
    if anomaly_scores is not None:
        plt.figure(figsize=(12, 6))
        plt.plot(anomaly_scores)
        
        if anomaly_indices is not None:
            for idx in anomaly_indices:
                plt.axvline(x=idx, color='r', linestyle='--', alpha=0.3)
        
        plt.title('Anomaly Scores')
        plt.savefig('anomaly_scores.png')


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HPCN Time Series Example')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--seq-length', type=int, default=50, help='Sequence length')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu or cuda)')
    parser.add_argument('--anomaly', action='store_true', help='Add anomalies to the data')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'
    
    # Generate data
    print("Generating data...")
    X, y = generate_sine_wave(n_samples=1000, seq_length=args.seq_length)
    
    # Add anomalies if requested
    anomaly_indices = None
    if args.anomaly:
        print("Adding anomalies...")
        X, y, anomaly_indices = create_anomalies(X, y)
    
    # Prepare data
    train_loader, test_loader, scaler = prepare_data(X, y, batch_size=args.batch_size)
    
    # Create model
    model = create_model(seq_length=args.seq_length, n_features=1)
    
    # Train model
    train_losses, test_losses, model = train_model(
        model, train_loader, test_loader, 
        epochs=args.epochs, device=args.device
    )
    
    # Plot training curves
    fig = plot_training_curves(train_losses, title='Training Loss')
    fig.savefig('time_series_training_loss.png')
    
    # Make predictions
    model.eval()
    X_tensor = torch.FloatTensor(X).to(args.device)
    with torch.no_grad():
        predictions = model(X_tensor)[:, -1].cpu().numpy()
    
    # Detect anomalies if requested
    anomaly_scores = None
    if args.anomaly:
        print("Detecting anomalies...")
        anomaly_scores = detect_anomalies(model, X, device=args.device)
    
    # Plot results
    plot_results(X, y, predictions, anomaly_scores, anomaly_indices, scaler)
    
    # Save model
    torch.save(model.state_dict(), 'time_series_model.pt')
    print("Model saved to time_series_model.pt")


if __name__ == '__main__':
    main()