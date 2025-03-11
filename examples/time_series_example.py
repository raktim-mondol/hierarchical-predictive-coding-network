"""
Time Series Example for Hierarchical Predictive Coding Network

This script demonstrates how to use the HPCN model with temporal dynamics
for time series prediction and pattern recognition.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys

# Add parent directory to path to import hpcn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpcn.model import HPCNWithTemporalDynamics
from hpcn.layers import PCLayer, PCLayerWithErrorUnits, LocalHebbianPCLayer


def generate_synthetic_data(num_samples=1000, seq_length=50, num_features=5):
    """
    Generate synthetic time series data with multiple features and patterns
    
    Args:
        num_samples (int): Number of sequences to generate
        seq_length (int): Length of each sequence
        num_features (int): Number of features per time step
        
    Returns:
        tuple: (train_data, test_data)
    """
    t = np.linspace(0, 4*np.pi, seq_length)
    
    # Generate different patterns
    data = []
    for _ in range(num_samples):
        # Base signals
        sin_signal = np.sin(t + np.random.uniform(0, np.pi))
        cos_signal = np.cos(2*t + np.random.uniform(0, np.pi))
        square_signal = np.sign(sin_signal)
        saw_signal = t % (np.pi/2)
        noise = np.random.normal(0, 0.1, seq_length)
        
        # Combine signals with random weights
        sequence = np.stack([
            sin_signal + 0.1*noise,
            cos_signal + 0.1*noise,
            square_signal + 0.1*noise,
            saw_signal + 0.1*noise,
            0.5*sin_signal + 0.5*cos_signal + 0.1*noise
        ]).T  # Shape: [seq_length, num_features]
        
        data.append(sequence)
    
    data = np.stack(data)  # Shape: [num_samples, seq_length, num_features]
    
    # Split into train and test
    train_size = int(0.8 * num_samples)
    train_data = data[:train_size]
    test_data = data[train_size:]
    
    return train_data, test_data


def create_data_loaders(train_data, test_data, batch_size=32):
    """
    Create PyTorch DataLoaders for the time series data
    
    Args:
        train_data (np.ndarray): Training data
        test_data (np.ndarray): Test data
        batch_size (int): Batch size
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    # Create datasets
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def create_model(model_type='standard', input_size=5, temporal_window=5):
    """
    Create HPCN model for time series
    
    Args:
        model_type (str): Type of model to create ('standard', 'error_units', or 'hebbian')
        input_size (int): Number of input features
        temporal_window (int): Size of temporal window
        
    Returns:
        HPCNWithTemporalDynamics: HPCN model with temporal dynamics
    """
    # Create layers based on model type
    if model_type == 'standard':
        layers = [
            PCLayer(input_size=input_size, hidden_size=32),
            PCLayer(input_size=32, hidden_size=16),
            PCLayer(input_size=16, hidden_size=8),
            PCLayer(input_size=8, hidden_size=16),
            PCLayer(input_size=16, hidden_size=32),
            PCLayer(input_size=32, hidden_size=input_size)
        ]
    elif model_type == 'error_units':
        layers = [
            PCLayerWithErrorUnits(input_size=input_size, hidden_size=32),
            PCLayerWithErrorUnits(input_size=32, hidden_size=16),
            PCLayerWithErrorUnits(input_size=16, hidden_size=8),
            PCLayerWithErrorUnits(input_size=8, hidden_size=16),
            PCLayerWithErrorUnits(input_size=16, hidden_size=32),
            PCLayerWithErrorUnits(input_size=32, hidden_size=input_size)
        ]
    elif model_type == 'hebbian':
        layers = [
            LocalHebbianPCLayer(input_size=input_size, hidden_size=32),
            LocalHebbianPCLayer(input_size=32, hidden_size=16),
            LocalHebbianPCLayer(input_size=16, hidden_size=8),
            LocalHebbianPCLayer(input_size=8, hidden_size=16),
            LocalHebbianPCLayer(input_size=16, hidden_size=32),
            LocalHebbianPCLayer(input_size=32, hidden_size=input_size)
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model with temporal dynamics
    model = HPCNWithTemporalDynamics(
        layers=layers,
        supervised=False,
        temporal_window=temporal_window
    )
    
    return model


def visualize_predictions(model, test_data, device='cpu'):
    """
    Visualize model predictions on test data
    
    Args:
        model (HPCNWithTemporalDynamics): Trained model
        test_data (torch.Tensor): Test data
        device (str): Device to use
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for first test sequence
        test_sequence = test_data[0:1].to(device)
        reconstructed = model.reconstruct(test_sequence)
        
        # Plot original vs reconstructed
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        fig.suptitle('Original vs Reconstructed Signals')
        
        for i in range(5):
            axes[i].plot(test_sequence[0, :, i].cpu().numpy(), 
                        label='Original', alpha=0.7)
            axes[i].plot(reconstructed[0, :, i].cpu().numpy(), 
                        label='Reconstructed', alpha=0.7)
            axes[i].set_title(f'Feature {i+1}')
            axes[i].legend()
        
        plt.tight_layout()
        plt.savefig('time_series_predictions.png')
        plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HPCN Time Series Example')
    parser.add_argument('--model-type', type=str, default='standard',
                        choices=['standard', 'error_units', 'hebbian'],
                        help='Type of HPCN model to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to train on (cpu or cuda)')
    parser.add_argument('--temporal-window', type=int, default=5,
                        help='Size of temporal window')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'
    
    # Generate synthetic data
    print("Generating synthetic time series data...")
    train_data, test_data = generate_synthetic_data(
        num_samples=1000,
        seq_length=50,
        num_features=5
    )
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        train_data, test_data, batch_size=args.batch_size
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        input_size=train_data.shape[2],  # number of features
        temporal_window=args.temporal_window
    )
    
    # Train model
    print("Training model...")
    model.to(args.device)
    losses = model.train_unsupervised(
        train_loader,
        epochs=args.epochs,
        device=args.device
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_error = model.evaluate(test_loader, device=args.device)
    print(f"Test reconstruction error: {test_error:.6f}")
    
    # Visualize results
    print("Generating visualizations...")
    visualize_predictions(model, torch.FloatTensor(test_data), device=args.device)
    print("Saved visualization to time_series_predictions.png")
    
    # Save model
    torch.save(model.state_dict(), f'time_series_{args.model_type}.pt')
    print(f"Model saved to time_series_{args.model_type}.pt")


if __name__ == '__main__':
    main()