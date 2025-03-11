"""
MNIST Example for Hierarchical Predictive Coding Network

This script demonstrates how to use the HPCN model on the MNIST dataset
for both unsupervised and supervised learning tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

# Add parent directory to path to import hpcn
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpcn.model import HPCN
from hpcn.layers import PCLayer, PCLayerWithErrorUnits, LocalHebbianPCLayer
from hpcn.utils import (
    visualize_reconstructions, 
    visualize_prediction_errors,
    visualize_representations,
    plot_training_curves
)


def load_mnist(batch_size=64):
    """
    Load MNIST dataset
    
    Args:
        batch_size (int, optional): Batch size. Defaults to 64.
        
    Returns:
        tuple: (train_loader, test_loader)
    """
    # Define transformations
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # Load datasets
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader


def create_model(model_type='standard', supervised=False):
    """
    Create HPCN model
    
    Args:
        model_type (str, optional): Type of model to create. Defaults to 'standard'.
        supervised (bool, optional): Whether to use supervised learning. Defaults to False.
        
    Returns:
        HPCN: HPCN model
    """
    # Define input size for MNIST
    input_size = 28 * 28  # Flattened 28x28 images
    
    # Create layers based on model type
    if model_type == 'standard':
        layers = [
            PCLayer(input_size=input_size, hidden_size=400),
            PCLayer(input_size=400, hidden_size=200),
            PCLayer(input_size=200, hidden_size=100)
        ]
    elif model_type == 'error_units':
        layers = [
            PCLayerWithErrorUnits(input_size=input_size, hidden_size=400),
            PCLayerWithErrorUnits(input_size=400, hidden_size=200),
            PCLayerWithErrorUnits(input_size=200, hidden_size=100)
        ]
    elif model_type == 'hebbian':
        layers = [
            LocalHebbianPCLayer(input_size=input_size, hidden_size=400),
            LocalHebbianPCLayer(input_size=400, hidden_size=200),
            LocalHebbianPCLayer(input_size=200, hidden_size=100)
        ]
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create model
    if supervised:
        model = HPCN(layers, supervised=True, output_size=10)
    else:
        model = HPCN(layers, supervised=False)
    
    return model


def train_unsupervised(model, train_loader, test_loader, epochs=10, device='cpu'):
    """
    Train HPCN in unsupervised mode
    
    Args:
        model (HPCN): HPCN model
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Device to train on. Defaults to 'cpu'.
        
    Returns:
        tuple: (losses, model)
    """
    print("Training in unsupervised mode...")
    
    # Train the model
    losses = model.train_unsupervised(train_loader, epochs=epochs, device=device)
    
    # Evaluate reconstruction error
    recon_error = model.evaluate(test_loader, device=device)
    print(f"Test reconstruction error: {recon_error:.6f}")
    
    # Visualize results
    # Get a batch of test data
    test_data, _ = next(iter(test_loader))
    
    # Visualize reconstructions
    fig_recon = visualize_reconstructions(model, test_data, n_samples=5, device=device)
    fig_recon.savefig('unsupervised_reconstructions.png')
    
    # Visualize prediction errors
    fig_errors = visualize_prediction_errors(model, test_data[:1], device=device)
    fig_errors.savefig('unsupervised_prediction_errors.png')
    
    # Plot training curve
    fig_loss = plot_training_curves(losses, title='Unsupervised Training Loss')
    fig_loss.savefig('unsupervised_training_loss.png')
    
    print("Saved visualizations to current directory.")
    
    return losses, model


def train_supervised(model, train_loader, test_loader, epochs=10, device='cpu'):
    """
    Train HPCN in supervised mode
    
    Args:
        model (HPCN): HPCN model
        train_loader (DataLoader): Training data loader
        test_loader (DataLoader): Test data loader
        epochs (int, optional): Number of epochs. Defaults to 10.
        device (str, optional): Device to train on. Defaults to 'cpu'.
        
    Returns:
        tuple: (losses, model)
    """
    print("Training in supervised mode...")
    
    # Train the model
    losses = model.train_supervised(train_loader, epochs=epochs, device=device)
    
    # Evaluate accuracy
    accuracy = model.evaluate(test_loader, device=device)
    print(f"Test accuracy: {accuracy:.2f}%")
    
    # Visualize results
    # Get a batch of test data
    test_data, test_labels = next(iter(test_loader))
    
    # Visualize representations
    fig_rep = visualize_representations(model, (test_data, test_labels), device=device)
    fig_rep.savefig('supervised_representations.png')
    
    # Visualize prediction errors
    fig_errors = visualize_prediction_errors(model, test_data[:1], device=device)
    fig_errors.savefig('supervised_prediction_errors.png')
    
    # Plot training curve
    fig_loss = plot_training_curves(losses, title='Supervised Training Loss')
    fig_loss.savefig('supervised_training_loss.png')
    
    print("Saved visualizations to current directory.")
    
    return losses, model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='HPCN MNIST Example')
    parser.add_argument('--mode', type=str, default='unsupervised', choices=['unsupervised', 'supervised'],
                        help='Training mode (unsupervised or supervised)')
    parser.add_argument('--model-type', type=str, default='standard', 
                        choices=['standard', 'error_units', 'hebbian'],
                        help='Type of HPCN model to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--device', type=str, default='cpu', help='Device to train on (cpu or cuda)')
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA is not available, using CPU instead.")
        args.device = 'cpu'
    
    # Load MNIST dataset
    train_loader, test_loader = load_mnist(batch_size=args.batch_size)
    
    # Create model
    supervised = args.mode == 'supervised'
    model = create_model(model_type=args.model_type, supervised=supervised)
    
    # Train model
    if args.mode == 'unsupervised':
        losses, model = train_unsupervised(model, train_loader, test_loader, 
                                          epochs=args.epochs, device=args.device)
    else:
        losses, model = train_supervised(model, train_loader, test_loader, 
                                        epochs=args.epochs, device=args.device)
    
    # Save model
    torch.save(model.state_dict(), f'mnist_{args.mode}_{args.model_type}.pt')
    print(f"Model saved to mnist_{args.mode}_{args.model_type}.pt")


if __name__ == '__main__':
    main()