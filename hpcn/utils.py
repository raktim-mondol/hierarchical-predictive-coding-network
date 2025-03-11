"""
Utilities for Hierarchical Predictive Coding Networks

This module provides utility functions for visualization, monitoring, and analysis
of Hierarchical Predictive Coding Networks.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from torchvision.utils import make_grid


def visualize_reconstructions(model, data, n_samples=10, device='cpu'):
    """
    Visualize original and reconstructed images
    
    Args:
        model (HPCN): Trained HPCN model
        data (torch.Tensor): Input data
        n_samples (int, optional): Number of samples to visualize. Defaults to 10.
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        plt.Figure: Matplotlib figure with visualizations
    """
    # Select a subset of samples
    indices = np.random.choice(len(data), min(n_samples, len(data)), replace=False)
    samples = data[indices].to(device)
    
    # Get reconstructions
    reconstructions = model.reconstruct(samples, device=device)
    
    # Convert to numpy for visualization
    samples = samples.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
    
    # Plot original and reconstructed images
    for i in range(n_samples):
        # Original
        if samples.shape[1] == 1:  # Grayscale
            axes[0, i].imshow(samples[i, 0], cmap='gray')
        else:  # RGB
            axes[0, i].imshow(np.transpose(samples[i], (1, 2, 0)))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstruction
        if reconstructions.shape[1] == 1:  # Grayscale
            axes[1, i].imshow(reconstructions[i, 0], cmap='gray')
        else:  # RGB
            axes[1, i].imshow(np.transpose(reconstructions[i], (1, 2, 0)))
        axes[1, i].set_title('Reconstructed')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    return fig


def visualize_prediction_errors(model, data, device='cpu'):
    """
    Visualize prediction errors at each layer
    
    Args:
        model (HPCN): Trained HPCN model
        data (torch.Tensor): Input data
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        plt.Figure: Matplotlib figure with visualizations
    """
    # Forward pass with error tracking
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        
        # Flatten if needed
        if len(data.shape) > 2:
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
            
        # Forward pass
        _, _, errors = model.forward(data_flat, return_all=True, return_errors=True)
    
    # Create figure
    n_layers = len(errors)
    fig, axes = plt.subplots(1, n_layers, figsize=(4*n_layers, 4))
    if n_layers == 1:
        axes = [axes]
    
    # Plot error magnitudes for each layer
    for i, layer_errors in enumerate(errors):
        # Compute mean squared error
        mse = torch.mean(layer_errors**2, dim=(0, 2))
        
        # Plot
        axes[i].plot(mse.cpu().numpy())
        axes[i].set_title(f'Layer {i+1} Errors')
        axes[i].set_xlabel('Prediction Step')
        axes[i].set_ylabel('Mean Squared Error')
    
    plt.tight_layout()
    return fig


def visualize_representations(model, data, layer_idx=-1, n_components=2, device='cpu'):
    """
    Visualize learned representations using dimensionality reduction
    
    Args:
        model (HPCN): Trained HPCN model
        data (torch.Tensor): Input data
        layer_idx (int, optional): Index of the layer to visualize. Defaults to -1 (last layer).
        n_components (int, optional): Number of components for dimensionality reduction. Defaults to 2.
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        plt.Figure: Matplotlib figure with visualizations
    """
    from sklearn.decomposition import PCA
    
    # Forward pass to get representations
    model.eval()
    with torch.no_grad():
        data_tensor = data[0].to(device)
        labels = data[1].cpu().numpy()
        
        # Flatten if needed
        if len(data_tensor.shape) > 2:
            data_tensor = data_tensor.reshape(data_tensor.shape[0], -1)
            
        # Forward pass
        _, representations = model.forward(data_tensor, return_all=True)
        
        # Get representations from the specified layer
        layer_representations = representations[layer_idx].cpu().numpy()
    
    # Apply dimensionality reduction
    pca = PCA(n_components=n_components)
    reduced_representations = pca.fit_transform(layer_representations)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot representations
    if n_components == 2:
        scatter = ax.scatter(reduced_representations[:, 0], reduced_representations[:, 1], 
                            c=labels, cmap='tab10', alpha=0.7)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
    elif n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(reduced_representations[:, 0], reduced_representations[:, 1], 
                            reduced_representations[:, 2], c=labels, cmap='tab10', alpha=0.7)
        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
    
    # Add legend
    legend = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend)
    
    ax.set_title(f'Layer {layer_idx+1} Representations')
    plt.tight_layout()
    return fig


def visualize_layer_activations(model, data, device='cpu'):
    """
    Visualize activations at each layer
    
    Args:
        model (HPCN): Trained HPCN model
        data (torch.Tensor): Input data (single sample)
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        plt.Figure: Matplotlib figure with visualizations
    """
    # Ensure we have a single sample
    if len(data.shape) == 4:  # Image data (B, C, H, W)
        data = data[0:1]
    elif len(data.shape) == 2:  # Flat data (B, F)
        data = data[0:1]
    
    # Forward pass to get representations
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        
        # Flatten if needed
        if len(data.shape) > 2:
            original_shape = data.shape
            data_flat = data.reshape(data.shape[0], -1)
        else:
            data_flat = data
            
        # Forward pass
        _, representations = model.forward(data_flat, return_all=True)
    
    # Create figure
    n_layers = len(representations)
    fig = plt.figure(figsize=(4*n_layers, 4))
    gs = GridSpec(1, n_layers, figure=fig)
    
    # Plot activations for each layer
    for i, rep in enumerate(representations):
        ax = fig.add_subplot(gs[0, i])
        
        # Get activations
        activations = rep[0].cpu().numpy()
        
        # Reshape for visualization if possible
        if np.sqrt(activations.shape[0]).is_integer():
            size = int(np.sqrt(activations.shape[0]))
            activations = activations.reshape(size, size)
            im = ax.imshow(activations, cmap='viridis')
        else:
            # Plot as a 1D array
            im = ax.imshow(activations.reshape(1, -1), cmap='viridis', aspect='auto')
        
        ax.set_title(f'Layer {i+1} Activations')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    return fig


def plot_training_curves(losses, title='Training Loss'):
    """
    Plot training loss curves
    
    Args:
        losses (list): List of losses
        title (str, optional): Plot title. Defaults to 'Training Loss'.
        
    Returns:
        plt.Figure: Matplotlib figure with plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(losses)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(True)
    plt.tight_layout()
    return fig


def compute_prediction_error_statistics(model, dataloader, device='cpu'):
    """
    Compute statistics of prediction errors
    
    Args:
        model (HPCN): Trained HPCN model
        dataloader (DataLoader): DataLoader for the data
        device (str, optional): Device to use. Defaults to 'cpu'.
        
    Returns:
        dict: Dictionary with error statistics
    """
    model.eval()
    
    # Initialize statistics
    total_samples = 0
    layer_errors = [[] for _ in range(len(model.layers))]
    
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            batch_size = data.shape[0]
            total_samples += batch_size
            
            # Flatten if needed
            if len(data.shape) > 2:
                data = data.reshape(data.shape[0], -1)
            
            # Forward pass
            _, _, errors = model.forward(data, return_all=True, return_errors=True)
            
            # Collect errors
            for i, layer_err in enumerate(errors):
                # Mean error per sample
                mean_err = torch.mean(layer_err**2, dim=(1, 2))
                layer_errors[i].append(mean_err)
    
    # Compute statistics
    stats = {}
    for i, errors in enumerate(layer_errors):
        errors = torch.cat(errors, dim=0).cpu().numpy()
        stats[f'layer_{i+1}'] = {
            'mean': np.mean(errors),
            'std': np.std(errors),
            'min': np.min(errors),
            'max': np.max(errors),
            'median': np.median(errors)
        }
    
    return stats