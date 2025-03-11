"""
Hierarchical Predictive Coding Network (HPCN) Model

This module implements the main HPCN model, which consists of multiple predictive coding
layers organized in a hierarchy. The model can be used for both supervised and unsupervised
learning tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from .layers import PCLayer


class HPCN(nn.Module):
    """
    Hierarchical Predictive Coding Network
    
    A neural network model based on the predictive coding theory from neuroscience,
    consisting of multiple predictive coding layers organized in a hierarchy.
    
    Args:
        layers (list): List of PCLayer instances
        supervised (bool, optional): Whether to use supervised learning. Defaults to False.
        output_size (int, optional): Size of the output layer for supervised learning. Defaults to None.
    """
    
    def __init__(self, layers, supervised=False, output_size=None):
        super(HPCN, self).__init__()
        
        self.layers = nn.ModuleList(layers)
        self.supervised = supervised
        
        # Add output layer for supervised learning
        if supervised and output_size is not None:
            self.output_layer = nn.Linear(layers[-1].hidden_size, output_size)
            nn.init.xavier_normal_(self.output_layer.weight)
        
    def forward(self, x, return_all=False, return_errors=False):
        """
        Forward pass through the HPCN
        
        Args:
            x (torch.Tensor): Input tensor
            return_all (bool, optional): Whether to return all layer representations. Defaults to False.
            return_errors (bool, optional): Whether to return prediction errors. Defaults to False.
            
        Returns:
            torch.Tensor or tuple: Output tensor or tuple of (outputs, representations, errors)
        """
        batch_size = x.shape[0]
        
        # Initialize lists to store representations and errors
        representations = []
        predictions = []
        errors = []
        
        # Forward pass through each layer
        current_input = x
        for i, layer in enumerate(self.layers):
            # Forward pass through the layer
            if return_errors:
                rep, pred, err = layer(current_input, None, return_errors=True)
                representations.append(rep)
                predictions.append(pred)
                errors.append(err)
            else:
                rep = layer(current_input, None, return_errors=False)
                representations.append(rep)
            
            # Update input for the next layer
            current_input = rep
        
        # Output for supervised learning
        if self.supervised:
            output = self.output_layer(representations[-1])
        else:
            output = representations[-1]
        
        # Return based on flags
        if return_all and return_errors:
            return output, representations, errors
        elif return_all:
            return output, representations
        else:
            return output
    
    def predict_input(self, representation=None, layer_idx=-1):
        """
        Generate a prediction of the input given a representation
        
        Args:
            representation (torch.Tensor, optional): Representation tensor. If None, uses the
                                                    representation from the last forward pass.
            layer_idx (int, optional): Index of the layer to start prediction from. Defaults to -1.
            
        Returns:
            torch.Tensor: Prediction of the input
        """
        if representation is None:
            raise ValueError("Representation must be provided")
        
        # Start from the specified layer
        current_rep = representation
        
        # Generate predictions layer by layer, from top to bottom
        for i in range(layer_idx, -1, -1):
            current_rep = self.layers[i].predict(current_rep)
        
        return current_rep
    
    def train_unsupervised(self, dataloader, epochs=10, learning_rate=0.001, device='cpu'):
        """
        Train the HPCN in an unsupervised manner
        
        Args:
            dataloader (DataLoader): DataLoader for the training data
            epochs (int, optional): Number of training epochs. Defaults to 10.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            device (str, optional): Device to train on. Defaults to 'cpu'.
            
        Returns:
            list: Training losses
        """
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, (data, _) in enumerate(pbar):
                    data = data.to(device)
                    
                    # Flatten the data if needed
                    if len(data.shape) > 2:
                        data = data.reshape(data.shape[0], -1)
                    
                    # Forward pass
                    _, representations, errors = self.forward(data, return_all=True, return_errors=True)
                    
                    # Compute reconstruction loss (prediction error at the input level)
                    input_prediction = self.predict_input(representations[-1])
                    recon_loss = F.mse_loss(input_prediction, data)
                    
                    # Compute prediction errors at each layer
                    error_loss = 0.0
                    for error in errors:
                        error_loss += torch.mean(error**2)
                    
                    # Total loss
                    loss = recon_loss + 0.1 * error_loss
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return losses
    
    def train_supervised(self, dataloader, epochs=10, learning_rate=0.001, device='cpu'):
        """
        Train the HPCN in a supervised manner
        
        Args:
            dataloader (DataLoader): DataLoader for the training data
            epochs (int, optional): Number of training epochs. Defaults to 10.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.001.
            device (str, optional): Device to train on. Defaults to 'cpu'.
            
        Returns:
            list: Training losses
        """
        if not self.supervised:
            raise ValueError("Model was not initialized for supervised learning")
        
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, (data, targets) in enumerate(pbar):
                    data, targets = data.to(device), targets.to(device)
                    
                    # Flatten the data if needed
                    if len(data.shape) > 2:
                        data = data.reshape(data.shape[0], -1)
                    
                    # Forward pass
                    outputs, representations, errors = self.forward(data, return_all=True, return_errors=True)
                    
                    # Compute task loss (classification)
                    task_loss = criterion(outputs, targets)
                    
                    # Compute prediction errors at each layer
                    error_loss = 0.0
                    for error in errors:
                        error_loss += torch.mean(error**2)
                    
                    # Total loss (weighted sum of task loss and prediction errors)
                    loss = task_loss + 0.01 * error_loss
                    
                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update statistics
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                    
                    pbar.set_postfix({
                        "loss": loss.item(),
                        "acc": 100. * correct / total
                    })
            
            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}, Accuracy: {100. * correct / total:.2f}%")
        
        return losses
    
    def evaluate(self, dataloader, device='cpu'):
        """
        Evaluate the HPCN on a test dataset
        
        Args:
            dataloader (DataLoader): DataLoader for the test data
            device (str, optional): Device to evaluate on. Defaults to 'cpu'.
            
        Returns:
            float: Accuracy (for supervised) or reconstruction error (for unsupervised)
        """
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            if self.supervised:
                correct = 0
                total = 0
                
                for data, targets in dataloader:
                    data, targets = data.to(device), targets.to(device)
                    
                    # Flatten the data if needed
                    if len(data.shape) > 2:
                        data = data.reshape(data.shape[0], -1)
                    
                    # Forward pass
                    outputs = self.forward(data)
                    
                    # Compute accuracy
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                
                accuracy = 100. * correct / total
                return accuracy
            else:
                total_error = 0.0
                
                for data, _ in dataloader:
                    data = data.to(device)
                    
                    # Flatten the data if needed
                    if len(data.shape) > 2:
                        data = data.reshape(data.shape[0], -1)
                    
                    # Forward pass with return_errors=True to ensure proper prediction
                    _, representations, _ = self.forward(data, return_all=True, return_errors=True)
                    
                    # Compute reconstruction error
                    input_prediction = self.predict_input(representations[-1])
                    error = F.mse_loss(input_prediction, data)
                    total_error += error.item()
                
                avg_error = total_error / len(dataloader)
                return avg_error
        
    def reconstruct(self, data, device='cpu'):
        """
        Reconstruct input data
        
        Args:
            data (torch.Tensor): Input data
            device (str, optional): Device to use. Defaults to 'cpu'.
            
        Returns:
            torch.Tensor: Reconstructed data
        """
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            data = data.to(device)
            
            # Flatten the data if needed
            if len(data.shape) > 2:
                original_shape = data.shape
                data = data.reshape(data.shape[0], -1)
            else:
                original_shape = None
            
            # Forward pass with return_errors=True to ensure proper prediction
            _, representations, _ = self.forward(data, return_all=True, return_errors=True)
            
            # Reconstruct input
            reconstructions = self.predict_input(representations[-1])
            
            # Reshape back to original shape if needed
            if original_shape is not None:
                reconstructions = reconstructions.reshape(original_shape)
            
            return reconstructions