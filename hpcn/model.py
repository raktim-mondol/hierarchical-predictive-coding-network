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
    """Base HPCN implementation"""
    
    def __init__(self, layers, supervised=False, output_size=None):
        super(HPCN, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.supervised = supervised
        if supervised and output_size is not None:
            self.output_layer = nn.Linear(layers[-1].hidden_size, output_size)
            nn.init.xavier_normal_(self.output_layer.weight)
    
    def forward(self, x, return_all=False, return_errors=False):
        batch_size = x.shape[0]
        representations = []
        predictions = []
        errors = []
        
        current_input = x
        for i, layer in enumerate(self.layers):
            if return_errors:
                rep, pred, err = layer(current_input, None, return_errors=True)
                representations.append(rep)
                predictions.append(pred)
                errors.append(err)
            else:
                rep = layer(current_input, None, return_errors=False)
                representations.append(rep)
            current_input = rep
        
        if self.supervised:
            output = self.output_layer(representations[-1])
        else:
            output = representations[-1]
        
        if return_all and return_errors:
            return output, representations, errors
        elif return_all:
            return output, representations
        else:
            return output
    
    def predict_input(self, representation=None, layer_idx=-1):
        if representation is None:
            raise ValueError("Representation must be provided")
        
        current_rep = representation
        for i in range(layer_idx, -1, -1):
            current_rep = self.layers[i].predict(current_rep)
        return current_rep


class HPCNWithTemporalDynamics(HPCN):
    """HPCN with Temporal Dynamics"""
    
    def __init__(self, layers, supervised=False, output_size=None, temporal_window=5):
        super(HPCNWithTemporalDynamics, self).__init__(layers, supervised, output_size)
        self.temporal_window = temporal_window
        self.temporal_memory = None
        self.current_batch_size = None
    
    def _init_temporal_memory(self, batch_size, device):
        """Initialize temporal memory for each layer"""
        self.temporal_memory = []
        for layer in self.layers:
            memory = torch.zeros(
                batch_size,
                self.temporal_window,
                layer.hidden_size,
                device=device
            )
            self.temporal_memory.append(memory)
        self.current_batch_size = batch_size
    
    def _update_temporal_memory(self, layer_idx, new_state):
        """Update temporal memory for a specific layer"""
        # Check if batch size has changed
        if new_state.shape[0] != self.current_batch_size:
            self._init_temporal_memory(new_state.shape[0], new_state.device)
        
        # Shift memory contents
        self.temporal_memory[layer_idx] = torch.roll(self.temporal_memory[layer_idx], shifts=1, dims=1)
        # Update most recent state
        self.temporal_memory[layer_idx][:, 0] = new_state
    
    def reset_temporal_memory(self):
        """Reset temporal memory"""
        self.temporal_memory = None
        self.current_batch_size = None
    
    def forward(self, x, return_all=False, return_errors=False):
        """Forward pass with temporal dynamics"""
        # Handle both 2D and 3D inputs
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add time dimension: [batch_size, 1, features]
        
        batch_size, seq_length, features = x.shape
        device = x.device
        
        # Reset temporal memory if batch size changed
        if self.current_batch_size != batch_size:
            self.reset_temporal_memory()
        
        # Initialize temporal memory if needed
        if self.temporal_memory is None:
            self._init_temporal_memory(batch_size, device)
        
        # Process each time step
        all_outputs = []
        all_representations = []
        all_errors = []
        
        for t in range(seq_length):
            x_t = x[:, t]  # [batch_size, features]
            
            # Lists for current timestep
            representations_t = []
            errors_t = []
            
            # Forward through layers
            current_input = x_t
            for i, layer in enumerate(self.layers):
                # Get temporal context
                if t >= 1 or self.temporal_memory[i][:, 0].sum() != 0:
                    # Compute temporal context as weighted average of past states
                    weights = torch.arange(self.temporal_window, 0, -1, device=device)
                    weights = F.softmax(weights.float(), dim=0)
                    context = torch.sum(self.temporal_memory[i] * weights.view(1, -1, 1), dim=1)
                    
                    # Ensure dimensions match before adding
                    if context.shape == current_input.shape:
                        current_input = current_input + 0.1 * context
                
                # Layer forward pass
                if return_errors:
                    rep, pred, err = layer(current_input, None, return_errors=True)
                    representations_t.append(rep)
                    errors_t.append(err)
                else:
                    rep = layer(current_input, None, return_errors=False)
                    representations_t.append(rep)
                
                # Update temporal memory and prepare next layer input
                self._update_temporal_memory(i, rep.detach())
                current_input = rep
            
            # Handle output
            if self.supervised:
                output_t = self.output_layer(representations_t[-1])
            else:
                output_t = representations_t[-1]
            
            # Store results
            all_outputs.append(output_t)
            if return_errors:
                all_errors.append(errors_t)
            all_representations.append(representations_t[-1])  # Only store the last layer's representation
        
        # Stack outputs and representations along time dimension
        outputs = torch.stack(all_outputs, dim=1)  # [batch_size, seq_length, output_size]
        representations = torch.stack(all_representations, dim=1)  # [batch_size, seq_length, hidden_size]
        
        # Return based on flags
        if return_all and return_errors:
            return outputs, representations, all_errors
        elif return_all:
            return outputs, representations
        else:
            return outputs
    
    def predict_input(self, representation, layer_idx=-1):
        """Predict input from representation, handling temporal sequences"""
        if isinstance(representation, list):
            representation = torch.stack(representation, dim=1)
        
        # If representation is 3D [batch_size, seq_length, hidden_size]
        if len(representation.shape) == 3:
            batch_size, seq_length, hidden_size = representation.shape
            
            # Process each time step
            predictions = []
            for t in range(seq_length):
                rep_t = representation[:, t]  # [batch_size, hidden_size]
                pred_t = super().predict_input(rep_t, layer_idx)
                predictions.append(pred_t)
            
            # Stack predictions along time dimension
            return torch.stack(predictions, dim=1)  # [batch_size, seq_length, input_size]
        else:
            # Handle single time step
            return super().predict_input(representation, layer_idx)
    
    def train_unsupervised(self, dataloader, epochs=10, learning_rate=0.001, device='cpu'):
        """Train with temporal dynamics"""
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            # Reset temporal memory at the start of each epoch
            self.reset_temporal_memory()
            
            with tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}") as pbar:
                for batch_idx, (data, *_) in enumerate(pbar):
                    data = data.to(device)
                    optimizer.zero_grad()
                    
                    # Forward pass
                    _, representations, errors = self.forward(data, return_all=True, return_errors=True)
                    
                    # Compute reconstruction loss
                    input_prediction = self.predict_input(representations)
                    recon_loss = F.mse_loss(input_prediction, data)
                    
                    # Compute prediction errors
                    error_loss = sum(sum(torch.mean(error**2) for error in errors_t) for errors_t in errors)
                    
                    # Total loss
                    loss = recon_loss + 0.1 * error_loss
                    
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    pbar.set_postfix({"loss": loss.item()})
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_epoch_loss:.6f}")
        
        return losses
    
    def evaluate(self, dataloader, device='cpu'):
        """Evaluate with temporal dynamics"""
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            total_error = 0.0
            num_batches = 0
            
            # Reset temporal memory before evaluation
            self.reset_temporal_memory()
            
            for data, *_ in dataloader:
                data = data.to(device)
                if len(data.shape) == 2:
                    data = data.unsqueeze(1)
                
                # Forward pass
                _, representations, _ = self.forward(data, return_all=True, return_errors=True)
                
                # Compute reconstruction error
                input_prediction = self.predict_input(representations)
                error = F.mse_loss(input_prediction, data, reduction='mean')
                total_error += error.item()
                num_batches += 1
            
            avg_error = total_error / num_batches
            return avg_error
    
    def reconstruct(self, data, device='cpu'):
        """Reconstruct temporal sequence"""
        self.to(device)
        self.eval()
        
        with torch.no_grad():
            # Reset temporal memory before reconstruction
            self.reset_temporal_memory()
            
            data = data.to(device)
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            
            outputs, representations, _ = self.forward(data, return_all=True, return_errors=True)
            reconstructions = self.predict_input(representations)
            return reconstructions