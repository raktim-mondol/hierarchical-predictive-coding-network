"""
Predictive Coding Layers

This module implements the core layers for the Hierarchical Predictive Coding Network.
Each layer consists of representation units, prediction units, and error units.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PCLayer(nn.Module):
    """
    Predictive Coding Layer
    
    A single layer in the Hierarchical Predictive Coding Network that implements
    the core predictive coding mechanism with representation units, prediction units,
    and error units.
    
    Args:
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden representation
        prediction_steps (int, optional): Number of prediction steps per forward pass. Defaults to 10.
        learning_rate (float, optional): Local learning rate for prediction error minimization. Defaults to 0.1.
        precision (float, optional): Precision (inverse variance) of the prediction errors. Defaults to 1.0.
    """
    
    def __init__(self, input_size, hidden_size, prediction_steps=10, learning_rate=0.1, precision=1.0):
        super(PCLayer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.prediction_steps = prediction_steps
        self.learning_rate = learning_rate
        self.precision = precision
        
        # Feedforward pathway (bottom-up)
        self.feedforward = nn.Linear(input_size, hidden_size)
        
        # Feedback pathway (top-down)
        self.feedback = nn.Linear(hidden_size, input_size)
        
        # Lateral connections for recurrent processing
        self.lateral = nn.Linear(hidden_size, hidden_size)
        
        # Initialize weights
        nn.init.xavier_normal_(self.feedforward.weight)
        nn.init.xavier_normal_(self.feedback.weight)
        nn.init.xavier_normal_(self.lateral.weight)
        
        # Activation function
        self.activation = nn.ReLU()
        
    def forward(self, x, top_down_prediction=None, return_errors=False):
        """
        Forward pass through the predictive coding layer
        
        Args:
            x (torch.Tensor): Input tensor
            top_down_prediction (torch.Tensor, optional): Prediction from the layer above. Defaults to None.
            return_errors (bool, optional): Whether to return prediction errors. Defaults to False.
            
        Returns:
            tuple: (representation, prediction, errors) if return_errors=True, else representation
        """
        batch_size = x.shape[0]
        
        # Initialize representation with feedforward pass
        representation = self.activation(self.feedforward(x))
        
        # Initialize prediction as zeros if no top-down prediction is provided
        if top_down_prediction is None:
            prediction = torch.zeros_like(x)
        else:
            prediction = top_down_prediction
        
        # Store prediction errors for each step
        all_errors = []
        
        # Iterative prediction refinement
        for _ in range(self.prediction_steps):
            # Generate prediction of the input
            prediction = self.feedback(representation)
            
            # Compute prediction error
            error = x - prediction
            all_errors.append(error)
            
            # Update representation based on prediction error
            delta = self.feedforward(error * self.precision)
            representation = representation + self.learning_rate * delta
            
            # Apply lateral/recurrent connections
            representation = representation + self.learning_rate * self.activation(self.lateral(representation))
            
            # Apply activation function
            representation = self.activation(representation)
        
        if return_errors:
            return representation, prediction, torch.stack(all_errors, dim=1)
        else:
            return representation
    
    def predict(self, representation):
        """
        Generate a prediction of the input given a representation
        
        Args:
            representation (torch.Tensor): Representation tensor
            
        Returns:
            torch.Tensor: Prediction of the input
        """
        return self.feedback(representation)


class PCLayerWithErrorUnits(PCLayer):
    """
    Predictive Coding Layer with Explicit Error Units
    
    An extension of the PCLayer that explicitly models error units as separate
    neural populations, more closely matching the biological implementation.
    
    Args:
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden representation
        prediction_steps (int, optional): Number of prediction steps per forward pass. Defaults to 10.
        learning_rate (float, optional): Local learning rate for prediction error minimization. Defaults to 0.1.
        precision (float, optional): Precision (inverse variance) of the prediction errors. Defaults to 1.0.
    """
    
    def __init__(self, input_size, hidden_size, prediction_steps=10, learning_rate=0.1, precision=1.0):
        super(PCLayerWithErrorUnits, self).__init__(
            input_size, hidden_size, prediction_steps, learning_rate, precision
        )
        
        # Error units transformation
        self.error_transform = nn.Linear(input_size, input_size)
        nn.init.xavier_normal_(self.error_transform.weight)
        
    def forward(self, x, top_down_prediction=None, return_errors=False):
        """
        Forward pass through the predictive coding layer with explicit error units
        
        Args:
            x (torch.Tensor): Input tensor
            top_down_prediction (torch.Tensor, optional): Prediction from the layer above. Defaults to None.
            return_errors (bool, optional): Whether to return prediction errors. Defaults to False.
            
        Returns:
            tuple: (representation, prediction, errors) if return_errors=True, else representation
        """
        batch_size = x.shape[0]
        
        # Initialize representation with feedforward pass
        representation = self.activation(self.feedforward(x))
        
        # Initialize prediction as zeros if no top-down prediction is provided
        if top_down_prediction is None:
            prediction = torch.zeros_like(x)
        else:
            prediction = top_down_prediction
        
        # Store prediction errors for each step
        all_errors = []
        
        # Iterative prediction refinement
        for _ in range(self.prediction_steps):
            # Generate prediction of the input
            prediction = self.feedback(representation)
            
            # Compute prediction error through error units
            raw_error = x - prediction
            error = self.error_transform(raw_error)  # Error units processing
            all_errors.append(error)
            
            # Update representation based on prediction error
            delta = self.feedforward(error * self.precision)
            representation = representation + self.learning_rate * delta
            
            # Apply lateral/recurrent connections
            representation = representation + self.learning_rate * self.activation(self.lateral(representation))
            
            # Apply activation function
            representation = self.activation(representation)
        
        if return_errors:
            return representation, prediction, torch.stack(all_errors, dim=1)
        else:
            return representation


class LocalHebbianPCLayer(PCLayer):
    """
    Predictive Coding Layer with Local Hebbian Learning
    
    A variant of the PCLayer that uses local Hebbian learning rules instead of
    backpropagation for weight updates, making it more biologically plausible.
    
    Args:
        input_size (int): Size of the input features
        hidden_size (int): Size of the hidden representation
        prediction_steps (int, optional): Number of prediction steps per forward pass. Defaults to 10.
        learning_rate (float, optional): Local learning rate for prediction error minimization. Defaults to 0.1.
        precision (float, optional): Precision (inverse variance) of the prediction errors. Defaults to 1.0.
        hebbian_lr (float, optional): Learning rate for Hebbian updates. Defaults to 0.01.
    """
    
    def __init__(self, input_size, hidden_size, prediction_steps=10, learning_rate=0.1, 
                 precision=1.0, hebbian_lr=0.01):
        super(LocalHebbianPCLayer, self).__init__(
            input_size, hidden_size, prediction_steps, learning_rate, precision
        )
        
        self.hebbian_lr = hebbian_lr
        
    def hebbian_update(self, pre, post):
        """
        Perform a Hebbian weight update
        
        Args:
            pre (torch.Tensor): Pre-synaptic activity
            post (torch.Tensor): Post-synaptic activity
            
        Returns:
            torch.Tensor: Weight update
        """
        # Simple Hebbian rule: weight change proportional to product of pre and post activity
        return torch.bmm(post.unsqueeze(2), pre.unsqueeze(1)).mean(0)
    
    def local_learning_step(self, x, representation, prediction):
        """
        Perform a local learning step using Hebbian updates
        
        Args:
            x (torch.Tensor): Input tensor
            representation (torch.Tensor): Current representation
            prediction (torch.Tensor): Current prediction
        """
        # Compute prediction error
        error = x - prediction
        
        # Update feedback weights (top-down) using Hebbian learning
        feedback_update = self.hebbian_update(representation, error)
        self.feedback.weight.data -= self.hebbian_lr * feedback_update
        
        # Update feedforward weights (bottom-up) using Hebbian learning
        feedforward_update = self.hebbian_update(x, representation)
        self.feedforward.weight.data += self.hebbian_lr * feedforward_update
        
        # Update lateral weights using Hebbian learning
        lateral_update = self.hebbian_update(representation, representation)
        self.lateral.weight.data += self.hebbian_lr * (lateral_update - 0.01 * self.lateral.weight.data)  # With weight decay