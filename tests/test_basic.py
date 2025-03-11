"""
Basic tests for the Hierarchical Predictive Coding Network.
"""

import torch
import pytest
import sys
import os

# Add parent directory to path to import hpcn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hpcn.model import HPCN, HPCNWithTemporalDynamics
from hpcn.layers import PCLayer, PCLayerWithErrorUnits, LocalHebbianPCLayer


def test_pc_layer():
    """Test PCLayer forward pass"""
    batch_size = 10
    input_size = 20
    hidden_size = 10
    
    # Create layer
    layer = PCLayer(input_size, hidden_size)
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, hidden_size)
    
    # Test with error return
    output, prediction, errors = layer(x, return_errors=True)
    
    # Check shapes
    assert output.shape == (batch_size, hidden_size)
    assert prediction.shape == (batch_size, input_size)
    assert errors.shape[0] == batch_size
    assert errors.shape[2] == input_size


def test_pc_layer_with_error_units():
    """Test PCLayerWithErrorUnits forward pass"""
    batch_size = 10
    input_size = 20
    hidden_size = 10
    
    # Create layer
    layer = PCLayerWithErrorUnits(input_size, hidden_size)
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, hidden_size)
    
    # Test with error return
    output, prediction, errors = layer(x, return_errors=True)
    
    # Check shapes
    assert output.shape == (batch_size, hidden_size)
    assert prediction.shape == (batch_size, input_size)
    assert errors.shape[0] == batch_size
    assert errors.shape[2] == input_size


def test_local_hebbian_pc_layer():
    """Test LocalHebbianPCLayer forward pass"""
    batch_size = 10
    input_size = 20
    hidden_size = 10
    
    # Create layer
    layer = LocalHebbianPCLayer(input_size, hidden_size)
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = layer(x)
    
    # Check output shape
    assert output.shape == (batch_size, hidden_size)
    
    # Test local learning step
    representation = output
    prediction = layer.predict(representation)
    layer.local_learning_step(x, representation, prediction)


def test_hpcn_model():
    """Test HPCN model forward pass"""
    batch_size = 10
    input_size = 28 * 28  # MNIST-like
    
    # Create layers
    layers = [
        PCLayer(input_size=input_size, hidden_size=400),
        PCLayer(input_size=400, hidden_size=200),
        PCLayer(input_size=200, hidden_size=100)
    ]
    
    # Create model
    model = HPCN(layers)
    
    # Create input
    x = torch.randn(batch_size, input_size)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 100)
    
    # Test with all returns
    output, representations, errors = model(x, return_all=True, return_errors=True)
    
    # Check shapes
    assert output.shape == (batch_size, 100)
    assert len(representations) == 3
    assert len(errors) == 3
    
    # Test predict_input
    reconstruction = model.predict_input(representations[-1])
    assert reconstruction.shape == (batch_size, input_size)


def test_hpcn_with_temporal_dynamics():
    """Test HPCNWithTemporalDynamics forward pass"""
    batch_size = 10
    seq_length = 5
    n_features = 1
    
    # Create layers
    layers = [
        PCLayer(input_size=n_features, hidden_size=32),
        PCLayer(input_size=32, hidden_size=16),
        PCLayer(input_size=16, hidden_size=8)
    ]
    
    # Create model
    model = HPCNWithTemporalDynamics(
        layers, 
        supervised=True, 
        output_size=n_features,
        temporal_window=3
    )
    
    # Create input
    x = torch.randn(batch_size, seq_length, n_features)
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, seq_length, n_features)
    
    # Test with all returns
    output, representations, errors = model(x, return_all=True, return_errors=True)
    
    # Check shapes
    assert output.shape == (batch_size, seq_length, n_features)
    assert len(representations) == seq_length
    assert len(errors) == seq_length


if __name__ == "__main__":
    # Run tests
    test_pc_layer()
    test_pc_layer_with_error_units()
    test_local_hebbian_pc_layer()
    test_hpcn_model()
    test_hpcn_with_temporal_dynamics()
    print("All tests passed!")