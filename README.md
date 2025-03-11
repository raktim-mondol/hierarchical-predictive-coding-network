# Hierarchical Predictive Coding Network (HPCN)

A PyTorch implementation of Hierarchical Predictive Coding Networks inspired by neuroscience principles.

## Neurological Theory: Predictive Coding

Predictive Coding is a neuroscience theory proposing that the brain functions as a hierarchical prediction machine. Each level of the cortical hierarchy generates predictions about the activity at lower levels, and discrepancies between these predictions and actual sensory input (prediction errors) drive learning and perception. This iterative process minimizes free energy, continuously refining internal models to better match sensory input.

## Model Architecture

The HPCN implements these principles through:

### 1. Hierarchical Structure
- Multiple layers organized in a hierarchy
- Each layer predicts the activity of the layer below
- Higher layers capture more abstract features

### 2. Dual Pathways
- **Bottom-Up (Feedforward)**: Transmits sensory data or input features upward
- **Top-Down (Feedback)**: Carries predictions downward to lower layers

### 3. Error Units
- Dedicated nodes compute prediction errors at each layer
- Errors drive learning and guide weight adjustments
- Larger errors trigger "surprise" signals, directing computational resources

### 4. Recurrent Processing
- Layers iteratively update predictions via feedback loops
- Mimics cortical recurrent connections for temporal integration
- Allows for refinement of predictions over multiple time steps

## Key Innovations

1. **Unsupervised Learning Efficiency**: Learns latent data structures by predicting inputs, reducing reliance on labeled data
2. **Noise Robustness**: Focuses on significant prediction errors, filtering irrelevant noise
3. **Temporal Processing**: Integrates time through recurrent connections, ideal for sequential data
4. **Dynamic Attention**: Allocates computational resources based on prediction error magnitude
5. **Local Learning Rules**: Uses biologically plausible learning mechanisms that don't rely solely on backpropagation

## Implementation Details

This repository contains a PyTorch implementation of HPCN with:

- Modular layer architecture with configurable parameters
- Support for both supervised and unsupervised learning
- Visualization tools for monitoring prediction errors and layer activations
- Example applications on standard datasets

## Getting Started

### Installation

```bash
git clone https://github.com/raktim-mondol/hierarchical-predictive-coding-network.git
cd hierarchical-predictive-coding-network
pip install -r requirements.txt
```

### Basic Usage

```python
from hpcn.model import HPCN
from hpcn.layers import PCLayer

# Create a simple HPCN with 3 layers
model = HPCN([
    PCLayer(input_size=784, hidden_size=400),
    PCLayer(input_size=400, hidden_size=200),
    PCLayer(input_size=200, hidden_size=100)
])

# Train on MNIST
model.train(train_loader, epochs=10)

# Evaluate
accuracy = model.evaluate(test_loader)
print(f"Test accuracy: {accuracy:.2f}%")
```

## Applications

The HPCN architecture is particularly well-suited for:

1. **Anomaly Detection**: Identifying unusual patterns by monitoring prediction errors
2. **Time Series Prediction**: Leveraging recurrent connections for sequential data
3. **Robust Perception**: Maintaining performance in noisy environments
4. **Generative Modeling**: Creating new samples through iterative refinement

## Theory and Background

The implementation is based on key papers in predictive coding:

- Rao, R. P., & Ballard, D. H. (1999). Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1), 79-87.
- Friston, K. (2005). A theory of cortical responses. Philosophical transactions of the Royal Society B: Biological sciences, 360(1456), 815-836.
- Spratling, M. W. (2017). A review of predictive coding algorithms. Brain and cognition, 112, 92-97.

## License

MIT License