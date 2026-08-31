# Growing-AI Roadmap

## Project Direction

Learn AI engineering by implementing core components and connecting them into
small, understandable systems: software engineering → deep learning →
Transformers → agents → reinforcement learning.

## Completed

- Scalar automatic differentiation for basic arithmetic, shared-node gradient accumulation, and reverse topological propagation.
- NumPy-backed Tensor automatic differentiation for arithmetic, broadcasting, two-dimensional matrix multiplication, reductions, `exp`, `log`, ReLU, and Leaky ReLU.
- Linear, Sequential, and MLP components with discoverable parameters.
- Composition-based Sigmoid, MSE training, per-sample SGD, and fixed-parameter epoch evaluation.
- Reproducible linear, `y = x^2`, XOR, and Two Moons datasets with independently selected Linear/MLP models.
- A command-line experiment runner that saves timestamped loss curves and two-dimensional prediction surfaces for XOR and Two Moons.
- Unit tests for current autograd, layer, model, activation, and loss behavior.

## Current Focus

Finish and verify the axis-aware `Tensor.sum(axis, keepdims)` behavior needed by later batch-wise normalization and loss calculations. Preserve the existing scalar experiments as a regression bench while the multi-class path is developed separately.

## Next

1. Add focused forward/backward tests for axis-aware reductions, including restored gradient shapes when `keepdims=False`.
2. Add a binary-classification contract with BCE, explicit threshold accuracy, and a held-out split before presenting XOR or Two Moons as evaluated classifiers.
3. Verify numerically stable Softmax/CrossEntropy on small synthetic logits before adding a multiclass dataset.
4. Add centered finite-difference gradient checks for representative Tensor operations and layer parameters.

Transformer, agent, and reinforcement-learning work remain long-term learning directions, not current repository capabilities.
