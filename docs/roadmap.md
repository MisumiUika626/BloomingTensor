# Growing-AI Roadmap

## Project Direction

Learn AI engineering by implementing core components and connecting them into
small, understandable systems: software engineering → deep learning →
Transformers → agents → reinforcement learning.

## Completed

- A small dataset, linear model, squared-loss trainer, SGD optimizer, and
  training entry point.
- Scalar automatic differentiation for basic arithmetic.
- NumPy-backed Tensor automatic differentiation for arithmetic, broadcasting,
  matrix multiplication, reductions, and ReLU.
- Tests covering the current autograd, model, and trainer components.

## Current Focus

Stabilize the Tensor-based autograd and linear-training path so component
interfaces, shapes, gradients, and tests agree end to end.

## Next

- Resolve gaps between the current model/trainer interfaces and their tests.
- Build the next small neural-network component only after the current training
  path is correct and understandable.
