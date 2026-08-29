# Growing-AI Architecture

## Principles

- Prefer simple, understandable designs with clear responsibilities.
- Separate data, computation, training, and parameter updates.
- Preserve learning value; avoid premature abstraction and hypothetical systems.

## Current Architecture Snapshot

```text
src/
├── agent/          # Agent experiments (currently a placeholder)
├── autograd/       # Scalar and Tensor automatic differentiation
├── configs/        # Per-experiment training configuration
├── datasets/       # Training data access
├── experiments.py  # Named experiment assembly
├── models/         # Learnable models such as Linear
├── nn/             # Neural-network components (currently a placeholder)
├── optimizers/     # Parameter update rules such as SGD
├── trainers/       # Loss and training responsibilities
└── main.py         # Command-line experiment entry point

tests/              # Autograd, model, and trainer tests
```

The current data flow is:

```text
Experiment name → Experiment assembly → Trainer
                                      ↓
Dataset → Model forward → Loss → Backward → Gradients → Optimizer update
```

`build_experiment(name)` creates the dataset, model, optimizer, training
configuration, and output filename for `linear`, `nonlinear`, `xor`, or
`twomoon`. `Trainer` receives the training objects through their shared
interfaces and does not branch on the experiment name.

## Evolution Policy

This structure is a snapshot of the current learning stage, not a permanent
contract. Architecture should evolve incrementally with the learning goals.
Modules may be added, merged, renamed, or removed when responsibilities change
or the current structure blocks understanding, testing, or extension.

Keep this snapshot aligned with significant structural changes, without
designing complex systems before they are implemented.
