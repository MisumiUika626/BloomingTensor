# Growing-AI Architecture

## Principles

- Prefer simple, understandable designs with clear responsibilities.
- Separate numerical operations, model composition, training, and parameter updates.
- Preserve learning value; avoid premature abstraction and hypothetical systems.

## Current architecture snapshot

```text
src/
├── autograd/
│   ├── engine.py       # Scalar Value automatic differentiation
│   └── tensor.py       # NumPy Tensor operations and reverse-mode gradients
├── nn/                 # Linear, activation, and Sequential components
├── models/mlp.py       # MLP construction from reusable nn components
├── datasets/           # Linear, nonlinear, XOR, and Two Moons samples
├── configs/            # Dataset-specific dimensions and training settings
├── optimizers/sgd.py   # Parameter update and gradient reset
├── trainers/trainer.py # MSE plus train/evaluate loops
├── experiments.py      # Dataset/model/config assembly
├── visualization.py    # Two-dimensional raw-output surfaces
└── main.py             # CLI, reporting, and plot orchestration

tests/                  # Autograd, layer, model, and loss tests
docs/                   # Architecture, experiments, roadmap, and notes
```

## Data flow

```text
dataset name + model name
            ↓
build_experiment(dataset_name, model_name)
            ↓
Dataset + Model + SGD + configuration
            ↓
zero_grad → forward → MSE → backward → SGD step
            ↓
fixed-parameter training-set evaluation → loss history → plots
```

`build_experiment(dataset_name, model_name)` first selects the dataset and its matching configuration, seeds NumPy, then independently constructs either `Linear` or `MLP`. `Trainer` receives objects through their small behavioral interfaces and does not branch on dataset or model names.

## Responsibility boundaries

- `Tensor` owns numerical values, graph edges, local derivative closures, topological traversal, and gradient storage.
- `Linear`, activations, `Sequential`, and `MLP` organize Tensor operations and expose trainable parameters.
- Dataset objects only implement `__len__()` and `__getitem__()`.
- `Trainer` owns the per-sample training sequence and fixed-parameter epoch evaluation.
- `SGD` only clears gradients and updates parameter data.
- `visualization.py` turns a trained two-input model into a raw-output surface without owning training behavior.
- `main.py` owns argument parsing, object orchestration, console output, and plot selection.

The current trainer uses one scalar target per sample and MSE. Multi-class logits, CrossEntropy, mini-batches, data splitting, and evaluation metrics require new explicit contracts rather than hidden branching in the existing trainer.

## Evolution policy

This structure is a snapshot of the current learning stage, not a permanent contract. Modules should be added, merged, renamed, or removed only when a verified learning step changes their responsibilities. Keep this document aligned with significant structural changes without designing systems before they are implemented.
