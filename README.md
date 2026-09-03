# BloomingTensor

Growing-AI is a small, NumPy-first learning project for understanding how automatic differentiation and neural-network training work below framework-level APIs. The repository now has two deliberately separate learning tracks built on one shared `Tensor`/`MLP`/`SGD` core:

| Track | Data and output | Loss and evaluation | Entry point |
| --- | --- | --- | --- |
| Toy foundations | Small 1-D/2-D datasets, usually one scalar output | Per-sample MSE and training-set inspection | `python3 -m src.main` |
| MNIST classification | `(B,784) -> (B,10)` logits | Mini-batch CrossEntropy and held-out test accuracy | `python3 -m src.main_mnist` |

The toy track preserves the first end-to-end experiments as a regression bench. The MNIST track adds a new classification contract; it does not replace or silently modify the old experiments.

This is an educational implementation, not a production deep-learning framework or a replacement for PyTorch.

## Why this project?

High-level frameworks make model construction convenient, but that convenience can hide the mechanisms that make training possible. This project implements a deliberately small stack so those mechanisms remain inspectable:

- how operations create a computation graph;
- how local derivatives compose through the chain rule;
- why gradients from multiple graph paths must accumulate;
- how broadcasted values receive gradients in their original shapes;
- how layers expose parameters to an optimizer; and
- how a training loop connects data, loss, `backward()`, and SGD.

NumPy provides array storage and numerical kernels. The graph construction, gradient rules, traversal, layer composition, loss, optimizer, datasets, and training loop are implemented in this repository.

## What is implemented

### Automatic differentiation

- A scalar `Value` engine for basic arithmetic and reverse-mode differentiation.
- A NumPy-backed `Tensor` with `float64` data and same-shaped gradient storage.
- Arithmetic operations: addition, subtraction, negation, multiplication, division, and scalar powers.
- Two-dimensional matrix multiplication.
- Reductions with `sum()` and `mean()`; the current branch also supports `sum(axis=..., keepdims=...)`.
- Differentiable `exp()`, `log()`, ReLU, and Leaky ReLU operations.
- Reverse topological traversal from a scalar output.
- Gradient accumulation when a value contributes through multiple graph paths.
- Gradient shape reduction for broadcasted addition and multiplication.

### Neural-network and training components

- `Linear`: affine transformation `x @ weight + bias` with parameter discovery.
- `Sequential`: ordered forward composition and flattened parameter collection.
- `MLP`: configurable stacks of `Linear` layers and hidden `LeakyReLU` activations.
- `LeakyReLU`, composition-based `Sigmoid`, and numerically stable row-wise `Softmax` activation objects.
- `ToyTrainer` for the original per-sample MSE experiments.
- Stable multi-class `CrossEntropyLoss`, integer targets, `argmax` accuracy, and `ClassificationTrainer`.
- A mini-batch `DataLoader` with shuffling and reproducible seeds.
- Explicit `zero_grad()` and `step()` operations in SGD.
- Loss-curve plotting for toy experiments, two-dimensional toy prediction surfaces, and MNIST sample visualization.

### Toy foundation experiments

| Dataset | Purpose | Current setup |
| --- | --- | --- |
| `linear` | End-to-end affine training sanity check | Three samples with three input features |
| `nonlinear` | Compare Linear and MLP capacity on `y = x^2` | Nine one-dimensional samples |
| `xor` | Show the limitation of a linear boundary and the effect of a hidden layer | Four binary-labelled points |
| `twomoon` | Learn a curved boundary on a reproducible noisy toy dataset | 200 two-dimensional samples |

The XOR and Two Moons runs currently optimize raw scalar outputs with MSE. A value of `0.5` can be used as an inspection threshold, but the repository does not yet provide a complete binary-classification pipeline with BCE, calibrated probabilities, a train/test split, or generalization metrics.

See [docs/experiments.md](docs/experiments.md) for commands, verified observations, interpretation limits, and suitable next experiments.

### MNIST classification track

MNIST is kept separate from the toy runner. Its current path is:

```text
IDX gzip files
→ normalized images (B,784) and integer targets (B,)
→ DataLoader
→ MLP [784,128,10]
→ stable CrossEntropyLoss
→ ClassificationTrainer
→ held-out test loss and accuracy
```

The standard 60,000/10,000 train/test files are downloaded locally under `data/mnist/raw/` and ignored by Git. Quick-run limits are available so the complete pipeline can be checked without starting full training.

### Experiment snapshots

The figures below were generated with the repository's default MLP configurations. They show training samples, raw model output, and the `0.5` inspection boundary; they are not test-set evaluations.

**XOR — four training points**

![MLP raw prediction surface on the XOR dataset](docs/assets/xor_mlp_prediction_surface.png)

**Two Moons — 200 noisy training points**

![MLP raw prediction surface on the Two Moons dataset](docs/assets/twomoon_mlp_prediction_surface.png)

## How it works

### Computation graph

Each operation performs its NumPy forward calculation immediately and returns a new `Tensor`. The result stores its parent tensors in `_prev`, an operation label in `_op`, and a local `_backward` closure containing that operation's derivative rule.

For example, a linear layer creates this data flow:

```text
X: (batch, in_features)
  @ W: (in_features, out_features)
  + b: (out_features,)
  -> Y: (batch, out_features)
```

The bias is broadcast in the forward pass. During backward propagation, its gradient is summed back to `(out_features,)`.

### Backward and gradient propagation

`Tensor.backward()` only accepts a scalar output. It recursively builds a topological ordering of the reachable graph, clears gradients within that graph, seeds the output gradient with `1`, and executes local backward functions in reverse order.

Local rules add into parent gradients with `+=`. This is essential for expressions such as `x * x + x`, where the same tensor affects the result through more than one path. Clearing the reachable graph before each pass also makes repeated calls on the same loss produce the same gradients instead of accumulating across separate backward calls.

### Broadcasting and shapes

NumPy performs broadcasting during forward addition and multiplication. `_unbroadcast_grad()` reverses that shape expansion by summing extra leading axes and axes whose original dimension was `1`. Matrix multiplication is intentionally limited to two-dimensional tensors, keeping its gradient rules explicit:

```text
dX = dY @ W.T
dW = X.T @ dY
```

## Quick start

Python 3.9 or newer is required by the current type-hint syntax. The repository is not packaged, so run commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m unittest discover -s tests -v
```

Run a toy foundation experiment by selecting a dataset and model independently:

```bash
python3 -m src.main --dataset nonlinear --model linear
python3 -m src.main --dataset nonlinear --model mlp
python3 -m src.main --dataset xor --model mlp
python3 -m src.main --dataset twomoon --model mlp
```

Each command prints epoch losses, final parameters, and predictions, then saves a timestamped log-scale loss curve under `plots/`. XOR and Two Moons also save a prediction-surface figure with the raw model output and a `0.5` inspection boundary. If Matplotlib cannot use its default cache directory, set `MPLCONFIGDIR` to a writable directory before running the command.

Run a small MNIST pipeline check:

```bash
python3 -m src.main_mnist --epochs 2 --train-limit 512 --test-limit 256
```

Run the configured full MNIST experiment:

```bash
python3 -m src.main_mnist
```

Save one MNIST sample as a PNG:

```bash
MPLCONFIGDIR=/tmp/growing-ai-matplotlib \
python3 -m src.show_mnist --split train --index 0
```

## Minimal autograd example

This example follows the current API from tensor creation through a scalar loss and backward propagation:

```python
from src.autograd.tensor import Tensor

x = Tensor([[1.0, 2.0], [3.0, 4.0]])
weight = Tensor([[2.0], [-1.0]])
bias = Tensor([0.5])
target = Tensor([[0.0], [0.0]])

prediction = x @ weight + bias
loss = ((prediction - target) ** 2).mean()
loss.backward()

print(loss.data)     # 3.25
print(weight.grad)   # [[ 8.], [11.]]
print(bias.grad)     # [3.]
```

## Project structure

```text
.
├── src/
│   ├── autograd/
│   │   ├── engine.py       # Scalar Value graph and backward pass
│   │   └── tensor.py       # NumPy Tensor operations and gradients
│   ├── nn/                 # Linear, activations, and Sequential
│   ├── models/mlp.py       # MLP assembly from reusable layers
│   ├── datasets/           # Toy datasets, MNIST IDX loader, and DataLoader
│   ├── configs/            # Separate toy and MNIST experiment settings
│   ├── optimizers/sgd.py   # Parameter updates and gradient reset
│   ├── trainers/
│   │   ├── trainer.py                # ToyTrainer: per-sample MSE
│   │   └── classification_trainer.py # Mini-batch CrossEntropy
│   ├── experiments.py      # Toy dataset/model/config assembly only
│   ├── main.py             # Toy experiment CLI
│   ├── main_mnist.py       # MNIST classification CLI
│   └── show_mnist.py       # MNIST sample PNG helper
├── tests/                  # Autograd, components, losses, metrics, and batches
└── docs/                   # Architecture, experiment, roadmap, and study notes
```

The module boundaries are intentionally simple: datasets provide samples, models perform forward computation, the trainer owns the training sequence, and the optimizer only updates parameters.

## Tests

The current suite contains 54 `unittest` cases. It checks:

- scalar arithmetic, chain rule behavior, and shared-node accumulation;
- Tensor arithmetic, scalar-backward enforcement, and repeatable backward passes;
- broadcasted addition and multiplication gradients;
- matrix-multiplication values, gradient values, and gradient shapes;
- reduction, ReLU, Leaky ReLU, Softmax, exponential, and logarithm behavior;
- Linear input validation, forward values, backward values, and parameter identity;
- MLP output shape, parameter collection, and seeded initialization;
- MSE and CrossEntropy forward/backward behavior;
- mini-batch boundaries and deterministic shuffling; and
- multi-class accuracy for correct, partial, and incorrect predictions;
- classification training updates, evaluation immutability, and history recording.

Run the full suite with:

```bash
python3 -m unittest discover -s tests -v
```

## Current limitations

- This is a learning implementation with a small API and no compatibility or performance guarantees.
- `Tensor.backward()` requires a scalar root; custom upstream gradients are not supported.
- Matrix multiplication supports only two-dimensional tensors.
- The toy trainer intentionally remains online and MSE-based; MNIST uses a separate mini-batch classification trainer.
- BCE is not implemented; XOR and Two Moons remain raw-output MSE demonstrations.
- Binary-labelled experiments use raw outputs rather than probability-calibrated predictions.
- Toy experiment reporting is training-set only. MNIST uses its standard held-out test set but has no validation split.
- There is no model serialization, device abstraction, packaging metadata, or CI configuration.
- Type hints exist in a few interfaces but are not comprehensive.

## Roadmap

The next steps are deliberately limited to gaps visible in the current code and experiment path:

1. Add focused persisted tests for axis-aware `Tensor.sum()` forward and backward behavior.
2. Add MNIST training curves, prediction grids, and a confusion matrix without mixing them into the toy visualizations.
3. Add a properly evaluated binary-classification path with BCE and a held-out split for XOR and Two Moons.
4. Add finite-difference gradient checks for representative Tensor and layer parameters.

Larger topics in the project's learning direction should be added only after their required tensor operations, losses, interfaces, and tests exist.

## Learning goals

The project favors small, inspectable implementations over broad API coverage. A component is useful here when its mathematical rule, data flow, engineering responsibility, and verification method can all be explained from the repository itself.
