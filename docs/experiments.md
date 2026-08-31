# Experiments and Showcase Guide

This document separates experiments that are reproducible in the current repository from experiments that are only candidates for future work. The current command-line interface accepts a dataset and model independently:

```bash
python3 -m src.main --dataset DATASET --model MODEL
```

`DATASET` is one of `linear`, `nonlinear`, `xor`, or `twomoon`. `MODEL` is `linear` or `mlp`.

## What the current runner measures

`Trainer.fit()` performs per-sample SGD. After each epoch it freezes the updated parameters, evaluates MSE over the full training dataset, and appends that average to the loss history. `src.main` plots this history on a logarithmic y-axis and saves it under `plots/`.

For XOR and Two Moons, the runner also evaluates the trained model over a two-dimensional grid. It saves the raw prediction surface, training samples, and a black `0.5` inspection boundary as a second timestamped figure.

The runner does not create a held-out split or report validation/test metrics. For XOR and Two Moons, the plotted `0.5` boundary is only a way to inspect the raw scalar outputs; those outputs are trained with MSE and are not calibrated probabilities.

## Reproducible experiments

| Dataset | Shape and target | Default configuration | Main question |
| --- | --- | --- | --- |
| `linear` | 3 samples, `x: (3,)`, scalar target | 100 epochs, lr `0.033`, seed `42` | Does the complete Tensor → Linear → MSE → backward → SGD path train? |
| `nonlinear` | 9 samples, `x: (1,)`, `y = x^2` | 500 epochs, lr `0.01`, seed `42` | Can a hidden layer represent a relation that one affine layer cannot? |
| `xor` | 4 samples, `x: (2,)`, label `0/1` | 500 epochs, lr `0.01`, seed `42` | Can the MLP fit a non-linearly separable label pattern? |
| `twomoon` | 200 samples, `x: (2,)`, label `0/1`, noise `0.15` | 500 epochs, lr `0.01`, seed `38` | How do Linear and MLP behave on a curved noisy boundary? |

### Linear versus MLP on nonlinear regression

```bash
python3 -m src.main --dataset nonlinear --model linear
python3 -m src.main --dataset nonlinear --model mlp
```

This is the cleanest capacity comparison in the repository because the target is explicitly `y = x^2`. Keep dataset, seed, learning rate, and epoch budget fixed when comparing the two models.

### XOR

```bash
python3 -m src.main --dataset xor --model linear
python3 -m src.main --dataset xor --model mlp
```

XOR provides only four points, so it is a mechanism check rather than evidence of generalization. The useful observation is whether the hidden layer can fit all four labels while a single affine boundary cannot under the same experiment configuration.

### Two Moons

```bash
python3 -m src.main --dataset twomoon --model linear
python3 -m src.main --dataset twomoon --model mlp
```

The dataset generator is implemented with NumPy inside the repository. It creates two noisy interleaving half-circles, shuffles them reproducibly, and does not require scikit-learn.

## Verified snapshot

The following values were reproduced from the current checkout on 2026-08-31. They are training-set observations from one fixed seed, not benchmark or generalization claims.

| Dataset | Model | First-epoch MSE | Final MSE | Training threshold accuracy |
| --- | --- | ---: | ---: | ---: |
| `nonlinear` | Linear | 4.0878 | 2.1563 | not applicable |
| `nonlinear` | MLP | 2.6699 | 0.0129 | not applicable |
| `xor` | Linear | 0.4173 | 0.2501 | 25.0% |
| `xor` | MLP | 0.3715 | 0.0002 | 100.0% |
| `twomoon` | Linear | 0.1401 | 0.0940 | 86.5% |
| `twomoon` | MLP | 0.1362 | 0.0239 | 99.0% |

Threshold accuracy uses `prediction >= 0.5`. Two Moons raw predictions exceeded the `[0, 1]` interval in this run, which is expected because the current output layer is linear and the loss is MSE.

## How to present these experiments honestly

For each figure or written result, record:

- dataset and model;
- layer sizes;
- random seed;
- learning rate and epoch count;
- loss definition;
- whether the metric is training, validation, or test; and
- what the result does not establish.

A decreasing training loss shows that the update path improves fit to the training samples. It does not by itself establish robustness, calibrated probabilities, or generalization.

## Candidate experiments — not implemented

These experiments match the current project's learning goals, but they should remain labelled as planned until their datasets, runners, tests, and metrics exist in the repository.

### 1. Centered finite-difference gradient checking

Compare selected autograd results with

```text
(f(x + h) - f(x - h)) / (2h)
```

This would directly test the most important mechanism in the project without adding a new model family. Start with scalar arithmetic, broadcasting, and one `Linear` parameter before attempting a whole-network check. The centered formula and practical cautions are described in the [CS231n gradient-checking notes](https://cs231n.github.io/neural-networks-3/).

### 2. Noisy regression with known coefficients

Generate a small NumPy dataset from known weights and controlled Gaussian noise, then compare learned and true coefficients on a held-out split. This would extend the current three-sample linear sanity check into an experiment about parameter recovery and generalization. Scikit-learn's official [`make_regression` documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_regression.html) provides a reference design; the repository can keep its own small NumPy implementation.

### 3. Concentric circles

Add a two-dimensional inner/outer-circle dataset and compare Linear with MLP under the same seed and training budget. It exercises a different nonlinear geometry from XOR and Two Moons while still fitting the current two-input, scalar-output model interface. The dataset shape is documented by scikit-learn's official [`make_circles` reference](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_circles.html).

For a defensible classification result, implement BCE and a held-out split before treating threshold accuracy as more than an inspection metric.

### 4. Seed, width, and learning-rate ablations

Run the existing nonlinear, XOR, and Two Moons experiments across several seeds while changing one variable at a time. Report the distribution of final training loss and, after a held-out split exists, test accuracy. This requires an aggregation runner but no new autograd primitive.

### 5. Multiclass spiral data

A spiral dataset is a useful visual test of nonlinear decision boundaries, but the standard experiment is multiclass. It should follow, not precede, stable multi-class logits, Softmax/CrossEntropy, integer targets, and accuracy. The [CS231n neural-network case study](https://cs231n.github.io/neural-networks-case-study/) is a reference for the data shape and the linear-versus-MLP comparison.

## Recommended order

1. Add finite-difference checks for the current autograd engine.
2. Add a held-out split and honest evaluation to the existing regression and binary-labelled experiments.
3. Implement BCE, then add concentric circles as another binary geometry.
4. Implement and test stable CrossEntropy before attempting multiclass spiral data or MNIST.
