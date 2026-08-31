"""CLI for the separate MNIST ten-class learning track."""

import argparse
from pathlib import Path

import numpy as np

from .configs import config_mnist
from .datasets.dataloader import DataLoader
from .datasets.dataset_mnist import Dataset
from .models.mlp import MLP
from .nn.losses import CrossEntropyLoss
from .optimizers.sgd import SGD
from .trainers.classification_trainer import ClassificationTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train the NumPy MLP on MNIST")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mnist/raw"),
        help="directory containing the four MNIST gzip files",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config_mnist.EPOCHS,
        help="number of training epochs",
    )
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="optional number of training samples for a quick run",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="optional number of test samples for a quick run",
    )
    return parser.parse_args()


def limit_dataset(dataset, limit):
    if limit is None:
        return
    if limit <= 0:
        raise ValueError("dataset limit must be positive")

    dataset.images = dataset.images[:limit]
    dataset.labels = dataset.labels[:limit]


def main():
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError("epochs must be positive")

    np.random.seed(config_mnist.RANDOM_SEED)

    train_dataset = Dataset(
        str(args.data_dir / "train-images-idx3-ubyte.gz"),
        str(args.data_dir / "train-labels-idx1-ubyte.gz"),
    )
    test_dataset = Dataset(
        str(args.data_dir / "t10k-images-idx3-ubyte.gz"),
        str(args.data_dir / "t10k-labels-idx1-ubyte.gz"),
    )

    limit_dataset(train_dataset, args.train_limit)
    limit_dataset(test_dataset, args.test_limit)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_mnist.BATCH_SIZE,
        shuffle=True,
        random_seed=config_mnist.RANDOM_SEED,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config_mnist.BATCH_SIZE,
        shuffle=False,
        random_seed=config_mnist.RANDOM_SEED,
    )

    layer_sizes = [
        config_mnist.INPUT_DIM,
        *config_mnist.HIDDEN_DIMS,
        config_mnist.OUTPUT_DIM,
    ]
    model = MLP(layer_sizes)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), config_mnist.LEARNING_RATE)
    trainer = ClassificationTrainer(criterion)

    print(
        f"MNIST: train={len(train_dataset)}, test={len(test_dataset)}, "
        f"batch_size={config_mnist.BATCH_SIZE}, layers={layer_sizes}"
    )

    history = trainer.fit(
        model,
        train_loader,
        test_loader,
        optimizer,
        epochs=args.epochs,
    )

    print(
        f"Final: test_loss={history['test_loss'][-1]:.4f}, "
        f"test_accuracy={history['test_accuracy'][-1]:.4f}"
    )


if __name__ == "__main__":
    main()
