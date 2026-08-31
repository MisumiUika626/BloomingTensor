"""Save one MNIST sample as an image without starting training."""

import argparse
from pathlib import Path

from .datasets.dataset_mnist import Dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Save one MNIST image as a PNG")
    parser.add_argument("--split", choices=("train", "test"), default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/mnist/raw"),
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    image_prefix = "train" if args.split == "train" else "t10k"
    dataset = Dataset(
        str(args.data_dir / f"{image_prefix}-images-idx3-ubyte.gz"),
        str(args.data_dir / f"{image_prefix}-labels-idx1-ubyte.gz"),
    )

    if not 0 <= args.index < len(dataset):
        raise IndexError(f"index must be between 0 and {len(dataset) - 1}")

    image, label = dataset[args.index]
    output = args.output or Path(f"plots/mnist_{args.split}_{args.index}.png")
    output.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    axes.imshow(image.reshape(28, 28), cmap="gray")
    axes.set_title(f"MNIST {args.split} #{args.index}, label={label}")
    axes.axis("off")
    figure.tight_layout()
    figure.savefig(output)
    plt.close(figure)

    print(f"Saved MNIST image: {output}")


if __name__ == "__main__":
    main()
