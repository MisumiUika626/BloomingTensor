import argparse
from pathlib import Path

from .autograd.tensor import Tensor
from .experiments import build_experiment
from .trainers.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Run a Growing-AI experiment")
    parser.add_argument(
        "--dataset",
        choices=("linear", "nonlinear", "xor", "twomoon"),
        default="twomoon",
    )
    parser.add_argument(
        "--model",
        choices=("linear", "mlp"),
        default="mlp",
    )
    args = parser.parse_args()

    import matplotlib.pyplot as plt

    experiment = build_experiment(args.dataset, args.model)

    dataset = experiment["dataset"]
    model = experiment["model"]
    optimizer = experiment["optimizer"]
    epochs = experiment["epochs"]

    trainer = Trainer()

    loss_history = trainer.fit(
        model,
        dataset,
        optimizer,
        epochs,
    )
    epoch_numbers = range(1, len(loss_history) + 1)

    plt.plot(epoch_numbers, loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training Loss: {args.dataset} data with {args.model.upper()}")
    plt.yscale("log")
    axes = plt.gca()
    axes.text(
        0.98,
        0.98,
        (
            f"lr = {experiment['learning_rate']}\n"
            f"seed = {experiment['random_seed']}\n"
            f"loss = {loss_history[0]:.4g} → {loss_history[-1]:.4g}"
        ),
        transform=axes.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    loss_curve_path = Path(experiment["loss_curve_filename"])
    loss_curve_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Saved loss curve: {loss_curve_path}")

    print("Final parameters:")

    for parameter in model.parameters():
        print(parameter.data)

    for index in range(len(dataset)):
        x, target = dataset[index]
        x_tensor = Tensor([x])

        prediction = model.forward(x_tensor)

        print(
            "input:",
            x,
            "prediction:",
            prediction.data.item(),
            "target:",
            target,
        )


if __name__ == "__main__":
    main()
