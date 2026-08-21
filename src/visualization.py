import matplotlib.pyplot as plt
import numpy as np

from .autograd.tensor import Tensor


def plot_binary_decision_region(model, dataset, save_path, title):
    """Plot the prediction surface learned by a binary classifier."""
    samples = np.asarray([dataset[index][0] for index in range(len(dataset))])
    targets = np.asarray([dataset[index][1] for index in range(len(dataset))])

    padding = 0.2
    x1_values = np.linspace(
        samples[:, 0].min() - padding,
        samples[:, 0].max() + padding,
        100,
    )
    x2_values = np.linspace(
        samples[:, 1].min() - padding,
        samples[:, 1].max() + padding,
        100,
    )
    grid_x1, grid_x2 = np.meshgrid(x1_values, x2_values)

    grid_points = np.column_stack((grid_x1.ravel(), grid_x2.ravel()))
    predictions = model.forward(Tensor(grid_points))
    prediction_grid = predictions.data.reshape(grid_x1.shape)

    figure, axes = plt.subplots()
    color_map = axes.contourf(
        grid_x1,
        grid_x2,
        prediction_grid,
        levels=np.linspace(0.0, 1.0, 21),
        cmap="coolwarm",
        extend="both",
    )
    axes.contour(
        grid_x1,
        grid_x2,
        prediction_grid,
        levels=[0.5],
        colors="black",
        linewidths=1.5,
    )
    axes.scatter(
        samples[:, 0],
        samples[:, 1],
        c=targets,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        edgecolors="black",
        s=100,
    )

    axes.set_xlabel("x1")
    axes.set_ylabel("x2")
    axes.set_title(title)
    axes.set_xlim(x1_values.min(), x1_values.max())
    axes.set_ylim(x2_values.min(), x2_values.max())
    axes.set_aspect("equal")
    figure.colorbar(color_map, ax=axes, label="Model prediction")
    figure.tight_layout()
    figure.savefig(save_path)
    plt.close(figure)
