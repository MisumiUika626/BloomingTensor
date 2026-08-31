from datetime import datetime

import numpy as np

from .models.mlp import MLP
from .nn.linear import Linear
from .optimizers.sgd import SGD


def build_experiment(dataset_name, model_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dataset_name == "linear":
        from .configs import config
        from .datasets.dataset import Dataset

        dataset_kwargs = {}

    elif dataset_name == "nonlinear":
        from .configs import config_nolinear as config
        from .datasets.dataset_nolinear import Dataset

        dataset_kwargs = {}

    elif dataset_name == "xor":
        from .configs import config_xor as config
        from .datasets.dataset_xor import Dataset

        dataset_kwargs = {}

    elif dataset_name == "twomoon":
        from .configs import config_twomoon as config
        from .datasets.dataset_twomoon import Dataset

        dataset_kwargs = {"random_seed": config.RANDOM_SEED}

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    np.random.seed(config.RANDOM_SEED)
    dataset = Dataset(**dataset_kwargs)

    if model_name == "linear":
        model = Linear(config.INPUT_DIM, config.OUTPUT_DIM)
    elif model_name == "mlp":
        layer_sizes = [config.INPUT_DIM, *config.HIDDEN_DIMS, config.OUTPUT_DIM]
        model = MLP(layer_sizes)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    optimizer = SGD(model.parameters(), config.LEARNING_RATE)

    return {
        "dataset_name": dataset_name,
        "model_name": model_name,
        "dataset": dataset,
        "model": model,
        "optimizer": optimizer,
        "epochs": config.EPOCHS,
        "learning_rate": config.LEARNING_RATE,
        "random_seed": config.RANDOM_SEED,
        "loss_curve_filename": (
            f"plots/loss_curve_{dataset_name}_{model_name}_{timestamp}.png"
        ),
        "decision_region_filename": (
            f"plots/decision_region_{dataset_name}_{model_name}_{timestamp}.png"
        ),
    }
