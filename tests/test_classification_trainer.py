import io
import unittest
from contextlib import redirect_stdout

import numpy as np

from src.datasets.dataloader import DataLoader
from src.models.mlp import MLP
from src.nn.losses import CrossEntropyLoss
from src.optimizers.sgd import SGD
from src.trainers.classification_trainer import ClassificationTrainer


class SmallClassificationDataset:
    def __init__(self):
        self.images = np.array(
            [[-2.0, 0.0], [-1.0, 0.2], [1.0, -0.2], [2.0, 0.0]]
        )
        self.labels = np.array([0, 0, 1, 1])

    def __len__(self):
        return len(self.labels)


class TestClassificationTrainer(unittest.TestCase):
    def setUp(self):
        np.random.seed(4)
        self.dataset = SmallClassificationDataset()
        self.model = MLP([2, 4, 2])
        self.criterion = CrossEntropyLoss()
        self.trainer = ClassificationTrainer(self.criterion)

    def test_train_epoch_updates_parameters(self):
        loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        optimizer = SGD(self.model.parameters(), learning_rate=0.05)
        parameters_before = [parameter.data.copy() for parameter in self.model.parameters()]

        average_loss = self.trainer.train_epoch(self.model, loader, optimizer)

        self.assertIsInstance(average_loss, float)
        self.assertTrue(
            all(
                not np.array_equal(before, parameter.data)
                for before, parameter in zip(parameters_before, self.model.parameters())
            )
        )

    def test_evaluate_does_not_change_parameters_or_gradients(self):
        loader = DataLoader(self.dataset, batch_size=3, shuffle=False)
        parameters_before = [parameter.data.copy() for parameter in self.model.parameters()]
        gradients_before = [parameter.grad.copy() for parameter in self.model.parameters()]

        loss, score = self.trainer.evaluate(self.model, loader)

        self.assertIsInstance(loss, float)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        for before, parameter in zip(parameters_before, self.model.parameters()):
            np.testing.assert_array_equal(parameter.data, before)
        for before, parameter in zip(gradients_before, self.model.parameters()):
            np.testing.assert_array_equal(parameter.grad, before)

    def test_fit_records_one_value_per_epoch(self):
        train_loader = DataLoader(
            self.dataset, batch_size=2, shuffle=True, random_seed=4
        )
        test_loader = DataLoader(self.dataset, batch_size=2, shuffle=False)
        optimizer = SGD(self.model.parameters(), learning_rate=0.05)

        with redirect_stdout(io.StringIO()):
            history = self.trainer.fit(
                self.model,
                train_loader,
                test_loader,
                optimizer,
                epochs=2,
            )

        self.assertEqual(set(history), {"train_loss", "test_loss", "test_accuracy"})
        self.assertTrue(all(len(values) == 2 for values in history.values()))


if __name__ == "__main__":
    unittest.main()
