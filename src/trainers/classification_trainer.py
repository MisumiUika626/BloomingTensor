from ..autograd.tensor import Tensor
from ..metrics import accuracy


class ClassificationTrainer:
    def __init__(self, criterion):
        self.criterion = criterion

    def train_epoch(self, model, data_loader, optimizer):
        total_loss = 0.0
        total_samples = 0

        for images, targets in data_loader:
            optimizer.zero_grad()

            image_tensor = Tensor(images)
            logits = model.forward(image_tensor)
            loss = self.criterion.forward(logits, targets)

            loss.backward()
            optimizer.step()

            batch_size = len(targets)
            total_loss += float(loss.data) * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    def evaluate(self, model, data_loader):
        total_loss = 0.0
        total_correct = 0.0
        total_samples = 0

        for images, targets in data_loader:
            image_tensor = Tensor(images)
            logits = model.forward(image_tensor)
            loss = self.criterion.forward(logits, targets)

            batch_size = len(targets)
            total_loss += float(loss.data) * batch_size
            total_correct += accuracy(logits, targets) * batch_size
            total_samples += batch_size

        average_loss = total_loss / total_samples
        average_accuracy = total_correct / total_samples
        return average_loss, average_accuracy

    def fit(self, model, train_loader, test_loader, optimizer, epochs):
        history = {
            "train_loss": [],
            "test_loss": [],
            "test_accuracy": [],
        }

        for epoch in range(epochs):
            train_loss = self.train_epoch(model, train_loader, optimizer)
            test_loss, test_accuracy = self.evaluate(model, test_loader)

            history["train_loss"].append(train_loss)
            history["test_loss"].append(test_loss)
            history["test_accuracy"].append(test_accuracy)

            print(
                f"Epoch {epoch + 1}/{epochs}, "
                f"train_loss={train_loss:.4f}, "
                f"test_loss={test_loss:.4f}, "
                f"test_accuracy={test_accuracy:.4f}"
            )

        return history
