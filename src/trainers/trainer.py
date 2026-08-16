# trainers/trainer.py
class Trainer:
    def compute_loss(self, prediction, target):
        # (1)the trainer needs the loss funtion
        squared_error = (target - prediction) ** 2  # Example squared loss function
        return squared_error.mean()
