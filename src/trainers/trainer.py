# trainers/trainer.py
class Trainer:
    #1.trainer needs the model
    def __init__(self, model):
        self.model =model
    def compute_loss(self,prediction,target):
        # (1)the trainer needs the loss funtion
        loss=(target-prediction)**2  # Example squared loss function
        return loss
    