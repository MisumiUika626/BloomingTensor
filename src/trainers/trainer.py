# trainers/trainer.py
class Trainer:
    #1.trainer needs the model
    def __init__(self, model):
        self.model =model
    def compute_loss(self,prediction,target):
        # (1)the trainer needs the loss funtion
        loss=(target-prediction)**2  # Example squared loss function
        return loss
    def compute_gradient(self,
                         x,
                         prediction,
                         target):
        # (2)the trainer needs the gradient compuatation
        #gradient=2*(prediction-target)
        #return gradient
        error = prediction-target

        weight_gradient=[
            2*error*xi
            for xi in x
        ]

        bias_gradient=2*error
        return {"weight": weight_gradient, "bias": bias_gradient}
