class SGD:
    def __init__ (self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate
    def step(self):
        for parameter in self.model.parameters():
            parameter.data-=(
                parameter.grad*self.learning_rate
            )

    def zero_grad(self):
        for parameter in self.model.parameters():
            parameter.grad = 0.0
