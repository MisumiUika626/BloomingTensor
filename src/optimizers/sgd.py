class SGD:
    def __init__ (self, model, learning_rate):
        self.model = model
        self.learning_rate = learning_rate

    def step(self, gradients):
        #print(gradients)
        #print(type(gradients["weight"]))
        #print(type(gradients["weight"][0]))

        weight_grads = gradients["weight"]
        bias_grad = gradients["bias"]

        self.model.weight = [
            w - self.learning_rate * gw
            for w,gw in zip(
                self.model.weight,
                weight_grads
            )
        ]

        self.model.bias -= (
        self.learning_rate * bias_grad
    )
