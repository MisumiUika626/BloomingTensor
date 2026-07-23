from .trainers.trainer import Trainer
from .optimizers.sgd import SGD
from .models.linear import Linear
from .datasets.dataset import Dataset
from .config import INPUT_DIM, LEARNING_RATE, EPOCHS

model = Linear(
            INPUT_DIM
        )

trainer=Trainer(model)
optimizer=SGD(model,LEARNING_RATE)
def main():
    dataset = Dataset()
    for epoch in range(EPOCHS):
        total_loss=0
        for index in range(len(dataset)):
            x, target = dataset[index]
            prediction=model.forward(x)
            loss=trainer.compute_loss(
                prediction,
                target
            )
            total_loss+=loss
            gradients=trainer.compute_gradient(
                x,
                prediction,
                target
            )
            optimizer.step(gradients)
            average_loss = total_loss / len(dataset)
            print(
                f"Epoch {epoch + 1}, "
                #f"Sample {index + 1}, "
                f"Loss: {average_loss}"
            )

        



if __name__=="__main__":
    main()
