from .autograd.tensor import Tensor
from .config import EPOCHS, INPUT_DIM, LEARNING_RATE, OUTPUT_DIM
from .datasets.dataset import Dataset
from .nn.linear import Linear
from .optimizers.sgd import SGD
from .trainers.trainer import Trainer


def main():
    dataset = Dataset()
    model = Linear(INPUT_DIM, OUTPUT_DIM)
    trainer = Trainer(model)
    optimizer = SGD(model, LEARNING_RATE)

    for epoch in range(EPOCHS):
        # 训练
        for index in range(len(dataset)):
            optimizer.zero_grad()

            x, target = dataset[index]
            x_tensor = Tensor([x])
            target_tensor = Tensor([[target]])

            prediction = model.forward(x_tensor)
            loss = trainer.compute_loss(
                prediction,
                target_tensor,
            ).mean()

            loss.backward()
            optimizer.step()

        # 评估：只计算 loss，不反向传播、不更新参数
        total_loss = 0.0

        for index in range(len(dataset)):
            x, target = dataset[index]
            x_tensor = Tensor([x])
            target_tensor = Tensor([[target]])

            prediction = model.forward(x_tensor)
            sample_loss = trainer.compute_loss(
                prediction,
                target_tensor,
            ).mean()

            total_loss += float(sample_loss.data)

        average_loss = total_loss / len(dataset)
        print(f"Epoch {epoch + 1}, Loss: {average_loss}")

    print("Final parameters:")

    for parameter in model.parameters():
        print(parameter.data)

    for index in range(len(dataset)):
        x, target = dataset[index]
        x_tensor = Tensor([x])

        prediction = model.forward(x_tensor)

        print(
            "input:",
            x,
            "prediction:",
            prediction.data.item(),
            "target:",
            target,
        )


if __name__ == "__main__":
    main()
