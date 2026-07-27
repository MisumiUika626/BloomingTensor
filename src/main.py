if __package__:
    from .trainers.trainer import Trainer
    from .optimizers.sgd import SGD
    from .models.linear import Linear
    from .datasets.dataset import Dataset
    from .config import INPUT_DIM, LEARNING_RATE, EPOCHS
else:
    # 兼容 `python3 src/main.py` 和 IDE 的“运行当前文件”。
    from trainers.trainer import Trainer
    from optimizers.sgd import SGD
    from models.linear import Linear
    from datasets.dataset import Dataset
    from config import INPUT_DIM, LEARNING_RATE, EPOCHS


def main():
    dataset = Dataset()
    model = Linear(INPUT_DIM)
    trainer = Trainer(model)
    optimizer = SGD(model, LEARNING_RATE)

    for epoch in range(EPOCHS):
        for index in range(len(dataset)):
            x, target = dataset[index]
            prediction = model.forward(x)
            gradients = trainer.compute_gradient(
                x,
                prediction,
                target
            )
            optimizer.step(gradients)

        # 固定本轮训练后的参数，再评估整个数据集。
        total_loss = 0
        for index in range(len(dataset)):
            x, target = dataset[index]
            prediction = model.forward(x)
            total_loss += trainer.compute_loss(
                prediction,
                target
            )

        average_loss = total_loss / len(dataset)
        print(
            f"Epoch {epoch + 1}, "
            f"Loss: {average_loss}"
        )


if __name__ == "__main__":
    main()
