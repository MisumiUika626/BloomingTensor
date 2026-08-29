from ..autograd.tensor import Tensor


class Trainer:
    def compute_loss(self, prediction, target):
        squared_error = (target - prediction) ** 2
        return squared_error.mean()

    def fit(self, model, dataset, optimizer, epochs):
        loss_history = []

        for epoch in range(epochs):
            # 训练
            for index in range(len(dataset)):
                optimizer.zero_grad()

                x, target = dataset[index]
                x_tensor = Tensor([x])
                target_tensor = Tensor([[target]])

                prediction = model.forward(x_tensor)
                loss = self.compute_loss(prediction, target_tensor)

                loss.backward()
                optimizer.step()

            # 评估：不反向传播、不更新参数
            total_loss = 0.0

            for index in range(len(dataset)):
                x, target = dataset[index]
                x_tensor = Tensor([x])
                target_tensor = Tensor([[target]])

                prediction = model.forward(x_tensor)
                sample_loss = self.compute_loss(
                    prediction,
                    target_tensor,
                )

                total_loss += float(sample_loss.data)

            average_loss = total_loss / len(dataset)
            loss_history.append(average_loss)

            print(f"Epoch {epoch + 1}, Loss: {average_loss}")

        return loss_history
