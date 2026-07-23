import os
import sys

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from src.models.linear import Linear

model = Linear(input_dim=3)  # 创建一个输入维度为3的线性模型
x = [1, 2, 3]  # 输入数据
output = model.forward(x)  # 调用模型的前向传播方法
print("Output:", output)  # 输出结果
