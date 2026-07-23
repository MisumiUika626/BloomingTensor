# Python 工程笔记

## 鸭子类型（Duck Typing）

### 概念

Python 更关注对象是否具有需要的行为，而不强制对象必须属于某个指定类型。

常见表述是：如果一个对象走起来像鸭子、叫起来像鸭子，就可以把它当作鸭子使用。

```python
class Trainer:
    def __init__(self, model):
        self.model = model
```

这里 `Trainer` 不需要导入或检查 `Linear` 类型。只要传入的 `model` 提供 Trainer 所需的属性和方法，就可以协作。

### C++ 类比

鸭子类型接近 C++ 模板的编译期行为：模板通常不关心参数的具体类型，只要求该类型支持模板代码中使用的操作。

```cpp
template <typename Model>
class Trainer {
    Model& model;
};
```

区别是：C++ 模板通常在编译期检查操作是否存在；Python 通常在运行到相关代码时才发现接口不满足。

### 当前项目中的应用

`Trainer` 通过构造函数接收模型：

```python
trainer = Trainer(model)
```

因此不需要在 `trainer.py` 中写：

```python
from models.linear import Linear
```

去掉对 `Linear` 的直接依赖后，未来传入 MLP 或其他模型时，不必因为模型类型变化而修改 Trainer。

### 收益与风险

- 收益：降低模块耦合，方便替换和扩展模型。
- 风险：缺少编译期类型检查；如果对象没有所需接口，错误会在运行时出现。
- 实践：小型项目可保持接口简单；项目变大后可使用类型注解、`Protocol` 或抽象基类明确约束。



## 相对导入中的 `.`

### 含义

导入路径开头的 `.` 表示“从当前包内部查找”。例如 `main.py` 位于 `src` 包中：

```python
from .models.linear import Linear
```

表示从当前的 `src` 包中查找 `models.linear`，而不是查找项目顶层的 `models`。

### C++ 类比

它类似于使用相对于当前模块的 include 路径，明确依赖属于当前工程模块，而不是依赖外部配置碰巧提供搜索路径。

### 运行方式

相对导入需要保留包上下文，应从项目根目录按模块运行：

```bash
python3 -m src.main
```

`-m` 表示运行模块。直接执行 `python3 src/main.py` 时，Python 会把它当作独立脚本，无法确定 `.` 所代表的包。

## 为什么 SGD 的 Loss 不一定单调下降

对单个样本，平方损失是：

```text
L = (prediction - target)²
```

权重梯度是 `2 × error × x`。逐样本 SGD 每次只使用当前样本的梯度，所以一次更新可能降低当前样本的 loss，却升高其他样本的 loss。整个数据集的平均 loss 不保证每一步都下降。

边更新参数边累加 loss，还会混合不同的参数状态：

```text
L1(更新前参数) + L2(第一次更新后参数) + L3(第二次更新后参数)
```

它不是同一组参数上的严格平均损失。正确评估方式是：一个 epoch 更新结束后固定参数，再对全部样本做一次只前向、不更新的计算。

结论：SGD 允许震荡，应观察较长时间的整体下降趋势；Batch Gradient Descent 使用全部样本的平均梯度，在学习率合适时更容易得到单调下降曲线。
