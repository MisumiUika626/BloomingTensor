#models/linear.py
class Linear:
    def __init__(self,input_dim):
        self.weight=[
            0.5 for _ in range(input_dim)
        ]
        self.bias = 0.1

    def forward(self,x):
        if len(x)!=len(self.weight):
            #如果维度不匹配就报错
            raise ValueError(
                f"expected input dimension{len(self.weight)},"
                f"but got {len(x)}"
            )

        result=0

        for w,value in zip(
            self.weight,
            x
        ):
            result += w*value

        result += self.bias

        return result
