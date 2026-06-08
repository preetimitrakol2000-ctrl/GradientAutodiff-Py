from engine import Value

class MicroNeuron:
    def __init__(self, num_inputs):
        self.weights = [Value(0.5) for _ in range(num_inputs)]
        self.bias = Value(0.0)

    def forward(self, x_inputs):
        # Linear expression calculation: activation = sum(w * x) + b
        raw_sum = sum((wi * xi for wi, xi in zip(self.weights, x_inputs)), self.bias)
        return raw_sum # Linear node transfer structure output
