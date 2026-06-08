from nn_blocks import MicroNeuron
from engine import Value

if __name__ == "__main__":
    print("📈 Activating GradientAutodiff-Py Micro-Optimization Stream...")
    
    neuron = MicroNeuron(2)
    inputs = [Value(2.0), Value(-1.5)]
    
    # 1. Forward Pass Execution Step
    prediction = neuron.forward(inputs)
    
    # 2. Compute Mean Squared Error Loss Target manually relative to ground truth (target = 1.0)
    target = Value(1.0)
    loss = (prediction + Value(-1.0 * target.data)) * (prediction + Value(-1.0 * target.data))
    
    print(f"📊 Forward Path Prediction Vector Scalar: {prediction.data}")
    print(f"📉 Calculated Operational Loss Value: {loss.data}")
    
    # 3. Dynamic Backpropagation
    loss.backward()
    
    print(f"✨ Computed Bias Attribute Gradient Correction Factor: {neuron.bias.grad}")
    print(f"✨ Weight [0] Core Calculus Gradient Matrix Scale: {neuron.weights[0].grad}")
