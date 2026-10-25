
import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(x, w, b):
    z = w * x + b
    return sigmoid(z)


def numerical_derivative(x, h=1e-5):
    return (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)


def analytic_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


def relative_error(numerical, analytic):
    denominator = max(abs(numerical), abs(analytic))
    if denominator == 0:
        return 0.0
    return abs(numerical - analytic) / denominator


# Test the sigmoid derivative at three values
test_values = [-3, 0, 2]

h = 1e-5

for z in test_values:
    numerical = numerical_derivative(z, h)
    analytic = analytic_derivative(z)
    error = relative_error(numerical, analytic)

    print(f"z = {z}")
    print(f"Numerical derivative: {numerical:.10f}")
    print(f"Analytic derivative:  {analytic:.10f}")
    print(f"Relative error:       {error:.2e}")
    print()

# Test prediction
p = predict(0, 1, 0)
print(f"Predicted probability: {p:.10f}")
