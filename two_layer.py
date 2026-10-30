from _linear_regression import dot_product
from neuron import sigmoid
import random
import math

random.seed(0)


X = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]

y = [0, 1, 1, 0]


def forward_pass(x, params):

    w1 = params[0:2]
    b1 = params[2]

    w2 = params[3:5]
    b2 = params[5]

    w3 = params[6:8]
    b3 = params[8]


    a1 = sigmoid(dot_product(x, w1) + b1)
    a2 = sigmoid(dot_product(x, w2) + b2)
    a3 = sigmoid(dot_product([a1,a2], w3) + b3)
    return a1, a2, a3



def backward(x, y, params):
    w1 = params[0:2]
    b1 = params[2]

    w2 = params[3:5]
    b2 = params[5]

    w3 = params[6:8]

    a1, a2, p = forward_pass(x, params)

    # Output neuron delta
    delta3 = (p - y)

    # Hidden neuron deltas
    delta1 = delta3 * w3[0] * a1 * (1 - a1)
    delta2 = delta3 * w3[1] * a2 * (1 - a2)

    # Gradients in the same order as params
    gradients = [
        delta1 * x[0],
        delta1 * x[1],
        delta1,
        delta2 * x[0],
        delta2 * x[1],
        delta2,
        delta3 * a1,
        delta3 * a2,
        delta3
    ]

    return gradients


def train(X, y, params, lr=1, epochs=10000):
    for epoch in range(epochs):
        gradient_totals = [0.0] * len(params)
        total_loss = 0.0

        for x_i, y_i in zip(X, y):
            a1, a2, p = forward_pass(x_i, params)

            epsilon = 1e-15
            p_safe = max(epsilon, min(1 - epsilon, p))
            total_loss += -(
               y_i * math.log(p_safe)
               + (1 - y_i) * math.log(1 - p_safe)
            )

            gradients = backward(x_i, y_i, params)

            for j, grad in enumerate(gradients):
                gradient_totals[j] += grad

        # Average gradients over the four XOR examples
        n = len(X)
        gradient_totals = [g / n for g in gradient_totals]

        # Update all parameters
        for j in range(len(params)):
            params[j] -= lr * gradient_totals[j]

        if (epoch + 1) % 1000 == 0:
            print(
                f"Epoch {epoch + 1}, "
                f"Loss: {total_loss / n:.6f}"
            )
    return params


if __name__ == "__main__":
    params = [
        random.uniform(-1, 1)
        for _ in range(9)
    ]

    print("Initial parameters:", params)

    params = train(X, y, params, lr=1, epochs=10000)

    print("\nFinal predictions:")
    for x_i, y_i in zip(X, y):
        a1, a2, p = forward_pass(x_i, params)

        print(
            f"Input: {x_i}, "
            f"Target: {y_i}, "
            f"a1: {a1:.4f}, "
            f"a2: {a2:.4f}, "
            f"p: {p:.4f}"
        )