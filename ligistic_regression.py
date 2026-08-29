import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(x, w, b):
    z= w * x + b
    return z

p = predict(0, 1, 0)


print(sigmoid(p))
