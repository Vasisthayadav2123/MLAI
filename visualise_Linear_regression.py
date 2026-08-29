import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



X = [1, 2, 3, 4, 5]

y = [3, 5, 7, 9, 11]

w = 0
b = 0

learning_rate = 0.01

def predict(x, w, b):
    return w * x + b

predictions = [predict(x, w, b) for x in X]

def mse(y_true, predictions):
    total_error = 0

    for actual, prediction in zip(y_true, predictions):
        total_error += (prediction - actual) ** 2

    return total_error / len(y_true)



def gradients(X, y, w, b):
    total_dw = 0
    total_db = 0
    n = len(X)

    for i in range(n):
        prediction = predict(X[i], w, b)
        error = prediction - y[i]
        total_dw += 2 * error * X[i]
        total_db += 2 * error

    dw = total_dw / n
    db = total_db / n

    return dw, db

learning_rate = 0.01
epochs = 30

fig, ax = plt.subplots()


ax.scatter(X, y, label="Actual Data")
line, = ax.plot([], [], label="Neuron Prediction")

ax.set_xlim(min(X) - 1, max(X) + 1)
ax.set_ylim(min(y) - 2, max(y) + 2)

ax.set_xlabel("X")
ax.set_ylabel("y")
ax.legend()


def update(epoch):
    global w, b
    predictions = [predict(x, w, b) for x in X]
    loss = mse(y, predictions)
    dw, db = gradients(X, y, w, b)
    w = w - learning_rate * dw
    b = b - learning_rate * db
    predictions = [predict(x, w, b) for x in X]
    line.set_data(X, predictions)
    ax.set_title(
        f"Epoch: {epoch} | Loss: {loss:.4f} | w: {w:.3f} | b: {b:.3f}"
    )

    return line,


animation = FuncAnimation(
    fig,
    update,
    frames=epochs,
    interval=100,
    repeat=False
)

plt.show()

print("\nFinal Results")
print("w:", w)
print("b:", b)