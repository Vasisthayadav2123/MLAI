
import math

x = [1, 2, 3, 4, 5, 6, 7, 8]
y = [0, 0, 0, 1, 0, 1, 1, 1]




def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def predict(x, w, b):
    z = w * x + b
    return sigmoid(z)

def accuracy(x, y, w, b):
    correct = 0
    for i in range(len(x)):
        prediction = predict(x[i], w, b)
        if prediction >= 0.5:
            predicted = 1
        else:
            predicted = 0
        if predicted == y[i]:
            correct += 1
    return correct / len(x)

def numerical_derivative(x, h=1e-5):
    return (sigmoid(x + h) - sigmoid(x - h)) / (2 * h)


def analytic_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def bce_loss( x , y , w , b):
    total_Loss = 0
    for i in range(len(x)):
        prediction = predict(x[i], w, b)
        prediction = max(1e-15, min(1 - 1e-15, prediction))
        if y[i] == 1:
            total_Loss += -math.log(prediction)
        else:    
            total_Loss += -math.log(1 - prediction)
    return total_Loss/len(x)


def gradients(x, y, w, b):
    total_dw = 0 
    total_db = 0 
    for i in range(len(x)):
        prediction = predict(x[i], w, b)
        error = prediction - y[i]
        total_dw += error * x[i]
        total_db += error
    return total_dw/len(x), total_db/len(x)


def train_model(x , y , w , b , learning_rate , epochs):
    for epoch in range(epochs):
        total_dw , total_db = gradients(x, y, w, b)
        w = w - learning_rate * total_dw
        b = b - learning_rate * total_db
        if epoch % 100 == 0:
            print(bce_loss(x, y , w, b))
    return w , b

if __name__ == "__main__":
    w = .5
    b = -1
    w, b = train_model(x, y, w, b, learning_rate=0.1, epochs=10000)
    print(w,b)

    acc = accuracy(x, y, w, b)
    print("Accuracy:", acc * 100, "%")