import math
from logistic_regression import sigmoid
from _linear_regression import dot_product



X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_and = [0, 0, 0, 1]
y_or  = [0, 1, 1, 1]
y_xor = [0, 1, 1, 0]

train_data = {
    "y_and": y_and,
    "y_or": y_or,
    "y_xor": y_xor
}


def predict(x, w, b):
    z = dot_product(x,w)+b
    return sigmoid(z)



def bce_loss(x , y , w , b ):
    total_Loss = 0
    for i in range(len(x)):
        prediction = predict(x[i], w , b)
        prediction = max(1e-15, min(1 - 1e-15, prediction))
        if y[i] == 1:
            total_Loss += -math.log(prediction)
        else:
            total_Loss += -math.log(1 - prediction)
    return total_Loss/len(x)

def gradients(x , y , w , b ):
    total_dw = [0] * len(w)
    total_db = 0
    for i in range(len(x)):
        prediction = predict(x[i], w , b)
        error = prediction -y[i]
        for j in range(len(w)):
            total_dw[j] += error * x[i][j]

        total_db += error
    total_dw = [total_dw[j]/len(x) for j in range(len(w))]
    total_db = total_db / len(x)
    return total_dw, total_db

def train_model(x , y , w , b , learning_rate , epochs):
    for epoch in range(epochs):
        dw , db = gradients(x, y, w, b)
        for j in range(len(w)):
            w[j] = w[j] - learning_rate * dw[j]
        b = b - learning_rate * db
        if epoch % 100 == 0:
            print(bce_loss(x , y , w , b))
    return w , b


def accuracy(x , y , w , b):
    correct_predictions = 0
    for i in range(len(x)):
        prediction = predict(x[i], w , b)
        predicted_label = 1 if prediction >= 0.5 else 0
        if predicted_label == y[i]:
            correct_predictions += 1
    return correct_predictions / len(x)

if __name__ == "__main__":

    w = [0, 0]
    b = 0

    for name,data in train_data.items():
        w, b = train_model(
            X,
            data,
            [0,0],
            0,
            learning_rate=0.1,
            epochs=3000
        )
        print(f"--------------{name}--------------")
        print(f"Final weights: {w}, Final bias: {b}")
        print(f"Accuracy: {accuracy(X, data, w, b)}")



