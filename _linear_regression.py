from linear_regression_rawdogg import mse

x = [
    [1200, 3, 2],
    [1500, 4, 3],
    [1000, 2, 1],
    [1800, 4, 3]
]

y = [200, 250, 170, 300]
w = [0.1, 10, 5]

b= 20

def dot_product(x, y):
    return sum(x[i] * y[i] for i in range(len(x)))


def predict(x, w , b):
    prediction = dot_product(x, w) + b
    return prediction



def gradients(x , y , w , b):
    total_dw = [0] * len(w)
    total_db = 0
    for i in range(len(x)):
        prediction = predict(x[i] , w , b)
        error = prediction - y[i]
        for j in range(len(w)):
            total_dw[j] += 2 * error * x[i][j]

        total_db += 2*error

    dw = [total_dw[j]/len(x) for j in range(len(w))]
    db = total_db/len(x)
    return dw , db

dw, db = gradients(x, y, w, b)

def train_model(x , y , w , b , learning_rate , epochs):
    for epoch in range(epochs):
        dw , db = gradients(x, y, w, b)
        for j in range(len(w)):
            w[j] = w[j] - learning_rate * dw[j]
        b = b - learning_rate * db
        if epoch % 100 == 0:
            print(mse(y , [predict(i, w, b) for i in x]))
    return w , b
        