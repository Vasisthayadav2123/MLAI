from linear_regression_rawdogg import mse

x = [
    [1200, 3, 2],
    [1500, 4, 3],
    [1000, 2, 1],
    [1800, 4, 3]
]

y = [200, 250, 170, 300]
w = [0, 0, 0]

b= 0

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

        total_db += error
        
    dw = [total_dw[j]/len(x) for j in range(len(w))]
    db = total_db/len(x)
    return dw , db

def standardize_data(values):
    mean = sum(values) / len(values)
    variance = (sum((x - mean) ** 2 for x in values) / len(values))
    std = variance ** 0.5
    standardized = [(x - mean) / std for x in values]
    return standardized


def train_model(x , y , w , b , learning_rate , epochs):
    for epoch in range(epochs):
        dw , db = gradients(x, y, w, b)
        for j in range(len(w)):
            w[j] = w[j] - learning_rate * dw[j]
        b = b - learning_rate * db
        if epoch % 100 == 0:
            print(mse(y , [predict(i, w, b) for i in x]))
    return w , b

if __name__ == "__main__":


    w = [0, 0, 0]
    b = 0

    w, b = train_model(
        x,
        y,
        w,
        b,
        learning_rate=0.0000001,
        epochs=2000
    )

    print("w:", w)
    print("b:", b)
            