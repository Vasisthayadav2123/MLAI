

x = [1, 2, 3, 4, 5]
y = [30, 40, 50, 60, 70]



w = 0
b = 0
learning_rate = 0.01


def predict(x , w , b):
    return(w * x + b)

def mse(y_true, y_pred):
    total = 0
    for i in range(len(y_true)):
        error = y_true[i] - y_pred[i]
        total += error**2
    return (total / len(y_true))



def gradients(x , y_true , w , b):
    total_dw , total_db = 0 , 0
    for i in range(len(x)):
        prediction = predict(x[i], w, b)
        error = prediction - y_true[i]
        total_dw += 2*error*x[i]
        total_db += 2*error 

    dw = total_dw/len(x)
    db = total_db/len(x)
    
    return dw , db

##for epoch in range(1000):
##    dw, db = gradients(x, y, w, b)
##
##    if epoch % 100 == 0:
##        ##print(mse(y , [predict(i, w, b) for i in x]))
##
##    w = w-learning_rate * dw
##    b = b-learning_rate * db


def train_model(x , y , w , b , learning_rate , epochs):
    for epoch in range(epochs):
        dw, db = gradients(x, y, w, b)

        w = w-learning_rate * dw
        b = b-learning_rate * db
        if epoch % 100 == 0:
            print(mse(y , [predict(i, w, b) for i in x]))
    return w, b

if __name__ == "__main__":
    w, b = train_model(x, y, w, b, learning_rate, 1000)
    print(predict(6, w, b))


