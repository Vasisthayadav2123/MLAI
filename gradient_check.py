import _linear_regression as lr


def loss(x, y, w, b):
    return lr.mse(y, [lr.predict(row, w, b) for row in x])


def relative_error(a, n):
    return abs(a - n) / max(abs(a), abs(n), 1e-12)


def numerical_gradients(x, y, w, b, h):
    num_dw = [0] * len(w)
    for j in range(len(w)):
        w_plus = w.copy()
        w_minus = w.copy()
        w_plus[j] += h
        w_minus[j] -= h
        num_dw[j] = (loss(x, y, w_plus, b) - loss(x, y, w_minus, b)) / (2 * h)

    num_db = (loss(x, y, w, b + h) - loss(x, y, w, b - h)) / (2 * h)
    return num_dw, num_db


def check_gradients(x, y, w, b, h=1e-5):
    dw, db = lr.gradients(x, y, w, b)
    num_dw, num_db = numerical_gradients(x, y, w, b, h)

    for j in range(len(w)):
        print(f"w[{j}]  analytic: {dw[j]:.6f}  numerical: {num_dw[j]:.6f}  rel_err: {relative_error(dw[j], num_dw[j]):.2e}")
    print(f"b     analytic: {db:.6f}  numerical: {num_db:.6f}  rel_err: {relative_error(db, num_db):.2e}")


if __name__ == "__main__":
    w = [0.1, 5, -3]
    b = 2
    check_gradients(lr.x, lr.y, w, b)
