import numpy as np

def sgd_for_user_n(X_hat_n, y_hat_n, lambd=0.1, lr=0.01, epochs=1000):
   
    s_n, d = X_hat_n.shape
    w_n = np.zeros(d)
    b_n = 0.0
    e_n = np.ones(s_n)

    for epoch in range(epochs):
        indices = np.random.permutation(s_n)

        for i in indices:
            x_i = X_hat_n[i]
            y_i = y_hat_n[i]

            y_pred_i = np.dot(x_i, w_n) + b_n

            error_i = y_pred_i - y_i

            grad_w = (1 / s_n) * (error_i * x_i + lambd * w_n)
            grad_b = (1 / s_n) * error_i

            w_n -= lr * grad_w
            b_n -= lr * grad_b

    return w_n, b_n

if __name__ == "__main__":
    default_X_hat = np.array([[0.99, 0.02], 
                               [0.01, 0.99]])
    default_y_hat = np.array([1, 4])

    use_default = input("Use default data? (yes/no): ").strip().lower()

    if use_default == "no":
        rows = int(input("Enter the number of items: "))
        columns = int(input("Enter the number of features: "))

        print("Enter the feature matrix row by row:")
        X_hat_list = []
        for i in range(rows):
            row = list(map(float, input(f"Row {i+1}: ").split()))
            X_hat_list.append(row)
        X_hat = np.array(X_hat_list)

        print("Enter the ratings vector:")
        y_hat = np.array(list(map(float, input().split())))
    else:
        X_hat = default_X_hat
        y_hat = default_y_hat

    lambd = float(input("Enter regularization coefficient (default 0.1): ") or 0.1)
    lr = float(input("Enter learning rate (default 0.01): ") or 0.01)
    epochs = int(input("Enter number of epochs (default 1000): ") or 1000)

    w, b = sgd_for_user_n(X_hat, y_hat, lambd=lambd, lr=lr, epochs=epochs)

    # Output results
    print("Results:")
    print(f"Weights (w): {w}")
    print(f"Bias (b): {b:.4f}")