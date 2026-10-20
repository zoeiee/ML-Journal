import numpy as np
import matplotlib.pyplot as plt

x_train = np.array([1, 2, 3, 4])
y_train = np.array([35,45,55,65])

w = 5
b = 20
alpha = 0.01

def compute_model_output(x,w,b):
    m = x.shape[0]
    f_wb = np.zeros(m)

    for i in range(m):
        f_wb[i] = w*x[i] + b
    return f_wb

y_hat = compute_model_output(x_train, w, b)
print(y_hat)

def compute_prediction_error(y, y_hat):
    m = y.shape[0]
    error = np.zeros(m)

    for i in range(m):
        error[i] = y_hat[i] - y[i]
    return error

individual_error = compute_prediction_error(y_train, y_hat)
print(individual_error)

# J_wb == cost function is the same thing
def J_wb(w, b, x, y):
    m = x.shape[0]
    cost_sum = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]
        cost_sum += error ** 2

    total_cost = cost_sum / (2 * m)
    return total_cost

def compute_gradient(w, b, x, y):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb = w * x[i] + b
        error = f_wb - y[i]

        dj_dw += error * x[i]
        dj_db += error

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_dw, dj_db

gradient = compute_gradient(w,b,x_train,y_train)
print(gradient)

def compute_cost(w, b, x, y):
    m = x.shape[0]
    cost_sum = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = (f_wb - y[i]) ** 2
        cost_sum += cost
        
    total_cost = cost_sum / (2 * m)
    return total_cost

def gradient_descent(x,y,w,b, alpha, n):
    cost_history = []

    for i in range(n):

        dj_dw, dj_db = compute_gradient(w,b,x,y)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = compute_cost(w,b,x,y)
        cost_history.append(cost)
    return w, b, cost_history

final_gradient = compute_gradient(w,b,x_train,y_train)
print(final_gradient)