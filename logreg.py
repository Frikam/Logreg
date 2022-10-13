'''
Реализовать на C++ метод машинного обучения — логистическую регрессию.
Реализовать три функции fit (обучение), prediction (предсказание), accuracy .
Программа начинает свою работу с функции fit, которая на вход получает двухмерный массив признаков X (размера MxN),
одномерный массив целевой переменной Y (классы 0 или 1).
После чего функция fit должна построить логистическую регрессию
на вход которой должны прийти данные из функции prediction — матрица признаков для предсказания (размера ZxN).
На выход функции prediction должен выйти массив предсказаний. Кроме этого, необходимо реализовать функцию accuracy,
которая будет оценивать качество классификации при помощи метрики accuracy.
Интерфейс программы — два файла train.txt (файл с массивом признаков и классов для обучения),
test.txt (файл с массивом признаков и классов для проверки результата обучения), вывод в консоль метрики accuracy.
'''

import numpy as np


def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s


def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = -(np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))) / m

    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    cost = np.squeeze(np.array(cost))
    grads = {"dw": dw,
             "db": db}

    return grads, cost


def fit(X, Y):

    w = np.zeros((X.shape[0], 1))
    b = float(0)

    costs = []

    num_iterations = 100
    learning_rate = 0.009,

    for i in range(num_iterations):

        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b -= learning_rate * db

    params = {"w": w,
              "b": b}

    return params


def predict(w, b, X):
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        if A[0, i] > 0.5:
            Y_pred[0, i] = 1
        else:
            Y_pred[0, i] = 0

    return Y_pred


def accuracy(Y_test, Y_pred):
    true = 0

    for i in range(len(Y_pred)):
        if Y_test[0, i] == Y_pred[0, i]:
            true += 1

    return true / len(Y_pred)
