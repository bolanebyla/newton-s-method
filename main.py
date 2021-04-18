import math
import numpy as np
import matplotlib.pylab as plt


def func(x):
    """
    Исследуемая функция.
    :param x:
    :return:
    """
    y = math.asin((x - 1) ** 2)
    return y


def show_func():
    """
    Построение графика функции.
    :return:
    """
    x = np.linspace(0, 2, 4001)
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = func(x[i])
    plt.plot(x, y)
    plt.grid()
    plt.show()


def dif1(x):
    """
    Первая производная (разностная).
    :param x:
    :return:
    """
    h = 1e-7
    y = (func(x + h) - func(x - h)) / (2 * h)
    return y


def dif2(x):
    """
    Вторая производная (разностная).
    :param x:
    :return:
    """
    h = 1e-7
    y = (func(x + h) - 2 * func(x) + func(x - h)) / (h ** 2)
    return y


def Newton(x):
    """
    Метод Ньютона (с разностными производными).
    :param x:
    :return:
    """
    eps = 1e-14
    x_error = []
    x2 = x
    for i in range(100):
        x1 = x2
        x2 = x1 - (dif1(x1) / dif2(x1))
        x_error.append([x2, x1, abs(x2 - x1)])
        if abs(x2 - x1) < eps:
            break
    if dif2(x2) > 0:
        print('Точка минимума: x = ', end='')
    elif dif2(x2) < 0:
        print('Точка максимума: x = ', end='')
    print(x2)
    R_theor = x_error[-2][2]
    print('Теоретическая ошибка: R_theor =', R_theor)
    return x2, x_error


def find_min():
    """
    Поиск точки минимума.
    :return:
    """
    x, x_error = Newton(0.5)
    x_0 = 1
    R_real = abs(x_0 - x)
    print('Реальная ошибка: R_real =', R_real)
    print('\nСписок ошибок:')
    for i in x_error:
        print(f'|{i[0]} - {i[1]}| = {i[2]}')


def dif1_a(x):
    """
    Первая производная (аналитическая).
    :param x:
    :return:
    """
    y = (2 * x - 2) / math.sqrt(1 - (x - 1) ** 4)
    return y


def dif2_a(x):
    """
    Вторая производная (аналитическая).
    :param x:
    :return:
    """
    y = (2 * (1 + (2 * (x - 1) ** 4) / (1 - (x - 1) ** 4))) / math.sqrt(1 - (x - 1) * 4)
    return y


def Newton_a(x):
    """
    Метод Ньютона (с аналитическими производными).
    :param x:
    :return:
    """
    eps = 1e-14
    x_error = []
    x2 = x
    for i in range(100):
        x1 = x2
        x2 = x1 - (dif1_a(x1) / dif2_a(x1))
        x_error.append([x2, x1, abs(x2 - x1)])
        if abs(x2 - x1) < eps:
            break
    if dif2_a(x2) > 0:
        print('Точка минимума: x = ', end='')
    elif dif2_a(x2) < 0:
        print('Точка максимума: x = ', end='')
    print(x2)
    R_theor = x_error[-2][2]
    print('Теоретическая ошибка: R_theor =', R_theor)
    return x2, x_error


def find_min_a():
    """
    Поиск точки минимума.
    :return:
    """
    x, x_error = Newton(0.5)
    x_0 = 1
    R_real = abs(x_0 - x)
    print('Реальная ошибка: R_real =', R_real)

    print('\nСписок ошибок:')
    for i in x_error: print(f'|{i[0]} - {i[1]}| = {i[2]}')


if __name__ == '__main__':
    find_min_a()
