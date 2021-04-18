import math
import numpy as np
import matplotlib.pylab as plt


def func(x):
    """
    Исследуемая функция.
    :param x:
    :return:
    """
    # y = (x - 1) ** 4 + (x - 1) ** 3 + (x - 1) ** 2
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


if __name__ == '__main__':
    show_func()
