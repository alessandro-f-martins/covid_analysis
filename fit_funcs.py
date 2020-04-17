import numpy as np


def exp_func(x, a, b, c):
    return a * np.exp(b * x) + c


def lin_func(x, x1, x0):
    return x1 * x + x0


def quad_func(x, x2, x1, x0):
    return x2 * x**2 + x1 * x + x0


def third_func(x, x3, x2, x1, x0):
    return x3 * x**3 + x2 * x**2 + x1 * x + x0


def fourth_func(x, x4, x3, x2, x1, x0):
    return x4 * x**4 + x3 * x**3 + x2 * x**2 + x1 * x + x0


def sixth_func(x, x6, x5, x4, x3, x2, x1, x0):
    return x6 * x**6 + x5 * x**5 + x4 * x**4 + x3 * x**3 \
        + x2 * x**2 + x1 * x + x0


def dummy_func1(x, x3, x2, x1, x0):
    return x3 * np.sin(x2 * x) + x1 * x + x0
