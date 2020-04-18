import numpy as np


fit_funcs = {
    'poly_1': lambda x, x1, x0: x1 * x + x0,
    'poly_2': lambda x, x2, x1, x0: x2 * x**2 + x1 * x + x0,
    'poly_3': lambda x, x3, x2, x1, x0: x3 * x**3 + x2 * x**2 + x1 * x + x0,
    'poly_4': lambda x, x4, x3, x2, x1, x0: x4 * x**4 + x3 * x**3 + x2 * x**2 +
    x1 * x + x0,
    'poly_5': lambda x, x5, x4, x3, x2, x1, x0: x5 * x**5 + x4 * x**4 +
    x3 * x**3 + x2 * x**2 + x1 * x + x0,
    'poly_6': lambda x, x6, x5, x4, x3, x2, x1, x0: x6 * x**6 + x5 * x**5 + x4
    * x**4 + x3 * x**3 + x2 * x**2 + x1 * x + x0,
    'exp': lambda x, a, b, c: a * np.exp(b * x) + c,
    'dummy_1': lambda x, x3, x2, x1, x0: x3 * np.sin(x2 * x) + x1 * x + x0
}
