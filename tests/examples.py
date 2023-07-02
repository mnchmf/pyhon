import numpy as np
import math


class Examples:

    @staticmethod
    def _get_function_values(x, q, hessian):
        f = np.dot(x, q.dot(x))
        g = 2 * q.dot(x)
        if hessian:
            h = 2 * q
            return f, g, h

        return f, g, 0

    @staticmethod
    def quad1(x, hessian):
        q = np.array([[1, 0], [0, 1]])
        return Examples._get_function_values(x, q, hessian)

    @staticmethod
    def quad2(x, hessian):
        q = np.array([[1, 0], [0, 100]])
        return Examples._get_function_values(x, q, hessian)

    @staticmethod
    def quad3(x, hessian):
        W = np.array([[(3**0.5)/2, -0.5],  [0.5, (3**0.5)/2]])
        q = W.transpose() @ np.array([[100, 0], [0, 1]]) @ W
        return Examples._get_function_values(x, q, hessian)

    @staticmethod
    def rosenbork(X, hessian):
        x = X[0]
        y = X[1]
        f = 100*(y - x**2)**2 + (1 - x)**2
        g = np.array([400*x**3 - 400*x*y + 2*x - 2, 200*(y - x**2)])
        h = None
        if hessian:
            h = np.array(([[1200*x**2 - 400*y + 2, -400*x], [-400*x, 200]]))

        return f, g, h


    @staticmethod
    def liner(X, hessian):
        x = X[0]
        y = X[1]
        f = x + y
        g = np.array([1, 1])
        h = None
        if hessian:
            h = np.zeros_like((2, 2))

        return f, g, h

    @staticmethod
    def exponent(X, hessian):
        x = X[0]
        y = X[1]
        f = math.exp(x + 3*y - 0.1) + math.exp(x - 3*y - 0.1) + math.exp(-x - 0.1)
        g = np.array([math.exp(x + 3*y - 0.1) + math.exp(x - 3*y - 0.1) - math.exp(-x - 0.1),
                      3*math.exp(x + 3 * y - 0.1) - 3*math.exp(x - 3 * y - 0.1)])
        h = None
        if hessian:
            h = np.array([[math.exp(x + 3*y - 0.1) + math.exp(x - 3*y - 0.1) + math.exp(-x - 0.1), 3*math.exp(x + 3 * y - 0.1) - 3*math.exp(x - 3 * y - 0.1)],
                         [3*math.exp(x + 3 * y - 0.1) - 3*math.exp(x - 3 * y - 0.1), 9*math.exp(x + 3 * y - 0.1) + 9*math.exp(x - 3 * y - 0.1)]])

        return f, g, h

    @staticmethod
    def ex1(X):
        x = X[0]
        y = X[1]
        z = X[2]
        f = x**2 + y**2 + (z + 1)**2
        g = np.array([2*x, 2*y, 2*z + 2])
        h = np.array([[2, 0, 0], [0, 2, 0], [0, 0, 2]])

        return f, g, h

    def ex1h1(X):
        x = X[0]
        return -x, np.array([-1, 0, 0]), np.zeros((3, 3))

    def ex1h2(X):
        y = X[1]
        return -y, np.array([0, -1, 0]), np.zeros((3, 3))

    def ex1h3(X):
        z = X[2]
        return -z, np.array([0, 0, -1]), np.zeros((3, 3))


    @staticmethod
    def ex2(X):
        x = X[0]
        y = X[1]
        f = -x + -y
        g = np.array([-1, -1])
        h = np.array([[0, 0], [0, 0]])

        return f, g, h

    # x <= 2
    def ex2h1(X):
        x = X[0]
        return x - 2, np.array([1, 0]), np.zeros((2, 2))

    # y < = 1
    def ex2h2(X):
        y = X[1]

        return y - 1, np.array([0, 1]), np.zeros((2, 2))

    # y > = 0
    def ex2h3(X):
        y = X[1]
        return -y, np.array([0, -1]), np.zeros((2, 2))

   #𝑦 ≥ −𝑥 + 1
    def ex2h4(X):
        y = X[1]
        x = X[0]
        return -y - x + 1, np.array([-1, -1]), np.zeros((2, 2))



