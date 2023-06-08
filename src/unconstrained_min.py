import matplotlib.pyplot as plt
import numpy as np
from src import utils


class Solver:

    _positions = []
    _values = []

    @staticmethod
    def _check_convergence(x_prev, x_next, f_prev, f_next, obj_tol, param_tol):
        if abs(f_next - f_prev) < obj_tol or np.linalg.norm(x_next - x_prev) < param_tol:
            return True
        return False

    def solve_gd(self, func, x0, obj_tol, param_tol, max_iter):
        self._positions = []
        self._values = []
        x_prev = x0
        success = False
        i = 0
        f_prev, g, h = func(x_prev, False)

        self._positions.append(x_prev)
        self._values.append(f_prev)
        while not success and i <= max_iter:
            f_prev, g, h = func(x_prev, False)
            direction = -g
            step_len = Solver.get_step_len_gd(x_prev, func, direction, f_prev)
            x_next = x_prev + direction * step_len
            f_next, g, h = func(x_next, False)

            i += 1
            success = Solver._check_convergence(x_prev, x_next, f_prev, f_next, obj_tol, param_tol)
            x_prev = x_next
            f_prev = f_next
            print('iter', i)
            print('x', x_prev)
            print('f', f_prev)
            self._positions.append(x_prev)
            self._values.append(f_prev)

        return f_prev, success

    def solve_nm(self, func, x0, obj_tol, param_tol, max_iter):
        self._positions = []
        self._values = []
        x_prev = x0
        success = False
        i = 0
        f_prev, g, h = func(x_prev, False)

        self._positions.append(x_prev)
        self._values.append(f_prev)
        while not success and i <= max_iter:
            f_prev, g, h = func(x_prev, True)
            direction = -np.linalg.solve(h, g)
            step_len = Solver.get_step_len_gd(x_prev, func, direction, f_prev)
            x_next = x_prev + direction * step_len
            f_next, g, h = func(x_next, False)

            i += 1
            success = Solver._check_convergence(x_prev, x_next, f_prev, f_next, obj_tol, param_tol)
            x_prev = x_next
            f_prev = f_next
            print('iter', i)
            print('x', x_prev)
            print('f', f_prev)
            self._positions.append(x_prev)
            self._values.append(f_prev)

    def solve_bfgs(self, func, x0, obj_tol, param_tol, max_iter):
        self._positions = []
        self._values = []
        x_prev = x0
        success = False
        i = 0
        f_prev, g, h = func(x_prev, False)
        B = np.identity(len(x0))
        self._positions.append(x_prev)
        self._values.append(f_prev)
        while not success and i <= max_iter:
            f_prev, g, h = func(x_prev, False)

            direction = -np.linalg.solve(B, g)
            step_len = Solver.get_step_len_gd(x_prev, func, direction, f_prev)
            x_next = x_prev + direction * step_len
            f_next, g_next, h = func(x_next, False)

            s = x_next - x_prev
            y = g_next - g
            Bs = B @ s
            sBs = np.dot(s, Bs)
            yTs = np.dot(y, s)
            B = B - np.divide(np.outer(Bs, Bs), sBs) + np.divide(np.outer(y, y), yTs)


            i += 1
            success = Solver._check_convergence(x_prev, x_next, f_prev, f_next, obj_tol, param_tol)
            x_prev = x_next
            f_prev = f_next
            print('iter', i)
            print('x', x_prev)
            print('f', f_prev)
            self._positions.append(x_prev)
            self._values.append(f_prev)

        return f_prev, success


    def solve_sr1(self, func, x0, obj_tol, param_tol, max_iter):
        self._positions = []
        self._values = []
        x_prev = x0
        success = False
        i = 0
        f_prev, g, h = func(x_prev, False)
        B = np.identity(len(x0))
        self._positions.append(x_prev)
        self._values.append(f_prev)
        while not success and i <= max_iter:
            f_prev, g, h = func(x_prev, False)

            direction = -np.linalg.solve(B, g)
            step_len = Solver.get_step_len_gd(x_prev, func, direction, f_prev)
            x_next = x_prev + direction * step_len
            f_next, g_next, h = func(x_next, False)

            s = x_next - x_prev
            y = g_next - g
            Bs = B @ s
            B = B + np.divide(np.outer(y - Bs, y - Bs), np.dot(y - Bs, s))


            i += 1
            success = Solver._check_convergence(x_prev, x_next, f_prev, f_next, obj_tol, param_tol)
            x_prev = x_next
            f_prev = f_next
            print('iter', i)
            print('x', x_prev)
            print('f', f_prev)
            self._positions.append(x_prev)
            self._values.append(f_prev)

        return f_prev, success


    @staticmethod
    def get_step_len_gd(x_prev, func, direction, f_prev):
        a = 1
        while True:
            f, g, h = func(x_prev + a*direction, False)
            if f <= f_prev + 0.01*a*np.dot(-direction, direction):
                return a
            else:
                a = a*0.5


