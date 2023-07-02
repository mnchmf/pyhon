import math

import numpy as np


def get_log(ineq_constraints, x0):
    n = np.shape(x0)[0]
    log_f = 0
    log_g = np.zeros(n)
    log_h = np.zeros((n, n))
    for c in ineq_constraints:
        f, g, h = c(x0)
        log_f += math.log(-f)
        log_g += (-1.0 / f) * g
        l = g / f
        helper = np.tile(
            l.reshape(l.shape[0], -1), (1, l.shape[0])
        ) * np.tile(l.reshape(l.shape[0], -1).T, (l.shape[0], 1))
        log_h += (h * f - helper) / f ** 2

    return -log_f, log_g, -log_h


def get_derivatives(f, x0, ineq_constraints, t):
    f_prev, g_prev, h_prev = f(x0)
    f_log, g_log, h_log = get_log(ineq_constraints, x0)
    x = t*f_prev + f_log
    # print (g_log, np.shape(g_log))
    # print(g_prev, np.shape(g_prev))
    # print(t, np.shape(t))
    y = g_log + t*g_prev
    z = t*h_prev + h_log
    return t*f_prev + f_log, g_log + t*g_prev, t*h_prev + h_log


def get_direction(h, A, g):
    if A is not None:
        m = np.shape(A)[0]
        upper = np.concatenate((h, A.T), axis=1)
        lower = np.concatenate((A, np.zeros((m, m))), axis=1)
        M = np.concatenate((upper, lower), axis=0)
        v = np.concatenate((-g, np.zeros(m)), axis=0)
        x = np.linalg.solve(M, v)
        return x[0:-m]
    else:
        v = -g
        M = h
        x = np.linalg.solve(M, v)
        return x


def get_step_len(x_prev, func, direction, f_prev, t, gradiant, iec):
    a = 1
    i = 0
    while i < 5:
        # f, g, h = get_derivatives(func, x_prev + a * direction, iec, t)
        f, g, h = func(x_prev + a * direction)
        i += 1
        if f <= f_prev + 0.001 * a * gradiant.dot(direction):
            return a
        else:
            a = a * 0.5
    return a


class ConstrainedSolver:
    _positions = []
    _values = []
    _constraints = []

    def interior_pt(self, f, ineq_constraints, eq_constraints_mat, eq_constraints_rhs, x0):
        print("start")
        t = 1
        m = len(ineq_constraints)
        f_prev, g_prev, h_prev = get_derivatives(f, x0, ineq_constraints, t)
        x_prev = x0

        while (m / t) > 10 ** -12:
            for i in range(10):
                p = get_direction(h_prev, eq_constraints_mat, g_prev)
                alpha = get_step_len(x_prev, f, p, f_prev, t, g_prev, ineq_constraints)

                x_next = x_prev + alpha * p

                f_next, g_next, h_next = get_derivatives(f, x_next, ineq_constraints, t)
                l = np.sqrt(np.dot(p, np.dot(h_next, p.T)))
                x_prev = x_next
                f_prev = f_next
                g_prev = g_next
                h_prev = h_next
                if 0.5 * (l ** 2) < 10**-8:
                    break

            print(x_prev)
            print(f(x_prev)[0])
            self._positions.append(x_prev)
            self._values.append(f(x_prev)[0])
            t *= 10
        for c in ineq_constraints:
            self._constraints.append(c(x_prev)[0])

        return x_prev, f_prev




