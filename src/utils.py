import matplotlib.pyplot as plt
import numpy as np

def contour(f, positions_gd, positions_nm, positions_bfgs, positions_sr1, x_limit, y_limit, title):

    x_p_gd = []*len(positions_gd)
    y_p_gd = []*len(positions_gd)
    for p in positions_gd:
        x_p_gd.append(p[0])
        y_p_gd.append(p[1])

    x_p_nm = [] * len(positions_nm)
    y_p_nm = [] * len(positions_nm)
    for p in positions_nm:
        x_p_nm.append(p[0])
        y_p_nm.append(p[1])

    x_p_bfgs = []*len(positions_bfgs)
    y_p_bfgs = []*len(positions_bfgs)
    for p in positions_bfgs:
        x_p_bfgs.append(p[0])
        y_p_bfgs.append(p[1])

    x_p_sr1 = [] * len(positions_sr1)
    y_p_sr1 = [] * len(positions_sr1)
    for p in positions_sr1:
        x_p_sr1.append(p[0])
        y_p_sr1.append(p[1])


    X = np.linspace(x_limit[0], x_limit[1], 100)
    Y = np.linspace(y_limit[0], y_limit[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros_like(X)
    for i in range(len(X)):
        for j in range(len(Y)):
            x = X[i, j]
            y = Y[i, j]
            position = np.array([x, y])
            Z[i, j], a, b = f(position, False)


    plt.figure()
    plt.contour(X, Y, Z, levels=10, title=title)

    plt.plot(x_p_gd, y_p_gd, "xr-", label="gradient decant")

    plt.plot(x_p_nm, y_p_nm, 'xb-', label="Newton Method")

    plt.plot(x_p_bfgs, y_p_bfgs, 'xy-', label="BFGS")

    plt.plot(x_p_sr1, y_p_sr1, 'xm-', label="SR1")
    plt.legend()
    plt.title(label=title)
    plt.show()


def plot(gd_values, nm_values, bfgs_values, sr1_values):
    i = 0
    iter_gd = []
    for i in range(len(gd_values)):
        iter_gd.append(i)

    iter_nm = []
    for i in range(len(nm_values)):
        iter_nm.append(i)

    iter_bfgs = []
    for i in range(len(bfgs_values)):
        iter_bfgs.append(i)

    iter_sr1 = []
    for i in range(len(sr1_values)):
        iter_sr1.append(i)
    plt.plot(iter_gd, gd_values, label="Gradient Decant")
    plt.plot(iter_nm, nm_values, label="Newton Method")

    plt.plot(iter_bfgs, bfgs_values, label="BFGS")
    plt.plot(iter_sr1, sr1_values, label="SR1")
    plt.legend()
    plt.show()










