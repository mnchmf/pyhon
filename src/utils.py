import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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


def plot_uc(values):
    i = 0
    iter_gd = []
    for i in range(len(values)):
        iter_gd.append(i)
    plt.plot(iter_gd, values)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.title('Value vs Iteration')
    plt.show()


def plot_feasible_regions_2d(fi_functions, x_positions):
    """
    Plot the feasible regions and the path of positions.

    Arguments:
    - fi_functions: List of functions fi, where each function should return the value,
                    gradient, and Hessian matrix when run with a position as input.
    - x_positions: List of positions x.

    """
    n_dimensions = len(x_positions[0])

    if n_dimensions != 2:
        raise ValueError("Exactly two dimensions are required for plotting in 2D.")

    # 2D Plot
    x = np.linspace(0, 2.1, 200)
    y = np.linspace(0, 2.1, 200)
    X, Y = np.meshgrid(x, y)

    fig, ax = plt.subplots()

    for fi in fi_functions:
        Z = fi([X, Y])[0]
        plt.contour(X, Y, Z, levels=[0], colors='black')

    x_path = [pos[0] for pos in x_positions]
    y_path = [pos[1] for pos in x_positions]
    plt.plot(x_path[1:-1], y_path[1:-1], 'bo-', linewidth=2, markersize=8)
    plt.scatter(x_path[0], y_path[0], c='g', marker='o', s=80, label='Start', linewidths=3)
    plt.scatter(x_path[-1], y_path[-1], c='r', marker='o', s=80, label='End',  linewidths=3)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Feasible Regions and Path')
    plt.legend()

    # Set the axis limits based on the feasible region
    min_x = min(min(x_path), np.min(X))
    max_x = max(max(x_path), np.max(X))
    min_y = min(min(y_path), np.min(Y))
    max_y = max(max(y_path), np.max(Y))
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)

    plt.tight_layout()
    plt.show()


def plot_feasible_regions_3d(path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    path = np.array(path)

    ax.plot_trisurf([1, 0, 0], [0, 1, 0], [0, 0, 1], color='lightgray', alpha=0.5)
    ax.plot(path[:, 0], path[:, 1], path[:, 2], label='Path')
    ax.scatter(path[-1][0], path[-1][1], path[-1][2], s=50, c='gold', marker='o', label='Final candidate')
    ax.set_title("Feasible Regions and Path")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.legend()
    ax.view_init(45, 45)
    plt.show()