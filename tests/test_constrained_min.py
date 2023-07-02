import unittest
from src.unconstrained_min import Solver
from tests.examples import Examples
import numpy as np
from src import utils
from src.constrained_min import ConstrainedSolver

class Test(unittest.TestCase):


    def test_qp(self):
        x0 = np.array([0.1, 0.2, 0.7])
        solver = ConstrainedSolver()
        solver.interior_pt(Examples.ex1, [Examples.ex1h1, Examples.ex1h2, Examples.ex1h3], np.array([[1, 1, 1]]), np.array([1]), x0)
        utils.plot_uc(solver._values)
        utils.plot_feasible_regions_3d(solver._positions)
        print("the final x_value: ", solver._positions[-1])
        print("the final y_value: ", solver._values[-1])
        print("the final Equality Constraint values  : ", solver._constraints)

    def test_lp(self):
        x0 = np.array([0.5, 0.75])
        solver = ConstrainedSolver()
        solver.interior_pt(Examples.ex2, [Examples.ex2h1, Examples.ex2h2, Examples.ex2h3, Examples.ex2h4], None, None, x0)
        utils.plot_uc(solver._values)
        utils.plot_feasible_regions_2d([Examples.ex2h1, Examples.ex2h2, Examples.ex2h3, Examples.ex2h4], solver._positions)
        print("the final x_value: ", solver._positions[-1])
        print("the final y_value: ", solver._values[-1])
        print("the final Equality Constraint values  : ", solver._constraints)

