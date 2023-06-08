import unittest
from src.unconstrained_min import Solver
from tests.examples import Examples
import numpy as np
from src import utils


class Test(unittest.TestCase):

    def test1(self):
        x0 = np.array([1, 1])
        solver_gd = Solver()
        self.assertEqual(solver_gd.solve_gd(Examples.quad1, x0, 10**-8, 10**-12, 100), (0, True))
        solver_nm = Solver()
        solver_nm.solve_nm(Examples.quad1, x0, 0.001, 0.001, 100)

        solver_bfgs = Solver()
        solver_bfgs.solve_bfgs(Examples.quad1, x0, 10**-8, 10**-12, 100)
        solver_sr1 = Solver()
        solver_bfgs.solve_sr1(Examples.quad1, x0, 10**-8, 10**-12, 100)
        utils.contour(Examples.quad1, solver_gd._positions, solver_nm._positions, solver_bfgs._positions, solver_sr1._positions, [-2, 2], [-2, 2], "x1^2 + x2^2")
        utils.plot(solver_gd._values, solver_nm._values, solver_bfgs._values, solver_sr1._values)

    def test2(self):
        x0 = np.array([1, 1])
        solver_gd = Solver()
        solver_gd.solve_gd(Examples.quad2, x0, 10**-8, 10**-12, 100)
        solver_nm = Solver()
        solver_nm.solve_nm(Examples.quad2, x0, 0.001, 0.001, 100)
        solver_bfgs = Solver()
        solver_bfgs.solve_bfgs(Examples.quad2, x0, 10**-8, 10**-12, 100)
        solver_sr1 = Solver()
        solver_sr1.solve_sr1(Examples.quad2, x0, 10**-8, 10**-12, 100)
        utils.contour(Examples.quad2, solver_gd._positions, solver_nm._positions, solver_bfgs._positions, solver_sr1._positions, [-2, 2], [-2, 2], "x1^2 + 100x2^2")
        utils.plot(solver_gd._values, solver_nm._values, solver_bfgs._values, solver_sr1._values)

    def test3(self):
        x0 = np.array([1, 1])
        solver_gd = Solver()
        solver_gd.solve_gd(Examples.quad3, x0, 10**-8, 10**-12, 100)
        solver_nm = Solver()
        solver_nm.solve_nm(Examples.quad3, x0, 0.001, 0.001, 100)
        solver_bfgs = Solver()
        solver_bfgs.solve_bfgs(Examples.quad3, x0, 10**-8, 10**-12, 100)
        solver_sr1 = Solver()
        solver_sr1.solve_sr1(Examples.quad3, x0, 10**-8, 10**-12, 100)
        utils.contour(Examples.quad3, solver_gd._positions, solver_nm._positions, solver_bfgs._positions, solver_sr1._positions, [-3, 3], [-3, 3], "100x1^2 + x2^2 rotated 30 degrees" )
        utils.plot(solver_gd._values, solver_nm._values, solver_bfgs._values, solver_sr1._values)


    def test4(self):
        x0 = np.array([-1, 2])
        solver_gd = Solver()
        solver_gd.solve_gd(Examples.rosenbork, x0, 10**-8, 10**-12, 1000)
        solver_nm = Solver()
        solver_nm.solve_nm(Examples.rosenbork, x0, 0.001, 0.001, 1000)
        solver_bfgs = Solver()
        solver_bfgs.solve_bfgs(Examples.rosenbork, x0, 10**-8, 10**-12, 1000)
        solver_sr1 = Solver()
        solver_sr1.solve_sr1(Examples.rosenbork, x0, 10**-8, 10**-12, 1000)
        utils.contour(Examples.rosenbork, solver_gd._positions, solver_nm._positions, solver_bfgs._positions, solver_sr1._positions, [-1.5, 1.5], [-.5, 2.1], "Rosenborck 2 degree")
        utils.plot(solver_gd._values, solver_nm._values, solver_bfgs._values, solver_sr1._values)


    def test5(self):
        x0 = np.array([1, 1])
        solver_gd = Solver()
        solver_gd.solve_gd(Examples.liner, x0, 10**-8, 10**-12, 100)
        utils.contour(Examples.liner, solver_gd._positions, [], [], [], [-200, 2], [-200, 2], "x + y")
        utils.plot(solver_gd._values, [], [], [])

    def test6(self):
        x0 = np.array([1, 1])
        solver_gd = Solver()
        solver_gd.solve_gd(Examples.exponent, x0, 10**-8, 10**-12, 100)
        solver_nm = Solver()
        solver_nm.solve_nm(Examples.exponent, x0, 0.001, 0.001, 100)
        solver_bfgs = Solver()
        solver_bfgs.solve_bfgs(Examples.exponent, x0, 10**-8, 10**-12, 100)
        solver_sr1 = Solver()
        solver_sr1.solve_sr1(Examples.exponent, x0, 10**-8, 10**-12, 100)
        utils.contour(Examples.exponent, solver_gd._positions, solver_nm._positions, solver_bfgs._positions, solver_sr1._positions, [-3, 3], [-3, 3], "exp(x1 -3x2 - 0.1) + exp(x1 - 3x2 - 0.1) + exp(-x1 - 0.1)")
        utils.plot(solver_gd._values, solver_nm._values, solver_bfgs._values, solver_sr1._values)



