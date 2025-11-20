from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from rich import print

from centroidal_walk.collocation.spline.polynomial4 import Polynomial4
from tests.test_util import *

from centroidal_walk.collocation.spline.spline_trajectory import *

class TestSplineTrajectory(TestCase):

	num_polynomials_for_com_trajectory = 5

	def test_polynomial4(self):
		x0 = np.ones(2)*5
		dx0 = np.ones(2)*3
		ddx0 = np.ones(2)*100
		x1 = np.ones(2)*-3#(-1)
		dx1 = np.ones(2)*-1
		deltaT = 2

		poly = Polynomial4(
			x0=x0,
			dx0=dx0,
			ddx0=ddx0,
			x1=x1,
			dx1=dx1,
			deltaT=deltaT,
			reshape=True
		)

		t_vals = np.arange(-0.1, deltaT+0.1, step=0.01)
		x_vals = np.array(poly.evaluate(t_vals))
		dx_vals = np.array(poly.evaluate_dx(t_vals))
		ddx_vals = np.array(poly.evaluate_ddx(t_vals))
		fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
		ax1.plot(t_vals, x_vals[0, :], label='x')
		ax1.scatter(np.array([0, deltaT]), np.array([x0[0], x1[0]]), label='given x0, x1')
		ax1.set_xlim(0, 3)
		#ax1.set_ylim(0, 4)
		ax1.legend()
		ax2.plot(t_vals, dx_vals[0, :], label='dx')
		ax2.scatter(np.array([0, deltaT]), np.array([dx0[0], dx1[0]]), label='given dx0, dx1')
		ax2.legend()
		ax3.plot(t_vals, ddx_vals[0, :], label='ddx')
		ax3.scatter(np.array([0]), np.array([ddx0[0]]), label='given ddx0')
		ax3.legend()
		#plt.xlim(0, 3)
		#plt.ylim(1, 4)

		save_fig(self)

		self.assertTrue(np.all(x0 == poly.evaluate(np.array([0]))))
		self.assertTrue(np.all(x1 == poly.evaluate(np.array([deltaT]))))
		self.assertTrue(np.all(dx0 == poly.evaluate_dx(np.array([0]))))
		self.assertTrue(np.all(dx1 == poly.evaluate_dx(np.array([deltaT]))))
		self.assertTrue(np.all(ddx0 == poly.evaluate_ddx(np.array([0]))))

		poly.evaluate(np.ones(4))
		#plt.show()
