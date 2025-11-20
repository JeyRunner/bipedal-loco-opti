from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
from rich import print
from tests.test_util import *

from centroidal_walk.collocation.spline.spline_trajectory import *

class TestSplineTrajectory(TestCase):

	num_polynomials_for_com_trajectory = 5

	def test_opti_single_spline3_trajectory_deltaT_flexible(self):
		opti = casadi.Opti()

		start_x = np.ones(1)
		start_dx = np.zeros(1)
		end_x = np.ones(1)*4
		end_dx = np.ones(1)*10

		x_trajectory = SplineTrajectory(
			2,
			start_x,
			start_dx,
			opti,
			param_opti_end_x=end_x,
			param_opti_end_dx=end_dx,
			spline_polynomial_degree=3,
			dimensions=1,
			#given_total_duration=6
			_debug__casadi_ifelse_nested=True
		)
		x_trajectory.set_initial_opti_values__zero(init_total_duration=6)

		# demo constraints
		opti.subject_to(x_trajectory.evaluate_x(1) == 2)
		opti.subject_to(x_trajectory.get_sum_detaT() == 6)

		opti.solver('ipopt')
		opti.solve()



		# plotting
		t_vals = np.arange(-1, 7, step=0.01)

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_x(t_vals)), label="x")
		#save_fig(self)

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_dx(t_vals)), label="dx", linestyle='dashed')
		#save_fig(self, 'dx')

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_ddx(t_vals)).T, label="ddx", linestyle='dashed')
		#save_fig(self, 'ddx')
		plt.legend()
		plt.show()



	def test_opti_single_spline3_trajectory_deltaT_flexible__rolling_shift_t(self):
		opti = casadi.Opti()

		start_x = np.ones(1)
		start_dx = np.zeros(1)
		end_x = np.ones(1)*4
		end_dx = np.ones(1)*10

		x_trajectory = SplineTrajectory(
			2,
			start_x,
			start_dx,
			opti,
			param_opti_end_x=end_x,
			param_opti_end_dx=end_dx,
			spline_polynomial_degree=3,
			dimensions=1,
			#given_total_duration=6
			_debug__casadi_ifelse_nested=True
		)
		x_trajectory.set_initial_opti_values__zero(init_total_duration=6)

		# demo constraints
		opti.subject_to(x_trajectory.evaluate_x(1) == 2)
		opti.subject_to(x_trajectory.get_sum_detaT() == 6)

		opti.solver('ipopt')
		opti.solve()



		# plotting
		t_vals = np.arange(-1, 7, step=0.01)

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_x(t_vals)), label="x")
		#save_fig(self)

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_dx(t_vals)), label="dx", linestyle='dashed')
		#save_fig(self, 'dx')

		#plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_ddx(t_vals)).T, label="ddx", linestyle='dashed')
		#save_fig(self, 'ddx')
		plt.legend()
		plt.show()


