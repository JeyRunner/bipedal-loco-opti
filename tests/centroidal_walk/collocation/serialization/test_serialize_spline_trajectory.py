from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import rich
from centroidal_walk.collocation.serialization.OptiDecorators import *
from centroidal_walk.collocation.spline.spline_trajectory import *
from centroidal_walk.collocation.opti.casadi_util.OptiLoadable import *


class TestSerializeSplineTrajectory(TestCase):

	def test_serialize_SplineTrajectory(self):
		opti = Opti()

		spline_parameters = dict(
			num_polynomials_for_trajectory=30,
			param_opti_start_x=np.ones(3),
			param_opti_start_dx=np.ones(3)*0.5,
			dimensions=3,
			#given_total_duration=3
		)

		trajectory = SplineTrajectory(**spline_parameters, opti=opti)
		trajectory.set_initial_opti_values__zero(init_total_duration=6)

		opti.subject_to(trajectory.evaluate_x(0.1) == 2)
		# opti.subject_to(trajectory.evaluate_x(1.1) == 2)
		# opti.subject_to(trajectory.evaluate_dx(1.1) == 2)
		# opti.subject_to(trajectory.evaluate_x(1.6) == 2)
		opti.subject_to(trajectory.get_sum_detaT() == 6)

		opti.solver('ipopt')
		opti.solve()
		serialized_solution = trajectory.serialize_opti_vars()
		serialized_solution_yaml = yaml.dump(serialized_solution, indent=2)
		rich.print('serialized_solution', serialized_solution)
		rich.print('serialized_solution', serialized_solution_yaml)



		# load
		opti_loadable = OptiLoadable()
		trajectory_loaded = SplineTrajectory(**spline_parameters, opti=opti_loadable)
		trajectory_loaded.deserialize_opti_vars_load_solution(
			serialized_solution
		)

		val = trajectory_loaded.evaluate_x(0.5)
		val_eval = opti_loadable.value(val)
		val_eval_orig = opti.value(trajectory.evaluate_x(0.5))
		#rich.print('val_loaded', val)
		rich.print('val_loaded_eval', val_eval)
		rich.print('val_eval_orig', val_eval_orig)


		# compare loaded and orig
		t_vals = np.arange(-1, 6+1, step=0.01)
		print('x_loaded')
		x_loaded = opti_loadable.value(trajectory_loaded.evaluate_x(t_vals))
		print('x_loaded DONE')
		print('\nx_original')
		x_original = opti.value(trajectory.evaluate_x(t_vals))
		print('\nx_original DONE')

		np.testing.assert_allclose(x_loaded, x_original)

		# plot both in one plot
		trajectory_loaded.evaluate_solution_and_plot_full_trajectory(show_plot=False)
		trajectory.evaluate_solution_and_plot_full_trajectory(show_plot=False)
		plt.show()


