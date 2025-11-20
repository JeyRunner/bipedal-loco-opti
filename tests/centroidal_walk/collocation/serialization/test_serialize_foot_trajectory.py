from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import rich
from centroidal_walk.collocation.serialization.OptiDecorators import *
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import *
from centroidal_walk.collocation.opti.casadi_util.OptiLoadable import *
from centroidal_walk.collocation.plotting import *


class TestSerializeSplineTrajectory(TestCase):

	def test_serialize_SplineTrajectory(self):
		opti = OptiWithSimpleBounds.create()

		foot_trajectory_parameters = dict(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			param_opti_start_foot_pos=np.ones(3),
			param_opti_start_foot_dpos=np.ones(3)*0.5,
			num_phases=5+4,
			total_duration=6,
			dimensions=3,
		)

		foot_trajectory = FootTrajectory(**foot_trajectory_parameters, opti=opti)
		foot_trajectory.set_initial_opti_values(init_for_duration_of_each_phase=1)

		opti.subject_to(foot_trajectory.evaluate_foot_pos(0.1) == 2)
		opti.subject_to(foot_trajectory.evaluate_foot_force(0.1) == 2)
		opti.subject_to(foot_trajectory.evaluate_foot_pos(3) == 2)
		opti.subject_to(foot_trajectory.evaluate_foot_pos(3.1) == 3)
		opti.subject_to(foot_trajectory.evaluate_foot_pos(3.3) == 2)
		opti.subject_to(foot_trajectory.evaluate_foot_pos(3.5) == 3)
		# opti.subject_to(trajectory.evaluate_x(1.1) == 2)
		# opti.subject_to(trajectory.evaluate_dx(1.1) == 2)
		# opti.subject_to(trajectory.evaluate_x(1.6) == 2)

		opti.solver('ipopt')
		opti.bake_solve_with_simple_var_bounds('ipopt', {}, {})
		opti.solve_with_simple_var_bounds()

		serialized_solution = foot_trajectory.serialize_opti_vars()
		serialized_solution_yaml = yaml.dump(serialized_solution, indent=2)
		rich.print('serialized_solution', serialized_solution)
		rich.print('serialized_solution', serialized_solution_yaml)



		# load
		opti_loadable = OptiLoadable()
		trajectory_loaded = FootTrajectory(**foot_trajectory_parameters, opti=opti_loadable)
		trajectory_loaded.deserialize_opti_vars_load_solution(
			yaml.load(serialized_solution_yaml, Loader=Loader)
		)

		val = trajectory_loaded.evaluate_foot_pos(0.5)
		val_eval = opti_loadable.value(val)
		val_eval_orig = opti.value(foot_trajectory.evaluate_foot_pos(0.5))
		#rich.print('val_loaded', val)
		rich.print('val_loaded_eval', val_eval)
		rich.print('val_loaded_duration', trajectory_loaded.phases[0].evaluate_solution_duration())
		rich.print('val_eval_orig', val_eval_orig)


		# compare loaded and orig
		t_vals = np.arange(-1, 6+1, step=0.01)
		print('x_loaded')
		x_pos_loaded = opti_loadable.value(trajectory_loaded.evaluate_foot_pos(t_vals))
		x_force_loaded = opti_loadable.value(trajectory_loaded.evaluate_foot_force(t_vals))
		print('x_loaded DONE')
		print('\nx_original')
		x_pos_original = opti.value(foot_trajectory.evaluate_foot_pos(t_vals))
		x_force_original = opti.value(foot_trajectory.evaluate_foot_force(t_vals))
		print('\nx_original DONE')

		np.testing.assert_allclose(x_pos_loaded, x_pos_original)
		np.testing.assert_allclose(x_force_loaded, x_force_original)

		# plot both in one plot
		plot_foot_trajectory(trajectory_loaded, only_plot_dimensions=np.s_[0])
		plt.figure()
		plot_foot_trajectory(foot_trajectory, only_plot_dimensions=np.s_[0])
		plt.show()

		#plt.show()


