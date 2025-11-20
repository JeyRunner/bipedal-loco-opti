from unittest import TestCase

import casadi
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import FlightPhase
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.plotting import plot_foot_trajectory
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory
from tests.test_util import *


class TestFootTrajectoryInit(TestCase):

	def dummy_solve(self, foot_trajectory):
		opti = foot_trajectory.opti

		# dummy constraints
		opti.subject_to(foot_trajectory.evaluate_foot_pos(1.9) == 1)
		opti.subject_to(foot_trajectory.evaluate_foot_pos(2.5) == 1)
		opti.subject_to(foot_trajectory.phases[0].duration < 1)
		opti.subject_to(foot_trajectory.evaluate_foot_force(1) == 0)

		opti.bake_solve_with_simple_var_bounds('ipopt', dict(), dict(max_iter=0))
		#opti.solver('ipopt')
		try:
			opti.solve_with_simple_var_bounds()
			#opti.solve()
		except:
			pass



	def test_foot_trajectory_init__first_phase_contact_len5(self):
		#opti = casadi.Opti()
		opti = OptiWithSimpleBounds.create()

		start_x = np.zeros(3)

		x_trajectory = FootTrajectory(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			first_phase_type=PhaseType.CONTACT,
			num_phases=5,
			dimensions=3,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			#param_opti_start_foot_force=start_x,
			#param_opti_start_foot_dforce=start_x,
			total_duration=5,
			opti=opti,

		)

		x_trajectory.set_initial_opti_values(
			init_z_force_for_contact=0,
			init_z_height_for_flight=1
		)

		self.dummy_solve(x_trajectory)


		plot_foot_trajectory(
			x_trajectory,
			plot_force_or_pos='pos',
			only_plot_dimensions=np.s_[2]
		)
		#plt.show()

		# check
		# each phase will have duration 1
		np.testing.assert_array_equal(x_trajectory.evaluate_solution_x(
			np.arange(0.5, 5, step=1)).foot_pos_trajectory.values[2],
			np.array([
				0, 1, 0, 1, 0
			])
		)


	def test_foot_trajectory_init__first_phase_flight_len6(self):
		#opti = casadi.Opti()
		opti = OptiWithSimpleBounds.create()

		start_x = np.zeros(3)

		x_trajectory = FootTrajectory(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			first_phase_type=PhaseType.FLIGHT,
			num_phases=6,
			dimensions=3,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			#param_opti_start_foot_force=start_x,
			#param_opti_start_foot_dforce=start_x,
			total_duration=6,
			opti=opti,

		)

		x_trajectory.set_initial_opti_values(
			init_z_force_for_contact=0,
			init_z_height_for_flight=1
		)

		self.dummy_solve(x_trajectory)


		plot_foot_trajectory(
			x_trajectory,
			plot_force_or_pos='pos',
			only_plot_dimensions=np.s_[2]
		)
		#plt.show()

		# check
		# each phase will have duration 1
		np.testing.assert_array_equal(x_trajectory.evaluate_solution_x(
			np.arange(0.5, 6, step=1)).foot_pos_trajectory.values[2],
			np.array([
				1, 0, 1, 0, 1, 0
			])
		)

