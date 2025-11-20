from unittest import TestCase

import casadi
import matplotlib.pyplot as plt
import numpy as np
from rich import print

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import FlightPhase
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory
from tests.centroidal_walk.collocation.foot_trajectory.test_foot_trajectory import TestFootTrajectory
from tests.test_util import *


class TestFootTrajectoryRollingShift(TestCase):


	def test_opti_foot_trajectory_single_rolling_shift_duration(self):
		opti = OptiWithSimpleBounds.create()

		start_x = np.zeros(3)
		total_duration = 4
		time_shift = opti.variable(1) #0.2

		x_trajectory = FootTrajectory(
			dimensions=3,
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			num_phases=4,
			phase_duration_min=0.1,
			first_phase_type=PhaseType.FLIGHT,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			total_duration=total_duration,
			rolling_time_shift_duration=time_shift,
			#fixed_phase_durations=np.ones(4)/4 * total_duration,
			foot_force_at_trajectory_end_and_start_variable=False,
			last_phase_duration_implicit=True,
			base_euler_angle_difference_end_to_start=np.ones(3)*2,
			opti=opti,
		)
		print(f'phase durations fixed {x_trajectory.fixed_phase_durations}')

		#opti.subject_to(time_shift >= 0.0)
		opti.subject_to(x_trajectory.evaluate_foot_force(0.25) == 0.4)
		opti.subject_to(x_trajectory.evaluate_foot_force(0.3) == 0.7)
		opti.subject_to(x_trajectory.evaluate_foot_force(0.5) == 0.9)
		opti.subject_to(x_trajectory.evaluate_foot_force(1.4) == 1)
		opti.subject_to(x_trajectory.evaluate_foot_force(1.6) == 1.5)
		opti.subject_to(x_trajectory.evaluate_foot_force(3.5) == 1.2)
		opti.subject_to(x_trajectory.evaluate_foot_force(3.9) == 1)
		opti.subject_to(x_trajectory.phases[0].duration < 1)
		opti.subject_to(x_trajectory.evaluate_foot_pos(0.9) == 0)
		opti.subject_to(x_trajectory.evaluate_foot_pos(2.5) == 0)
		opti.subject_to(list(x_trajectory.get_flight_phases())[-1].foot_position.poly_list[-1].evaluate(0.5) == 0)

		x_trajectory.set_initial_opti_values()
		TestFootTrajectory.dummy_solve(x_trajectory,
									   use_simple_var_bounds=True,
									   add_dummy_constraints=False)

		#self.assertTrue(opti.stats()['success'])

		t_vals = np.arange(0, 6.1, step=0.0001)
		x_trajectory_evaluated = x_trajectory.evaluate_solution_x(t_vals)
		#print(x_trajectory_evaluated.foot_pos_trajectory.values)

		# test compare evaluate_foot_pos
		x_trajectory_evaluated_pos_via_mx = opti.value(x_trajectory.evaluate_foot_force(t_vals))
		#self.assertTrue(np.all(np.abs(x_trajectory_evaluated.foot_pos_trajectory.values - x_trajectory_evaluated_pos_via_mx) < 1e-5))


		#plt.show()
		# plot phases individual
		if False:
			for i, p in enumerate(x_trajectory.phases):
				print(f'phase {i} duration is {p.evaluate_solution_duration()}')
				if isinstance(p, FlightPhase):
					plt.figure()
					#p._debug_evaluate_solution_plot_force_or_pos()
					p.foot_position.evaluate_solution_and_plot_full_trajectory(show_plot=False, step_t=0.0001)
					plt.title(f"pos of phase {i}")
					save_fig(self, f'_pos_phase{i}_spline')

		# plot all phases together
		plt.figure()
		plt.plot(t_vals, x_trajectory_evaluated_pos_via_mx[0, :], label='mx evaluated')
		plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
		# plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
		plt.scatter(x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.t_values,
					x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.x_values[0, :] #.swapaxes(1, 0) # make time first axis
					)

		for phase_end_t in x_trajectory_evaluated.phase_end_times:
			plt.vlines(x=phase_end_t, ymin=0, ymax=1, colors='grey', linestyles='dashed')
			print('phase end', phase_end_t)
		plt.legend()
		#save_fig(self)

		for p in x_trajectory.phases:
			print('phase duration', opti.value(p.duration))
		print('time shift', opti.value(time_shift))

		#print("##", opti.value(x_trajectory.phases[1].evaluate_foot_pos(np.array([0.5, 0.75, 0.9]))))
		plt.show()



