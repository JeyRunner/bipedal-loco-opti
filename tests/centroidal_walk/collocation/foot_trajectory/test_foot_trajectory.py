from unittest import TestCase

import casadi
from rich import print

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import FlightPhase
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory
from tests.test_util import *


class TestFootTrajectory(TestCase):

	@staticmethod
	def dummy_solve(foot_trajectory, use_simple_var_bounds=True, add_dummy_constraints=True, solve=True):
		opti = foot_trajectory.opti

		# dummy constraints
		if add_dummy_constraints:
			opti.subject_to(foot_trajectory.evaluate_foot_pos(1.9) == 1)
			opti.subject_to(foot_trajectory.evaluate_foot_pos(2.5) == 1)
			opti.subject_to(foot_trajectory.phases[0].duration < 1)
			opti.subject_to(foot_trajectory.evaluate_foot_force(1) == 0)

		max_iter = 250 if solve else 0

		p_opts = {
		}
		s_opts = {
			"max_iter": max_iter,
		}
		if use_simple_var_bounds:
			opti.bake_solve_with_simple_var_bounds('ipopt', p_opts, s_opts)
		else:
			opti.solver('ipopt', p_opts, s_opts)

		try:
			if use_simple_var_bounds:
				opti.solve_with_simple_var_bounds()
			else:
				opti.solve()
		except:
			pass


	def test_opti_foot_trajectory_single_deltaT_flexible(self):
		opti = casadi.Opti()#OptiWithSimpleBounds.create()

		start_x = np.zeros(2)

		x_trajectory = FootTrajectory(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			num_phases=5,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			#param_opti_start_foot_force=start_x,
			#param_opti_start_foot_dforce=start_x,
			total_duration=5.0,
			opti=opti,
		)


		# add some constraints to spline trajectory#
		# test evaluate function
		#opti.subject_to(x_trajectory.get_total_duration() == 5)
		#opti.subject_to(x_trajectory.evaluate_foot_pos(0.4) == 2)
		#opti.subject_to(x_trajectory.evaluate_foot_pos(1) == 1)
		opti.subject_to(x_trajectory.evaluate_foot_pos(1.9) == 1)
		opti.subject_to(x_trajectory.evaluate_foot_pos(2.5) == 1)
		#opti.subject_to(x_trajectory.evaluate_foot_pos(3.1) == 1.5)
		opti.subject_to(x_trajectory.evaluate_foot_pos(3.2) == 2)
		opti.subject_to(x_trajectory.evaluate_foot_pos(4.5) == 3)
		#opti.subject_to(x_trajectory.phases[1].evaluate_foot_pos(0.5) == [1, 1])
		#opti.subject_to(x_trajectory.phases[2].evaluate_foot_pos(0.5) == [0.5, 0.5])
		#opti.subject_to(x_trajectory.phases[3].evaluate_foot_pos(0.5) == [0, 0])
		#opti.subject_to(x_trajectory.phases[3].evaluate_foot_pos(0.75) == [1, 0])
		#opti.subject_to(x_trajectory.phases[3].evaluate_foot_pos(0.999) == [0, 0])
		opti.subject_to(x_trajectory.phases[0].duration < 1)
		#opti.subject_to(x_trajectory.phases[-1].evaluate_foot_pos(0.5) > 2)

		opti.subject_to(x_trajectory.evaluate_foot_force(1) == 0)


		# run solver
		opti.solver('ipopt')
		#opti.bake_solve_with_simple_var_bounds('ipopt', dict(), dict())
		# deltaT should not be 0 because we divide by it in the polynomial calc
		x_trajectory.set_initial_opti_values()
		try:
			#opti.solve_with_simple_var_bounds()
			opti.solve()
		except Exception as error:
			print(error)
			opti.debug.show_infeasibilities()

		#self.assertTrue(opti.stats()['success'])

		t_vals = np.arange(0, 6.1, step=0.0001)
		x_trajectory_evaluated = x_trajectory.evaluate_solution_x(t_vals)
		#print(x_trajectory_evaluated.foot_pos_trajectory.values)

		# test compare evaluate_foot_pos
		x_trajectory_evaluated_pos_via_mx = opti.value(x_trajectory.evaluate_foot_pos(t_vals))
		#self.assertTrue(np.all(np.abs(x_trajectory_evaluated.foot_pos_trajectory.values - x_trajectory_evaluated_pos_via_mx) < 1e-5))


		#plt.show()
		# plot phases individual
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
		plt.plot(t_vals, x_trajectory_evaluated.foot_pos_trajectory.values[0, :], label='np')
		#plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
		plt.scatter(x_trajectory_evaluated.foot_pos_trajectory.spline_connection_points.t_values,
					x_trajectory_evaluated.foot_pos_trajectory.spline_connection_points.x_values[0, :] #.swapaxes(1, 0) # make time first axis
					)
		for phase_end_t in x_trajectory_evaluated.phase_end_times:
			plt.vlines(x=phase_end_t, ymin=0, ymax=1, colors='grey', linestyles='dashed')
		plt.legend()
		save_fig(self)

		print("##", opti.value(x_trajectory.phases[1].evaluate_foot_pos(np.array([0.5, 0.75, 0.9]))))

		#plt.show()

		self.assertTrue(opti.stats()['success'])






	def test_opti_foot_trajectory_is_dx_of_other_spline(self):
		opti = OptiWithSimpleBounds.create() # casadi.Opti()

		start_x = np.zeros(2)
		total_duration = 6

		com_trajectory = SplineTrajectory(
			num_polynomials_for_trajectory=8,
			param_opti_start_x=start_x,
			param_opti_start_dx=np.zeros(2),
			opti=opti,
			given_total_duration=total_duration,
			spline_polynomial_degree=3,
		)

		foot_trajectory = FootTrajectory(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			num_phases=5,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			# param_opti_start_foot_force=start_x,
			# param_opti_start_foot_dforce=start_x,
			total_duration=total_duration,
			opti=opti
		)


		# add some constraints to spline trajectory#
		# test evaluate function
		#opti.subject_to(foot_trajectory.get_total_duration() == total_duration)
		# enforce dynamics as middle of each com trajectory polynomial
		constraint_points_t = np.zeros(0)
		t = 0
		for p in com_trajectory.poly_list:
			# test enforce for multiple points
			num_constraint_points = 3
			for i in range(num_constraint_points):
				deltaT = (i + 1) * (p.deltaT / (num_constraint_points + 1))
				constraint_points_t = np.hstack([constraint_points_t, t + deltaT])
				# deltaT = (i + 0) * (p.deltaT / (num_constraint_points + 0))
				print(f"> enforce constraint at deltaT = {deltaT} (deltaT total is {p.deltaT})")
				opti.subject_to(p.evaluate_dx(deltaT) == foot_trajectory.evaluate_foot_force(t + deltaT))
				# limit dx
				opti.subject_to(opti.bounded(-1, foot_trajectory.evaluate_foot_force(t+deltaT), 1))
			# + self.x_opti_foot1.evaluate_foot_pos(t+deltaT))
			t += p.deltaT

		# final com pos
		opti.subject_to(com_trajectory.poly_list[-1].x1 == [3, 1])
		# add something from foot pos to constraints so that foot pos var are part of opti
		opti.subject_to(foot_trajectory.evaluate_foot_pos(2) == 0)



		# run solver
		opti.bake_solve_with_simple_var_bounds('ipopt')
		# deltaT should not be 0 because we divide by it in the polynomial calc
		foot_trajectory.set_initial_opti_values()
		try:
			opti.solve_with_simple_var_bounds()
		except Exception as error:
			print(error)
			opti.debug.show_infeasibilities()

		#self.assertTrue(opti.stats()['success'])

		t_vals = np.arange(0, 6.1, step=0.0001)
		x_trajectory_evaluated = foot_trajectory.evaluate_solution_x(t_vals)

		# test compare evaluate_foot_pos
		x_trajectory_evaluated_force_via_mx = opti.value(foot_trajectory.evaluate_foot_force(t_vals))


		# plot com
		plt.figure()
		com_trajectory.evaluate_solution_and_plot_full_trajectory(show_plot=False)
		save_fig(self, '_com_pos')

		#plt.show()
		# plot phases individual
		for i, p in enumerate(foot_trajectory.phases):
			print(f'phase {i} duration is {p.evaluate_solution_duration()}')
			if isinstance(p, FlightPhase):
				plt.figure()
				#p._debug_evaluate_solution_plot_force_or_pos()
				p.foot_position.evaluate_solution_and_plot_full_trajectory(show_plot=False, step_t=0.0001)
				plt.title(f"pos of phase {i}")
				save_fig(self, f'_pos_phase{i}_spline')

		# plot all phases together
		plt.figure()
		plt.plot(t_vals, x_trajectory_evaluated_force_via_mx[0, :], label='approx dx - mx')
		plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='approx dx - np')
		plt.plot(t_vals, opti.value(com_trajectory.evaluate_dx(t_vals)[0, :]), label='true dx')
		plt.scatter(constraint_points_t, opti.value(com_trajectory.evaluate_dx(constraint_points_t)[0, :]), label='dx constraints', color='black')

		#plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
		plt.scatter(x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.t_values,
					x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.x_values[0, :] #.swapaxes(1, 0) # make time first axis
					)
		for phase_end_t in x_trajectory_evaluated.phase_end_times:
			plt.vlines(x=phase_end_t, ymin=0, ymax=1, colors='grey', linestyles='dashed')
		plt.legend()
		save_fig(self, '__fforce_approx_dx')

		print("##", opti.value(foot_trajectory.phases[1].evaluate_foot_pos(np.array([0.5, 0.75, 0.9]))))

		plt.show()


		# test compare evaluate_foot_pos
		self.assertTrue(np.all(
			np.abs(x_trajectory_evaluated.foot_force_trajectory.values - x_trajectory_evaluated_force_via_mx) < 1e-5))

		self.assertTrue(opti.stats()['success'])







	def test_opti_foot_trajectory_is_ddx_of_other_spline__2constraints(self):
		opti = OptiWithSimpleBounds.create()

		start_x = np.zeros(2)
		total_duration = 6

		com_trajectory = SplineTrajectory(
			num_polynomials_for_trajectory=14,#14,
			param_opti_start_x=start_x,
			param_opti_start_dx=np.zeros(2),
			opti=opti,
			given_total_duration=total_duration,
			spline_polynomial_degree=4,
			ddx_consistency_constraint=True,
			ddx_consistency_constraint_manually=True # setting this to true seams to yield better results
		)

		foot_trajectory = FootTrajectory(
			num_polynomials_foot_pos_in_flight=3,
			num_polynomials_foot_force_in_contact=3,
			num_phases=5,
			param_opti_start_foot_pos=start_x,
			param_opti_start_foot_dpos=start_x,
			#param_opti_start_foot_force=start_x,
			#param_opti_start_foot_dforce=start_x,
			opti=opti,
			total_duration=total_duration
		)


		# add some constraints to spline trajectory#
		# test evaluate function
		opti.subject_to(foot_trajectory.get_total_duration() == total_duration)
		# enforce dynamics as middle of each com trajectory polynomial
		constraint_points_t = np.zeros(0)
		t = 0
		for p in com_trajectory.poly_list:
			# test enforce for multiple points
			num_constraint_points = 2
			for i in range(num_constraint_points):
				deltaT = (i) * (p.deltaT / (num_constraint_points))
				constraint_points_t = np.hstack([constraint_points_t, t + deltaT])
				print(f"> enforce constraint at deltaT = {deltaT} (deltaT total is {p.deltaT})")
				opti.subject_to(p.evaluate_ddx(deltaT) == foot_trajectory.evaluate_foot_force(t + deltaT))
				# limit ddx
				opti.subject_to(opti.bounded(-4, foot_trajectory.evaluate_foot_force(t+deltaT), 4))
			# + self.x_opti_foot1.evaluate_foot_pos(t+deltaT))
			t += p.deltaT

		# final com pos
		opti.subject_to(com_trajectory.poly_list[-1].x1 == [-2, 1])
		opti.subject_to(com_trajectory.evaluate_x(np.array([2])) > 2)
		opti.subject_to(com_trajectory.evaluate_x(np.array([3])) > -0.5)
		opti.subject_to(com_trajectory.evaluate_x(np.array([3])) < 0.5)
		opti.subject_to(com_trajectory.evaluate_x(np.array([3.7])) == 1)
		opti.subject_to(com_trajectory.evaluate_x(np.array([4])) == 1)

		# add something from foot pos to constraints so that foot pos var are part of opti
		opti.subject_to(foot_trajectory.evaluate_foot_pos(2) == 0)



		# run solver
		p_opts = {
			"expand": False,
			# auto convert MX to SX expression (here it improves efficiency), but seams to yield to more wrong jacobian values
		}
		# ipopt options
		s_opts = {
			"max_iter": 250,
			# testing
			"check_derivatives_for_naninf": "yes",
			"derivative_test": "first-order",
			# "derivative_test_perturbation": 1e-8,
			"point_perturbation_radius": 1000000,
			# "derivative_test_print_all": "yes",

			#"jacobian_approximation": "finite-difference-values",  # "exact" # here exact yields correct values
			# "jacobian_approximation": "exact",  # "exact" does not work
			#"hessian_approximation": "limited-memory",  # seems not to be forwarded correctly to ipopt
			"linear_solver": "mumps"
		}
		opti.bake_solve_with_simple_var_bounds('ipopt', p_opts, s_opts)
		# deltaT should not be 0 because we divide by it in the polynomial calc
		foot_trajectory.set_initial_opti_values(
			init_z_force_for_contact=0
		)
		try:
			opti.solve_with_simple_var_bounds()
		except Exception as error:
			print(error)
			opti.debug.show_infeasibilities()

		t_vals = np.arange(0, 6.1, step=0.0001)
		x_trajectory_evaluated = foot_trajectory.evaluate_solution_x(t_vals)

		# test compare evaluate_foot_pos
		x_trajectory_evaluated_force_via_mx = opti.value(foot_trajectory.evaluate_foot_force(t_vals))
		self.assertTrue(np.all(np.abs(x_trajectory_evaluated.foot_force_trajectory.values - x_trajectory_evaluated_force_via_mx) < 1e-5))


		# plot com
		plt.figure()
		com_trajectory.evaluate_solution_and_plot_full_trajectory(show_plot=False)
		save_fig(self, '_com_pos')

		#plt.show()
		# plot phases individual
		for i, p in enumerate(foot_trajectory.phases):
			print(f'phase {i} duration is {p.evaluate_solution_duration()}')
			if isinstance(p, FlightPhase):
				plt.figure()
				#p._debug_evaluate_solution_plot_force_or_pos()
				p.foot_position.evaluate_solution_and_plot_full_trajectory(show_plot=False, step_t=0.0001)
				plt.title(f"pos of phase {i}")
				save_fig(self, f'_pos_phase{i}_spline')

		# plot all phases together
		plt.figure()
		plt.plot(t_vals, x_trajectory_evaluated_force_via_mx[0, :], label='true ddx - mx')
		plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='true ddx - np')
		plt.plot(t_vals, opti.value(com_trajectory.evaluate_ddx(t_vals)[0, :]), label='approx com_ddx')
		plt.scatter(constraint_points_t, opti.value(com_trajectory.evaluate_ddx(constraint_points_t)[0, :]), label='dx constraints', color='black')

		#plt.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
		plt.scatter(x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.t_values,
					x_trajectory_evaluated.foot_force_trajectory.spline_connection_points.x_values[0, :] #.swapaxes(1, 0) # make time first axis
					)
		for phase_end_t in x_trajectory_evaluated.phase_end_times:
			plt.vlines(x=phase_end_t, ymin=0, ymax=1, colors='grey', linestyles='dashed')
		plt.legend()
		save_fig(self, '__fforce_approx_dx')

		print("##", opti.value(foot_trajectory.phases[1].evaluate_foot_pos(np.array([0.5, 0.75, 0.9]))))

		#plt.show()

		self.assertTrue(opti.stats()['success'])
