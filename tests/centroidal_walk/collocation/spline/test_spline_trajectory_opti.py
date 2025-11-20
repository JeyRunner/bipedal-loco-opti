from unittest import TestCase

from rich import print

from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from tests.test_util import *

from centroidal_walk.collocation.spline.spline_trajectory import *

class TestSplineTrajectory(TestCase):

	num_polynomials_for_com_trajectory = 5

	def test_opti_single_spline3_trajectory_deltaT_flexible(self):
		opti = OptiWithSimpleBounds()

		start_x = np.zeros(2)
		start_dx = np.zeros(2)
		end_x = np.ones(2)*4
		end_dx = np.ones(2)*10

		x_trajectory = SplineTrajectory(
			self.num_polynomials_for_com_trajectory,
			start_x,
			start_dx,
			opti,
			param_opti_end_x=end_x,
			param_opti_end_dx=end_dx,
			spline_polynomial_degree=3,
			dimensions=2,
			#given_total_duration=6
			_debug__casadi_ifelse_nested=True
		)

		#print(x_trajectory.x_opti_vars[x_trajectory.id_poly_deltaT][0])
		#print('jac deltaT ', casadi.jacobian(MX.sym('t', 1), x_trajectory.x_opti_vars[x_trajectory.id_poly_deltaT][0]))
		#return


		# add some constraints to spline trajectory#
		# test evaluate function
		opti.subject_to(x_trajectory.get_sum_detaT() == 6)

		#opti.subject_to(x_trajectory.poly_list[0].evaluate(0.5) == [-1, 2])
		#opti.subject_to(x_trajectory.poly_list[0].x1 == [2, 4])
		#opti.subject_to(x_trajectory.poly_list[-1].x1 == [-2, -1])
		#opti.subject_to(x_trajectory.poly_list[-2].evaluate(0.5) == [2, 1])
		opti.subject_to(x_trajectory.evaluate_x(0.5) == [2, 1])
		opti.subject_to(x_trajectory.evaluate_x(1) == [5, -1])
		opti.subject_to(x_trajectory.evaluate_x(1.5) == [5, -1])
		opti.subject_to(x_trajectory.evaluate_x(2.5) == [5, -1])
		#opti.subject_to(x_trajectory.evaluate_dx(3.5) == [-0.1, 0.1])  # @todo did work before with dx(3.5) == [0, 0]
		opti.subject_to(x_trajectory.evaluate_dx(3.5) == [0, 0])  # @todo did work before with dx(3.5) == [0, 0]
		opti.subject_to(x_trajectory.evaluate_x(5) == [9, 2])
		opti.subject_to(x_trajectory.evaluate_dx(5) == [0, 0])


		# run solver
		p_opts = {
			"expand": True,  # auto convert MX to SX expression (here it improves efficiency), but seams to yield to more wrong jacobian values
		}
		# ipopt options
		s_opts = {
			"max_iter": 50,
			# testing
			"check_derivatives_for_naninf": "yes",
			"derivative_test": "first-order",
			#"derivative_test_perturbation": 1e-8,
			"point_perturbation_radius": 1000000,
			#"derivative_test_print_all": "yes",

			"jacobian_approximation": "finite-difference-values",  # "exact"
			#"jacobian_approximation": "exact",  # "exact" does not work
			#"gradient_approximation": "finite-difference-values",
			"gradient_approximation": "exact",
			"hessian_approximation": "limited-memory",  # seems not to be forwarded correctly to ipopt
			#"fixed_variable_treatment": "make_constraint"

			"linear_solver": "mumps"
		}
		opti.solver('ipopt', p_opts, s_opts)
		#print(casadi.doc_nlpsol('ipopt'))
		#return


		# deltaT should not be 0 because we divide by it in the polynomial calc
		x_trajectory.set_initial_opti_values__zero(init_total_duration=6)
		opti.solve()
		self.assertTrue(opti.stats()['success'])

		# test evaluate via casadi MX expression
		#t_vals = SX.sym('test', 3)
		t_vals = np.array([0.5, 1, 2])#.reshape(1, -1)#.reshape(-1, 1)
		print(t_vals)
		#print(">> x_trajectory.poly_list[0] at [0.5, 1, 1.5] is ", opti.value(x_trajectory.poly_list[0].evaluate(t_vals)))
		#print(">> x_trajectory at [0.5, 1, 1.5] is ", opti.value(x_trajectory.evaluate_x(t_vals)))

		plt.figure()

		# check if np eval and solution eval via MX is the same
		t_vals = np.arange(0, 6, step=0.005)
		x_vals_sol_ref = x_trajectory.evaluate_solution_x(t_vals)
		x_vals_sol_mx = opti.value(x_trajectory.evaluate_x(t_vals))
		#plt.plot(t_vals, x_vals_sol_ref.T, linewidth=3, linestyle='dashed')
		#print("ref", x_vals_sol_ref)
		#print("mx ", x_vals_sol_mx)

		self.assertTrue(np.all(np.abs(x_vals_sol_mx - x_vals_sol_ref) < 1e-10))

		plt.figure()
		x_trajectory.evaluate_solution_and_plot_full_trajectory(step_t=0.00001, show_plot=False)
		save_fig(self)

		plt.figure()
		plt.plot(t_vals, x_trajectory.evaluate_solution_dx(t_vals).T, label="dx")
		save_fig(self, 'dx')

		plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_ddx(t_vals)).T, label="ddx")
		save_fig(self, 'ddx')






	def test_opti_single_spline4_trajectory_deltaT_fixed(self):
		opti = casadi.Opti()

		start_x = np.zeros(2)
		start_dx = np.zeros(2)
		end_x = np.ones(2)*4
		end_dx = np.ones(2)*10

		x_trajectory = SplineTrajectory(
			self.num_polynomials_for_com_trajectory,
			start_x,
			start_dx,
			opti,
			param_opti_end_x=end_x,
			param_opti_end_dx=end_dx,
			spline_polynomial_degree=4,
			given_total_duration=6,
			ddx_consistency_constraint=True,
			ddx_consistency_constraint_manually=False
		)


		# add some constraints to spline trajectory#
		# test evaluate function
		#opti.subject_to(x_trajectory.get_sum_detaT() == 6)
		opti.subject_to(x_trajectory.evaluate_x(0.5) == [2, 1])
		opti.subject_to(x_trajectory.evaluate_x(1) == [5, -1])
		opti.subject_to(x_trajectory.evaluate_x(1.5) == [5, -1])
		opti.subject_to(x_trajectory.evaluate_x(2.5) == [5, -1])
		opti.subject_to(x_trajectory.evaluate_dx(3.5) == [0, 0])
		opti.subject_to(x_trajectory.evaluate_x(5) == [9, 2])
		opti.subject_to(x_trajectory.evaluate_dx(5) == [0, 0])


		# run solver
		opti.solver('ipopt')
		# deltaT should not be 0 because we divide by it in the polynomial calc
		x_trajectory.set_initial_opti_values__zero()
		opti.solve()
		self.assertTrue(opti.stats()['success'])

		# test evaluate via casadi MX expression
		t_vals = np.array([0.5, 1, 2])#.reshape(1, -1)#.reshape(-1, 1)
		print(t_vals)


		# check if np eval and solution eval via MX is the same
		t_vals = np.arange(0, 6, step=0.001)
		x_vals_sol_ref = x_trajectory.evaluate_solution_x(t_vals)
		x_vals_sol_mx = opti.value(x_trajectory.evaluate_x(t_vals))
		#plt.figure()
		#plt.plot(t_vals, x_vals_sol_ref.T, linewidth=3, linestyle='dashed')
		#print("ref", x_vals_sol_ref)
		#print("mx ", x_vals_sol_mx)

		self.assertTrue(np.all(np.abs(x_vals_sol_mx - x_vals_sol_ref) < 1e-10))

		plt.figure()
		x_trajectory.evaluate_solution_and_plot_full_trajectory(step_t=0.00001, show_plot=False)
		save_fig(self)

		plt.figure()
		plt.plot(t_vals, x_trajectory.evaluate_solution_dx(t_vals).T, label="dx")
		save_fig(self, 'dx')

		plt.figure()
		plt.plot(t_vals, opti.value(x_trajectory.evaluate_ddx(t_vals)).T, label="ddx")


		# check ddx continuity condition
		t = np.zeros(1)
		for i in range(len(x_trajectory.poly_list)-1):
			p = x_trajectory.poly_list[i]
			t += opti.value(p.deltaT)
			pre_ddx_end = np.array(opti.value(p.evaluate_ddx(p.deltaT)))
			next_ddx0 = np.array(opti.value(x_trajectory.poly_list[i+1].ddx0))
			next_eval_ddx0 = np.array(opti.value(x_trajectory.poly_list[i+1].evaluate_ddx(0)))
			self.assertTrue(np.all(pre_ddx_end == next_eval_ddx0))

			plt.scatter(t.repeat(2, 0), pre_ddx_end.reshape(-1, 1), label='pre_ddx_end')
			plt.scatter(t.repeat(2, 0), next_ddx0.reshape(-1, 1), label='pre_ddx_end')
			plt.scatter(t.repeat(2, 0), next_eval_ddx0.reshape(-1, 1), label='next_eval_ddx0')

		save_fig(self, 'ddx')
		#plt.show()





	def test_opti_spline3_two_trajectories_one_dt_of_other(self):
		opti = OptiWithSimpleBounds()

		start_x = np.zeros(2)
		start_dx = np.zeros(2)

		x_trajectory = SplineTrajectory(
			self.num_polynomials_for_com_trajectory,
			start_x,
			start_dx,
			opti,
			spline_polynomial_degree=3,
			given_total_duration=6
		)
		dx_trajectory = SplineTrajectory(
			self.num_polynomials_for_com_trajectory,
			start_x,
			start_dx,
			opti,
			spline_polynomial_degree=3,
			#given_total_duration=6
		)


		# add some constraints to spline trajectory#
		# test evaluate function



		# enforce dynamics as middle of each com trajectory polynomial
		t = 0
		num_constraint_points = 3 # @todo worked with 5 (when bug was in sline (to many opti vars))
		for i, p in enumerate(x_trajectory.poly_list):
			# test enforce for multiple points
			#num_points = 3 #if i < (len(x_trajectory.poly_list)-1) else 4
			print(f"for poly {i} using {num_constraint_points} constraints points")
			for i in range(num_constraint_points):
				deltaT = (i + 1) * (p.deltaT / (num_constraint_points + 1))
				print(f"> enforce constraint at deltaT = {deltaT} (deltaT total is {p.deltaT})")
				opti.subject_to(p.evaluate_dx(deltaT) == dx_trajectory.evaluate_x(t + deltaT))
			t += p.deltaT

		# final x pos
		opti.subject_to(x_trajectory.evaluate_x(5) == [7, 1])

		# dx constraints
		opti.subject_to(dx_trajectory.get_sum_detaT() == 6)
		opti.subject_to(dx_trajectory.poly_list[-2].deltaT >= 2)



		# run solver
		p_opts = {
			"expand": True,
			# auto convert MX to SX expression (here it improves efficiency), but seams to yield to more wrong jacobian values
		}
		# ipopt options
		s_opts = {
			"max_iter": 150,
			# testing
			"check_derivatives_for_naninf": "yes",
			"derivative_test": "first-order",
			# "derivative_test_perturbation": 1e-8,
			"point_perturbation_radius": 1000000,
			# "derivative_test_print_all": "yes",

			#"jacobian_approximation": "finite-difference-values",  # "exact" # here exact yields correct values
			# "jacobian_approximation": "exact",  # "exact" does not work
			#"hessian_approximation": "limited-memory",  # seems not to be forwarded correctly to ipopt
			# "fixed_variable_treatment": "make_constraint"

			"linear_solver": "mumps"
		}
		opti.solver('ipopt', p_opts, s_opts)
		# deltaT should not be 0 because we divide by it in the polynomial calc
		x_trajectory.set_initial_opti_values__zero()
		dx_trajectory.set_initial_opti_values__zero(init_total_duration=6)
		opti.solve()
		#opti.solve_as_nlp(p_opts, s_opts)
		self.assertTrue(opti.stats()['success'])

		plt.figure()
		x_trajectory.evaluate_solution_and_plot_full_trajectory(step_t=0.00001, only_plot_dimensions=0, show_plot=False)
		save_fig(self, 'x')

		# compare true derivative with dx_trajectory
		t_vals = np.arange(0, 6, 0.001)
		dx_true = x_trajectory.evaluate_solution_dx(t_vals)
		plt.figure()
		plt.plot(t_vals, dx_true[0], linestyle='dashed', label='dx_x_trajectory', linewidth=3)

		dx_trajectory.evaluate_solution_and_plot_full_trajectory(step_t=0.00001, only_plot_dimensions=0, show_plot=False)
		plt.title(f"x and dx splines (num consistency constraints per poly {num_constraint_points})")
		plt.legend()
		#plt.show()
		save_fig(self)

		dx_trajectory_x, dx_trajectory_spline_points = dx_trajectory.evaluate_solution_x(t_vals, output_spline_connection_points=True)
		dx_trajectory_approx_error = dx_true[0] - dx_trajectory_x[0]
		dx_trajectory_approx_error_max = np.max(np.abs(dx_trajectory_approx_error))
		print("dx_trajectory_approx_error_max = ", dx_trajectory_approx_error_max)

		plt.figure()
		plt.plot(t_vals, dx_trajectory_approx_error, label="dx approx error")
		print(dx_trajectory_spline_points)
		plt.scatter(dx_trajectory_spline_points.t_values, np.zeros(dx_trajectory_spline_points.t_values.shape[0]))
		plt.title(f"dx approx error (max = {dx_trajectory_approx_error_max}, \n num consistency constraints per poly {num_constraint_points})")
		save_fig(self, "dx_approx_error")
		#plt.show()

		self.assertLess(dx_trajectory_approx_error_max, 0.2)
