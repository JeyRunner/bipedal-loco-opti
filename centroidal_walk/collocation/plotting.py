import numpy as np
from matplotlib import pyplot as plt

from centroidal_walk.collocation.opti.centroidal_dyn_poly import CentroidalDynPoly
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.plotting_data import FootConstraintTimePoints
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory


def plot_foot_trajectory(x_opti_foot_trajectory: FootTrajectory, plot_force_or_pos='pos', only_plot_dimensions=None):
	opti = x_opti_foot_trajectory.opti
	total_duration = opti.value(x_opti_foot_trajectory.get_total_duration())

	t_vals = np.arange(0, total_duration, step=0.001)
	foot_eval_trajectory = x_opti_foot_trajectory.evaluate_solution_x(t_vals)
	FootTrajectory.plot_evaluated_foot_trajectory(
		t_vals,
		foot_eval_trajectory,
		plot_force_or_pos=plot_force_or_pos,
		only_plot_dimensions=only_plot_dimensions
	)

	for p in x_opti_foot_trajectory.phases:
		print(f'>> phase duration {p.evaluate_solution_duration():5.3f}s ({p.get_phase_type()})')
	plt.legend()




def plot_com_and_foot_trajectories_xyz_threaded(
		collocation_opti: CentroidalDynPoly,
		x_opti_com_pos: SplineTrajectory,
		x_opti_com_angle: SplineTrajectory,
		x_opti_feet: list[FootTrajectory],
		constraints_t_dynamics: np.ndarray,
		constraints_t_feet: list[FootConstraintTimePoints]
		):
	import threading, time
	def plot_func():
		print("# showing plot")
		plot_com_and_foot_trajectories_xyz(collocation_opti,
										   x_opti_com_pos,
										   x_opti_com_angle,
										   x_opti_feet,
										   constraints_t_dynamics,
										   constraints_t_feet)
		time.sleep(2.4)

	thread = threading.Thread(target=plot_func)
	thread.start()
	thread.join()




def plot_com_and_foot_trajectories_xyz(
		collocation_opti: CentroidalDynPoly,
		x_opti_com_pos: SplineTrajectory,
		x_opti_com_angle: SplineTrajectory,
		x_opti_feet: list[FootTrajectory],
		constraints_t_dynamics: np.ndarray,
		constraints_t_feet: list[FootConstraintTimePoints],
		show_plot=True,
		show_non_blocking=True,
		show_plots_angular=True,
		show_angular_dyn=True
		):
	#plt.style.use('science')
	opti = x_opti_com_pos.opti
	total_duration = opti.value(x_opti_com_pos.given_total_duration)
	base_poly_duration = total_duration/len(x_opti_com_pos.poly_list)


	# get com angular torques true and approx
	# manually fill this array (does not work for multiple t values)
	if show_angular_dyn:
		t_vals_angle = np.arange(0, total_duration + base_poly_duration - 1e-5, step=base_poly_duration / 2)
		angular_dyn_com_torques_vals = np.zeros((3, t_vals_angle.shape[0]))
		angular_dyn_torques_from_feet_vals = np.zeros((3, t_vals_angle.shape[0]))
		if x_opti_com_angle is not None:
			for i in range(angular_dyn_com_torques_vals.shape[1]):
				angular_dyn_com_torques, angular_dyn_torques_from_feet = get_com_angular_dyn_torques(
					collocation_opti,
					x_opti_com_pos,
					x_opti_com_angle,
					x_opti_feet,
					t_vals=t_vals_angle[i]
				)
				angular_dyn_com_torques_vals[:, i] = angular_dyn_com_torques
				angular_dyn_torques_from_feet_vals[:, i] = angular_dyn_torques_from_feet



	def add_row_header(ax, title):
		pad = 5
		ax.annotate(title, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
					xycoords=ax.yaxis.label, textcoords='offset points',
					size='large', ha='right', va='center')

	step_t = 0.0001
	t_vals = np.arange(0, total_duration, step=0.001)

	fig, axes = plt.subplots(nrows=1 + len(x_opti_feet)*2 + 1, ncols=3, sharex=True)

	feet_trajectories = []
	for foot_i in range(len(x_opti_feet)):
		feet_trajectories.append(x_opti_feet[foot_i].evaluate_solution_x(t_vals))
	print('> plotting: evaluated foot trajectories')

	# for all columns: x,y,z
	col_titles = ['X', 'Y', 'Z']
	for k_dim in range(len(axes[0])):
		ax_row = 0
		ax_col = axes[:, k_dim]
		ax_col[0].set_title(col_titles[k_dim])

		# fill rows
		# com pos
		if k_dim == 0 : add_row_header(ax_col[0], 'CoM \nPos')
		x_opti_com_pos.evaluate_solution_and_plot_full_trajectory(show_plot=False,
																  step_t=step_t, only_plot_dimensions=k_dim,
																  axis=ax_col[0])

		# feet
		for foot_i in range(len(x_opti_feet)):
			ax_row = foot_i*2 + 1

			# foot pos
			if k_dim == 0: add_row_header(ax_col[ax_row], f'Foot {foot_i} \nPos')
			foot_eval_trajectory = feet_trajectories[foot_i] #x_opti_feet[foot_i].evaluate_solution_x(t_vals)
			FootTrajectory.plot_evaluated_foot_trajectory(
				t_vals,
				foot_eval_trajectory,
				'pos',
				only_plot_dimensions=k_dim,
				axis=ax_col[ax_row])

			# foot pos constraint points
			constraints_t_feet_pos_np = np.array(opti.value(
				constraints_t_feet[foot_i].constraint_points_t__feet_pos
			))
			ax_col[ax_row].scatter(constraints_t_feet_pos_np,
								   np.array(opti.value(x_opti_feet[foot_i].evaluate_foot_pos(
									   constraints_t_feet_pos_np
								   )))[k_dim, :],
								   label='kin constr.',
								   marker='x',
								   color='black', alpha=0.8, s=18)
			ax_col[ax_row].legend()

			# foot force
			if k_dim == 0: add_row_header(ax_col[ax_row+1], f'Foot {foot_i} \nForce')
			foot_eval_trajectory = x_opti_feet[foot_i].evaluate_solution_x(t_vals)
			FootTrajectory.plot_evaluated_foot_trajectory(
				t_vals,
				foot_eval_trajectory,
				'force',
				only_plot_dimensions=k_dim,
				axis=ax_col[ax_row+1])

			# foot force constraint points
			constraints_t_feet_contact_force_np = np.array(opti.value(
				constraints_t_feet[foot_i].constraint_points_t__feet_contact_force
			))
			ax_col[ax_row+1].scatter(constraints_t_feet_contact_force_np,
								   np.array(opti.value(x_opti_feet[foot_i].evaluate_foot_force(
									   constraints_t_feet_contact_force_np
								   )))[k_dim, :],
								   label='f constr.',
								   marker='x',
								   color='black', alpha=0.8, s=18)
			ax_col[ax_row + 1].legend()

		ax_row = 1 + len(x_opti_feet)*2


		# com ddpos
		if k_dim == 0: add_row_header(ax_col[ax_row], f'CoM ddpos')
		com_ddx_approx = np.array(opti.value(x_opti_com_pos.evaluate_ddx(t_vals)))
		ax_col[ax_row].plot(t_vals, com_ddx_approx[k_dim, :], label='approx ddCoM')

		com_ddx_true = opti.value(
			collocation_opti.get_dynamics_com_ddpos(foot_force=collocation_opti.evaluate_sum_foot_forces(t_vals))
		)
		ax_col[ax_row].plot(t_vals, com_ddx_true[k_dim, :], label='true')

		ax_col[ax_row].scatter(constraints_t_dynamics,
							   np.array(opti.value(x_opti_com_pos.evaluate_ddx(constraints_t_dynamics)))[k_dim, :],
							   label='dyn. constr.',
							   marker='x',
							   color='black', alpha=0.8, s=18)

		ax_col[ax_row].legend()




	# fix spacing
	fig.tight_layout(pad=0.05, h_pad=0.05, w_pad=0.00)
	fig.subplots_adjust(left=0.1, top=0.94, bottom=0.05, right=0.99)

	__plt_maximize_window()



	#############
	# plot com angular part and ddangle
	if show_plots_angular:
		if x_opti_com_angle is not None:
			fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True)
			for k_dim in range(len(axes[0])):
				ax_row = 0
				ax_col = axes[:, k_dim]
				ax_col[0].set_title(col_titles[k_dim])

				# angle
				if k_dim == 0: add_row_header(ax_col[ax_row], 'CoM \nAngle')
				x_opti_com_angle.evaluate_solution_and_plot_full_trajectory(show_plot=False,
																			step_t=step_t, only_plot_dimensions=k_dim,
																			axis=ax_col[0])
				ax_row += 1

				# ddangle
				if show_angular_dyn:
					if k_dim == 0: add_row_header(ax_col[ax_row], f'CoM \nangular acc')
					ax_col[ax_row].plot(t_vals_angle, angular_dyn_com_torques_vals[k_dim, :], label='approx torque \non com')
					ax_col[ax_row].plot(t_vals_angle, angular_dyn_torques_from_feet_vals[k_dim, :], label='actual torque \non com by feet')

					ax_col[ax_row].scatter(
						constraints_t_dynamics,
						 #np.array(opti.value(x_opti_com_angle.evaluate_ddx(constraints_t_dynamics)))[k_dim, :],
						 get_com_angular_dyn_torques_from_feet(collocation_opti, x_opti_com_pos, x_opti_feet, constraints_t_dynamics)[k_dim, :],
						 label='dyn. constr.',
						 marker='x',
						 color='black', alpha=0.8, s=18)

					ax_col[ax_row].legend()



	__plt_maximize_window()
	if show_plot and not show_non_blocking:
		plt.show()
	elif show_non_blocking:
		plt.draw()
		plt.pause(0.01)




def get_true_com_angular_acc(collocation_opti, x_opti_com_pos, x_opti_com_angle, x_opti_feet, t_vals):
	assert False, "deprecated"
	com_ddx_true = np.array(collocation_opti.opti.value(
		collocation_opti.get_dynamics_com_angular_acc(
			com_pos=x_opti_com_pos.evaluate_x(t_vals),
			com_angle=x_opti_com_angle.evaluate_x(t_vals),
			com_dangle=x_opti_com_angle.evaluate_dx(t_vals),
			feet_positions=[foot.evaluate_foot_pos(t_vals) for foot in x_opti_feet],
			feet_forces=[foot.evaluate_foot_force(t_vals) for foot in x_opti_feet],
		)
	))
	return com_ddx_true


def get_com_angular_dyn_torques(collocation_opti: CentroidalDynPoly, x_opti_com_pos, x_opti_com_angle, x_opti_feet, t_vals):
	angular_dyn_com_torques, angular_dyn_torques_from_feet = collocation_opti.get_dynamics_com_angular_acc_violation(
			com_pos=x_opti_com_pos.evaluate_x(t_vals),
			com_angle=x_opti_com_angle.evaluate_x(t_vals),
			com_dangle=x_opti_com_angle.evaluate_dx(t_vals),
			com_ddangle=x_opti_com_angle.evaluate_ddx(t_vals),
			feet_positions=[foot.evaluate_foot_pos(t_vals) for foot in x_opti_feet],
			feet_forces=[foot.evaluate_foot_force(t_vals) for foot in x_opti_feet],
		)
	np_angular_dyn_com_torques = np.array(collocation_opti.opti.value(angular_dyn_com_torques))
	np_angular_dyn_torques_from_feet = np.array(collocation_opti.opti.value(angular_dyn_torques_from_feet))
	return np_angular_dyn_com_torques, np_angular_dyn_torques_from_feet

def get_com_angular_dyn_torques_from_feet(collocation_opti: CentroidalDynPoly, x_opti_com_pos, x_opti_feet, t_vals):
	angular_dyn_com_torques, angular_dyn_torques_from_feet = collocation_opti.get_dynamics_com_angular_acc_violation(
			com_pos=x_opti_com_pos.evaluate_x(t_vals),
			com_angle=np.zeros(3),  # not used for torques from feet
			com_dangle=np.zeros(3),  # not used for torques from feet
			com_ddangle=np.zeros(3),  # not used for torques from feet
			feet_positions=[foot.evaluate_foot_pos(t_vals) for foot in x_opti_feet],
			feet_forces=[foot.evaluate_foot_force(t_vals) for foot in x_opti_feet],
		)
	np_angular_dyn_torques_from_feet = np.array(collocation_opti.opti.value(angular_dyn_torques_from_feet))
	return np_angular_dyn_torques_from_feet





def plot_com_pos_derivative(x_opti_com: SplineTrajectory, k_dim=2):
	opti = x_opti_com.opti
	total_duration = opti.value(x_opti_com.given_total_duration)

	step_t = 0.0001
	t_vals = np.arange(0, total_duration, step=step_t)

	com_dx = opti.value(x_opti_com.evaluate_dx(t_vals))
	plt.plot(t_vals, com_dx[k_dim, :], label='true')



def __plt_maximize_window():
	figManager = plt.get_current_fig_manager()
	if plt.get_backend() == 'QtAgg':
		figManager.window.showMaximized()
	if plt.get_backend() == 'TkAgg':
		figManager.resize(*figManager.window.maxsize())