from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from biped_params import *
from centroidal_walk.collocation.opti.collocation_opti_data import ParamsToOptimize
from centroidal_walk.collocation.opti.initial_gait import INIT_GAIT_TYPE



# run
opti = CentroidalDynPolyOpti(
    mass=biped_example_mass,
    InertiaMatrix=biped_example_InertiaMatrix,
    use_two_feet=True,
    foot_force_max_z=1000,#150,
    foot_kin_constraint_box_center_rel=biped_example_foot_kin_constraint_box_center_rel,
    foot_kin_constraint_box_size=biped_example_foot_kin_constraint_box_size,  # test
    total_duration=biped_example_total_duration,#3,
    num_phases=biped_example_num_steps,
    fixed_phase_durations=None,#hopper_example_initial_phase_durations
    base_poly_duration=0.1,#0.05,
    use_angular_dynamics=True,
    # additional variables and constraints:
    foot_force_at_trajectory_end_and_start_variable=True,
    max_com_angle_xy_abs=0.2,  # can help with convergence
    #additional_intermediate_foot_force_constraints=True,  # just works well in combination with max_com_angle_xy_abs
    additional_foot_flight_smooth_constraints=True,
    params_to_optimize=ParamsToOptimize(
        param_opti_start_foot_pos=True,
    )
)

# when goal is close enough to start use
# no additional constraints, no max_com_angle_xy_abs
# init_gait_type=INIT_GAIT_TYPE.ALL_ZERO_NO_MOVEMENT

# test additional constraints
# opti.add_additional_constraints__allways_at_least_one_foot_ground_contact(
#     type=CentroidalDynPolyOpti.AllwaysAtLeastOneFootGroundContactConstraintType.EVERY_DT,
#     add_every_dt=0.1,
#     #n_constraints_per_phase=16
# )

# WORKS WELL:
#opti.add_additional_constraint__com_lin_z_range_of_motion(max_z_deviation_from_init=0.05)
#opti.add_additional_constraint__com_angular_acc()
opti.add_additional_cost_or_constraint__com_linear_acc(
    constraint_max_acc=np.array([10, 10, 3])
)

opti.bake_solver(just_show_init_values=False, max_iter=200)




##################
# solve and viz
def solve_opti(com_end_pos):
	print('\n\n#########################')
	print('> com_end_pos: ', com_end_pos)

	opti.solve_opti(
		start_com_pos=biped_example_start_com_pos,
		start_feet_pos=biped_example_start_feet_pos,
		end_com_pos=com_end_pos,
		init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT
	)
	print('>> solution valid: ', opti.solution_is_feasible)

	opti.viz3d.com_target_marker_fixed = False
	opti.plot_animate_all(show_plots=False,
												show_plots_angular=False,
												show_plots_angular_dyn=True,
												playback_once=False)


def on_com_target_pos_changed(pos_2d):
	com_end_pos = np.zeros(3)
	com_end_pos[2] = biped_example_end_com_pos[2]
	com_end_pos[:2] = pos_2d
	solve_opti(com_end_pos)


opti.setup_vis3d(on_com_target_pos_changed)

# run fist
on_com_target_pos_changed(pos_2d=np.array([0.0, 0.0]))