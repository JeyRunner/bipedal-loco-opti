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


# solve with params
opti.solve_opti(
    just_show_init_values=False,
    start_com_pos=biped_example_start_com_pos,
    start_feet_pos=biped_example_start_feet_pos,
    end_com_pos=biped_example_end_com_pos,
    #np.array([
    #    1, 1, 0.4#2, 1, 0.4
        #0.5, 0.3, 0.4  # test
        #0.0, 0.2, 0.4  # works
    #]),
    max_iter=500,
    #init_gait_type=INIT_GAIT_TYPE.ALL_ZERO_NO_MOVEMENT  # works also: INIT_GAIT_TYPE.ALL_FEET_JUMP
    init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT # INIT_GAIT_TYPE.ALL_FEET_JUMP
)
# works good: INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES

opti.plot_animate_all()