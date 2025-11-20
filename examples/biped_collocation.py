from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from biped_params import *
from centroidal_walk.collocation.opti.initial_gait import INIT_GAIT_TYPE



# run
opti = CentroidalDynPolyOpti(
    mass=biped_example_mass,
    InertiaMatrix=biped_example_InertiaMatrix,
    use_two_feet=True,
    foot_force_max_z=1000,
    foot_kin_constraint_box_center_rel=biped_example_foot_kin_constraint_box_center_rel,
    foot_kin_constraint_box_size=biped_example_foot_kin_constraint_box_size,  # test
    total_duration=biped_example_total_duration,#3,
    num_phases=biped_example_num_steps + 1,
    looping_last_contact_phase=True,
    feet_first_phase_type=[PhaseType.CONTACT, PhaseType.FLIGHT],
    fixed_phase_durations=None,#hopper_example_initial_phase_durations
    base_poly_duration=0.1,#0.05,
    use_angular_dynamics=True,
    # additional variables and constraints:
    foot_force_at_trajectory_end_and_start_variable=False,
    #max_com_angle_xy_abs=0.2,  # can help with convergence
    #additional_intermediate_foot_force_constraints=False,  # just works well in combination with max_com_angle_xy_abs
    #additional_foot_flight_smooth_constraints=True
)


# solve with params
opti.solve_opti(
    just_show_init_values=False,
    start_com_pos=biped_example_start_com_pos,
    start_feet_pos=biped_example_start_feet_pos,
    end_com_pos=biped_example_end_com_pos,
    max_iter=400,
    init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT  # works also: INIT_GAIT_TYPE.ALL_FEET_JUMP
    #init_gait_type=INIT_GAIT_TYPE.ALL_FEET_JUMP
)
# works good: INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES

opti.plot_animate_all()