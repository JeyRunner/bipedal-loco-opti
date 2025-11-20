from centroidal_walk.collocation.opti import initial_gait
from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from centroidal_walk.collocation.opti.initial_gait import INIT_GAIT_TYPE
from hopper_params import *


InertiaMatrixRealistic = np.array([
    [1.209, 0.2, 0],
    [0.2, 5.58, 0],
    [0, 0, 6],
])

InertiaMatrixCube = np.array([
    # example of a cube
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])



# run
opti = CentroidalDynPolyOpti(
    mass=hopper_example_mass,
    InertiaMatrix=hopper_example_InertiaMatrix,
    use_two_feet=False,
    foot_force_max_z=1000,
    foot_kin_constraint_box_center_rel=hopper_example_foot_kin_constraint_box_center_rel,
    foot_kin_constraint_box_size=hopper_example_foot_kin_constraint_box_size,#np.array([0.1, 0.15, 0.1]),  # test
    total_duration=hopper_example_total_duration,#3,
    num_phases=hopper_example_num_steps,
    # problem formulation parameters
    fixed_phase_durations=None,#hopper_example_initial_phase_durations,
    base_poly_duration=0.1,
    use_angular_dynamics=True,
    # additional variables and constraints:
    foot_force_at_trajectory_end_and_start_variable=True,
    # max_com_angle_xy_abs=0.1,  # can help with convergence
    # additional_intermediate_foot_force_constraints=False
    additional_foot_flight_smooth_constraints=True
)




# solve with params
opti.solve_opti(
    just_show_init_values=False,
    start_com_pos=np.array([
        0, 0.0, 0.4 # test
    ]),
    start_feet_pos=np.array([
        [0, 0, 0],  # foot 1
    ]),
    end_com_pos=np.array([
        #2, 1, 0.4
        1, 1, 0.4
        #0.5, 0.3, 0.4  # test
        #0.0, 0.2, 0.4  # works
    ]),
    max_iter=500,
    init_gait_type=INIT_GAIT_TYPE.ALL_ZERO_NO_MOVEMENT
)

opti.plot_animate_all()