import numpy as np

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from centroidal_walk.collocation.opti.initial_gait import INIT_GAIT_TYPE

# just reusing some of the parameters of the bipad here
from examples.biped_params import *


quadruped_example_InertiaMatrix = np.array([
    [0.946438,   0.000938112,      0.00595386],
    [0.000938112,     1.94478,    0.00146328],
    [0.00595386,   0.00146328,   2.01835],
])

# parameters
quadruped_example_foot_x_offset = 0.1
quadruped_example_foot_y_offset = 0.15
biped_example_foot_z_offset = 0.4
quadruped_example_foot_kin_constraint_box_center_rel = [
    np.array([biped_example_foot_x_offset, quadruped_example_foot_y_offset, -biped_example_foot_z_offset]),
    np.array([-biped_example_foot_x_offset, quadruped_example_foot_y_offset, -biped_example_foot_z_offset]),
    np.array([biped_example_foot_x_offset, -quadruped_example_foot_y_offset, -biped_example_foot_z_offset]),
    np.array([-biped_example_foot_x_offset, -quadruped_example_foot_y_offset, -biped_example_foot_z_offset]),
]
#biped_example_foot_kin_constraint_box_size = np.array([0.1, 0.15, 0.1]) #also works
quadruped_example_foot_kin_constraint_box_size = np.array([0.075, 0.15, 0.1])


quadruped_example_start_com_pos=np.array([
        0, 0.0, biped_example_foot_z_offset
    ])
quadruped_example_start_feet_pos=np.array([
    [biped_example_foot_x_offset, quadruped_example_foot_y_offset, 0],
    [-biped_example_foot_x_offset, quadruped_example_foot_y_offset, 0],
    [biped_example_foot_x_offset, -quadruped_example_foot_y_offset, 0],
    [-biped_example_foot_x_offset, -quadruped_example_foot_y_offset, 0],
])

quadruped_example_end_com_pos=np.array([
    0.1, 0.3, 0.4  # middle params
])


fixed_phase_durations = [
    np.array([0.8, 0.2, 0.8, 0.2, 0.8]),
    np.array([0.2, 0.8, 0.2, 0.8, 0.2]),
    np.array([0.2, 0.8, 0.2, 0.8, 0.2]),
    np.array([0.8, 0.2, 0.8, 0.2, 0.8]),
]
for i in range(len(fixed_phase_durations)):
    fixed_phase_durations[i] = (fixed_phase_durations[i]/np.sum(fixed_phase_durations[i]))*biped_example_total_duration

# custom timing init
num_phases_feet = 3
num_phases = num_phases_feet*2-1
init_flight_phase_durations = [
    np.ones((num_phases_feet-1,))*0.2,
    np.ones((num_phases_feet,))*0.2,
    np.ones((num_phases_feet,))*0.2,
    np.ones((num_phases_feet-1,))*0.2,
]
init_contact_phase_durations = [
    np.ones((num_phases_feet,))*0.8,
    np.ones((num_phases_feet-1,))*0.8,
    np.ones((num_phases_feet-1,))*0.8,
    np.ones((num_phases_feet,))*0.8,
]

# create opti
opti = CentroidalDynPolyOpti(
    mass=6,
    InertiaMatrix=quadruped_example_InertiaMatrix,
    num_feet=4,
    feet_first_phase_type=[PhaseType.CONTACT, PhaseType.FLIGHT, PhaseType.FLIGHT, PhaseType.CONTACT],
    foot_force_max_z=1000,
    foot_kin_constraint_box_center_rel=quadruped_example_foot_kin_constraint_box_center_rel,
    foot_kin_constraint_box_size=quadruped_example_foot_kin_constraint_box_size,  # test
    total_duration=biped_example_total_duration,#3,
    num_phases=num_phases,
    #fixed_phase_durations=fixed_phase_durations,

    phase_duration_min=0.0,
    #contact_phase_duration_min=0.1,

    base_poly_duration=0.05, # 0.05
    use_angular_dynamics=True,
    # additional variables and constraints:
    foot_force_at_trajectory_end_and_start_variable=True,
    max_com_angle_xy_abs=0.002,  # can help with convergence
    additional_intermediate_foot_force_constraints=False,  # just works well in combination with max_com_angle_xy_abs
    additional_foot_flight_smooth_constraints=True
)

opti.add_additional_cost_or_constraint__com_linear_acc(
    constraint_max_acc=np.array([10, 10, 3])
)
opti.add_additional_constraint__com_lin_z_range_of_motion(max_z_deviation_from_init=0.05)



# baking takes some time
opti.bake_solver(
    max_iter=400,
    just_show_init_values=False,
)


# solve with params
opti.solve_opti(
    start_com_pos=quadruped_example_start_com_pos,
    start_feet_pos=quadruped_example_start_feet_pos,
    end_com_pos=quadruped_example_end_com_pos,
    init_gait_type=INIT_GAIT_TYPE.ALL_FEET_JUMP,

	# or to fix gait timings:
    #init_gait_type=INIT_GAIT_TYPE.TIMINGS_GIVEN,
    #given_flight_phase_durations=init_flight_phase_durations,
    #given_contact_phase_durations=init_contact_phase_durations,
)
# works good: INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES

opti.plot_animate_all(show_plots=True)