import numpy as np



hopper_example_total_duration = 3#4.4
hopper_example_num_steps = 9
hopper_example_initial_phase_durations = [
    np.ones(hopper_example_num_steps) * (hopper_example_total_duration / hopper_example_num_steps)
]

hopper_example_mass = 15
hopper_example_InertiaMatrix = np.eye(3)
# hopper_example_InertiaMatrix = np.array([  #np.eye(3)
#     [1.2,   0,      0.2],
#     [0,     5.5,    0.01],
#     [0.2,   0.01,   6],
# ])

# params
hopper_example_foot_kin_constraint_box_center_rel = [
    np.array([0, 0, -0.4]),    # one foot
]
# hopper_example_foot_kin_constraint_box_size = np.array([0.05, 0.15, 0.1]) #also works
hopper_example_foot_kin_constraint_box_size = np.array([0.1, 0.15, 0.1])
