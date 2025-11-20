import numpy as np



biped_example_total_duration = 3/2#4.4
biped_example_num_steps = 5+0
biped_example_initial_phase_durations = [
    np.ones(biped_example_num_steps) * (biped_example_total_duration / biped_example_num_steps),
    np.ones(biped_example_num_steps) * (biped_example_total_duration / biped_example_num_steps)
]

#biped_example_mass = 1.254 #10 #5 # 10
biped_example_mass = 10 #5 # 10
#biped_example_InertiaMatrix = np.eye(3)
biped_example_InertiaMatrix = np.array([
    [1.2,   0,      0.2],
    [0,     5.5,    0.01],
    [0.2,   0.01,   6],
])

# for bolt
biped_example_mass = 1.25407789
# the intertia matrix is rotated around z -> not correct
biped_example_InertiaMatrix = np.array(
    [[1.90789887e-02, -3.19631689e-07, 2.68055723e-03],
     [-3.19631689e-07, 2.94731563e-02, -2.58141431e-07],
     [2.68055723e-03, -2.58141431e-07, 3.11954631e-02]]
)


# params
# biped_example_foot_x_offset = 0.1235
# biped_example_foot_z_offset = 0.3386
biped_example_foot_x_offset = 0.2
biped_example_foot_y_offset = 0.02319275
biped_example_foot_z_offset = 0.4

# for bolt
biped_example_foot_x_offset = 0.123499
biped_example_foot_y_offset = -0.02319275
biped_example_foot_z_offset = 0.309

biped_example_foot_kin_constraint_box_center_rel = [
    np.array([biped_example_foot_x_offset, biped_example_foot_y_offset, -biped_example_foot_z_offset]),
    np.array([-biped_example_foot_x_offset, biped_example_foot_y_offset, -biped_example_foot_z_offset]),
]
#biped_example_foot_kin_constraint_box_size = np.array([0.1, 0.15, 0.1]) #also works
biped_example_foot_kin_constraint_box_size = np.array([0.05, 0.1, 0.05])

# for bolt
biped_example_foot_kin_constraint_box_size = np.array([0.07, 0.125, 0.04])


biped_example_start_com_pos = np.array([
        0, 0.0, biped_example_foot_z_offset
    ])
biped_example_start_feet_pos = np.array([
    [biped_example_foot_x_offset, biped_example_foot_y_offset, 0],  # foot 1
    [-biped_example_foot_x_offset, biped_example_foot_y_offset, 0],  # foot 1
])
biped_example_start_feet_pos[:, 0:2] += biped_example_start_com_pos[:2]

biped_example_end_com_pos = np.array([
        #0.2, 0.2, 0
        #0, 0, 0.4

        #-0.05, 0.05, 0  # easy params
        #0.2, 0.5, 0
        #0.1, 0.15, 0
        #0.1, 0.15, 0
        #0.15, 0.0, 0
        #0, 0.0, 0

        #-0.2, 0.8, 0.4  # harder params
        -0.2, 0.4, 0.4  # middle params
], dtype=np.float64)

biped_example_end_com_pos[2] = biped_example_foot_z_offset