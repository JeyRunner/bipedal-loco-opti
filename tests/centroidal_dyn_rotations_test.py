import unittest

import numpy as np

from centroidal_walk.centroidal_dyn_rotations import *
from tests.test_util import to_np


class TestEulerAngleConversions(unittest.TestCase):


    def test_R_world_frame_to_com_frame_euler(self):
        angles = np.array([1, 2.5, 0.5])
        R = to_np(R_world_frame_to_com_frame_euler(angles))

        same = np.abs(np.linalg.inv(R) - R.T) < 0.0000001
        print("R ", R)
        print()
        print("R^1 ", np.linalg.inv(R))
        print("R.T ", R.T)
        print("same? ", same)
        self.assertTrue(np.all(same))


    def test_E_angular_vel_to_euler_rates(self):
        step_size = 0.3
        for angle_x in [0]:
            for angle_y in np.arange(0, np.pi, step=step_size):
                for angle_z in np.arange(0, np.pi, step=step_size):
                    angles = np.array([angle_x, angle_y, angle_z])
                    E = to_np(E_euler_rates_to_angular_vel(angles))
                    #E_inv_ref = solve(E, SX.eye(3))
                    E_inv_ref = np.linalg.inv(E)
                    E_inv = to_np(E_angular_vel_to_euler_rates(angles))
                    #print(E.print_dense())
                    #print(E_inv)
                    print(E_inv)
                    print(E_inv_ref)
                    same = np.abs(E_inv - E_inv_ref) < 0.0000001
                    print('same? \n', same)
                    self.assertTrue(same.all())
                    print('\n')

if __name__ == '__main__':
    unittest.main()