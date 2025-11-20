from centroidal_walk.centroidal_dyn_rotations import *
from centroidal_walk.centroidal_dyn_util import *


# Two Foot Humanoid with centroidal dynamics
# in 3d
#
# Uses SX graph (optimized for speed)
#
# coord sys
#      z^
#       |
#       |    / x
#       |   /
#       |  /
#       | /
#       |/
#       +-------------> y
# depth is in x dir
# order in arrays: x,y,z
class CentroidalDynPoly:

    d: int = 3
    num_feet: int

    # robot properties
    mass: int
    InertiaMatrix: np.array
    use_two_feet: bool

    gravity_acc = 9.80665




    def __init__(self, mass: int, InertiaMatrix: np.array, num_feet: int):
        self.num_feet = num_feet
        self.use_two_feet = num_feet == 2
        self.mass = mass
        self.InertiaMatrix = InertiaMatrix



    def get_dynamics_com_ddpos(self, foot_force):
        return foot_force/self.mass - np.array([0, 0, self.gravity_acc])


    def get_dynamics_com_ddpos_violation(self, com_ddpos: MX, feet_forces: list[MX]):
        sum_foot_forces = MX.zeros(self.d)
        for i in range(self.num_feet):
            sum_foot_forces += feet_forces[i]
        return com_ddpos - self.get_dynamics_com_ddpos(sum_foot_forces)


    def get_dynamics_com_angular_acc(self,
                                     com_pos: MX,
                                     com_angle: MX,
                                     com_dangle: MX,
                                     feet_positions: list[MX],
                                     feet_forces: list[MX]
                                     ):
        """
        This returns the angular acc of the com, NOT the euler change rates acc!
        To use this the com euler acc has to be transformed to euler change acc.
        """
        assert len(feet_forces) == len(feet_positions) == self.num_feet
        assert False, "This function is deprecated and not implemented correctly"

        # angular com value in the global frame
        angular_vel_glob = E_euler_rates_to_angular_vel(com_angle) @ com_dangle

        com_torques_from_feet = MX.zeros(3)
        for foot_i in range(self.num_feet):
            com_to_foot_vec = com_pos - feet_positions[foot_i] # + 1e-10  # fix NaN error
            com_torques_from_feet = (
                    com_torques_from_feet
                    + casadi.cross(feet_forces[foot_i], com_to_foot_vec)
            )

        # transform inertia matrix (expressed in com frame) into global frame (according to body rotation)
        # (R_com_to_world @ I_com @ R_world_to_com)   @   (vec_in_world)
        #R_world_frame_to_com_frame = R_world_frame_to_com_frame_euler(com_angle)
        #InertiaMatrix_global = R_world_frame_to_com_frame.T @ self.InertiaMatrix @ R_world_frame_to_com_frame
        InertiaMatrix_global = np.zeros((3, 3))

        angular_acc_on_com = np.linalg.inv(InertiaMatrix_global) @ (
                com_torques_from_feet
                - casadi.cross(angular_vel_glob, InertiaMatrix_global @ angular_vel_glob)
        )
        return angular_acc_on_com



    def euler_acc_to_angular_acc(self, com_angle, com_dangle, com_ddangle):
        """
        Transforms the given euler acceleration rates (com_ddangle) to angular velocity.
        :return: angular acceleration in global frame
        """
        dangle_y = com_dangle[1]
        dangle_z = com_dangle[2]
        E_euler_rates_to_angular_vel__dt = (
                  E_euler_rates_to_angular_vel__dangleY(com_angle) * dangle_y
                + E_euler_rates_to_angular_vel__dangleZ(com_angle) * dangle_z
        )
        angular_acc = (
                E_euler_rates_to_angular_vel__dt @ com_dangle
              + E_euler_rates_to_angular_vel(com_angle) @ com_ddangle
        )
        return angular_acc



    def get_dynamics_com_angular_acc_violation(self,
                                               com_pos: MX,
                                               com_angle: MX,
                                               com_dangle: MX,
                                               com_ddangle: MX,
                                               feet_positions: list[MX],
                                               feet_forces: list[MX]
                                               ):
        """
        Constraint the resulting violation value to == 0.
        Which means constraining the returned values like this:  angular_dyn_com_torques == angular_dyn_torques_from_feet.
        Get the angular dynamics values for given values.
        :return: (angular_dyn_com_torques, angular_dyn_torques_from_feet) these two values should be equal
        """
        assert len(feet_forces) == len(feet_positions) == self.num_feet

        # angular com value in the global frame
        angular_vel_glob = E_euler_rates_to_angular_vel(com_angle) @ com_dangle
        angular_acc_glob = self.euler_acc_to_angular_acc(com_angle, com_dangle, com_ddangle)

        # transform inertia matrix (expressed in com frame) into global frame (according to body rotation)
        # (R_com_to_world @ I_com @ R_world_to_com)   @   (vec_in_world)
        R_world_frame_to_com_frame = R_world_frame_to_com_frame_euler(com_angle)
        #print('R_world_frame_to_com_frame', R_world_frame_to_com_frame)
        InertiaMatrix_global =  R_world_frame_to_com_frame.T @ self.InertiaMatrix @ R_world_frame_to_com_frame

        # the toque value that must act on the com to create the current angular acc
        angular_dyn_com_torques = (
            InertiaMatrix_global @ angular_acc_glob
            + casadi.cross(angular_vel_glob, InertiaMatrix_global @ angular_vel_glob)
        )

        # the actual toque that acts on the com coming from the feet forces
        angular_dyn_torques_from_feet = MX.zeros(3)
        for foot_i in range(self.num_feet):
            com_to_foot_vec = com_pos - feet_positions[foot_i]  # + 1e-10  # fix NaN error
            angular_dyn_torques_from_feet = (
                    angular_dyn_torques_from_feet
                    + casadi.cross(feet_forces[foot_i], com_to_foot_vec)
            )

        return angular_dyn_com_torques, angular_dyn_torques_from_feet

