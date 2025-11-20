from casadi import *
import numpy as np
import matplotlib.pyplot as plt
from centroidal_walk.visualization import simulate_and_plot



def R_world_frame_to_com_frame_euler(com_angles):
    """
    Get transformation matrix to rotate from world to com frame (via euler zyx angles).
    Does ${\bf z'}=R_{i j k}({\bf u})\;{\bf z}$,
    where $z$ is the vector in world frame and $z'$ is the vector in com frame.
    :param com_angles: com angels
    :return: Transform matrix
    """
    # rotations: here we use Z-Y-X order Euler angles
    #   φ = around x-axis
    #   θ = around y-axis
    #   ψ = around z-axis
    sin_x = casadi.sin(com_angles[0])  # φ
    cos_x = casadi.cos(com_angles[0])  # φ

    sin_y = casadi.sin(com_angles[1])  # θ
    cos_y = casadi.cos(com_angles[1])  # θ

    sin_z = casadi.sin(com_angles[2])  # ψ
    cos_z = casadi.cos(com_angles[2])  # ψ

    R = vertcat(
        horzcat(cos_y*cos_z,                        cos_y*sin_z,                            -sin_y),
        horzcat(sin_x*sin_y*cos_z - cos_x*sin_z,    sin_x*sin_y*sin_z + cos_x*cos_z,        cos_y*sin_x),
        horzcat(cos_x*sin_y*cos_z + sin_x*sin_z,    cos_x*sin_y*sin_z - sin_x*cos_z,        cos_y*cos_x),
    )
    return R



def E_angular_vel_to_euler_rates(com_angles, testing=False):
    """
    Get Matrix to map (global) angular velocity to (global) euler angle (change) rates.
    :param com_angles: com angels
    :return: Transform matrix
    """
    # rotations: here we use Z-Y-X order Euler angles
    #   φ = around x-axis
    #   θ = around y-axis
    #   ψ = around z-axis
    sin_y = casadi.sin(com_angles[1])
    cos_y = casadi.cos(com_angles[1])
    sin_z = casadi.sin(com_angles[2])
    cos_z = casadi.cos(com_angles[2])
    # Matrix to map (global) angular velocity to (global) euler angle (change) rates
    #if testing:
    #    E = (1/cos_y) * SX([
    #        [cos_z,         sin_z,          0],
    #        [-cos_y*sin_z,  cos_y*cos_z,    0],
    #        [cos_z*sin_y,   sin_z*sin_y,    cos_y]
    #    ])
    #    return E
    E = (1/cos_y) * vertcat(
        horzcat(cos_z,         sin_z,          0),
        horzcat(-cos_y*sin_z,  cos_y*cos_z,    0),
        horzcat(cos_z*sin_y,   sin_z*sin_y,    cos_y)
    )
    print('E', E)
    return E


def E_euler_rates_to_angular_vel(com_angles, testing=False, as_jax=False):
    """
    Get Matrix to map (global) euler angle (change) rates to (global) angular velocity.
    :param com_angles: com angels
    :return: Transform matrix
    """
    # rotations: here we use Z-Y-X order Euler angles
    #   φ = around x-axis
    #   θ = around y-axis
    #   ψ = around z-axis
    if not as_jax:
        sin_y = casadi.sin(com_angles[1])
        cos_y = casadi.cos(com_angles[1])
        sin_z = casadi.sin(com_angles[2])
        cos_z = casadi.cos(com_angles[2])
        if testing:
            E = SX([
                [cos_y*cos_z,   -sin_z,     0],
                [cos_y*sin_z,   cos_z,      0],
                [-sin_y,        0,          1]
            ])
            return E
        E = vertcat(
            horzcat(cos_y*cos_z,   -sin_z,     0),
            horzcat(cos_y*sin_z,   cos_z,      0),
            horzcat(-sin_y,        0,          1),
        )
    else:
        # as jax
        import jax.numpy as jnp
        sin_y = jnp.sin(com_angles[1])
        cos_y = jnp.cos(com_angles[1])
        sin_z = jnp.sin(com_angles[2])
        cos_z = jnp.cos(com_angles[2])
        E = jnp.array([
            [ cos_y * cos_z, -sin_z, 0 ],
            [ cos_y * sin_z, cos_z, 0 ],
            [ -sin_y, 0, 1 ],
        ])
    return E




def E_euler_rates_to_angular_vel__dangleY(com_angles):
    """
    Get derivative by the y-angle of the Matrix which maps (global) euler angle (change) rates to (global) angular velocity.
    See $$\frac{\partial E}{\partial \theta}$$ in "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation" p. 24.

    :param com_angles: com angels
    :return: Transform matrix
    """
    # rotations: here we use Z-Y-X order Euler angles
    #   φ = around x-axis
    #   θ = around y-axis
    #   ψ = around z-axis
    sin_y = casadi.sin(com_angles[1])
    cos_y = casadi.cos(com_angles[1])
    sin_z = casadi.sin(com_angles[2])
    cos_z = casadi.cos(com_angles[2])
    E_dY = vertcat(
        horzcat(-cos_z*sin_y,      0,     0),
        horzcat(-sin_z*sin_y,      0,     0),
        horzcat(-cos_y,            0,     0),
    )
    return E_dY


def E_euler_rates_to_angular_vel__dangleZ(com_angles):
    """
    Get derivative by the z-angle of the Matrix which maps (global) euler angle (change) rates to (global) angular velocity.
    See $$\frac{\partial E}{\partial \psi}$$ in "Representing Attitude: Euler Angles, Unit Quaternions, and Rotation" p. 24.

    :param com_angles: com angels
    :return: Transform matrix
    """
    # rotations: here we use Z-Y-X order Euler angles
    #   φ = around x-axis
    #   θ = around y-axis
    #   ψ = around z-axis
    sin_y = casadi.sin(com_angles[1])
    cos_y = casadi.cos(com_angles[1])
    sin_z = casadi.sin(com_angles[2])
    cos_z = casadi.cos(com_angles[2])
    E_dY = vertcat(
        horzcat(-cos_y*sin_z,      -cos_z,     0),
        horzcat(cos_y*cos_z,        -sin_z,    0),
        horzcat(0,                  0,         0),
    )
    return E_dY