from scipy.spatial.transform import Rotation as R
import numpy as np


def get_range_for_dimensional_index(index, dimensions_per_index=3, second_dim=False):
    """
    Get index range for the ith sub index, where each index uses dimensions_per_index elements
    :param index:
    :param dimensions_per_index:
    :return:
    """
    if second_dim:
        return np.s_[index*dimensions_per_index : (index+1)*dimensions_per_index, :]
    else:
        return np.s_[index*dimensions_per_index : (index+1)*dimensions_per_index]


def get_foot_pos_global(pos_pos_rel_to_body, com_pos, com_rotation):
    #rot_matrix = R.from_euler('zyx', com_rotation, degrees=False).as_matrix()
    # use zxy because visualization lib uses that
    rot_matrix = R.from_euler('zxy', com_rotation, degrees=False).as_matrix()
    rotated = rot_matrix @ pos_pos_rel_to_body
    return rotated + com_pos



def rad_to_deg(rad):
    return (rad / np.pi) * 180



def create_contact_schedule(num_steps, time_steps, use_two_feet=False):
    contact_steps__total_changes = num_steps + num_steps - 1
    time_steps_per_contact_steps = round(time_steps / (num_steps + num_steps - 1))

    num_feet = 2 if use_two_feet else 1
    foot_in_contact = np.zeros(shape=(num_feet, time_steps))
    for i in range(time_steps):
        first_phase = int(i / time_steps_per_contact_steps) == 0
        foot1_in_contact = int(i / time_steps_per_contact_steps) % 2 == 0 or int(i / time_steps_per_contact_steps) >= contact_steps__total_changes
        foot_in_contact[0, i] = foot1_in_contact
        if use_two_feet:
            if first_phase:
                foot2_in_contact = True
            else:
                foot2_in_contact = not foot1_in_contact
            foot_in_contact[1, i] = foot2_in_contact
    return foot_in_contact
