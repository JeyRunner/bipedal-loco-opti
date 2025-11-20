from dataclasses import dataclass

import casadi

from centroidal_walk.collocation.serialization.yaml_util import SerializableOptiVariables, serializable_opti_variables


@serializable_opti_variables([
    'constraint_points_t__feet_pos',
    'constraint_points_t__feet_contact_force'
])
@dataclass
class FootConstraintTimePoints(SerializableOptiVariables):
    constraint_points_t__feet_pos: casadi.MX
    constraint_points_t__feet_contact_force: casadi.MX

    def __init__(self, opti):
        SerializableOptiVariables.__init__(self, opti)
        self.constraint_points_t__feet_contact_force = casadi.MX.zeros(1)
        self.constraint_points_t__feet_pos = casadi.MX.zeros(1)

    def append_timepoint_to_feet_pos_constraints(self, t):
        self.constraint_points_t__feet_pos = casadi.horzcat(self.constraint_points_t__feet_pos, t)

    def append_timepoint_to_feet_force_constraints(self, t):
        self.constraint_points_t__feet_contact_force = casadi.horzcat(self.constraint_points_t__feet_contact_force, t)


