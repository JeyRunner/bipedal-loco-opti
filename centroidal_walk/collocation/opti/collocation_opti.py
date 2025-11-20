from typing import List

import casadi
import numpy as np
from numpy import ndarray

from centroidal_walk.casadi_util_functions import scalar_prod
from centroidal_walk.collocation.opti import initial_gait
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import Phase, ContactPhase, FlightPhase
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.opti.collocation_opti_data import CentroidalDynPolyOptiData, ParamsToOptimize
from centroidal_walk.collocation.plotting import plot_com_and_foot_trajectories_xyz
from centroidal_walk.collocation.plotting_data import FootConstraintTimePoints
from centroidal_walk.collocation.spline.spline_trajectory import *
from centroidal_walk.visualization import simulate_and_plot
from centroidal_walk.visualization.RobotAnimation3D import RobotAnimation3D
from centroidal_walk.collocation.serialization.OptiDecorators import *


class CentroidalDynPolyOpti(CentroidalDynPolyOptiData):
    """
    Optimize over walking trajectories for walking robots.
    Allows to fully optimize gait pattern and phase timings.
    """

    # number of dimensions
    d = 3


    # opti problem formulation parameters
    num_phases: int

    foot_kin_constraint_box_size: ndarray
    foot_kin_constraint_box_center_rel: list[ndarray]

    use_angular_dynamics: bool
    fixed_phase_durations: list[ndarray] | None
    phase_duration_min: float

    foot_force_at_trajectory_end_and_start_variable: bool
    foot_force_max_z: float
    max_com_angle_xy_abs: float | None

    total_duration: float
    base_poly_duration: float
    num_polynomials_for_com_trajectory: int



    # opti problem
    opti: OptiWithSimpleBounds #casadi.Opti

    # opti parameters: start state
    param_opti_start_com_pos: casadi.Opti.parameter
    param_opti_start_com_dpos: casadi.Opti.parameter
    param_opti_start_com_angle: casadi.Opti.parameter
    param_opti_start_com_dangle: casadi.Opti.parameter
    param_opti_start_foot_pos: casadi.Opti.parameter    # 2d: first index is foot number
    param_opti_start_foot_dpos: casadi.Opti.parameter   # 2d: first index is foot number
    param_opti_end_com_pos: casadi.Opti.parameter

    # opti variables
    x_opti_com_pos: SplineTrajectory  # list of polynomials that represent the com trajectory
    x_opti_com_angle: SplineTrajectory  # list of polynomials that represent the com trajectory
    x_opti_feet: list[FootTrajectory]  # trajectories for all feet
    x_opti_feet__start_pos: list[Opti.variable] # variables for the initial feet positions


    # time points where dynamics are enforced by constraints
    constraint_points_t__dynamics: np.array
    # time points where foot constraints are enforced
    constraint_points_t__feet: list[FootConstraintTimePoints]


    # additional cost and constraints
    additional_foot_flight_smooth_vel_constraints: bool
    additional_intermediate_foot_force_constraints: bool
    # other additional cost and constraints
    additional_costs_and_constraints_parameters: dict




    def __init__(self,
                 mass: int,
                 InertiaMatrix: np.array,
                 foot_kin_constraint_box_center_rel: list[np.ndarray],
                 foot_kin_constraint_box_size: np.ndarray,
                 foot_force_max_z: float,
                 num_feet: int = None,
                 use_two_feet: bool = True,
                 total_duration=4.4,
                 num_phases=1 + 2 + 2,
                 params_to_optimize=ParamsToOptimize(),
                 looping_last_contact_phase: bool = False,
                 looping_last_contact_phase_time_shift_fixed: float = None,
                 feet_first_phase_type=[PhaseType.CONTACT, PhaseType.CONTACT],
                 base_poly_duration=0.1,
                 num_polynomials_for_com_trajectory=None,
                 phase_duration_min=0.1,
                 contact_phase_duration_min=None,
                 fixed_phase_durations: list[np.ndarray] = None,
                 use_angular_dynamics=True,
                 foot_force_at_trajectory_end_and_start_variable=False,
                 max_com_angle_xy_abs: float = None,
                 additional_intermediate_foot_force_constraints=False,
                 additional_foot_flight_smooth_constraints=False,
                 ):
        """

        :param mass:            mass of the robot in kg
        :param InertiaMatrix:
        :param use_two_feet:    Deprecated, sets num_feet to 2 is num_feet is not set.
        :param num_feet:        The number of feet, this has to match number of elements in foot_kin_constraint_box_center_rel.
                                If set to None (default), use_two_feet will determine this value.
        :param foot_kin_constraint_box_center_rel:  center position of constraint box (relative to com pos)
                                                    in which a foot is allowed to move. One center pos per foot.
        :param foot_kin_constraint_box_size:    size (in each direction from box center) (xyz) of constraint box
                                                in which a foot is allowed to move.
                                                E.g. 1 means the foot can move in +1 and -1 from the box center.
        :param foot_force_max_z:                The forces of the feet in z have to be in the range [0, foot_force_max_z].
        :param total_duration:                  The whole movement has this fixed duration in seconds.
        :param num_phases:                      The number of phases for each foot. A phase is either a contact or flight.
                                                This number has to be odd since the first and last phase are allways contact phases.
                                                E.g. with num_phases=3 you get for each foot: Contact -> Flight -> Contact
        :param base_poly_duration               The number of polynomials for the base trajectory (linar and angular)
                                                is calculated by total_duration/base_poly_duration
                                                if num_polynomials_for_com_trajectory is None.
                                                A lower value leads to the approx error of the dynamics beeing lower
                                                but longer solving times.
        :param num_polynomials_for_com_trajectory:
                                                if this is not none it overwrites the value
                                                calculated with base_poly_duration for the number of base polynomials.
        :param fixed_phase_durations:           When this parameter is None the phase duration are optimized.
                                                Otherwise, this contains arrays for the phase duration for each foot.
        :param max_com_angle_xy_abs:            if set additional constraints are added to ensure that the com angle in xy
                                                stays in range [-max_com_angle_xy_abs, max_com_angle_xy_abs]
        :param additional_intermediate_foot_force_constraints:
                                                if true the force constraints (friction cone, only push on ground)
                                                will also be ensured at one additional point in the middle of each force
                                                polynomial for each contact phase.
                                                Thus, additional num_contact_phase*(force_polies_per_phase-1)
                                                constraints will be added.
        :param additional_foot_flight_smooth_constraints:
                                                if true additional constraint will be added so that the foot xy velocity in
                                                the middle of the flight phase has to match the avg vel. during that phase.
                                                Also, pos in the middle of flight has to be
                                                the middle between previous and next step pos.
                                                Note that this assumes two polynomial for the foot motion during each
                                                flight phase.
        :param params_to_optimize:              set which parameters will not be set as initial conditions
                                                but which will be optimized.
        """
        super().__init__(mass=mass, InertiaMatrix=InertiaMatrix, use_two_feet=use_two_feet, num_feet=num_feet,
                         foot_kin_constraint_box_center_rel=foot_kin_constraint_box_center_rel,
                         foot_kin_constraint_box_size=foot_kin_constraint_box_size, foot_force_max_z=foot_force_max_z,
                         total_duration=total_duration,
                         num_phases=num_phases,
                         params_to_optimize=params_to_optimize,
                         looping_last_contact_phase=looping_last_contact_phase,
                         looping_last_contact_phase_time_shift_fixed=looping_last_contact_phase_time_shift_fixed,
                         feet_first_phase_type=feet_first_phase_type,
                         base_poly_duration=base_poly_duration,
                         num_polynomials_for_com_trajectory=num_polynomials_for_com_trajectory,
                         phase_duration_min=phase_duration_min, contact_phase_duration_min=contact_phase_duration_min,
                         fixed_phase_durations=fixed_phase_durations,
                         use_angular_dynamics=use_angular_dynamics,
                         foot_force_at_trajectory_end_and_start_variable=foot_force_at_trajectory_end_and_start_variable,
                         max_com_angle_xy_abs=max_com_angle_xy_abs,
                         additional_intermediate_foot_force_constraints=additional_intermediate_foot_force_constraints,
                         additional_foot_flight_smooth_vel_constraints=additional_foot_flight_smooth_constraints,
                         as_just_solution_loader=False)
        # state
        self.__baked_solver = False

        # cost term (optional)
        self.cost = MX.zeros(1)

        # make constraints
        self.__create_constraints()

        if looping_last_contact_phase:
            self.__add_constraints_for_looping_gait()




    def __create_constraints(self):
        print()

        # enforce dynamics
        dt_constraints__dynamics = self.base_poly_duration
        self.constraint_points_t__dynamics = np.zeros(0)
        t_phase_start = 0
        time_points_dynamic_constraints = np.arange(0, self.total_duration, step=dt_constraints__dynamics)
        time_points_dynamic_constraints = np.hstack([time_points_dynamic_constraints, self.total_duration])
        for t in time_points_dynamic_constraints:
            self.constraint_points_t__dynamics = np.hstack([self.constraint_points_t__dynamics, t])
            print(f"> enforce dynamics constraint at t = {t} (deltaT total is {self.total_duration})")

            # values at time t
            foot_force = self.evaluate_sum_foot_forces(t)  # this function already handles edge case of t=total_duration
            feet_positions = [foot.evaluate_foot_pos(t) for foot in self.x_opti_feet]
            feet_forces = [foot.evaluate_foot_force(t) for foot in self.x_opti_feet]
            com_pos = self.x_opti_com_pos.evaluate_x(t)
            com_ddpos = self.x_opti_com_pos.evaluate_ddx(t)
            com_angle = self.x_opti_com_angle.evaluate_x(t)
            com_dangle = self.x_opti_com_angle.evaluate_dx(t)
            com_ddangle = self.x_opti_com_angle.evaluate_ddx(t)

            # when at end all splines will result in zero
            # fix this by directly getting the end values
            if t == self.total_duration:
                feet_positions, feet_forces = self.__evaluate_final_feet_pos_and_forces()
                com_pos = self.x_opti_com_pos.poly_list[-1].x1
                com_ddpos = self.x_opti_com_pos.poly_list[-1].evaluate_ddx(self.x_opti_com_pos.poly_list[-1].deltaT)
                com_angle = self.x_opti_com_angle.poly_list[-1].x1
                com_dangle = self.x_opti_com_angle.poly_list[-1].dx1
                com_ddangle = self.x_opti_com_angle.poly_list[-1].evaluate_ddx(self.x_opti_com_angle.poly_list[-1].deltaT)

            # linear com acc
            dynamics_com_ddpos_violation = self.get_dynamics_com_ddpos_violation(
                com_ddpos=com_ddpos,
                feet_forces=feet_forces
            )
            self.opti.subject_to(dynamics_com_ddpos_violation == 0)


            # angular com acc
            dynamics_com_ddangle_as_angular_acc = self.euler_acc_to_angular_acc(
                com_angle=com_angle,
                com_dangle=com_dangle,
                com_ddangle=com_ddangle
            )
            if self.use_angular_dynamics:
                # real angular acc
                # dynamics_com_angular_acc = self.get_dynamics_com_angular_acc(
                #     com_pos=com_pos,
                #     com_dangle=com_dangle,
                #     feet_positions=feet_positions,
                #     feet_forces=feet_forces,
                # )
                # self.opti.subject_to(dynamics_com_ddangle_as_angular_acc == dynamics_com_angular_acc)
                angular_dyn_com_torques, angular_dyn_torques_from_feet = self.get_dynamics_com_angular_acc_violation(
                    com_pos=com_pos,
                    com_angle=com_angle,
                    com_dangle=com_dangle,
                    com_ddangle=com_ddangle,
                    feet_positions=feet_positions,
                    feet_forces=feet_forces
                )
                self.opti.subject_to(angular_dyn_com_torques == angular_dyn_torques_from_feet)

                # constrain com angle
                # @todo currently ignore last time (poly values will be zero here because of casadi.ifelse)
                if self.max_com_angle_xy_abs is not None and not t >= self.total_duration:
                    self.opti.subject_to(self.opti.bounded(-self.max_com_angle_xy_abs, self.x_opti_com_angle.evaluate_x(t)[0], self.max_com_angle_xy_abs))
                    self.opti.subject_to(self.opti.bounded(-self.max_com_angle_xy_abs, self.x_opti_com_angle.evaluate_x(t)[1], self.max_com_angle_xy_abs))
                    print(f"> com angle in xy stays in range at t = {t} (deltaT total is {self.total_duration})")

            else:
                self.opti.subject_to(dynamics_com_ddangle_as_angular_acc == 0)

            # test additional constraint at end
            #if t == self.total_duration:
            #    self.opti.subject_to(com_ddpos == dynamics_com_ddpos)
            #    self.opti.subject_to(dynamics_com_ddangle_as_angular_acc == dynamics_com_angular_acc)
        print('>> total number of dynamics constraint time points: ', len(self.constraint_points_t__dynamics))
        print()


        # constraints for all feet
        self.constraint_points_t__feet = []
        for i, foot in enumerate(self.x_opti_feet):
            print(f'\n>> Constraints for foot  {i}')
            self.__add_foot_constraints(foot, i)

        print()



        # final com pos, ddpos
        # -> already fixed by parameters
        #self.opti.subject_to(self.x_opti_com_pos.poly_list[-1].x1 == self.param_opti_end_com_pos)
        #self.opti.subject_to(self.x_opti_com_pos.poly_list[-1].dx1 == np.zeros(self.d))
        #self.opti.subject_to(self.x_opti_com_pos.evaluate_ddx(self.total_duration-1e-5) == np.zeros(self.d))

        if self.use_angular_dynamics:
            # final com angle
            self.opti.subject_to(self.x_opti_com_angle.poly_list[-1].x1[0:2] == np.zeros(2))
            #self.opti.subject_to_var_bounds(0, self.x_opti_com_angle.poly_list[-1].x1[0:2], 0)
            #self.opti.subject_to(self.x_opti_com_angle.poly_list[-1].x1[2] == 0) # z rotation
            # final com dangle, ddangle
            self.opti.subject_to(self.x_opti_com_angle.poly_list[-1].dx1 == np.zeros(self.d))
            #self.opti.subject_to_var_bounds(0, self.x_opti_com_angle.poly_list[-1].dx1, 0)






    def __add_foot_constraints(self, foot: FootTrajectory, foot_index: int):
        """
        Add all kinematic and force constraints for one foot trajectory (for all its phases).
        """
        # save constraint time point for visualization
        self.constraint_points_t__feet.append(FootConstraintTimePoints(self.opti))
        constraint_points_t__foot = self.constraint_points_t__feet[-1]


        # kinematic foot constraints
        for t in np.arange(0, self.total_duration, step=0.04):  # 0.08): # try also 0.02
            # enforce foot distance to com
            self.__add_foot_phase_pos_distance_constraints(
                foot,
                foot_phase=None,
                t=t,
                foot_kin_constraint_box_center_rel=self.foot_kin_constraint_box_center_rel[foot_index]
            )
            constraint_points_t__foot.append_timepoint_to_feet_pos_constraints(t)
            print(f"> enforce foot kin constraint at t = {t} (t total is {self.total_duration})")
        print()


        # constraints for foot phases
        t = MX.zeros(1)
        for phase_i, phase in enumerate(foot.phases):
            print(f'## Foot phase {phase_i} ({phase.get_phase_type()})')

            if phase.get_phase_type() == PhaseType.CONTACT:
                # foot force constraints
                t_phase = MX.zeros(1)
                for force_poly_i, force_poly in enumerate(phase.foot_force.poly_list):
                    # ignore first poly since here x0 is fixed zero anyway
                    if force_poly_i > 0:
                        # normal force constraints only at poly connection points
                        print(f"> enforce foot force constraint at x0 of poly {force_poly_i} (of 0-{len(phase.foot_force.poly_list)-1} polies) (t={t})")
                        self.__add_foot_force_constraints_poly(force_poly)
                        constraint_points_t__foot.append_timepoint_to_feet_force_constraints(t)
                    # force constraints at more intermediate points of each force poly
                    if self.additional_intermediate_foot_force_constraints:
                        force_sub_t_points = 1
                        for at_i in range(1, force_sub_t_points+1):
                           force_at_t = (force_poly.deltaT/(force_sub_t_points+1))*at_i
                           self.__add_foot_force_constraints(phase, t_phase + force_at_t)
                           print(f"> enforce foot force constraint at t_local={force_at_t} of poly {force_poly_i} (of 0-{len(phase.foot_force.poly_list) - 1} polies)")
                           constraint_points_t__foot.append_timepoint_to_feet_force_constraints(t + force_at_t)
                    t_phase = t_phase + force_poly.deltaT
                    t = t + force_poly.deltaT

            elif phase.get_phase_type() == PhaseType.FLIGHT:
                # foot in z allways on or above ground
                for pos_poly_i, pos_poly in enumerate(phase.foot_position.poly_list):
                    if phase_i == 0 and pos_poly_i == 0:
                        # ignore first poly of first flight phase since here x0 is fixed to the start foot pos
                        continue
                    elif pos_poly_i == 0:
                        # first poly is allways fixed to last contact position
                        continue
                    print(f"> enforce foot above ground constraint at x0 of poly {pos_poly_i} (of 0-{len(phase.foot_position.poly_list)-1} polies)")
                    # comment from towr:
                    # When using interior point solvers such as IPOPT to solve the problem, this
                    # constraint also keeps the foot nodes far from the terrain, causing a leg
                    # lifting during swing-phase. This is convenient.
                    # > pos during flight phase
                    max_distance_above_terrain = 1e20
                    #self.opti.subject_to(max_distance_above_terrain > pos_poly.x0[2] >= 0)
                    self.opti.subject_to(self.opti.bounded(0, pos_poly.x0[2], max_distance_above_terrain))
                t = t + phase.duration


        # additional foot force constraint for start of first phase and end of last phase
        foot_last_phase_is_contact = foot.phases[-1].get_phase_type() == PhaseType.CONTACT
        foot_first_phase_is_contact = foot.phases[0].get_phase_type() == PhaseType.CONTACT
        if self.foot_force_at_trajectory_end_and_start_variable:
            if foot_first_phase_is_contact:
                self.__add_foot_force_constraints_poly(list(foot.get_contact_phases())[0].foot_force.poly_list[0], enforce_at_x0=True)
                constraint_points_t__foot.append_timepoint_to_feet_force_constraints(0)
            if foot_last_phase_is_contact:
                self.__add_foot_force_constraints_poly(list(foot.get_contact_phases())[-1].foot_force.poly_list[-1], enforce_at_x0=False) # at x1
                constraint_points_t__foot.append_timepoint_to_feet_force_constraints(self.total_duration)


        def terrain_height(pos_xy):
            # return pos_xy[1]*0.001
            return 0

        # foot z-pos is zero for contact phases
        for phase_i, phase in enumerate(foot.get_contact_phases()):
            # skip first since the initial foot pos is given and to an opti variable
            # except when first phase is not contact but flight phase
            if phase_i > 0 or not foot_first_phase_is_contact:
                self.opti.subject_to(phase.foot_position[2] == terrain_height(phase.foot_position[0:2]))
                print(f"> enforce foot on ground constraint for contact phase {phase_i} (of 0-{len(list(foot.get_contact_phases())) - 1} contact phases)")

        # foot z-pos is zero at end of last phase when last phase is flight
        if not foot_last_phase_is_contact:
            last_phase: FlightPhase = foot.phases[-1]
            self.opti.subject_to(last_phase.foot_position.poly_list[-1].x1[2] == terrain_height(last_phase.foot_position.poly_list[-1].x1[0:2]))
            print(
                f"> enforce foot on ground constraint for last phase (is flight)")
        if self.params_to_optimize.param_opti_start_foot_pos:
            # fix z for each foot
            self.opti.subject_to(self.param_opti_start_foot_pos[:, 2] == terrain_height(self.param_opti_start_foot_pos[0:2]))

        # optionally constraint max foot vel in flight phase
        # for phase in foot.get_flight_phases():
        #     max_foot_vel = 1#0.7
        #     foot_vel_avg = (phase.foot_position.poly_list[-1].x1 - phase.foot_position.poly_list[0].x0) / phase.duration
        #     self.opti.subject_to(self.opti.bounded(-max_foot_vel, foot_vel_avg, max_foot_vel))


        # constrain foot xy velocity in the middle of the flight phase has to match the avg vel. during that phase.
        # Also, pos in the middle of flight has to be the middle between previous and next step pos.
        # Note that this assumes two polynomial for the foot motion during each flight phase.
        if self.additional_foot_flight_smooth_vel_constraints:
            for phase_i, phase in enumerate(foot.get_flight_phases()):
                assert len(phase.foot_position.poly_list) == 2, "this constraint only works for two polies per flight phase"
                pos_start = phase.foot_position.poly_list[0].x0[0:2] # just xy
                pos_end = phase.foot_position.poly_list[-1].x1[0:2]
                distance = pos_end - pos_start
                avg_vel = distance / phase.duration
                poly_middle_vel = phase.foot_position.poly_list[0].dx1[0:2]
                # @todo ensure not via constraint but directly via parametrization (need less opti vars)
                self.opti.subject_to(avg_vel == poly_middle_vel)

                middle_pos = pos_start + 0.5*distance
                poly_middle_pos = phase.foot_position.poly_list[0].x1[0:2]
                self.opti.subject_to(middle_pos == poly_middle_pos)


        # foot constraints: fix total duration
        # -> handled in FootTrajectory

        # starting values for foot pos
        #print('##', foot.evaluate_foot_pos(0))
        #print('#!', self.x_opti_feet__start_pos[foot_index])
        #self.opti.subject_to(foot.evaluate_foot_pos(0) == self.param_opti_start_foot_pos[foot_index])
        #self.opti.subject_to(foot.phases[0].foot_position == self.param_opti_start_foot_pos[foot_index])

        # @todo this needs to be a bound on the variables (at forwarded as those to ipopt) not a constraint
        # @todo bounds on force z also needs to be ipopt bound! not a constraint
        #self.opti.subject_to(self.x_opti_feet__start_pos[foot_index][:] == self.param_opti_start_foot_pos[foot_index, :].T)



    def __add_foot_force_constraints_poly(self, poly: Polynomial3, enforce_at_x0=True):
        """
        Add constraints for the foot forces for one foot force polynomial of a contact phase.
        This will add constraints:
            - ensure that force in z is positive (>= 0)
            - ensure xy force obey friction pyramid
        :param poly: the polynomial of the force trajectory of a contact phase to constrain
        :param enforce_at_x0: if true the constraint will be enforced at the start (x0) of the poly,
                              otherwise at its end (x1)
        """
        if enforce_at_x0:
            force_z = poly.x0[2]
            force_xy = poly.x0[0:2]
        else:
            force_z = poly.x1[2]
            force_xy = poly.x1[0:2]
        self.__add_foot_force_constraints_generic(force_xy, force_z)
        # force_limit_z = self.foot_force_max_z #500 #200 #200 #90 #250#180 #60 #20
        # # just push
        # self.opti.subject_to(self.opti.bounded(0, force_z, force_limit_z))
        #
        # # friction pyramid
        # # 1 means 45° angle of friction pyramid walls
        # #      ^= xy-forces could be as large as z force
        # friction_mu_coefficient = 0.5 #0.35 #0.2 #0.5 #0.2
        # self.opti.subject_to(
        #     self.opti.bounded(-friction_mu_coefficient*force_z,
        #                       force_xy,
        #                       friction_mu_coefficient*force_z)
        # )


    def __add_foot_force_constraints(self, phase, deltaT):
        """
        Add constraints for the foot forces of a contact phase at a specific time of that phase.
        This will add constraints:
            - ensure that force in z is positive (>= 0)
            - ensure xy force obey friction pyramid
        :param phase: the contact phase to constrain
        :param deltaT: add the constraint for this time point (local time of phase, t=0 is at phase start)
        """
        foot_force = phase.evaluate_foot_force(deltaT)
        force_z = foot_force[2]
        force_xy = foot_force[0:2]
        self.__add_foot_force_constraints_generic(force_xy, force_z)
        # force_limit_z = self.foot_force_max_z #500 #200 #200 #90 #250#180 #60 #20
        # just push
        # print('>> force_z ', force_z)
        # self.opti.subject_to(self.opti.bounded(0, force_z, force_limit_z))
        #
        # # friction pyramid
        # # 1 means 45° angle of friction pyramid walls
        # #      ^= xy-forces could be as large as z force
        # friction_mu_coefficient = 0.5 #0.2 #0.5 #0.2
        # self.opti.subject_to(
        #     self.opti.bounded(-friction_mu_coefficient*force_z,
        #                       phase.evaluate_foot_force(deltaT)[0:2],
        #                       friction_mu_coefficient*force_z)
        # )

    def __add_foot_force_constraints_generic(self, force_xy: MX, force_z: MX):
        """
        Add constraints for the foot forces of a contact phase.
        This will add constraints:
            - ensure that force in z is positive (>= 0)
            - ensure xy force obey friction pyramid
        """
        force_limit_z = self.foot_force_max_z
        # just push
        self.opti.subject_to(self.opti.bounded(0, force_z, force_limit_z))

        # friction pyramid
        # 1 means 45° angle of friction pyramid walls
        #      ^= xy-forces could be as large as z force
        friction_mu_coefficient = 0.5 #0.35 #0.2 #0.5 #0.2
        self.opti.subject_to(
            self.opti.bounded(-friction_mu_coefficient*force_z,
                              force_xy,
                              friction_mu_coefficient*force_z)
        )



    def __add_foot_phase_pos_distance_constraints(self, foot: FootTrajectory,
                                                  foot_phase: Phase,
                                                  t: float,
                                                  foot_kin_constraint_box_center_rel: np.ndarray,
                                                  deltaT=None,
                                                  enforce_at_foot_x1=False,
                                                  use_com_last_x1=False):
        """
        Add constraints for the foot position at time t.
        This will add constraints:
            - that foot is in bounding box relative to com pos
        :param foot_phase: [optional] if this is given constraints will be added for the given flight phase,
                                      otherwise at the global time t.
        :param enforce_at_foot_x1: just used when 'foot_phase' is not none.
                                    If true add constraint for last pos value (x1) of the flight phase,
                                    otherwise add constraint at local phase time deltaT.
        """
        # foot pos
        if foot_phase is not None:
            assert self.x_opti_rolling_time_shift_duration is not None, "phase based foot contains do not work in combination will rolling"
            if enforce_at_foot_x1:
                foot_pos = foot_phase.foot_position.poly_list[-1].x1
            else:
                foot_pos = foot_phase.evaluate_foot_pos(deltaT)
        else:
            foot_pos = foot.evaluate_foot_pos(t)

        # com pos
        if use_com_last_x1:
            com_pos = self.x_opti_com_pos.poly_list[-1].x1
        else:
            com_pos = self.x_opti_com_pos.evaluate_x(t)


        actual_foot_distance_rotated = self.__get_foot_distance_to_com_in_com_frame(
            com_pos=com_pos,
            com_angles=self.x_opti_com_angle.evaluate_x(t),
            foot_pos=foot_pos
        )
        self.opti.subject_to(self.opti.bounded(-self.foot_kin_constraint_box_size,
                                               actual_foot_distance_rotated + foot_kin_constraint_box_center_rel,
                                               self.foot_kin_constraint_box_size))

    def __get_foot_distance_to_com_in_com_frame(self, com_pos, com_angles, foot_pos):
        actual_foot_distance = com_pos - foot_pos
        # foot distance in com frame:
        if self.use_angular_dynamics:
            # rotate foot kin bounding box according to body rotation
            Rot = R_world_frame_to_com_frame_euler(com_angles=com_angles)
            actual_foot_distance_rotated = Rot @ actual_foot_distance
        else:
            actual_foot_distance_rotated = actual_foot_distance
        return actual_foot_distance_rotated




    def __evaluate_final_feet_pos_and_forces(self):
        feet_positions = []
        feet_forces = []
        for foot in self.x_opti_feet:
            # special case for when last phase is looping
            if foot.rolling_time_shift_duration is not None:
                tol = 1e-6
                feet_positions.append(foot.evaluate_foot_pos(self.total_duration - tol))
                feet_forces.append(foot.evaluate_foot_force(self.total_duration - tol))
                continue

            if foot.phases[-1].get_phase_type() == PhaseType.CONTACT:
                feet_positions.append(foot.phases[-1].foot_position)
                feet_forces.append(foot.phases[-1].foot_force.poly_list[-1].x1)
            else:
                feet_positions.append(foot.phases[-1].foot_position.poly_list[-1].x1)
                feet_forces.append(np.array([0, 0, 0]))
        return feet_positions, feet_forces



    def __add_constraints_for_looping_gait(self):
        """
        When looping_last_contact_phase is true these constraints will be added
        to ensure that the produced trajectory is looping.
        Currently just supports start and end rotation is the same -> thus no turning.
        This includes:
            - start and end feet pos match (relative to the base in the rotated base frame)
            - NOT, SINCE NOT REQUIRED: start and end feet velocities match (relative to the base in the rotated base frame)
            - start and end com velocity (pos and rotation) matches
            - start and end com rotation matches
        """
        print(f"> add constrains for looping gait")

        # com pos and angle
        self.opti.subject_to(self.x_opti_com_angle.poly_list[0].x0 == self.x_opti_com_angle.poly_list[-1].x1)
        self.opti.subject_to(self.x_opti_com_angle.poly_list[0].dx0 == self.x_opti_com_angle.poly_list[-1].dx1)
        # TODO: check this if start or end dcom pos is not parametric:
        self.opti.subject_to(self.x_opti_com_pos.poly_list[0].dx0 == self.x_opti_com_pos.poly_list[-1].dx1)
        # self.opti.subject_to(self.x_opti_com_pos.poly_list[0].x0[-1] == self.x_opti_com_pos.poly_list[-1].x1[-1])

        # feet pos rel to base
        # Note that a foot trajectory may be looping
        for foot_i in range(len(self.x_opti_feet)):
            # t_eval_com = 0.0
            # if self.foot_id_with_looping_last_phase == foot_i:
            #     # use the time of the end of the looping phase
            #     t_phase_end = self.x_opti_feet[foot_i].phases[-1].duration
            #     t_eval_com = t_phase_end
            # else:
            #     pass
            # if self.foot_id_with_looping_last_phase == foot_i:
            #     pass

            com_pos_start = self.x_opti_com_pos.evaluate_x(0.0)
            com_angles_start = self.x_opti_com_angle.evaluate_x(0.0)
            com_pos_end = self.x_opti_com_pos.poly_list[-1].x1
            com_angles_end = self.x_opti_com_angle.poly_list[-1].x1

            # get start and end pos no matter the phase type
            # Note: without added offset to last phase of looped trajectory
            foot_pos_abs_start = self.x_opti_feet[foot_i].phases[0].evaluate_foot_pos_start()
            foot_pos_abs_end = self.x_opti_feet[foot_i].phases[-1].evaluate_foot_pos_end()

            foot_pos_start = self.__get_foot_distance_to_com_in_com_frame(
                com_pos=com_pos_start, com_angles=com_angles_start, foot_pos=foot_pos_abs_start)
            foot_pos_end = self.__get_foot_distance_to_com_in_com_frame(
                com_pos=com_pos_end, com_angles=com_angles_end, foot_pos=foot_pos_abs_end)
            self.opti.subject_to(foot_pos_start == foot_pos_end)

            # make velocities match
            # -> not required since first and last phases need to have differnt typs -> vel at start and end is allways zero
            # foot_dpos_abs_start = self.x_opti_feet[foot_i].phases[0].evaluate_foot_pos_start(derivative=True)
            # foot_dpos_abs_end = self.x_opti_feet[foot_i].phases[-1].evaluate_foot_pos_end(derivative=True)
            # def rotate_vel_in_base_frame(com_angles, vel):
            #     Rot = R_world_frame_to_com_frame_euler(com_angles=com_angles)
            #     return Rot @ vel
            # foot_dpos_start = rotate_vel_in_base_frame(com_angles=com_angles_start, vel=foot_dpos_abs_start)
            # foot_dpos_end = rotate_vel_in_base_frame(com_angles=com_angles_start, vel=foot_dpos_abs_end)
            # self.opti.subject_to(foot_dpos_start == foot_dpos_end)

            print(f"> add constrains for looping gait: foot pos for foot {foot_i}")






    ###################################################################################
    ## additional optional constraints and costs ######################################

    @additional_cost_or_constraint
    def add_additional_constraint__com_lin_z_range_of_motion(self, max_z_deviation_from_init):
        """
        Limit com z pos deviation from start com z value at discrete time points.
        """
        add_every_dt = self.base_poly_duration / 4
        for t in np.arange(0, self.total_duration, step=add_every_dt):
            com_pos_z = self.x_opti_com_pos.evaluate_x(t)[2]
            self.opti.subject_to(self.opti.bounded(
                self.param_opti_start_com_pos[-1] - max_z_deviation_from_init,
                com_pos_z,
                self.param_opti_start_com_pos[-1] + max_z_deviation_from_init
            ))
            print(f"> enforce com_lin_z_range_of_motion at t = {t} (t total is {self.total_duration})")


    @additional_cost_or_constraint
    def add_additional_constraint__com_angular_acc(self, max_xy_acc=None, max_z_acc=None):
        """
        Limit com angular acceleration at discrete time points (one per base polynomial).
        """
        for t in np.arange(0, self.total_duration, step=self.base_poly_duration):
            if max_z_acc is not None:
                com_angle_z = self.x_opti_com_angle.evaluate_ddx(t)[2]
                self.opti.subject_to(self.opti.bounded(
                    -max_z_acc,
                    com_angle_z,
                    max_z_acc
                ))
            if max_xy_acc is not None:
                com_angle_xy = self.x_opti_com_angle.evaluate_ddx(t)[0:2]
                self.opti.subject_to(self.opti.bounded(
                    -max_xy_acc,
                    com_angle_xy,
                    max_xy_acc
                ))
            print(f"> enforce add_additional_constraint__com_angular_acc at t = {t} (t total is {self.total_duration})")


    @additional_cost_or_constraint
    def add_additional_constraint__com_angular_vel(self, max_xy_vel=None, max_z_vel=None):
        """
        Limit com angular vel at discrete time points (one per base polynomial).
        """
        for t in np.arange(0, self.total_duration, step=self.base_poly_duration):
            if max_z_vel is not None:
                com_angle_vel_z = self.x_opti_com_angle.evaluate_dx(t)[2]
                self.opti.subject_to(self.opti.bounded(
                    -max_z_vel,
                    com_angle_vel_z,
                    max_z_vel
                ))
            if max_xy_vel is not None:
                com_angle_vel_xy = self.x_opti_com_angle.evaluate_dx(t)[0:2]
                self.opti.subject_to(self.opti.bounded(
                    -max_xy_vel,
                    com_angle_vel_xy,
                    max_xy_vel
                ))
            print(f"> enforce add_additional_constraint__com_angular_vel at t = {t} (t total is {self.total_duration})")



    class AllwaysAtLeastOneFootGroundContactConstraintType:
        EVERY_DT = 0,
        AT_N_POINTS_OF_FOOT_FLIGHT_PHASES = 1
    @additional_cost_or_constraint
    def add_additional_constraint__allways_at_least_one_foot_ground_contact(self,
                                                                             type: AllwaysAtLeastOneFootGroundContactConstraintType,
                                                                             add_every_dt=0.04,
                                                                             n_constraints_per_phase=3):
        """
        Add a constraint set that ensures that the force by the feet in z on the com is allways > 0 (AT_N_POINTS_OF_FOOT_FLIGHT_PHASES).
        Or that the com acc in z is allways > -9 (EVERY_DT).
        This will ensure that allways one foot has ground contact.
        NOTE: THIS DOES NOT WORK VERY WELL
        """
        assert not self.additional_constraints__allways_at_least_one_foot_ground_contact, "can't add this constraint set twice."
        assert self.num_feet == 2, "just works with two feet"
        assert False, "Do not use this, it does not work well"

        # since ipopt can't really handle > 0
        min_force_z = 0.01

        if type == self.AllwaysAtLeastOneFootGroundContactConstraintType.EVERY_DT:
            # enforce at discrete time steps
            print(">> allways_at_least_one_foot_ground_contact -> EVERY_DT")
            for t in np.arange(0, self.total_duration, step=add_every_dt):  # 0.08): # try also 0.02
                # enforce total force > 0
                #sum_forces_z_abs = MX.zeros(1)
                # for foot in self.x_opti_feet:
                #     sum_forces_z_abs += MX.fabs(foot.evaluate_foot_force(t))
                #foot_forces_z = self.evaluate_sum_foot_forces(t)[-1]
                # self.opti.subject_to(sum_forces_z_abs > min_force_z)
                self.opti.subject_to(self.x_opti_com_pos.evaluate_ddx(t)[2] > -9)
                print(f"> enforce allways_at_least_one_foot_ground_contact at t = {t} (t total is {self.total_duration})")

        elif type == self.AllwaysAtLeastOneFootGroundContactConstraintType.AT_N_POINTS_OF_FOOT_FLIGHT_PHASES:
            # enforce at n points of flight phases of each foot
            print(">> allways_at_least_one_foot_ground_contact -> AT_N_POINTS_OF_FOOT_FLIGHT_PHASES")
            def add_constraint_for_flight_phases_of_foot(foot_in_flight: FootTrajectory, other_foot: FootTrajectory):
                t_phase_start = MX.zeros(1)
                for phase_foot_0 in foot_in_flight.phases:
                    if phase_foot_0.get_phase_type() == PhaseType.FLIGHT:
                        # make sure that during flight phase the other foot pushes on the ground
                        for i in range(n_constraints_per_phase):
                            t_phase = (phase_foot_0.duration / n_constraints_per_phase) * i
                            t_global = t_phase_start + t_phase
                            foot1_force_z = other_foot.evaluate_foot_force(t_phase_start)
                            self.opti.subject_to(foot1_force_z > min_force_z)
                            print(
                                f"> enforce allways_at_least_one_foot_ground_contact at t = {t_global} (t total is {self.total_duration})")
                    t_phase_start += phase_foot_0.duration
            # add for each foot combi
            add_constraint_for_flight_phases_of_foot(self.x_opti_feet[0], self.x_opti_feet[1])
            add_constraint_for_flight_phases_of_foot(self.x_opti_feet[1], self.x_opti_feet[0])
        print()


    @additional_cost_or_constraint
    def add_additional_cost__com_rotation_acc_z(self, weight=0.1):
        for t in np.arange(0, self.total_duration, step=self.base_poly_duration):  # 0.08): # try also 0.02
            self.cost += scalar_prod((self.x_opti_com_angle.evaluate_ddx(t)[-1]))*weight
            print(f"> add cost com_rotation_acc at t = {t} (t total is {self.total_duration})")


    @additional_cost_or_constraint
    def add_additional_cost_or_constraint__com_linear_acc(self,
                                                          as_constraint=True,
                                                          constraint_max_acc: np.ndarray=1,
                                                          cost_weight=0.1):
        """
        Limit com linear acc.
        :param as_constraint: If true enforce this as constraint, then com linar acc is constraint to constraint_max_acc
                              (manhattan distance).
                              If false a penalty term for the squared com linear acc is added to the cost.
        :param constraint_max_acc:
        :param cost_weight:   When formulated as a cost, the cost term is weighted by this value.
        """
        for t in np.arange(0, self.total_duration, step=self.base_poly_duration):  # 0.08): # try also 0.02
            if not as_constraint:
                self.cost += scalar_prod((self.x_opti_com_pos.evaluate_ddx(t))*cost_weight)
            else:
                self.opti.subject_to(self.opti.bounded(
                    -constraint_max_acc,
                    self.x_opti_com_pos.evaluate_ddx(t),
                    constraint_max_acc
                ))
            print(f"> add cost add_additional_cost_or_constraint__com_linear_acc at t = {t} (t total is {self.total_duration})")


    @additional_cost_or_constraint
    def add_additional_cost_or_constraint__feet_at_nominal_at_end_of_trajectory(self,
                                                                                as_constraint=True,
                                                                                constraint_box_size_mul_factor=0.5,
                                                                                cost_weight=0.1):
        """
        At the end of the trajectory make the feet stand below the com (as close at possible to the nominal pose in xy).
        :param as_constraint: If true enfroce this via constraint:
                              At the end of the trjectory the feets max deviation from the nominal pose in xy is
                              'foot_kin_constraint_box_size[0:2] * constraint_box_size_mul_factor'.
                              If false add a cost for the devation of the feet from the nominal pose at the final timestep.
                              This cost is scaled via cost_weight.
        """
        for foot_i, foot in enumerate(self.x_opti_feet):
            t_phase_start = MX.zeros(1)
            for foot_phase in foot.phases:
                if foot_phase.get_phase_type() == PhaseType.CONTACT and foot_phase == foot.phases[-1]:
                    foot_phase: ContactPhase = foot_phase
                    t_global = t_phase_start + foot_phase.duration - 1e-2
                    # during contact phase foot should be close to nominal pos below com
                    distance_to_com_xy = self.__get_foot_distance_to_com_in_com_frame(
                        com_pos=self.x_opti_com_pos.evaluate_x(t_global),
                        com_angles=self.x_opti_com_angle.evaluate_x(t_global),
                        foot_pos=foot_phase.foot_position
                    )[0:2]
                    if not as_constraint:
                        self.cost += scalar_prod(distance_to_com_xy + self.foot_kin_constraint_box_center_rel[foot_i][0:2])*cost_weight
                    else:
                        max_diff = self.foot_kin_constraint_box_size[0:2] * constraint_box_size_mul_factor
                        self.opti.subject_to(self.opti.bounded(
                            -max_diff,
                            (distance_to_com_xy + self.foot_kin_constraint_box_center_rel[foot_i][0:2]),
                            max_diff
                        ))
                    print(f"> add add_additional_cost_or_constraint__feet_at_nominal_at_end_of_last_contact at t = {t_global} (t total is {self.total_duration})")
                t_phase_start += foot_phase.duration


    @additional_cost_or_constraint
    def add_additional_constraint__feet_max_velocity_gloval(self, max_foot_vel):
        """
        Restrict the maximum foot velocity in global catesian coodinates (approx at middle of each fligh phase).
        This is usefull when the phase_duration_min is set to a very low value (e.g. =0) to avoid instantainus jumps
        of the feet in the flight phase.
        :param max_foot_vel: This can be a vector or a scalar value.
                             The foot velocity has to be in range of +- max_foot_vel (manhattan distance).
        """
        assert self.additional_foot_flight_smooth_vel_constraints, "this constraint should also be enabled"
        for foot_i, foot in enumerate(self.x_opti_feet):
            for phase_i, phase in enumerate(foot.get_flight_phases()):
                assert len(phase.foot_position.poly_list) == 2, "this constraint only works for two polies per flight phase"
                poly_middle_vel = phase.foot_position.poly_list[0].dx1[0:2]
                # @note this works because with the additional_foot_flight_smooth_constraints
                #       poly_middle_vel is the avg velocity of the foot in the flight phase
                self.opti.subject_to(self.opti.bounded(
                    -max_foot_vel,
                    poly_middle_vel,
                    max_foot_vel
                ))
                print(f"> add add_additional_constraint__feet_max_velocity_gloval at middle of flight poly")



    @additional_cost_or_constraint
    def add_additional_constraint__feet_max_step_distance(self, max_foot_step_distance, via_flight_phases_start_end):
        """
        Restrict the maximum foot step distance (eucledian).
        """
        if via_flight_phases_start_end:
            for foot_i, foot in enumerate(self.x_opti_feet):
                flight_phases = list(foot.get_flight_phases())
                for phase_i, phase in enumerate(flight_phases):
                    pos0 = phase.evaluate_foot_pos_start()
                    pos1 = phase.evaluate_foot_pos_end()
                    distance = pos0 - pos1

                    self.opti.subject_to(self.opti.bounded(
                        -max_foot_step_distance ** 2,
                        distance.T @ distance,
                        max_foot_step_distance ** 2
                    ))
                    print(
                        f"> add add_additional_constraint__feet_max_step_distance for flight phase {phase_i} (start to end pos)")
        else:
            for foot_i, foot in enumerate(self.x_opti_feet):
                contact_phases = list(foot.get_contact_phases())
                for phase_i, phase in enumerate(contact_phases):
                    if phase_i >= len(contact_phases) - 1:
                        break
                    pos0 = phase.foot_position
                    pos1 = contact_phases[phase_i+1].foot_position
                    distance = pos0 - pos1

                    self.opti.subject_to(self.opti.bounded(
                        -max_foot_step_distance**2,
                        distance.T @ distance,
                        max_foot_step_distance**2
                    ))
                    print(f"> add add_additional_constraint__feet_max_step_distance for contact phases {phase_i} - {phase_i+1}")


    @additional_cost_or_constraint
    def add_additional_constraint__feet_phases_max_duration_diff_to_previous_phase(self,
                                                                                   max_duration_diff_seconds,
                                                                                   diff_to_last_same_phase_type=True,
                                                                                   ignore_first_and_last_contact_phases=True
                                                                                   ):
        """
				Restrict the maximum phase duration difference.
        """
        for foot_i, foot in enumerate(self.x_opti_feet):
            phases = foot.phases
            for phase_i, phase in enumerate(phases):
                #is_pre_last = phase_i == len(phases) - 2
                if phase_i >= len(phases) - 1:
                    break

                phase_next_idx = phase_i+1
                phase_next = phases[phase_next_idx]
                if diff_to_last_same_phase_type:
                    if phase_i >= len(phases) - 2:
                        break
                    phase_next_idx = phase_i+2
                    phase_next = phases[phase_next_idx]
                    assert phase_next.get_phase_type() == phase.get_phase_type()

                if ((phase_i == 0 and phase.get_phase_type() == PhaseType.CONTACT) or
                    (phase_next == phases[-1] and phase_next.get_phase_type() == PhaseType.CONTACT)) and ignore_first_and_last_contact_phases:
                    print(f"> add add_additional_constraint__feet_phases_max_duration_diff_to_previous_phase"
                          f" ignore phases {phase_i} - {phase_next_idx}")
                    continue


                d0 = phase.duration
                d1 = phase_next.duration
                diff = d0 - d1

                self.opti.subject_to(self.opti.bounded(
                    -max_duration_diff_seconds,
                    diff,
                    max_duration_diff_seconds
                ))
                print(f"> add add_additional_constraint__feet_phases_max_duration_diff_to_previous_phase for phases {phase_i} - {phase_next_idx}")

    @additional_cost_or_constraint
    def add_additional_constraint__feet_phases_max_duration_diff_between_feet(self,
                                                                                   max_duration_diff_seconds,
                                                                                   ignore_first_and_last_contact_phases=True
                                                                                   ):
        """
				Restrict the maximum phase duration difference.
				"""
        for foot_i, foot in enumerate(self.x_opti_feet):
            for phase_i, phase in enumerate(foot.phases):
                if ((phase_i == 0 and phase.get_phase_type() == PhaseType.CONTACT) or
                    (phase_i == foot.phases[-1] and phase.get_phase_type() == PhaseType.CONTACT)) and ignore_first_and_last_contact_phases:
                    print(f"> add add_additional_constraint__feet_phases_max_duration_diff_between_feet"
                          f" ignore phases {phase_i}")
                    continue
                  
                # to all other feet after this one
                for foot_j, foot_other in enumerate(self.x_opti_feet[foot_i+1:]):
                    d0 = phase.duration
                    d1 = foot_other.phases[phase_i].duration
                    diff = d0 - d1

                    self.opti.subject_to(self.opti.bounded(
                        -max_duration_diff_seconds,
                        diff,
                        max_duration_diff_seconds
                    ))
                    print(
                        f"> add add_additional_constraint__feet_phases_max_duration_diff_between_feet "
                        f"between feet {foot_i} - {foot_j+1} for phase {phase_i}")






    ###################################################################################
    ## solving ########################################################################


    def bake_solver(self, just_show_init_values=False, max_iter=1000, jit=False, expand_to_SX=True):
        # add cost if cost was defined
        if self.cost is not None and not self.cost.is_zero():
            print('>> adding cost term')
            #print('>> also add final com pos to cost term')
            #self.cost += scalar_prod()
            self.opti.minimize(self.cost)

        print('> bake_solver ...')
        # solver parameters
        p_opts = {
            "expand": expand_to_SX,   # auto convert MX to SX expression (here it improves efficiency)
            # IMPORTANT: will create real variables bound forwarded to ipopt (these bounds will allways be honored)
            "detect_simple_bounds": False, # but we just want to apply this to some constraints NOT all.
            # @todo solve this via manually creating nlpsol from opti: https://groups.google.com/g/casadi-users/c/hX6bTw6lCSw/m/-VfUVyn_BAAJ
            # @todo this seems to sometimes improve convergence -> implement ^

            "jit": jit,  # test jit compiling the nlp
            "compiler": "shell",
            "jit_options": {
                "compiler": "gcc",  # ccache gcc
                "verbose": True,
                "flags": ['-pipe', '-fPIC'], #'-O2'
                "temp_suffix": False
            },
            'print_time': 2
        }
        # ipopt options
        ipopt_opts = {
            "max_iter": max_iter,

            # testing
            "check_derivatives_for_naninf": "yes",
            "derivative_test": "first-order",
            #"derivative_test_perturbation": 1e-7,
            #"point_perturbation_radius": 1000000,
            "derivative_test_tol": 1e-3,

            # gradient of the constraints
            # "jacobian_approximation": "finite-difference-values",
            "jacobian_approximation": "exact",  # "exact" #does not work as well as 'finite-difference-values'
            # "gradient_approximation": "finite-difference-values",
            # gradient of the objective (unused here)
            #"gradient_approximation": "exact",
            "hessian_approximation": "limited-memory",  # essential to converge
            "tol": 0.001, # same as in towr
            "linear_solver": "mumps"
        }
        if just_show_init_values:
            ipopt_opts['max_iter'] = 0

        # @todo for now just check
        if self.num_feet >= 2:
            ipopt_opts['derivative_test'] = "none"

        # create solver
        self.opti.solver('ipopt', p_opts, ipopt_opts)
        if not self.feet_last_phase_duration_implicit:
            self.opti.callback(lambda i: print('>> total duration [foot0]: ', self.opti.debug.value(self.x_opti_feet[0].get_total_duration())))

        self.opti.bake_solve_with_simple_var_bounds('ipopt', p_opts, ipopt_opts)
        self.__baked_solver = True
        print('> bake_solver DONE')


    def solve_opti(self,
                   start_com_pos: np.ndarray,
                   start_feet_pos: np.ndarray,
                   end_com_pos: np.ndarray,
                   start_com_dpos: np.ndarray = None,
                   start_foot_dpos: np.ndarray = None,

                   init_gait_type: initial_gait.INIT_GAIT_TYPE = initial_gait.INIT_GAIT_TYPE.ALL_FEET_JUMP,
                   given_flight_phase_durations: list[np.array] = None,
                   given_contact_phase_durations: list[np.array] = None,
                   just_show_init_values=False,
                   max_iter=1000,
                   check_solution_feasable=True
                   ):
        if not self.__baked_solver:
            self.bake_solver(just_show_init_values, max_iter)
        assert self.__baked_solver, "call bake_solver() first"

        # set if value is variable
        def set_value(not_set: bool, parameter, value):
            if not not_set:
                self.opti.set_value(parameter, value)

        # set start state parameters
        # com
        set_value(False, self.param_opti_start_com_pos, start_com_pos)
        set_value(self.params_to_optimize.param_opti_start_com_dpos,
                  self.param_opti_start_com_dpos, start_com_dpos if start_com_dpos is not None else np.zeros(3))
        set_value(self.params_to_optimize.param_opti_start_com_angle,
                  self.param_opti_start_com_angle, np.zeros(self.d))
        set_value(self.params_to_optimize.param_opti_start_com_dangle,
                  self.param_opti_start_com_dangle, np.zeros(self.d))

        # feet
        start_foot_dpos = start_foot_dpos if start_foot_dpos is not None else np.zeros((self.num_feet, self.d))
        set_value(self.params_to_optimize.param_opti_start_foot_pos,
                  self.param_opti_start_foot_pos, start_feet_pos)
        set_value(self.params_to_optimize.param_opti_start_foot_dpos,
                  self.param_opti_start_foot_dpos, start_foot_dpos)

        # end pos for com
        set_value(False, self.param_opti_end_com_pos, end_com_pos)

        # init values
        # deltaT should not be 0 because we divide by it in the polynomial calc
        initial_gait.gen_initial_gait(
            self,
            init_gait_type=init_gait_type,
            start_feet_pos=start_feet_pos,
            given_flight_phase_durations=given_flight_phase_durations,
            given_contact_phase_durations=given_contact_phase_durations
        )

        # run solver
        print('> start solving ...')
        solver_stats = {}
        try:
            #self.opti.solve()
            solver_stats = self.opti.solve_with_simple_var_bounds()
        except Exception as error:
            print(error)
            #self.opti.debug.show_infeasibilities()

        if solver_stats['success'] == True:
            print('> found solution!')
            self.solution_converged_to_optimal = True
        else:
            self.solution_converged_to_optimal = False
        self.solution_solver_stats = solver_stats


        # get results
        poly_vals = self.opti.value(self.x_opti_com_pos.x_opti_vars)
        #print(poly_vals)
        #print(self.opti.value(self.x_opti_com_x.poly_list[0].x1))

        #Polynomial3.create_from_flat_param_array(poly_vals[:, 0:1], self.d).plot(0)
        # plot
        #self.x_opti_com_poly[0].create_new_poly_with_solved_opti_values(self.opti).plot()
        #self.x_opti_com_angle.evaluate_solution_and_plot_full_trajectory(show_plot=False)



        if check_solution_feasable:
            self.solution_is_feasible = self.evaluate_solution__check_solution_feasible()


    def set_initial_opti_values(self):
        assert False, "unused"
        return
        #for p in self.x_opti_com_poly:
        #    # deltaT should not be 0 because we divide by it in the polynomial calc
        #    self.opti.set_initial(p.deltaT, 1)
        start_com_pos = np.array(self.opti.value(self.param_opti_start_com_pos))[0:2]
        end_com_pos = np.array(self.opti.value(self.param_opti_end_com_pos))[0:2]
        self.x_opti_com_pos.set_initial_opti_values__x1_z(
            initial_x1_z_middle=self.opti.value(self.param_opti_start_com_pos)[-1],#0.8,
            initial_x1_z_end=self.opti.value(self.param_opti_end_com_pos)[-1],
            interpolation_xy_start=start_com_pos,
            interpolation_xy_end=end_com_pos
        )

        # init values for all feet
        for foot_i,  foot in enumerate(self.x_opti_feet):
            start_foot_pos = np.array(self.opti.value(self.param_opti_start_foot_pos[foot_i, :]))[0:2]
            end_foot_pos = start_foot_pos + (end_com_pos - start_com_pos)
            foot.set_initial_opti_values(
                init_for_duration_of_each_phase=self.total_duration / len(foot.phases),
                init_z_height_for_flight=0,#0.25,
                init_z_dheight_for_flight=0*5,
                init_z_force_for_contact=self.mass*self.gravity_acc, #+ 50,#150,
                init_z_dforce_for_contact=self.mass*self.gravity_acc,#150
                interpolation_xy_start=start_foot_pos,
                interpolation_xy_end=end_foot_pos
            )

        print("## init force to ", self.mass*self.gravity_acc)
