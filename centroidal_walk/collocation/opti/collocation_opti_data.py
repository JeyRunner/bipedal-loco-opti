from typing import List

import numpy as np
import yaml
from numpy import ndarray

from centroidal_walk.casadi_util_functions import scalar_prod
from centroidal_walk.collocation.opti import initial_gait
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import Phase, ContactPhase
from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.foot_trajectory.foot_trajectory import FootTrajectory
from centroidal_walk.collocation.opti.casadi_util.OptiLoadable import OptiLoadable
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.plotting import plot_com_and_foot_trajectories_xyz
from centroidal_walk.collocation.plotting_data import FootConstraintTimePoints
from centroidal_walk.collocation.spline.spline_trajectory import *
from centroidal_walk.visualization import simulate_and_plot
from centroidal_walk.visualization.RobotAnimation3D import RobotAnimation3D
from centroidal_walk.collocation.serialization.OptiDecorators import *


@dataclass
class ParamsToOptimize:
    """
    Define which parameters are fixed and which are optimized.
    Setting to True means the parm will be optimized.
    """
    param_opti_start_com_dpos: bool = False
    param_opti_start_com_angle: bool = False
    param_opti_start_com_dangle: bool = False
    param_opti_start_foot_pos: bool = False
    param_opti_start_foot_dpos: bool = False


class CentroidalDynPolyOptiData(CentroidalDynPoly):
    """
    The Data (splines and trajectories) for CentroidalDynPolyOpti.
    This calls can be used to load serialized solution from CentroidalDynPolyOpti and evaluate, display them.
    """

    # number of dimensions
    d = 3


    # opti problem formulation parameters
    num_phases: int
    feet_first_phase_type: list[PhaseType]

    foot_kin_constraint_box_size: ndarray
    foot_kin_constraint_box_center_rel: list[ndarray]

    use_angular_dynamics: bool
    fixed_phase_durations: list[ndarray] | None
    phase_duration_min: float
    feet_last_phase_duration_implicit: bool

    foot_force_max_z: float
    foot_force_at_trajectory_end_and_start_variable: bool
    max_com_angle_xy_abs: float | None

    total_duration: float
    base_poly_duration: float
    num_polynomials_for_com_trajectory: int

    solution_converged_to_optimal: bool
    solution_is_feasible: bool
    solution_solver_stats: dict
    as_just_solution_loader: bool



    # opti problem
    opti: OptiWithSimpleBounds | OptiLoadable

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
    # list of polynomials that represent the com trajectory (zyx-intrinsic-euler = xyz-intrinsic-euler angles or yaw, pitch, roll)
    x_opti_com_angle: SplineTrajectory
    x_opti_feet: list[FootTrajectory]  # trajectories for all feet
    x_opti_feet__start_pos: list[Opti.variable] # variables for the initial feet positions

    # shifting of all phases of a foot (optional)
    x_opti_rolling_time_shift_duration: Opti.variable


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
                 num_phases=1+2+2,
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
                 additional_foot_flight_smooth_vel_constraints=False,
                 as_just_solution_loader=True
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
        :param additional_foot_flight_smooth_vel_constraints:
                                                if true additional constraint will be added so that the foot xy velocity in
                                                the middle of the flight phase has to match the avg vel. during that phase.
                                                Also, pos in the middle of flight has to be
                                                the middle between previous and next step pos.
                                                Note that this assumes two polynomial for the foot motion during each
                                                flight phase.
        :param as_just_solution_loader:         if true, this will not use the real casadi.Opti but just OptiLoadable.
                                                Thus, just solutions can be loaded and evaluated. Solving is not supported.
                                                if false, utilize a real solver.
        :param params_to_optimize:              set which parameters will not be set as initial conditions
                                                but which will be optimized.
        """
        if num_feet is None:
            num_feet = 2 if use_two_feet else 1

        super().__init__(mass, InertiaMatrix, num_feet)
        self.foot_force_max_z = foot_force_max_z
        self.num_phases = num_phases
        self.looping_last_contact_phase = looping_last_contact_phase
        self.looping_last_contact_phase_time_shift_fixed = looping_last_contact_phase_time_shift_fixed
        self.feet_first_phase_type = feet_first_phase_type
        self.fixed_phase_durations = fixed_phase_durations
        self.use_angular_dynamics = use_angular_dynamics
        self.foot_force_at_trajectory_end_and_start_variable = foot_force_at_trajectory_end_and_start_variable
        self.phase_duration_min = phase_duration_min
        self.params_to_optimize = params_to_optimize
        self.contact_phase_duration_min = contact_phase_duration_min
        self.max_com_angle_xy_abs = max_com_angle_xy_abs
        self.viz3d: RobotAnimation3D = None
        self.as_just_solution_loader = as_just_solution_loader
        self.constraint_points_t__feet = None
        self.constraint_points_t__dynamics = None
        self.solution_converged_to_optimal = False
        solution_is_feasable = 'Unknown'

        # additional constraint sets
        self.additional_intermediate_foot_force_constraints = additional_intermediate_foot_force_constraints
        self.additional_foot_flight_smooth_vel_constraints = additional_foot_flight_smooth_vel_constraints

        # additional constraint and cost sets which can be added by separate functions
        # saves if a constraint/cost is used and with which parameters
        # the key is the constraint name and the values are its parameters
        # if a name is not in that dict the constraint is not used
        self.additional_costs_and_constraints_parameters: dict = {}

        # parameters
        self.foot_kin_constraint_box_center_rel = foot_kin_constraint_box_center_rel
        self.foot_kin_constraint_box_size = foot_kin_constraint_box_size
        assert len(foot_kin_constraint_box_center_rel) == num_feet, \
            "number of feet has to match number of elements in foot_kin_constraint_box_center_rel"

        self.total_duration = total_duration #4.4 #2#4.4
        self.base_poly_duration = base_poly_duration #0.05
        # for com pos and angles
        if num_polynomials_for_com_trajectory is None:
            self.num_polynomials_for_com_trajectory = int(self.total_duration/self.base_poly_duration)
        else:
            self.num_polynomials_for_com_trajectory = num_polynomials_for_com_trajectory
        print('## use num_polynomials_for_com_trajectory = ', self.num_polynomials_for_com_trajectory)


        ###################################
        ### create opti problem
        #self.opti = casadi.Opti()
        if as_just_solution_loader:
            # dummy opti
            self.opti = OptiLoadable()
        else:
            # real opti
            self.opti = OptiWithSimpleBounds.create()
            # self.opti.describe_variable = describe_variable

        self.__create_trajectories_and_parameters()


    def __create_trajectories_and_parameters(self):
        def param_or_var(as_variable: bool, *dimensions):
            if as_variable:
                return self.opti.variable(*dimensions)
            else:
                return self.opti.parameter(*dimensions)

        # parameters: start state
        # @todo add class param with dict which of these are optimized and which are fixed as parameter
        self.param_opti_start_com_pos = self.opti.parameter(self.d)
        self.param_opti_start_com_dpos = param_or_var(self.params_to_optimize.param_opti_start_com_dpos, self.d)
        self.param_opti_start_com_angle = param_or_var(self.params_to_optimize.param_opti_start_com_angle, self.d)
        self.param_opti_start_com_dangle = param_or_var(self.params_to_optimize.param_opti_start_com_dangle, self.d)
        self.param_opti_start_foot_pos = param_or_var(self.params_to_optimize.param_opti_start_foot_pos, self.num_feet, self.d)
        self.param_opti_start_foot_dpos = param_or_var(self.params_to_optimize.param_opti_start_foot_dpos, self.num_feet, self.d)
        self.param_opti_end_com_pos = self.opti.parameter(self.d)

        # opti trajectories
        # com
        self.x_opti_com_pos = SplineTrajectory(
            self.num_polynomials_for_com_trajectory,
            self.param_opti_start_com_pos,
            self.param_opti_start_com_dpos,
            param_opti_end_x=self.param_opti_end_com_pos,
            #param_opti_end_dx=np.zeros(self.d),  # final com acc zero
            param_opti_end_dx=None,  # final com acc zero
            opti=self.opti,
            spline_polynomial_degree=3,
            ddx_consistency_constraint=True,
            ddx_consistency_constraint_manually=True,
            given_total_duration=self.total_duration,
            dimensions=self.d,
            scope_name=['com_pos'],
            # _debug__casadi_ifelse_nested=True
        )
        self.x_opti_com_angle = SplineTrajectory(
            self.num_polynomials_for_com_trajectory,
            self.param_opti_start_com_angle,
            self.param_opti_start_com_dangle,
            opti=self.opti,
            # param_opti_end_dx=np.zeros(self.d),
            spline_polynomial_degree=3,
            ddx_consistency_constraint=True,
            ddx_consistency_constraint_manually=True,
            given_total_duration=self.total_duration,
            dimensions=self.d,
            scope_name=['com_angle'],
            # _debug__casadi_ifelse_nested=True
        )

        self.foot_id_with_looping_last_phase = None
        if self.looping_last_contact_phase:
            assert not self.foot_force_at_trajectory_end_and_start_variable, "when using looping to generate looping trajectories, start and end forces have to be fixed"
            assert self.num_feet == 2, "just supported for bipeds"
            for i in range(self.num_feet):
                # for a looping trajectory we need a foot trajectory with first phase as flight and last phase as contact
                # -> contact forces will be looped
                if (FootTrajectory.infer_if_last_phase_is_contact_phase(self.feet_first_phase_type[i], self.num_phases)
                        and self.feet_first_phase_type[i] == PhaseType.FLIGHT):
                    #assert self.foot_id_with_looping_last_phase is None, "just can have one foot trajectory with looping"
                    if self.foot_id_with_looping_last_phase is None:
                        self.foot_id_with_looping_last_phase = i
            assert self.foot_id_with_looping_last_phase is not None, ("could not find usable foot trajectory for looping "
                                                          "(needs to have first phase flight and last phase contact)")

        # create feet trajectories (each foot has multiple phases)
        self.feet_last_phase_duration_implicit = True
        self.x_opti_feet: list[FootTrajectory] = []
        self.x_opti_feet__start_pos: list[Opti.variable] = []
        self.x_opti_rolling_time_shift_duration = None
        for i in range(self.num_feet):
            # foot start variable
            # self.x_opti_feet__start_pos.append(self.opti.variable(self.d))
            # describe_variable(self.opti, self.x_opti_feet__start_pos[-1], 'x_opti_feet__start_pos',
            #                  [f'foot_{i}_trajectory', 'foot_position'])

            # handle looping trajectory
            foot__looping_last_contact_phase_time_shift_fixed = None
            base_euler_angle_difference_end_to_start = np.zeros(3)
            if i == self.foot_id_with_looping_last_phase:
                print(f'>> use foot {i} for looping trajectory')
                assert foot__looping_last_contact_phase_time_shift_fixed is None
                if foot__looping_last_contact_phase_time_shift_fixed is None:
                    foot__looping_last_contact_phase_time_shift_fixed = self.opti.variable(1)
                    describe_variable(self.opti, foot__looping_last_contact_phase_time_shift_fixed,
                                      'x_opti_feet__rolling_time_shift_duration',
                                      [f'foot_{i}_trajectory'])
                else:
                    foot__looping_last_contact_phase_time_shift_fixed = self.looping_last_contact_phase_time_shift_fixed
                # to rotate the looped force value we need the rotation difference between start and end of the base
                base_euler_angle_difference_end_to_start = (self.x_opti_com_angle.poly_list[0].x0
                                                            - self.x_opti_com_angle.poly_list[-1].x1)
                self.x_opti_rolling_time_shift_duration = foot__looping_last_contact_phase_time_shift_fixed

            foot_trajectory = FootTrajectory(
                num_polynomials_foot_pos_in_flight=2,  # enough to lift the foot, using 3 may produce jittery motions
                num_polynomials_foot_force_in_contact=3,
                num_phases=self.num_phases,
                rolling_time_shift_duration=foot__looping_last_contact_phase_time_shift_fixed,
                first_phase_type=self.feet_first_phase_type[i],
                param_opti_start_foot_pos=self.param_opti_start_foot_pos[i, :].T,
                param_opti_start_foot_dpos=self.param_opti_start_foot_dpos[i, :].T, # bring values in second dim to first dim
                opti=self.opti,
                dimensions=self.d,
                total_duration=self.total_duration,
                base_euler_angle_difference_end_to_start=base_euler_angle_difference_end_to_start,
                fixed_phase_durations=None if self.fixed_phase_durations is None else self.fixed_phase_durations[i],
                # phase duration range (relevant when optimizing phase durations)
                phase_duration_min=self.phase_duration_min,
                contact_phase_duration_min=self.contact_phase_duration_min,
                phase_duration_max=10,
                last_phase_duration_implicit=self.feet_last_phase_duration_implicit,
                foot_force_at_trajectory_end_and_start_variable=self.foot_force_at_trajectory_end_and_start_variable,
                scope_name=[f'foot_{i}_trajectory']
            )
            self.x_opti_feet.append(foot_trajectory)

        # print the vars
        if not self.as_just_solution_loader:
            self.print_all_opti_vars()




    def evaluate_sum_foot_forces(self, time) -> MX:
        """
        Get the sum of all foot force for a time point (can also be array of time points).
        :param time: float or array
        """
        forces = MX.zeros(3)
        for foot in self.x_opti_feet:
            foot_force = foot.evaluate_foot_force(time)
            # workaround when time is >= total_duration
            # currently just suppored when time is a scalar
            if not hasattr(time, 'shape') or time.shape == (1, 1) or time.shape == ():
                if time >= self.total_duration:
                    # assume last phase is contact phase
                    if foot.phases[-1].get_phase_type() == PhaseType.CONTACT:
                        foot_force = foot.phases[-1].foot_force.poly_list[-1].x1
                    else:
                        foot_force = 0  # flight phase has no force
                    print('last foot_force: ', foot_force)
            elif np.any(time >= self.total_duration):
                print("[[WARNING]] evaluate_sum_foot_forces", "time >= total_duration")
            forces = forces + foot_force
        return forces




    ###############################################################
    ## evaluation of solution

    def value(self, expression: MX):
        """
        Get the value of an expression, use e.g. to get trajectory data.
        Example:
        o.value(o.x_opti_com_pos.evaluate_x(np.arange(0, 10)))

        :param expression:
        :return: the value as DM or np array
        """
        return self.opti.value(expression)


    class FootForceExtremeValues:
        pass
    def evaluate_solution__foot_force_extreme_values(self) -> list[list[np.ndarray]]:
        """
        Get the most extreme values (seperate for x,y,z) for each foot force.
        Use this to check if the foot z force values are allways positive.
        :return: for each foot a list of the most extreme foot forces.
                 The inner list for each foot containt serpate extreme value for x, y, z (for each min, max)
                 (for each also the other 3d values are included).
        """
        feet_force_exteme_vals = []

        for foot in self.x_opti_feet:
            foot_force_exteme_vals = []
            # add start values
            for d in range(0, 3):
                # seperate 3d values for each extreme in each dim
                v = np.zeros((3, 2))
                v[:, 0] = np.inf # min
                v[:, 1] = -np.inf # max
                foot_force_exteme_vals.append(v)

            # go over all contact phases
            for contact_phase in foot.get_contact_phases():
                t = np.linspace(0, self.value(contact_phase.duration)-1e-5, num=len(contact_phase.foot_force.poly_list)*2*3)
                force_vals = self.value(contact_phase.evaluate_foot_force(t))
                force_vals_max = np.max(force_vals, axis=1)
                force_vals_arg_max = np.argmax(force_vals, axis=1)
                force_vals_min = np.min(force_vals, axis=1)
                force_vals_arg_min = np.argmin(force_vals, axis=1)

                for dim in range(0, 3):
                    # max
                    is_max = 1
                    if foot_force_exteme_vals[dim][dim, is_max] < force_vals_max[dim]:
                        foot_force_exteme_vals[dim][:, is_max] = force_vals[:, force_vals_arg_max[dim]]
                    is_max = 0
                    if foot_force_exteme_vals[dim][dim, is_max] > force_vals_min[dim]:
                        foot_force_exteme_vals[dim][:, is_max] = force_vals[:, force_vals_arg_min[dim]]
            feet_force_exteme_vals.append(foot_force_exteme_vals)
        return feet_force_exteme_vals


    def evaluate_solution__foot_forces_z_min(self) -> float:
        """
        Get the minimum z force over all feet over the whole trajectory
        """
        feet_force_extreme_vals = self.evaluate_solution__foot_force_extreme_values()
        z_min_force = np.inf
        for foot in feet_force_extreme_vals:
            z_min_force = min(z_min_force, foot[-1][-1, 0])
        return z_min_force


    def evaluate_solution__max_dyn_lin_error(self) -> (float, float):
        """
        Get the maximum devaition (and abs sum) of the actual linear dynamics values and the approximated ones
        (approx by the com trajectory).
        :return (max, abs sum)
        """
        t = np.linspace(0, self.total_duration - 1e-5, num=self.num_polynomials_for_com_trajectory*8)
        violation_lin = self.get_dynamics_com_ddpos_violation(
            com_ddpos=self.x_opti_com_pos.evaluate_x(t),
            feet_forces=[foot.evaluate_foot_force(t) for foot in self.x_opti_feet]
        )
        violation = self.value(violation_lin)
        violation_lin_max = np.max(violation)
        violation_lin_sum= np.sum(np.abs(violation))
        return violation_lin_max, violation_lin_sum


    def evaluate_solution__check_solution_feasible(self, lin_dyn_error_tolerance=6) -> bool:
        valid = self.evaluate_solution__foot_forces_z_min() >= 0
        violation_lin_max, violation_lin_sum = self.evaluate_solution__max_dyn_lin_error()
        valid &= np.abs(violation_lin_max) <= lin_dyn_error_tolerance
        return valid


    def evaluate_solution__feet_in_contact(self, times):
        """
        Check if the feet are in contact at the given times.
        Note when time values in times are greate than the total duration, the corresponding values in will be false in the returned array.
        :param times: array of times
        :return: array of booleans
        """
        in_contact = np.zeros((2, times.shape[0]))
        for foot_i, foot in enumerate(self.x_opti_feet):
            contact_phases = list(foot.get_contact_phases())
            for contact_phase_i, contact_phase in enumerate(contact_phases):
                start_t = self.value(contact_phase.start_t)
                end_t = start_t + self.value(contact_phase.duration)

                # follow range definition in FootTrajectory.__evaluate_x_generic_func
                in_contact[foot_i] = np.logical_or(
                    in_contact[foot_i],
                    np.logical_and(times >= start_t, times < end_t)
                )
                #rich.print(f'foot {foot_i}', start_t, end_t)
            # handle last wraped phase of looping foot
            if foot_i == self.foot_id_with_looping_last_phase:
                phase = list(foot.get_contact_phases())[-1]
                # the last phase of a foot with looping is allways a contact phase
                assert phase.get_phase_type() == PhaseType.CONTACT
                start_t = 0.0
                end_t = self.value(foot.rolling_time_shift_duration)
                in_contact[foot_i] = np.logical_or(
                    in_contact[foot_i],
                    np.logical_and(times >= start_t, times < end_t)
                )

        return in_contact







    ###############################################################
    ## load and save parms and solution

    @staticmethod
    def create_solution_loader(opti_parameters: dict) -> 'CentroidalDynPolyOptiData':
        """
        Create CentroidalDynPolyOptiData(as_just_solution_loader=True) with all opti parameters from a given dict
        containing these parameters.
        :param opti_parameters: the opti parameters as dict (all parameters of CentroidalDynPolyOptiData.__init__(...)).
                                Get these parameter via opti.serialize_opti_parameters()
        """

        def without_keys(d, keys):
            return {x: d[x] for x in d if x not in keys}
        params = without_keys(opti_parameters, [
            'feet_last_phase_duration_implicit',
            'additional_costs_and_constraints_parameters',
            'num_feet'
        ])
        optiData = CentroidalDynPolyOptiData(as_just_solution_loader=True, **params)
        optiData.additional_costs_and_constraints_parameters = opti_parameters['additional_costs_and_constraints_parameters']
        optiData.feet_last_phase_duration_implicit = opti_parameters['feet_last_phase_duration_implicit']
        optiData.num_feet = opti_parameters['num_feet']
        return optiData

    @staticmethod
    def create_solution_loader_from_yaml(opti_parameters_yaml_file: str) -> 'CentroidalDynPolyOptiData':
        """
        Create CentroidalDynPolyOptiData(as_just_solution_loader=True) with all opti parameters from a given yaml file
        containing these parameters.
        :param opti_parameters_yaml_file: the opti parameters are in this yaml file
                                          (all parameters of CentroidalDynPolyOptiData.__init__(...)).
                                          Get these parameter via opti.serialize_opti_parameters_yaml()
        """
        with open(opti_parameters_yaml_file) as file:
            params = yaml.load(file, yaml.Loader)
            return CentroidalDynPolyOptiData.create_solution_loader(params)




    def serialize_opti_parameters(self):
        param_names = [
            "mass",
            "InertiaMatrix",
            "use_two_feet",
            "num_feet",

            "num_phases",
            "feet_first_phase_type",
            "foot_kin_constraint_box_size",
            "foot_kin_constraint_box_center_rel",

            "looping_last_contact_phase",
            "looping_last_contact_phase_time_shift_fixed",

            "use_angular_dynamics",
            "fixed_phase_durations",
            "phase_duration_min",
            "feet_last_phase_duration_implicit",

            "foot_force_max_z",
            "foot_force_at_trajectory_end_and_start_variable",
            "max_com_angle_xy_abs",

            "total_duration",
            "base_poly_duration",
            "num_polynomials_for_com_trajectory",

            "additional_foot_flight_smooth_vel_constraints",
            "additional_intermediate_foot_force_constraints",
            "additional_costs_and_constraints_parameters",
        ]
        # serialize to dict
        params = {}
        for k in param_names:
            params[k] = vars(self)[k]
        return params




    # member variables to serialize for saving solution
    _variables_to_serialize_for_solution_keys = [
        'param_opti_start_com_pos',
        'param_opti_start_com_dpos',
        'param_opti_start_com_angle',
        'param_opti_start_com_dangle',
        'param_opti_start_foot_pos',
        'param_opti_start_foot_dpos',
        'param_opti_end_com_pos',
        'x_opti_com_pos',
        'x_opti_com_angle',
        'x_opti_feet',
        'x_opti_feet__start_pos',
        'x_opti_rolling_time_shift_duration'
    ]
    _variables_to_serialize_for_constraint_points_keys = [
        "constraint_points_t__dynamics",
        "constraint_points_t__feet",
    ]


    def load_solution(self, solution_dict):
        """
        Load a solution (with parameters) from a dict that was created with .serialize_solution().
        """
        assert self.as_just_solution_loader, (
            "Solution can just be loaded when as_just_solution_loader is true. "
            "You should create an instance with CentroidalDynPolyOptiData.create_solution_loader(...)"
        )

        self.solution_converged_to_optimal = solution_dict['solution_converged_to_optimal']
        if 'solution_is_feasible' in solution_dict:
            self.solution_is_feasible = solution_dict['solution_is_feasible']
        self.solution_solver_stats = solution_dict['solution_solver_stats']

        constraint_points = solution_dict['constraint_points']
        self.constraint_points_t__dynamics = constraint_points['constraint_points_t__dynamics']

        # create feet constraint points
        if self.constraint_points_t__feet is None:
            self.constraint_points_t__feet = [FootConstraintTimePoints(self.opti) for f in self.x_opti_feet]
        # load feet constraint points
        for i, f in enumerate(self.constraint_points_t__feet):
            f.constraint_points_t__feet_pos = constraint_points['constraint_points_t__feet'][i]['constraint_points_t__feet_pos']
            f.constraint_points_t__feet_contact_force = constraint_points['constraint_points_t__feet'][i]['constraint_points_t__feet_contact_force']

        # load opti values
        SerializableOptiVariables.deserialize_opti_vars_load_solution_static(
            self,
            self._variables_to_serialize_for_solution_keys,
            solution_dict['values']
        )


    def serialize_solution(self) -> dict:
        """
        Dump solution (with parameters) to a dict so that it can be saved as yaml and loaded later.
        """
        d = {
            'solution_converged_to_optimal': self.solution_converged_to_optimal,
            'solution_is_feasible': self.solution_is_feasible,
            'solution_solver_stats': {
                key: self.solution_solver_stats[key] for key in [
                    'iter_count',
                    'return_status',
                    't_wall_total',
                    't_proc_total'
                ]
            }
        }
        # serialize to dict
        constraint_points = {
            'constraint_points_t__dynamics': self.constraint_points_t__dynamics,
            'constraint_points_t__feet': [f.serialize_opti_vars() for f in self.constraint_points_t__feet]
        }
        d['constraint_points'] = constraint_points
        solution_dict = SerializableOptiVariables.serialize_opti_vars_static(
            self,
            self._variables_to_serialize_for_solution_keys
        )
        d['values'] = solution_dict
        return d



    def load_solution_from_yaml(self, filename: str):
        with open(filename) as file:
            solution_dict = yaml.load(file, yaml.Loader)
            self.load_solution(solution_dict)

    def serialize_solution_to_yaml(self, filename: str):
        return yaml.dump(self.serialize_solution(), indent=2)








    ###############################################################
    ## plotting and vis


    def setup_vis3d(self, on_target_maker_changed_callback=None):
        assert self.viz3d is None
        self.viz3d = RobotAnimation3D(
            # for testing
            foot_kin_constraint_box_center_rel=self.foot_kin_constraint_box_center_rel,
            foot_kin_constraint_box_size=self.foot_kin_constraint_box_size,
            num_feet=self.num_feet,
        )
        self.viz3d.set_target_pos_changed_callback(on_target_maker_changed_callback)

    def plot_animate_all(self,
                         show_plots=True,
                         show_animate=True,
                         show_plots_angular_dyn=True,
                         show_plots_angular=False,
                         playback_once=False,
                         animate_show_new_window=False):
        # show results
        for i, foot in enumerate(self.x_opti_feet):
            print(f"\n## Foot {i}:")
            for p_i, p in enumerate(foot.phases):
                print(f'>> phase {p_i} duration {p.evaluate_solution_duration():5.3f}s ({p.get_phase_type()})')
            if self.foot_id_with_looping_last_phase == i:
                print(f'=>> looping phase shifting duration {self.value(self.x_opti_rolling_time_shift_duration)} ')


        if show_plots:
            # plot
            plot_com_and_foot_trajectories_xyz(
                self,
                x_opti_com_pos=self.x_opti_com_pos,
                x_opti_com_angle=self.x_opti_com_angle if self.use_angular_dynamics else None,
                x_opti_feet=self.x_opti_feet,
                constraints_t_dynamics=self.constraint_points_t__dynamics,
                constraints_t_feet=self.constraint_points_t__feet,
                show_non_blocking=False,
                show_angular_dyn=show_plots_angular_dyn,
                show_plots_angular=show_plots_angular
                # show_plot=False
            )


        # evaluate trajectories
        t_vals = np.arange(0, self.total_duration, step=0.01)
        x_vals_com_pos = self.x_opti_com_pos.evaluate_solution_x(t_vals)
        x_vals_com_angle = self.x_opti_com_angle.evaluate_solution_x(t_vals)
        #x_vals_com_angle = np.zeros((self.d, t_vals.shape[0]))

        x_vals_foot1 = self.x_opti_feet[0].evaluate_solution_x(t_vals)
        if self.num_feet > 1:
            x_vals_foot2 = self.x_opti_feet[1].evaluate_solution_x(t_vals)

        x_feet_pos_vals = []
        u_feet_force_vals = []
        for foot_i in range(self.num_feet):
            foot_trajectory = self.x_opti_feet[foot_i].evaluate_solution_x(t_vals)
            x_feet_pos_vals.append(foot_trajectory.foot_pos_trajectory.values)
            u_feet_force_vals.append(foot_trajectory.foot_force_trajectory.values)


        if show_animate:
            if self.viz3d is None:
                self.setup_vis3d()
            self.viz3d.animate_humanoid(
                t_vals,
                x_com_pos_vals=x_vals_com_pos,
                x_com_rotation_vals=x_vals_com_angle,  # np.zeros((self.d, t_vals.shape[0])),
                x_feet_pos_vals=x_feet_pos_vals,
                u_feet_force_vals=u_feet_force_vals,
                initially_start=True,
                once=playback_once,
                show_new_window=animate_show_new_window,
                loop_duration_s=self.total_duration
            )



    def print_all_opti_vars(self):
        print_all_opti_vars(self.opti)