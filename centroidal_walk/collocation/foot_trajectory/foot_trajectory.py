from typing import Callable

import numpy as np

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases import Phase, ContactPhase, FlightPhase
from centroidal_walk.collocation.foot_trajectory.trajectory_eval_structs import *
from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import OptiWithSimpleBounds
from centroidal_walk.collocation.spline.spline_trajectory import *
from centroidal_walk.collocation.serialization.yaml_util import *


@serializable_opti_variables([
    'opti_start_foot_force',
    'opti_start_foot_dforce',
    'opti_end_foot_force',
    'opti_end_foot_dforce',
    'phase_durations',
    'phases'
])
class FootTrajectory(SerializableOptiVariables):
    """
    Foot trajectory consisting of foot pos and force.
    Consists of multiple phases.
    For each phase the position/force is either constant or represented by chain of polynomials (spline).
    When in a phase the force is constant zero the pos is represented by a spline.
    When in a phase pos is constant the force is represented by a spline.
    """
    opti: OptiWithSimpleBounds
    d: int
    num_polynomials_per_non_const_phase: int
    num_phases: int


    # opti parameters: start state
    param_opti_start_foot_pos: casadi.Opti.parameter
    param_opti_start_foot_dpos: casadi.Opti.parameter

    # variable foot force and start of first contact phase and end of last contact phase
    opti_start_foot_force: casadi.Opti.variable
    opti_start_foot_dforce: casadi.Opti.variable
    opti_end_foot_force: casadi.Opti.variable
    opti_end_foot_dforce: casadi.Opti.variable

    # contact and flight phases
    phases: list[Phase]
    phase_durations: casadi.Opti.variable


    def __init__(self,
                 num_polynomials_foot_pos_in_flight,
                 num_polynomials_foot_force_in_contact,
                 num_phases,
                 param_opti_start_foot_pos, param_opti_start_foot_dpos,
                 #param_opti_start_foot_force, param_opti_start_foot_dforce,
                 opti: OptiWithSimpleBounds,
                 total_duration: float,
                 rolling_time_shift_duration: Opti.variable = None,
                 base_euler_angle_difference_end_to_start=None,
                 first_phase_type: PhaseType = PhaseType.CONTACT,
                 dimensions=2,
                 phase_duration_min=0,
                 phase_duration_max=1000,
                 contact_phase_duration_min=None,
                 fixed_phase_durations: np.ndarray = None,
                 foot_force_at_trajectory_end_and_start_variable=True,
                 last_phase_duration_implicit=False,
                 scope_name: list[str] = []
                 ):
        """
        Create spline consisting of multiple polynomials. The spline parameters are optimizable.
        Consistency at the spline boundaries is ensured for x and dx.

        :param total_duration:       all phase durations have to sum up to this values
        :param num_polynomials_for_trajectory: number of polynomials that make the spline
        :param num_phases:       number of gait phases to create
        :param param_opti_start_x: start boundary value x for first polynomial
        :param param_opti_start_dx: start boundary value dx for first polynomial
        :param opti:
        :param given_total_duration: total duration of the foot movement.
        :param last_phase_duration_implicit: then the last phase duration is not an optimization parameter but
                                            total_duration - (sum of all previous phases)
        :param rolling_time_shift_duration: shift all phases by this duration backwards.
         The time the last phase is shifted past the total duration, this last phase reappears at the beginning.
         Thereby the last phase needs to be a contact phase.
        :param base_euler_angle_difference_end_to_start: set this to (base_final_angle - base_initial_angle).
                                                        Just required when rolling_time_shift_duration is set.
        """
        self.num_polynomials_foot_pos_in_flight = num_polynomials_foot_pos_in_flight
        self.num_polynomials_foot_force_in_contact = num_polynomials_foot_force_in_contact
        self.num_phases = num_phases
        self.first_phase_type = first_phase_type
        self.param_opti_start_foot_pos = param_opti_start_foot_pos
        self.param_opti_start_foot_dpos = param_opti_start_foot_dpos
        #self.param_opti_start_foot_force = param_opti_start_foot_force
        #self.param_opti_start_foot_dforce = param_opti_start_foot_dforce
        self.opti = opti
        self.total_duration = total_duration
        self.d = dimensions
        self.phase_duration_min = phase_duration_min
        self.phase_duration_max = phase_duration_max
        self.contact_phase_duration_min = contact_phase_duration_min
        self.fixed_phase_durations = fixed_phase_durations
        self.foot_force_at_trajectory_end_and_start_variable = foot_force_at_trajectory_end_and_start_variable
        self.last_phase_duration_implicit = last_phase_duration_implicit
        self.scope_name = scope_name#


        last_phase_is_contact = self.infer_if_last_phase_is_contact_phase(first_phase_type, num_phases)

        self.rolling_time_shift_duration = rolling_time_shift_duration
        self.base_euler_angle_difference_end_to_start = base_euler_angle_difference_end_to_start
        if rolling_time_shift_duration is not None:
            # assert self.fixed_phase_durations is None, "currently just supported with optimized durations"
            assert last_phase_is_contact, ("with rolling_time_shift_duration last phase has to be a contact phase."
                                           "Since foot positions are expressed globally, warping makes no sense for those.")
            print('>> use rolling_time_shift_duration: CAUTION just makes sense when all start and end rotations and feet positions/forces are the same. \n'
                  '\tSince the last foot contact phase will be wrapped (thus the contact forces).')
            assert base_euler_angle_difference_end_to_start is not None
            # if its opti variable
            if not np.isscalar(rolling_time_shift_duration):
                shift_min = 0.0
                self.opti.subject_to_var_bounds(shift_min, rolling_time_shift_duration, self.total_duration)
                # improves solving
                opti.subject_to(rolling_time_shift_duration >= shift_min)


        # start and end opti values
        self.opti_start_foot_force = None
        self.opti_start_foot_dforce = None
        self.opti_end_foot_force = None
        self.opti_end_foot_dforce = None
        if self.foot_force_at_trajectory_end_and_start_variable:
            if first_phase_type == PhaseType.CONTACT:
                # first contact phase has non-zero force at start
                self.opti_start_foot_force = self.opti.variable(self.d)
                self.opti_start_foot_dforce = self.opti.variable(self.d)
                describe_variable(self.opti, self.opti_start_foot_force,
                                  var_name='foot_trajectory__opti_start_foot_force', scope=scope_name + ['foot_force'])
                describe_variable(self.opti, self.opti_start_foot_dforce,
                                  var_name='foot_trajectory__opti_start_foot_dforce', scope=scope_name + ['foot_force'])
            if last_phase_is_contact:
                # last contact phase has non-zero force at end
                self.opti_end_foot_force = self.opti.variable(self.d)
                self.opti_end_foot_dforce = self.opti.variable(self.d)
                describe_variable(self.opti, self.opti_end_foot_force, var_name='foot_trajectory__opti_end_foot_force', scope=scope_name + ['foot_force'])
                describe_variable(self.opti, self.opti_end_foot_dforce, var_name='foot_trajectory__opti_end_foot_dforce', scope=scope_name + ['foot_force'])

        # phase durations
        if self.fixed_phase_durations is None:
            self.phase_durations = self.opti.variable(
                self.num_phases
                - (1 if last_phase_duration_implicit else 0)
            )
            describe_variable(self.opti, self.phase_durations, var_name='phase_durations', scope=scope_name)



        # phases
        self.create_phases()

        # ensure that all phase duration sum up to given total_duration
        if self.fixed_phase_durations is None:
            self.opti.subject_to(self.get_total_duration() == self.total_duration)
        #else:
        #    assert self.get_total_duration() == self.total_duration, f"got get_total_duration={self.get_total_duration()} but should be {self.total_duration}"


    @staticmethod
    def infer_if_last_phase_is_contact_phase(first_phase_type, num_phases):
        return (first_phase_type == PhaseType.CONTACT and num_phases % 2 == 1) or (
                                 first_phase_type == PhaseType.FLIGHT and num_phases % 2 == 0
        )


    def create_phases(self):
        print(f"create phases of {self.scope_name} ...")
        last_foot_pos = self.param_opti_start_foot_pos
        last_foot_dpos = self.param_opti_start_foot_dpos
        #last_foot_force = self.param_opti_start_foot_force  # @todo: this does not make sense, force and beginning and end is allways zero
        #last_foot_dforce = self.param_opti_start_foot_dforce

        # rolling time shift
        time_shift = self.rolling_time_shift_duration if self.rolling_time_shift_duration is not None else 0.0
        last_end_t = MX.zeros(1) + time_shift

        # assert self.num_phases % 2 == 1, "need an odd number of phases, since the first and last phase is allways a constact phase"

        # phase type based on phase index
        def phase_type_is_contact_from_index(index):
            first = index % 2 == 0
            if self.first_phase_type == PhaseType.CONTACT:
                return first
            else:
                return not first



        self.phases: list[Phase] = []
        for i in range(self.num_phases):
            phase = None
            is_first_phase = i == 0
            is_last_phase = i == self.num_phases-1

            # handle case where phase durations are fixed
            phase_duration = None
            if self.fixed_phase_durations is not None:
                phase_duration = self.fixed_phase_durations[i]
            else:
                # when phase duration are variable
                if not is_last_phase:
                    phase_duration = self.phase_durations[i]
                # handle last phase separate
                else:
                    if self.last_phase_duration_implicit:
                        phase_duration = self.total_duration - self.get_sum_of_phase_durations_excluding_last_phase()
                        print('####', phase_duration)
                    # last phase duration explicit (own opti var)
                    else:
                        phase_duration = self.phase_durations[i]


            # contact
            if phase_type_is_contact_from_index(i):
                print(f"> phase {i} is a contact phase")

                # the force of the previous flight phase will be zero, so at the start of this phase it is zero too
                start_foot_force = np.zeros(self.d)
                start_foot_dforce = np.zeros(self.d)
                # but for the first contact phase the force can be non-zero at the start (there is no previous flight phase)
                if is_first_phase and self.foot_force_at_trajectory_end_and_start_variable:
                    start_foot_force = self.opti_start_foot_force
                    start_foot_dforce = self.opti_start_foot_dforce
                # the force in next flight phase will be zero, so end the end of this phase it is zero too
                end_foot_force = np.zeros(self.d)
                end_foot_dforce = np.zeros(self.d)
                # but for the last contact phase the force can be non-zero at the end (there is no next flight phase)
                if is_last_phase and self.foot_force_at_trajectory_end_and_start_variable:
                    end_foot_force = self.opti_end_foot_force
                    end_foot_dforce = self.opti_end_foot_dforce

                phase = ContactPhase(
                    self.num_polynomials_foot_force_in_contact,
                    start_t=last_end_t,
                    start_foot_pos=last_foot_pos,
                    start_foot_dpos=None,   # foot pos constant
                    start_foot_force=start_foot_force,
                    start_foot_dforce=start_foot_dforce,
                    end_foot_force=end_foot_force,
                    end_foot_dforce=end_foot_dforce,
                    opti=self.opti,
                    dimensions=self.d,
                    phase_duration=phase_duration,  # is None by default, then duration is flexible
                    scope_name=self.scope_name + [f'phase_{i}']
                )
                # at the end just new force, position was constant
                last_foot_force = phase.foot_force.poly_list[-1].x1
                last_foot_dforce = phase.foot_force.poly_list[-1].dx1

            # fight
            else:
                print(f"> phase {i} is a flight phase")
                phase = FlightPhase(
                    self.num_polynomials_foot_pos_in_flight,
                    start_t=last_end_t,
                    start_foot_pos=last_foot_pos,
                    start_foot_dpos=last_foot_dpos,
                    start_foot_force=None,  # foot force is constant zero
                    start_foot_dforce=None,
                    opti=self.opti,
                    dimensions=self.d,
                    phase_duration=phase_duration,  # is None by default, then duration is flexible
                    scope_name=self.scope_name + [f'phase_{i}']
                )
                # at the end just new position, force was constant
                last_foot_pos = phase.foot_position.poly_list[-1].x1
                last_foot_dpos = phase.foot_position.poly_list[-1].dx1

            # constrain phase duration when these are not fixed
            if self.fixed_phase_durations is None:
                use_var_bounds = True
                duration_min, duration_max = self.__get_phase_duration_range(phase)
                #self.opti.subject_to(self.opti.bounded(self.phase_duration_min, phase.duration, self.phase_duration_max))
                if (not (self.last_phase_duration_implicit and is_last_phase)) and use_var_bounds:  # skip last if is implicit
                    if isinstance(self.opti, OptiWithSimpleBounds):
                        self.opti.subject_to_var_bounds(duration_min, phase.duration, duration_max)
                    else:
                        print('>> WARN: not using subject_to_var_bounds since opti is not OptiWithSimpleBounds')
                        self.opti.subject_to(self.opti.bounded(duration_min, phase.duration, duration_max))
                else:
                    self.opti.subject_to(
                        self.opti.bounded(duration_min, phase.duration, duration_max))

            # add new phase
            self.phases.append(phase)
            last_end_t = last_end_t + phase.duration

        assert len(self.phases) == self.num_phases

        if self.rolling_time_shift_duration is not None and (self.fixed_phase_durations is None
                                                             or not np.isscalar(self.rolling_time_shift_duration)):
            # assume that rolling_time_shift_duration can at maximum shift last phase to the start
            self.opti.subject_to(self.rolling_time_shift_duration < self.phases[-1].duration)
            print('>> add rolling_time_shift_duration shorter than last phase duration constraint')
      
    
    
    def __get_phase_duration_range(self, phase):
        duration_min = self.phase_duration_min
        duration_max = self.phase_duration_max
        if phase.get_phase_type() == PhaseType.CONTACT:
            if self.contact_phase_duration_min is not None:
                duration_min = min(duration_min, self.contact_phase_duration_min)
        return duration_min, duration_max



    def get_sum_of_phase_durations_excluding_last_phase(self):
        duration = MX.zeros(1)
        for i in range(self.num_phases - 1):
            print(i)
            duration = duration + self.phase_durations[i]
        return duration


    def get_total_duration(self):
        duration = MX.zeros(1) #if self.fixed_phase_durations is None else 0.0
        for p in self.phases:
            duration = duration + p.duration
        return duration


    def set_initial_opti_values(self,
                                init_for_duration_of_each_phase: np.ndarray=1,
                                init_durations_contact_phases: np.ndarray = None,
                                init_durations_flight_phases: np.ndarray = None,
                                init_z_height_for_flight=2,
                                init_z_dheight_for_flight=1,
                                init_z_force_for_contact=10,
                                init_z_dforce_for_contact=10,
                                interpolation_xy_start=None,
                                interpolation_xy_end=None,
                                use_interpolted_dx_for_foot_pos=True
                                ):
        """
        Set initial values for opti parameters.
        This will initialize the phase duration parameters to non-zero, otherwise we would get division by zero errors.
        Note that the first phase is allways a contact phase.

        :param init_for_duration_of_each_phase: A single value, all phases will have equal duration.
                                                When init_durations_contact_phases or init_durations_flight_phases is also specified.
                                                The corresponding phases will use the values of these arrays.
        :param init_durations_contact_phases:   array of initial durations of the contact phases.
        :param init_durations_flight_phases:    array of initial durations of the flight phases.
        """
        if interpolation_xy_start is None:
            interpolation_xy_start = np.zeros(self.d-1)
        if interpolation_xy_end is None:
            interpolation_xy_end = np.zeros(self.d-1)
        delta_pos_xy_per_flight_phase = (interpolation_xy_end - interpolation_xy_start) / (len(list(self.get_flight_phases())))

        # phase durations
        if self.fixed_phase_durations is None:
            if init_durations_contact_phases is None:
                # math.ceil(self.num_phases/2)
                init_durations_contact_phases = np.ones(len(list(self.get_contact_phases()))) * init_for_duration_of_each_phase
            if init_durations_flight_phases is None:
                init_durations_flight_phases = np.ones(len(list(self.get_flight_phases()))) * init_for_duration_of_each_phase

            # assert self.phases[-1].get_phase_type() == PhaseType.CONTACT, "last phase needs to be a contact phase"

            # init deltaT
            i_contact_phase = 0
            i_flight_phase = 0
            for i, phase in enumerate(self.phases):
                # when implicit last phase duration is not optimized
                if self.last_phase_duration_implicit and i == len(self.phases) - 1:
                    break
                if phase.get_phase_type() == PhaseType.CONTACT:
                    self.opti.set_initial(phase.duration, init_durations_contact_phases[i_contact_phase])
                    i_contact_phase += 1
                elif phase.get_phase_type() == PhaseType.FLIGHT:
                    self.opti.set_initial(phase.duration, init_durations_flight_phases[i_flight_phase])
                    i_flight_phase += 1


        # flight phases
        last_xy_pos = interpolation_xy_start
        for phase_i, phase in enumerate(self.get_flight_phases()):
            middle_poly_i = len(phase.foot_position.poly_list)/2.0

            # for i, poly in enumerate(p.foot_position.poly_list):
            #     # just set x1 values
            #     # skip last since it has to be zero
            #     if i < len(p.foot_position.poly_list)-1:
            #         init_x1 = np.zeros(self.d)
            #         init_x1[-1] = init_z_height_for_flight
            #         init_x1[0:2] =  last_xy_pos
            #         self.opti.set_initial(poly.x1, init_x1)
            #
            #         # set derivative
            #         init_dx1 = np.zeros(self.d)
            #         init_dx1[-1] = init_z_dheight_for_flight * (1 if i+1 < middle_poly_i else -1)
            #         self.opti.set_initial(poly.dx1, init_dx1)
            #         #print('dx_init_with_sign', init_dx1[2])
            flight_phase_duration = phase.duration if self.fixed_phase_durations is not None else init_durations_flight_phases[phase_i]
            phase.foot_position.set_initial_opti_values__x1_z(
                initial_x1_z_middle=init_z_height_for_flight,
                init_total_duration=flight_phase_duration,
                interpolation_xy_start=last_xy_pos,
                interpolation_xy_end=last_xy_pos + delta_pos_xy_per_flight_phase,
                use_interpolted_dx=use_interpolted_dx_for_foot_pos
            )
            #print("## End: ", last_xy_pos + delta_pos_xy_per_flight_phase)
            last_xy_pos += delta_pos_xy_per_flight_phase

        # force init value
        init_force_x = np.zeros(self.d)
        init_force_x[-1] = init_z_force_for_contact

        # contact phases
        contact_phases = list(self.get_contact_phases())
        for phase in contact_phases:
            middle_poly_i = len(phase.foot_force.poly_list)/2.0

            for i, poly in enumerate(phase.foot_force.poly_list):
                # just set x1 values
                # skip last since it has to be zero
                if i < len(phase.foot_force.poly_list)-1:
                    self.opti.set_initial(poly.x1, init_force_x)

                    # set derivative
                    init_dx1 = np.zeros(self.d)
                    init_dx1[-1] = init_z_dforce_for_contact * (1 if i+1 < middle_poly_i else -1)
                    self.opti.set_initial(poly.dx1, init_dx1)

        # first and last poly of force
        # just if force at end and start of trajectory is variable
        if self.foot_force_at_trajectory_end_and_start_variable:
            if self.phases[0].get_phase_type() == PhaseType.CONTACT:
                # x0, dx0 of first poly of first phase can be non-zero
                self.opti.set_initial(contact_phases[0].foot_force.poly_list[0].x0, init_force_x)
                self.opti.set_initial(contact_phases[0].foot_force.poly_list[0].dx0, np.zeros(self.d))

            if self.phases[-1].get_phase_type() == PhaseType.CONTACT:
                # x1, dx1 of last poly of last phase can be non-zero
                self.opti.set_initial(contact_phases[-1].foot_force.poly_list[-1].x1, init_force_x)
                self.opti.set_initial(contact_phases[-1].foot_force.poly_list[-1].dx1, np.zeros(self.d))



    def __evaluate_x_generic_func(self, t, func_to_evaluate,
                                  fn_map_last_poly_looped_vales: Callable[[float], float] = lambda values: values):
        """
        Get trajectory value of 'func_to_evaluate' for a given time t (t can also be an array of time values).
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        :param func_to_evaluate(p, t) takes time as parameter and gives an SX/MX expression as output, is member function of Phase
        :param fn_map_last_poly_looped_vales: map the values of the last poly which are looped back to the trajectory beginning.
                                        (poly values) -> values.
        """
        time_shift = self.rolling_time_shift_duration if self.rolling_time_shift_duration is not None else 0.0

        # reshape to t to shape (self.d, #num_time_values)
        t = Polynomial3.repeat_t_in_first_dim(t, self.d)

        x_vals = MX.zeros((self.d, 1)) #t.shape[0]))
        t_start = MX.zeros(self.d) + time_shift # repeated in first dim
        for p in self.phases:
            t_end = t_start + p.duration
            # special case for last phase when rolling_time_shift_duration is used
            # t_end may be larger than total_duration
            if p == self.phases[-1] and self.rolling_time_shift_duration is not None:
                t_end = self.total_duration

            x_vals = x_vals + SplineTrajectory.val_if_in_range(t_start, t_end, t, value=func_to_evaluate(p, t - t_start))
            #x_vals = x_vals + func_to_evaluate(p, t - t_start)
            t_start = t_start + p.duration


        # special case for last phase when rolling_time_shift_duration is used
        # map end of last phase to start
        if self.rolling_time_shift_duration is not None:
            phase = self.phases[-1]
            t_start = t + phase.duration - self.rolling_time_shift_duration
            t_end = self.rolling_time_shift_duration
            x_vals = x_vals + SplineTrajectory.val_if_in_range(
                0.0, t_end, t,
                value=fn_map_last_poly_looped_vales(func_to_evaluate(phase, t_start))
            )

        return x_vals


    def evaluate_foot_pos(self, t):
        """
        Get foot position value for a given time t (t=0 is at the spline start).
        Use this for constraints during optimization.
        t can be a scalar or array.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        """
        return self.__evaluate_x_generic_func(
            t, lambda phase, t : phase.evaluate_foot_pos(t),
            # foot pos before first poly matches starting foot pos
            fn_map_last_poly_looped_vales=lambda values: self.phases[0].evaluate_foot_pos_start()
        )

    def evaluate_foot_force(self, t):
        """
        Get foot force value for a given time t (t=0 is at the spline start).
        Use this for constraints during optimization.
        t can be a scalar or array.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        """
        def rotate_looped_last_poly_force(force_value):
            if self.rolling_time_shift_duration is not None:
                # @todo check this:
                return R_world_frame_to_com_frame_euler(self.base_euler_angle_difference_end_to_start) @ force_value
            else:
                return force_value
        return self.__evaluate_x_generic_func(
            t, lambda phase, t : phase.evaluate_foot_force(t),
            # foot force before first poly needs to be rotated based on orientation difference between start end of the base
            fn_map_last_poly_looped_vales=rotate_looped_last_poly_force
        )



    def get_contact_phases(self) -> list[ContactPhase]:
        return filter(lambda p: p.get_phase_type() == PhaseType.CONTACT, self.phases)

    def get_flight_phases(self) -> list[FlightPhase]:
        return filter(lambda p: p.get_phase_type() == PhaseType.FLIGHT, self.phases)



    ######################################################
    ## plotting and evaluation

    def evaluate_solution_x(self, t: np.array, use_individual_poly_evaluation_functions=False) -> FootTrajectorySolutionEvaluated:
        """
        Get solution (for foot pos and force) from solver and evaluate it for time(s) t.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :param t:
        :param use_individual_poly_evaluation_functions: evaluate each polynomial individually with the soluton values using numpy.
                                                            This may deviate from the real solution, better set this to False.
                                                            If set to false this just apply to trajectory.##.values
                                                            and not to trajectory.##.spline_connection_points
        :return:
        """
        #total_duration = self.opti.value(self.get_total_duration())
        #assert not np.any(np.logical_or(t < 0, t > total_duration)), \
        #        f"t has to be in range [0, total_duration={total_duration}]"
        num_time_points = t.shape[0]
        trajectory = FootTrajectorySolutionEvaluated()
        trajectory.foot_pos_trajectory = PhaseTrajectoryEvaluated(
            np.zeros((self.d, num_time_points)),
            SplineTrajectory.EvalSplineConnectionPoints(self.d),  # empty points list
        )
        trajectory.foot_force_trajectory = PhaseTrajectoryEvaluated(
            np.zeros((self.d, num_time_points)),
            SplineTrajectory.EvalSplineConnectionPoints(self.d),  # empty points list
        )
        trajectory.phase_types=[]
        trajectory.phase_end_times=np.zeros(self.num_phases)
        #trajectory.phase_end_times[0] = 0


        def handle_phase(phase, i, start_t, t_vals, t_max_duration, end_t, t_shift=None, start_t_insert_at=None):
            nonlocal trajectory
            insert_at = start_t
            if start_t_insert_at is not None:
                insert_at = start_t_insert_at

            p_pos = phase.evaluate_solution_foot_pos(t_vals, t_max_duration=t_max_duration)
            if use_individual_poly_evaluation_functions:
                trajectory.foot_pos_trajectory.values += p_pos.values
            trajectory.foot_pos_trajectory.spline_connection_points.append(p_pos.spline_connection_points, start_t=insert_at)

            p_force = phase.evaluate_solution_foot_force(t_vals, t_max_duration=t_max_duration)
            if use_individual_poly_evaluation_functions:
                trajectory.foot_force_trajectory.values += p_force.values
            trajectory.foot_force_trajectory.spline_connection_points.append(p_force.spline_connection_points, start_t=insert_at)

            trajectory.phase_end_times[i] = end_t

            if start_t_insert_at is None:
                trajectory.phase_types.append(phase.get_phase_type())

        rolling_time_shift_duration_evaluated = None if self.rolling_time_shift_duration is None else self.opti.value(self.rolling_time_shift_duration)
        time_shift = rolling_time_shift_duration_evaluated if self.rolling_time_shift_duration is not None else 0.0
        start_t = 0 + time_shift
        # for all phases
        for i, phase in enumerate(self.phases):
            handle_phase(phase, i,
                         start_t=start_t,
                         end_t=start_t + phase.evaluate_solution_duration(),
                         t_vals=t-start_t,
                         t_max_duration=self.total_duration-start_t)
            start_t += phase.evaluate_solution_duration()

        # special case for last phase when rolling_time_shift_duration is used
        # map end of last phase to start
        if self.rolling_time_shift_duration is not None:
            phase = self.phases[-1]
            t_start = self.opti.value(phase.duration) - rolling_time_shift_duration_evaluated
            t_end = rolling_time_shift_duration_evaluated
            # @todo make this work when start and end body rotation is not the same -> rotate forces by rotation diff of start and end
            handle_phase(phase,
                         i=len(self.phases)-1,
                         start_t=t_start,
                         end_t=t_end,
                         t_vals=t+t_start,
                         t_max_duration=self.opti.value(phase.duration),
                         start_t_insert_at=-t_start)

        # use the existing eval functions of this calls if requested
        if not use_individual_poly_evaluation_functions:
            trajectory.foot_pos_trajectory.values = self.opti.value(self.evaluate_foot_pos(t))
            trajectory.foot_force_trajectory.values = self.opti.value(self.evaluate_foot_force(t))

        return trajectory



    @staticmethod
    def plot_evaluated_foot_trajectory(t_vals, trajectory: FootTrajectorySolutionEvaluated,
                                       plot_force_or_pos='pos', only_plot_dimensions: np.s_ = None,
                                       axis=None
                                       ):
        if axis is None:
            axis = plt

        trajectory_to_use = trajectory.foot_pos_trajectory
        color = None
        if plot_force_or_pos == 'force':
            trajectory_to_use = trajectory.foot_force_trajectory
            color = 'C3'

        if only_plot_dimensions is None:
            only_plot_dimensions = np.s_[0:trajectory_to_use.values.shape[0]]
        if not isinstance(only_plot_dimensions, int):
            num_dims = (only_plot_dimensions.end - only_plot_dimensions.start) / (
                        only_plot_dimensions.step if only_plot_dimensions.step is not None else 1)
        else:
            # if only_plot_dimensions is scalar
            num_dims = 1
        print("num_dims", num_dims)
            
        axis.plot(t_vals, trajectory_to_use.values[only_plot_dimensions, :].T, label=plot_force_or_pos, color=color)
        # axis.plot(t_vals, x_trajectory_evaluated.foot_force_trajectory.values[0, :], label='np')
        axis.scatter(trajectory_to_use.spline_connection_points.t_values
                        .reshape(-1, 1)
                        .repeat(num_dims, axis=1),
                    trajectory_to_use.spline_connection_points.x_values[only_plot_dimensions, :].T,
                    # .swapaxes(1, 0) # make time first axis
                     color=color
                    )

        y_max = np.max(trajectory_to_use.values[only_plot_dimensions])
        y_min = np.min(trajectory_to_use.values[only_plot_dimensions])

        # plot phase types
        axis.vlines(x=0, ymin=y_min, ymax=y_max, colors='grey', linestyles='dashed')

        last_phase_end_t = 0
        for i, phase_end_t in enumerate(trajectory.phase_end_times):
            axis.vlines(x=phase_end_t, ymin=y_min, ymax=y_max, colors='grey', linestyles='dashed')
            if trajectory.phase_types[i] == PhaseType.CONTACT:
                axis.fill_between([last_phase_end_t, phase_end_t], y1=y_min, y2=y_max, color='black', alpha=0.2)
            last_phase_end_t = phase_end_t
        axis.legend()
