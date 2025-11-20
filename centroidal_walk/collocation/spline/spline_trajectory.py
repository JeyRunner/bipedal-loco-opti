from builtins import staticmethod

from centroidal_walk.collocation.spline.polynomial3 import Polynomial3
from centroidal_walk.collocation.opti.centroidal_dyn_poly import *
from centroidal_walk.collocation.spline.polynomial4 import Polynomial4
from centroidal_walk.collocation.opti.casadi_util.opti_util import *
from centroidal_walk.collocation.serialization.yaml_util import *

@serializable_opti_variables([
    'x_opti_vars',
    'x_opti_vars_last_poly_x',
    'x_opti_vars_last_poly_dx',
    'x_opti_vars_last_poly_deltaT'
])
class SplineTrajectory(SerializableOptiVariables):
    """
    Chain of polynomials to build a trajectory.
    """
    opti: casadi.Opti
    d: int #= 2
    num_polynomials_for_trajectory: int


    # opti parameters: start state
    param_opti_start_x: casadi.Opti.parameter
    param_opti_start_dx: casadi.Opti.parameter

    # opti variables
    x_opti_vars: casadi.Opti.variable
    x_opti_vars_last_poly_x: casadi.Opti.variable # last poly may have less free variables than others (as final x, dx may be fixed)
    x_opti_vars_last_poly_dx: casadi.Opti.variable # last poly may have less free variables than others (as final x, dx may be fixed)
    x_opti_vars_last_poly_deltaT: casadi.Opti.variable

    poly_list: list[Polynomial3]  # list of polynomials that represent the trajectory

    # indices for a 3d polynomial represented as flat variable array: x_opti_vars
    # -> now part of polynomial class
    #id_poly_x0 = get_range_for_dimensional_index(0, d)
    #id_poly_dx0 = get_range_for_dimensional_index(1, d)
    id_poly_x1: np.s_
    id_poly_dx1: np.s_
    id_poly_deltaT: np.s_  # the time is just 1d
    id_poly_ddx0: np.s_


    def __init__(self,
                 num_polynomials_for_trajectory,
                 param_opti_start_x, param_opti_start_dx,
                 opti: casadi.Opti,
                 spline_polynomial_degree=3,
                 ddx_consistency_constraint=False,
                 ddx_consistency_constraint_manually=False,
                 param_opti_end_x=None, param_opti_end_dx=None,
                 dx1_z_fixed_zero=False,
                 given_total_duration=None,
                 rolling_time_shift: float = None,
                 given_total_duration_portion_deltaT_for_first_and_last_poly: float = None,
                 _debug__casadi_ifelse_nested=False,
                 dimensions=2,
                 scope_name=''
                 ):
        """
        Create spline consisting of multiple polynomials. The spline parameters are optimizable.
        Consistency at the spline boundaries is ensured for x and dx.

        :param num_polynomials_for_trajectory: number of polynomials that make the spline
        :param spline_polynomial_degree: can have value 3 or -> use Polynomial3 or Polynomial
        :param param_opti_start_x: start boundary value x for first polynomial
        :param param_opti_start_dx: start boundary value dx for first polynomial
        :param ddx_consistency_constraint: enforce ddx consistency constraint at polynomial boundaries
                                            (ddx at end of previous same as at beginning of next).
        :param ddx_consistency_constraint_manually: enforce ddx consistency constraint via opt constraint
                                                    and not implicitly via parametrization of splines.
        :param dx1_z_fixed_zero:        when true the z component of all polynomial dx1 values (except for the last dx1)
                                        will have a fixed zero value and will not be optimized.
        :param given_total_duration: optional external parameter for deltaT for the polynomials.
                        When deltaT is not given, each polynomial has its own optimizable deltaT.
                        Otherwise, the deltaT of each polynomial is fixed to the given 'given_total_duration/num_polynomials_for_trajectory'.
                        Note that a given deltaT can also be a casadi Opti variable.
        :param rolling_time_shift: will shift the whole trajectory to the right in time, the part that is shifted beyond
                                    the total time will reappear on the left side.
        :param given_total_duration_portion_deltaT_for_first_and_last_poly:
                                    part of the given_total_duration the first and last poly get as deltaT.
                                    The other polynomial get the rest.
        """
        SerializableOptiVariables.__init__(self, opti)

        assert spline_polynomial_degree == 3 or spline_polynomial_degree == 4
        assert (not ddx_consistency_constraint) or ddx_consistency_constraint and (
                (spline_polynomial_degree == 3 and ddx_consistency_constraint_manually) or
                (spline_polynomial_degree == 4)
        ), "for polynomial3 ddx_consistency_constraint can just be enforced manually (via solver constraint)"

        self.spline_polynomial_degree = spline_polynomial_degree
        self.num_polynomials_for_trajectory = num_polynomials_for_trajectory
        self.param_opti_start_x = param_opti_start_x
        self.param_opti_start_dx = param_opti_start_dx
        self.ddx_consistency_constraint = ddx_consistency_constraint
        self.ddx_consistency_constraint_manual = ddx_consistency_constraint_manually
        self.opti = opti
        self.given_total_duration = given_total_duration
        self. given_total_duration_portion_deltaT_for_first_and_last_poly = given_total_duration_portion_deltaT_for_first_and_last_poly
        self._debug__casadi_ifelse_nested = _debug__casadi_ifelse_nested
        self.d = dimensions

        # slices for opti variables
        need_ddx0_variables = spline_polynomial_degree == 4 and (not (ddx_consistency_constraint and not ddx_consistency_constraint_manually))
        id_last = 0
        self.id_poly_x1 = get_range_for_dimensional_index(0, self.d, True)
        id_last = self.d
        # dx1 may just have 2 entries per poly when dx1_z_fixed_zero is true
        self.dx1_num_vars = self.d if not dx1_z_fixed_zero else self.d - 1
        self.id_poly_dx1 = np.s_[id_last : id_last + self.dx1_num_vars, :]
        id_last += self.dx1_num_vars
        if need_ddx0_variables:
            self.id_poly_ddx0 = np.s_[id_last : id_last + self.d, :]
            id_last += self.d
        #index_last = 3 if need_ddx0_variables else 2
        self.id_poly_deltaT = np.s_[id_last: id_last + 1, :]  # the time is just 1d


        # create opti variables
        # note that polynomial3 and polynomial4 both have the same number of variables (ddx0 via continuity condition)
        num_vars_per_poly = (
            #Polynomial3.get_required_num_parameters_for_dim(self.d, deltaT_variable=(given_total_duration is None))
            self.d + self.dx1_num_vars
            + (1 if (given_total_duration is None) else 0)
            # for ddx at polynomial3
            + (self.d if need_ddx0_variables else 0)
        )
        self.x_opti_vars = self.opti.variable(
            num_vars_per_poly,
            self.num_polynomials_for_trajectory-1 # last polynomial has separate parameters
        )
        #self.opti.update_user_dict(self.x_opti_vars, {'var_name': 'spline_trajectory_x_opti_vars', 'scope': scope_name})


        # last poly number of variables depends on if last x,dx are given
        self.x_opti_vars_last_poly_x  = self.opti.variable(self.d) if param_opti_end_x is None else None
        self.x_opti_vars_last_poly_dx = self.opti.variable(self.d) if param_opti_end_dx is None else None
        self.x_opti_vars_last_poly_deltaT = self.opti.variable(1) if given_total_duration is None else None

        # for polynomial4 we need an additional variable for the ddx0 of the first polynomial
        self.x_opti_vars_first_poly_ddx0 = self.opti.variable(self.d) if spline_polynomial_degree == 4 else None

        self.poly_list: list[Polynomial3] = []
        self.create_polynomials(self.param_opti_start_x, self.param_opti_start_dx, param_opti_end_x, param_opti_end_dx, given_total_duration)


        # add description for variables
        describe_variable(self.opti, self.x_opti_vars, var_name='spline_trajectory_x_opti_vars', scope=scope_name)
        describe_variable(self.opti, self.x_opti_vars_last_poly_x, var_name='x_opti_vars_last_poly_x', scope=scope_name)
        describe_variable(self.opti, self.x_opti_vars_last_poly_dx, var_name='x_opti_vars_last_poly_dx', scope=scope_name)
        describe_variable(self.opti, self.x_opti_vars_last_poly_deltaT, var_name='x_opti_vars_last_poly_deltaT', scope=scope_name)
        describe_variable(self.opti, self.x_opti_vars_first_poly_ddx0, var_name='x_opti_vars_first_poly_ddx0', scope=scope_name)



    def create_polynomials(self, start_x, start_dx, end_x, end_dx, given_total_duration=None):
        # create com polynomials
        assert len(self.poly_list) == 0
        #self.poly_list.clear()
        last_x1 = start_x
        last_dx1 = start_dx
        # ddx at end of last and start of next poly
        last_ddx1 = self.x_opti_vars_first_poly_ddx0

        for i in range(self.num_polynomials_for_trajectory):
            is_last = i == self.num_polynomials_for_trajectory-1
            is_first = i == 0

            if given_total_duration is None:
                if not is_last:
                    deltaT = self.x_opti_vars[self.id_poly_deltaT][:, i]
                else:
                    deltaT = self.x_opti_vars_last_poly_deltaT
            else:
                # clac deltaT based on the given total duration
                if self.given_total_duration_portion_deltaT_for_first_and_last_poly is None:
                    deltaT = given_total_duration/self.num_polynomials_for_trajectory
                else:
                    if is_last or is_first:
                        deltaT = self.given_total_duration_portion_deltaT_for_first_and_last_poly*self.given_total_duration
                    else:
                        deltaT = ((given_total_duration * (1 - 2*self.given_total_duration_portion_deltaT_for_first_and_last_poly))
                                  / (self.num_polynomials_for_trajectory-2))
                    print("## deltaT ", deltaT)

            # polynom variables
            if not is_last:
                x1 = self.x_opti_vars[self.id_poly_x1][:, i]
                dx1 = MX.zeros(self.d)  # for the case where id_poly_dx1 contains just x,y and z should be fixed to zero
                dx1[0:self.dx1_num_vars] += self.x_opti_vars[self.id_poly_dx1][:, i]
            else:
                x1 = self.x_opti_vars_last_poly_x
                dx1 = self.x_opti_vars_last_poly_dx
                # when end values are not given
                if end_x is not None:
                    x1 = end_x
                if end_dx is not None:
                    dx1 = end_dx

            # create polynomial 3
            if self.spline_polynomial_degree == 3:
                p = Polynomial3(
                    x0=     last_x1,
                    dx0=    last_dx1,
                    x1=     x1,
                    dx1=    dx1,
                    deltaT= deltaT
                )
                # ddx consistency constraint
                if self.ddx_consistency_constraint_manual:
                    if i > 0: # skip first
                        self.opti.subject_to(p.evaluate_ddx(0) == last_ddx1)
                    last_ddx1 = p.evaluate_ddx(p.deltaT)

            # create polynomial 4
            elif self.spline_polynomial_degree == 4:
                p = Polynomial4(
                    x0=     last_x1,
                    dx0=    last_dx1,
                    ddx0=   last_ddx1,
                    x1=     x1,
                    dx1=    dx1,
                    deltaT= deltaT
                )
                # skip last poly
                if i < self.num_polynomials_for_trajectory-1:
                    if not self.ddx_consistency_constraint:
                        last_ddx1 = self.x_opti_vars[self.id_poly_ddx0][:, i] # next ddx0 is variable
                    else:
                        # add ddx consistency constraint
                        if not self.ddx_consistency_constraint_manual:
                            last_ddx1 = p.evaluate_ddx(deltaT)
                        else:
                            # keep ddx0 opti parameter but add explicit consistency constraint
                            next_ddx0 = self.x_opti_vars[self.id_poly_ddx0][:, i]
                            self.opti.subject_to(next_ddx0 == p.evaluate_ddx(deltaT))
                            last_ddx1 = next_ddx0


            last_x1 = p.x1
            last_dx1 = p.dx1
            self.poly_list.append(p)

            # keep deltaT positive
            if given_total_duration is None:
                self.opti.subject_to(p.deltaT >= 0)


    def get_sum_detaT(self):
        sum = MX.zeros(self.d)
        for p in self.poly_list:
            sum += p.deltaT
        return sum


    def set_initial_opti_values__zero(self, init_total_duration: float=None):
        """
        Set initial values for opti parameters so that all x1, dx1 values are zero.
        """
        self.set_initial_opti_values__x1_z(initial_x1_z_middle=0, init_total_duration=init_total_duration)


    def set_initial_opti_values__x1_z(self,
                                      initial_x1_z_middle=1,
                                      initial_x1_z_end=0,
                                      init_total_duration: float=None,
                                      interpolation_xy_start=None,
                                      interpolation_xy_end=None,
                                      use_interpolted_dx=True
                                      ):
        """
        Set initial values for opti parameters so that z x1 values of all splines are set to 'initial_x1_z_middle'.
        Z Value at end are set to 'initial_x1_z_end'.
        XY values are interpolated between 'interpolation_xy_start' and 'interpolation_xy_end'.
        :param init_total_duration: just required if total duration is not fixed
        """
        if interpolation_xy_start is None:
            interpolation_xy_start = np.zeros(2)
        if interpolation_xy_end is None:
            interpolation_xy_end = np.zeros(2)

        total_duration = self.given_total_duration
        if self.given_total_duration is None or not (isinstance(self.given_total_duration, float) or isinstance(self.given_total_duration, int)):
            assert init_total_duration is not None, "init_total_duration has to be given if total_duration is not fixed"
            total_duration = init_total_duration

        vel_approx = (interpolation_xy_end - interpolation_xy_start) / total_duration
        delta_pos_per_poly = vel_approx / len(self.poly_list)
        # print("## interpolation_xy_end - interpolation_xy_start", interpolation_xy_end - interpolation_xy_start)
        # print("## vel_approx", vel_approx)
        # print("## delta_pos_per_poly", delta_pos_per_poly)
        # print("## num polies", len(self.poly_list))


        last_xy = interpolation_xy_start + delta_pos_per_poly*total_duration
        for p in self.poly_list:
            # just when optimize over deltaTs
            # deltaT should not be 0 because we divide by it in the polynomial calc
            if self.given_total_duration is None:
                self.opti.set_initial(p.deltaT, init_total_duration)
            x1 = np.zeros(self.d)
            dx1 = np.zeros(self.d)
            x1[-1] = initial_x1_z_middle

            if self.d >= 3:
                x1[0:2] = last_xy
                if use_interpolted_dx:
                    dx1[0:2] = vel_approx

            # polies in the middle
            if p != self.poly_list[-1]:
                self.opti.set_initial(p.x1, x1)
                self.opti.set_initial(p.dx1, dx1)

            # last poly
            # skip last if end values (x1, dx1) are fixed
            if p == self.poly_list[-1] and self.x_opti_vars_last_poly_x is not None:
                x1[-1] = initial_x1_z_end  # height to end value
                self.opti.set_initial(p.x1, x1)
                self.opti.set_initial(p.dx1, 0)  # dx1 zero

            last_xy += delta_pos_per_poly*total_duration



    ## helper
    @staticmethod
    def val_if_in_range(range_start_t, range_end_t, t, value):
        """
        Outputs value when t is in range [range_start_t, range_end_t).
        This is used during optimization, t has to be a scalar.
        :return: casadi.MX expression
        """
        zero_value = 0
        if hasattr(value, 'shape'):
            zero_value = np.zeros(value.shape)
        return casadi.if_else(casadi.logic_and(t >= range_start_t, t < range_end_t),
                              value,  # if true
                              zero_value,      # if false
                              )#, short_circuit=False) # @todo test short_circuit


    def __evaluate_x_generic_func(self, t, func_to_evaluate):
        """
        Get spline value of 'func_to_evaluate' for a given time t (t=0 is at the spline start).
        t can be a scalar or array.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        :param func_to_evaluate(p, t) takes time as parameter and gives an SX/MX expression as output, is member function of Polynomial3/4
        """
        #if len(t.shape) < 2:
        #    t = t.reshape(-1, 1)
        t = Polynomial3.repeat_t_in_first_dim(t, self.d)

        x_vals = MX.zeros((self.d, 1)) #t.shape[0]))
        t_start = MX.zeros(1)

        # normal evaluation (each spline has own ifelse to activate it)
        if not self._debug__casadi_ifelse_nested:
            for p in self.poly_list:
                x_vals = x_vals + SplineTrajectory.val_if_in_range(t_start, t_start + p.deltaT, t, value=func_to_evaluate(p, t - t_start))
                t_start = t_start + p.deltaT

        else:
            # nest if_else (fully ensures that at a time just one spline is active)
            # here also first poly is also active for t<0
            for p_i, p in enumerate(self.poly_list):
                t_local = t - t_start
                if p_i == 0:
                    # first poly is also active for t<0
                    x_vals = func_to_evaluate(p, t_local)
                else:
                    # nest if_else
                    x_vals = casadi.if_else(t_local < 0, x_vals, func_to_evaluate(p, t_local))
                t_start = t_start + p.deltaT

        #x_vals[:, t>t_start] = SX.nan
        return x_vals

    def evaluate_x(self, t):
        """
        Get spline value for a given time t (t=0 is at the spline start).
        Use this for constraints during optimization.
        t can be a scalar or array.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        """
        return self.__evaluate_x_generic_func(t, lambda p, t : p.evaluate(t))

    def evaluate_dx(self, t):
        """
        Get spline first derivative for a given time t (t=0 is at the spline start).
        t can be a scalar or array.
        Use this for constraints during optimization.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        """
        return self.__evaluate_x_generic_func(t, lambda p, t : p.evaluate_dx(t))

    def evaluate_ddx(self, t):
        """
        Get spline second derivative for a given time t (t=0 is at the spline start).
        t can be a scalar or array.
        Use this for constraints during optimization.
        This will return an SX expression
        :param t   either a scalar or an array with shape (#num_time_values) or (#num_time_values,1)
        """
        return self.__evaluate_x_generic_func(t, lambda p, t : p.evaluate_ddx(t))





    ######################################################
    ## plotting and evaluation
    class EvalSplineConnectionPoints:
        t_values: np.array      # timepoints of the connection points
        x_values: np.array      # [x, time-index] the x value of the function of this spline at that timepoint

        def __init__(self, d):
            self.t_values = np.array([])
            self.x_values = np.zeros((d, 0))

        def append(self, other_spline_connection_points: 'EvalSplineConnectionPoints', start_t: float, remove_before_t_zero = True):
            """
            Appends another EvalSplineConnectionPoints to this object.
            :param start_t:  this time value will be added to the appended other_spline_connection_points
            """
            t_vals_new = other_spline_connection_points.t_values + start_t
            t_values_larger_zero = t_vals_new >= 0
            if not remove_before_t_zero:
                t_values_larger_zero = np.ones_like(t_values_larger_zero)
            self.t_values = np.hstack([self.t_values, t_vals_new[t_values_larger_zero]])
            self.x_values = np.hstack([self.x_values, other_spline_connection_points.x_values[:, t_values_larger_zero]])

    def __evaluate_solution_generic_func(self, t, func_to_evaluate,
                                         t_upper_bound: float = None,
                                         make_nans_for_t_out_of_range=True,
                                         output_spline_connection_points=False) -> (np.array, EvalSplineConnectionPoints):
        """
        Get spline value of 'func_to_evaluate' for a given time t (t=0 is at the spline start).
        This will pull the real optimized values out of opti first.
        t needs to be a scalar.
        This will return n np.array.
        :param func_to_evaluate(p, t) takes time as parameter and gives a np.array expression as output, is member function of Polynomial3
        :param make_nans_for_t_out_of_range if true replace values where t is out of range of spline with NaN.
        :return: x_vals,
                spline_connection_points: list[EvalSplineConnectionPoint]  points of the spline end and beginnings
        """
        #def one_in_range(range_start_t, range_end_t, t):
        #    return np.heaviside(t-range_start_t, 1) * np.heaviside(-t+range_end_t, 1)

        x_vals = np.zeros((self.d, t.shape[0]))
        spline_connection_points = SplineTrajectory.EvalSplineConnectionPoints(self.d)
        spline_connection_points.t_values = np.zeros(len(self.poly_list)+1)
        spline_connection_points.x_values = np.zeros((self.d, len(self.poly_list)+1))

        t_start = 0
        for i, p in enumerate(self.poly_list):
            p_eval = p.create_new_poly_with_solved_opti_values(self.opti)
            #x_vals += func_to_evaluate(p_eval, (t - t_start)) * one_in_range(t_start, t_start + p_eval.deltaT, t)
            p_x_vals = func_to_evaluate(p_eval, (t - t_start))
            p_x_vals = np.array(p_x_vals)
            p_x_vals[:, np.logical_or(t < t_start, t >= t_start+p_eval.deltaT)] = 0
            x_vals += p_x_vals
            spline_connection_points.t_values[i] = t_start
            spline_connection_points.x_values[:, i:i+1] = p_eval.x0

            t_start += p_eval.deltaT

            # add last connection point
            if i == len(self.poly_list)-1:
                spline_connection_points.t_values[-1] = t_start
                spline_connection_points.x_values[:, -1:] = p_eval.x1

        # remove all after t_upper_bound
        if t_upper_bound is not None:
            x_vals[:, t > t_upper_bound] = 0

        if make_nans_for_t_out_of_range:
            x_vals[:, t > t_start] = np.nan
            if t_upper_bound is not None:
                x_vals[:, t > t_upper_bound] = np.nan
        if output_spline_connection_points:
            return x_vals, spline_connection_points
        else:
            return x_vals

    def evaluate_solution_x(self, t: np.array,
                            make_nans_for_t_out_of_range=True,
                            t_upper_bound: float = None,
                            output_spline_connection_points=False) -> (np.array, EvalSplineConnectionPoints):
        """
        Get solution from solver and evaluate it for time(s) t.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :param t:
        :param output_spline_connection_points: return second spline connection point list
        :param make_nans_for_t_out_of_range if true replace values where t is out of range of spline with NaN.
        :return: x_vals,
                 spline_connection_points: list[EvalSplineConnectionPoint]  points of the spline end and beginnings
        """
        return self.__evaluate_solution_generic_func(t, lambda p, ts : p.evaluate(ts),
                                                     t_upper_bound,
                                                     make_nans_for_t_out_of_range,
                                                     output_spline_connection_points)

    def evaluate_solution_dx(self, t: np.array,
                             make_nans_for_t_out_of_range=True,
                             t_upper_bound: float = None,
                             output_spline_connection_points=False) -> (np.array, EvalSplineConnectionPoints):
        """
        Get solution from solver and evaluate it for time(s) t.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :param t:
        :param output_spline_connection_points: return second spline connection point list
        :param make_nans_for_t_out_of_range if true replace values where t is out of range of spline with NaN.
        :return: x_vals,
                 spline_connection_points: list[EvalSplineConnectionPoint]  points of the spline end and beginnings
        """
        return self.__evaluate_solution_generic_func(t, lambda p, ts : p.evaluate_dx(ts),
                                                     t_upper_bound,
                                                     make_nans_for_t_out_of_range,
                                                     output_spline_connection_points)


    def evaluate_solution_full_trajectory_individual(self, step_t=0.01):
        """
        Get output values over time for full trajectory.
        This will return an individual value array for each polynomial of this trajectory.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :return:
        """
        poly_list_evaluated = []
        total_time = 0
        for p in self.poly_list:
            p_eval = p.create_new_poly_with_solved_opti_values(self.opti)
            poly_list_evaluated.append(p_eval)
            total_time += p.deltaT

        t_vals = []
        x_vals = []
        t_start = 0
        for p in poly_list_evaluated:
            p_t_vals = np.arange(0, p.deltaT+step_t, step=step_t)
            p_x_vals = p.evaluate(p_t_vals)
            t_vals.append(p_t_vals + t_start)
            x_vals.append(p_x_vals)
            t_start += p.deltaT

        return t_vals, x_vals, poly_list_evaluated

    def evaluate_solution_full_trajectory(self, step_t=0.01):
        """
        Get output values over time for full trajectory.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :return:
        """
        t_vals, x_vals, poly_list_evaluated = self.evaluate_solution_full_trajectory_individual(step_t)
        t_vals_all = np.zeros((1))
        x_vals_all = np.zeros((self.d, 1))
        for p_t_vals, p_x_vals, p in zip(t_vals, x_vals, poly_list_evaluated):
            t_vals_all = np.hstack([t_vals_all, p_t_vals])
            x_vals_all = np.hstack([x_vals_all, p_x_vals])
        return t_vals_all, x_vals_all




    def evaluate_solution_and_plot_full_trajectory(self, step_t=0.01, show_plot=True, only_plot_dimensions: np.s_ = None, axis=None):
        """
        Get output values over time for full trajectory.
        This will pull the real optimized values out of opti first.
        Therefore, opt.solve() has to be called before.
        :return:
        """
        if axis is None:
            axis = plt

        if only_plot_dimensions is None:
            only_plot_dimensions = np.s_[0:self.d]
        if not isinstance(only_plot_dimensions, int):
            num_dims = (only_plot_dimensions.stop - only_plot_dimensions.start) / (
                        only_plot_dimensions.step if only_plot_dimensions.step is not None else 1)
        else:
            # if only_plot_dimensions is scalar
            num_dims = 1


        t_vals, x_vals, poly_list_evaluated = self.evaluate_solution_full_trajectory_individual(step_t)
        for p_t_vals, p_x_vals, p in zip(t_vals, x_vals, poly_list_evaluated):
            p_x_vals = np.array(p_x_vals)
            axis.plot(p_t_vals, p_x_vals.swapaxes(0, 1)[:, only_plot_dimensions])  # make time the first axis
            #print("p.deltaT", p.deltaT)
            t_start = np.ones_like(p.x0[only_plot_dimensions])*p_t_vals[0]
            t_end = np.ones_like(p.x0[only_plot_dimensions])*(p_t_vals[0] + p.deltaT)

            if num_dims == 1:
                axis.scatter(t_start, p.x0[only_plot_dimensions], c='red', s=18)
                axis.scatter(t_end, p.x1[only_plot_dimensions], c='red', s=18)
            else:
                axis.scatter(t_start, p.x0[only_plot_dimensions], c='red', s=18)
                axis.scatter(t_end, p.x1[only_plot_dimensions], c='red', s=18)

            d_t = 0.05
            d_t_p_scaled = d_t #/ p.deltaT
            if num_dims == 1:
                disp_t = np.hstack([t_start - d_t, t_start + d_t])
                disp_x = np.hstack([-p.dx0[only_plot_dimensions] * d_t_p_scaled, p.dx0[only_plot_dimensions] * d_t_p_scaled]) + p.x0[only_plot_dimensions]
                axis.plot(disp_t, disp_x, c='black', linestyle='dashed')
            else:
                disp_t = np.hstack([t_start - d_t, t_start + d_t]).T
                disp_x = np.hstack([-p.dx0 * d_t_p_scaled, p.dx0 * d_t_p_scaled]).T + p.x0.T
                axis.plot(disp_t, disp_x, c='black', linestyle='dashed')

        if show_plot:
            axis.show()
