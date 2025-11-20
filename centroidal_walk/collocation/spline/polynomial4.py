from builtins import staticmethod

from centroidal_walk.collocation.spline.polynomial3 import Polynomial3
from centroidal_walk.collocation.opti.centroidal_dyn_poly import *
import numpy._typing as npt







class Polynomial4():
    """
    4. degree polynomial with hermite parametrization.
    """

    # may have multiple dimension when the polynom output is also multidimensional
    x0: npt.ArrayLike
    dx0: npt.ArrayLike
    ddx0: npt.ArrayLike
    x1: npt.ArrayLike
    dx1: npt.ArrayLike

    deltaT: float



    @staticmethod
    def get_required_num_parameters_for_dim(polynom_output_dimensions: int, deltaT_variable=True):
        """
        Get number of required (scalar) parameters for a Polynomial3 with `polynom_output_dimensions` output dimensions.
        When is deltaT_variable=False, deltaT will not be counted as variable.
        """
        return polynom_output_dimensions*5 + (1 if deltaT_variable else 0)


    @staticmethod
    def create_from_flat_param_array(parameter_array, polynom_output_dimensions: int, reshape=False):
        """
        Create polynomial from flat parameter array.
        The flatted order of parameters in the array is: x0, dx0, x1, dx1, deltaT.
        Each parameter in the array occupies polynom_output_dimensions elements in the parameter_array.
        :param parameter_array:
        :param polynom_output_dimensions:
        :return:
        """
        # indices for a 3d polynomial represented as flat variable array
        id_poly_x0 = get_range_for_dimensional_index(0, polynom_output_dimensions)
        id_poly_dx0 = get_range_for_dimensional_index(1, polynom_output_dimensions)
        id_poly_ddx0 = get_range_for_dimensional_index(2, polynom_output_dimensions)
        id_poly_x1 = get_range_for_dimensional_index(3, polynom_output_dimensions)
        id_poly_dx1 = get_range_for_dimensional_index(4, polynom_output_dimensions)
        id_poly_deltaT = np.s_[5*polynom_output_dimensions : 5*polynom_output_dimensions + 1]  # the time is just 1d

        return Polynomial4(
            x0=     parameter_array[id_poly_x0],
            dx0=    parameter_array[id_poly_dx0],
            ddx0=    parameter_array[id_poly_ddx0],
            x1=     parameter_array[id_poly_x1],
            dx1=    parameter_array[id_poly_dx1],
            deltaT= parameter_array[id_poly_deltaT],
            reshape=reshape
        )


    def __init__(self, x0, dx0, ddx0, x1, dx1, deltaT, reshape=False):
        self.x0 = x0
        self.dx0 = dx0
        self.ddx0 = ddx0
        self.x1 = x1
        self.dx1 = dx1
        self.deltaT = deltaT
        # add second dimension for time
        if reshape:
            self.x0 = self.x0.reshape(-1, 1)
            self.dx0 = self.dx0.reshape(-1, 1)
            self.ddx0 = self.ddx0.reshape(-1, 1)
            self.x1 = self.x1.reshape(-1, 1)
            self.dx1 = self.dx1.reshape(-1, 1)



    def get_coefficients(self):
        """
        Get polynom coefficients.
        Note that x0 is allways at t=0 and x1 at t=deltaT.
        """
        # Hermite Polynom parametrization
        a0 = self.x0
        #a1 = (self.deltaT**(-1)) * self.dx0
        a1 = self.dx0
        #a2 = (self.deltaT**(-2)) * self.ddx0/2
        a2 = self.ddx0/2

        #a3 = (self.deltaT**(-3)) * ( 4*self.x1 - self.dx1 - 4*self.x0 - 3*self.dx0 - self.ddx0)
        a3 = ( 4*self.x1 - self.dx1 - 4*self.x0 - 3*self.dx0 - self.ddx0)
        #a4 = (self.deltaT**(-4)) * (-3*self.x1 + 3*self.x0 + 2*self.dx0 + 0.5 * self.ddx0 + self.dx1)
        a4 = (-3*self.x1 + 3*self.x0 + 2*self.dx0 + 0.5 * self.ddx0 + self.dx1)

        return a0, a1, a2, a3, a4



    def evaluate(self, t):
        """
        Get polynom value at time t.
        Note that x0 is allways at t=0 and x1 at t=deltaT.
        """
        a0, a1, a2, a3, a4 = self.get_coefficients()
        # deal with t array, just when it is not a np array
        # only do that when necessary, this is not required when called from evaluate_solution_x (otherwise it slow)
        if not isinstance(t, float) and t.shape[0] > 1 and not self.__contains_just_solution_values_as_np_arrays(): #@todo should be not __containst...
            # either time is 1d or time values are in fist dim
            # or t values are already in the second dim
            assert len(t.shape) == 1 or (len(t.shape) == 2)
            # reshape t so that it has t vals in second dim and repeats these self.d times in the first dim
            t = Polynomial3.repeat_t_in_first_dim(t, self.x0.shape[0])
            num_t = t.shape[1]

            a0 =  casadi.repmat(a0, 1, num_t) # repeat in second axis the number of time values
            a1 =  casadi.repmat(a1, 1, num_t) # repeat in second axis the number of time values
            a2 =  casadi.repmat(a2, 1, num_t) # repeat in second axis the number of time values
            a3 =  casadi.repmat(a3, 1, num_t) # repeat in second axis the number of time values
            a4 =  casadi.repmat(a4, 1, num_t) # repeat in second axis the number of time values

        t = t/self.deltaT
        x = a0 + a1*t + a2*(t**2) + a3*(t**3) + a4*(t**4)
        return x

    def evaluate_dx(self, t):
        """
        Get polynom first derivative value at time t.
        Note that x0 is allways at t=0 and x1 at t=deltaT.
        """
        t = t/self.deltaT
        a0, a1, a2, a3, a4 = self.get_coefficients()
        x = a1 + 2*a2*(t**1) + 3*a3*(t**2) + 4*a4*(t**3)
        return x

    def evaluate_ddx(self, t):
        """
        Get polynom second derivative value at time t.
        Note that x0 is allways at t=0 and x1 at t=deltaT.
        """
        t = t/self.deltaT
        a0, a1, a2, a3, a4 = self.get_coefficients()
        x = 2*a2 + 2*3*a3*(t**1) + 3*4*a4*(t**2)
        return x





    def plot(self, t0=0, tend=None):
        if tend == None:
            tend = self.deltaT

        t_vals = np.arange(t0, tend + (tend-t0)/100, step=(tend-t0)/100)
        x_vals = self.evaluate(t_vals).swapaxes(0, 1)
        print(x_vals.shape)
        plt.plot(t_vals, x_vals)
        plt.scatter(np.zeros_like(self.x0), self.x0, c='red')
        plt.scatter(np.ones_like(self.x0)*self.deltaT, self.x1, c='red')
        plt.show()


    def create_new_poly_with_solved_opti_values(self, opti: casadi.Opti):
        """
        Create a new polynomial with real values from an optimization result.
        opti.solve() has to be called before
        """
        return Polynomial4(
            x0=     opti.value(self.x0),
            dx0=    opti.value(self.dx0),
            ddx0=   opti.value(self.ddx0) if not isinstance(self.ddx0, np.ndarray) else self.ddx0,
            x1=     opti.value(self.x1) if not isinstance(self.x1, np.ndarray) else self.x1,
            dx1=    opti.value(self.dx1) if not isinstance(self.dx1, np.ndarray) else self.dx1,
            deltaT= opti.value(self.deltaT),
            reshape=True
        )


    def __contains_just_solution_values_as_np_arrays(self):
        """
        returns true if all variables (x0, ...) are np arrays and not casadi values.
        """
        return isinstance(self.x0, np.ndarray) \
            and isinstance(self.dx0, np.ndarray) \
            and isinstance(self.ddx0, np.ndarray) \
            and isinstance(self.x1, np.ndarray) \
            and isinstance(self.dx1, np.ndarray) \
            and isinstance(self.deltaT, np.ndarray)