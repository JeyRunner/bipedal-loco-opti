import casadi
import numpy
import numpy as np
import rich
from casadi import OptiAdvanced, MX, DM, Opti, jacobian, Sparsity, MetaVar, nlpsol


class OptiLoadable():
	"""
	A dummy opti class just for loading a solution and evaluation of values.
	This class is just for inferance/evaluation not for solving.
	"""

	class Variable:
		loaded_value: np.ndarray
		def __init__(self):
			self.loaded_value = None



	_next_var_index: int
	_variables: dict #[Variable]
	_variables_user_dicts: dict

	def __init__(self):
		self._next_var_index = 0
		self._variables = {}
		self._variables_user_dicts = {}


	def variable(self, rows, columns=1):
		"""
		Create a new variable, analog to Opti.variable(...)
		"""
		var = MX.sym(f'opti0_x_{self._next_var_index}', rows, columns)
		self._variables[var] = DM.zeros((rows, columns)) # default value
		self._next_var_index += 1
		return var


	def parameter(self, rows, columns=1):
		"""
		Create a new parameter, analog to Opti.parameter(...)
		"""
		return self.variable(rows, columns)



	def subject_to(self, *args):
		"""
		Dummy constraint, will do noting since this class is just for inference/evaluation not for solving.
		"""
		pass

	def subject_to_var_bounds(self, *args):
		"""
		Dummy constraint, will do noting since this class is just for inference/evaluation not for solving.
		"""
		pass

	def bounded(self, *args):
		"""
		Dummy bounded, will do noting since this class is just for inference/evaluation not for solving.
		"""
		pass

	def set_initial(self, *args):
		"""
		Dummy set_initial, will do noting since this class is just for inference/evaluation not for solving.
		"""


	@property
	def advanced(self):
		return self

	def symvar(self):
		return list(self._variables.keys())



	def set_variable_value(self, variable: MX, value: np.ndarray):
		"""
		Set the value of a variable to the value of solution.
		Do this for all variables before calling .value(...)
		:param variable:
		:param value: assing this value to the varialbe
		"""
		self._variables[variable] = value


	def value(self, expression: MX, as_np=True, np_rm_last_index_if_empty=True) -> np.ndarray:
		"""
		Get the value of a variable, analog to Opti.value(...).
		Note that before a solution has to be loaded.
		"""
		if isinstance(expression, np.ndarray):
			return expression

		all_variables = list(self._variables.keys())
		all_variable_values = list(self._variables.values())

		f = casadi.Function('f_value', all_variables, [expression])
		value: dict = f.call(all_variable_values)
		assert len(value) == 1

		value_dm = value[0]
		if as_np:
			arr = np.array(value_dm)
			if np_rm_last_index_if_empty and len(arr.shape) == 2 and arr.shape[1] <= 1 and arr.shape[0] <= 1:
				return float(arr[0, 0])
			elif np_rm_last_index_if_empty and len(arr.shape) == 2 and arr.shape[-1] <= 1:
				return arr[:, 0]
			else:
				return arr
		else:
			return value_dm

	def update_user_dict(self, variable: MX, user_dict: dict):
		self._variables_user_dicts[variable] = user_dict

	def user_dict(self, variable: MX):
		if variable in self._variables_user_dicts:
			return self._variables_user_dicts[variable]
		else:
			return {}
