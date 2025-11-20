from dataclasses import dataclass
from pathlib import Path

import casadi
import yaml
import numpy as np
from yaml import Loader
from ast import literal_eval
import rich

from centroidal_walk.collocation.opti.casadi_util.OptiWithSimpleBounds import *


# better store np arrays
def ndarray_representer(dumper: yaml.Dumper, array: np.ndarray) -> yaml.Node:
	#return dumper.represent_sequence('!np', array.tolist())
	style = '|'
	# if len(array.shape) <= 1:
	# 	style = None
	return dumper.represent_scalar(
		'!np',
		np.array2string(array, separator=', ', precision=100),
		style=style
	)
def ndarray_constructor(loader: yaml.Loader, node):
	arr_str = loader.construct_scalar(node)
	arr_lists = literal_eval(arr_str)
	array = np.array(arr_lists) #np.fromstring(arr_str, sep=',  ')
	# print('arr_str', arr_lists)
	# print('>> ', array)
	return array

yaml.add_representer(np.ndarray, ndarray_representer)
yaml.add_constructor('!np', ndarray_constructor)







def serializable_opti_variables(variables_to_serializable: list[str]):
	def inner_decorator(self):
		# print('## run decorator')
		# variables_to_serizalize_dict = {}
		# rich.print(self.__dict__)
		# var_dict = self.__dict__['__annotations__']
		# for v in variables_to_serizalize:
		# 	variables_to_serizalize_dict[v] = var_dict[v]
		# setattr(self, 'variables_to_serizalize_dict', variables_to_serizalize_dict)
		# rich.print('variables_to_serizalize_dict', variables_to_serizalize_dict)
		setattr(self, '_variables_to_serialize_keys', variables_to_serializable)
		return self
	return inner_decorator


class SerializableOptiVariables:
	opti: OptiWithSimpleBounds
	_variables_to_serialize_keys: list[str]


	def __init__(self, opti: OptiWithSimpleBounds):
		self.opti = opti
		pass

	@staticmethod
	def serialize_opti_vars_static(self, variables_to_serialize_keys):
		var_dict = {}
		for k in variables_to_serialize_keys:
			var = vars(self)[k]
			var_value = None
			if var is None:
				continue
			elif isinstance(var, casadi.MX):
				var_value = self.opti.value(var)
			elif isinstance(var, np.ndarray):
				var_value = var
			elif isinstance(var, SerializableOptiVariables):
				var_value = var.serialize_opti_vars()
				#assert False, 'not implemented'
			elif isinstance(var, list):
				var_value = []
				for el in var:
					assert isinstance(el, SerializableOptiVariables), 'list elements need to be SerializableOptiVariables'
					var_value.append(el.serialize_opti_vars())
				#assert False, 'not implemented'
			else:
				assert False, (f"member variable {k} of class {self.__class__} is not an MX variable "
							   f"or SerializableOptiVariables or list of SerializableOptiVariables.")
			var_dict[k] = var_value
		#rich.print(var_dict)
		return var_dict #yaml.dump(var_dict, indent=2)

	def serialize_opti_vars(self):
		return SerializableOptiVariables.serialize_opti_vars_static(self, self._variables_to_serialize_keys)


	@staticmethod
	def deserialize_opti_vars_load_solution_static(self, _variables_to_serialize_keys, variables_values_dict: dict):
		var_dict = {}
		for k in _variables_to_serialize_keys:
			var = vars(self)[k]
			if var is None:
				continue
			value = variables_values_dict[k]

			if isinstance(var, casadi.MX):
				self.opti.set_variable_value(var, value)
			elif isinstance(var, SerializableOptiVariables):
				var.deserialize_opti_vars_load_solution(value)
			elif isinstance(var, list):
				assert isinstance(var, list), f'{k} needs to be a list'
				for el_i, el in enumerate(var):
					assert isinstance(el, SerializableOptiVariables), 'list elements need to be SerializableOptiVariables'
					el.deserialize_opti_vars_load_solution(value[el_i])
			else:
				assert False, f"member variable {k} of class {self.__class__} is not an MX variable or SerializableOptiVariables."

	def deserialize_opti_vars_load_solution(self,  variables_values_dict: dict):
		SerializableOptiVariables.deserialize_opti_vars_load_solution_static(
			self,
			self._variables_to_serialize_keys,
			variables_values_dict
		)
