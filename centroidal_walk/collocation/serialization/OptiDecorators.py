import functools
import inspect
from inspect import getfullargspec

import rich


def additional_cost_or_constraint(func):
	name: str = func.__name__
	assert name.startswith('add_additional_')
	name_short = name.removeprefix('add_additional_')
	# name_short = name.removeprefix('add_additional_constraint__')
	# name_short = name_short.removeprefix('add_additional_cost__')
	# name_short = name_short.removeprefix('add_additional_cost_or_constraint__')
	signature = inspect.signature(func)

	@functools.wraps(func)
	def wrapper(*args, **kwargs):
		print()
		print(f">> {name}")
		bound_args = signature.bind(*args, **kwargs)
		bound_args.apply_defaults()
		func_arguemnts_dict = bound_args.arguments

		self = func_arguemnts_dict['self']
		func_arguemnts_dict.pop('self', None)
		assert self is not None, f"{name} needs to be a method of a class"

		# rich.print('bound_args', func_arguemnts_dict)
		# rich.print('self', self)

		assert hasattr(self, 'additional_costs_and_constraints_parameters'), "the opti instance needs this member."
		constraints_dict: dict = self.additional_costs_and_constraints_parameters
		assert name_short not in constraints_dict, f"you can't add a constraint/cost twice, '{name_short}' was already added."

		# save parameters
		constraints_dict[name_short] = func_arguemnts_dict
		# call
		return func(*args, **kwargs)
	return wrapper