from unittest import TestCase
import rich
from centroidal_walk.collocation.serialization.OptiDecorators import *

class TestOpti:
	def __init__(self):
		self.additional_costs_and_constraints_parameters = {}

	@additional_cost_or_constraint
	def add_additional_constraint__test_func(self, a, b=2, c=10):
		pass

	@additional_cost_or_constraint
	def add_additional_constraint__2(self, x=10):
		pass



class TestOptiDecorators(TestCase):

	def test_additional_cost_or_constraint(self):
		o = TestOpti()
		o.add_additional_constraint__test_func(2, c=22)
		o.add_additional_constraint__2()
		rich.print(o.additional_costs_and_constraints_parameters)

		self.assertEquals(
			o.additional_costs_and_constraints_parameters,
			{'constraint__test_func': {'a': 2, 'b': 2, 'c': 22}, 'constraint__2': {'x': 10}}
		)
