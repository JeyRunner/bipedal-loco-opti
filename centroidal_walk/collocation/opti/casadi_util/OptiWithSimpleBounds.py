import casadi
import numpy as np
import rich
from casadi import OptiAdvanced, MX, DM, Opti, jacobian, Sparsity, MetaVar, nlpsol


class OptiWithSimpleBounds(OptiAdvanced):

	solved_with_simple_bounds: bool
	baked_with_simple_bounds: bool
	solved_with_simple_bounds_solver_result: dict

	# key is the index of the variable
	class VarBounds:
		sym: MX
		lbx: DM
		ubx: DM
	simple_var_bounds_values: dict[VarBounds]


	@staticmethod
	def create_from_opti(opti: Opti) -> 'OptiWithSimpleBounds':
		"""
		Create OptiWithSimpleBounds from an existing opti object.
		Not that this will create a copy of the opti object, thus don't use the original opti object afterwards.
		:return: the new OptiWithSimpleBounds object.
		"""
		advanced = opti.advanced
		# patch object
		advanced.__class__ = OptiWithSimpleBounds
		# add attrs
		setattr(advanced, 'solved_with_simple_bounds', False)
		setattr(advanced, 'baked_with_simple_bounds', False)
		setattr(advanced, 'solved_with_simple_bounds_solver_result', None)
		setattr(advanced, 'simple_var_bounds_values', {})
		return advanced

	@staticmethod
	def create():
		"""
		Create OptiWithSimpleBounds object to be used inplace of Opti.
		"""
		return OptiWithSimpleBounds.create_from_opti(Opti())

	def __init__(self, *args):
		assert False, ("Never create this object directly, allways use: \n"
					   "OptiWithSimpleBounds.create()")


	def subject_to_var_bounds(self, min: DM, variable: MX, max: DM):
		"""
		Add simple bounds on optimization variables.
		If you do not want to restrict either min or max, just use (-)DM.inf() as value.

		Example usage:
		opti.subject_to_var_bounds(-DM.inf(), var_x[0:2], 42)
		opti.subject_to_var_bounds(0, var_x, 42)
		"""
		# helper
		def get_sym_slice_elements():
			# get symbols
			f = casadi.Function("f", [], [variable], {"allow_free": True})
			assert len(f.free_mx()) == 1, "currently just support using one var at a time"
			sym: MX = f.free_mx()[0]
			assert sym.is_symbolic(), "you need to provided a variable to constrain"

			# Evaluate jacobian of expr wrt symbols (taken from optistack_internal.cpp (set_value_internal))
			Jf = casadi.Function("f", [], [jacobian(variable, sym)], [], ['J'])#, {"compact": True})
			J: DM = Jf()['J']
			#print('J', J)
			sp_JT: Sparsity = J.T.sparsity()
			#print(sp_JT)
			sliced_elements_of_var = sp_JT.row()
			return sym, sliced_elements_of_var


		var_sym = None
		var_elements_to_constrain = []

		# if is just a var
		if variable.is_symbolic():
			var_sym = variable
			var_elements_to_constrain = np.arange(0, var_sym.rows())
		else:
			# slice of var
			sym, sliced_elements_of_var = get_sym_slice_elements()
			print(f'variable bounds for sym ({variable}) {sym} elements {sliced_elements_of_var}')
			var_sym = sym
			var_elements_to_constrain = sliced_elements_of_var

		var_i = self.__get_var_sym_index(var_sym)
		# print('variable index', var_i)

		# add var bounds
		var_bounds = None
		if var_i not in self.simple_var_bounds_values:
			var_nx = var_sym.rows()
			var_bounds = OptiWithSimpleBounds.VarBounds()
			var_bounds.sym = var_sym
			var_bounds.lbx = -DM.inf(var_nx)
			var_bounds.ubx = DM.inf(var_nx)
			self.simple_var_bounds_values[var_i] = var_bounds
		else:
			var_bounds = self.simple_var_bounds_values[var_i]

		# check that bounds are not constraint already
		if not (casadi.logic_all(var_bounds.lbx[var_elements_to_constrain] == -DM.inf()) and
			casadi.logic_all(var_bounds.ubx[var_elements_to_constrain] == DM.inf())):
			assert False, \
				(f"Variable {var_sym} has already defined bounds (from previous call to subject_to_var_bounds) for range {var_elements_to_constrain}.\n"
				  f"	lbx {var_bounds.lbx}\n"
				  f"	ubx {var_bounds.ubx}\n")

		# set bounds
		var_bounds.lbx[var_elements_to_constrain] = min
		var_bounds.ubx[var_elements_to_constrain] = max
		# print('lbx ', var_bounds.lbx)
		# print('ubx ', var_bounds.ubx)
		# print()

	def __get_var_sym_index(self, var_sym) -> int:
		var_meta: MetaVar = self.get_meta(var_sym)
		var_i = var_meta.i
		return var_i

	def __get_var_lower_and_upper_bounds(self) -> (DM, DM):
		"""
		Get upper and lower bounds for variables x.
		opti.bake() has to be called before.
		:return: lbx, ubx
		"""
		lbx = -DM.inf(self.nx)
		ubx = DM.inf(self.nx)
		print("__get_var_lower_and_upper_bounds:")
		for var in self.simple_var_bounds_values.keys():
			bounds_dict: OptiWithSimpleBounds.VarBounds = self.simple_var_bounds_values[var]
			var_meta: MetaVar = self.get_meta(bounds_dict.sym)
			start = var_meta.start
			stop = var_meta.stop
			lbx[start:stop] = bounds_dict.lbx
			ubx[start:stop] = bounds_dict.ubx
			print(f"> var '{var}' index range  {start} : {stop},  bounds = {bounds_dict}")
		#print('lbx', lbx)
		#print('ubx', ubx)
		print()
		return lbx, ubx


	def bake_solve_with_simple_var_bounds(self, solver: str, p_opts={}, solver_opts={}):
		self.solver(solver, p_opts, solver_opts)

		# get the args how the internal solver would be called:
		advanced: OptiAdvanced = self
		advanced.bake()
		# advanced.solve_prepare()
		# solver_args = advanced.arg()
		# print('orig solver args', solver_args)

		# put the solver options into the options
		if solver_opts is not None:
			p_opts[solver] = solver_opts

		# create new solver
		self.solver_with_bounds = nlpsol('solver', 'ipopt', {
			'x': self.x,
			'p': self.p,
			'f': self.f,
			'g': self.g,
		}, p_opts)
		self.baked_with_simple_bounds = True


	def solve_with_simple_var_bounds(self) -> dict:
		"""
		Solve with previously defined simple_var_bounds.
		Note that this will create a new internal nlpsolver and copy back the results into this opti object.
		Thus, some methods of this opti object, e.g. opti.callback(...), will not work.
		But the resulting values can be normally obtained by using opti.value(...).
		:return the solver stats (contains e.g.iter_count )
		"""
		assert self.baked_with_simple_bounds, "call bake_solve_with_simple_var_bounds(...) first"

		# get the args how the internal solver would be called:
		advanced: OptiAdvanced = self
		advanced.bake()
		advanced.solve_prepare()
		solver_args = advanced.arg()
		#print('orig solver args', solver_args)

		# get the lower and upper variable bounds
		lbx, ubx = self.__get_var_lower_and_upper_bounds()

		# convert upper and lower bounds on constraint g to -> SX -> DM
		# lbg_val = Function('lbg_func', [], [self.lbg], [], ['ret']).expand()()['ret']
		# ubg_val = Function('ubg_func', [], [self.ubg], [], ['ret']).expand()()['ret']
		# lbg_val = self.value(self.lbg)
		# ubg_val = self.value(self.ubg)
		# #print('opti_adv.lbg', self.lbg)
		# print('lbg_val', lbg_val)
		# print()
		# #print('opti_adv.ubg', self.ubg)
		# print('ubg_val', ubg_val)
		# print()

		# test
		# not_simple_indx, lbx, ubx, lam_f, lam_b = casadi.detect_simple_bounds(
		#     opti.x, opti.p, opti.g, opti.lbg, opti.ubg)
		# print('new lbx', lbx)

		# get initial values (which were set via opti.set_initial(...))
		# x0 = self.value(self.x, self.initial())
		# print('x0', x0)

		solved_with_simple_bounds_solver_result = self.solver_with_bounds(
			**solver_args,
			lbx=lbx,
			ubx=ubx
		)

		# copy back results form solver to x values of this object
		advanced.res(solved_with_simple_bounds_solver_result)

		self.solved_with_simple_bounds = True
		return self.solver_with_bounds.stats()


	def stats(self, *args) -> dict:
		if self.solved_with_simple_bounds:
			return self.solver_with_bounds.stats(*args)
		return super().stats(*args)


