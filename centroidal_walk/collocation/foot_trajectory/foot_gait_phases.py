import casadi
import matplotlib.pyplot as plt
from abc import abstractmethod, ABC

from centroidal_walk.collocation.foot_trajectory.trajectory_eval_structs import *
from centroidal_walk.collocation.opti.casadi_util.opti_util import describe_variable
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory
from centroidal_walk.collocation.serialization.yaml_util import *



class Phase(ABC):
	"""
	Generic foot gait phase.
	"""
	d = 2
	opti: casadi.Opti
	duration: casadi.Opti.variable
	start_t: casadi.MX  # time when this phase starts (variable) @todo unused -> remove

	def __init__(self,
				 start_t,
				 opti: casadi.Opti,
				 dimensions,
				 phase_duration: float = None,
				 scope_name: list[str] = []
				 ):
		self.start_t = start_t
		self.opti = opti
		self.d = dimensions
		# variable phase duration
		if phase_duration is None:
			self.duration = opti.variable(1)
			describe_variable(self.opti, self.duration, 'duration', scope_name)
			#opti.subject_to(self.duration >= 0)
			#opti.subject_to(opti.bounded(phase_duration_min, self.duration, phase_duration_max))
		else:
			# fixed phase duration
			self.duration = phase_duration

	@abstractmethod
	def evaluate_foot_pos(self, t):
		"""
		Get MX expression for foot pos at times t (at the beginning of the phase is t=0).
		"""
		assert False, "This function needs to be overloaded by specific phase type"

	@abstractmethod
	def evaluate_foot_pos_start(self, derivative=False):
		"""
		Get MX expression for foot pos at phase start (t=0).
		"""
		assert False, "This function needs to be overloaded by specific phase type"

	@abstractmethod
	def evaluate_foot_pos_end(self, derivative=False):
		"""
		Get MX expression for foot pos at phase end (t=duration).
		Note that this can not be evaluated with evaluate_foot_pos.
		"""
		assert False, "This function needs to be overloaded by specific phase type"


	@abstractmethod
	def evaluate_foot_dpos(self, t):
		"""
		Get MX expression for foot pos derivative at times t (at the beginning of the phase is t=0).
		"""
		assert False, "This function needs to be overloaded by specific phase type"

	@abstractmethod
	def evaluate_foot_force(self, t):
		"""
		Get MX expression for foot force at times t (at the beginning of the phase is t=0).
		"""
		assert False, "This function needs to be overloaded by specific phase type"

	@abstractmethod
	def get_phase_type(self) -> PhaseType:
		pass

	def evaluate_solution_foot_pos(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		"""
		Get solution from solver and evaluate it for time(s) t.
		This will pull the real optimized values out of opti first.
		:param t:
		:param t_shift: shift the output in time by this value, if None do not shift.
		:param t_max_duration: relevant when using rolling_time_shift_duration. This will be set to the total duration minus duration of all previous phases.
								Acts as an upper bound on the phase duration. The used phase duration is min(phase_duration, t_max_duration)
		"""
		pass

	def evaluate_solution_foot_force(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		"""
		Get solution from solver and evaluate it for time(s) t.
		This will pull the real optimized values out of opti first.
		:param t:
		:param t_shift: shift the output in time by this value, if None do not shift.
		:param t_max_duration: relevant when using rolling_time_shift_duration. This will be set to the total duration minus duration of all previous phases.
								Acts as an upper bound on the phase duration. The used phase duration is min(phase_duration, t_max_duration)
		"""
		pass

	def evaluate_solution_duration(self) -> float:
		"""
		Get solution from solver and evaluate it for duration of this phase.
		This will pull the real optimized values out of opti first.
		"""
		return self.opti.value(self.duration)


	def _debug_evaluate_solution_plot_force_or_pos(self):
		pass




@serializable_opti_variables(
	['foot_force']
)
class ContactPhase(Phase, SerializableOptiVariables):
	foot_position: casadi.MX
	foot_force: SplineTrajectory

	foot_force_start_dx_opti: casadi.MX
	foot_force_end_dx_opti: casadi.MX

	def __init__(self,
				 num_polynomials_foot_force_in_contact,
				 start_t,
				 start_foot_pos, start_foot_dpos, start_foot_force, start_foot_dforce,
				 end_foot_force, end_foot_dforce,
				 opti: casadi.Opti,
				 dimensions,
				 phase_duration: float = None,
				 scope_name: list[str] = []
				 ):
		super().__init__(start_t, opti, dimensions, phase_duration, scope_name + ['ContactPhase'])

		# variables: test dx at start and end flexible
		#self.foot_force_start_dx_opti = opti.variable(self.d) #if start_foot_dforce is None else start_foot_dforce
		#self.foot_force_end_dx_opti = opti.variable(self.d) #if end_foot_dforce is None else start_foot_dforce
		#self.foot_force_start_dx_opti = start_foot_dforce

		self.foot_position = start_foot_pos
		self.foot_force = SplineTrajectory(
			num_polynomials_foot_force_in_contact,
			param_opti_start_x=start_foot_force,
			param_opti_start_dx=start_foot_dforce,
			opti=opti,
			param_opti_end_x=end_foot_force,  # the force in next phase will be zero, so end the end of this phase it is zero too, or variable at last phase
			param_opti_end_dx=end_foot_dforce,	# the force in next phase will be zero, so end the end of this phase it is zero too, or variable at last phase
			given_total_duration=self.duration,#None
			#given_total_duration_portion_deltaT_for_first_and_last_poly=0.15, # 0.25, # for hopper 0.15
			dimensions=self.d,
			scope_name=scope_name + ['ContactPhase', 'foot_force']
		)
		# enforce total time if time of spline parts is flexible
		# opti.subject_to(self.foot_force.get_sum_detaT() == self.duration)

		# for now ensure end values via constraints
		# the force in next phase will be zero, so end the end of this phase it is zero too
		# @todo this can be enforced directly by using end values as x1, dx1 in SplineTrajectory
		#self.foot_force.poly_list[-1].x1 = 0
		#self.foot_force.poly_list[-1].dx1 = 0
		#opti.subject_to(self.foot_force.poly_list[-1].x1 == 0)
		#opti.subject_to(self.foot_force.poly_list[-1].dx1 == 0)

	def evaluate_foot_pos(self, t):
		# need to make sure that we just output the pos when this phase is active
		# @todo this is enforced twice, but only need once
		#return SplineTrajectory.val_if_in_range(range_start_t=-0.001, range_end_t=self.duration, t=t, value=self.foot_position)
		return self.foot_position

	def evaluate_foot_pos_start(self, derivative=False):
		return self.foot_position if not derivative else 0.0

	def evaluate_foot_pos_end(self, derivative=False):
		return self.foot_position if not derivative else 0.0


	def evaluate_foot_dpos(self, t):
		# need to make sure that we just output the pos when this phase is active
		return casadi.MX.zeros(self.d)
		#return SplineTrajectory.val_if_in_range(range_start_t=0, range_end_t=self.duration, t=t,
		#										value=0)

	def evaluate_foot_force(self, t):
		return self.foot_force.evaluate_x(t)

	def get_phase_type(self) -> PhaseType:
		return PhaseType.CONTACT

	########
	# Evaluation of solved variables
	def evaluate_solution_foot_pos(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		values = np.repeat(self.opti.value(self.foot_position)[:, np.newaxis], t.shape[0], axis=1)
		values[:, np.logical_or(t < 0, t >= 0+self.opti.value(self.duration))] = 0
		values[:, t >= t_max_duration] = 0
		return PhaseTrajectoryEvaluated(
			values=values,
			spline_connection_points=SplineTrajectory.EvalSplineConnectionPoints(self.d)	# empty points list
		)

	def evaluate_solution_foot_force(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		x_vals, spline_connection_points = self.foot_force.evaluate_solution_x(
			t,
			t_upper_bound=t_max_duration,
			make_nans_for_t_out_of_range=False,
			output_spline_connection_points=True)
		return PhaseTrajectoryEvaluated(x_vals, spline_connection_points)




@serializable_opti_variables(
	['foot_position']
)
class FlightPhase(Phase, SerializableOptiVariables):
	foot_position: SplineTrajectory

	# foot_force: casadi.SX # is allways zero

	def __init__(self, num_polynomials_foot_pos_in_flight,
				 start_t,
				 start_foot_pos, start_foot_dpos, start_foot_force, start_foot_dforce,
				 # end_foot_pos, end_foot_dpos,
				 opti: casadi.Opti,
				 dimensions,
				 phase_duration: float = None,
				 scope_name: list[str] = []
				 ):
		super().__init__(start_t, opti, dimensions, phase_duration, scope_name + ['FlightPhase'])
		self.foot_position = SplineTrajectory(
			num_polynomials_foot_pos_in_flight,  # 2
			start_foot_pos,
			start_foot_dpos,
			opti,
			param_opti_end_dx=np.zeros(self.d), # position in next phase will be constant, so end the end of this phase its derivative is zero two
			given_total_duration=self.duration,
			dx1_z_fixed_zero=True, # dx1 values are fixed to zero and not optimized
			dimensions=self.d,
			scope_name=scope_name + ['FlightPhase', 'foot_position']
		)
		self.foot_force = start_foot_force
		# for now ensure end values via constraints
		# @todo this can be enforced directly by using end values as x1, dx1 in SplineTrajectory
		# position in next phase will be constant, so end the end of this phase its derivative is zero two
		#opti.subject_to(self.foot_position.poly_list[-1].dx1 == 0)

	def evaluate_foot_pos(self, t):
		return self.foot_position.evaluate_x(t)

	def evaluate_foot_pos_start(self, derivative=False):
		return self.foot_position.poly_list[0].x0 if not derivative else self.foot_position.poly_list[0].dx0

	def evaluate_foot_pos_end(self, derivative=False):
		return self.foot_position.poly_list[-1].x1 if not derivative else self.foot_position.poly_list[-1].dx1

	def evaluate_foot_dpos(self, t):
		return self.foot_position.evaluate_dx(t)

	def evaluate_foot_force(self, t):
		return 0

	def get_phase_type(self) -> PhaseType:
		return PhaseType.FLIGHT


	########
	# Evaluation of solved variables
	def evaluate_solution_foot_pos(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		x_vals, spline_connection_points = self.foot_position.evaluate_solution_x(
			t,
			make_nans_for_t_out_of_range=False,
			t_upper_bound=t_max_duration,
			output_spline_connection_points=True)
		return PhaseTrajectoryEvaluated(x_vals, spline_connection_points)

	def evaluate_solution_foot_force(self, t: np.array, t_max_duration: float) -> PhaseTrajectoryEvaluated:
		return PhaseTrajectoryEvaluated(
			values=np.zeros((self.d, t.shape[0])),
			spline_connection_points=SplineTrajectory.EvalSplineConnectionPoints(self.d)	# empty points list
		)

	def _debug_evaluate_solution_plot_force_or_pos(self):
		self.foot_position.evaluate_solution_and_plot_full_trajectory(show_plot=False)
		plt.title("pos of phase")