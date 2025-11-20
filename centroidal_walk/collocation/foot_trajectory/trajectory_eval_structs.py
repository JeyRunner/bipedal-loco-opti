import numpy as np

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.spline.spline_trajectory import SplineTrajectory

class PhaseTrajectoryEvaluated:
	values: np.array
	spline_connection_points: SplineTrajectory.EvalSplineConnectionPoints


	def __init__(self, values, spline_connection_points):
		self.values = values
		self.spline_connection_points = spline_connection_points



class FootTrajectorySolutionEvaluated:
	foot_pos_trajectory: PhaseTrajectoryEvaluated
	foot_force_trajectory: PhaseTrajectoryEvaluated

	phase_end_times: np.array
	phase_types: PhaseType
