from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from centroidal_walk.collocation.opti.collocation_opti_data import CentroidalDynPolyOptiData

from centroidal_walk.collocation.opti.initial_gait import gen_initial_gait, INIT_GAIT_TYPE
from rich import print

from centroidal_walk.collocation.spline.polynomial4 import Polynomial4
from tests.test_util import *

from centroidal_walk.collocation.spline.spline_trajectory import *

class TestInitialGait(TestCase):

	def dummy_solve(self, opti: CentroidalDynPolyOpti):
		opti.opti.set_value(opti.param_opti_start_com_pos, np.zeros(opti.d))
		opti.opti.set_value(opti.param_opti_start_com_dpos, np.array([
			0, 0, 0
		]))
		opti.opti.set_value(opti.param_opti_start_com_angle, np.zeros(opti.d))
		opti.opti.set_value(opti.param_opti_start_com_dangle, np.zeros(opti.d))

		# feet
		opti.opti.set_value(opti.param_opti_start_foot_pos, np.zeros((opti.num_feet, opti.d)))
		opti.opti.set_value(opti.param_opti_start_foot_dpos, np.zeros((opti.num_feet, opti.d)))

		# end pos for com
		opti.opti.set_value(opti.param_opti_end_com_pos, np.zeros((opti.d)))

		opti.opti.bake_solve_with_simple_var_bounds('ipopt', dict(), dict(max_iter=0))
		#opti.solver('ipopt')
		try:
			opti.opti.solve_with_simple_var_bounds()
			#opti.solve()
		except Exception as e:
			raise e
		# opti.solve_opti(
		#
		# )

	def test_4feet_ALTERNATING_SINGLE_FOOT_FLIGHT(self):
		opti = CentroidalDynPolyOptiData(
			num_feet=4,
			mass=1,
			num_phases=5+2,
			InertiaMatrix=None,
			foot_kin_constraint_box_size=None,
			foot_kin_constraint_box_center_rel=[None, None, None, None],
			foot_force_max_z=None
		)
		gen_initial_gait(opti, init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT)


	def test_2feet_ALTERNATING_SINGLE_FOOT_FLIGHT(self):
		opti = CentroidalDynPolyOpti(
			num_feet=2,
			mass=1,
			num_phases=5+2,
			InertiaMatrix=np.eye(3),
			foot_kin_constraint_box_size=np.ones(3),
			foot_kin_constraint_box_center_rel=[np.zeros(3), np.zeros(3)],
			base_poly_duration=1.0,
			foot_force_max_z=1000,
		)

		opti.solve_opti(
			start_com_pos=np.array([0,0, 1]),
			start_feet_pos=np.zeros((2, 3)),
			end_com_pos=np.array([2, 4, 0.8]),
			init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT,
			just_show_init_values=True
		)
		#gen_initial_gait(opti, init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES)
		#self.dummy_solve(opti)


		opti.plot_animate_all(show_animate=False, show_plots_angular_dyn=False)



	def test_2feet_phases4_first_flight_ALTERNATING_SINGLE_FOOT_FLIGHT(self):
		opti = CentroidalDynPolyOpti(
			num_feet=2,
			mass=1,
			num_phases=4,
			feet_first_phase_type=[PhaseType.CONTACT, PhaseType.FLIGHT],
			InertiaMatrix=np.eye(3),
			foot_kin_constraint_box_size=np.ones(3),
			foot_kin_constraint_box_center_rel=[np.zeros(3), np.zeros(3)],
			base_poly_duration=1.0,
			foot_force_max_z=1000,
		)

		opti.solve_opti(
			start_com_pos=np.array([0, 0, 1]),
			start_feet_pos=np.zeros((2, 3)),
			end_com_pos=np.array([2, 4, 0.8]),
			init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT,
			just_show_init_values=True
		)
		#gen_initial_gait(opti, init_gait_type=INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES)
		#self.dummy_solve(opti)


		opti.plot_animate_all(show_animate=False, show_plots_angular_dyn=False)
