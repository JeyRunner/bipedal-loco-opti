from unittest import TestCase

import matplotlib.pyplot as plt
import numpy as np
import rich

from centroidal_walk.collocation.opti.collocation_opti import CentroidalDynPolyOpti
from centroidal_walk.collocation.opti.collocation_opti_data import CentroidalDynPolyOptiData
from centroidal_walk.collocation.opti.initial_gait import INIT_GAIT_TYPE
from centroidal_walk.collocation.serialization.OptiDecorators import *
from centroidal_walk.collocation.spline.spline_trajectory import *
from centroidal_walk.collocation.opti.casadi_util.OptiLoadable import *
from examples.biped_params import *


class TestSerializeSplineTrajectory(TestCase):

	def test_serialize_CentroidalDynPolyOptiData(self):
		opti = CentroidalDynPolyOpti(
			mass=biped_example_mass,
			InertiaMatrix=biped_example_InertiaMatrix,
			use_two_feet=True,
			foot_force_max_z=1000,  # 150,
			foot_kin_constraint_box_center_rel=biped_example_foot_kin_constraint_box_center_rel,
			foot_kin_constraint_box_size=biped_example_foot_kin_constraint_box_size,  # test
			total_duration=3,  # 3,
			num_phases=biped_example_num_steps,
			fixed_phase_durations=None,  # hopper_example_initial_phase_durations
			base_poly_duration=1,  # 0.05,
			use_angular_dynamics=True,
			# additional variables and constraints:
			foot_force_at_trajectory_end_and_start_variable=True,
			max_com_angle_xy_abs=0.2,  # can help with convergence
			# additional_intermediate_foot_force_constraints=True,  # just works well in combination with max_com_angle_xy_abs
			additional_foot_flight_smooth_constraints=True
		)

		opti.add_additional_constraint__com_lin_z_range_of_motion(max_z_deviation_from_init=0.05)
		opti.add_additional_constraint__com_angular_acc(max_xy_acc=1)

		opti_params = opti.serialize_opti_parameters()
		rich.print(opti_params)

		# solve with params
		end_com_pos = np.ones(3)*0.1
		end_com_pos[-1] = biped_example_foot_z_offset
		opti.solve_opti(
			just_show_init_values=False,
			start_com_pos=biped_example_start_com_pos,
			start_feet_pos=biped_example_start_feet_pos,
			end_com_pos=end_com_pos,
			max_iter=1,
			init_gait_type=INIT_GAIT_TYPE.ALL_FEET_JUMP
		)

		# save solution
		solution_serialized = opti.serialize_solution()
		rich.print(solution_serialized)



		# load params
		opti_loaded = CentroidalDynPolyOptiData.create_solution_loader(opti_params)
		rich.print(opti_loaded.serialize_opti_parameters())
		self.assertDictEqual(
			opti_params,
			opti_loaded.serialize_opti_parameters()
		)

		# load solution
		opti_loaded.load_solution(solution_serialized)
		self.assertEquals(
			yaml.dump(solution_serialized, indent=2),
			yaml.dump(opti_loaded.serialize_solution(), indent=2)
		)



		# compare trajectories
		# compare loaded and orig
		t_vals = np.arange(-1, 6+1, step=0.01)
		def trajectories_equal(f_orig, f_loaded):
			x_orig = opti.value(f_orig(t_vals))
			x_loaded = opti_loaded.value(f_loaded(t_vals))
			np.testing.assert_allclose(x_orig, x_loaded)

			# plt.plot(t_vals, x_orig[0].T, label='orig')
			# plt.plot(t_vals, x_loaded[0].T, label='loaded', linestyle='dashed')
			# plt.legend()
			# plt.show()

		trajectories_equal(opti.x_opti_com_pos.evaluate_x, opti_loaded.x_opti_com_pos.evaluate_x)
		trajectories_equal(opti.x_opti_com_pos.evaluate_dx, opti_loaded.x_opti_com_pos.evaluate_dx)
		trajectories_equal(opti.x_opti_com_pos.evaluate_ddx, opti_loaded.x_opti_com_pos.evaluate_ddx)

		trajectories_equal(opti.x_opti_com_angle.evaluate_x, opti_loaded.x_opti_com_angle.evaluate_x)
		trajectories_equal(opti.x_opti_com_angle.evaluate_dx, opti_loaded.x_opti_com_angle.evaluate_dx)
		trajectories_equal(opti.x_opti_com_angle.evaluate_ddx, opti_loaded.x_opti_com_angle.evaluate_ddx)

		for foot_i in range(opti.num_feet):
			trajectories_equal(opti.x_opti_feet[foot_i].evaluate_foot_pos, opti_loaded.x_opti_feet[foot_i].evaluate_foot_pos)
			trajectories_equal(opti.x_opti_feet[foot_i].evaluate_foot_force, opti_loaded.x_opti_feet[foot_i].evaluate_foot_force)



		#opti_loaded.plot_animate_all()