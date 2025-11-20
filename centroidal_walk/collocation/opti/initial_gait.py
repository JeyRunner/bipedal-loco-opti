#from centroidal_walk.collocation import collocation_opti
import numpy as np

from centroidal_walk.collocation.foot_trajectory.foot_gait_phases_types import PhaseType
from centroidal_walk.collocation.spline.spline_trajectory import *
from enum import Enum


class INIT_GAIT_TYPE(Enum):
	ALL_FEET_JUMP = 0
	ALL_ZERO_NO_MOVEMENT = 1
	ALTERNATING_SINGLE_SUPPORT_PHASES = 1
	#ALTERNATING_SINGLE_AND_DOUBLE_SUPPORT_PHASES = 2
	ALTERNATING_SINGLE_FOOT_FLIGHT = 3
	TIMINGS_GIVEN = 4


def gen_initial_gait(
		collo_opti: 'collocation_opti.CentroidalDynPolyOpti',
		init_gait_type: INIT_GAIT_TYPE,
		start_feet_pos: np.array,
		given_flight_phase_durations: list[np.array] = None,
		given_contact_phase_durations: list[np.array] = None,
		):
	"""
	This will initialize the opti variables of the feet and com to imitate a simple gait sequence.
	This is depended on the initial and final conditions.
	:param collo_opti:
	:param init_gait_type: the gait type to use as initialization.
	:param start_feet_pos: the initial position of the feet.
	"""
	total_duration = collo_opti.total_duration
	num_feet = len(collo_opti.x_opti_feet)

	print()
	print('## gen_initial_gait:')

	# simple init
	if init_gait_type == INIT_GAIT_TYPE.ALL_ZERO_NO_MOVEMENT:
		collo_opti.x_opti_com_pos.set_initial_opti_values__x1_z(
			initial_x1_z_middle=0,
			initial_x1_z_end=0,
			interpolation_xy_start=np.zeros(2),
			interpolation_xy_end=np.zeros(2)
		)
		for foot_i, foot in enumerate(collo_opti.x_opti_feet):
			num_phases = len(foot.phases)
			phase_duration_nominal = total_duration / num_phases
			foot.set_initial_opti_values(
				init_for_duration_of_each_phase=phase_duration_nominal,
				init_z_height_for_flight=0,
				init_z_dheight_for_flight=0,
				init_z_force_for_contact=0,
				init_z_dforce_for_contact=0,
				interpolation_xy_start=np.zeros(2),
				interpolation_xy_end=np.zeros(2),
				use_interpolted_dx_for_foot_pos=False
			)
			print(f'# foot {foot_i}:')
			print('> init_durations_contact_phases', phase_duration_nominal)
			print('> init_durations_flight_phases', phase_duration_nominal)
			print()
		return



	# init com as interpolation
	start_com_pos = np.array(collo_opti.opti.value(collo_opti.param_opti_start_com_pos))[0:2]
	end_com_pos = np.array(collo_opti.opti.value(collo_opti.param_opti_end_com_pos))[0:2]
	collo_opti.x_opti_com_pos.set_initial_opti_values__x1_z(
		initial_x1_z_middle=collo_opti.opti.value(collo_opti.param_opti_start_com_pos)[-1],  # 0.8,
		initial_x1_z_end=collo_opti.opti.value(collo_opti.param_opti_end_com_pos)[-1],
		interpolation_xy_start=start_com_pos,
		interpolation_xy_end=end_com_pos
	)


	feet_first_phase_is_contact = np.array(
		[foot.phases[0].get_phase_type() == PhaseType.CONTACT for foot in collo_opti.x_opti_feet]
	)
	feet_all_first_phase_is_contact = np.all(feet_first_phase_is_contact)
	feet_all_first_phase_is_flight = np.all(np.logical_not(feet_first_phase_is_contact))
	feet_atleastone_first_phase_is_contact = np.any(feet_first_phase_is_contact)
	feet_atleastone_first_phase_is_flight = np.any(np.logical_not(feet_first_phase_is_contact))

	# init values for all feet
	for foot_i, foot in enumerate(collo_opti.x_opti_feet):
		num_phases = len(foot.phases)
		num_phases_contact = len(list(foot.get_contact_phases()))
		num_phases_flight = len(list(foot.get_flight_phases()))

		phase_duration_nominal = total_duration / num_phases
		init_durations_contact_phases = np.ones(num_phases_contact)
		init_durations_flight_phases = np.ones(num_phases_flight)

		# gait init types
		if init_gait_type == INIT_GAIT_TYPE.ALL_FEET_JUMP:
			init_durations_contact_phases *= phase_duration_nominal
			init_durations_flight_phases *= phase_duration_nominal

		elif init_gait_type == INIT_GAIT_TYPE.ALTERNATING_SINGLE_SUPPORT_PHASES:
			assert num_phases_contact == num_phases_flight + 1 and (
				foot.phases[0].get_phase_type() == PhaseType.CONTACT
			), "assume that first and last phase is a contact phase"
			long_contact_duration = (total_duration / (num_phases+1))*2
			phase_duration_nominal_reduced = (total_duration - long_contact_duration) / (num_phases - 1) # without extended phase
			init_durations_contact_phases *= phase_duration_nominal_reduced
			init_durations_flight_phases *= phase_duration_nominal_reduced
			if foot_i % 2 == 0:
				init_durations_contact_phases[0] = long_contact_duration
			else:
				init_durations_contact_phases[-1] = long_contact_duration

		if init_gait_type == INIT_GAIT_TYPE.ALTERNATING_SINGLE_FOOT_FLIGHT:
			if feet_all_first_phase_is_flight:
				raise RuntimeError("All feet have a flight as first phase, this is not supported for ALTERNATING_SINGLE_FOOT_FLIGHT")
			elif feet_all_first_phase_is_contact:
				#init_durations_contact_phases *= phase_duration_nominal
				#init_durations_flight_phases *= phase_duration_nominal
				#long_contact_duration = (total_duration / (num_phases+1))*2
				#phase_duration_nominal_reduced = (total_duration - long_contact_duration) / (num_phases - 1) # without extended phase
				#for i in range(num_phases_flight):
				#	if (i + foot_i) % num_feet:
				#		#init_durations_contact_phases[i] = 0.5
				#		init_durations_flight_phases[i] = phase_duration_nominal_reduced
				#	else:
				#init_durations_contact_phases /= num_feet
				init_durations_flight_phases /= num_feet
				init_durations_contact_phases[0] *= (foot_i+1) / num_feet
				init_durations_contact_phases[-1] *= (num_feet-foot_i) / num_feet
			# normalize
			#init_durations_contact_phases *= (total_duration-np.sum(init_durations_flight_phases))/np.sum(init_durations_flight_phases)
			sum_units = np.sum(init_durations_contact_phases) + np.sum(init_durations_flight_phases)
			init_durations_contact_phases *= total_duration/sum_units
			init_durations_flight_phases *= total_duration/sum_units
			#if foot_i % num_feet:
			#	init_durations_contact_phases[0] = long_contact_duration
			#else:
			#	init_durations_contact_phases[-1] = long_contact_duration

		if init_gait_type == INIT_GAIT_TYPE.TIMINGS_GIVEN:
			de_norm_factor = (total_duration)*(1/(np.sum(given_contact_phase_durations[foot_i]) + np.sum(given_flight_phase_durations[foot_i])))
			init_durations_contact_phases = given_contact_phase_durations[foot_i]*de_norm_factor
			init_durations_flight_phases = given_flight_phase_durations[foot_i]*de_norm_factor
		else:
			assert given_contact_phase_durations is None
			assert given_flight_phase_durations is None



		print(f'# foot {foot_i}:')
		print('> init_durations_contact_phases', init_durations_contact_phases)
		print('> init_durations_flight_phases', init_durations_flight_phases)
		print()

		# check
		sum_durations = np.sum(init_durations_contact_phases) + np.sum(init_durations_flight_phases)
		assert np.isclose(sum_durations, total_duration), \
			f"The duration of all contact and flight phases ({sum_durations}s) does not sum up to the given total duration ({total_duration}s)"

		#continue

		# start_foot_pos = np.array(collo_opti.opti.value(collo_opti.param_opti_start_foot_pos[foot_i, :]))[0:2]
		start_foot_pos = np.copy(start_feet_pos[foot_i, 0:2])

		end_foot_pos = start_foot_pos + (end_com_pos - start_com_pos)
		foot.set_initial_opti_values(
			#init_for_duration_of_each_phase=collo_opti.total_duration / len(foot.phases),
			init_durations_contact_phases=init_durations_contact_phases,
			init_durations_flight_phases=init_durations_flight_phases,
			init_z_height_for_flight=0,#0.25,
			init_z_dheight_for_flight=0*5,
			init_z_force_for_contact=collo_opti.mass * collo_opti.gravity_acc,  # + 50,#150,
			init_z_dforce_for_contact=0,#collo_opti.mass * collo_opti.gravity_acc,  # 150
			interpolation_xy_start=start_foot_pos,
			interpolation_xy_end=end_foot_pos,
			use_interpolted_dx_for_foot_pos=False
		)


	# set feet start pos when initial foot pos is optimized
	if collo_opti.params_to_optimize.param_opti_start_foot_pos:
		collo_opti.opti.set_initial(collo_opti.param_opti_start_foot_pos, start_feet_pos)
		print("> init param_opti_start_foot_pos to ", start_feet_pos)

	print("> init force to ", collo_opti.mass * collo_opti.gravity_acc)
