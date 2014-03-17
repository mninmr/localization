import utils
import argparse
from time import sleep
import numpy as np
import math
# Throughout the documentation we use the following abbreviations
# M - number of observations used at each timestep
# N - the number of starting positions
# T - number of timesteps
# X - width of the map
# Y - height of the map

#DO NOT MODIFY ANY TYPE SIGNATURES, all functions will be tested.
#You may add any additional helper function you wish

def generate_init_states(lidar_map, num_states):
	"""
	:param lidar_map: the input lidar map represented as a matrix
	:param num_states: - an integer representing the number of initial states to create, N
	:return: an Nx3 element numpy array representing N states. Each state should be initialized in
		an open space in the map (represented by a zero at the position).
		Initial phi should be a randomly generated float between 0 and 2pi.
	NOTE: You must index into the lidar_map in [y,x].
	For example, to access the cell at x=2,y=4 you would use the line:
	lidar_map[4,2]
	(This is a result of MatPlotLib's graphing software.)
	Additionally:
	height, width = lidar_map.shape
	"""

   	arr = []; ct = 0

   	while ct < num_states:
   		xcoord = np.random.randint(low=0, high=lidar_map.shape[1]-1)
   		ycoord = np.random.randint(low=0, high=lidar_map.shape[0]-1)
 		
 		# check space is unoccupied
	   	while lidar_map[ycoord][xcoord] == 1: # unfortunate numpy convension
	   		xcoord = np.random.randint(low=0, high=lidar_map.shape[1]-1)
	   		ycoord = np.random.randint(low=0, high=lidar_map.shape[0]-1)

   		phi = np.random.uniform(low=0.0, high=2*math.pi)
   		arr.append([xcoord, ycoord, phi])
   		ct += 1

   	return np.array(arr, dtype=float)

def update_state(state, control):
	"""
	:param state: - a 3 element numpy array of (xcoord, ycoord, phi)
	:param control: - a 2 element numpy array of (delta phi, velocity)
	:returns: a new state in the form of a numpy array, with the degree updated by the change
		in degree given in the control and  the x and y coordinates updated by moving 
		velocity in the new direction (update direction, then position)
	"""
	state[2] += control[0]
	state[0] += control[1] * np.cos(state[2])
	state[1] += control[1] * np.sin(state[2])

	return state

def measurement_prob(state, measurements, lidar_map):
	"""
	:param state: an (xcoord, ycoord, phi) numpy array representing one current position and direction
	:param measurement: Mx2 numpy arrays of M (displacement, distance) pairs representing the recorded distance to an object
		at heading displacement from the current phi of the agent
	:return: a probability (0-1) of seeing the given measurement given the current position

	You should use a gaussian distribution with a mean of 0 and a variance of 3
	to calculate this probability of the difference between observed and actual.
	"""

	prob = 1
	
	for measurement in measurements:

		actual_displacement = calculate_actual_displacement(state, measurement, lidar_map)
		observed_displacement = measurement[1]
		displacement_diff =  observed_displacement - actual_displacement

		prob *= utils.gauss_prob(displacement_diff,0,3)

	return prob

def calculate_actual_displacement(state, measurement, lidar_map):
	"""
	Calculate the actual distance to the wall of the current measurement direction
	"""

	xcoord = state[0]; ycoord = state[1]; phi = state[2]
	direction = measurement[0] + phi
	step = 0

	updated_xcoord = xcoord
	updated_ycoord = ycoord

	# check indices
	if updated_ycoord < 0:
		updated_ycoord = 0
	if updated_xcoord < 0:
		updated_xcoord = 0
	if updated_ycoord > lidar_map.shape[0]-1:
		updated_ycoord = lidar_map.shape[0]-1
	if updated_xcoord > lidar_map.shape[1]-1:
		updated_xcoord = lidar_map.shape[1]-1

	while lidar_map[round(updated_ycoord)][round(updated_xcoord)] == 0:
		updated_ycoord = ycoord + step*np.sin(direction)
		updated_xcoord = xcoord + step*np.cos(direction)
		step += 1

		# check indices
		if updated_ycoord < 0 or updated_xcoord < 0 or \
			updated_ycoord > lidar_map.shape[0]-1 or updated_xcoord > lidar_map.shape[1]-1:
			break

	return step

def resample(states, weights):
	"""
	Creates a new states vector by resampling them according to the weights.
	This step should also add gaussian noise with mean of 0 and variance of 0.5.

	:param states: an Nx3 array of tuples (xcoord, ycoord, phi)
	:param weights: a Nx1 array of resampling weights
	:return: an Nx3 array of tuples (xcoord, ycoord, phi)
	"""
	
	toret = []
	for ii in range(0, states.shape[0]):
		factor = np.random.multinomial(1, weights, size=1)[0]
		index = indices(factor, lambda x:x==1)[0]
		new_state = add_noise(states[index])
		toret.append(new_state)

	return np.array(toret, dtype=float)

def indices(a, func):
    return [i for (i, val) in enumerate(a) if func(val)]

def add_noise(state):
	"""
	Add Gaussian noise to the state variables
	"""
	xcoord = state[0]; ycoord = state[1]; phi = state[2]

	# add noise
	xcoord += utils.gauss_sample(.0, .5)
	ycoord += utils.gauss_sample(.0, .5)
	phi += utils.gauss_sample(.0, .1*math.pi)

	return [xcoord, ycoord, phi]

def particle_filter(start_states, controls, measurements, lidar_map, grounding, vis=False):
	"""
	:param start_states: - an Nx3 array of (x-coordinate, y-coordinate, phi), the initial candidate points.
	:param controls: - a Tx2 array of (degree change, velocity), the movement at each time step.
	:param measurements: - a TxMx2 array of (angle, distance), the measured distance to an obstacle in that direction at that timestep.
	:param lidar_map: - an XxY array representing a discretized map. Each position in the array represents
				whether the space is empty, represented by a 0, or filled (an obstacle), represented by a 1.
				Generated by calling utils.load_map on the appropriate .pgm file.
	:param grounding: - a (T+1)x3 array representing the true position of the agent from initialization to end
    	:param vis: - a flag dictating whether to run the visualizer or not. You should NOT make any changes that result
				in the visualizer being called when vis=False
    	:returns: An Nx3 array of (x-coordinate, y-coordinate, phi) representing the final states.
    	"""
    
	if vis:
		utils.init_vis(lidar_map, start_states, grounding[0,:])

	updated_states = start_states

	# This is the main loop of the particle filter.
	# This  code will display the updated locations of your particles, with the true position in red.
	for t in range(len(measurements)):  #len(measurements) is the number of timesteps

		prob = []
		for state in updated_states:

			backup = state
			new_state = update_state(state, controls[t]) # freaking python list mutable pass by reference F*CK
			if not check_updated_state_validity(new_state, lidar_map):
				state = backup # invalid updated state restored
			
			prob.append(measurement_prob(state, measurements[t], lidar_map))

		if sum(prob):
			updated_states = resample(updated_states, normalize(prob))
		else:
			updated_states = generate_init_states(lidar_map, updated_states.shape[0])

		if vis:
			utils.update_vis(lidar_map, updated_states, grounding[t+1,:])
			sleep(.1)
	return updated_states

def normalize(distribution):
	"""
	Normalize a given probability distribution 
	"""
	normalizer = sum(distribution)
	if normalizer:
		return distribution / normalizer
	else:
		return distribution

def check_updated_state_validity(state, lidar_map):
	"""
	Check if the updated state is out of bound
	"""

	xcoord = state[0]; ycoord = state[1]

	if xcoord < 0.0 or xcoord > lidar_map.shape[1]-1:
		return False
	if ycoord < 0.0 or ycoord > lidar_map.shape[0]-1:
		return False
	if lidar_map[ycoord][xcoord] == 1:
		return False
	return True


#MAIN METHOD - DO NOT MODIFY
def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('-t', '--test', help="A test directory containing map.pgm, measure.csv, control.csv, and ground.csv files", required=True)
	parser.add_argument('-s', '--states', help='The file containing the starting states to use')
	parser.add_argument('-v', '--visualizer', action='store_const', const=True, help='Add this flag to turn on the visualizer', default=False)
	parser.add_argument('-n', '--numstart', type=int, default=200)

	args = parser.parse_args()
	lmap = utils.load_map('tests/' + args.test + '/map.pgm')
	controls = utils.load_csv('tests/' + args.test + '/control.csv') #a Tx2 array of T (delta phi, velocity)'s
	measurements = utils.load_measurements('tests/' + args.test + '/measure.csv') #a TxMx2 array of T sets of M measurements containing a degree and a measured distance at that degree
	true_start = utils.load_csv('tests/' + args.test + '/ground.csv')
	if args.states:
		start_posns = utils.load_csv(args.states) #a Nx3 array of N (x,y,phi)'s
	else:
		start_posns = generate_init_states(lmap, args.numstart)

	print("Using particle_filter function...")
	particle_filter(start_posns, controls, measurements, lmap, true_start, args.visualizer)

if __name__ == "__main__":
	main()
