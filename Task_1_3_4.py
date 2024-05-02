import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse

class Node:

	def __init__(self, value, number, connections=None):

		self.index = number
		self.connections = connections
		self.value = value
		self.parent = None

	def get_neighbours(self):

		return np.where(np.array(self.connections)==1)[0]


class Queue:

	def __init__(self):
		self.queue = []

	def push(self, item):
		self.queue.append(item)
		
	def pop(self):
		if not self.is_empty():
			return self.queue.pop(0)
	
	def is_empty(self):
		return len(self.queue)==0


class Network: 

	def __init__(self, nodes=None):

		if nodes is None:
			self.nodes = []
		else:
			self.nodes = nodes 

	def get_mean_degree(self):
		'''
		This Function Calulates the mean degree of a network of nodes
		(Average number of connections made by each node)
		'''

		total_degree = 0

		#Uses a for loop to cycle through each node
		for node_number in self.nodes:

			#Count the number of connection at that node, adding to 'total_degree'
			total_degree += node_number.connections.count(1)
		
		#Returns total number of connections divided by number of nodes
		return total_degree/len(self.nodes)

	def get_mean_clustering(self):
		'''
		This Function calculates the Mean Clustering Coefficient, 
		measures the fraction of a node's neighbours that connect to one another
		'''

		mean_coefficient = 0

		# Uses a for loop to look through each node
		for node in self.nodes:
			count = 0

			#Get's the neighnours of the node, and number of neighbours
			node_neighbours = node.get_neighbours()
			neigh_num = len(node_neighbours)

			#For each of the neighbours of the original node, get their neighbours
			for neighbour in node_neighbours:
				neighbour_neighbours = self.nodes[neighbour].get_neighbours()

				#Checking if they share a same neighbour
				for check_neighbours in neighbour_neighbours:

					#If neighbour is shared the count is increased by 1
					if check_neighbours in node_neighbours:
						count += 1

			#Calculating the possible connections for each node
			possible_connections = (neigh_num*(neigh_num - 1))/2

			#For networks with very few nodes, there may be no connections 
			if possible_connections == 0:
				node_coefficient = 0

			#Calulate the node coefficient for the specific node
			else:
				node_coefficient = (count/2)/possible_connections

			#Sums all node coefficients
			mean_coefficient += node_coefficient
		

		#Returns the mean node coefficient
		return mean_coefficient/len(self.nodes)

	def get_mean_path_length(self):
		'''
		This Function calulates the mean path length, 
		which is the average distance to all nodes it is connected to
		'''
		
		total = 0

		#Uses 2 for loops to compare every node to one another
		for node_number_start in self.nodes:
			for node_number in self.nodes:

				#Ensures not trying to find path length of the same node started at
				if node_number_start != node_number:

					path_queue = Queue()
					#Pushes the node, trying to find the distance from, to the front of the list
					path_queue.push(node_number_start)
					path = []

					while not path_queue.is_empty():
						#Check the node at the top of the queue
						check_node = path_queue.pop()

						#Checks if reach the goal node
						if check_node == node_number:
							break
						
						#Loops through neighbours nodes pushing them to the list, to be checked
						for neighbour_node in check_node.get_neighbours():
							neighbour = self.nodes[neighbour_node]
							if neighbour_node not in path:
								path_queue.push(neighbour)
								path.append(neighbour_node)

								neighbour.parent = check_node

					check_node = node_number
					node_number_start.parent = None
					route = []
					
					# Creates a list containing the route from the start to goal node
					while check_node.parent:
						route.append(check_node)
						check_node = check_node.parent
					route.append(check_node)
					
					#Checks the length of the list
					total += len(route)-1
		
		n = len(self.nodes)
	
		mean_path = total/(n*(n-1))
		
		return mean_path


	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []

		# Creates N nodes
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		# Assigns random connectivity based on given probability
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1


	def make_ring_network(self, N, neighbour_range=1):
		'''
		This function makes a ring network of size N.
		
		The ring has a range of 1 meaning each node is connected to its neighbours only.
		'''
		#stores the nodes in a list with corresponding value, number and connections
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))
		
		#loops through the nodes in the ring network
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(N):
				
				#modulus ensures that the code will loop through the index of neighbours	
				#prev_index is the previous node before the current node in the loop
				prev_index = (neighbour_index - 1) % N 
				#next_index is the next node after the current node in the loop
				next_index = (neighbour_index + 1) % N 
				
				#adds a connection to the previous node in relation to the current node in the loop
				self.nodes[neighbour_index].connections[prev_index] = 1
				#adds a connection to the next node in relation to the current node in the loop
				self.nodes[neighbour_index].connections[next_index] = 1

	def make_small_world_network(self, N, re_wire_prob=0.2):
		'''
		This function makes a small world network of size N.
		
		The default re-wire probability is set to 0.2 unless a value is input as a command line argument.
		If a value for the re_wire probability is input then this becomes the value that is used as the
		re_wire probability.
		The re-wire probability determines the number of connections that are randomly changed.
		The number of connections within the small world network remains the same before and after being
		re-wired.
		After being re-wired some nodes will have no nodes and some will have multiple.
		If the re-wire probability is set to 0 the small world network will be a ring network with a neighbour
		range of 2. 
		If the re-wire probability is set to a high value (for example, 0.95) then the small world network will be
		very random and will resemble a random network more closely.
		'''
		
		#stores the nodes in a list with corresponding value, number and connections
		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))
		
		#loops through the nodes in the small world network
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(N):
				
				#the variables prev_index_1 and next_index_1 represent the neighbour of the current node in the loop (on either side)
				prev_index_1 = (neighbour_index - 1) % N 
				next_index_1 = (neighbour_index + 1) % N 
				#the variables prev_index_2 and next_index_2 represent the second node away from the current node in the loop (on either side)
				prev_index_2 = (neighbour_index - 2) % N 
				next_index_2 = (neighbour_index + 2) % N 
				
				#adds a connection to the previous node in relation to the current node in the loop
				self.nodes[neighbour_index].connections[prev_index_1] = 1
				#adds a connection to the next node in relation to the current node in the loop
				self.nodes[neighbour_index].connections[next_index_1] = 1
				#adds a connection to the previous second node away from the current node in the loop
				self.nodes[neighbour_index].connections[prev_index_2] = 1
				#adds a connection to the next second node away from the current node in the loop
				self.nodes[neighbour_index].connections[next_index_2] = 1
		
		#loops through the nodes in the small world network
		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				
				#variable re_wire_prob is the probability that a node gets rewired
				if np.random.random() < re_wire_prob:
					#adds a connection from the current node in the loop to a random node
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1	
					
					#loops through the nodes in the small world network
					for neighbour_index in range(index+1, N):
						#variable node_removal_prob is set to 0.25 as each node starts with 4 connections
						#if one connection is added then one connection needs to be removed
						node_removal_prob = 0.25
						if np.random.random() < node_removal_prob:
							#removes the connection between the current node in the loop and another random node
							#that it is connected to
							node.connections[neighbour_index] = 0
							self.nodes[neighbour_index].connections[index] = 0
				

	def plot(self):
		
		#Generates a figure
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		# Based on the number of nodes, creates the size of the network
		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		# Positioning the Nodes
		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

			#Drawing connections/edges to neighbours
			for neighbour_index in range(i+1, num_nodes):
				if node.connections[neighbour_index]:
					neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
					neighbour_x = network_radius * np.cos(neighbour_angle)
					neighbour_y = network_radius * np.sin(neighbour_angle)

					ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

def test_networks():

	#Ring network
	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number-1)%num_nodes] = 1
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing ring network")
	assert(network.get_mean_degree()==2), network.get_mean_degree()
	assert(network.get_mean_clustering()==0), network.get_clustering()
	assert(network.get_mean_path_length()==2.777777777777778), network.get_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [0 for val in range(num_nodes)]
		connections[(node_number+1)%num_nodes] = 1
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing one-sided network")
	assert(network.get_mean_degree()==1), network.get_mean_degree()
	assert(network.get_mean_clustering()==0),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==5), network.get_mean_path_length()

	nodes = []
	num_nodes = 10
	for node_number in range(num_nodes):
		connections = [1 for val in range(num_nodes)]
		connections[node_number] = 0
		new_node = Node(0, node_number, connections=connections)
		nodes.append(new_node)
	network = Network(nodes)

	print("Testing fully connected network")
	assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
	assert(network.get_clustering()==1),  network.get_mean_clustering()
	assert(network.get_path_length()==1), network.get_mean_path_length()

	print("All tests passed")


'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
	'''
	This function should return the *change* in agreement that would result if the cell at (row, col) was to flip it's value
	Inputs: population (numpy array)
			row (int)
			col (int)
			external (float)
	Returns:
			change_in_agreement (float)
	'''

	#Your code for task 1 goes here

	return np.random * population

def ising_step(population, external=0.0):
	'''
	This function will perform a single update of the Ising model
	Inputs: population (numpy array)
			external (float) - optional - the magnitude of any external "pull" on opinion
	'''
	
	n_rows, n_cols = population.shape
	row = np.random.randint(0, n_rows)
	col  = np.random.randint(0, n_cols)

	agreement = calculate_agreement(population, row, col, external=0.0)

	if agreement < 0:
		population[row, col] *= -1

	#Your code for task 1 goes here

def plot_ising(im, population):
	'''
	This function will display a plot of the Ising model
	'''

	new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
	im.set_data(new_im)
	plt.pause(0.1)

def test_ising():
	'''
	This function will test the calculate_agreement function in the Ising model
	'''

	print("Testing ising model calculations")
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1)==4), "Test 1"

	population[1, 1] = 1.
	assert(calculate_agreement(population,1,1)==-4), "Test 2"

	population[0, 1] = 1.
	assert(calculate_agreement(population,1,1)==-2), "Test 3"

	population[1, 0] = 1.
	assert(calculate_agreement(population,1,1)==0), "Test 4"

	population[2, 1] = 1.
	assert(calculate_agreement(population,1,1)==2), "Test 5"

	population[1, 2] = 1.
	assert(calculate_agreement(population,1,1)==4), "Test 6"

	"Testing external pull"
	population = -np.ones((3, 3))
	assert(calculate_agreement(population,1,1,1)==3), "Test 7"
	assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
	assert(calculate_agreement(population,1,1,10)==14), "Test 9"
	assert(calculate_agreement(population,1,1,-10)==-6), "Test 10"

	print("Tests passed")


def ising_main(population, alpha=None, external=0.0):
	
	fig = plt.figure()
	ax = fig.add_subplot(111)
	ax.set_axis_off()
	im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

	# Iterating an update 100 times
	for frame in range(100):
		# Iterating single steps 1000 times to form an update
		for step in range(1000):
			ising_step(population, external)
		print('Step:', frame, end='\r')
		plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

#def defuant_main():
	#Your code for task 2 goes here

#def test_defuant():
	#Your code for task 2 goes here


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def task3_flags():
	'''
	This function creates the flags for the randomly generated network
	'''
	# Creates the parser
	parser = argparse.ArgumentParser(description='FCP')

	#Creates the arguments
	parser.add_argument('-network', nargs = '?', type = int, default = 6 )
	parser.add_argument('-test_network', action='store_true')

	args = parser.parse_args()

	if args.test_network:
		print('Testing Task 3')

	network_size = args.network

	return network_size


def run_task3():
	network1 = Network()

	size_network = task3_flags()
	network1.make_random_network(size_network, 0.7)
	network1.plot()
	plt.show()
	
	print('Mean Degree:', network1.get_mean_degree())
	print('Mean Path Length:', network1.get_mean_path_length())
	print('Mean Clustering Co-efficient:', network1.get_mean_clustering())

def command_line_args_4():
	'''
	This function creates the flags for the ring network and the small world network.
	'''
	
	#creates the parser
	parser = argparse.ArgumentParser()
	
	#creates the arguments
	parser.add_argument('-ring_network', action = 'store_true')
	parser.add_argument('-small_world', action = 'store_true')
	parser.add_argument('-re_wire', type = float, default = 0.2)
	parser.add_argument('-N', type = int)

	args = parser.parse_args()

	#if there is no argument input for N then it is set to 10
	if args.N is None:
		args.N = 10
		
	if args.ring_network:
		ring_network = True 
	
	if args.small_world:
		parser.add_argument('N', type = int)
		small_world = True 
	
	return args

def run_task_4(args):
	'''
	This function runs task 4 with the values inputted into the command line
	'''
	
	args = command_line_args_4()

	if args.ring_network:
		ring = Network()
		N = args.N 
		ring.make_ring_network(N)
		ring.plot()

	if args.small_world:
		small_world = Network()
		re_wire_prob = args.re_wire 
		N = args.N 
		small_world.make_small_world_network(N, re_wire_prob)
		small_world.plot()

def main():
	run_task3()
	run_task_4(command_line_args_4)

if __name__=="__main__":
	main()