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
		mean distance to all nodes it is connected to
		'''
		
		total = 0

		#Uses 2 for loops to compare every node to one another
		for node_number_start in self.nodes:
			for node_number in self.nodes:

				#Ensures not trying to find path length of the same node started at
				if node_number_start != node_number:

					path_queue = Queue()
					#Pushes the node trying to find the distance from to the front of the list
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
		print('indent')
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		print('indent')
		#Your code for task 4 goes here

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
	assert(network.get_clustering()==0), network.get_clustering()
	assert(network.get_path_length()==2.777777777777778), network.get_path_length()

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
	assert(network.get_clustering()==0),  network.get_clustering()
	assert(network.get_path_length()==5), network.get_path_length()

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
	assert(network.get_mean_clustering()==1),  network.get_mean_clustering()
	assert(network.get_mean_path_length()==1), network.get_mean_path_length()

	print("All tests passed")

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


def main():
	"""main"""
	network1 = Network()

	size_network = task3_flags()
	network1.make_random_network(size_network, 0.7)
	network1.plot()
	plt.show()
	
	print('Mean Degree:', network1.get_mean_degree())
	print('Mean Path Length:', network1.get_mean_path_length())
	print('Mean Clustering Co-efficient:', network1.get_mean_clustering())

if __name__=="__main__":
	main()
