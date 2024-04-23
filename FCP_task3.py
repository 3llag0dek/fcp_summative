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
		This Function Calulates the mean degree of 
		'''
	
		total_degree = 0
		for node_number in self.nodes:
			total_degree += node_number.connections.count(1)
		
		return total_degree/len(self.nodes)
	
	def get_mean_clustering(self):

		mean_coefficient = 0
		for node in self.nodes:
			count = 0
			node_neighbours = node.get_neighbours()
			neigh_num = len(node_neighbours)
			for neighbour in node_neighbours:
				neighbour_neighbours = self.nodes[neighbour].get_neighbours()
				for check_neighbours in neighbour_neighbours:
					if check_neighbours in node_neighbours:
						count += 1
			possible_connections = (neigh_num*(neigh_num - 1))/2
			if possible_connections == 0:
				node_coefficient = 0
			else:
				node_coefficient = (count/2)/possible_connections

			mean_coefficient += node_coefficient

		return mean_coefficient/len(self.nodes)



		#Your code for task 3 goes here

	def get_mean_path_length(self):
		
		total = 0
		for node_number_start in self.nodes:
			for node_number in self.nodes:
				if node_number_start != node_number:
					path_queue = Queue()
					path_queue.push(node_number_start)
					path = []

					while not path_queue.is_empty():
						check_node = path_queue.pop()

						if check_node == node_number:
							break

						for neighbour_node in check_node.get_neighbours():
							neighbour = self.nodes[neighbour_node]
							if neighbour_node not in path:
								path_queue.push(neighbour)
								path.append(neighbour_node)

								neighbour.parent = check_node

					check_node = node_number
					node_number_start.parent = None
					route = []

					while check_node.parent:
						route.append(check_node)
						check_node = check_node.parent
					route.append(check_node)
					
					total += len(route)-1
		
		n = len(self.nodes)
		mean_path = total/(n*(n-1))
		
		return mean_path




		#Your code for task 3 goes here
	def make_random_network(self, N, connection_probability):
		'''
		This function makes a *random* network of size N.
		Each node is connected to each other node with probability p
		'''

		self.nodes = []
		for node_number in range(N):
			value = np.random.random()
			connections = [0 for _ in range(N)]
			self.nodes.append(Node(value, node_number, connections))

		for (index, node) in enumerate(self.nodes):
			for neighbour_index in range(index+1, N):
				if np.random.random() < connection_probability:
					node.connections[neighbour_index] = 1
					self.nodes[neighbour_index].connections[index] = 1

	def make_ring_network(self, N, neighbour_range=1):
		print('poop')
		#Your code  for task 4 goes here

	def make_small_world_network(self, N, re_wire_prob=0.2):
		print('poop')
		#Your code for task 4 goes here

	def plot(self):

		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.set_axis_off()

		num_nodes = len(self.nodes)
		network_radius = num_nodes * 10
		ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
		ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

		for (i, node) in enumerate(self.nodes):
			node_angle = i * 2 * np.pi / num_nodes
			node_x = network_radius * np.cos(node_angle)
			node_y = network_radius * np.sin(node_angle)

			circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
			ax.add_patch(circle)

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
	parser = argparse.ArgumentParser(description='FCP')

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



	
	#You should write some code for handling flags here

if __name__=="__main__":
	main()
