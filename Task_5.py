import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx as nx
import unittest
import time
from matplotlib.animation import FuncAnimation


def defuant(population, threshold = 0.2, beta=0.2, use_network=False, numNodes = 10, num_timesteps=100, show_network_delay=0.2):
    '''
    Executes timesteps of the changing opinions of the persons in the population array
    '''

    if use_network == False:
        # Get the number of rows and columns in population
        num_rows, num_cols = population.shape
        # for allowed number of timesteps
        for timestep in range(num_timesteps-1):
            # for each person in the population
            for i in range(num_rows):
                # get left or right neighbour index randomly (circular)
                randomNeighbourIndex = (i + ((2*random.randint(0, 1))-1)) % num_rows
                # if two people have difference of opinion no larger than threshold
                if abs(population[i,timestep] - population[randomNeighbourIndex,timestep]) < threshold:
                    # tilt each person's and neighbour's opinions towards each other
                    population[i,timestep+1] = population[i,timestep] + beta * (population[randomNeighbourIndex,timestep] - population[i,timestep])  
                    population[randomNeighbourIndex,timestep + 1] = population[randomNeighbourIndex,timestep] + beta * (population[i,timestep] - population[randomNeighbourIndex,timestep])
                else:
                    population[i,timestep+1] = population[i,timestep]

        # Plot histogram of opinions at the last timestep
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.hist(population[:, -1], bins=10, range=(0, 1))
        plt.xlabel('Opinion')
        plt.ylabel('Frequency')
        plt.title('Histogram of Opinion at Last Timestep')
        # Plot opinion over timestep for each person
        plt.subplot(1, 2, 2)
        for person in population:
            plt.plot(person)
        plt.xlabel('Timestep')
        plt.ylabel('Opinion')
        plt.title('Opinion Over Timestep')

        plt.tight_layout()
        plt.show()

    else:
        # Define parameters for the Watts-Strogatz small-world graph
        num_nodes = numNodes
        k_neighbors = 4 
        p_rewiring = 0.5
        #define array of mean opinion
        meanOpinion = [0.0]*num_timesteps
        # Create a Watts-Strogatz small-world graph
        G = nx.watts_strogatz_graph(num_nodes, k_neighbors, p_rewiring)

        # Assign initial random values between 0 and 1 to each node as opinions
        for node in G.nodes():
            G.nodes[node]['value'] = random.random()

        fig, ax = plt.subplots(figsize=(6, 6))

        for timestep in range(num_timesteps):
            ax.clear()
            for node in G.nodes():
                #adding total opinions up in timestep
                meanOpinion[timestep] += G.nodes[node]['value']
                neighbors = list(G.neighbors(node))
                neighbor = random.choice(neighbors)  # Choose a random neighbor
                if abs(G.nodes[node]['value'] - G.nodes[neighbor]['value']) < threshold:
                    # Update opinions based on the threshold condition
                    G.nodes[node]['value'] += beta * (G.nodes[neighbor]['value'] - G.nodes[node]['value'])
                    G.nodes[neighbor]['value'] += beta * (G.nodes[node]['value'] - G.nodes[neighbor]['value'])
            node_colors = [G.nodes[node]['value'] for node in G.nodes()]
            nx.draw(G, pos=nx.circular_layout(G), ax=ax, node_color=node_colors, cmap=plt.cm.Reds, with_labels=True)
            ax.set_title('Opinion Dynamics on Small-World Network (Timestep {})'.format(timestep))
            plt.pause(show_network_delay)
        #finding mean of each timestep
        for i in range(len(meanOpinion)):
            meanOpinion[i] = meanOpinion[i]/numNodes
        # Plotting the mean opinion over all timesteps
        # close previous plot
        plt.close()
        plt.plot(range(num_timesteps), meanOpinion)
        plt.xlabel('Timestep')
        plt.ylabel('Mean Opinion')
        plt.title('Mean Opinion Over 100 Timesteps')
        plt.grid(True)
        plt.show()
        


def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add the main flags
    parser.add_argument('--test_defuant', action='store_true', help='Add this flag to enable test_defuant')
    parser.add_argument('--defuant', action='store_true', help='Add this flag to enable defuant')
    parser.add_argument('--use_network', type=int, default=None, help='add this flag to enable network, value for size of network')

    # Add optional flags with default values
    parser.add_argument('--beta', type=float, default=0.2, help='Value for beta (default: 0.2)')
    parser.add_argument('--threshold', type=float, default=0.2, help='Value for threshold (default: 0.2)')

    return parser.parse_args()

class TestDefuantSimulation(unittest.TestCase):
    
    def test_defuant(self):
        print("Executing test_defuant_function")
        population = np.zeros((100, 100))
        population[:,0] = np.random.rand(100)  # Fill the first column with random values
        
        # Call defuant function with some threshold and beta values
        threshold = 0.2
        defuant(population, threshold)
        
        # Assert that the population array is modified after calling defuant function
        self.assertFalse(np.all(population == 0))  # Check if all elements are not zero
        
        # Assert that opinions at the last timestep have been updated
        self.assertNotEqual(population[:, -1].tolist(), [0] * 100)  # Check if any opinion is not zero
        print("No errors detected")

def main():

    # Create a 100x100 2D array filled with zeros
    population = np.zeros((100, 100))

    # Fill the first row with 100 random values between 0 and 1 representing 100 peoples random opinions
    population[:,0] = np.random.rand(100)
    # get arguments
    args = parse_args()
    
    # Call functions based on flags
    if args.test_defuant:
        unittest.main(argv=[''], verbosity=2, exit=False)
    if args.defuant:
        if args.use_network is not None:
            defuant(population, args.threshold, args.beta, True, args.use_network)
        else:
            defuant(population, args.threshold, args.beta, False)

if __name__ == "__main__":
    main()
