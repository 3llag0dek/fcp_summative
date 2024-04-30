import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import unittest


def defuant(threshold = 0.2, beta=0.2, use_network=False, num_nodes = 10, num_timesteps=100, show_network_delay=0.2):
    """
    Simulate the Deffuant model for opinion dynamics.

    :param threshold: Threshold value for opinion difference to initiate interaction
    :param beta: Weighting factor for updating opinions during interaction
    :param use_network: Boolean flag indicating whether to use a small-world network
    :param num_nodes: Number of nodes in the small-world network
    :param num_timesteps: Number of timesteps to simulate
    :param show_network_delay: Delay between network visualization updates
    :return: If not using a network, returns the population array with opinions, otherwise, returns None
    """
    if use_network == False:
        # Simulation without network
        # Create a 100x100 2D array filled with zeros
        population = np.zeros((100, 100))
        # Fill the first column with 100 random values between 0 and 1 representing 100 peoples random opinions
        population[:, 0] = np.random.rand(100)
        # copy first column to all other columns
        for i in range(len(population)):
            population[i, :] = population[i, 0]

        # Get the number of rows and columns in population
        num_rows = population.shape[0]
        # for allowed number of timesteps
        for timestep in range(num_timesteps-1):
            # randomly choose a person from the population``
            for i in range(250):
                random_person_index = random.randint(0,num_rows-1)
                # get left or right neighbour index randomly (circular)
                randomNeighbourIndex = (random_person_index + ((2*random.randint(0, 1))-1)) % num_rows
                # if two people have difference of opinion no larger than threshold
                if abs(population[random_person_index,timestep] - population[randomNeighbourIndex,timestep]) < threshold:
                    # tilt each person's and neighbour's opinions towards each other
                    population[random_person_index,timestep+1] = population[random_person_index,timestep] + beta * (population[randomNeighbourIndex,timestep] - population[random_person_index,timestep])  
                    population[randomNeighbourIndex,timestep + 1] = population[randomNeighbourIndex,timestep] + beta * (population[random_person_index,timestep] - population[randomNeighbourIndex,timestep])
                else:
                    population[random_person_index,timestep+1] = population[random_person_index,timestep]

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

        plt.suptitle(f'Coupling: {beta}, Threshold: {threshold}', fontsize=14, ha='center')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to make room for suptitle
        plt.show()

        return population

    else:
        # Simulation with small-world network
        # Define parameters for the Watts-Strogatz small-world graph
        k_neighbors = 4 
        p_rewiring = 0.2
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
            for i in range(num_nodes):
                #assign random node 
                random_node = random.choice(list(G.nodes()))
                #adding total opinions up in timestep
                neighbours = list(G.neighbors(random_node))
                # Choose a random neighbor
                neighbour = random.choice(neighbours)  
                if abs(G.nodes[random_node]['value'] - G.nodes[neighbour]['value']) < threshold:
                    # Update opinions based on the threshold condition
                    G.nodes[random_node]['value'] += beta * (G.nodes[neighbour]['value'] - G.nodes[random_node]['value'])
                    G.nodes[neighbour]['value'] += beta * (G.nodes[random_node]['value'] - G.nodes[neighbour]['value'])
            for node in G.nodes():
                meanOpinion[timestep] += G.nodes[node]['value']
            node_colors = [G.nodes[node]['value'] for node in G.nodes()]
            nx.draw(G, pos=nx.circular_layout(G), ax=ax, node_color=node_colors, cmap=plt.cm.Reds, with_labels=True)
            ax.set_title('Opinion Dynamics on Small-World Network (Timestep {})'.format(timestep))
            plt.pause(show_network_delay)
        #finding mean of each timestep
        for i in range(len(meanOpinion)):
            meanOpinion[i] = meanOpinion[i]/num_nodes
        # Plotting the mean opinion over all timesteps
        # close previous plot
        plt.close()
        plt.plot(range(num_timesteps), meanOpinion)
        plt.xlabel('Timestep')
        plt.ylabel('Mean Opinion')
        plt.title('Mean Opinion Over 100 Timesteps')
        plt.grid(True)
        plt.show()
        

# Function to parse command-line arguments
def parse_args():
    """
    Parse command-line arguments.

    :return: Parsed arguments
    """
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Model for opinion dynamics in populations")
    # Add the main flags
    parser.add_argument('--test_defuant', action='store_true', help='Add this flag to enable test_defuant')
    parser.add_argument('--defuant', action='store_true', help='Add this flag to enable defuant')
    parser.add_argument('--use_network', type=int, default=None, help='add this flag to enable small worlds network with a value for size of network')

    # Add optional flags with default values
    parser.add_argument('--beta', type=float, default=0.2, help='Value for beta (default: 0.2)')
    parser.add_argument('--threshold', type=float, default=0.2, help='Value for threshold (default: 0.2)')

    return parser.parse_args()

class TestDefuantSimulation(unittest.TestCase):
    
    def test_defuant(self):
        # Unit test for the defuant function
        print("Executing test_defuant_function")
        
        
        # Call defuant function with some threshold and beta values
        
        population = defuant(threshold = 0.3, beta = 0.4)
        
        # Assert that the population array is modified after calling defuant function
        self.assertFalse(np.all(population == 0))  # Check if all elements are not zero
        
        # Assert that opinions at the last timestep have been updated
        self.assertNotEqual(population[:, -1].tolist(), [0] * 100)  # Check if any opinion is not zero
        print("No errors detected")

def main():
    # Main function to parse arguments and call appropriate functions
    # get arguments
    args = parse_args()
    # Call functions based on flags
    if args.test_defuant:
        unittest.main(argv=[''], verbosity=2, exit=False)
    if args.defuant:
        if args.use_network is not None:
            defuant(args.threshold, args.beta, True, args.use_network)
        else:
            defuant(args.threshold, args.beta, False)

if __name__ == "__main__":
    main()
