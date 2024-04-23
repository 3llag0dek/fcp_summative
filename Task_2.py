import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import unittest

def defuant(population, threshold):
    '''
    Executes 100 timesteps of the changing opinions of the persons in the population array
    '''
    print("Executing defuant_function")
    args = parse_args()
    # Get the number of rows and columns in population
    num_rows, num_cols = population.shape
    # for allowed number of timesteps
    for timestep in range(num_cols-1):
        # for each person in the population
        for i in range(num_rows):
            # get left or right neighbour index randomly (circular)
            randomNeighbourIndex = (i + ((2*random.randint(0, 1))-1)) % num_rows
            # if two people have difference of opinion no larger than threshold
            if abs(population[i,timestep] - population[randomNeighbourIndex,timestep]) < threshold:
                # tilt each person's and neighbour's opinions towards each other
                population[i,timestep+1] = population[i,timestep] + args.beta * (population[randomNeighbourIndex,timestep] - population[i,timestep])  
                population[randomNeighbourIndex,timestep + 1] = population[randomNeighbourIndex,timestep] + args.beta * (population[i,timestep] - population[randomNeighbourIndex,timestep])
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

def parse_args():
    parser = argparse.ArgumentParser(description="Your script description here")

    # Add the main flags
    parser.add_argument('--test_defuant', action='store_true', help='Add this flag to enable test_defuant')
    parser.add_argument('--defuant', action='store_true', help='Add this flag to enable defuant')

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
        defuant(population, args.threshold)

if __name__ == "__main__":
    main()
