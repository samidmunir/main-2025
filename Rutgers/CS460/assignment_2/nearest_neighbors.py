import argparse
import numpy as NP

# Function to compute the Euclidean distance between two configurations.
def euclidean_distance(config1, config2):
    return NP.linalg.norm(NP.array(config1) - NP.array(config2))

# Parse the configurations from the file.
def load_configurations(filename):
    with open(filename, 'r') as file:
        configurations = [list(map(float, line.strip().split())) for line in file]
    
    return configurations

# Nearest neighbor search using linear search.
def find_nearest_neighbors(target, configurations, k):
    distances = []
    
    # Calculate distance from target to each configuration.
    for config in configurations:
        dist  = euclidean_distance(target, config)
        distances.append((config, dist))
        
    # Sort by distance and return the k-nearest configurations.
    distances.sort(key=lambda x: x[1])
    
    return [config for config, dist in distances[:k]]

# Main function.
def main():
    parser = argparse.ArgumentParser(description = 'Nearest neighbors with linear search.')
    
    # Define the required command-line arguments.
    parser.add_argument('--robot', required = True, choices = ['arm', 'freeBody'], help = 'Type of robot (arm or freeBody).')
    parser.add_argument('--target', required = True, nargs = '+', type = float, help = 'Target configration')
    parser.add_argument('--k', required = True, type = int, help = 'Number of nearest neighbors to find.')
    parser.add_argument('--configs', required = True, type = str, help = 'File containing the robot configurations.')
    
    # Parse arguments.
    args = parser.parse_args()
    
    # Load the configurations from the file.
    configurations = load_configurations(args.configs)
    
    # Find the nearest neighbors.
    neighbors = find_nearest_neighbors(args.target, configurations, args.k)
    
    # Output the results.
    print(f'Target configuration: {args.target}')
    print(f'{args.k} Nearest Neighbors:')
    for neighbor in neighbors:
        print(neighbor)
        
if __name__ == '__main__':
    main()