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