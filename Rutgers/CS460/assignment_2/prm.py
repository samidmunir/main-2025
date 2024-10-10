import argparse
import heapq
import json
import random
import time
import matplotlib.pyplot as PLT
import numpy as NP
from scipy.spatial import KDTree

# Check if the robot's configuration collides with any obstacles.
def is_collision_free(config, environment):
    # The robot is represented as a point (config).
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        
        # Check if the congig is inside the obstacle's bounding box.
        if (abs(x - config[0]) * 2 < width) and (abs(y - config[1]) * 2 < height):
            return False
        
        return True

# Genereate N random configurations in the environment.
def generate_random_configurations(N, environment):
    configurations = []
    
    while (len(configurations) < N):
        config = [random.uniform(0, 20), random.uniform(0, 20)] # Random (x, y) in 20x20 space.
        if is_collision_free(config, environment):
            configurations.append(config)
    
    return configurations

# Find k-nearest neighbors using KDTree.
def find_nearest_neighbors(config, configurations, k):
    tree = KDTree(configurations)
    distances, indices = tree.query(config, k)
    
    return indices

# Build the PRM roadmap.
def build_prm(N, k, environment):
    configurations = generate_random_configurations(N, environment)
    edges = []
    
    # Connect each node to its k-nearest neighbors.
    for i, config in enumerate(configurations):
        neighbors = find_nearest_neighbors(config, configurations, k)
        for neighbor in neighbors:
            # Connect config to its neighbor if the edge is collision-free.
            if neighbor != i and is_collision_free_path(configurations[neighbor], environment):
                edges.append((i, neighbor))
                
    return configurations, edges

# Check if the path between two configurations is collision-free.
def is_collision_free_path(config1, config2, environment, num_steps = 10):
    # Linearly interpolate between the two configurations.
    for step in NP.linspace(0, 1, num_steps):
        intermediate = (1 - step) * NP.array(config1) + step * NP.array(config2)
        
        if not is_collision_free(intermediate, environment):
            return False
        
    return True

# TODO: implement the A* search algorithm to find the shortest path from the start configuration to the goal configuration.

# Reconstruct the path from A* search.
def reconstruct_path(came_from, current):
    path = [current]
    
    while current in came_from:
        current = came_from[current]
        path.append(current)
    
    return path[::-1]

# TODO: Visualize the PRM roadmap and the solution path.

# Load the environment from a file.
def scene_from_file(filename):
    with open(filename, 'r') as file:
        environment = json.load(file)
    
    return environment