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

# Visualize the PRM roadmap and the solution path.
def visualize_prm(configurations, edges, path, environment):
    fig, ax = PLT.subplots()
    
    # Draw the obstacles.
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color = '#0000ff', alpha = 0.5)
        ax.add_patch(obstacle_rectangle)
    
    # Draw the nodes (configurations).
    for config in configurations:
        ax.plot(config[0], config[1], 'bo', markersize = 5)
    
    # Draw the edges.
    for edge in edges:
        config1 = configurations[edge[0]]
        config2 = configurations[edge[1]]
        ax.plot([config1[0], config2[0]], [config1[1], config2[1]], 'k-', alpha = 0.3)
    
    # Draw the solution path.
    for i in range(len(path) - 1):
        config1 = configurations[path[i]]
        config2 = configurations[path[i + 1]]
        ax.plot([config1[0], config2[0]], [config1[1], config2[1]], 'g-', linewidth = 2)
                
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    PLT.gca().set_aspect('equal', adjustable = 'box')
    PLT.show()

# Load the environment from a file.
def scene_from_file(filename):
    with open(filename, 'r') as file:
        environment = json.load(file)
    
    return environment