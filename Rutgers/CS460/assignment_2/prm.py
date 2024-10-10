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