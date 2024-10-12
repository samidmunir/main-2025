import argparse
import json
import random
import matplotlib.pyplot as PLT
import numpy as NP

# Function to check if a configuration collides with any obstacles.
def is_collision_free(config, environment):
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        
        if (abs(x - config[0]) * 2 < width) and (abs(y - config[1]) * 2 < height):
            return False
    return True

# Function to find the nearest node in the tree.
def nearest_node(tree, sampled_config):
    nodes = NP.array([node['config'] for node in tree])
    distances = NP.linalg.norm(nodes - NP.array(sampled_config), axis = 1)
    nearest_idx = NP.argmin(distances)
    return nearest_idx

# Function to extend the tree towards a sampled configuration.
def extend_tree(tree, sampled_config, step_size, environment):
    nearest_idx = nearest_node(tree, sampled_config)
    nearest_config = tree[nearest_idx]['config']
    
    # Compute direction and step toward the sampled config.
    direction = NP.array(sampled_config) - NP.array(nearest_config)
    length = NP.linalg.norm(direction)
    direction = direction / length # normalize direction
    
    # Step towards the sampled configuration.
    new_config = NP.array(nearest_config) + direction * min(step_size, length)
    
    if is_collision_free(new_config, environment):
        new_node = {'config': new_config, 'parent': nearest_idx}
        tree.append(new_node)
        return len(tree) - 1 # return index of the new node.
    return None

# RRT algorithm.
def rrt(start, goal, goal_radius, environment, max_iters = 1000, step_size = 0.5, goal_bias = 0.05):
    tree = [{'config': start, 'parent': None}]
    
    for _ in range(max_iters):
        # Randomly sample a new configuration.
        if random.random() < goal_bias:
            sampled_config = goal # goal bias
        else:
            sampled_config = [random.uniform(0, 20), random.uniform(0, 20)]
        
        # Extend the tree towards the sampled configuration.
        new_node_idx = extend_tree(tree, sampled_config, step_size, environment)
        
        # Check if the new node is within the goal radius.
        if new_node_idx is not None:
            new_node_config = tree[new_node_idx]['config']
            if NP.linalg.norm(NP.array(new_node_config) - NP.array(goal)) <= goal_radius:
                # reached the goal.
                return tree, new_node_idx
    # Return the tree if goal not reached within max_iters.
    return tree, None