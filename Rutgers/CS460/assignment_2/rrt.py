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

# Function to reconstruct the path from the start to the goal.
def reconstruct_path(tree, goal_idx):
    path = []
    current_idx = goal_idx
    while current_idx is not None:
        path.append(tree[current_idx]['config'])
        current_idx = tree[current_idx]['parent']
    return path[::-1] # reverse the path to get the start to goal.

# Function to visualize the RRT path and the solution path.
def visualize_rrt(tree, path, environment):
    fig, ax = PLT.subplots()
    
    # Draw obstacles.
    for obstacles in environment:
        x, y = obstacles['center']
        width, height = obstacles['width'], obstacles['height']
        obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color = 'black', alpha = 0.5)
        ax.add_patch(obstacle_rectangle)
    
    # Draw the tree.
    for node in tree:
        if node['parent'] is not None:
            parent_node = tree[node['parent']]
            ax.plot([node['config'][0], parent_node['config'][0]], [node['config'][1], parent_node['config'][1]], 'k-', alpha = 0.5)
    
    # Draw the solution path.
    if path:
        path = NP.array(path)
        ax.plot(path[:, 0], path[:, 1], 'g-', linewidth = 2)
    
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    PLT.gca.set_aspect('equal', adjustable = 'box')
    PLT.show()