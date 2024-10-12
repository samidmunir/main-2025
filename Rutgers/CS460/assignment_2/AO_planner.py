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

# Function to find the nearest node in the tree
def nearest_node(tree, sampled_config):
    nodes = NP.array([node['config'] for node in tree])
    distances = NP.linalg.norm(nodes - NP.array(sampled_config), axis=1)
    nearest_idx = NP.argmin(distances)
    return nearest_idx

# Function to extend the tree towards a sampled configuration
def extend_tree(tree, sampled_config, step_size, environment):
    nearest_idx = nearest_node(tree, sampled_config)
    nearest_config = tree[nearest_idx]['config']

    # Compute direction and step toward the sampled config
    direction = NP.array(sampled_config) - NP.array(nearest_config)
    length = NP.linalg.norm(direction)
    direction = direction / length  # Normalize direction

    # Step towards the sampled configuration
    new_config = NP.array(nearest_config) + direction * min(step_size, length)

    if is_collision_free(new_config, environment):
        new_node = {'config': new_config, 'parent': nearest_idx, 'cost': tree[nearest_idx]['cost'] + NP.linalg.norm(new_config - nearest_config)}
        tree.append(new_node)
        return len(tree) - 1  # Return index of the new node
    return None

# Function to find nearby nodes for rewiring
def find_nearby_nodes(tree, new_node_config, radius):
    nodes = NP.array([node['config'] for node in tree])
    distances = NP.linalg.norm(nodes - NP.array(new_node_config), axis=1)
    nearby_indices = NP.where(distances < radius)[0]
    return nearby_indices

# RRT algorithm (optimized - RRT*)
def rrt_star(start, goal, goal_radius, environment, max_iters=1000, step_size=0.5, goal_bias=0.05, radius=1.0):
    tree = [{'config': start, 'parent': None, 'cost': 0.0}]
    
    for _ in range(max_iters):
        # Randomly sample a new configuration
        if random.random() < goal_bias:
            sampled_config = goal  # Goal bias
        else:
            sampled_config = [random.uniform(0, 20), random.uniform(0, 20)]

        # Extend the tree towards the sampled configuration
        new_node_idx = extend_tree(tree, sampled_config, step_size, environment)

        if new_node_idx is not None:
            new_node_config = tree[new_node_idx]['config']

            # Rewire the tree by checking nearby nodes to minimize cost
            nearby_nodes = find_nearby_nodes(tree, new_node_config, radius)
            for near_idx in nearby_nodes:
                if near_idx != new_node_idx:
                    near_config = tree[near_idx]['config']
                    cost_via_new_node = tree[new_node_idx]['cost'] + NP.linalg.norm(NP.array(new_node_config) - NP.array(near_config))

                    # If the new node provides a lower-cost path, rewire
                    if cost_via_new_node < tree[near_idx]['cost']:
                        if is_collision_free_path(tree[new_node_idx]['config'], tree[near_idx]['config'], environment):
                            tree[near_idx]['parent'] = new_node_idx
                            tree[near_idx]['cost'] = cost_via_new_node

            # Check if the new node is within the goal radius
            if NP.linalg.norm(NP.array(new_node_config) - NP.array(goal)) <= goal_radius:
                return tree, new_node_idx
    
    # Return the tree if the goal is not reached within max_iters
    return tree, None

# Function to check if the path between two configurations is collision-free
def is_collision_free_path(config1, config2, environment, num_steps=10):
    for step in NP.linspace(0, 1, num_steps):
        intermediate = (1 - step) * NP.array(config1) + step * NP.array(config2)
        if not is_collision_free(intermediate, environment):
            return False
    return True