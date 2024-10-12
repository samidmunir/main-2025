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

# Reconstruct the path from the start to the goal
def reconstruct_path(tree, goal_idx):
    path = []
    current_idx = goal_idx
    while current_idx is not None:
        path.append(tree[current_idx]['config'])
        current_idx = tree[current_idx]['parent']
    return path[::-1]

# Visualize the RRT* and the solution path
def visualize_rrt_star(tree, path, environment):
    fig, ax = PLT.subplots()

    # Draw obstacles
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        rect = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color='red', alpha=0.5)
        ax.add_patch(rect)

    # Draw the tree
    for node in tree:
        if node['parent'] is not None:
            parent_node = tree[node['parent']]
            ax.plot([node['config'][0], parent_node['config'][0]], [node['config'][1], parent_node['config'][1]], 'k-', alpha=0.5)

    # Draw the solution path
    if path:
        path = NP.array(path)
        ax.plot(path[:, 0], path[:, 1], 'g-', linewidth=2)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    PLT.gca().set_aspect('equal', adjustable='box')
    PLT.show()

# Load the environment from a file
def scene_from_file(filename):
    with open(filename, 'r') as file:
        environment = json.load(file)
    return environment

# Main function
def main():
    parser = argparse.ArgumentParser(description='PRM*/RRT* Path Planning')
    
    # Define the required command-line arguments
    parser.add_argument('--robot', required = True, choices = ['arm', 'freeBody'], help = 'Type of robot (arm or freeBody)')
    parser.add_argument('--start', required = True, nargs = '+', type = float, help = 'Start configuration')
    parser.add_argument('--goal', required = True, nargs = '+', type = float, help = 'Goal configuration')
    parser.add_argument('--goal_rad', required = True, type = float, help = 'Goal radius')
    parser.add_argument('--map', required = True, type = str, help = 'File containing the environment')

    # Parse arguments
    args = parser.parse_args()

    # Load the environment
    environment = scene_from_file(args.map)

    # Perform RRT*
    tree, goal_idx = rrt_star(args.start, args.goal, args.goal_rad, environment)

    # If a path is found, reconstruct and visualize it
    if goal_idx is not None:
        path = reconstruct_path(tree, goal_idx)
        visualize_rrt_star(tree, path, environment)
    else:
        print("No path found to the goal.")

if __name__ == '__main__':
    main()