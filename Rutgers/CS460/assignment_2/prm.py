import argparse
import heapq
import json
import random
import time
import matplotlib.pyplot as PLT
import matplotlib.animation as ANIM
import numpy as NP
from scipy.spatial import KDTree

# Check if the robot's configuration collides with any obstacles.
def is_collision_free(config, environment):
    # The robot is represented as a point (config).
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        
        # Check if the config is inside the obstacle's bounding box.
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

# Build the PRM roadmap
def build_prm(N, k, environment):
    configurations = generate_random_configurations(N, environment)
    edges = []
    
    # Connect each node to its k-nearest neighbors
    for i, config in enumerate(configurations):
        neighbors = find_nearest_neighbors(config, configurations, k)
        for neighbor in neighbors:
            # Connect config to its neighbor if the edge is collision-free
            if neighbor != i and is_collision_free_path(config, configurations[neighbor], environment):
                edges.append((i, neighbor))
    
    return configurations, edges

# Check if the path between two configurations is collision-free
def is_collision_free_path(config1, config2, environment, num_steps=10):
    # Linearly interpolate between the two configurations
    for step in NP.linspace(0, 1, num_steps):
        intermediate = (1 - step) * NP.array(config1) + step * NP.array(config2)
        if not is_collision_free(intermediate, environment):
            return False  # If any intermediate step collides, the path is not valid
    return True

"""
# Build the PRM roadmap.
def build_prm(N, k, environment):
    configurations = generate_random_configurations(N, environment)
    edges = []
    
    # Connect each node to its k-nearest neighbors.
    for i, config in enumerate(configurations):
        neighbors = find_nearest_neighbors(config, configurations, k)
        for neighbor in neighbors:
            # Connect config to its neighbor if the edge is collision-free.
            if neighbor != i and is_collision_free_path(config, configurations[neighbor], environment):
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
"""

# A* search algorithm to find the shortest path from the start configuration to the goal configuration.
def astar_search(start_idx, goal_idx, configurations, edges):
    def heuristic(config1, config2):
        return NP.linalg.norm(NP.array(config1) - NP.array(config2))
     
    # Priority queue for A*.
    open_set = [(0, start_idx)]
    heapq.heapify(open_set)
    
    came_from = {}
    g_score = {i: float('inf') for i in range(len(configurations))}
    g_score[start_idx] = 0
    
    f_score = {i: float('inf') for i in range(len(configurations))}
    f_score[start_idx] = heuristic(configurations[start_idx], configurations[goal_idx])
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            return reconstruct_path(came_from, current)

        for neighbor in [edge[1] for edge in edges if edge[0] == current] + [edge[0] for edge in edges if edge[1] == current]:
            tentative_g_score = g_score[current] + heuristic(configurations[current], configurations[neighbor])
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(configurations[neighbor], configurations[goal_idx])
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return [] # Return an empty path if no solution found.

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

# A* path-finding algorithm with animation support (new function).
def astar_search_animated(start_idx, goal_idx, configurations, edges):
    def heuristic(config1, config2):
        return NP.linalg.norm(NP.array(config1) - NP.array(config2))
    
    open_set = [(0, start_idx)]
    heapq.heapify(open_set)
    
    came_from = {}
    g_score = {i: float('inf') for i in range(len(configurations))}
    g_score[start_idx] = 0
    
    f_score = {i: float('inf') for i in range(len(configurations))}
    f_score[start_idx] = heuristic(configurations[start_idx], configurations[goal_idx])
    
    explored_edges = [] # To store edges explored during search.
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal_idx:
            return reconstruct_path(came_from, current), explored_edges
        
        # Get the neighbors for the current node.
        for neighbor in [edge[1] for edge in edges if edge[0] == current] + [edge[0] for edge in edges if edge[1] == current]:
            tentative_g_score = g_score[current] + heuristic(configurations[current], configurations[neighbor])
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(configurations[neighbor], configurations[goal_idx])
                
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
                # Record the explored edge for animation.
                explored_edges.append((current, neighbor))
    
    return [], explored_edges # Return empty path if no solution is found.

# Function to animate the A* search process.
def animate_astar_search(configurations, edges, environment, path, explored_edges):
    fig, ax = PLT.subplots()
    
    # Draw the obstacles.
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color = '#ff0000', alpha = 0.5)
        ax.add_patch(obstacle_rectangle)
    
    # Draw the nodes (configurations).
    ax.scatter([c[0] for c in configurations], [c[1] for c in configurations], color = '#0000ff', s = 10)
    
    # Set limits for the environment.
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    def update(frame):
        # Clear the current frame.
        for patch in ax.patches[:]:
            patch.remove()
        for line in ax.lines[:]:
            line.remove()
        
        # Re-draw obstacles.
        for obstacle in environment:
            x, y = obstacle['center']
            width, height = obstacle['width'], obstacle['height']
            obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color = '#ff0000', alpha = 0.5)
            ax.add_patch(obstacle_rectangle)
        
        # Draw explored edges up to the current frame.
        for i in range(min(frame, len(explored_edges))):
            edge = explored_edges[i]
            config1 = configurations[edge[0]]
            config2 = configurations[edge[1]]
            ax.plot([config1[0], config2[0]], [config1[1], config2[1]], 'gray', alpha = 0.5)
        
        # If the path has been found, draw it.
        if frame >= len(explored_edges):
            for i in range(len(path) - 1):
                config1 = configurations[path[i]]
                config2 = configurations[path[i + 1]]
                ax.plot([config1[0], config2[0]], [config1[1], config2[1]], 'g-', linewidth = 2)
        
    # Create the animation.
    animation = ANIM.FuncAnimation(fig, update, frames = len(explored_edges) + 20, interval = 100, repeat = False)
    
    PLT.gca().set_aspect('equal', adjustable = 'box')
    PLT.show()

# Main function.
def main():
    parser = argparse.ArgumentParser(description = 'PRM Path Planning')
    
    # Define the required command-line arguments.
    parser.add_argument('--robot', required = True, choices = ['arm', 'freeBody'], help = 'Type of robot (arm or freeBody)')
    parser.add_argument('--start', required = True, nargs = '+', type = float, help = 'Start configuration')
    parser.add_argument('--goal', required = True, nargs = '+', type = float, help = 'Goal configuration')
    parser.add_argument('--map', required = True, type = str, help = 'File containing the environment')
    
    # Parse arguments.
    args = parser.parse_args()
    
    # Load the environment.
    environment = scene_from_file(args.map)
    
    # Generate the PRM roadmap.
    N = 500 # number of nodes
    k = 6 # number of nearest neighbors
    configurations, edges = build_prm(N, k, environment)
    
    # Add start and goal to the roadmap.
    configurations.append(args.start)
    configurations.append(args.goal)
    start_idx = len(configurations) - 2
    goal_idx = len(configurations) - 1
    
    # Connect start and goal to the roadmap.
    edges += [(start_idx, neighbor) for neighbor in find_nearest_neighbors(args.start, configurations[:-2], k)]
    edges += [(goal_idx, neighbor) for neighbor in find_nearest_neighbors(args.goal, configurations[:-2], k)]
    
    # Find the path using A* search.
    path = astar_search(start_idx, goal_idx, configurations, edges)
    
    # If a path is found, visualize the roadmap and the solution.
    if path:
        visualize_prm(configurations, edges, path, environment)
        path, explored_edges = astar_search_animated(start_idx, goal_idx, configurations, edges)
        animate_astar_search(configurations, edges, environment, path, explored_edges)
    else:
        print('No path found between the start and goal configurations.')

if __name__ == "__main__":
    main()