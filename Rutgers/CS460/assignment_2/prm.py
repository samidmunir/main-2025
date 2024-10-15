import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import rotate
import networkx as nx
from queue import PriorityQueue

# Function to load environment map
def load_map(filename):
    environment = {"obstacles": []}
    with open(filename, 'r') as file:
        width, height = map(float, file.readline().split())
        environment["width"] = width
        environment["height"] = height
        
        for line in file:
            x, y, w, h, orientation = map(float, line.split())
            obstacle = {
                "center": (x, y),
                "width": w,
                "height": h,
                "orientation": orientation
            }
            environment["obstacles"].append(obstacle)
    
    return environment

# Function to generate random nodes in the configuration space
def generate_random_node(robot_type):
    if robot_type == 'arm':
        # Generate random joint angles
        return np.random.uniform(0, 2 * np.pi, size=2)  # For 2-joint arm
    elif robot_type == 'freeBody':
        # Generate random position and orientation
        x = np.random.uniform(0, 20)
        y = np.random.uniform(0, 20)
        theta = np.random.uniform(-np.pi, np.pi)
        return (x, y, theta)

# Function to create a rectangle polygon from the center, width, height, and orientation
def create_rectangle(x, y, width, height, orientation):
    rect = Polygon([(-width / 2, -height / 2), (width / 2, -height / 2),
                    (width / 2, height / 2), (-width / 2, height / 2)])
    rotated_rect = rotate(rect, np.degrees(orientation), origin=(0, 0))
    rotated_rect = Polygon([(point[0] + x, point[1] + y) for point in rotated_rect.exterior.coords])
    return rotated_rect

# Function to check for collisions
def check_collision(node, obstacles, robot_type):
    if robot_type == 'arm':
        return False  # You can add collision checking for arm robots here
    elif robot_type == 'freeBody':
        robot = create_rectangle(node[0], node[1], 0.5, 0.3, node[2])
        for obstacle in obstacles:
            obstacle_rect = create_rectangle(obstacle["center"][0], obstacle["center"][1],
                                             obstacle["width"], obstacle["height"], obstacle["orientation"])
            if robot.intersects(obstacle_rect):
                return True
    return False

# A* Search Algorithm
def a_star_search(graph, start, goal):
    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not queue.empty():
        current = queue.get()[1]

        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + np.linalg.norm(np.array(graph.nodes[current]['pos']) - np.array(graph.nodes[neighbor]['pos']))
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + np.linalg.norm(np.array(graph.nodes[goal]['pos']) - np.array(graph.nodes[neighbor]['pos']))
                queue.put((priority, neighbor))
                came_from[neighbor] = current

    return came_from

# Reconstruct the path after A* search
def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

# Function to connect nodes (k-nearest neighbors)
def connect_nodes(graph, nodes, k, obstacles, robot_type):
    for i, node in enumerate(nodes):
        # Find k-nearest neighbors
        distances = [np.linalg.norm(np.array(node) - np.array(other)) for other in nodes]
        nearest_neighbors = np.argsort(distances)[1:k+1]  # Skip itself
        
        for neighbor in nearest_neighbors:
            if not check_collision(node, obstacles, robot_type):
                graph.add_edge(i, neighbor, weight=distances[neighbor])

# Visualize the PRM graph
def visualize_prm(graph, nodes, environment):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in environment["obstacles"]:
        rect = plt.Rectangle((obstacle["center"][0] - obstacle["width"] / 2,
                              obstacle["center"][1] - obstacle["height"] / 2),
                              obstacle["width"], obstacle["height"], 
                              angle=np.degrees(obstacle["orientation"]),
                              edgecolor='black', facecolor='gray', lw=2)
        ax.add_patch(rect)

    # Plot nodes and edges
    for edge in graph.edges:
        start_node = graph.nodes[edge[0]]['pos']
        end_node = graph.nodes[edge[1]]['pos']
        ax.plot([start_node[0], end_node[0]], [start_node[1], end_node[1]], 'b-')

    for i, node in enumerate(nodes):
        ax.plot(node[0], node[1], 'ro')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.show()

# Animate the solution path
def animate_solution_path(path, nodes, environment, robot_type):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in environment["obstacles"]:
        rect = plt.Rectangle((obstacle["center"][0] - obstacle["width"] / 2,
                              obstacle["center"][1] - obstacle["height"] / 2),
                              obstacle["width"], obstacle["height"], 
                              angle=np.degrees(obstacle["orientation"]),
                              edgecolor='black', facecolor='gray', lw=2)
        ax.add_patch(rect)

    # Animate the robot moving along the path
    for i in range(len(path) - 1):
        start_node = nodes[path[i]]
        next_node = nodes[path[i + 1]]

        ax.plot([start_node[0], next_node[0]], [start_node[1], next_node[1]], 'r-', lw=2)
        plt.pause(0.5)  # Pause to create animation effect

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="PRM Algorithm")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'],
                        help="Type of robot: 'arm' or 'freeBody'")
    parser.add_argument('--start', type=float, nargs='+', required=True,
                        help="Start configuration for the robot")
    parser.add_argument('--goal', type=float, nargs='+', required=True,
                        help="Goal configuration for the robot")
    parser.add_argument('--map', type=str, required=True,
                        help="Path to the map file")
    args = parser.parse_args()

    # Load the environment map
    environment = load_map(args.map)

    # Initialize graph for PRM
    graph = nx.Graph()

    # Generate random nodes and add them to the graph
    nodes = [generate_random_node(args.robot) for _ in range(5000)]
    
    for i, node in enumerate(nodes):
        graph.add_node(i, pos=node)
    
    # Connect nodes (PRM step)
    connect_nodes(graph, nodes, 6, environment["obstacles"], args.robot)

    # Visualize the PRM
    visualize_prm(graph, nodes, environment)

    # Convert start and goal to nodes
    start_node = generate_random_node(args.robot)  # You can use the provided start configuration here
    goal_node = generate_random_node(args.robot)  # You can use the provided goal configuration here

    # Perform A* search
    came_from = a_star_search(graph, start_node, goal_node)

    # Reconstruct the path
    path = reconstruct_path(came_from, start_node, goal_node)

    # Animate the solution path
    animate_solution_path(path, nodes, environment, args.robot)

if __name__ == "__main__":
    main()
