# IMPORTS
import argparse as ARGPRS
import time as TIME
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
from matplotlib.animation import FuncAnimation
import numpy as NP
import random as RANDOM
from component_1 import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    OBSTACLE_MIN_SIZE,
    OBSTACLE_MAX_SIZE
)
from nearest_neighbors import (
    get_k_nearest_freeBody_robot_configurations,
    get_k_nearest_arm_robot_configurations,
    get_arm_robot_joint_positions
)
from collision_checking import (
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT,
    is_colliding,
    get_polygon_corners,
    is_colliding_link,
    point_in_circle
)

# CONSTANTS
K_NEIGHBORS = 5
NUMBER_OF_SAMPLES = 250

import heapq

# Utility Functions
def euclidean_distance(node1, node2):
    """Calculate Euclidean distance between two configurations."""
    return NP.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

def a_star_search(prm_graph, start, goal):
    """A* search to find the shortest path."""
    open_list = [(0, start)]
    g_costs = {start: 0}
    came_from = {start: None}

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in prm_graph.get(current, []):
            neighbor = neighbor[0]
            tentative_g_cost = g_costs[current] + euclidean_distance(current, neighbor)

            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + euclidean_distance(neighbor, goal)
                heapq.heappush(open_list, (f_cost, neighbor))
                came_from[neighbor] = current

    return []  # No valid path found

def reconstruct_path(came_from, current):
    """Reconstruct the path from start to goal."""
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def interpolate_path(path, steps_per_segment=10):
    """Interpolate between consecutive configurations for smoother animation."""
    interpolated_path = []
    for i in range(len(path) - 1):
        start, end = NP.array(path[i]), NP.array(path[i + 1])
        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            interpolated_point = (1 - alpha) * start + alpha * end
            interpolated_path.append(tuple(interpolated_point))
    interpolated_path.append(path[-1])
    return interpolated_path

def is_edge_collision_free(start, end, obstacles):
    """Check if the edge between two configurations is collision-free."""
    for i in range(10):
        alpha = i / 10
        interp = (1 - alpha) * NP.array(start) + alpha * NP.array(end)
        corners = get_polygon_corners((interp[0], interp[1]), interp[2], 
                                      FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
        for obs in obstacles:
            obs_corners = get_polygon_corners((obs[0], obs[1]), obs[4], obs[2], obs[3])
            if is_colliding(corners, obs_corners):
                return False
    return True

def build_prm(robot_type, samples, environment):
    """Build the PRM graph."""
    PRM = {tuple(sample): [] for sample in samples}
    if robot_type == 'freeBody':
        for sample in samples:
            neighbors = get_k_nearest_freeBody_robot_configurations(samples, sample, orientation_weight = 0.25, k=K_NEIGHBORS)
            for neighbor in neighbors:
                if is_edge_collision_free(sample, neighbor[0], environment):
                    PRM[tuple(sample)].append(neighbor)
    return PRM

# Visualization Functions
def visualize_path(path, prm_graph, environment):
    """Visualize the PRM graph and solution path."""
    fig, ax = PLT.subplots()
    for obs in environment:
        corners = get_polygon_corners((obs[0], obs[1]), obs[4], obs[2], obs[3])
        ax.add_patch(PTCHS.Polygon(corners, closed=True, color='gray', alpha=0.5))
    for node, neighbors in prm_graph.items():
        ax.plot(node[0], node[1], 'bo', markersize=3)
        for neighbor in neighbors:
            ax.plot([node[0], neighbor[0][0]], [node[1], neighbor[0][1]], 'k-', alpha=0.6)
    for i in range(len(path) - 1):
        ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'r-', linewidth=2)
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    PLT.show()

def animate_freeBody_robot(path, environment, steps_per_segment=10):
    """Animate the freeBody robot along the path."""
    interpolated_path = interpolate_path(path, steps_per_segment)
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    for obs in environment:
        corners = get_polygon_corners((obs[0], obs[1]), obs[4], obs[2], obs[3])
        ax.add_patch(PTCHS.Polygon(corners, closed=True, color='gray', alpha=0.5))
    robot_patch = PTCHS.Rectangle(
        (interpolated_path[0][0] - FREE_BODY_ROBOT_WIDTH / 2, 
         interpolated_path[0][1] - FREE_BODY_ROBOT_HEIGHT / 2),
        FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, 
        angle=NP.degrees(interpolated_path[0][2]), color='blue'
    )
    ax.add_patch(robot_patch)

    def update(frame):
        x, y, theta = interpolated_path[frame]
        robot_patch.set_xy((x - FREE_BODY_ROBOT_WIDTH / 2, y - FREE_BODY_ROBOT_HEIGHT / 2))
        robot_patch.angle = NP.degrees(theta)
        return robot_patch,

    anim = FuncAnimation(fig, update, frames=len(interpolated_path), interval=50, blit=True, repeat=False)
    PLT.show()
    
def generate_freeBody_robot_sample_configurations(number_of_samples: int, start: tuple, goal: tuple) -> list:
    SAMPLES = []
    SAMPLES = SAMPLES + [start]
    for _ in range(number_of_samples):
        x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
        y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
        theta = RANDOM.uniform(0, 2 * NP.pi)
        
        SAMPLES.append((x, y, theta))
    
    SAMPLES = SAMPLES + [goal]
    
    return SAMPLES
    
def scene_from_file(filename: str) -> list:
    print(f'\nscene_from_file({filename}) called...')
    
    OBSTACLES = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        
        for LINE in LINES:
            VALUES = LINE.strip().split(',')
            
            x = float(VALUES[0])
            y = float(VALUES[1])
            width = float(VALUES[2])
            height = float(VALUES[3])
            theta = float(VALUES[4])
            
            OBSTACLE = (x, y, width, height, theta)
            
            OBSTACLES.append(OBSTACLE)
    
    print(f'\tEnvironment loaded from FILE <{filename}>.')
    
    return OBSTACLES

def parse_arguments():
    """Parse command-line arguments."""
    parser = ARGPRS.ArgumentParser(description='PRM for Arm and FreeBody Robots.')
    parser.add_argument('--robot', choices=['arm', 'freeBody'], required=True)
    parser.add_argument('--start', nargs='+', type=float, required=True)
    parser.add_argument('--goal', nargs='+', type=float, required=True)
    parser.add_argument('--map', required=True)
    return parser.parse_args()

def main():
    """Main function to run the PRM."""
    args = parse_arguments()
    environment = scene_from_file(args.map)
    samples = generate_freeBody_robot_sample_configurations(NUMBER_OF_SAMPLES, tuple(args.start), tuple(args.goal))
    prm = build_prm(args.robot, samples, environment)
    path = a_star_search(prm, tuple(args.start), tuple(args.goal))
    if path:
        animate_freeBody_robot(path, environment)

if __name__ == '__main__':
    main()
