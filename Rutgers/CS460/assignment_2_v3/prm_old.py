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

from nearest_neighbors_v2 import (
    get_k_nearest_freeBody_robot_configurations,
    get_k_nearest_arm_robot_configurations,
    get_arm_robot_joint_positions,
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH
)

from collision_checking_v2 import (
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT,
    JOINT_RADIUS,
    is_colliding,
    get_polygon_corners,
    is_colliding_link,
    point_in_circle
)

# CONSTANTS
K_NEIGHBORS = 5
NUMBER_OF_SAMPLES = 250

import heapq

"""
    function generate_arm_robot_sample_configurations(number_of_samples: int, start: tuple, goal: tuple) -> list:
"""
def generate_arm_robot_sample_configurations(number_of_samples: int, start: tuple, goal: tuple) -> list:
    SAMPLES = []
    SAMPLES = SAMPLES + [start, goal]
    for _ in range(number_of_samples):
        theta_1 = RANDOM.uniform(0, 2 * NP.pi)
        theta_2 = RANDOM.uniform(0, 2 * NP.pi)
        BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = theta_1, theta_2 = theta_2)
        SAMPLE = (theta_1, theta_2, BASE, JOINT, END_EFFECTOR)
        
        SAMPLES.append(SAMPLE)
    
    return SAMPLES

"""
    function euclidean_distance(node_1: tuple, node_2: tuple) -> float
"""
def euclidean_distance(node_1: tuple, node_2: tuple) -> float:
    EUC_DIST = NP.sqrt((node_1[0] - node_2[0]) ** 2 + (node_1[1] - node_2[1]) ** 2)
    
    return EUC_DIST

"""
    function a_star_search(prm_graph: dict, start: tuple, goal: tuple) -> list:
"""
def a_star_search(prm_graph: dict, start: tuple, goal: tuple) -> list:
    # Ensure start and goal nodes are properly formatted as tuples
    start = tuple(map(float, start))
    goal = tuple(map(float, goal))

    # Initialize priority queue (open list)
    open_list = []
    heapq.heappush(open_list, (0, start))

    # Store the actual cost (g-cost) to reach each node
    g_costs = {start: 0}

    # Track where each node came from (for path reconstruction)
    came_from = {start: None}

    # Continue until all nodes are explored or goal is found
    while open_list:
        # Get the node with the lowest f-cost
        _, current = heapq.heappop(open_list)

        # If we reached the goal, reconstruct and return the path
        if current == goal:
            return reconstruct_path(came_from, current)

        """
            function reconstruct_path(came_from: dict, current: tuple) -> list
            - Reconstruct the path from the goal to the start.
        """
        def reconstruct_path(came_from: dict, current: tuple) -> list:
            path = [current]
            while came_from[current] is not None:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Reverse the path to go from start to goal


        # Explore neighbors of the current node
        for neighbor in prm_graph.get(current, []):
            neighbor = neighbor[0]  # Extract the configuration tuple from neighbor list

            # Calculate the tentative g-cost for the neighbor
            tentative_g_cost = g_costs[current] + euclidean_distance(current, neighbor)

            # If this path is better, record it
            if neighbor not in g_costs or tentative_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = tentative_g_cost
                f_cost = tentative_g_cost + euclidean_distance(neighbor, goal)
                heapq.heappush(open_list, (f_cost, neighbor))
                came_from[neighbor] = current

    # If no path is found, return an empty list
    return []

def visualize_path(path: list, prm_graph: dict, environment: list):
    fig, ax = PLT.subplots()

    # Plot obstacles
    for obstacle in environment:
        x, y, width, height, theta = obstacle
        obstacle_corners = get_polygon_corners((x, y), theta, width, height)
        obstacle_patch = PTCHS.Polygon(obstacle_corners, closed=True, color='gray', alpha=0.5)
        ax.add_patch(obstacle_patch)

    # Plot PRM graph: Nodes and edges
    for node, neighbors in prm_graph.items():
        ax.plot(node[0], node[1], 'bo', markersize=3)  # Nodes as blue circles
        for neighbor in neighbors:
            neighbor = neighbor[0]  # Ensure correct unpacking
            ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 'k-', linewidth=0.5, alpha=0.6)

    # Plot the solution path
    for i in range(len(path) - 1):
        x1, y1 = path[i][:2]  # Unpack (x, y) from each path node
        x2, y2 = path[i + 1][:2]
        ax.plot([x1, x2], [y1, y2], 'r-', linewidth=2.0)  # Path in red

    # Set plot limits and aspect ratio
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    PLT.title('PRM Solution Path Visualization')
    PLT.show()

"""
    function is_edge_collision_free(start: tuple, end: tuple, obstacles: list) -> bool:
"""
def is_edge_collision_free(start: tuple, end: tuple, obstacles: list) -> bool:
    NUM_STEPS = 10 # number of points to interpolate along the edge.
    for i in range(NUM_STEPS + 1):
        # linear interpolation between start and end.
        ALPHA = i / NUM_STEPS
        INTERMEDIATE = ((1 - ALPHA) * NP.array(start) + ALPHA * NP.array(end))
        
        # Get polygon corners for the interpolated configuration.
        CORNERS = get_polygon_corners(center = (INTERMEDIATE[0], INTERMEDIATE[1]), theta = INTERMEDIATE[2], width = FREE_BODY_ROBOT_WIDTH, height = FREE_BODY_ROBOT_HEIGHT)
        
        # Check if this configuration collides with any obstacles.
        for (x, y, width, height, theta) in obstacles:
            OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
            
            if is_colliding(CORNERS, OBSTACLE_CORNERS):
                return False # collision detected
    
    return True # no collision along the path.            
            
"""
    function visualize_freeBody_PRM(prm: dict, environment: list) -> None:
"""
def visualize_freeBody_PRM(prm: dict, environment: list) -> None:
    FIGURE, AXES = PLT.subplots()
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = '#ff0000', color = '#ff0000', fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    for NODE, NEIGHBORS in prm.items():
        AXES.plot(NODE[0], NODE[1], 'bo', markersize = 2.5)
    
        for NEIGHBOR in NEIGHBORS:
                x1, y1 = NODE[:2]
                x2, y2 = NEIGHBOR[0][0], NEIGHBOR[0][1]
                AXES.plot([x1, x2], [y1, y2], 'k-', linewidth = 0.5, alpha = 0.5)
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Free Body Robot PRM')
    
    PLT.show()

"""
    function generate_freeBody_robot_sample_configurations(number_of_samples: int, start: tuple, goal: tuple) -> list
"""
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

"""
    function build_prm(robot_type: str, samples: list, environment: list) -> dict:
"""
def build_prm(robot_type: str, samples: list, environment: list) -> dict:
    PRM = {SAMPLE: [] for SAMPLE in samples}
    
    # samples = samples + [start, goal]
    
    NEIGHBORS = []
    
    if robot_type == 'arm':
        print('*** NOT YET SUPPORTED ***')
    elif robot_type == 'freeBody':
        for SAMPLE in samples:
            NEIGHBORS = get_k_nearest_freeBody_robot_configurations(samples, target_config = SAMPLE, orientation_weight = 0.25, k = K_NEIGHBORS)
            SAMPLE_CORNERS = get_polygon_corners(SAMPLE[:2], SAMPLE[2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
            
            for NEIGHBOR in NEIGHBORS:
                NEIGHBOR_CORNERS = get_polygon_corners(NEIGHBOR[0][:2], NEIGHBOR[0][2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
                
                for OBSTACLE in environment:
                    OBSTACLE_CORNERS = get_polygon_corners(center = (OBSTACLE[0], OBSTACLE[1]), width = OBSTACLE[2], height = OBSTACLE[3], theta = OBSTACLE[4])
                    
                    if not is_colliding(SAMPLE_CORNERS, OBSTACLE_CORNERS) and not is_colliding(NEIGHBOR_CORNERS, OBSTACLE_CORNERS) and is_edge_collision_free(start = SAMPLE, end = (NEIGHBOR[0][0], NEIGHBOR[0][1], NEIGHBOR[0][2]), obstacles = environment):
                        PRM[SAMPLE].append(NEIGHBOR)
                        
    return PRM

"""
    function interpolate_path(path: list, steps_per_segment: int) -> list:
"""
def interpolate_path(path, steps_per_segment=10):
    """Interpolate positions and angles between consecutive path points."""
    interpolated_path = []

    for i in range(len(path) - 1):
        start = NP.array(path[i])
        end = NP.array(path[i + 1])

        # Generate interpolated points between start and end
        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            interpolated_point = (1 - alpha) * start + alpha * end
            interpolated_path.append(tuple(interpolated_point))

    interpolated_path.append(path[-1])  # Add the final goal point
    return interpolated_path

"""
    function animate_freeBody_robot(path: list, environment: list, steps_per_segment: int) -> None:
"""
def animate_freeBody_robot(path: list, environment: list, steps_per_segment=10):
    """Animate the freeBody robot moving along the given interpolated path."""
    interpolated_path = interpolate_path(path, steps_per_segment)

    # Set up the figure and axes
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    PLT.title("FreeBody Robot Animation")

    # Plot obstacles in the environment
    for obstacle in environment:
        x, y, width, height, theta = obstacle
        corners = get_polygon_corners((x, y), theta, width, height)
        obstacle_patch = PTCHS.Polygon(corners, closed=True, color='gray', alpha=0.5)
        ax.add_patch(obstacle_patch)

    # Initialize the robot as a rectangle
    robot_patch = PTCHS.Rectangle(
        (interpolated_path[0][0] - FREE_BODY_ROBOT_WIDTH / 2, 
         interpolated_path[0][1] - FREE_BODY_ROBOT_HEIGHT / 2),
        FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT,
        angle=NP.degrees(interpolated_path[0][2]), fill=True, color='blue'
    )
    ax.add_patch(robot_patch)

    def update(frame):
        """Update the robot's position and orientation for each frame."""
        x, y, theta = interpolated_path[frame]
        robot_patch.set_xy((x - FREE_BODY_ROBOT_WIDTH / 2, y - FREE_BODY_ROBOT_HEIGHT / 2))
        robot_patch.angle = NP.degrees(theta)
        return robot_patch,

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(interpolated_path), interval=50, blit=True, repeat=False
    )

    PLT.show()

"""
    function scene_from_file(filename: str) -> list:
"""
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

"""
    function parse_arguments() -> dict:
"""
def parse_arguments() -> dict:
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--start', type = float, nargs = '+', help = 'Start configuration of robot (arm or freeBody).')
    
    PARSER.add_argument('--goal', type = float, nargs = '+', help = 'Start configuration of robot (arm or freeBody).')
    
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg environment.')
    
    return PARSER.parse_args()

"""
    function main():
"""
def main():
    print('\n4. Probabalistic Road-Maps\n')
    
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(filename = ARGS.map)
    
    if ARGS.robot == 'arm':
        START_BASE, START_JOINT, START_END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = ARGS.start[0], theta_2 = ARGS.start[1])
        START = (ARGS.start[0], ARGS.start[1], START_BASE, START_JOINT, START_END_EFFECTOR)
        
        GOAL_BASE, GOAL_JOINT, GOAL_END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = ARGS.goal[0], theta_2 = ARGS.goal[1])
        GOAL = (ARGS.goal[0], ARGS.goal[1], GOAL_BASE, GOAL_JOINT, GOAL_END_EFFECTOR)
        
        SAMPLES = generate_arm_robot_sample_configurations(number_of_samples = NUMBER_OF_SAMPLES, start = START, goal = GOAL)
    elif ARGS.robot == 'freeBody':
        SAMPLES = generate_freeBody_robot_sample_configurations(number_of_samples = NUMBER_OF_SAMPLES, start = tuple(ARGS.start), goal = tuple(ARGS.goal))
        PRM = build_prm(robot_type = 'freeBody', samples = SAMPLES, environment = ENVIRONMENT)
        # visualize_freeBody_PRM(prm = PRM, environment = ENVIRONMENT)
        PATH = a_star_search(PRM, tuple(ARGS.start), tuple(ARGS.goal))
        visualize_path(PATH, PRM, environment = ENVIRONMENT)

if __name__ == '__main__':
    main()