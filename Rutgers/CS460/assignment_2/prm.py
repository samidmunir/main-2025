# IMPORTS
import argparse as ARGPRS
import heapq
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
from matplotlib.animation import FuncAnimation
import math
import networkx as nx
import numpy as NP
import random as RANDOM
from utils import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT,
    get_polygon_corners,
    is_colliding,
    is_colliding_link,
    get_arm_robot_forward_kinematics,
    point_in_circle,
    point_in_rotated_rectangle,
    check_line_intersects_obstacle
)
from nearest_neighbors import (
    get_k_nearest_free_body_robot_configurations,
    get_k_nearest_arm_robot_configurations,
    get_euclidean_distance
)

from collision_checking import (
    handle_drawing_arm_robot
)

# CONSTANTS
K_NEIGHBORS = 6
NUMBER_OF_SAMPLES = 1000
JOINT_RADIUS = 0.5
PRM_NEIGHBOR_RADIUS = 0.4

def interpolate_path_arm(path, environment, steps_per_segment = 10):
    interpolated_path = []
    for i in range(len(path) - 1):
        start, end = NP.array((path[i][0], path[i][1])), NP.array((path[i + 1][0], path[i + 1][1]))
        
        delta = (end - start + NP.pi) % (2 * NP.pi) - NP.pi
        
        for step in range(steps_per_segment):
            alpha = step / steps_per_segment
            interpolated_point = start + alpha * delta
            print('interpolated_point:', interpolated_point)
            theta_1, theta_2 = interpolated_point
            BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (theta_1, theta_2))
            COLLIDING = False
            for OBSTACLE in environment:
                x, y, width, height, theta = OBSTACLE
                OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
                COLLIDING = (
                    is_colliding_link(BASE, JOINT, OBSTACLE_CORNERS) or (is_colliding_link(JOINT, END_EFFECTOR, OBSTACLE_CORNERS)) or point_in_circle(BASE, (x, y), JOINT_RADIUS) or point_in_circle(JOINT, (x, y), JOINT_RADIUS) or point_in_circle(END_EFFECTOR, (x, y), JOINT_RADIUS or point_in_rotated_rectangle(BASE[0], BASE[1], x, y, width, height, theta) or point_in_rotated_rectangle(JOINT[0], JOINT[1], x, y, width, height, theta) or point_in_rotated_rectangle(END_EFFECTOR[0], END_EFFECTOR[1], x, y, width, height, theta)) or check_line_intersects_obstacle(BASE, JOINT, x, y, width, height, theta) or check_line_intersects_obstacle(JOINT, END_EFFECTOR, x, y, width, height, theta)
                )
                if COLLIDING:
                    break
            if not COLLIDING:
                interpolated_path.append(tuple(interpolated_point))
    interpolated_path.append(path[-1])
    return interpolated_path
        

# def interpolate_path_arm(path, steps_per_segment=10):
#     interpolated_path = []
#     for i in range(len(path) - 1):
#         start, end = NP.array((path[i][0], path[i][1])), NP.array((path[i + 1][0], path[i + 1][1]))
        
#         delta = (end - start + NP.pi) % (2 * NP.pi) - NP.pi
        
#         for step in range(steps_per_segment):
#             alpha = step / steps_per_segment
#             interpolated_point = start + alpha * delta
#             print('interpolated_point:', interpolated_point)
#             interpolated_path.append(tuple(interpolated_point))
#     interpolated_path.append(path[-1])
#     return interpolated_path

def animate_arm_robot(path, environment, steps_per_segment=10):
    # Interpolate the path for smooth animation
    interpolated_path = interpolate_path_arm(path, environment, steps_per_segment)

    # Set up the figure and axes
    fig, ax = PLT.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    ax.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

    # Draw the environment (obstacles)
    for obstacle in environment:
        x, y, width, height, theta = obstacle
        corners = get_polygon_corners(center=(x, y), width=width, height=height, theta=theta)
        ax.add_patch(PTCHS.Polygon(corners, closed=True, color='gray', alpha=0.5))

    # Initialize the base, joint, and end-effector positions for the arm robot
    theta_1, theta_2 = path[0][:2]
    base, joint, end_effector = get_arm_robot_forward_kinematics((theta_1, theta_2))

    # Draw the initial configuration of the arm robot
    lines = [
        ax.plot([base[0], joint[0]], [base[1], joint[1]], 'k-', linewidth = 2.0, color = '#000000')[0],  # First arm link
        ax.plot([joint[0], end_effector[0]], [joint[1], end_effector[1]], 'k-', linewidth = 2.0, color = '#000000')[0],  # Second arm link
    ]
    base_patch = ax.plot(base[0], base[1], 'bo', markersize = 2.0)[0]
    joint_patch = ax.plot(joint[0], joint[1], 'ro', markersize=2.0)[0]
    end_effector_patch = ax.plot(end_effector[0], end_effector[1], 'go', markersize = 2.0)[0]
    
    # Create a list to store the trace of the end-effector
    trace, = ax.plot([], [], 'g.', markersize = 2.0) # green dots for the trace.
    
    trace_x, trace_y = [], []
    
    def init():
        trace_x.clear()
        trace_y.clear()
        trace.set_data([], [])
        return lines + [base_patch, joint_patch, end_effector_patch, trace]

    def update(frame):
        """Update the arm robot's configuration for each frame."""
        theta_1, theta_2 = interpolated_path[frame][:2]
        base, joint, end_effector = get_arm_robot_forward_kinematics((theta_1, theta_2))

        # Update the arm links
        lines[0].set_data([base[0], joint[0]], [base[1], joint[1]])
        lines[1].set_data([joint[0], end_effector[0]], [joint[1], end_effector[1]])

        # Update the joint position
        base_patch.set_data([base[0]], [base[1]])
        joint_patch.set_data([joint[0]], [joint[1]])
        end_effector_patch.set_data([end_effector[0]], [end_effector[1]])
        
        # add the current-end effector position to the trace.
        trace_x.append(end_effector[0])
        trace_y.append(end_effector[1])
        trace.set_data(trace_x, trace_y)

        return lines + [base_patch, joint_patch, end_effector_patch, trace]

    # Create the animation
    anim = FuncAnimation(
        fig, update, frames=len(interpolated_path), init_func = init, interval=50, blit=False, repeat=True
    )

    # Display the animation
    PLT.show()

def visualize_path_arm_robot(path: list, prm: dict, environment: list) -> None:
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    # Plot the environment.
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
            
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, color = '#ff0000', fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    # Plot the PRM
    for NODE, NEIGHBORS in prm.items():
        AXES.plot(NODE[4][0], NODE[4][1], 'bo', markersize = 1.0)
        for NEIGHBOR in NEIGHBORS:
                x1, y1 = NODE[4][0], NODE[4][1]
                x2, y2 = NEIGHBOR[4][0], NEIGHBOR[4][1]
                AXES.plot([x1, x2], [y1, y2], 'k-', linewidth = 0.5, alpha = 0.5)
    
    # Plot the solution path
    for i in range(len(path) - 1):
        x1, y1 = path[i][4][:2]  # Unpack (x, y) from each path node
        x2, y2 = path[i + 1][4][:2]
        AXES.plot([x1, x2], [y1, y2], 'r-', linewidth=2.0)  # Path in red
    
    PLT.title(f'PRM w/Path ARM ROBOT')
    PLT.show(block = True)
    PLT.pause(1)
    PLT.close(FIGURE)

def angular_distance(theta1, theta2):
    diff = (theta2 - theta1 + math.pi) % (2 * math.pi) - math.pi
    return abs(diff)

def get_configuration_distance(node1, node2):
    """Calculate the total angular distance between two configurations."""
    # print('\nget_configuration_distance() called...')
    # print('\tnode1:', node1)
    # print('\tnode2:', node2)
    theta1_1, theta1_2 = node1[:2]
    theta2_1, theta2_2 = node2[:2]

    # Sum the angular distances for both joints
    return angular_distance(theta1_1, theta2_1) + angular_distance(theta1_2, theta2_2)

def a_star_search_arm(prm, samples):
    # The start and goal configurations are the first and last keys in the PRM
    start = list(prm.keys())[0]
    print(f'\nstart: {start}')
    goal = list(prm.keys())[-1]
    print(f'goal: {goal}')

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {start: None}
    g_score = {start: 0}
    f_score = {start: get_configuration_distance(node1 = start, node2 = goal)}

    while open_set:
        _, current = heapq.heappop(open_set)
        # print('\ncurrent:', current, '\n')

        if current == goal:
            return reconstruct_path(came_from, current)

        i = 0
        for neighbor in prm[current]:
            # print(f'\tneighbor #{(i + 1)}: ({neighbor[0]}, {neighbor[1]})')
            neighbor_node = neighbor  # Extract the neighbor node
            tentative_g_score = g_score[current] + get_configuration_distance(current, neighbor_node)

            if tentative_g_score < g_score.get(neighbor_node, float('inf')):
                came_from[neighbor_node] = current
                g_score[neighbor_node] = tentative_g_score
                f_score[neighbor_node] = tentative_g_score + get_configuration_distance(neighbor_node, goal)
                heapq.heappush(open_set, (f_score[neighbor_node], neighbor_node))
            i += 1

    return None  # No path found

def reconstruct_path(came_from, current):
    """Reconstruct the path from the goal to the start."""
    path = [current]
    while current in came_from and came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path

def build_arm_robot_prm(environment: list, samples: list, start_config: tuple, goal_config: tuple) -> dict:
    # Convert samples list to list of (theta_1, theta_2, BASE, JOINT, END_EFFECTOR)
    samples_with_forward_kinematics = []
    for sample in samples:
        theta_1, theta_2 = sample
        BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (theta_1, theta_2))
        samples_with_forward_kinematics.append((theta_1, theta_2, BASE, JOINT, END_EFFECTOR))
    
    samples = samples_with_forward_kinematics
    
    PRM = {tuple(sample): [] for sample in samples}
    
    for sample in samples:
        neighbors = get_k_nearest_arm_robot_configurations(configurations = samples, target_configuration = sample, k = K_NEIGHBORS)
        aligned_neighbors = []
        for neighbor in neighbors:
            (reduced, dist) = neighbor
            aligned_neighbors.append(reduced)
        neighbors = aligned_neighbors
        for neighbor in neighbors:
            # print(f'\tneighbor[0]: {neighbor[0]}')
            if get_euclidean_distance(point = (sample[4][0], sample[4][1]), target_point = (neighbor[4][0], neighbor[4][1])) <= PRM_NEIGHBOR_RADIUS:
                PRM[tuple(sample)].append(neighbor)
    
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)

    # Drawing obstacles in environment.
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
            
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, color = '#ff0000', fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    # Drawing PRM
    for NODE, NEIGHBORS in PRM.items():
        # print(f'\nNODE: {NODE}')
        # print(f'\tNEIGHBORS[0]: {NEIGHBORS[0]}')
        # print(f'\tNEIGHBORS[1]: {NEIGHBORS[1]}')
        AXES.plot(NODE[4][0], NODE[4][1], 'bo', markersize = 1.0)
    
        for NEIGHBOR in NEIGHBORS:
                x1, y1 = NODE[4][0], NODE[4][1]
                x2, y2 = NEIGHBOR[4][0], NEIGHBOR[4][1]
                AXES.plot([x1, x2], [y1, y2], 'k-', linewidth = 0.5, alpha = 0.5)
    
    PLT.title(f'Arm robot PRM.')
    PLT.show(block = True)
    PLT.pause(1)
    PLT.close(FIGURE)
    
    return PRM

def generate_sample_arm_robot_configurations(environment: list, number_of_samples: int, start_configuration: tuple, goal_configuration: tuple) -> list:
    # FIGURE, AXES = PLT.subplots()
    # AXES.set_aspect('equal')
    # AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    # AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    SAMPLES = []
    
    SAMPLES = SAMPLES + [start_configuration]
    
    for _ in range(number_of_samples):
        theta_1 = RANDOM.uniform(0.01746, 2 * NP.pi)
        theta_2 = RANDOM.uniform(0.01746, 2 * NP.pi)
        
        BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (theta_1, theta_2))
        COLLIDING = False
        for OBSTACLE in environment:
            x, y, width, height, theta = OBSTACLE
            OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
            
            OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, color = '#ff0000', fill = True, alpha = 0.5)
        
            # AXES.add_patch(OBSTACLE_RECTANGLE)
        
            COLLIDING = (
                is_colliding_link(BASE, JOINT, OBSTACLE_CORNERS) or (is_colliding_link(JOINT, END_EFFECTOR, OBSTACLE_CORNERS)) or point_in_circle(BASE, (x, y), JOINT_RADIUS) or point_in_circle(JOINT, (x, y), JOINT_RADIUS) or point_in_circle(END_EFFECTOR, (x, y), JOINT_RADIUS or point_in_rotated_rectangle(BASE[0], BASE[1], x, y, width, height, theta) or point_in_rotated_rectangle(JOINT[0], JOINT[1], x, y, width, height, theta) or point_in_rotated_rectangle(END_EFFECTOR[0], END_EFFECTOR[1], x, y, width, height, theta)) or check_line_intersects_obstacle(BASE, JOINT, x, y, width, height, theta) or check_line_intersects_obstacle(JOINT, END_EFFECTOR, x, y, width, height, theta)
            )
            
            if COLLIDING:
                break
            
        if not COLLIDING:
            SAMPLES.append((theta_1, theta_2))
        
    SAMPLES = SAMPLES + [goal_configuration]
    
    # for SAMPLE in SAMPLES:
    #     theta_1, theta_2 = SAMPLE
    #     BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (theta_1, theta_2))
        # handle_drawing_arm_robot(FIGURE = FIGURE, AXES = AXES, base = BASE, joint1 = JOINT, end_effector = END_EFFECTOR, joint_color = '#000000', line_color = '#000000')
    
    # PLT.title(f'Testing valid arm robot configurations.')
    # PLT.show(block = True)
    # PLT.pause(1)
    # PLT.close(FIGURE)
    
    return SAMPLES

def euclidean_distance(node1, node2):
    return NP.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

def a_star_search(prm_graph, start, goal):
    start = list(prm_graph.keys())[0]
    print(f'\nstart: {start}')
    goal = list(prm_graph.keys())[-1]
    print(f'goal: {goal}')
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
    path = [current]
    while came_from[current] is not None:
        current = came_from[current]
        path.append(current)
    return path[::-1]

def interpolate_path(path, steps_per_segment=10):
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
    PRM = {tuple(sample): [] for sample in samples}
    if robot_type == 'freeBody':
        for sample in samples:
            neighbors = get_k_nearest_free_body_robot_configurations(samples, sample, angle_weight = 0.25, k = K_NEIGHBORS)
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
    ARGS = parse_arguments()
    
    environment = scene_from_file(ARGS.map)
    
    if ARGS.robot == 'arm':
        samples = generate_sample_arm_robot_configurations(environment = environment, number_of_samples = NUMBER_OF_SAMPLES, start_configuration = ARGS.start, goal_configuration = ARGS.goal)
        prm = build_arm_robot_prm(environment = environment, samples = samples, start_config = ARGS.start, goal_config = ARGS.goal)
        path = a_star_search_arm(prm = prm, samples = samples)
        # print('\nPATH:', path)
        if path:    
            print('\nPATH[0]:', path[0])
            print('PATH[-1]:', path[-1])
            visualize_path_arm_robot(path = path, prm = prm , environment = environment)
            animate_arm_robot(path = path, environment = environment, steps_per_segment = 20)
        else:
            print('No path found between the start and goal configurations.')
    elif ARGS.robot == 'freeBody':
        samples = generate_freeBody_robot_sample_configurations(NUMBER_OF_SAMPLES, tuple(ARGS.start), tuple(ARGS.goal))
        prm = build_prm(ARGS.robot, samples, environment)
        visualize_freeBody_PRM(prm, environment)
        path = a_star_search(prm, start = ARGS.start, goal = ARGS.goal)
        if path:
            visualize_path(path, prm, environment)
            animate_freeBody_robot(path, environment)
        else:
            print('No path found between the start and goal configurations.')

if __name__ == '__main__':
    main()