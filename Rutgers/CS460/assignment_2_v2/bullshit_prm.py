# """
#     4. PRM: prm.py
# """

# """
#     Python IMPORTS
# """
# import argparse
# import heapq
# import math as MATH
# import random as RANDOM
# import matplotlib.pyplot as PLT
# import matplotlib.patches as PTCHS
# import numpy as NP

# """
#     CONSTANTS (from component_1.py)
# """
# from component_1 import (
#         ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, 
#         FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT,ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH, PRM_NUM_SAMPLES
#     )
# """
#     IMPORTS (from nearest_neighbors.py)
# """
# from nearest_neighbors import (
#         calculate_arm_robot_end_effector_positions, 
#         find_k_nearest_end_effector_positions, 
#         find_k_nearest_free_body_configs
#     )
# """
#     IMPORTS (from collision_checking.py)
# """
# from collision_checking import (
#     is_colliding, 
#     is_colliding_link,
#     is_line_intersecting,
#     get_polygon_corners
# )

# ######################################################################

# # freeBody Robot
# """
#     function generate_random_configs_free_body_robot():
# """
# def generate_random_configs_free_body_robot(num_samples: int) -> list:
#     RANDOM_CONFIGS = []
    
#     for i in range(num_samples):
#         x = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#         y = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#         theta = RANDOM.uniform(0, 2 * NP.pi)
        
#         RANDOM_CONFIGS.append((x, y, theta))
        
#     return RANDOM_CONFIGS

# """
#     function visualize_scene_free_body_robot():
# """
# def visualize_scene_free_body_robot(obstacles: list, random_samples: list, start_config, goal_config):
#     FIGURE, AXES = PLT.subplots()
    
#     # Adding obstacles to the environment.
#     for OBSTACLE in obstacles:
#         OBSTACLE_COLOR = '#ff0000'
#         x, y, width, height, orientation = OBSTACLE
#         OBSTACLE_RECTANGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(orientation), color = '#ff0000', edgecolor = '#ff0000', alpha = 0.5)
#         AXES.add_patch(OBSTACLE_RECTANGLE)
    
#     # Adding random samples to the environment.
#     for RANDOM_SAMPLE in random_samples:
#         X, Y = RANDOM_SAMPLE[0], RANDOM_SAMPLE[1]
#         AXES.plot(X, Y, 'o', color = '#000000', alpha = 0.5)
    
#     # Adding start configuration to the environment.
#     START_CONFIG_X, START_CONFIG_Y = start_config[:2]
#     START_CONFIG_COLOR = '#00ff00'
#     START_CONFIG_RECTANGLE = PTCHS.Rectangle((START_CONFIG_X, START_CONFIG_Y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(start_config[2]), color = START_CONFIG_COLOR)
#     AXES.add_patch(START_CONFIG_RECTANGLE)
    
#     # Adding goal configuration to the environment.
#     GOAL_CONFIG_X, GOAL_CONFIG_Y = goal_config[:2]
#     GOAL_CONFIG_COLOR = '#00ffff'
#     GOAL_CONFIG_RECTANGLE = PTCHS.Rectangle((GOAL_CONFIG_X, GOAL_CONFIG_Y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(goal_config[2]), color = GOAL_CONFIG_COLOR)
#     AXES.add_patch(GOAL_CONFIG_RECTANGLE)
    
#     # Setting the environment properties.
#     AXES.set_aspect('equal')
#     AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#     AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
#     PLT.show()

# ######################################################################

# # arm Robot

# """
#     function get_arm_robot_joint_positions():
# """
# def get_arm_robot_joint_positions(theta_1, theta_2):
#     joint_1_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
#     joint_1_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    
#     end_effector_x = joint_1_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
#     end_effector_y = joint_1_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    
#     return (0, 0), (joint_1_x, joint_1_y), (end_effector_x, end_effector_y)

# """
#     function generate_random_configs_arm_robot():
# """
# def generate_random_configs_arm_robot(num_samples: int) -> list:
#     RANDOM_CONFIGS = []
    
#     for i in range(num_samples):
#         theta_1 = RANDOM.uniform(0, 2 * NP.pi)
#         theta_2 = RANDOM.uniform(0, 2 * NP.pi)
        
#         RANDOM_CONFIGS.append((theta_1, theta_2))
        
#     return RANDOM_CONFIGS


# """
#     function get_random_arm_robot_samples():
# """
# def get_random_arm_robot_samples(random_configs: list) -> list:
#     RANDOM_SAMPLES = []
    
#     for CONFIG in random_configs:
#         RANDOM_SAMPLE = get_arm_robot_joint_positions(CONFIG[0], CONFIG[1])
        
#         RANDOM_SAMPLES.append((CONFIG, RANDOM_SAMPLE)) # appending THETA_1, THETA_2, BASE, JOINT_1, END_EFFECTOR
        
#     return RANDOM_SAMPLES
    
# """
#     function visualize_scene_arm_robot():
# """
# def visualize_scene_arm_robot(obstacles: list, random_samples: list, start_config, goal_config):
#     FIGURE, AXES = PLT.subplots()
    
#     # Adding obstacles to the environment.
#     for OBSTACLE in obstacles:
#         x, y, width, height, angle = OBSTACLE
#         OBSTACLE_CORNERS = get_polygon_corners((x, y), width, height, angle)
        
#         OBSTACLE_COLOR = '#ff0000'
#         OBSTACLE_POLYGON = PTCHS.Polygon(OBSTACLE_CORNERS, color = OBSTACLE_COLOR, fill = True, closed = True, alpha = 0.5)
        
#         AXES.add_patch(OBSTACLE_POLYGON)
        
    
#     # Drawing randomly sampled arm robot *END_EFFECTOR* positions.
#     for RANDOM_SAMPLE in random_samples:
#         CONFIG, (BASE, JOINT_1, END_EFFECTOR) = RANDOM_SAMPLE
        
#         END_EFFECTOR_X = END_EFFECTOR[0]
#         END_EFFECTOR_Y = END_EFFECTOR[1]
        
#         AXES.plot(END_EFFECTOR_X, END_EFFECTOR_Y, 'o', color = '#000000')
        
#     # Drawing START_END_EFFECTOR
#     START_CONFIG_BASE = start_config[0]
#     START_CONFIG_JOINT_1 = start_config[1]
#     START_CONFIG_END_EFFECTOR = start_config[2]
    
#     AXES.plot(START_CONFIG_END_EFFECTOR[0], START_CONFIG_END_EFFECTOR[1], 'o', color = '#00ff00')
    
#     # Drawing GOAL_END_EFFECTOR
#     GOAL_CONFIG_BASE = goal_config[0]
#     GOAL_CONFIG_JOINT_1 = goal_config[1]
#     GOAL_CONFIG_END_EFFECTOR = goal_config[2]
    
#     AXES.plot(GOAL_CONFIG_END_EFFECTOR[0], GOAL_CONFIG_END_EFFECTOR[1], 'o', color = '#ff00ff')
        
#     AXES.set_aspect('equal')
#     AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#     AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
#     PLT.show()

# ######################################################################

# """
#     function build_free_body_robot_road_map():
# """
# def build_free_body_robot_road_map(samples, obstacles, k):
#     ROAD_MAP = {SAMPLE: [] for SAMPLE in samples}
    
#     for SAMPLE in samples:
#         SAMPLE_CORNERS = get_polygon_corners((SAMPLE[0], SAMPLE[1]), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, SAMPLE[2])
#         NEIGHBORS = find_k_nearest_free_body_configs(samples, k, SAMPLE)
#         for NEIGHBOR in NEIGHBORS:
#             NEIGHBOR_CORNERS = get_polygon_corners((NEIGHBOR[0], NEIGHBOR[1]), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, NEIGHBOR[2])
#             for OBSTACLE in obstacles:
#                 OBSTACLE_CORNERS = get_polygon_corners((OBSTACLE[0], OBSTACLE[1]), OBSTACLE[2], OBSTACLE[3], OBSTACLE[4])
#                 if not is_colliding(SAMPLE_CORNERS, OBSTACLE_CORNERS) and not is_colliding(NEIGHBOR_CORNERS, OBSTACLE_CORNERS):
#                     ROAD_MAP[SAMPLE].append(NEIGHBOR)
#                     ROAD_MAP[NEIGHBOR].append(SAMPLE) # bidirectional connection.
#     return ROAD_MAP

# """
#     function visualize_free_body_robot_road_map():
# """
# def visualize_free_body_robot_road_map(samples, obstacles, road_map, robot_type: str):
#     FIGURE, AXES = PLT.subplots()
    
#     # Adding the obstacles to the environment.
#     for OBSTACLE in obstacles:
#         OBSTACLE_COLOR = '#ff0000'
#         x, y, width, height, orientation = OBSTACLE
#         OBSTACLE_RECTANGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(orientation), color = '#ff0000', edgecolor = '#ff0000', alpha = 0.5)
#         AXES.add_patch(OBSTACLE_RECTANGLE)
    
#     # Adding the samples to the environment.
#     for SAMPLE in samples:
#         X, Y = SAMPLE[0], SAMPLE[1]
#         # AXES.plot(X, Y, 'o', color = '#000000', alpha = 0.5)
#         AXES.scatter(X, Y, color = '#000000', s = 10, alpha = 0.5)
        
#         # Adding the road map connections.
#         for NEIGHBOR in road_map[SAMPLE]:
#             NX, NY = NEIGHBOR[0], NEIGHBOR[1]
#             AXES.plot([X, NX], [Y, NY], color = '#000000', lw = 0.25, alpha = 0.50)
    
#     # Setting the environment properties.
#     AXES.set_aspect('equal')
#     AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#     AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
#     PLT.show()

# """
#     function scene_from_file():
# """
# def scene_from_file(filename: str) -> list:
#     OBSTACLES = []
    
#     with open(filename, 'r') as FILE:
#         for LINE in FILE:
#             x, y, width, height, angle = map(float, LINE.strip().split(','))
#             OBSTACLES.append((x, y, width, height, angle))
    
#     return OBSTACLES

# """
#     function parse_arguments():
# """
# def parse_arguments():
#     PARSER = argparse.ArgumentParser(description = 'Path Planning with PRM')
    
#     PARSER.add_argument('--robot', choices = ['arm', 'freeBody'], required = True, type = str, help = 'Type of robot (arm or freeBody)')
#     PARSER.add_argument('--start', nargs = '+', type = float, required = True, help = 'Start configuration of robot')
#     PARSER.add_argument('--goal', nargs = '+', type = float, required = True, help = 'Goal configuration of robot')
#     PARSER.add_argument('--map', type = str, required = True, help = 'Filename containing map of environment')
    
#     return PARSER.parse_args()

# """
#     function main():
# """
# def main():
#     ARGS = parse_arguments()
    
#     if ARGS.robot == 'arm':
#         OBSTACLES = scene_from_file(ARGS.map)
#         RANDOM_CONFIGS = generate_random_configs_arm_robot(num_samples = PRM_NUM_SAMPLES)
#         RANDOM_SAMPLES = get_random_arm_robot_samples(random_configs = RANDOM_CONFIGS)
#         visualize_scene_arm_robot(obstacles = OBSTACLES, random_samples = RANDOM_SAMPLES, start_config = get_arm_robot_joint_positions(ARGS.start[0], ARGS.start[1]), goal_config = get_arm_robot_joint_positions(ARGS.goal[0], ARGS.goal[1]))
#     elif ARGS.robot == 'freeBody':
#         OBSTACLES = scene_from_file(ARGS.map)
#         RANDOM_SAMPLES = generate_random_configs_free_body_robot(num_samples = PRM_NUM_SAMPLES)
#         # visualize_scene_free_body_robot(obstacles = OBSTACLES, random_samples = RANDOM_SAMPLES, start_config = ARGS.start, goal_config = ARGS.goal)
#         START_CONFIG = tuple(ARGS.start)
#         GOAL_CONFIG = tuple(ARGS.goal)
#         RANDOM_SAMPLES.append(START_CONFIG)
#         RANDOM_SAMPLES.append(GOAL_CONFIG)
#         ROAD_MAP = build_free_body_robot_road_map(obstacles = OBSTACLES, samples = RANDOM_SAMPLES, k = PRM_NUM_SAMPLES)
#         visualize_free_body_robot_road_map(obstacles = OBSTACLES, samples = RANDOM_SAMPLES, road_map = ROAD_MAP, robot_type = 'freeBody')
#         # print('\n', RANDOM_SAMPLES[0])
#         # print('\n', OBSTACLES[0])
#         # print(ROAD_MAP[RANDOM_SAMPLES[0]])

# if __name__ == '__main__':
#     main()

import argparse
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as NP
import heapq

from collision_checking import get_polygon_corners, project, get_axes

# Normalize angle between 0 and 360
def normalize_angle(angle):
    return angle % 360

# Function to calculate the positions of each joint and end-effector for a 2-joint arm robot
def calculate_arm_positions(joint1_angle, joint2_angle, joint1_length=10.0, joint2_length=10.0):
    # Base position
    base_x, base_y = 0, 0

    # Joint 1 position
    joint1_x = joint1_length * math.cos(math.radians(joint1_angle))
    joint1_y = joint1_length * math.sin(math.radians(joint1_angle))

    # End-effector position (Joint 2 relative to Joint 1)
    end_effector_x = joint1_x + joint2_length * math.cos(math.radians(joint1_angle + joint2_angle))
    end_effector_y = joint1_y + joint2_length * math.sin(math.radians(joint1_angle + joint2_angle))

    return (base_x, base_y), (joint1_x, joint1_y), (end_effector_x, end_effector_y)

# Function to generate random samples in the environment
def generate_samples(num_samples, width, height, obstacle_check_fn, robot_type):
    samples = []
    for _ in range(num_samples):
        while True:
            if robot_type == 'arm':
                joint1 = random.uniform(0, 360)
                joint2 = random.uniform(0, 360)
                config = (joint1, joint2)
            else:  # freeBody robot
                x = random.uniform(-20, 20)
                y = random.uniform(-20, 20)
                orientation = random.uniform(0, 360)
                config = (x, y, orientation)
            if not obstacle_check_fn(config):
                samples.append(config)
                break
    return samples

"""
    function is_colliding():
"""
def is_colliding(robot_corners, obstacle_corners):
    AXES = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])
    
    for AXIS in AXES:
        min_1, max_1 = project(robot_corners, AXIS)
        min_2, max_2 = project(obstacle_corners, AXIS)
        
        if max_1 < min_2 or max_2 < min_1:
            return False
    
    return True

# Check for collision between a robot and an obstacle
def is_collision(config, obstacles):
    if len(config) == 3:  # freeBody robot
        x, y, orientation = config
        CONFIG_CORNERS = get_polygon_corners((x, y), 0.3, 0.5, orientation)
        for obstacle in obstacles:
            OBSTACLE_CORNERS = get_polygon_corners((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], obstacle[4])
            # obs_x, obs_y, obs_width, obs_height, orientation = obstacle
            # if obs_x - obs_width / 2 <= x <= obs_x + obs_width / 2 and obs_y - obs_height / 2 <= y <= obs_y + obs_height / 2:
            #     return True
            if is_colliding(CONFIG_CORNERS, OBSTACLE_CORNERS):
                return True
    return False

# Get k-nearest neighbors for a given configuration
def get_k_nearest_neighbors(config, samples, k):
    distances = [(other, math.sqrt((config[0] - other[0]) ** 2 + (config[1] - other[1]) ** 2)) for other in samples]
    distances.sort(key=lambda x: x[1])
    return [neighbor for neighbor, _ in distances[:k]]

# Build the PRM roadmap by connecting samples to their k-nearest neighbors
def build_roadmap(samples, obstacles, k, obstacle_check_fn):
    roadmap = {sample: [] for sample in samples}
    for sample in samples:
        neighbors = get_k_nearest_neighbors(sample, samples, k)
        for neighbor in neighbors:
            # for OBSTACLE in obstacles:
                # if not obstacle_check_fn((sample, OBSTACLE)) and not obstacle_check_fn((neighbor, OBSTACLE)):
                #     roadmap[sample].append(neighbor)
                #     roadmap[neighbor].append(sample)  # Bidirectional connection
            for obstacle in obstacles:
                
                if not obstacle_check_fn((sample, neighbor)) and not obstacle_check_fn((neighbor, obstacle)):
                    roadmap[sample].append(neighbor)
                    roadmap[neighbor].append(sample)  # Bidirectional connection
    return roadmap

# Heuristic function for A* search (Euclidean distance)
def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# A* search algorithm to find the shortest path in the roadmap
def astar(roadmap, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    g_score = {sample: float('inf') for sample in roadmap}
    g_score[start] = 0
    f_score = {sample: float('inf') for sample in roadmap}
    f_score[start] = heuristic(start, goal)

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        for neighbor in roadmap[current]:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Visualize the roadmap and the found path for both arm robot and freeBody robot
def visualize_roadmap(samples, obstacles, roadmap, path=None, robot_type='freeBody'):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        obs_x, obs_y, obs_width, obs_height, orientation = obstacle
        rect = patches.Rectangle(
            (obs_x - obs_width / 2, obs_y - obs_height / 2),
            obs_width, obs_height, edgecolor='black', facecolor='gray', angle = orientation
        )
        ax.add_patch(rect)

    # Plot roadmap (nodes and edges)
    for sample in roadmap:
        if robot_type == 'arm':
            # Calculate the end-effector position for the arm robot
            base, joint1, end_effector = calculate_arm_positions(sample[0], sample[1])
            # Plot the arm as connected line segments
            ax.plot([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], 'b-', lw=2, label="Arm")
            # Plot the joints
            ax.scatter([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], color='red', s=50, zorder=5, label="Joints")
        else:
            x, y, _ = sample  # Use x, y for the freeBody robot
            ax.scatter(x, y, color='blue', s=10)

        for neighbor in roadmap[sample]:
            if robot_type == 'arm':
                # Calculate the end-effector position for the arm robot's neighbor
                base_n, joint1_n, end_effector_n = calculate_arm_positions(neighbor[0], neighbor[1])
                ax.plot([end_effector[0], end_effector_n[0]], [end_effector[1], end_effector_n[1]], color='blue', lw=0.5)
            else:
                nx, ny, _ = neighbor
                ax.plot([x, nx], [y, ny], color='blue', lw=0.5)

    # Plot the path if available
    if path:
        for i in range(len(path) - 1):
            if robot_type == 'arm':
                base_1, joint1_1, end_effector_1 = calculate_arm_positions(path[i][0], path[i][1])
                base_2, joint1_2, end_effector_2 = calculate_arm_positions(path[i + 1][0], path[i + 1][1])
                ax.plot([end_effector_1[0], end_effector_2[0]], [end_effector_1[1], end_effector_2[1]], color='red', lw=2)
            else:
                x1, y1, _ = path[i]
                x2, y2, _ = path[i + 1]
                ax.plot([x1, x2], [y1, y2], color='red', lw=2)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Load obstacles from the map file
def load_obstacles(filename):
    obstacles = []
    with open(filename, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split(',')))
            if len(parts) == 5:
                x, y, width, height, orientation = parts
                # For simplicity, we ignore orientation for now in the obstacle definition
                obstacles.append((x, y, width, height, orientation))
    return obstacles

# Main function to parse input arguments and run the PRM algorithm
def main():
    parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
    parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
    parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation for freeBody or joint1, joint2 for arm)')
    parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation for freeBody or joint1, joint2 for arm)')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
    args = parser.parse_args()

    # Load obstacles from the map file
    obstacles = load_obstacles(args.map)

    # Generate random samples in free space
    num_samples = 500  # You can adjust this
    k = 6  # Number of nearest neighbors
    width, height = 20, 20  # Environment dimensions

    start_config = tuple(args.start)
    goal_config = tuple(args.goal)
    
    # Add start and goal to samples
    samples = generate_samples(num_samples, width, height, lambda config: is_collision(config, obstacles), args.robot)
    samples.append(start_config)
    samples.append(goal_config)

    # Build the roadmap
    roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)

    # Find the shortest path using A* search
    path = astar(roadmap, start_config, goal_config)

    # Visualize the roadmap and the path
    visualize_roadmap(samples, obstacles, roadmap, path, args.robot)

if __name__ == "__main__":
    main()