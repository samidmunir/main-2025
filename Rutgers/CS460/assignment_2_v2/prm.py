"""
    4. PRM: prm.py
"""

"""
    Python IMPORTS
"""
import argparse
import heapq
import math as MATH
import random as RANDOM
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

"""
    CONSTANTS (from component_1.py)
"""
from component_1 import (
        ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, 
        FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT,ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH, PRM_NUM_SAMPLES
    )
"""
    IMPORTS (from nearest_neighbors.py)
"""
from nearest_neighbors import (
        calculate_arm_robot_end_effector_positions, find_k_nearest_end_effector_positions, find_k_nearest_free_body_configs
    )
"""
    IMPORTS (from collision_checking.py)
"""
from collision_checking import (
    is_colliding, 
    is_colliding_link
)

"""
    function is_line_intersecting():
"""
def is_line_intersecting(p1, p2, q1, q2):
    def orientation(a, b, c):
        VALUE = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        
        return 0 if VALUE == 0 else (1 if VALUE > 0 else -1)
    
    O1 = orientation(p1, p2, q1)
    O2 = orientation(p1, p2, q2)
    O3 = orientation(q1, q2, p1)
    O4 = orientation(q1, q2, p2)
    
    return (O1 != O2) and (O3 != O4)

"""
    function is_colliding_link():
"""
def is_colliding_link(link_start, link_end, obstacle_corners):
    for i in range(len(obstacle_corners)):
        CORNER_1 = obstacle_corners[i]
        CORNER_2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        
        if is_line_intersecting(link_start, link_end, CORNER_1, CORNER_2):
            return True
    
    return False

"""
    function get_polygon_corners():
"""
def get_polygon_corners(center, width, height, angle):
    w, h = width / 2, height / 2
    CORNERS = NP.array(
        [
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ]
    )
    
    COS_THETA, SIN_THETA = NP.cos(angle), NP.sin(angle)
    ROTATION_MATRIX = NP.array(
        [
            [COS_THETA, -SIN_THETA],
            [SIN_THETA, COS_THETA]
        ]
    )
    
    ROTATED_CORNERS = CORNERS @ ROTATION_MATRIX.T
    
    return ROTATED_CORNERS + NP.array(center)

######################################################################

# freeBody Robot
"""
    function generate_random_configs_free_body_robot():
"""
def generate_random_configs_free_body_robot(num_samples: int) -> list:
    RANDOM_CONFIGS = []
    
    for i in range(num_samples):
        x = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
        y = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
        theta = RANDOM.uniform(0, 2 * NP.pi)
        
        RANDOM_CONFIGS.append((x, y, theta))
        
    return RANDOM_CONFIGS

"""
    function visualize_scene_free_body_robot():
"""
def visualize_scene_free_body_robot(obstacles: list, random_samples: list, start_config, goal_config):
    FIGURE, AXES = PLT.subplots()
    
    # Adding obstacles to the environment.
    for OBSTACLE in obstacles:
        OBSTACLE_COLOR = '#ff0000'
        x, y, width, height, orientation = OBSTACLE
        OBSTACLE_RECTANGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(orientation), color = '#ff0000', edgecolor = '#ff0000', alpha = 0.5)
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    # Adding random samples to the environment.
    for RANDOM_SAMPLE in random_samples:
        X, Y = RANDOM_SAMPLE[0], RANDOM_SAMPLE[1]
        AXES.plot(X, Y, 'o', color = '#000000', alpha = 0.5)
    
    # Adding start configuration to the environment.
    START_CONFIG_X, START_CONFIG_Y = start_config[:2]
    START_CONFIG_COLOR = '#00ff00'
    START_CONFIG_RECTANGLE = PTCHS.Rectangle((START_CONFIG_X, START_CONFIG_Y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(start_config[2]), color = START_CONFIG_COLOR)
    AXES.add_patch(START_CONFIG_RECTANGLE)
    
    # Adding goal configuration to the environment.
    GOAL_CONFIG_X, GOAL_CONFIG_Y = goal_config[:2]
    GOAL_CONFIG_COLOR = '#00ffff'
    GOAL_CONFIG_RECTANGLE = PTCHS.Rectangle((GOAL_CONFIG_X, GOAL_CONFIG_Y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(goal_config[2]), color = GOAL_CONFIG_COLOR)
    AXES.add_patch(GOAL_CONFIG_RECTANGLE)
    
    # Setting the environment properties.
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
    PLT.show()

######################################################################

# arm Robot

"""
    function get_arm_robot_joint_positions():
"""
def get_arm_robot_joint_positions(theta_1, theta_2):
    joint_1_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    joint_1_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    
    end_effector_x = joint_1_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    end_effector_y = joint_1_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    
    return (0, 0), (joint_1_x, joint_1_y), (end_effector_x, end_effector_y)

"""
    function generate_random_configs_arm_robot():
"""
def generate_random_configs_arm_robot(num_samples: int) -> list:
    RANDOM_CONFIGS = []
    
    for i in range(num_samples):
        theta_1 = RANDOM.uniform(0, 2 * NP.pi)
        theta_2 = RANDOM.uniform(0, 2 * NP.pi)
        
        RANDOM_CONFIGS.append((theta_1, theta_2))
        
    return RANDOM_CONFIGS


"""
    function get_random_arm_robot_samples():
"""
def get_random_arm_robot_samples(random_configs: list) -> list:
    RANDOM_SAMPLES = []
    
    for CONFIG in random_configs:
        RANDOM_SAMPLE = get_arm_robot_joint_positions(CONFIG[0], CONFIG[1])
        
        RANDOM_SAMPLES.append((CONFIG, RANDOM_SAMPLE)) # appending THETA_1, THETA_2, BASE, JOINT_1, END_EFFECTOR
        
    return RANDOM_SAMPLES
    
"""
    function visualize_scene_arm_robot():
"""
def visualize_scene_arm_robot(obstacles: list, random_samples: list, start_config, goal_config):
    FIGURE, AXES = PLT.subplots()
    
    # Adding obstacles to the environment.
    for OBSTACLE in obstacles:
        x, y, width, height, angle = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners((x, y), width, height, angle)
        
        OBSTACLE_COLOR = '#ff0000'
        OBSTACLE_POLYGON = PTCHS.Polygon(OBSTACLE_CORNERS, color = OBSTACLE_COLOR, fill = True, closed = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_POLYGON)
        
    
    # Drawing randomly sampled arm robot *END_EFFECTOR* positions.
    for RANDOM_SAMPLE in random_samples:
        CONFIG, (BASE, JOINT_1, END_EFFECTOR) = RANDOM_SAMPLE
        
        END_EFFECTOR_X = END_EFFECTOR[0]
        END_EFFECTOR_Y = END_EFFECTOR[1]
        
        AXES.plot(END_EFFECTOR_X, END_EFFECTOR_Y, 'o', color = '#000000')
        
    # Drawing START_END_EFFECTOR
    START_CONFIG_BASE = start_config[0]
    START_CONFIG_JOINT_1 = start_config[1]
    START_CONFIG_END_EFFECTOR = start_config[2]
    
    AXES.plot(START_CONFIG_END_EFFECTOR[0], START_CONFIG_END_EFFECTOR[1], 'o', color = '#00ff00')
    
    # Drawing GOAL_END_EFFECTOR
    GOAL_CONFIG_BASE = goal_config[0]
    GOAL_CONFIG_JOINT_1 = goal_config[1]
    GOAL_CONFIG_END_EFFECTOR = goal_config[2]
    
    AXES.plot(GOAL_CONFIG_END_EFFECTOR[0], GOAL_CONFIG_END_EFFECTOR[1], 'o', color = '#ff00ff')
        
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
    PLT.show()

######################################################################

"""
    function build_free_body_robot_road_map():
"""
def build_free_body_robot_road_map(samples, obstacles, k):
    ROAD_MAP = {SAMPLE: [] for SAMPLE in samples}
    
    for SAMPLE in samples:
        NEIGHBORS = find_k_nearest_free_body_configs(samples, k, SAMPLE)
        for NEIGHBOR in NEIGHBORS:
            ROAD_MAP[SAMPLE].append(NEIGHBOR)
            ROAD_MAP[NEIGHBOR].append(SAMPLE) # bidirectional connection.
    
    return ROAD_MAP
"""
    function scene_from_file():
"""
def scene_from_file(filename: str) -> list:
    OBSTACLES = []
    
    with open(filename, 'r') as FILE:
        for LINE in FILE:
            x, y, width, height, angle = map(float, LINE.strip().split(','))
            OBSTACLES.append((x, y, width, height, angle))
    
    return OBSTACLES

"""
    function parse_arguments():
"""
def parse_arguments():
    PARSER = argparse.ArgumentParser(description = 'Path Planning with PRM')
    
    PARSER.add_argument('--robot', choices = ['arm', 'freeBody'], required = True, type = str, help = 'Type of robot (arm or freeBody)')
    PARSER.add_argument('--start', nargs = '+', type = float, required = True, help = 'Start configuration of robot')
    PARSER.add_argument('--goal', nargs = '+', type = float, required = True, help = 'Goal configuration of robot')
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containing map of environment')
    
    return PARSER.parse_args()

"""
    function build_prm():
"""
"""
def build_prm(robot_type: str, configs, k = 6, obstacles = []):
    GRAPH = {i: [] for i in range(len(configs))}
    if robot_type == 'arm':
        END_EFFECTOR_POSITIONS, _ = calculate_arm_robot_end_effector_positions(configs, configs[0])
        for i, position in enumerate(END_EFFECTOR_POSITIONS):
            NEIGHBORS = find_k_nearest_end_effector_positions(END_EFFECTOR_POSITIONS, position[2:], k)
            for j in NEIGHBORS:
                if is_collision_free_path(configs[i], configs[j], obstacles, robot_type):
                    GRAPH[i].append((j[0], NP.linalg.norm(configs[i] - configs[j[0]])))
    else:
        for i, config in enumerate(configs):
            neighbors = find_k_nearest_free_body_configs(configs, k, config)
            for j in neighbors:
                print(f'\nconfig', config)
                print(f'j', j)
                if is_collision_free_path(config_1=config, config_2=j, obstacles=obstacles, robot_type=robot_type):
                    GRAPH[i].append((j, NP.linalg.norm(config - configs[j])))
    return GRAPH
"""

"""
    function main():
"""
def main():
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        OBSTACLES = scene_from_file(ARGS.map)
        RANDOM_CONFIGS = generate_random_configs_arm_robot(num_samples = PRM_NUM_SAMPLES)
        RANDOM_SAMPLES = get_random_arm_robot_samples(random_configs = RANDOM_CONFIGS)
        visualize_scene_arm_robot(obstacles = OBSTACLES, random_samples = RANDOM_SAMPLES, start_config = get_arm_robot_joint_positions(ARGS.start[0], ARGS.start[1]), goal_config = get_arm_robot_joint_positions(ARGS.goal[0], ARGS.goal[1]))
    elif ARGS.robot == 'freeBody':
        OBSTACLES = scene_from_file(ARGS.map)
        RANDOM_SAMPLES = generate_random_configs_free_body_robot(num_samples = PRM_NUM_SAMPLES)
        visualize_scene_free_body_robot(obstacles = OBSTACLES, random_samples = RANDOM_SAMPLES, start_config = ARGS.start, goal_config = ARGS.goal)
        build_free_body_robot_road_map(obstacles = OBSTACLES, samples = RANDOM_SAMPLES, k = PRM_NUM_SAMPLES)

if __name__ == '__main__':
    main()