"""
    2. Nearest neighbors with linear search approach: nearest_neighbors.py
        - main()
        - parse_arguments()
"""

# IMPORTS
import argparse as ARGPRS
import math as MATH
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
import time as TIME

from component_1 import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    OBSTACLE_MIN_SIZE,
    OBSTACLE_MAX_SIZE
)

# CONSTANTS
ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5
FREE_BODY_ROBOT_WIDTH = 0.5
FREE_BODY_ROBOT_HEIGHT = 0.3

"""
    function get_angular_difference(theta_1, theta_2) -> float:
"""
def get_angular_difference(theta_1: float, theta_2: float) -> float:
    ANGR_DIFF = NP.abs(theta_1 - theta_2) % (2 * NP.pi)
    
    return min(ANGR_DIFF, 2 * NP.pi - ANGR_DIFF)

"""
    function get_k_nearest_freeBody_robot_configurations(configs, target_config, k) -> list:
"""
def get_k_nearest_freeBody_robot_configurations(configs: list, target_config: tuple, orientation_weight: float, k: int) -> list:
    CONFIG_DISTS = []
    
    TARGET_X, TARGET_Y, TARGET_THETA = target_config
    
    for CONFIG in configs:
        CONFIG_DIST = get_euclidean_distance((CONFIG[0], CONFIG[1]), (TARGET_X, TARGET_Y)) + orientation_weight * get_angular_difference(CONFIG[2], TARGET_THETA)
    
        CONFIG_DISTS.append((CONFIG, CONFIG_DIST))
    
    CONFIG_DISTS.sort(key = lambda W_DIST: W_DIST[1])
    K_NEAREST_CONFIGS = CONFIG_DISTS[:k]
    
    return K_NEAREST_CONFIGS

"""
    function handle_drawing_freeBody_robot(figure, axes, config: tuple):
"""
def handle_drawing_freeBody_robot(figure, axes, config: tuple, edge_color: str, fill_color: str):
    FIGURE = figure
    AXES = axes
    
    x, y, theta = config
    
    ROBOT_RECTANGLE = PTCHS.Rectangle((x, y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(theta), color = fill_color, edgecolor = edge_color, linewidth = 1.0, alpha = 0.50)
    
    AXES.add_patch(ROBOT_RECTANGLE)
    
"""
    function visualize_scene_freeBody_robot():
"""
def visualize_scene_freeBody_robot(configs: list, k_nearest_configs: list, target_config: tuple) -> None:
    FIGURE, AXES = PLT.subplots()
    
    for CONFIG in configs:
        handle_drawing_freeBody_robot(figure = FIGURE, axes = AXES, config = CONFIG, edge_color = '#000000', fill_color = '#000000')
    
    for CONFIG in k_nearest_configs:
        handle_drawing_freeBody_robot(figure = FIGURE, axes = AXES, config = CONFIG[0], edge_color = '#0000ff', fill_color = '#0000ff')
        
    handle_drawing_freeBody_robot(figure = FIGURE, axes = AXES, config = target_config, edge_color = '#00ff00', fill_color = '#00ff00')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Free-body robot configurations')
    
    PLT.show()

"""
    function load_sample_freeBody_configs(filename: str) -> list:
"""
def load_sample_freeBody_configs(filename: str) -> list:
    # print(f'\nload_sample_freeBody_configs({filename}) called...')
    
    CONFIGS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        
        for LINE in LINES:
            VALUES = LINE.strip().split()
            
            x, y, theta = VALUES[0], VALUES[1], VALUES[2]
            
            CONFIG = (float(x), float(y), float(theta))
            
            CONFIGS.append(CONFIG)
    
    # TIME.sleep(2)
    # print(f'\tSample free body configurations loaded from FILE <{filename}>.')
    
    return CONFIGS

"""
    function get_euclidean_distance(point, target_point) -> float:
"""
def get_euclidean_distance(point, target_point):
    EUCLIDEAN_DIST = NP.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
    
    return EUCLIDEAN_DIST

"""
    function get_k_nearest_arm_robot_configurations(configs, target_config) -> list:
"""
def get_k_nearest_arm_robot_configurations(configs, target_config, k: int):
    
    CONFIGS_DISTS = []
    
    TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR = get_arm_robot_joint_positions(target_config[0], target_config[1])
    
    for CONFIG in configs:
        END_EFFECTOR = CONFIG[4]
        EUCLIDEAN_DIST = get_euclidean_distance(END_EFFECTOR, TARGET_END_EFFECTOR)
        CONFIGS_DISTS.append((CONFIG, EUCLIDEAN_DIST))
    
    CONFIGS_DISTS.sort(key = lambda EUC_DIST: EUC_DIST[1])
    
    SORTED_END_EFFECTOR_POSITIONS = sorted(configs)
    
    K_NEAREST_CONFIGS = CONFIGS_DISTS[:k]
    
    return K_NEAREST_CONFIGS

"""
    function handle_drawing_arm_robot(config: tuple) -> None:
"""
def handle_drawing_arm_robot(figure, axes, config: tuple, joint_color: str, line_color: str) -> None:
    # print(f'\nhandle_drawing_arm_robot({config}) called...')
    
    FIGURE = figure
    AXES = axes
    
    THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = config
    
    AXES.plot([BASE[0], JOINT[0]], [BASE[1], JOINT[1]], marker = 'o', color = line_color, linewidth = 1.0)
    AXES.plot([JOINT[0], END_EFFECTOR[0]], [JOINT[1], END_EFFECTOR[1]], marker = 'o', color = line_color, linewidth = 1.0)
    
    AXES.plot(BASE[0], BASE[1], marker = 'o', color = joint_color, label = 'Base')
    AXES.plot(JOINT[0], JOINT[1], marker = 'o', color = joint_color, label = 'Joint')
    AXES.plot(END_EFFECTOR[0], END_EFFECTOR[1], marker = 'o', color = joint_color, label = 'End-effector')

"""
    function visualize_scene_arm_robot(configs: list) -> None:
"""
def visualize_scene_arm_robot(configs: list, k_nearest_configs: list, target_config) -> None:
    # print(f'\nvisualize_scene_arm_robot({configs}) called...')
    
    FIGURE, AXES = PLT.subplots()
    
    for CONFIG in configs:
        handle_drawing_arm_robot(figure = FIGURE, axes = AXES, config = CONFIG, line_color = '#000000', joint_color = '#000000')
        
    for CONFIG in k_nearest_configs:
        handle_drawing_arm_robot(figure = FIGURE, axes = AXES, config = CONFIG[0], line_color = '#00ffff', joint_color = '#00ffff')

    TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = target_config[0], theta_2 = target_config[1])
    TARGET_CONFIG = (target_config[0], target_config[1], TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR)
    
    handle_drawing_arm_robot(figure = FIGURE, axes = AXES, config = TARGET_CONFIG, line_color = '#00ff00', joint_color = '#00ff00')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Arm robot configurations')
    
    # TODO: add legend to display/plot.
    
    PLT.show()

"""
    function get_arm_robot_joint_positions(theta_1, theta_2) -> list:
"""
def get_arm_robot_joint_positions(theta_1: float, theta_2: float) -> list:
    # print(f'\nget_arm_robot_joint_positions({theta_1}, {theta_2}) called...')
    
    BASE = (0, 0)
    
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    
    return (BASE, JOINT, END_EFFECTOR)

"""
    function load_sample_arm_robot_configs() -> list:
"""
def load_sample_arm_robot_configs(filename: str) -> list:
    # print(f'\nload_sample_arm_robot_configs() called...')
    
    CONFIGS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        for LINE in LINES:
            VALUES = LINE.strip().split()
            THETA_1, THETA_2 = VALUES
            THETA_1, THETA_2 = float(THETA_1), float(THETA_2)
            BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = THETA_1, theta_2 = THETA_2)

            CONFIGS.append((THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR))
    
    return CONFIGS

"""
    function parse_arguments()
    - function to parse command-line arguments.
    - return arguments object as ARGS
"""
def parse_arguments():
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--target', type = float, nargs = '+', required = True, help = 'Target pose of robot. (N = 2 for arm, N = 3 for freeBody).')
    
    PARSER.add_argument('-k', type = int, required = True, default = 3, help = 'Number of nearest neighbors to consider.')
    
    PARSER.add_argument('--configs', type = str, required = True, help = 'Filename containg list of arm/freeBody configurations.')
    
    return PARSER.parse_args()

"""
    function main():
    - Main function to run the program.
"""
def main():
    print('\n2. Nearest neighbors with linear search approach\n')
    
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        CONFIGS = load_sample_arm_robot_configs(filename = ARGS.configs)
        K_NEAREST_CONFIGS = get_k_nearest_arm_robot_configurations(configs = CONFIGS, target_config = ARGS.target, k = ARGS.k)
        visualize_scene_arm_robot(configs = CONFIGS, k_nearest_configs = K_NEAREST_CONFIGS, target_config = ARGS.target)
    elif ARGS.robot == 'freeBody':
        CONFIGS = load_sample_freeBody_configs(filename = ARGS.configs)
        K_NEAREST_CONFIGS = get_k_nearest_freeBody_robot_configurations(configs = CONFIGS, target_config = ARGS.target, orientation_weight = 0.25, k = ARGS.k)
        visualize_scene_freeBody_robot(configs = CONFIGS, k_nearest_configs = K_NEAREST_CONFIGS, target_config = ARGS.target)

if __name__ == '__main__':
    main()