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

"""
    function get_euclidean_distance(point, target_point) -> float:
"""
def get_euclidean_distance(point, target_point):
    EUCLIDEAN_DIST = NP.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
    
    return EUCLIDEAN_DIST

"""
    function handle_drawing_arm_robot(config: tuple) -> None:
"""
def handle_drawing_arm_robot(figure, axes, config: tuple) -> None:
    print(f'\nhandle_drawing_arm_robot({config}) called...')
    
    FIGURE = figure
    AXES = axes
    
    THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = config
    
    AXES.plot([BASE[0], JOINT[0]], [BASE[1], JOINT[1]], marker = 'o', color = '#000000')
    AXES.plot([JOINT[0], END_EFFECTOR[0]], [JOINT[1], END_EFFECTOR[1]], marker = 'o', color = '#000000')
    
    AXES.plot(BASE[0], BASE[1], marker = 'o', color = '#000000', label = 'Base')
    AXES.plot(JOINT[0], JOINT[1], marker = 'o', color = '#ff0000', label = 'Joint')
    AXES.plot(END_EFFECTOR[0], END_EFFECTOR[1], marker = 'o', color = '#00ff00', label = 'End-effector')

"""
    function visualize_scene_arm_robot(configs: list) -> None:
"""
def visualize_scene_arm_robot(configs: list) -> None:
    print(f'\nvisualize_scene_arm_robot({configs}) called...')
    
    FIGURE, AXES = PLT.subplots()
    
    for CONFIG in configs:
        handle_drawing_arm_robot(figure = FIGURE, axes = AXES, config = CONFIG)
    
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
    print(f'\nget_arm_robot_joint_positions({theta_1}, {theta_2}) called...')
    
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
    print(f'\nload_sample_arm_robot_configs() called...')
    
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
        CONFIGS = load_sample_arm_robot_configs(ARGS.configs)
        visualize_scene_arm_robot(configs = CONFIGS)
    elif ARGS.robot == 'freeBody':
        print('\n*** NOT YET SUPPORTED ***\n')

if __name__ == '__main__':
    main()