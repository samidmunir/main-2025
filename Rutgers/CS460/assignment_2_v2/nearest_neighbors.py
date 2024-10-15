"""
    2. Nearest neighbors with linear search approach: nearest_neighbors.py
"""

import argparse
import math as MATH
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

"""
    CONSTANTS (from component_1.py)
"""
from component_1 import ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH

"""
    function euclidean_distance():
    - so this function takes in two points (x1, y1) & (x2, y2) as input, and returns the Euclidean distance between them.
    - these points represent the end-effector (x, y) coordinates and we use this attribute to perform the linear-search nearest-neighbors approach.
"""
def euclidean_distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    
    return MATH.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

"""
    function calculate_arm_robot_end_effector_positions():
    - this function calculates the positions of all the end-effectors in the configs list based on their given two angles (theta_0 and theta_1).
    - it returns this list as well as the positions of the target_end_effector.
    - this is crucial because our environment is not unit of radians, but a unit of x, y coordinates. Using the angles in the config files, and the target angles, we can compute the positions of the end-effectors and implement nearest-neighbors linear-search on them.
"""
def calculate_arm_robot_end_effector_positions(configs: list, target_config):
    TARGET_END_EFFECTOR_POSITIONS = get_arm_position(theta_0 = target_config[0], theta_1 = target_config[1])[2]
    END_EFFECTOR_POSITIONS = []
    for CONFIG in configs:
        end_effector_position_x, end_effector_position_y = get_arm_position(theta_0 = CONFIG[0], theta_1 = CONFIG[1])[2]
        END_EFFECTOR_POSITIONS.append((CONFIG[0], CONFIG[1], end_effector_position_x, end_effector_position_y))
    
    return END_EFFECTOR_POSITIONS, TARGET_END_EFFECTOR_POSITIONS

def find_k_nearest_configs(configs_distances: list, k: int) -> list:
    SORTED_CONFIGS_DISTS = sorted(configs_distances, key = lambda x: x[2])
    
    K_CLOSEST_CONFIGS = SORTED_CONFIGS_DISTS[:k]
    
    return K_CLOSEST_CONFIGS

"""
    function find_k_nearest_end_effector_positions():
    - this function is essential in the nearest-neighbors with linear search approach.
    - given a list of end_effector_positions (which are computed using the calculate_arm_robot_end_effector_positions() function), the target_end_effector_position, and the value of k, it returns a list of the k closest end_effector_positions that are closest to the target end-effector position.
"""
def find_k_nearest_end_effector_positions(end_effector_positions: list, target_end_effector_position, k: int) -> list:
    SORTED_END_EFFECTOR_POSITIONS = sorted(end_effector_positions, key = lambda x: euclidean_distance((x[2], x[3]), target_end_effector_position))
    
    K_CLOSEST_END_EFFECTOR_POSITIONS = SORTED_END_EFFECTOR_POSITIONS[:k]
    
    return K_CLOSEST_END_EFFECTOR_POSITIONS

"""
    function load_arm_robot_configs():
    - this functions reads a list of arm-robot configurations from a .txt file and returns a list of tuples in the form (theta_1, theta_2).
"""
def load_arm_robot_configs(filename: str) -> list:
    CONFIGS = []
    with open(filename, 'r') as FILE:
        for LINE in FILE:
            theta_1, theta_2 = map(float, LINE.strip().split())
            CONFIGS.append((theta_1, theta_2))
    
    return CONFIGS

"""
    function get_arm_position():
    - this function is so important as it calculates the position of the base, joint_1, and end-effector of the 2-joint arm robot, given the angles of the two joints, theta_0 and theta_1.
"""
def get_arm_position(theta_0, theta_1):
    # Joint 1 (base)
    x1, y1, = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_0), ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_0)
    
    # Joint 2 (end effector)
    x2 = x1 + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_0 + theta_1)
    y2 = y1 + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_0 + theta_1)
    
    # Return positions of the base, joint 1, and joint 2.
    return (0, 0), (x1, y1), (x2, y2)

"""
    function visualize_arm_robot():
    - this function handles drawing of 2-joint arm robot on the passed in matplotlib AXES.
    - draws the base joint, first joint, and end-effector.
    - draws the two links in between the two joints.
"""
def visualize_arm_robot(FIGURE, AXES, base, joint1, end_effector, joint_color, line_color):
    # Line from base_joint to joint1
    AXES.plot([base[0], joint1[0]], [base[1], joint1[1]], marker = 'o', color = f'{line_color}', label = 'First Arm Link')
    # Line from joint1 to end_effector
    AXES.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], marker = 'o', color = f'{line_color}', label = 'Second Arm Link')

    # Mark the base joint, joint1, and end-effector
    AXES.plot(base[0], base[1], marker = 'o', color = f'{joint_color}', label = 'Base Joint')
    AXES.plot(joint1[0], joint1[1], marker = 'o', color = f'{joint_color}', label = 'Joint 1')
    AXES.plot(end_effector[0], end_effector[1], marker = 'o', color = f'{joint_color}', label = 'End Effector')

"""
    function visualize_arm_robot_scene()
    - takes as input the CONFIGS, K_CLOSEST_CONFIGS, TARGET_CONFIG.
    - visualizes the CONFIGS, K_CLOSEST_CONFIGS, and TARGET_CONFIG.
"""
def visualize_arm_robot_scene(CONFIGS, K_CLOSEST_CONFIGS, TARGET_CONFIG):
    FIGURE, AXES = PLT.subplots()    
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
    # Visualize the configs.
    for CONFIG in CONFIGS:
        base, joint1, end_effector = get_arm_position(theta_0 = CONFIG[0], theta_1 = CONFIG[1])
        visualize_arm_robot(FIGURE, AXES, base, joint1, end_effector, '#000000', '#000000')
    
    for CONFIG in K_CLOSEST_CONFIGS:
        base, joint1, end_effector = get_arm_position(theta_0 = CONFIG[0], theta_1 = CONFIG[1])
        visualize_arm_robot(FIGURE, AXES, base, joint1, end_effector, '#0000ff', '#0000ff')
        
    target_base, target_joint1, target_end_effector = get_arm_position(theta_0 = TARGET_CONFIG[0], theta_1 = TARGET_CONFIG[1])
    visualize_arm_robot(FIGURE, AXES, target_base, target_joint1, target_end_effector, '#00ff00', '#00ff00')
    
    AXES.set_title('2-Joint Arm Robot k Nearest Configurations')
    # AXES.legend()
    
    PLT.show()

#TODO: Implement calculate_free_body_robot_euclidean_difference()
def calculate_free_body_robot_euclidean_difference():
    pass

# TODO: Implement load_free_body_robot_configs()
def load_free_body_robot_configs(filename: str) -> list:
    pass

# TODO: Implement visualize_free_body_robot_scene()
def visualize_free_body_robot_scene():
    pass

"""
    function parse_arguments()
    - parses command-line arguments and returns the parsed argumnets as an object.
"""
def parse_arguments():
    PARSER = argparse.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm or freeBody).')
    PARSER.add_argument('--target', type = float, nargs = '+', required = True, help = 'Target configuration for the robot.')
    PARSER.add_argument('-k', type = int, default = 3, help = 'Number of nearest configuratoins to find.')
    PARSER.add_argument('--configs', type = str, required = True, help = 'File containg the robot configurations.')
    
    return PARSER.parse_args()

"""
    function main()
    - handles main administration flow of the program.
    - parses command-line arguments.
"""
def main():
    ARGS = parse_arguments()
    
    CONFIGS = None
    
    if ARGS.robot == 'arm':
        CONFIGS = load_arm_robot_configs(filename = ARGS.configs)
        TARGET_THETA_1, TARGET_THETA_2 = ARGS.target
        number_of_nearest_neighbors = ARGS.k
        END_EFFECTOR_POSITIONS, TARGET_END_EFFECTOR_POSITIONS = calculate_arm_robot_end_effector_positions(configs = CONFIGS, target_config = (TARGET_THETA_1, TARGET_THETA_2))
        K_CLOSEST_END_EFFECTOR_POSITIONS = find_k_nearest_end_effector_positions(end_effector_positions = END_EFFECTOR_POSITIONS, target_end_effector_position = TARGET_END_EFFECTOR_POSITIONS, k = number_of_nearest_neighbors)
        visualize_arm_robot_scene(CONFIGS = CONFIGS, K_CLOSEST_CONFIGS = K_CLOSEST_END_EFFECTOR_POSITIONS, TARGET_CONFIG = (TARGET_THETA_1, TARGET_THETA_2))
    elif ARGS.robot == 'freeBody':
        pass

if __name__ == '__main__':
    main()