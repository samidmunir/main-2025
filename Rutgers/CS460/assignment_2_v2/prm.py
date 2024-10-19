"""
    4. PRM: prm.py
"""

"""
    Python IMPORTS
"""
import argparse
import math as MATH
import random as RANDOM
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

"""
    CONSTANTS (from component_1.py)
"""
from component_1 import ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH, FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT
"""
    IMPORTS (from nearest_neighbors.py)
"""
"""
    IMPORTS (from collision_checking.py)
"""

######################################################################

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
def visualize_scene_arm_robot(obstacles: list, random_samples: list):
    FIGURE, AXES = PLT.subplots()
    
    PLT.title('Arm Robot Path Planning with PRM')
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
    PLT.show()


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
    function main():
"""
def main():
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        OBSTACLES = scene_from_file(ARGS.map)
        RANDOM_CONFIGS = generate_random_configs_arm_robot(num_samples = 5000)
        RANDOM_SAMPLES = get_random_arm_robot_samples(random_configs = RANDOM_CONFIGS)
    elif ARGS.robot == 'freeBody':
        pass

if __name__ == '__main__':
    main()