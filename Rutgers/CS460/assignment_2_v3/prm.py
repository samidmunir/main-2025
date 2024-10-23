# IMPORTS
import argparse as ARGPRS
import time as TIME
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
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
    get_arm_robot_joint_positions,
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH
)

from collision_checking import (
    JOINT_RADIUS,
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT,
    is_colliding,
    get_polygon_corners,
    is_colliding_link,
    point_in_circle
)

# CONSTANTS
K_NEIGHBORS = 6
NUMBER_OF_SAMPLES = 500

"""
    function generate_arm_robot_sample_configurations(number_of_samples: int) -> list:
"""
def generate_arm_robot_sample_configurations(number_of_samples: int) -> list:
    SAMPLES = []
    for _ in range(number_of_samples):
        theta_1 = RANDOM.uniform(0, 2 * NP.pi)
        theta_2 = RANDOM.uniform(0, 2 * NP.pi)
        BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = theta_1, theta_2 = theta_2)
        SAMPLE = (theta_1, theta_2, BASE, JOINT, END_EFFECTOR)
        
        SAMPLES.append(SAMPLE)
    
    return SAMPLES

"""
    function is_edge_collision_free(start, end, obstacles) -> bool:
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
    function generate_freeBody_robot_sample_configurations(number_of_samples: int) -> list
"""
def generate_freeBody_robot_sample_configurations(number_of_samples: int) -> list:
    SAMPLES = []
    for _ in range(number_of_samples):
        x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
        y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
        theta = RANDOM.uniform(0, 2 * NP.pi)
        
        SAMPLES.append((x, y, theta))
    
    return SAMPLES

"""
    function build_prm(robot_type: str, samples: list, environment: list) -> dict:
"""
def build_prm(robot_type: str, samples: list, environment: list) -> dict:
    PRM = {SAMPLE: [] for SAMPLE in samples}
    
    NEIGHBORS = []
    
    if robot_type == 'arm':
        for SAMPLE in samples:
            NEIGHBORS = get_k_nearest_arm_robot_configurations(configs = samples, target_config = SAMPLE, k = K_NEIGHBORS)
            
            for NEIGHBOR in NEIGHBORS:
                
                for OBSTACLE in environment:
                    x, y, width, height, theta = OBSTACLE
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
                    else:
                        pass
                        
    return PRM

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
    function parse_arguments()
    - function to parse command-line arguments.
    - return arguments object as ARGS
"""
def parse_arguments():
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--start', type = float, nargs = '+', help = 'Start configuration of robot (arm or freeBody).')
    
    PARSER.add_argument('--goal', type = float, nargs = '+', help = 'Start configuration of robot (arm or freeBody).')
    
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg environment.')
    
    return PARSER.parse_args()

"""
    function main():
    - Main function to run the program.
"""
def main():
    print('\n3. Collision checking\n')
    
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(filename = ARGS.map)
    
    if ARGS.robot == 'arm':
        SAMPLES = generate_arm_robot_sample_configurations(number_of_samples = NUMBER_OF_SAMPLES)
        build_prm(robot_type = 'arm', samples = SAMPLES, environment = ENVIRONMENT)
    elif ARGS.robot == 'freeBody':
        SAMPLES = generate_freeBody_robot_sample_configurations(number_of_samples = NUMBER_OF_SAMPLES)
        PRM = build_prm(robot_type = 'freeBody', samples = SAMPLES, environment = ENVIRONMENT)
        visualize_freeBody_PRM(prm = PRM, environment = ENVIRONMENT)

if __name__ == '__main__':
    main()