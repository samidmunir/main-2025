"""
    1. Generating environments: component_1.py
        - function generate_environment(number_of_obstacles: int) -> list:
        - function scene_to_file(environment: list, filename: str) -> None:
        - function scene_from_file(filename: str) -> list:
        - function visualize_scene(environment: list) -> None:
"""

# IMPORTS
from utils import (
    get_polygon_corners,
)

import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
import time as TIME

# CONSTANTS
ENVIRONMENT_WIDTH_MIN = -10.0
ENVIRONMENT_WIDTH_MAX = 10.0
ENVIRONMENT_HEIGHT_MIN = -10.0
ENVIRONMENT_HEIGHT_MAX = 10.0

OBSTACLE_MIN_SIZE = 0.5
OBSTACLE_MAX_SIZE = 2.0

"""
    function generate_environment(number_of_obstacles: int) -> list:
    - this function will generate a randomized environment with the number of randomized obstacles = number_of_obstacles.
    - this function returns a list of random obstacles.
    - each obstacle is a tuple in the form:
        > (x, y, width, height, theta, corners)
            * corners are the 4 corners of the obstacle rectangle.
            * will use utils.get_polygon_corners() to compute corners() tuple.
"""
def generate_environment(number_of_obstacles: int) -> list:
    print(f'\ngenerate_environment({number_of_obstacles}) called...')
    
    ENVIRONMENT = []
    
    for _ in range(number_of_obstacles):
        x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
        y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
        width = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        height = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        theta = RANDOM.uniform(0, 2 * NP.pi)
        
        corners = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        c1, c2, c3, c4 = corners
        
        OBSTACLE = (x, y, width, height, theta, c1, c2, c3, c4)
        
        ENVIRONMENT.append(OBSTACLE)
        
    print(f'\tsuccessfully generated environment with {number_of_obstacles}.')
    
    return ENVIRONMENT

"""
    function scene_to_file(environment: list, filename: str) -> None:
    - this function takes the environment list and stores each obstacle in the file refered to by filename.
    - each line will contain the configuration tuple.
        > comma-separated values of obstacle configuration.
"""
def scene_to_file(environment: list, filename: str) -> None:
    print(f'\nscene_to_file({filename}) called...')
    
    with open(filename, 'w') as FILE:
        for OBSTACLE in environment:
            FILE.write(f'{OBSTACLE[0]}, {OBSTACLE[1]}, {OBSTACLE[2]}, {OBSTACLE[3]}, {OBSTACLE[4]}, {OBSTACLE[5]}, {OBSTACLE[6]}, {OBSTACLE[7]}, {OBSTACLE[8]}\n')

    print(f'\tenvironment saved to FILE <{filename}>.')
    
"""
    function scene_from_file(environment: list, filename: str) -> None:
    - this function loads in the environment as described in the file referred to by filename.
    - we expect each line to contain a tuple containing the obstacle's configuration.
    - we will store each obstacle configuration in the environment list and return it.
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
            c1 = float(VALUES[5])
            c2 = float(VALUES[6])
            c3 = float(VALUES[7])
            c4 = float(VALUES[8])
            
            OBSTACLE = (x, y, width, height, theta, c1, c2, c3, c4)
            
            OBSTACLES.append(OBSTACLE)
    
    TIME.sleep(2)
    print(f'\tenvironment loaded from FILE <{filename}>.')
    
    return OBSTACLES

"""
    function visualize_scene(environment: list):
"""
def visualize_scene(environment: list) -> None:
    print(f'\nvisualize_scene() called...')
    
    FIGURE, AXES = PLT.subplots()
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        
        OBSTACLE_RECTAGNGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(theta), color = '#ff0000', edgecolor = '#ff0000')
        
        AXES.add_patch(OBSTACLE_RECTAGNGLE)
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title(f'Randomly generated environment: {len(environment)} obstacles')
    
    PLT.show(block = False)