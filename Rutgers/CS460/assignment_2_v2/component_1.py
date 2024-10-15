"""
    1. Generating Environments: component_1.py
        - function generate_environments()
        - function scene_to_file()
        - function scene_from_file()
        - function visualize_scene()
"""

import random as RANDOM
import math as MATH
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

"""
    CONSTANTS
"""
ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION = -10.0, 10.0 # [20.0, 20.0]
OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE = 0.5, 2.0
# OBSTACLE_SIZE_PADDING = 2.5
OBSTACLE_MIN_ORIENTATION, OBSTACLE_MAX_ORIENTATION = 0.0, 2.0 * NP.pi # in radians

ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5


"""
    function generate_environments(number_of_obstacles: int)
"""
def generate_environment(number_of_obstacles: int):
    OBSTACLES = []
    
    for _ in range(number_of_obstacles):
        x = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
        y = RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
        width = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        height = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        orientation = RANDOM.uniform(OBSTACLE_MIN_ORIENTATION, OBSTACLE_MAX_ORIENTATION) # in radians
        
        OBSTACLE = (x, y, width, height, orientation)
        
        OBSTACLES.append(OBSTACLE)
    
    return OBSTACLES

"""
    function scene_to_file(environment: list, filename: str)
"""
def scene_to_file(environment: list, filename: str):
    with open(filename, 'w') as FILE:
        for OBSTACLE in environment:
            x, y, width, height, orientation = OBSTACLE
            FILE.write(f'{x}, {y}, {width}, {height}, {orientation}\n')
            
"""
    function scene_from_file(filename: str) -> list
"""
def scene_from_file(filename: str) -> list:
    OBSTACLES = []
    
    with open(filename, 'r') as FILE:
        for LINE in FILE:
            x, y, width, height, orientation = map(float, LINE.strip().split(','))
            OBSTACLE = (x, y, width, height, orientation)
            OBSTACLES.append(OBSTACLE)
    
    return OBSTACLES

"""
    function visualize_scene(environment: list)
"""
def visualize_scene(environment: list):
    FIGURE, AXES = PLT.subplots()
    
    for OBSTACLE in environment:
        x, y, width, height, orientation = OBSTACLE
        OBSTACLE_RECTANGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(orientation), color = '#ff0000', edgecolor = '#ff0000', alpha = 0.5)
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    
    PLT.show()