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