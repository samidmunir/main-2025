"""
    1. Generating environments: component_1.py
        - function generate_environment(number_of_obstacles: int) -> list:
        - function scene_to_file(environment: list, filename: str):
        - function scene_from_file(filename: str) -> list:
        - function visualize_scene(environment: list):
"""

# IMPORTS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
import time

# CONSTANTS

"""
    function generate_environment(number_of_obstacles: int) -> list:
"""
def generate_environment(number_of_obstacles: int) -> list:
    print(f'\ngenerate_environment({number_of_obstacles}) called...')
    
    OBSTACLES = []
    
    for i in range(number_of_obstacles):
        
        x = RANDOM.uniform(-10, 10)
        y = RANDOM.uniform(-10, 10)
        width = RANDOM.uniform(0.5, 2.0)
        height = RANDOM.uniform(0.5, 2.0)
        theta = RANDOM.uniform(0, (2 * NP.pi))
        
        OBSTACLE = (x, y, width, height, theta)
        
        OBSTACLES.append(OBSTACLE)
    
    time.sleep(2)
    print(f'\tenvironment generation successfully complete.')
    
    return OBSTACLES
