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

"""
    function scene_to_file(environment: list, filename: str):
"""
def scene_to_file(environment: list, filename: str):
    print(f'\nscene_to_file({filename}) called...')
    
    with open(filename, 'w') as FILE:
        for OBSTCALE in environment:
            FILE.write(f'{OBSTCALE[0]}, {OBSTCALE[1]}, {OBSTCALE[2]}, {OBSTCALE[3]}, {OBSTCALE[4]}\n')
    
    time.sleep(2)
    print(f'\tEnvironment saved to FILE <{filename}>.')
    
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
    
    time.sleep(2)
    print(f'\tEnvironment loaded from FILE <{filename}>.')
    
    return OBSTACLES

"""
    function visualize_scene(environment: list):
"""
def visualize_scene(environment: list):
    print(f'\nvisualize_scene() called...')
    
    FIGURE, AXES = PLT.subplots()
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        
        OBSTACLE_RECTAGNGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(theta), color = '#ff0000', edgecolor = '#ff0000')