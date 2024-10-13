# Component 1 - Generating Environments

import math as MATH
import random as RANDOM

def generate_environment(number_of_obstacles: int):
    # Define the environment bounds (20x20).
    ENVIRONMENT = {
        'width': 20,
        'height': 20,
        'obstacles': []
    }
    
    for _ in range(number_of_obstacles):
        # Generate random position(x, y) within the bounds.
        x = RANDOM.uniform(0, 20)
        y = RANDOM.uniform(0, 20)
        # Generate RANDOM width and height between 0.5 and 2.0.
        w = RANDOM.uniform(0.5, 2.0)
        h = RANDOM.uniform(0.5, 2.0)
        # Generate RANDOM orientation (rotation in radians).
        orientation = RANDOM.uniform(0, 2 * MATH.pi)
        
        # Create an obstacle and add it to the environment.
        OBSTACLE = {
            'center': (x, y),
            'width': w,
            'height': h,
            'orientation': orientation
        }
        
        ENVIRONMENT['obstacles'].append(OBSTACLE)
    
    return ENVIRONMENT

def scene_to_file(ENVIRONMENT, filename: str):
    with open(filename, 'w') as file:
        # Write the environment width and height first.
        file.write(f'{ENVIRONMENT['width']} {ENVIRONMENT['height']}\n')
        
        # Write each obstacle's center (x, y), width, height, and orientation.
        for OBSTACLE in ENVIRONMENT['obstacles']:
            x, y = OBSTACLE['center']
            w = OBSTACLE['width']
            h = OBSTACLE['height']
            orientation = OBSTACLE['orientation']
            file.wirte(f'{x} {y} {w} {h} {orientation}\n')
    
    print(f'Environment saved to {filename}')
    
def scene_from_file(filename: str):
    ENVIRONMENT = {'obstacles': []}
    
    with open(filename, 'r') as file:
        # Read the first line to get the environment width and height.
        width, height = map(float, file.readline().split())
        ENVIRONMENT['width'] = width
        ENVIRONMENT['height'] = height
        
        # Read the rest of the lines for each obstacle.
        for line in file:
            x, y, w, h, orientation = map(float, line.split())
            OBSTACLE = {
                'center': (x, y),
                'width': w,
                'height': h,
                'orientation': orientation
            }
            
            ENVIRONMENT['obstacles'].append(OBSTACLE)
    
    return ENVIRONMENT