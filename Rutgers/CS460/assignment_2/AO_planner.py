import argparse
import json
import random
import matplotlib.pyplot as PLT
import numpy as NP

# Function to check if a configuration collides with any obstacles.
def is_collision_free(config, environment):
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        
        if (abs(x - config[0]) * 2 < width) and (abs(y - config[1]) * 2 < height):
            return False
    return True