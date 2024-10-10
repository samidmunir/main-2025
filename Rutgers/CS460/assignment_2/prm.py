import argparse
import heapq
import json
import random
import time
import matplotlib.pyplot as PLT
import numpy as NP
from scipy.spatial import KDTree

# Check if the robot's configuration collides with any obstacles.
def is_collision_free(config, environment):
    # The robot is represented as a point (config).
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        
        # Check if the congig is inside the obstacle's bounding box.
        if (abs(x - config[0]) * 2 < width) and (abs(y - config[1]) * 2 < height):
            return False
        
        return True