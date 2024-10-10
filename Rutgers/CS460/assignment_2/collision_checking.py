import argparse
import json
import random
import time
import matplotlib.pyplot as PLT
import numpy as NP

# Function to check if two rectangles (obstacles and robot) collide.
def is_collision(obstacle, robot):
    # Extract center and dimensions of both the obstacle and the robot.
    x1, y1 = obstacle['center']
    w1, h1 = obstacle['width'], obstacle['height']
    
    x2, y2 = robot['center']
    w2, h2 = robot['width'], robot['height']
    
    # Check for collision using axis-aligned bounding box (AABB) mehthod.
    if (abs(x1 - x2) * 2 < (w1 + w2)) and (abs(y1 - y2) * 2 < (h1 + h2)):
        return True
    return False