import argparse
import math as MATH
import random as RANDOM
import time as TIME
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS

from component_1 import (
    ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, 
    ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH, 
    FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT
)

"""
    function get_polygon_corners():
    # TODO: Calculate the world coordinates of the rectangle's corners.
"""

"""
    function get_end_effector_position():
    # TODO: Calculate the (x, y) position of the end-effector using forward kinematics.
"""

"""
    function is_line_intersecting():
    # TODO: Check if two line segments (p1-p2 and q1-q2) intersect.
"""

"""
    function is_colliding_link():
    # TODO: Check if a robot link intersects with any edge of the obstacle.
"""

"""
    function visualize_scene_arm():
"""
def visualize_scene_arm():
    pass

"""
    function scene_from_file():
"""
def scene_from_file(filename: str) -> list:
    OBSTACLES = []
    
    with open(filename, 'r') as FILE:
        for LINE in FILE:
            x, y, width, height, angle = map(float, LINE.strip().split(','))
            OBSTACLES.append((x, y, width, height, angle))
    
    return OBSTACLES

"""
    function parse_arguments():
"""
def parse_arguments():
    PARSER = argparse.ArgumentParser(description = 'Collision Detection between Robot & Obstacles')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm or freeBody)')
    
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg map of environment/obstacles')
    
    return PARSER.parse_args()

"""
    function main():
"""
def main():
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(ARGS.map)
    
    if ARGS.robot == 'arm':
        for i in range(10):
            theta_1 = RANDOM.uniform(0, 2 * MATH.pi)
            theta_2 = RANDOM.uniform(0, 2 * MATH.pi)
            # TODO: call function visualize_scene_arm()
    elif ARGS.robot == 'freeBody':
        pass

if __name__ == '__main__':
    main()