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
    function is_line_intersecting():
"""
def is_line_intersecting(p1, p2, q1, q2):
    def orientation(a, b, c):
        VALUE = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        
        return 0 if VALUE == 0 else (1 if VALUE > 0 else -1)
    
    O1 = orientation(p1, p2, q1)
    O2 = orientation(p1, p2, q2)
    O3 = orientation(q1, q2, p1)
    O4 = orientation(q1, q2, p2)
    
    return (O1 != O2) and (O3 != O4)
    
"""
    function is_colliding_link():
"""
def is_colliding_link(link_start, link_end, obstacle_corners):
    for i in range(len(obstacle_corners)):
        CORNER_1 = obstacle_corners[i]
        CORNER_2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        
        if is_line_intersecting(link_start, link_end, CORNER_1, CORNER_2):
            return True
    
    return False

"""
    function get_polygon_corners():
"""
def get_polygon_corners(center, width, height, angle):
    w, h = width / 2, height / 2
    CORNERS = NP.array(
        [
            [-w, -h],
            [w, -h],
            [w, h],
            [-w, h]
        ]
    )
    
    COS_THETA, SIN_THETA = NP.cos(angle), NP.sin(angle)
    ROTATION_MATRIX = NP.array(
        [
            [COS_THETA, -SIN_THETA],
            [SIN_THETA, COS_THETA]
        ]
    )
    
    ROTATED_CORNERS = CORNERS @ ROTATION_MATRIX.T
    
    return ROTATED_CORNERS + NP.array(center)

"""
    function get_end_effector_position():
"""
def get_end_effector_position(theta_1, theta_2):
    joint_1_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    joint_1_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    
    end_effector_x = joint_1_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    end_effector_y = joint_1_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    
    return (joint_1_x, joint_1_y), (end_effector_x, end_effector_y)

"""
    function visualize_scene_arm():
"""
def visualize_scene_arm(environment, theta_1, theta_2, iteration):
    FIGURE, AXES = PLT.subplots()
    
    BASE = (0, 0)
    (joint_1_x, joint_1_y), (end_effector_x, end_effector_y) = get_end_effector_position(theta_1, theta_2)
    JOINT_1 = (joint_1_x, joint_1_y)
    END_EFFECTOR = (end_effector_x, end_effector_y)
    
    for OBSTACLE in environment:
        x, y, width, height, angle = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners((x, y), width, height, angle)

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