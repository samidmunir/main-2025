import argparse
import math as MATH
import random as RANDOM
import time as TIME
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS

"""
    CONSTANTS (from component_1.py)
"""
from component_1 import (
        ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, 
        FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, 
        ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH
    )

"""
    function project():
"""
def project(corners, axis):
    PROJECTIONS = NP.dot(corners, axis)
    
    return NP.min(PROJECTIONS), NP.max(PROJECTIONS)

"""
    function get_axes():
"""
def get_axes(corners):
    EDGES = NP.diff(NP.vstack([corners, corners[0]]), axis = 0)
    
    return NP.array([[-edge[1], edge[0]] for edge in EDGES]) / NP.linalg.norm(EDGES, axis = 1, keepdims = True)

"""
    function is_colliding():
"""
def is_colliding(robot_corners, obstacle_corners):
    AXES = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])
    
    for AXIS in AXES:
        min_1, max_1 = project(robot_corners, AXIS)
        min_2, max_2 = project(obstacle_corners, AXIS)
        
        if max_1 < min_2 or max_2 < min_1:
            return False
    
    return True

"""
    function visualize_scene_free_body_robot():
"""
def visualize_scene_free_body_robot(environment, config, iteration):
    FIGURE, AXES = PLT.subplots()
    
    ROBOT_CORNERS = get_polygon_corners(config[:2], config[2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
    
    for OBSTACLE in environment:
        x, y, width, height, angle = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners((x, y), width, height, angle)
        
        COLLIDING = is_colliding(ROBOT_CORNERS, OBSTACLE_CORNERS)
        
        COLOR = '#ff0000' if COLLIDING else '#000000'
        OBSTACLE_PATCH = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = COLOR, fill = False)
        AXES.add_patch(OBSTACLE_PATCH)
    
    ROBOT_PATCH = PTCHS.Polygon(ROBOT_CORNERS, closed = True, edgecolor = '#0000ff', fill = True, alpha = 0.5)
    AXES.add_patch(ROBOT_PATCH)
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    PLT.title(f'Free Body Robot Collision Test (Iteration #{iteration})')
    
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

########################################################################

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
    function visualize_arm_robot():
"""
def visualize_arm_robot(FIGURE, AXES, base, joint_1, end_effector, joint_color, line_color):
    AXES.plot([base[0], joint_1[0]], [base[1], joint_1[1]], marker = 'o', color = f'{line_color}', label = 'Link 1')
    AXES.plot([joint_1[0], end_effector[0]], [joint_1[1], end_effector[1]], marker = 'o', color = f'{line_color}', label = 'Link 2')
    
    AXES.plot(base[0], base[1], marker = 'o', color = f'{joint_color}', label = 'Base')
    AXES.plot(joint_1[0], joint_1[1], marker = 'o', color = f'{joint_color}', label = 'Joint 1')
    AXES.plot(end_effector[0], end_effector[1], marker = 'o', color = f'{line_color}', label = 'End Effector')

"""
    function visualize_scene_arm_robot():
"""
def visualize_scene_arm_robot(environment, theta_1, theta_2, iteration):
    FIGURE, AXES = PLT.subplots()
    
    BASE = (0, 0)
    (joint_1_x, joint_1_y), (end_effector_x, end_effector_y) = get_end_effector_position(theta_1, theta_2)
    JOINT_1 = (joint_1_x, joint_1_y)
    END_EFFECTOR = (end_effector_x, end_effector_y)
    
    for OBSTACLE in environment:
        x, y, width, height, angle = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners((x, y), width, height, angle)
        
        COLLIDING = (
            is_colliding_link(BASE, JOINT_1, OBSTACLE_CORNERS) or is_colliding_link(JOINT_1, END_EFFECTOR, OBSTACLE_CORNERS)
        )
        
        COLOR = '#ff0000' if COLLIDING else '#000000'
        OBSTACLE_POLYGON = PTCHS.Polygon(OBSTACLE_CORNERS, color = COLOR, edgecolor = COLOR, fill = True, closed = True)
        AXES.add_patch(OBSTACLE_POLYGON)
        
    visualize_arm_robot(FIGURE, AXES, BASE, JOINT_1, END_EFFECTOR, joint_color = '#000000', line_color = '#000000')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    PLT.title(f'ARM Robot Collision Test (Iteration #{iteration})')
    
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

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
            THETA_1 = RANDOM.uniform(0, 2 * MATH.pi)
            THETA_2 = RANDOM.uniform(0, 2 * MATH.pi)
            visualize_scene_arm_robot(ENVIRONMENT, THETA_1, THETA_2, (i + 1))
            TIME.sleep(1)
    elif ARGS.robot == 'freeBody':
        for i in range(10):
            CONFIG = (
                RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
                RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
                RANDOM.uniform(0, 2 * NP.pi)
            )
            visualize_scene_free_body_robot(ENVIRONMENT, CONFIG, (i + 1))
            TIME.sleep(1)

if __name__ == '__main__':
    main()