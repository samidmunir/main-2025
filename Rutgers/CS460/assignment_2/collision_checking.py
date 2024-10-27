import argparse as ARGPRS
import matplotlib.pylab as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM

from utils import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    JOINT_RADIUS,
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT,
    is_colliding,
    point_in_circle,
    get_polygon_corners,
    is_colliding_link,
    get_arm_robot_forward_kinematics
)

from gen_env import (
    scene_from_file
)
    
def handle_drawing_free_body_robot(FIGURE, AXES, robot_corners, fill_color, line_color):
    ROBOT_RECTANGLE = PTCHS.Polygon(robot_corners, closed = True, edgecolor = line_color, color = fill_color, fill = True)
    
    AXES.add_patch(ROBOT_RECTANGLE)

def handle_drawing_arm_robot(FIGURE, AXES, base, joint1, end_effector, joint_color, line_color):
    # Line from base to joint1
    AXES.plot([base[0], joint1[0]], [base[1], joint1[1]], marker='o', color=f'{line_color}', label='First Arm Link')
    # Line from joint1 to end-effector
    AXES.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], marker='o', color=f'{line_color}', label='Second Arm Link')

    # Mark the base, joint1, and end-effector
    AXES.plot(base[0], base[1], marker='o', ms=1.5, color='#000000', label='Base Joint')
    AXES.plot(joint1[0], joint1[1], marker='o', ms=1.5, color=f'{joint_color}', label='Joint 1')
    AXES.plot(end_effector[0], end_effector[1], marker='o', ms=1.5, color=f'{joint_color}', label='End Effector')

def visualize_scene_free_body_robot(environment: list, configuration: tuple, iteration: int) -> None:
    FIGURE, AXES = PLT.subplots()
    
    ROBOT_CORNERS = get_polygon_corners(configuration[:2], configuration[2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        
        COLLIDING = is_colliding(ROBOT_CORNERS, OBSTACLE_CORNERS)
        
        COLOR = '#ff0000' if COLLIDING else '#000000'
        
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = COLOR, color = COLOR, fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
        
    handle_drawing_free_body_robot(FIGURE, AXES, ROBOT_CORNERS, '#0000ff', '#0000ff')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title(f'FreeBody robot collision test (Iteration # {iteration})')
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

def visualize_scene_arm_robot(environment: list, configuration: tuple, iteration: int) -> None:
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (configuration[0], configuration[1]))
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        
        COLLIDING = (
            is_colliding_link(BASE, JOINT, OBSTACLE_CORNERS) or (is_colliding_link(JOINT, END_EFFECTOR, OBSTACLE_CORNERS)) or point_in_circle(BASE, (x, y), JOINT_RADIUS) or point_in_circle(JOINT, (x, y), JOINT_RADIUS) or point_in_circle(END_EFFECTOR, (x, y), JOINT_RADIUS)
        )
        
        COLOR = '#ff0000' if COLLIDING else '#000000'
        
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = COLOR, color = COLOR, fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
        
    handle_drawing_arm_robot(FIGURE, AXES, BASE, JOINT, END_EFFECTOR, '#00ff00', line_color = '#0000ff')
    
    PLT.title(f'Arm robot collision test (Iteration # {iteration})')
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

def parse_arguments():
    ARG_PARSER = ARGPRS.ArgumentParser(description = 'Collision checking.')
    
    ARG_PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody)')
    ARG_PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg environment.')
    
    return ARG_PARSER.parse_args()

def main():
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(ARGS.map)
    
    if ARGS.robot == 'arm':
        for i in range(10):
            theta_1 = RANDOM.uniform(0.01746, 2 * NP.pi)
            theta_2 = RANDOM.uniform(0.01746, 2 * NP.pi)
            RANDOM_CONFIGURATION = (theta_1, theta_2)
            visualize_scene_arm_robot(environment = ENVIRONMENT, configuration = RANDOM_CONFIGURATION, iteration = (i + 1))
    elif ARGS.robot == 'freeBody':
        for i in range(10):
            x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
            y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
            theta = RANDOM.uniform(0.01746, 2 * NP.pi)
            RANDOM_CONFIGURATION = (x, y, theta)
            visualize_scene_free_body_robot(environment = ENVIRONMENT, configuration = RANDOM_CONFIGURATION, iteration = (i + 1))

if __name__ == '__main__':
    main()