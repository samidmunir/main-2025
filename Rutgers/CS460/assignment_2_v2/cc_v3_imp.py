import argparse as ARGPRS
import time as TIME
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
from component_1 import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    OBSTACLE_MIN_SIZE,
    OBSTACLE_MAX_SIZE
)

from nearest_neighbors import (
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH
)

# """
#     function point_in_circle(point, circle_center, radius):
# """
# def point_in_circle(point, circle_center, radius):
#     DISTANCE = NP.sqrt((point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2)
    
#     return DISTANCE <= radius

# """
#     function separating_axis_theorem(rectangle_1, rectangle_2):
# """
# def separating_axis_theorem(rectangle_1, rectangle_2):
#     tolerance = 1e-1
#     def get_axes(rectangle):
#         axes = []
#         for i in range(len(rectangle)):
#             # Calculate the edge between cosecutive corners.
#             edge = (rectangle[i][0] - rectangle[i - 1][0], rectangle[i][1] - rectangle[i - 1][1])
#             # Get the perpendicular axis (normal).
#             axis = (-edge[1], edge[0])
#             # Normalize the axis (to get unit vectors).
#             magnitude = NP.sqrt(axis[0] ** 2 + axis[1] ** 2)
#             axes.append((axis[0] / magnitude, axis[1] / magnitude))
            
#         return axes
    
#     def project(rectangle, axis):
#         return [NP.dot(corner, axis) for corner in rectangle]

#     for axis in get_axes(rectangle_1) + get_axes(rectangle_2):
#         projection_1 = project(rectangle_1, axis)
#         projection_2 = project(rectangle_2, axis)
#         # Check for overlap between the projections.
#         if max(projection_1) < min(projection_2) or max(projection_2) < min(projection_1):
#             return False # no collision detected
#     return True # collision detected

# """
#     function rotate_point(point, angle, origin = (0, 0))
# """
# def rotate_point(point, angle, origin = (0, 0)):
#     ox, oy = origin
#     px, py = point
    
#     cos_theta = NP.cos(angle)
#     sin_theta = NP.sin(angle)
    
#     qx = ox + cos_theta * (px - ox) - sin_theta * (py - oy)
#     qy = oy + sin_theta * (px - ox) + cos_theta * (py - oy)
    
#     return qx, qy

# """
#     function create_rectangle(x, y, width, height, angle):
# """
# def create_rectangle(x, y, width, height, angle):
#     CORNERS = [
#         (x - width / 2, y - height / 2),
#         (x + width / 2, y - height / 2),
#         (x + width / 2, y + height / 2),
#         (x - width / 2, y + height / 2)
#     ]
    
#     ROTATED_CORNERS = [
#         rotate_point(CORNER, angle, (x, y)) for CORNER in CORNERS
#     ]
    
#     return ROTATED_CORNERS

# """
#     function get_link_coordinates(base, legnth, angle):
# """
# def get_link_coordinates(base, length, angle):
#     end_x = base[0] + length * NP.cos(angle)
#     end_y = base[1] + length * NP.sin(angle)
    
#     return end_x, end_y

# """
#     function is_arm_robot_colliding(environment: list, config: tuple) -> bool:
# """
# def is_arm_robot_colliding(figure, axes, environment: list, obstacle, config: tuple) -> bool:
#     # # print(f'\nis_arm_robot_colliding({environment}, {config}) called...')
    
#     # FIGURE = figure
#     # AXES = axes
    
#     # joint_radius = 0.075
    
#     # THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = config
    
#     # # Calculate the coordinates of the joints and end-effector.
#     # JOINT = get_link_coordinates(BASE, 2.0, THETA_1)
#     # END_EFFECTOR = get_link_coordinates(JOINT, 1.5, THETA_1 + THETA_2)
    
#     # # Create rectangles representing the arm's links.
#     # arm_link_1 = create_rectangle((BASE[0] + JOINT[0]) / 2, (BASE[1] + JOINT[1]) / 2, width = 0.1, height = 2.0, angle = THETA_1)
    
#     # arm_link_2 = create_rectangle((JOINT[0] + END_EFFECTOR[0]) / 2, (JOINT[1] + END_EFFECTOR[1]) / 2, width = 0.1, height = 1.5, angle = THETA_1 + THETA_2)
    
#     # x, y, width, height, theta = obstacle
#     # OBSTACLE_RECTANGLE = create_rectangle(x, y, width, height, theta)
    
#     # if separating_axis_theorem(arm_link_1, OBSTACLE_RECTANGLE) or separating_axis_theorem(arm_link_2, OBSTACLE_RECTANGLE):
#     #     return True
    
#     # if point_in_circle(BASE, (x, y), joint_radius) or point_in_circle(JOINT, (x, y), joint_radius) or point_in_circle(END_EFFECTOR, (x, y), joint_radius):
#     #     return True
    
#     # return False
    
#     # Check for collisions with each obstacle.
#     # for OBSTACLE in environment:
#     #     x, y, width, height, theta = OBSTACLE
#     #     OBSTACLE_RECTANGLE = create_rectangle(x, y, width, height, theta)
        
#     #     # Check for collision for arm links.
#     #     if separating_axis_theorem(arm_link_1, OBSTACLE_RECTANGLE) or separating_axis_theorem(arm_link_2, OBSTACLE_RECTANGLE):
#     #         return True # collision detected with a link.
        
#     #     # Check collision for joints (as circles).
#     #     if point_in_circle(BASE, (x, y), joint_radius) or point_in_circle(JOINT, (x, y), joint_radius) or point_in_circle(END_EFFECTOR, (x, y), joint_radius):
#     #         return True
    
#     # return False
        

"""
    function get_arm_robot_joint_positions(theta_1, theta_2) -> list:
"""
def get_arm_robot_joint_positions(theta_1: float, theta_2: float) -> list:
    # print(f'\nget_arm_robot_joint_positions({theta_1}, {theta_2}) called...')
    
    BASE = (0, 0)
    
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    
    return (BASE, JOINT, END_EFFECTOR)

"""
    function handle_drawing_arm_robot(config: tuple) -> None:
"""
def handle_drawing_arm_robot(figure, axes, config: tuple, joint_color: str, line_color: str) -> None:
    # print(f'\nhandle_drawing_arm_robot({config}) called...')
    
    FIGURE = figure
    AXES = axes
    
    THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = config
    
    AXES.plot([BASE[0], JOINT[0]], [BASE[1], JOINT[1]], marker = 'o', color = line_color, linewidth = 1.0)
    AXES.plot([JOINT[0], END_EFFECTOR[0]], [JOINT[1], END_EFFECTOR[1]], marker = 'o', color = line_color, linewidth = 1.0)
    
    AXES.plot(BASE[0], BASE[1], marker = 'o', markersize = 2.5, color = joint_color, label = 'Base')
    AXES.plot(JOINT[0], JOINT[1], marker = 'o', markersize = 2.5, color = joint_color, label = 'Joint')
    AXES.plot(END_EFFECTOR[0], END_EFFECTOR[1], marker = 'o', markersize = 2.5, color = joint_color, label = 'End-effector')

"""
    function visualize_scene_arm_robot(environment: list, config: tuple):
"""
def visualize_scene_arm_robot(environment: list, config: tuple, iteration: int):
    print(f'\nvisualize_scene() called...')
    
    FIGURE, AXES = PLT.subplots()
    
    # for OBSTACLE in environment:
    #     x, y, width, height, theta = OBSTACLE
        
    #     COLLIDING = is_arm_robot_colliding()
        
    #     OBSTACLE_RECTAGNGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(theta), color = '#000000', edgecolor = '#000000', alpha = 0.5)
        
    #     AXES.add_patch(OBSTACLE_RECTAGNGLE)
    
    # TODO: generate new CONFIG tuple with (theta_1, theta_2, base, joint, end_effector)
    BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = config[0], theta_2 = config[1])
    CONFIG = (config[0], config[1], BASE, JOINT, END_EFFECTOR)
    handle_drawing_arm_robot(FIGURE, AXES, CONFIG, '#000000', '#000000')
    
    # COLLIDING = is_arm_robot_colliding(environment, CONFIG)
    # if COLLIDING:
    #     print('\n*** COLLISION ***')
    
    # OBSTACLE_COLOR = '#ff0000' if COLLIDING else '#000000'
    
    NUMBER_OF_COLLISIONS = 0        
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        
        if is_arm_robot_colliding(FIGURE, AXES, environment, OBSTACLE, CONFIG):
            OBSTACLE_COLOR = '#ff0000'
            NUMBER_OF_COLLISIONS += 1
        else:
            OBSTACLE_COLOR = '#000000'  # default color for non-colliding obstacles
        
        OBSTACLE_RECTAGNGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(theta), color = OBSTACLE_COLOR, edgecolor = OBSTACLE_COLOR, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTAGNGLE)
        
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title(f'Collision checking (arm robot)[{iteration}] - # of collisions: {NUMBER_OF_COLLISIONS}')
    
    PLT.show()
    
    # TIME.sleep(1)
    
    PLT.close()

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
    
    TIME.sleep(2)
    print(f'\tEnvironment loaded from FILE <{filename}>.')
    
    return OBSTACLES

"""
    function parse_arguments()
    - function to parse command-line arguments.
    - return arguments object as ARGS
"""
def parse_arguments():
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg environment.')
    
    return PARSER.parse_args()

"""
    function main():
    - Main function to run the program.
"""
def main():
    print('\n3. Collision checking\n')
    
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(ARGS.map)
    
    if ARGS.robot == 'arm':
        for i in range(10):
            theta_1 = RANDOM.uniform(0.0, 2 * NP.pi)
            theta_2 = RANDOM.uniform(0.0, 2 * NP.pi)
            RAND_CONFIG = (theta_1, theta_2)
            visualize_scene_arm_robot(environment = ENVIRONMENT, config = RAND_CONFIG, iteration = (i + 1))
            # TIME.sleep(1)
    elif ARGS.robot == 'freeBody':
        print('\n*** Not yet supported ***')

if __name__ == '__main__':
    main()