# IMPORTS
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

# CONSTANTS
JOINT_RADIUS = 0.1
FREE_BODY_ROBOT_WIDTH = 0.5
FREE_BODY_ROBOT_HEIGHT = 0.3

"""
    function handle_drawing_freeBody_robot(figure, axes, x, y, theta, fill_color, line_color):
"""
def handle_drawing_freeBody_robot(FIGURE, AXES, robot_corners, fill_color, line_color):
    ROBOT_RECTANGLE = PTCHS.Polygon(robot_corners, closed = True, edgecolor = line_color, color = fill_color, fill = True)
    
    AXES.add_patch(ROBOT_RECTANGLE)

"""
    function project(corners, axis):
"""
def project(corners, axis):
    """Project corners onto the given axis."""
    projections = NP.dot(corners, axis)
    return NP.min(projections), NP.max(projections)

"""
    function get_axes(corners):
"""
def get_axes(corners):
    """Compute axes perpendicular to the edges of the polygon."""
    edges = NP.diff(NP.vstack([corners, corners[0]]), axis=0)
    return NP.array([[-edge[1], edge[0]] for edge in edges]) / NP.linalg.norm(edges, axis=1, keepdims=True)

"""
    function is_colliding(robot_corners, obstacle_corners):
"""
def is_colliding(robot_corners, obstacle_corners):
    """Check collision between two polygons using SAT."""
    axes = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])

    for axis in axes:
        min1, max1 = project(robot_corners, axis)
        min2, max2 = project(obstacle_corners, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

"""
    function visualize_scene_freeBody_robot(environment: list, config: tuple, iteration_num: int):
"""
def visualize_scene_freeBody_robot(environment: list, config: tuple, iteration_num: int):
    FIGURE, AXES = PLT.subplots()
    
    ROBOT_CORNERS = get_polygon_corners(config[:2], config[2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        
        COLLIDING = is_colliding(ROBOT_CORNERS, OBSTACLE_CORNERS)
        
        COLOR = '#ff0000' if COLLIDING else '#000000'
        
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = COLOR, color = COLOR, fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
        
    handle_drawing_freeBody_robot(FIGURE, AXES, ROBOT_CORNERS, '#0000ff', '#0000ff')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title(f'FreeBody robot collision test (Iteration # {iteration_num})')
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

"""
    function point_in_circle(point, circle_center, radius):
"""
def point_in_circle(point, circle_center, radius):
    DISTANCE = NP.sqrt((point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2)
    
    return DISTANCE <= radius

"""
    function is_line_intersecting(p1: tuple, p2: tuple, q1: tuple, q2: tuple)
"""
def is_line_intersecting(p1, p2, q1, q2):
    """Check if two line segments (p1-p2 and q1-q2) intersect."""
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        return 0 if val == 0 else (1 if val > 0 else -1)

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    return (o1 != o2) and (o3 != o4)

"""
    function is_colliding_link(link_start: tuple, link_end: tuple, obstacle_corners):
"""
def is_colliding_link(link_start, link_end, obstacle_corners):
    """Check if a robot link intersects with any edge of the obstacle."""
    for i in range(len(obstacle_corners)):
        corner1 = obstacle_corners[i]
        corner2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        if is_line_intersecting(link_start, link_end, corner1, corner2):
            return True
    return False

"""
    function get_polygon_corners(center: tuple, width: float, height: float, theta: float):
"""
def get_polygon_corners(center, theta, width, height):
    """Calculate the world coordinates of the rectangle's corners."""
    w, h = width / 2, height / 2
    corners = NP.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    cos_theta, sin_theta = NP.cos(theta), NP.sin(theta)
    rotation_matrix = NP.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_corners = corners @ rotation_matrix.T
    return rotated_corners + NP.array(center)

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
    function handle_drawing_arm_robot(figure, axes, base, joint, end_effector, joint_color, line_color):
"""
def handle_drawing_arm_robot(FIGURE, AXES, base, joint1, end_effector, joint_color, line_color):
    """Draw the arm robot with markers at joints and links."""
    # Line from base to joint1
    AXES.plot([base[0], joint1[0]], [base[1], joint1[1]], marker='o', color=f'{line_color}', label='First Arm Link')
    # Line from joint1 to end-effector
    AXES.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], marker='o', color=f'{line_color}', label='Second Arm Link')

    # Mark the base, joint1, and end-effector
    AXES.plot(base[0], base[1], marker='o', ms=1.5, color='#000000', label='Base Joint')
    AXES.plot(joint1[0], joint1[1], marker='o', ms=1.5, color=f'{joint_color}', label='Joint 1')
    AXES.plot(end_effector[0], end_effector[1], marker='o', ms=1.5, color=f'{joint_color}', label='End Effector')

"""
    function visualize_scene_arm_robot(environment: list, config: tuple, iteration_num: int) -> None:
"""
def visualize_scene_arm_robot(environment: list, config: tuple, iteration_num: int) -> None:
    FIGURE, AXES = PLT.subplots()
    
    BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = config[0], theta_2 = config[1])
    
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
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title(f'Arm robot collision test (Iteration # {iteration_num})')
    PLT.show(block = False)
    PLT.pause(1)
    PLT.close(FIGURE)

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
            visualize_scene_arm_robot(environment = ENVIRONMENT, config = RAND_CONFIG, iteration_num = (i + 1))
            # TIME.sleep(1)
    elif ARGS.robot == 'freeBody':
        for i in range(10):
            x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
            y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
            theta = RANDOM.uniform(0.0, 2 * NP.pi)
            RAND_CONFIG = (x, y, theta)
            visualize_scene_freeBody_robot(environment = ENVIRONMENT, config = RAND_CONFIG, iteration_num = (i + 1))
            # TIME.sleep(1)

if __name__ == '__main__':
    main()