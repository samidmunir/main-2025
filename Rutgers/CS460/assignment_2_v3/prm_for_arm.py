# IMPORTS
import argparse as ARGPRS
import time as TIME
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM

from point_in_obstacle_test import point_in_rotated_rectangle

from component_1 import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX
)

from nearest_neighbors_v2 import (
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH
)

from collision_checking_v2 import *

# CONSTANTS
# JOINT_RADIUS = 1.757
# JOINT_RADIUS = 2.26
JOINT_RADIUS = 2.26

def get_rotated_rectangle_points(cx, cy, width, height, theta):
    """
    Calculate the (x, y) coordinates of the corners of a rotated rectangle.
    
    Args:
        cx, cy: Center coordinates of the rectangle.
        width: Width of the rectangle.
        height: Height of the rectangle.
        theta: Rotation angle of the rectangle (in radians).

    Returns:
        A list of 4 (x, y) tuples representing the corners of the rotated rectangle.
    """
    # Half dimensions
    half_width = width / 2
    half_height = height / 2

    # Unrotated rectangle corners (relative to the center)
    corners = [
        (-half_width, -half_height),  # Bottom-left
        (half_width, -half_height),   # Bottom-right
        (half_width, half_height),    # Top-right
        (-half_width, half_height)    # Top-left
    ]

    # Apply rotation to each corner
    cos_t = NP.cos(theta)
    sin_t = NP.sin(theta)

    rotated_corners = [
        (
            cx + corner[0] * cos_t - corner[1] * sin_t,
            cy + corner[0] * sin_t + corner[1] * cos_t
        )
        for corner in corners
    ]

    return rotated_corners

def point_in_polygon(point, polygon):
    """
    Check if a point is inside a polygon using the Ray-Casting Algorithm.
    
    Args:
        point: (x, y) coordinates of the point.
        polygon: A list of (x, y) tuples representing the polygon vertices.

    Returns:
        True if the point is inside the polygon, False otherwise.
    """
    px, py = point
    n = len(polygon)
    inside = False

    # Loop through each edge of the polygon
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]  # Next vertex (with wrap-around)

        # Check if the point is on an edge crossing the y-axis of the point
        if ((y1 > py) != (y2 > py)) and \
           (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside

    return inside

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
    
    # TIME.sleep(2)
    # print(f'\tEnvironment loaded from FILE <{filename}>.')
    
    return OBSTACLES

"""
    function parse_arguments()
"""
def parse_arguments():
    PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    PARSER.add_argument('--map', type = str, required = True, help = 'Filename containg environment.')
    
    return PARSER.parse_args()

"""
    function visualize_scene_arm_robot(environment: list, config: tuple, iteration_num: int) -> None:
"""
def visualize_prm_arm_robot(environment: list, end_effector_positions: list,) -> None:
    FIGURE, AXES = PLT.subplots()
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
        
        # COLLIDING = (
        #     is_colliding_link(BASE, JOINT, OBSTACLE_CORNERS) or (is_colliding_link(JOINT, END_EFFECTOR, OBSTACLE_CORNERS)) or point_in_circle(BASE, (x, y), JOINT_RADIUS) or point_in_circle(JOINT, (x, y), JOINT_RADIUS) or point_in_circle(END_EFFECTOR, (x, y), JOINT_RADIUS)
        # )
        
        # COLOR = '#ff0000' if COLLIDING else '#000000'
        COLOR = '#ff0000'
        
        OBSTACLE_RECTANGLE = PTCHS.Polygon(OBSTACLE_CORNERS, closed = True, edgecolor = COLOR, color = COLOR, fill = True, alpha = 0.5)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    for end_effector_position in end_effector_positions:
        AXES.plot(end_effector_position[0], end_effector_position[1], 'bo', ms = 0.5)
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Arm robot PRM')
    PLT.show(block = True)
    # PLT.pause(1)
    PLT.close(FIGURE)

"""
    function visualize_scene_arm_robot(environment: list, config: tuple, iteration_num: int) -> None:
"""
def visualize_arm_robot(environment: list, config: tuple) -> None:
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
    
    PLT.show(block = False)
    PLT.pause(2)
    PLT.close(FIGURE)

"""
    function main():
    - Main function to run the program.
"""
def main():
    print('\n4. Arm Robot PRM\n')
    
    ARGS = parse_arguments()
    
    ENVIRONMENT = scene_from_file(ARGS.map)
    
    if ARGS.robot == 'arm':
        # generate random VALID configuratoins using theta_1 and theta_2 randomly generated.
        valid_joint_samples = []
        base = (0, 0)
        for i in range(100):
            theta_1 = RANDOM.uniform(0.0, 2 * NP.pi)
            joint_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
            joint_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
            joint = (joint_x, joint_y)
            COLLIDING = False
            for OBSTACLE in ENVIRONMENT:
                x, y, width, height, theta = OBSTACLE
                # OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
                OBSTACLE_CORNERS = get_rotated_rectangle_points(x, y, width, height, theta)
                # COLLIDING = (
                #     is_colliding_link(base, joint, OBSTACLE_CORNERS) or point_in_circle(base, (x, y), JOINT_RADIUS) or point_in_circle(joint, (x, y), JOINT_RADIUS) or point_in_rotated_rectangle(base[0], base[1], x, y, width, height, theta) or point_in_rotated_rectangle(joint[0], joint[1], x, y, width, height, theta)
                # )
                COLLIDING = (
                    is_colliding_link(base, joint, OBSTACLE_CORNERS) or point_in_rotated_rectangle(base[0], base[1], x, y, width, height, theta) or point_in_rotated_rectangle(joint[0], joint[1], x, y, width, height, theta) or point_in_circle(base, (x, y), JOINT_RADIUS) or point_in_circle(joint, (x, y), JOINT_RADIUS)
                )
                
                if COLLIDING: break
                
            if not COLLIDING:
                valid_joint_samples.append((theta_1, 0.0))
        
        # for valid_config in valid_joint_samples:
        #     visualize_arm_robot(environment = ENVIRONMENT, config = valid_config)
        
        end_effector_positions = []
        valid_end_effectors = []
        print(len(valid_joint_samples))
        for valid_joint in valid_joint_samples:
            theta_1 = valid_joint[0]
            joint_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
            joint_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
            joint = (joint_x, joint_y)
            
            for i in range(100):
                theta_2 = RANDOM.uniform(0.0, 2 * NP.pi)
                end_effector_x = joint_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
                end_effector_y = joint_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
                end_effector = (end_effector_x, end_effector_y)
                COLLIDING = False
                for OBSTACLE in ENVIRONMENT:
                    x, y, width, height, theta = OBSTACLE
                    # OBSTACLE_CORNERS = get_polygon_corners(center = (x, y), width = width, height = height, theta = theta)
                    OBSTACLE_CORNERS = get_rotated_rectangle_points(x, y, width, height, theta)
                    COLLIDING = (
                        is_colliding_link(base, joint, OBSTACLE_CORNERS) or is_colliding_link(joint, end_effector, OBSTACLE_CORNERS) or point_in_circle(base, (x, y), JOINT_RADIUS) or point_in_circle(joint, (x, y), JOINT_RADIUS) or point_in_circle(end_effector, (x, y), JOINT_RADIUS)
                        or point_in_rotated_rectangle(end_effector[0], end_effector[1], x, y, width, height, theta) or point_in_rotated_rectangle(joint[0], joint[1], x, y, width, height, theta)
                    )
                    
                    # COLLIDING_II = (
                    #     not point_in_polygon(base, OBSTACLE_CORNERS)
                    # )
                    
                    if not COLLIDING:
                        end_effector_positions.append((end_effector_x, end_effector_y))
                
                    if COLLIDING: break
                
                if not COLLIDING:
                    valid_end_effectors.append((theta_1, theta_2))
        
        print(len(valid_end_effectors))
        print(len(end_effector_positions))
        # for valid_end_effector in valid_end_effectors:
            # visualize_arm_robot(environment = ENVIRONMENT, config = valid_end_effector)
        # end_effector_positions = RANDOM.sample(end_effector_positions, 5000)
        print(f'reduced len(end_effector_positions): {len(end_effector_positions)}')
        visualize_prm_arm_robot(environment = ENVIRONMENT, end_effector_positions = end_effector_positions)

if __name__ == '__main__':
    main()