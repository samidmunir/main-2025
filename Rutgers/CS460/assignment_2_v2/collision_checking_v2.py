# import argparse
# import math as MATH
# import random as RANDOM
# import time as TIME
# import numpy as NP
# import matplotlib.pyplot as PLT
# import matplotlib.patches as PTCHS

# from component_1 import (
#     ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION, 
#     FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, 
#     ARM_ROBOT_LINK_1_LENGTH, ARM_ROBOT_LINK_2_LENGTH
# )

# def scene_from_file(filename: str) -> list:
#     """Load obstacles from a file with (x, y, width, height, angle)."""
#     OBSTACLES = []
#     with open(filename, 'r') as FILE:
#         for LINE in FILE:
#             x, y, width, height, angle = map(float, LINE.strip().split(','))
#             OBSTACLES.append((x, y, width, height, angle))
#     return OBSTACLES

# def get_polygon_corners(center, theta, width, height):
#     """Calculate the world coordinates of the rectangle's corners."""
#     w, h = width / 2, height / 2
#     corners = NP.array([[-w, -h], [w, -h], [w, h], [-w, h]])

#     cos_theta, sin_theta = NP.cos(theta), NP.sin(theta)
#     rotation_matrix = NP.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

#     rotated_corners = corners @ rotation_matrix.T
#     return rotated_corners + NP.array(center)

# def get_end_effector_position(theta1, theta2):
#     """Calculate the (x, y) position of the end-effector using forward kinematics."""
#     joint1_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta1)
#     joint1_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta1)

#     end_effector_x = joint1_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta1 + theta2)
#     end_effector_y = joint1_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta1 + theta2)

#     return (joint1_x, joint1_y), (end_effector_x, end_effector_y)

# def is_line_intersecting(p1, p2, q1, q2):
#     """Check if two line segments (p1-p2 and q1-q2) intersect."""
#     def orientation(a, b, c):
#         val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
#         return 0 if val == 0 else (1 if val > 0 else -1)

#     o1 = orientation(p1, p2, q1)
#     o2 = orientation(p1, p2, q2)
#     o3 = orientation(q1, q2, p1)
#     o4 = orientation(q1, q2, p2)

#     return (o1 != o2) and (o3 != o4)

# def is_colliding_link(link_start, link_end, obstacle_corners):
#     """Check if a robot link intersects with any edge of the obstacle."""
#     for i in range(len(obstacle_corners)):
#         corner1 = obstacle_corners[i]
#         corner2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
#         if is_line_intersecting(link_start, link_end, corner1, corner2):
#             return True
#     return False

def get_axes(corners):
    """Compute axes perpendicular to the edges of the polygon."""
    edges = NP.diff(NP.vstack([corners, corners[0]]), axis=0)
    return NP.array([[-edge[1], edge[0]] for edge in edges]) / NP.linalg.norm(edges, axis=1, keepdims=True)

def is_colliding(robot_corners, obstacle_corners):
    """Check collision between two polygons using SAT."""
    axes = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])

    for axis in axes:
        min1, max1 = project(robot_corners, axis)
        min2, max2 = project(obstacle_corners, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

# def get_axes(corners):
#     """Compute axes perpendicular to the edges of the polygon."""
#     edges = NP.diff(NP.vstack([corners, corners[0]]), axis=0)
#     return NP.array([[-edge[1], edge[0]] for edge in edges]) / NP.linalg.norm(edges, axis=1, keepdims=True)

# def project(corners, axis):
#     """Project corners onto the given axis."""
#     projections = NP.dot(corners, axis)
#     return NP.min(projections), NP.max(projections)

# def visualize_scene_arm(environment, theta1, theta2, iteration_num):
#     """Visualize the arm robot and color colliding obstacles in red."""
#     FIGURE, AXES = PLT.subplots()

#     (joint1_x, joint1_y), (end_effector_x, end_effector_y) = get_end_effector_position(theta1, theta2)

#     base = (0, 0)
#     joint1 = (joint1_x, joint1_y)
#     end_effector = (end_effector_x, end_effector_y)

#     for obstacle in environment:
#         x, y, width, height, theta = obstacle
#         obstacle_corners = get_polygon_corners((x, y), theta, width, height)

#         colliding = (
#             is_colliding_link(base, joint1, obstacle_corners) or 
#             is_colliding_link(joint1, end_effector, obstacle_corners)
#         )

#         color = 'red' if colliding else 'black'
#         patch = PTCHS.Polygon(obstacle_corners, closed=True, edgecolor=color, fill=False)
#         AXES.add_patch(patch)

#     AXES.plot([0, joint1_x], [0, joint1_y], color='blue', linewidth=2)
#     AXES.plot([joint1_x, end_effector_x], [joint1_y, end_effector_y], color='green', linewidth=2)

#     AXES.set_aspect('equal')
#     AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
#     AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)

#     PLT.title(f'Arm Robot Collision Test (Iteration #{iteration_num})')
#     PLT.show(block=False)
#     PLT.pause(1)
#     PLT.close(FIGURE)

def project(corners, axis):
    """Project corners onto the given axis."""
    projections = NP.dot(corners, axis)
    return NP.min(projections), NP.max(projections)

def visualize_scene_free_body(environment, config, iteration_num):
    """Visualize the free-body robot and color colliding obstacles in red."""
    FIGURE, AXES = PLT.subplots()

    robot_corners = get_polygon_corners(config[:2], config[2], FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT)
    for obstacle in environment:
        x, y, width, height, theta = obstacle
        obstacle_corners = get_polygon_corners((x, y), theta, width, height)

        colliding = is_colliding(robot_corners, obstacle_corners)

        color = 'red' if colliding else 'black'
        patch = PTCHS.Polygon(obstacle_corners, closed=True, edgecolor=color, fill=False)
        AXES.add_patch(patch)

    robot_patch = PTCHS.Polygon(robot_corners, closed=True, edgecolor='blue', fill=True, alpha=0.5)
    AXES.add_patch(robot_patch)

    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)

    PLT.title(f'Free Body Robot Collision Test (Iteration #{iteration_num})')
    PLT.show(block=False)
    PLT.pause(1)
    PLT.close(FIGURE)

# def parse_arguments():
#     """Parse command-line arguments."""
#     PARSER = argparse.ArgumentParser(description='Collision Checking with Obstacles')
#     PARSER.add_argument('--robot', required=True, choices=['arm', 'freeBody'], help='Type of robot')
#     PARSER.add_argument('--map', required=True, help='Environment map file')
#     return PARSER.parse_args()

# def main():
#     """Main function to run the collision checking."""
#     ARGS = parse_arguments()
#     ENVIRONMENT = scene_from_file(ARGS.map)

#     if ARGS.robot == 'arm':
#         for i in range(10):
#             theta1 = RANDOM.uniform(0, 2 * MATH.pi)
#             theta2 = RANDOM.uniform(0, 2 * MATH.pi)
#             visualize_scene_arm(ENVIRONMENT, theta1, theta2, i + 1)
#             TIME.sleep(1)
#     elif ARGS.robot == 'freeBody':
#         for i in range(10):
#             CONFIG = (
#                 RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
#                 RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
#                 RANDOM.uniform(0, 2 * MATH.pi)
#             )
#             visualize_scene_free_body(ENVIRONMENT, CONFIG, i + 1)
#             TIME.sleep(1)

# if __name__ == '__main__':
#     main()

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

def scene_from_file(filename: str) -> list:
    """Load obstacles from a file with (x, y, width, height, angle)."""
    OBSTACLES = []
    with open(filename, 'r') as FILE:
        for LINE in FILE:
            x, y, width, height, angle = map(float, LINE.strip().split(','))
            OBSTACLES.append((x, y, width, height, angle))
    return OBSTACLES

def get_polygon_corners(center, theta, width, height):
    """Calculate the world coordinates of the rectangle's corners."""
    w, h = width / 2, height / 2
    corners = NP.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    cos_theta, sin_theta = NP.cos(theta), NP.sin(theta)
    rotation_matrix = NP.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_corners = corners @ rotation_matrix.T
    return rotated_corners + NP.array(center)

def get_end_effector_position(theta1, theta2):
    """Calculate the (x, y) position of the end-effector using forward kinematics."""
    joint1_x = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta1)
    joint1_y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta1)

    end_effector_x = joint1_x + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta1 + theta2)
    end_effector_y = joint1_y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta1 + theta2)

    return (joint1_x, joint1_y), (end_effector_x, end_effector_y)

def visualize_arm_robot(FIGURE, AXES, base, joint1, end_effector, joint_color, line_color):
    """Draw the arm robot with markers at joints and links."""
    # Line from base to joint1
    AXES.plot([base[0], joint1[0]], [base[1], joint1[1]], marker='o', color=f'{line_color}', label='First Arm Link')
    # Line from joint1 to end-effector
    AXES.plot([joint1[0], end_effector[0]], [joint1[1], end_effector[1]], marker='o', color=f'{line_color}', label='Second Arm Link')

    # Mark the base, joint1, and end-effector
    AXES.plot(base[0], base[1], marker='o', color=f'{joint_color}', label='Base Joint')
    AXES.plot(joint1[0], joint1[1], marker='o', color=f'{joint_color}', label='Joint 1')
    AXES.plot(end_effector[0], end_effector[1], marker='o', color=f'{joint_color}', label='End Effector')

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

def is_colliding_link(link_start, link_end, obstacle_corners):
    """Check if a robot link intersects with any edge of the obstacle."""
    for i in range(len(obstacle_corners)):
        corner1 = obstacle_corners[i]
        corner2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        if is_line_intersecting(link_start, link_end, corner1, corner2):
            return True
    return False

def visualize_scene_arm(environment, theta1, theta2, iteration_num):
    """Visualize the arm robot and color colliding obstacles in red."""
    FIGURE, AXES = PLT.subplots()

    # Calculate positions
    (joint1_x, joint1_y), (end_effector_x, end_effector_y) = get_end_effector_position(theta1, theta2)
    base = (0, 0)
    joint1 = (joint1_x, joint1_y)
    end_effector = (end_effector_x, end_effector_y)

    # Plot obstacles and check for collisions
    for obstacle in environment:
        x, y, width, height, theta = obstacle
        obstacle_corners = get_polygon_corners((x, y), theta, width, height)

        colliding = (
            is_colliding_link(base, joint1, obstacle_corners) or 
            is_colliding_link(joint1, end_effector, obstacle_corners)
        )

        color = 'red' if colliding else 'black'
        patch = PTCHS.Polygon(obstacle_corners, closed=True, edgecolor=color, fill=False)
        AXES.add_patch(patch)

    # Visualize the arm robot using the provided visualization function
    visualize_arm_robot(FIGURE, AXES, base, joint1, end_effector, joint_color='green', line_color='blue')

    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)
    AXES.set_ylim(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION)

    PLT.title(f'Arm Robot Collision Test (Iteration #{iteration_num})')
    PLT.show(block=False)
    PLT.pause(1)
    PLT.close(FIGURE)

def parse_arguments():
    """Parse command-line arguments."""
    PARSER = argparse.ArgumentParser(description='Collision Checking with Obstacles')
    PARSER.add_argument('--robot', required=True, choices=['arm', 'freeBody'], help='Type of robot')
    PARSER.add_argument('--map', required=True, help='Environment map file')
    return PARSER.parse_args()

def main():
    """Main function to run the collision checking."""
    ARGS = parse_arguments()
    ENVIRONMENT = scene_from_file(ARGS.map)

    if ARGS.robot == 'arm':
        for i in range(10):
            theta1 = RANDOM.uniform(0, 2 * MATH.pi)
            theta2 = RANDOM.uniform(0, 2 * MATH.pi)
            visualize_scene_arm(ENVIRONMENT, theta1, theta2, i + 1)
            TIME.sleep(1)
    elif ARGS.robot == 'freeBody':
        for i in range(10):
            CONFIG = (
                RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
                RANDOM.uniform(ENVIRONMENT_MIN_POSITION, ENVIRONMENT_MAX_POSITION),
                RANDOM.uniform(0, 2 * MATH.pi)
            )
            visualize_scene_free_body(ENVIRONMENT, CONFIG, i + 1)
            TIME.sleep(1)

if __name__ == '__main__':
    main()
