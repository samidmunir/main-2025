# IMPORTS
import numpy as NP

ENVIRONMENT_WIDTH_MIN = -10.0
ENVIRONMENT_WIDTH_MAX = 10.0
ENVIRONMENT_HEIGHT_MIN = -10.0
ENVIRONMENT_HEIGHT_MAX = 10.0

OBSTACLE_MIN_SIZE = 0.5
OBSTACLE_MAX_SIZE = 2.0

ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5
BASE = (0, 0)
JOINT_RADIUS = 0.05
FREE_BODY_ROBOT_WIDTH = 0.5
FREE_BODY_ROBOT_HEIGHT = 0.3

def get_arm_robot_forward_kinematics(configuration: tuple) -> tuple:
    theta_1, theta_2 = configuration
    
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    
    return (BASE, JOINT, END_EFFECTOR)

def project(corners, axis):
    projections = NP.dot(corners, axis)
    return NP.min(projections), NP.max(projections)

def get_axes(corners):
    edges = NP.diff(NP.vstack([corners, corners[0]]), axis=0)
    return NP.array([[-edge[1], edge[0]] for edge in edges]) / NP.linalg.norm(edges, axis=1, keepdims=True)

def is_colliding(robot_corners, obstacle_corners):
    axes = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])

    for axis in axes:
        min1, max1 = project(robot_corners, axis)
        min2, max2 = project(obstacle_corners, axis)
        if max1 < min2 or max2 < min1:
            return False
    return True

def point_in_circle(point, circle_center, radius):
    DISTANCE = NP.sqrt((point[0] - circle_center[0]) ** 2 + (point[1] - circle_center[1]) ** 2)
    
    return DISTANCE <= radius

def is_line_intersecting(p1, p2, q1, q2):
    def orientation(a, b, c):
        val = (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])
        return 0 if val == 0 else (1 if val > 0 else -1)

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    return (o1 != o2) and (o3 != o4)

def get_polygon_corners(center, theta, width, height):
    w, h = width / 2, height / 2
    corners = NP.array([[-w, -h], [w, -h], [w, h], [-w, h]])

    cos_theta, sin_theta = NP.cos(theta), NP.sin(theta)
    rotation_matrix = NP.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    rotated_corners = corners @ rotation_matrix.T
    return rotated_corners + NP.array(center)

def is_colliding_link(link_start, link_end, obstacle_corners):
    for i in range(len(obstacle_corners)):
        corner1 = obstacle_corners[i]
        corner2 = obstacle_corners[(i + 1) % len(obstacle_corners)]
        if is_line_intersecting(link_start, link_end, corner1, corner2):
            return True
    return False