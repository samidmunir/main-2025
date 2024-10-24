"""
    Utility file containing helper functions.
    - utils.py
        * CONSTANTS
        > function get_polygon_corners(center: (float, float), width: float, height: float, theta: float) -> NP.ndarray:)
"""

# IMPORTS
import numpy as NP

# CONSTANTS
ENVIRONMENT_WIDTH_MIN = -10.0
ENVIRONMENT_WIDTH_MAX = 10.0
ENVIRONMENT_HEIGHT_MIN = -10.0
ENVIRONMENT_HEIGHT_MAX = 10.0

OBSTACLE_MIN_SIZE = 0.5
OBSTACLE_MAX_SIZE = 2.0

ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5
FREE_BODY_ROBOT_WIDTH = 0.5
FREE_BODY_ROBOT_HEIGHT = 0.3

"""
    function get_polygon_corners(center: tuple, width: float, height: float, theta: float) -> NP.ndarray:
    - this function returns the world (environment) coordinates of a polygon.
"""
def get_polygon_corners(center: tuple, width: float, height: float, theta: float) -> NP.ndarray:
    width_prime, height_prime = width / 2, height / 2
    CORNERS = NP.array(
        [
            [-width_prime, -height_prime],
            [width_prime, -height_prime],
            [width_prime, height_prime],
            [-width_prime, height_prime]
        ]
    )
    
    COS_THETA, SIN_THETA = NP.cos(theta), NP.sin(theta)
    ROTATION_MATRIX = NP.array(
        [
            [COS_THETA, -SIN_THETA],
            [SIN_THETA, COS_THETA]
        ]
    )
    
    ROTATED_CORNERS = CORNERS @ ROTATION_MATRIX.T
    
    return ROTATED_CORNERS + NP.array(center)

"""
    function get_arm_robot_joint_positions(theta_1: float, theta_2: float) -> tuple:
    - this function returns a tuple of the cartesian coordinates of the BASE, JOINT, and END_EFFECTOR of the arm-robot, based on the angles theta_1 & theta_2.
"""
def get_arm_robot_joint_positions(theta_1: float, theta_2: float) -> tuple:
    BASE = (0, 0)
    
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    
    return (BASE, JOINT, END_EFFECTOR)

"""
    function load_sample_arm_robot_configurations(filename: str) -> list:
    - this function loads in a list of arm-robot configurations from the file specified by filename.
    - we want to store the configuration of each arm-robot as a tuple of the following form: (theta_1, theta_2, BASE, JOINT, END_EFFECTOR).
    - we will compute the BASE, JOINT, END_EFFECTOR positions (x, y) given the two angles theta_1 & theta_2 and (the length of the links as constants).
    - call utility function get_arm_robot_joint_positions(theta_1: float, theta_2: float)
"""
def load_sample_arm_robot_configurations(filename: str) -> list: