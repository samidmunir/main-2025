"""
    Utility file containing helper functions.
    - utils.py
        * CONSTANTS
        > function get_polygon_corners(center: (float, float), width: float, height: float, theta: float) -> NP.ndarray:)
        
        > function get_arm_robot_joint_positions(theta_1: float, theta_2: float) -> tuple:
        
        > function load_sample_arm_robot_configurations(filename: str) -> list:
        
        > function get_euclidean_distance(point, target_point) -> float:
        
        > function get_knn_nearest_arm_robot_configurations(configurations, target_configuration) -> list:
        
        > function handle_arm_robot_visualization(figure, axes, configuration:
        
        > function visualize_knn_scene_arm_robot(configurations: list) -> None:
        
        > function load_sample_free_body_robot_configurations(filename: str) -> list:
        
        > function get_angular_difference(theta_1, theta_2) -> float:
        
        > funcion get_k_nearest_free_body_robot_configurations(configurations: list, target_configuration: tuple, orientation_weight: float, k: int) -> list:
        
        > function handle_free_body_robot_visualization(figure, axes, configuration: tuple) -> None:
        
        > function visualize_knn_scene_free_body_robot():
"""

# IMPORTS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
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
    - return the list of configurations.
"""
def load_sample_arm_robot_configurations(filename: str) -> list:
    CONFIGURATIONS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        
        for LINE in LINES:
            VALUES = LINE.strip().split()
            THETA_1, THETA_2 = VALUES
            THETA_1, THETA_2 = float(THETA_1), float(THETA_2)
            BASE, JOINT, END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = THETA_1, theta_2 = THETA_2)

            CONFIGURATIONS.append((THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR))
    
    return CONFIGURATIONS

"""
    function get_euclidean_distance(point, target_point) -> float:
    - this function calculates the Euclidean distance between two points (x, y).
"""
def get_euclidean_distance(point, target_point) -> float:
    EUCLIDEAN_DIST = NP.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
    
    return EUCLIDEAN_DIST

"""
    function get_k_nearest_arm_robot_configurations(configs, target_configuration) -> list:
    - this function finds the k nearest configurations to the target configuration in the list of configurations using the position of the end-effector (x, y).
"""
def get_k_nearest_arm_robot_configurations(configurations, target_configuration, k: int) -> list:
    
    CONFIGS_DISTS = []
    
    TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR = get_arm_robot_joint_positions(target_configuration[0], target_configuration[1])
    
    for CONFIG in configurations:
        END_EFFECTOR = CONFIG[4]
        EUCLIDEAN_DIST = get_euclidean_distance(point = END_EFFECTOR, target_point = TARGET_END_EFFECTOR)
        CONFIGS_DISTS.append((CONFIG, EUCLIDEAN_DIST))
    
    CONFIGS_DISTS.sort(key = lambda EUC_DIST: EUC_DIST[1])
    
    SORTED_END_EFFECTOR_POSITIONS = sorted(configurations)
    
    K_NEAREST_CONFIGS = CONFIGS_DISTS[:k]
    
    return K_NEAREST_CONFIGS

"""
    function handle_arm_robot_visualization(figure, axes, configuration: tuple, joint_color: str, line_color: str) -> None:
    - this function handles the drawing process of the 2-link/2-arm robot.
"""
def handle_arm_robot_visualization(figure, axes, configuration: tuple, joint_color: str, line_color: str) -> None:
    
    THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = configuration
    
    axes.plot([BASE[0], JOINT[0]], [BASE[1], JOINT[1]], marker = 'o', color = line_color, linewidth = 1.0)
    axes.plot([JOINT[0], END_EFFECTOR[0]], [JOINT[1], END_EFFECTOR[1]], marker = 'o', color = line_color, linewidth = 1.0)
    
    axes.plot(BASE[0], BASE[1], marker = 'o', color = joint_color, label = 'Base')
    axes.plot(JOINT[0], JOINT[1], marker = 'o', color = joint_color, label = 'Joint')
    axes.plot(END_EFFECTOR[0], END_EFFECTOR[1], marker = 'o', color = joint_color, label = 'End-effector')

"""
    function visualize_knn_scene_arm_robot(configurations: list) -> None:
    - this function handles the primary visualization fot the target configuration, the list of all configurations, and the k-nearest configurations.
"""
def visualize_knn_scene_arm_robot(configurations: list, k_nearest_configurations: list, target_configuration: tuple) -> None:
    FIGURE, AXES = PLT.subplots()
    
    for CONFIG in configurations:
        handle_arm_robot_visualization(figure = FIGURE, axes = AXES, configuration = CONFIG, line_color = '#000000', joint_color = '#000000')
        
    for CONFIG in k_nearest_configurations:
        handle_arm_robot_visualization(figure = FIGURE, axes = AXES, configuration = CONFIG[0], line_color = '#00ffff', joint_color = '#00ffff')

    TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR = get_arm_robot_joint_positions(theta_1 = target_configuration[0], theta_2 = target_configuration[1])
    target_configuration = (target_configuration[0], target_configuration[1], TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR)
    
    handle_arm_robot_visualization(figure = FIGURE, axes = AXES, configuration = target_configuration, line_color = '#00ff00', joint_color = '#00ff00')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Arm robot configurations')
    
    # TODO: add legend to display/plot.
    
    PLT.show()
    
"""
    function load_sample_free_body_robot_configurations(filename: str) -> list:
    - this function loads in a list of free-body-robot configurations from the file specified by filename.
    - we want to store the configuration of each arm-robot as a tuple of the following form: (x, y, theta).
    - return the list of free-body-robot configurations.
"""
def load_sample_free_body_robot_configurations(filename: str) -> list:
    print(f'\nload_sample_freeBody_configs({filename}) called...')
    
    CONFIGURATIONS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        
        for LINE in LINES:
            VALUES = LINE.strip().split()
            x, y, theta = VALUES[0], VALUES[1], VALUES[2]
            
            CONFIGURATION = (float(x), float(y), float(theta))
            
            CONFIGURATIONS.append(CONFIGURATION)
    
    print(f'\tsample free body configurations loaded from FILE <{filename}>.')
    
    return CONFIGURATIONS

"""
    function get_angular_difference(theta_1, theta_2) -> float:
    - this function calculates the angular difference between two angles, taking into account the periodicity of the angle.
"""
def get_angular_difference(theta_1: float, theta_2: float) -> float:
    ANGR_DIFF = NP.abs(theta_1 - theta_2) % (2 * NP.pi)
    
    return min(ANGR_DIFF, 2 * NP.pi - ANGR_DIFF)

"""
    funcion get_k_nearest_free_body_robot_configurations(configurations: list, target_configuration: tuple, orientation_weight: float, k: int) -> list:
    - this function finds the k nearest configurations to the target configuration in the list of configurations using the position of the center of the robot and its orientation (angle).
"""
def get_k_nearest_free_body_robot_configurations(configurations: list, target_configuration: tuple, orientation_weight: float, k: int) -> list:
    CONFIGURATION_DISTANCES = []
    
    TARGET_X, TARGET_Y, TARGET_THETA = target_configuration
    
    for CONFIGURATION in configurations:
        CONFIG_DIST = get_euclidean_distance((CONFIGURATION[0], CONFIGURATION[1]), (TARGET_X, TARGET_Y)) + orientation_weight * get_angular_difference(CONFIGURATION[2], TARGET_THETA)
    
        CONFIGURATION_DISTANCES.append((CONFIGURATION, CONFIG_DIST))
    
    CONFIGURATION_DISTANCES.sort(key = lambda W_DIST: W_DIST[1])
    K_NEAREST_CONFIGS = CONFIGURATION_DISTANCES[:k]
    
    return K_NEAREST_CONFIGS

"""
    function handle_free_body_robot_visualization(figure, axes, configuration: tuple) -> None:
    - this function handles drawing/visualization of the free-body robot.
"""
def handle_free_body_robot_visualization(figure, axes, configuration: tuple, edge_color: str, fill_color: str) -> None:
    FIGURE = figure
    AXES = axes
    
    x, y, theta = configuration
    
    ROBOT_RECTANGLE = PTCHS.Rectangle((x, y), FREE_BODY_ROBOT_WIDTH, FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(theta), color = fill_color, edgecolor = edge_color, linewidth = 1.0, alpha = 0.75)
    
    AXES.add_patch(ROBOT_RECTANGLE)
    
"""
    function visualize_knn_scene_free_body_robot():
    - this function handles the visualization of the target configuration, the list of configurations, and the k-nearest configurations.
"""
def visualize_knn_scene_free_body_robot(configurations: list, k_nearest_configurations: list, target_configuration: tuple) -> None:
    FIGURE, AXES = PLT.subplots()
    
    for CONFIG in configurations:
        handle_free_body_robot_visualization(figure = FIGURE, axes = AXES, configuration = CONFIG, edge_color = '#000000', fill_color = '#000000')
    
    for CONFIG in k_nearest_configurations:
        handle_free_body_robot_visualization(figure = FIGURE, axes = AXES, configuration = CONFIG[0], edge_color = '#0000ff', fill_color = '#0000ff')
        
    handle_free_body_robot_visualization(figure = FIGURE, axes = AXES, configuration = target_configuration, edge_color = '#00ff00', fill_color = '#00ff00')
    
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    PLT.title('Free-body robot configurations')
    
    PLT.show()