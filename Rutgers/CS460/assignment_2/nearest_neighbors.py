import argparse as ARGPRS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

from utils import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH,
    BASE,
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT
)

def handle_drawing_free_body_robot(figure, axes, configuration: tuple, color: str):
    x, y, theta = configuration
    
    ROBOT_RECTANGLE = PTCHS.Rectangle(xy = (x, y), width = FREE_BODY_ROBOT_WIDTH, height = FREE_BODY_ROBOT_HEIGHT, angle = NP.rad2deg(theta), color = color, linewidth = 1.0, alpha = 0.75)
    axes.add_patch(ROBOT_RECTANGLE)

def handle_drawing_arm_robot(figure, axes, configuration: tuple, color: str) -> None:
    THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR = configuration
    
    axes.plot(BASE[0], BASE[1], marker = 'o', markersize = 0.75, color = color)
    axes.plot(JOINT[0], JOINT[1], marker = 'o', markersize = 0.75, color = color)
    axes.plot(END_EFFECTOR[0], END_EFFECTOR[1], markersize = 0.75, marker = 'o', color = color)
    
    axes.plot([BASE[0], JOINT[0]], [BASE[1], JOINT[1]], marker = 'o', color = color, linewidth = 1.0)
    axes.plot([JOINT[0], END_EFFECTOR[0]], [JOINT[1], END_EFFECTOR[1]], marker = 'o', color = color, linewidth = 1.0)

def visualize_scene_free_body_robot(configurations: list, k_nearest_configurations: list, target_configuration: tuple) -> None:
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    for CONFIGURATION in configurations:
        handle_drawing_free_body_robot(figure = FIGURE, axes = AXES, configuration = CONFIGURATION, color = '#000000')
        
    for CONFIGURATION in k_nearest_configurations:
        handle_drawing_free_body_robot(figure = FIGURE, axes = AXES, configuration = CONFIGURATION[0], color = '#00ffff')
        
    handle_drawing_free_body_robot(figure = FIGURE, axes = AXES, configuration = target_configuration, color = '#00ff00')
    
    PLT.title('Free-body robot configurations')
    PLT.show()

def visualize_scene_arm_robot(configurations: list, k_nearest_configurations: list, target_configuration: tuple) -> None:
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    for CONFIGURATION in configurations:
        handle_drawing_arm_robot(figure = FIGURE, axes = AXES, configuration = CONFIGURATION, color = '#000000')
        
    for CONFIGURATION in k_nearest_configurations:
        handle_drawing_arm_robot(figure = FIGURE, axes = AXES, configuration = CONFIGURATION[0], color = '#00ffff')
        
    TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (target_configuration[0], target_configuration[1]))
    TARGET_CONFIGURATION = (target_configuration[0], target_configuration[1], TARGET_BASE, TARGET_JOINT, TARGET_END_EFFECTOR)
    handle_drawing_arm_robot(figure = FIGURE, axes = AXES, configuration = TARGET_CONFIGURATION, color = '#00ff00')
    
    PLT.title('Arm robot configurations')
    PLT.show()

def get_angular_difference(theta_1: float, theta_2: float) -> float:
    ANGULAR_DIFFERENCE = NP.abs(theta_1 - theta_2) % (2 * NP.pi)
    
    return min(ANGULAR_DIFFERENCE, 2 * NP.pi - ANGULAR_DIFFERENCE)

def get_euclidean_distance(point: tuple, target_point: tuple) -> float:
    EUCLIDEAN_DISTANCE = NP.sqrt((point[0] - target_point[0]) ** 2 + (point[1] - target_point[1]) ** 2)
    
    return EUCLIDEAN_DISTANCE

def get_k_nearest_free_body_robot_configurations(configurations: list, target_configuration: tuple, angle_weight: float, k: int) -> list:
    CONFIGURATION_DISTANCES = []
    
    TARGET_X, TARGET_Y, TARGET_THETA = target_configuration
    
    for CONFIGURATION in configurations:
        CONFIGURATION_DISTANCE = get_euclidean_distance((CONFIGURATION[0], CONFIGURATION[1]), (TARGET_X, TARGET_Y)) + angle_weight * get_angular_difference(CONFIGURATION[2], TARGET_THETA)
        
        CONFIGURATION_DISTANCES.append((CONFIGURATION, CONFIGURATION_DISTANCE))
    
    CONFIGURATION_DISTANCES.sort(key = lambda DIST: DIST[1])
    
    K_NEAREST_CONFIGURATIONS = CONFIGURATION_DISTANCES[:k]
    
    return K_NEAREST_CONFIGURATIONS

def get_k_nearest_arm_robot_configurations(configurations: list, target_configuration: tuple, k: int) -> list:
    CONFIGURATION_DISTANCES = []
    
    _, _, TARGET_END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (target_configuration[0], target_configuration[1]))
    
    for CONFIGURATION in configurations:
        END_EFFECTOR = CONFIGURATION[4]
        EUCLIDEAN_DISTANCE = get_euclidean_distance(point = END_EFFECTOR, target_point = TARGET_END_EFFECTOR)
        CONFIGURATION_DISTANCES.append((CONFIGURATION, EUCLIDEAN_DISTANCE))
    
    CONFIGURATION_DISTANCES.sort(key = lambda EUC_DIST: EUC_DIST[1])
    
    K_NEAREST_CONFIGURATIONS = CONFIGURATION_DISTANCES[:k]
    
    return K_NEAREST_CONFIGURATIONS

def get_arm_robot_forward_kinematics(configuration: tuple) -> tuple:
    theta_1, theta_2 = configuration
    
    JOINT_X = ARM_ROBOT_LINK_1_LENGTH * NP.cos(theta_1)
    JOINT_Y = ARM_ROBOT_LINK_1_LENGTH * NP.sin(theta_1)
    JOINT = (JOINT_X, JOINT_Y)
    
    END_EFFECTOR_X = JOINT_X + ARM_ROBOT_LINK_2_LENGTH * NP.cos(theta_1 + theta_2)
    END_EFFECTOR_Y = JOINT_Y + ARM_ROBOT_LINK_2_LENGTH * NP.sin(theta_1 + theta_2)
    END_EFFECTOR = (END_EFFECTOR_X, END_EFFECTOR_Y)
    
    return (BASE, JOINT, END_EFFECTOR)

def load_sample_free_body_robot_configurations(filename: str) -> list:
    print(f'load_sample_free_body_robot_configurations({filename}) called...')
    
    CONFIGURATIONS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        for LINE in LINES:
            VALUES = LINE.strip().split()
            x, y, theta = float(VALUES[0]), float(VALUES[1]), float(VALUES[2])
            CONFIGURATION = (x, y, theta)
            CONFIGURATIONS.append(CONFIGURATION)
    
    return CONFIGURATIONS

def load_sample_arm_robot_configurations(filename: str) -> list:
    print(f'load_sample_arm_robot_configurations({filename}) called...')
    
    CONFIGURATIONS = []
    
    with open(filename, 'r') as FILE:
        LINES = FILE.readlines()
        for LINE in LINES:
            VALUES = LINE.strip().split()
            THETA_1, THETA_2 = float(VALUES[0]), float(VALUES[1])
            BASE, JOINT, END_EFFECTOR = get_arm_robot_forward_kinematics(configuration = (THETA_1, THETA_2))
            CONFIGURATION = (THETA_1, THETA_2, BASE, JOINT, END_EFFECTOR)
            CONFIGURATIONS.append(CONFIGURATION)
    
    return CONFIGURATIONS

def parse_arguments():
    ARG_PARSER = ARGPRS.ArgumentParser(description = 'Nearest neighbors with linear search approach.')
    
    ARG_PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    ARG_PARSER.add_argument('--target', type = float, nargs = '+', required = True, help = 'Target configuration of the robot. (N = 2 for arm, N = 3 for freeBody)')
    ARG_PARSER.add_argument('-k', type = int, required = True, default = 3, help = 'Number of nearest neighbors to consider.')
    ARG_PARSER.add_argument('--configs', type = str, required = True, help = 'Filename containing list of random arm/freeBody robot configurations.')
    
    return ARG_PARSER.parse_args()

def main():
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        RANDOM_CONFIGURATIONS = load_sample_arm_robot_configurations(filename = ARGS.configs)
        K_NEAREST_CONFIGURATIONS = get_k_nearest_arm_robot_configurations(configurations = RANDOM_CONFIGURATIONS, target_configuration = ARGS.target, k = ARGS.k)
        visualize_scene_arm_robot(configurations = RANDOM_CONFIGURATIONS, k_nearest_configurations = K_NEAREST_CONFIGURATIONS, target_configuration = ARGS.target)
    elif ARGS.robot == 'freeBody':
        RANDOM_CONFIGURATIONS = load_sample_free_body_robot_configurations(filename = ARGS.configs)
        K_NEAREST_CONFIGURATIONS = get_k_nearest_free_body_robot_configurations(configurations = RANDOM_CONFIGURATIONS, target_configuration = ARGS.target, angle_weight = 0.25, k = ARGS.k)
        visualize_scene_free_body_robot(configurations = RANDOM_CONFIGURATIONS, k_nearest_configurations = K_NEAREST_CONFIGURATIONS, target_configuration = ARGS.target)

if __name__ == '__main__':
    main()