# import argparse
# import numpy as NP

# # Function to parse the configurations from a file
# def load_configs(filename: str):
#     CONFIGS = []
#     with open(filename, 'r') as file:
#         for line in file:
#             config = list(map(float, line.strip().split()))
#             CONFIGS.append(config)
    
#     return NP.array(CONFIGS)

# # Function to parse arguments.
# def parse_args():
#     parser = argparse.ArgumentParser(description = 'Nearest neighbors with linear search.')
#     parser.add_argument('--robot', type = str, required = True, choices = ['arm', 'freeBody'],
#                         help = "Type of robot: 'arm' or 'freeBody'")
#     parser.add_argument('--target', type = float, nargs = '+', required = True,
#                         help = "Target configuration for the robot")
#     parser.add_argument('-k', type = int, required = True,
#                         help = "Number of nearest neighbors to find")
#     parser.add_argument('--configs', type = str, required = True,
#                         help = "File containing the configurations")
    
#     return parser.parse_args()

# # Function to calculate the distance between two configurations.
# def compute_distance(config1, config2, robot_type):
#     if robot_type == 'arm':
#         return NP.linalg.norm(config1 - config2)
#     elif robot_type == 'freeBody':
#         # freeBody configuration: distance for (x, y, theta)
#         position_delta = NP.linalg.norm(NP.array(config1[:2]) - NP.array(config2[:2]))
#         orientation_delta = abs(config1[2] - config2[2])
        
#         return position_delta + orientation_delta

# import matplotlib.pyplot as PLT
# import matplotlib.patches as PTCHS
# import numpy as NP

# # Visualize the arm (2-joint) robot with customizable colors
# def visualize_arm_robot(config, ax, link1_color='red', link2_color='blue', base_joint_color='green', middle_joint_color='magenta', end_effector_color='orange'):
#     joint1_angle, joint2_angle = config

#     # Arm lengths
#     length1 = 1.5
#     length2 = 2.0

#     # Base position (fixed at origin)
#     base_x, base_y = 0, 0

#     # Calculate position of the first joint (middle joint)
#     joint1_x = base_x + length1 * NP.cos(joint1_angle)
#     joint1_y = base_y + length1 * NP.sin(joint1_angle)

#     # Calculate position of the end effector
#     end_effector_x = joint1_x + length2 * NP.cos(joint1_angle + joint2_angle)
#     end_effector_y = joint1_y + length2 * NP.sin(joint1_angle + joint2_angle)

#     # Plot base to joint1 (first link)
#     ax.plot([base_x, joint1_x], [base_y, joint1_y], color=link1_color, lw=4, label='Link 1')
    
#     # Plot joint1 to end effector (second link)
#     ax.plot([joint1_x, end_effector_x], [joint1_y, end_effector_y], color=link2_color, lw=4, label='Link 2')
    
#     # Mark the joints and the end-effector
#     ax.plot(base_x, base_y, 'o', markersize=10, markerfacecolor=base_joint_color, label='Base Joint')
#     ax.plot(joint1_x, joint1_y, 'o', markersize=10, markerfacecolor=middle_joint_color, label='Middle Joint')
#     ax.plot(end_effector_x, end_effector_y, 'o', markersize=10, markerfacecolor=end_effector_color, label='End Effector')

#     # Set limits and labels
#     ax.set_xlim(-5, 5)
#     ax.set_ylim(-5, 5)
#     ax.set_xlabel('X-axis')
#     ax.set_ylabel('Y-axis')

#     # Add legend
#     ax.legend()
    
# # Visualize the freeBody robot (rectangle)
# def visualize_freebody_robot(config, ax):
#     x, y, theta = config
    
#     # Rectangle parameters: width, height
#     width, height = 0.5, 0.3
    
#     # Create a rectangle patch
#     rect = PTCHS.Rectangle((x - width / 2, y - height / 2), width, height, angle = NP.degrees(theta),
#                              edgecolor = 'b', facecolor = 'b', lw = 2)
#     ax.add_patch(rect)

#     # Set limits
#     ax.set_xlim(0, 20)
#     ax.set_ylim(0, 20)
    
#     # Mark the center
#     # ax.plot(x, y, 'bo', markersize = 10)
#     PLT.show()

# # Main visualization function to handle both robots
# def visualize_robot(config, robot_type, link1_color='red', link2_color='blue', base_joint_color='green', middle_joint_color='magenta', end_effector_color='orange'):
#     fig, ax = PLT.subplots()

#     if robot_type == 'arm':
#         visualize_arm_robot(config, ax, link1_color, link2_color, base_joint_color, middle_joint_color, end_effector_color)
#     elif robot_type == 'freeBody':
#         visualize_freebody_robot(config, ax)

#     ax.set_aspect('equal')
#     PLT.show()

# # Function to visualize all configurations with different colors
# def visualize_configs(target, configs, nearest_neighbors, robot_type):
#     fig, ax = PLT.subplots()

#     # Extract only the configuration part from nearest_neighbors (ignore distances)
#     nearest_configs = [config for _, config in nearest_neighbors]

#     # Plot all configurations in gray (remaining ones)
#     for config in configs:
#         # Use np.array_equal to compare the configurations
#         if not any(NP.array_equal(config, nc) for nc in nearest_configs) and not NP.array_equal(config, target):
#             visualize_robot(config, robot_type, link1_color='gray', link2_color='gray', 
#                             base_joint_color='gray', middle_joint_color='gray', end_effector_color='gray')

#     # Plot the nearest neighbors in blue
#     for config in nearest_configs:
#         visualize_robot(config, robot_type, link1_color='blue', link2_color='blue', 
#                         base_joint_color='blue', middle_joint_color='blue', end_effector_color='blue')

#     # Plot the target configuration in green
#     visualize_robot(target, robot_type, link1_color='green', link2_color='green', 
#                     base_joint_color='green', middle_joint_color='green', end_effector_color='green')

#     # Set aspect ratio and show all configurations in one window
#     ax.set_aspect('equal')
#     PLT.show()

# def main():
#     ARGS = parse_args()
    
#     # Load configurations from file.
#     CONFIGS = load_configs(ARGS.configs)
#     TARGET = NP.array(ARGS.target)
    
#     # Compute distances from target to each configuration.
#     DISTANCES = []
#     for config in CONFIGS:
#         distance = compute_distance(config, TARGET, ARGS.robot)
#         DISTANCES.append((distance, config))
    
#     # Sort CONFIGS by distance and select the k nearest neighbors.
#     DISTANCES.sort(key = lambda x: x[0])
#     NEAREST_NEIGHBORS = DISTANCES[:ARGS.k]
    
#     # Output the nearest neighbors.
#     print(f'Target configuration: {TARGET}')
#     print(f'Nearest {ARGS.k} Neighbors:')
#     for distance, configuration in NEAREST_NEIGHBORS:
#         print(f'Configuration: {configuration}, Distance: {distance:.2f} meters')
    
#     # Visualize the target, nearest neighbors, and the rest of the configurations.
#     visualize_configs(TARGET, CONFIGS, NEAREST_NEIGHBORS, ARGS.robot)

# if __name__ == "__main__":
#     main()

import argparse
import numpy as np
import matplotlib.pyplot as plt

# Function to load configurations from a file
def load_configs(filename):
    configs = []
    with open(filename, 'r') as file:
        for line in file:
            config = list(map(float, line.strip().split()))
            configs.append(config)
    return np.array(configs)

# Function to calculate distance between two configurations
def compute_distance(config1, config2, robot_type):
    if robot_type == 'arm':
        # Arm configuration: use difference in joint angles
        return np.linalg.norm(np.array(config1) - np.array(config2))
    elif robot_type == 'freeBody':
        # FreeBody configuration: distance for (x, y, theta)
        position_dist = np.linalg.norm(np.array(config1[:2]) - np.array(config2[:2]))
        orientation_dist = abs(config1[2] - config2[2])
        return position_dist + orientation_dist

# Visualization functions
def visualize_arm_robot(config, ax, link1_color='gray', link2_color='gray', base_joint_color='gray', middle_joint_color='gray', end_effector_color='gray'):
    joint1_angle, joint2_angle = config

    # Arm lengths
    length1 = 1.5
    length2 = 2.0

    # Base position (fixed at origin)
    base_x, base_y = 0, 0

    # Calculate position of the first joint (middle joint)
    joint1_x = base_x + length1 * np.cos(joint1_angle)
    joint1_y = base_y + length1 * np.sin(joint1_angle)

    # Calculate position of the end effector
    end_effector_x = joint1_x + length2 * np.cos(joint1_angle + joint2_angle)
    end_effector_y = joint1_y + length2 * np.sin(joint1_angle + joint2_angle)

    # Plot base to joint1 (first link)
    ax.plot([base_x, joint1_x], [base_y, joint1_y], color=link1_color, lw=4)
    
    # Plot joint1 to end effector (second link)
    ax.plot([joint1_x, end_effector_x], [joint1_y, end_effector_y], color=link2_color, lw=4)
    
    # Mark the joints and the end-effector
    ax.plot(base_x, base_y, 'o', markersize=10, markerfacecolor=base_joint_color)
    ax.plot(joint1_x, joint1_y, 'o', markersize=10, markerfacecolor=middle_joint_color)
    ax.plot(end_effector_x, end_effector_y, 'o', markersize=10, markerfacecolor=end_effector_color)

# def visualize_freebody_robot(config, ax, rectangle_color='gray'):
#     x, y, theta = config
    
#     # Rectangle parameters: width, height
#     width, height = 0.5, 0.3
    
#     # Create a rectangle patch with the specified color
#     rect = plt.Rectangle((x - width / 2, y - height / 2), width, height, angle=np.degrees(theta),
#                          edgecolor=rectangle_color, facecolor='none', lw=2)
#     ax.add_patch(rect)

#     # We are no longer plotting a dot in the center for the freeBody robot

# def visualize_robot(config, robot_type, ax, link1_color='gray', link2_color='gray', base_joint_color='gray', middle_joint_color='gray', end_effector_color='gray'):
#     if robot_type == 'arm':
#         visualize_arm_robot(config, ax, link1_color, link2_color, base_joint_color, middle_joint_color, end_effector_color)
#     elif robot_type == 'freeBody':
#         # Use link1_color for the rectangle color in freeBody visualization
#         visualize_freebody_robot(config, ax, rectangle_color=link1_color)

# # Function to visualize all configurations on the same plot
# def visualize_configs(target, configs, nearest_neighbors, robot_type):
#     fig, ax = plt.subplots()

#     # Extract only the configuration part from nearest_neighbors (ignore distances)
#     nearest_configs = [config for _, config in nearest_neighbors]

#     # Plot all configurations in gray (remaining ones)
#     for config in configs:
#         if not any(np.array_equal(config, nc) for nc in nearest_configs) and not np.array_equal(config, target):
#             visualize_robot(config, robot_type, ax, link1_color='gray', link2_color='gray', 
#                             base_joint_color='gray', middle_joint_color='gray', end_effector_color='gray')

#     # Plot the nearest neighbors in blue
#     for config in nearest_configs:
#         visualize_robot(config, robot_type, ax, link1_color='blue', link2_color='blue', 
#                         base_joint_color='blue', middle_joint_color='blue', end_effector_color='blue')

#     # Plot the target configuration in green
#     visualize_robot(target, robot_type, ax, link1_color='green', link2_color='green', 
#                     base_joint_color='green', middle_joint_color='green', end_effector_color='green')

#     ax.set_aspect('equal')
#     plt.show()

def visualize_freebody_robot(config, ax, rectangle_color='gray'):
    x, y, theta = config
    
    # Rectangle parameters: width, height
    width, height = 0.5, 0.3
    
    # Create a rectangle patch with the specified color
    rect = plt.Rectangle((x - width / 2, y - height / 2), width, height, angle=np.degrees(theta),
                         edgecolor=rectangle_color, facecolor='none', lw=2)
    ax.add_patch(rect)

def visualize_robot(config, robot_type, ax, link1_color='gray', link2_color='gray', base_joint_color='gray', middle_joint_color='gray', end_effector_color='gray'):
    if robot_type == 'arm':
        visualize_arm_robot(config, ax, link1_color, link2_color, base_joint_color, middle_joint_color, end_effector_color)
    elif robot_type == 'freeBody':
        # Use link1_color for the rectangle color in freeBody visualization
        visualize_freebody_robot(config, ax, rectangle_color=link1_color)

def visualize_configs(target, configs, nearest_neighbors, robot_type):
    fig, ax = plt.subplots()

    # Set plot limits based on robot type
    if robot_type == 'freeBody':
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 20)
    elif robot_type == 'arm':
        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)

    nearest_configs = [config for _, config in nearest_neighbors]

    # Plot all configurations
    for config in configs:
        if not any(np.array_equal(config, nc) for nc in nearest_configs) and not np.array_equal(config, target):
            visualize_robot(config, robot_type, ax, link1_color='gray')

    # Plot nearest neighbors
    for config in nearest_configs:
        visualize_robot(config, robot_type, ax, link1_color='blue')

    # Plot the target configuration
    visualize_robot(target, robot_type, ax, link1_color='green')

    ax.set_aspect('equal')
    plt.show()

# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Nearest Neighbors with Linear Search")
    
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'],
                        help="Type of robot: 'arm' or 'freeBody'")
    parser.add_argument('--target', type=float, nargs='+', required=True,
                        help="Target configuration for the robot")
    parser.add_argument('-k', type=int, required=True,
                        help="Number of nearest neighbors to find")
    parser.add_argument('--configs', type=str, required=True,
                        help="File containing the configurations")

    return parser.parse_args()

def main():
    args = parse_args()

    # Load configurations from file
    configs = load_configs(args.configs)
    target = np.array(args.target)

    # Compute distances from target to each configuration
    distances = []
    for config in configs:
        dist = compute_distance(config, target, args.robot)
        distances.append((dist, config))
    
    # Sort configurations by distance and select the k nearest neighbors
    distances.sort(key=lambda x: x[0])
    nearest_neighbors = distances[:args.k]

    # Output the nearest neighbors
    print(f"Target Configuration: {target}")
    print(f"Nearest {args.k} Neighbors:")
    for dist, config in nearest_neighbors:
        print(f"Config: {config}, Distance: {dist}")

    # Visualize the target, nearest neighbors, and the rest of the configurations
    visualize_configs(target, configs, nearest_neighbors, args.robot)

if __name__ == '__main__':
    main()
