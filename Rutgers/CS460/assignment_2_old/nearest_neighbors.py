import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to compute the Euclidean distance between two configurations
def euclidean_distance(config1, config2, robot_type):
    if robot_type == 'arm':
        # For arm robot, config1 and config2 are lists of joint angles [theta1, theta2]
        return np.linalg.norm(np.array(config1) - np.array(config2))
    
    elif robot_type == 'freeBody':
        # For freeBody robot, config1 and config2 are lists [x, y, orientation]
        return np.linalg.norm(np.array(config1) - np.array(config2))

# Parse the configurations from the file (.txt format)
def load_configurations(filename):
    configurations = []
    with open(filename, 'r') as file:
        for line in file:
            values = list(map(float, line.strip().split()))
            configurations.append(values)
    return configurations

# Nearest neighbor search using linear search
def find_nearest_neighbors(target, configurations, k, robot_type):
    distances = []
    
    # Calculate distance from target to each configuration
    for config in configurations:
        dist = euclidean_distance(target, config, robot_type)
        distances.append((config, dist))
    
    # Sort by distance and return the k-nearest configurations
    distances.sort(key=lambda x: x[1])
    return [config for config, dist in distances[:k + 1]]  # k + 1 positions

# Visualization for the arm robot with color-coding for k-nearest, target, and others
def visualize_arm(configurations, neighbors, target):
    fig, ax = plt.subplots()

    link1_length = 2  # Length of the first arm link
    link2_length = 1.5  # Length of the second arm link
    
    # Plot all M configurations in gray
    for config in configurations:
        theta1, theta2 = config
        
        # Compute joint positions based on the angles
        joint1_x = link1_length * np.cos(np.radians(theta1))
        joint1_y = link1_length * np.sin(np.radians(theta1))
        
        joint2_x = joint1_x + link2_length * np.cos(np.radians(theta1 + theta2))
        joint2_y = joint1_y + link2_length * np.sin(np.radians(theta1 + theta2))

        # Draw all configurations (in gray)
        ax.plot([0, joint1_x], [0, joint1_y], 'gray', linewidth=1, alpha=0.5)  # First link
        ax.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'gray', linewidth=1, alpha=0.5)  # Second link
        ax.scatter([joint1_x, joint2_x], [joint1_y, joint2_y], color='gray', alpha=0.5)  # Joints
    
    # Highlight the k-nearest neighbors (in blue)
    for config in neighbors:
        theta1, theta2 = config
        
        # Compute joint positions based on the angles
        joint1_x = link1_length * np.cos(np.radians(theta1))
        joint1_y = link1_length * np.sin(np.radians(theta1))
        
        joint2_x = joint1_x + link2_length * np.cos(np.radians(theta1 + theta2))
        joint2_y = joint1_y + link2_length * np.sin(np.radians(theta1 + theta2))

        # Draw the k-nearest neighbors
        ax.plot([0, joint1_x], [0, joint1_y], 'blue', linewidth=3)  # First link
        ax.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'blue', linewidth=3)  # Second link
        ax.scatter([joint1_x, joint2_x], [joint1_y, joint2_y], color='blue')  # Joints
    
    # Highlight the target configuration (in green)
    theta1, theta2 = target
    joint1_x = link1_length * np.cos(np.radians(theta1))
    joint1_y = link1_length * np.sin(np.radians(theta1))
    
    joint2_x = joint1_x + link2_length * np.cos(np.radians(theta1 + theta2))
    joint2_y = joint1_y + link2_length * np.sin(np.radians(theta1 + theta2))
    
    ax.plot([0, joint1_x], [0, joint1_y], 'green', linewidth=3, label='Target')  # Green for the target
    ax.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'green', linewidth=3)
    ax.scatter([joint1_x, joint2_x], [joint1_y, joint2_y], color='green')  # Joints for the target

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(f"Arm Robot: All M Configurations and {len(neighbors)} Nearest Neighbors + Target")
    plt.show()


# Visualization for the freeBody robot with color-coding for k-nearest, target, and others
def visualize_freebody(configurations, neighbors, target):
    fig, ax = plt.subplots()

    # Plot all M configurations in gray
    for config in configurations:
        x, y, orientation = config
        
        # Create a rectangle representing the robot
        rect = patches.Rectangle((x - 0.5, y - 0.3), 1, 0.6, angle=orientation, color='gray', alpha=0.5)
        ax.add_patch(rect)

    # Highlight the k-nearest neighbors (in blue)
    for config in neighbors:
        x, y, orientation = config
        rect = patches.Rectangle((x - 0.5, y - 0.3), 1, 0.6, angle=orientation, color='blue', alpha=0.7)
        ax.add_patch(rect)

    # Highlight the target configuration (in green)
    x, y, orientation = target
    rect = patches.Rectangle((x - 0.5, y - 0.3), 1, 0.6, angle=orientation, color='green', alpha=0.7, label='Target')
    ax.add_patch(rect)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title(f"FreeBody Robot: All M Configurations and {len(neighbors)} Nearest Neighbors + Target")
    plt.show()


# Main function
# Main function
def main():
    parser = argparse.ArgumentParser(description='Nearest neighbors with linear search and visualization.')
    
    # Define the required command-line arguments
    parser.add_argument('--robot', required=True, choices=['arm', 'freeBody'], help='Type of robot (arm or freeBody)')
    parser.add_argument('--target', required=True, nargs='+', type=float, help='Target configuration')
    parser.add_argument('-k', required=True, type=int, help='Number of nearest neighbors to find')
    parser.add_argument('--configs', required=True, type=str, help='.txt file containing robot configurations')
    
    # Parse arguments
    args = parser.parse_args()

    # Load configurations from the file
    configurations = load_configurations(args.configs)

    # Find the nearest neighbors
    neighbors = find_nearest_neighbors(args.target, configurations, args.k, args.robot)

    # Visualize based on robot type
    if args.robot == 'arm':
        print(f"Target Configuration (arm): {args.target}")
        print(f"{args.k} Nearest Neighbors (arm):")
        visualize_arm(configurations, neighbors, args.target)
    
    elif args.robot == 'freeBody':
        print(f"Target Configuration (freeBody): {args.target}")
        print(f"{args.k} Nearest Neighbors (freeBody):")
        visualize_freebody(configurations, neighbors, args.target)

if __name__ == '__main__':
    main()

