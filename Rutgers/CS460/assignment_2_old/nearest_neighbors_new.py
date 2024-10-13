import argparse
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Normalize angle between 0 and 360
def normalize_angle(angle):
    return angle % 360

# Function to calculate Euclidean distance for arm robot (2D) with weighting
def distance_arm(config1, config2, weights=(1.0, 1.0)):
    joint1_diff = normalize_angle(config1[0]) - normalize_angle(config2[0])
    joint2_diff = normalize_angle(config1[1]) - normalize_angle(config2[1])
    return math.sqrt(weights[0] * joint1_diff**2 + weights[1] * joint2_diff**2)

# Function to calculate the corners of the freeBody rectangle based on position and orientation
def get_rectangle_corners(x, y, orientation, width=1.0, height=0.5):
    # Convert orientation to radians
    angle_rad = math.radians(orientation)
    
    # Calculate the four corners of the rectangle (relative to the center)
    corners = [
        (-width / 2, -height / 2), (width / 2, -height / 2),  # Bottom-left, bottom-right
        (width / 2, height / 2), (-width / 2, height / 2)     # Top-right, top-left
    ]
    
    # Rotate and translate each corner based on the orientation and position
    rotated_corners = [
        (
            x + corner[0] * math.cos(angle_rad) - corner[1] * math.sin(angle_rad),
            y + corner[0] * math.sin(angle_rad) + corner[1] * math.cos(angle_rad)
        )
        for corner in corners
    ]
    
    return rotated_corners

# Function to calculate minimum distance between two rectangles based on their corners
def min_distance_between_rectangles(corners1, corners2):
    min_distance = float('inf')
    for x1, y1 in corners1:
        for x2, y2 in corners2:
            dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if dist < min_distance:
                min_distance = dist
    return min_distance

# Function to calculate distance between two freeBody robot configurations
def distance_freeBody(config1, config2):
    # Get the corners of both rectangles
    corners1 = get_rectangle_corners(config1[0], config1[1], config1[2])
    corners2 = get_rectangle_corners(config2[0], config2[1], config2[2])
    
    # Calculate the minimum distance between the rectangles
    return min_distance_between_rectangles(corners1, corners2)

# Function to parse configurations from a file
def load_configs(filename, robot_type):
    configs = []
    with open(filename, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split()))
            if robot_type == 'arm' and len(parts) == 2:
                configs.append(tuple(parts))
            elif robot_type == 'freeBody' and len(parts) == 3:
                configs.append(tuple(parts))
    return configs

# Function to find k nearest neighbors using linear search for both robots
def find_nearest_neighbors_linear(robot_type, target, configs, k):
    if robot_type == 'arm':
        weights = (1.0, 1.0)  # Weights for arm robot
        distances = [(config, distance_arm(config, target, weights)) for config in configs]
    elif robot_type == 'freeBody':
        distances = [(config, distance_freeBody(config, target)) for config in configs]
    
    # Sort configurations by distance and select the k nearest
    distances.sort(key=lambda x: x[1])
    return [config for config, _ in distances[:k]]

# Function to draw the arm robot as a two-joint arm
def draw_arm_robot(config, ax, color='blue'):
    joint1_length = 1.0  # Length of the first joint
    joint2_length = 1.0  # Length of the second joint

    # Joint 1 coordinates
    joint1_x = joint1_length * math.cos(math.radians(config[0]))
    joint1_y = joint1_length * math.sin(math.radians(config[0]))

    # Joint 2 coordinates (relative to joint 1)
    joint2_x = joint1_x + joint2_length * math.cos(math.radians(config[0] + config[1]))
    joint2_y = joint1_y + joint2_length * math.sin(math.radians(config[0] + config[1]))

    # Draw arm links
    ax.plot([0, joint1_x], [0, joint1_y], color=color, lw=2)  # Base to joint 1
    ax.plot([joint1_x, joint2_x], [joint1_y, joint2_y], color=color, lw=2)  # Joint 1 to joint 2

    # Draw joints as circles
    ax.scatter([0, joint1_x, joint2_x], [0, joint1_y, joint2_y], color=color, s=100, zorder=5)

# Function to draw the freeBody robot as a rectangle
def draw_freeBody_robot(config, ax, color='blue'):
    width, height = 1.0, 0.5  # Dimensions of the freeBody robot

    # Create a rectangle for the robot with the correct orientation
    rect = patches.Rectangle(
        (config[0] - width / 2, config[1] - height / 2),  # Position (x, y)
        width, height, angle=config[2],  # Orientation (angle in degrees)
        edgecolor=color, facecolor=color, alpha=0.5, zorder=5
    )
    ax.add_patch(rect)

# Function to visualize all configurations with target and nearest neighbors highlighted
def visualize_all_configs(target, nearest_neighbors, all_configs, robot_type):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all configurations in gray
    for config in all_configs:
        if robot_type == 'arm':
            draw_arm_robot(config, ax, color='gray')
        elif robot_type == 'freeBody':
            draw_freeBody_robot(config, ax, color='gray')

    # Plot nearest neighbors in blue
    for neighbor in nearest_neighbors:
        if robot_type == 'arm':
            draw_arm_robot(neighbor, ax, color='blue')
        elif robot_type == 'freeBody':
            draw_freeBody_robot(neighbor, ax, color='blue')

    # Plot target configuration in green
    if robot_type == 'arm':
        draw_arm_robot(target, ax, color='green')
    elif robot_type == 'freeBody':
        draw_freeBody_robot(target, ax, color='green')

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Target and {len(nearest_neighbors)} Nearest Neighbors")
    plt.xlabel("x / Joint 1 Angle (degrees)" if robot_type == 'arm' else "x position")
    plt.ylabel("y / Joint 2 Angle (degrees)" if robot_type == 'arm' else "y position")
    plt.show()

# Main function to handle argument parsing and program execution
def main():
    parser = argparse.ArgumentParser(description="Find nearest neighbors for robot configurations.")
    parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
    parser.add_argument('--target', nargs='+', type=float, required=True, help='Target configuration (N=2 for arm, N=3 for freeBody)')
    parser.add_argument('-k', type=int, required=True, help='Number of nearest neighbors to find')
    parser.add_argument('--configs', type=str, required=True, help='File containing configurations')
    
    args = parser.parse_args()

    # Validate target configuration length
    if args.robot == 'arm' and len(args.target) != 2:
        parser.error("Arm robot requires exactly 2 target configuration values (joint angles).")
    elif args.robot == 'freeBody' and len(args.target) != 3:
        parser.error("FreeBody robot requires exactly 3 target configuration values (x, y, orientation).")

    # Load configurations from file
    all_configs = load_configs(args.configs, args.robot)

    # Find the nearest neighbors using the improved linear search
    nearest_neighbors = find_nearest_neighbors_linear(args.robot, tuple(args.target), all_configs, args.k)

    # Visualize all configurations with target and nearest neighbors highlighted
    visualize_all_configs(tuple(args.target), nearest_neighbors, all_configs, args.robot)

if __name__ == "__main__":
    main()
