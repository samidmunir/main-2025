import argparse
import json
import time
import numpy as NP
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS

# Function to compute the Euclidean distance between two configurations.
def euclidean_distance(config1, config2, robot_type):
    print(config1)
    if robot_type == 'arm':
        # For arm robot, config1 and config2 are lists of joint angles [theta1, theta2]
        return NP.linalg.norm(NP.array(config1) - NP.array(config2))
    
    elif robot_type == 'freeBody':
        # For freeBody robot, config1 and config2 are dictionaries with x, y, and orientation
        pos1 = NP.array([config1['center'][0], config1['center'][1], config1['rotation']])
        pos2 = NP.array([config2['center'][0], config2['center'][1], config2['rotation']])
        return NP.linalg.norm(pos1 - pos2)

# Parse the configurations from the file.
def load_configurations(filename):
    with open(filename, 'r') as file:
        configurations = json.load(file)
    
    return configurations

# Nearest neighbor search using linear search.
def find_nearest_neighbors(target, configurations, k, robot_type):
    distances = []
    
    # Calculate distance from target to each configuration.
    for config in configurations:
        dist  = euclidean_distance(target, config, robot_type)
        distances.append((config, dist))
        
    # Sort by distance and return the k-nearest configurations.
    distances.sort(key=lambda x: x[1])
    
    return [config for config, dist in distances[:k + 1]]

# Function to visualize the arm robot.
def visualize_arm_robot(configurations):
    fig, ax = PLT.subplots()

    link1_length = 2.0 # length of the first arm link.
    link2_length = 1.5 # length of the second arm link.

    for config in configurations:
        theta1, theta2 = config
        
        # Compute joint positions based on the angles
        joint1_x = link1_length * NP.cos(NP.radians(theta1))
        joint1_y = link1_length * NP.sin(NP.radians(theta1))
        
        joint2_x = link2_length * NP.cos(NP.radians(theta1 + theta2))
        joint2_y = link2_length * NP.sin(NP.radians(theta1 + theta2))
        
        # Draw the arm.
        ax.plot([0, joint1_x], [0, joint1_y], 'b-', linewidth = 3) # first link.
        ax.plot([joint1_x, joint2_x], [joint1_y, joint2_y], 'r-', linewidth = 3) # second link.
        ax.scatter([joint1_x, joint2_x], [joint1_y, joint2_y], color = 'black') # joints
    
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    PLT.gca().set_aspect('equal', adjustable = 'box')
    PLT.title(f'Arm Robot: {len(configurations)} positions')
    PLT.show()

        
# TODO: Function to visualize the freeBody robot.

# Main function.
def main():
    parser = argparse.ArgumentParser(description = 'Nearest neighbors with linear search.')
    
    # Define the required command-line arguments.
    parser.add_argument('--robot', required = True, choices = ['arm', 'freeBody'], help = 'Type of robot (arm or freeBody).')
    parser.add_argument('--target', required = True, nargs = '+', type = float, help = 'Target configration')
    parser.add_argument('--k', required = True, type = int, help = 'Number of nearest neighbors to find.')
    parser.add_argument('--configs', required = True, type = str, help = 'File containing the robot configurations.')
    
    # Parse arguments.
    args = parser.parse_args()
    
    # Load the configurations from the file.
    configurations = load_configurations(args.configs)
    
    # Find the nearest neighbors.
    neighbors = find_nearest_neighbors(args.target, configurations, args.k, args.robot)
    
    # Output the results.
    if args.robot == 'arm':
        print(f'Target configuration: {args.target}')
        print(f'{args.k + 1} Nearest Neighbors (arm):')
        for neighbor in neighbors:
            print(neighbor)
        visualize_arm_robot(neighbors)
    elif args.robot == 'freeBody':
        pass
        
if __name__ == '__main__':
    main()