import argparse
import json
import random
import time
import matplotlib.pyplot as PLT
import numpy as NP

# Function to check if two rectangles (obstacles and robot) collide.
def is_collision(obstacle, robot):
    # Extract center and dimensions of both the obstacle and the robot.
    x1, y1 = obstacle['center']
    w1, h1 = obstacle['width'], obstacle['height']
    
    x2, y2 = robot['center']
    w2, h2 = robot['width'], robot['height']
    
    # Check for collision using axis-aligned bounding box (AABB) mehthod.
    if (abs(x1 - x2) * 2 < (w1 + w2)) and (abs(y1 - y2) * 2 < (h1 + h2)):
        return True
    return False

# Function to visualize the environment and color obstacles based on collision status.
def visualize_scene(environment, robot, colliding_obstacles):
    fig, ax = PLT.subplots()
    
    # Plot each obstacle, coloring it red if it collides, green otherwise.
    for i, obstacle in enumerate(environment):
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        color = '#ff0000' if i in colliding_obstacles else '#00ff00'
        obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, color = color, alpha = 0.5)
        ax.add_patch(obstacle_rectangle)
    
    # Plot the robot.
    rx, ry = robot['center']
    rwidth, rheight = robot['width'], robot['height']
    robot_rectangle = PLT.Rectangle((rx - rwidth / 2, ry - rheight / 2), rwidth, rheight, color = '#0000ff', alpha = 0.7)
    ax.add_patch(robot_rectangle)
    
    # Set limits for the 20x20 environment.
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    PLT.gca().set_aspect('equal', adjustable = 'box')
    PLT.show()
    
# Function to simulate the robot moving through the environment and checking for collisions.
def simulate_robot(environment):
    robot_size = {'width': 0.5, 'height': 0.3}
    
    for _ in range(10): # run for 10 seconds.
        # Randomly place the robot in the environment.
        rx = random.uniform(0, 20)
        ry = random.uniform(0, 20)
        robot = {'center': (rx, ry), 'width': robot_size['width'], 'height': robot_size['height']}
        
        # Check for collisions with each obstacle.
        colliding_obstacles = []
        for i, obstacle in enumerate(environment):
            if is_collision(obstacle, robot):
                colliding_obstacles.append(i)
        
        # Visualize the environment with colored obstacles based on collisoin status.
        visualize_scene(environment, robot, colliding_obstacles)
        
        # Wait for 1 second before spawning the robot in a new random position.
        time.sleep(1)
        
# Load the environment from a file.
def scene_from_file(filename):
    with open(filename, 'r') as file:
        environment = json.load(file)
    
    return environment

# Main function.
def main():
    parser = argparse.ArgumentParser(description = 'Collision Checking for Robots in an Environment.')
    
    # Define the required command-line arguments.
    parser.add_argument('--robot', required = True, choices = ['arm', 'freeBody'], help = 'Type of robot (arm or freeBody).')
    parser.add_argument('--map', required = True, type = str, help = 'File containing the environment')
    
    # Parse arguments.
    args = parser.parse_args()
    
    # Load the environment.
    environment = scene_from_file(args.map)
    
    # Simulate the robot and check for collisions.
    simulate_robot(environment)

if __name__ == '__main__':
    main()