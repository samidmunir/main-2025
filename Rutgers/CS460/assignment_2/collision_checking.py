# import argparse
# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Function to load environment map
# def load_map(filename):
#     environment = {"obstacles": []}
#     with open(filename, 'r') as file:
#         width, height = map(float, file.readline().split())
#         environment["width"] = width
#         environment["height"] = height
        
#         for line in file:
#             x, y, w, h, orientation = map(float, line.split())
#             obstacle = {
#                 "center": (x, y),
#                 "width": w,
#                 "height": h,
#                 "orientation": orientation
#             }
#             environment["obstacles"].append(obstacle)
    
#     return environment

# # Function to generate random poses
# def generate_random_pose():
#     x = np.random.uniform(0, 20)
#     y = np.random.uniform(0, 20)
#     theta = np.random.uniform(-np.pi, np.pi)
#     return (x, y, theta)

# # Function to check if a rectangle (robot) collides with any obstacles
# def check_collision(robot, obstacles):
#     for obstacle in obstacles:
#         # Simple collision check logic (bounding box overlap or similar)
#         # We can use matplotlib's patches for more advanced collision detection if needed
#         pass
#     return False  # Return True if collision happens

# # Function to visualize the environment and the robot
# def visualize_environment(environment, robot_pose, colliding_obstacles):
#     fig, ax = plt.subplots()

#     # Plot obstacles
#     for obstacle in environment["obstacles"]:
#         color = 'green' if obstacle not in colliding_obstacles else 'red'
#         rect = patches.Rectangle((obstacle['center'][0] - obstacle['width'] / 2, 
#                                   obstacle['center'][1] - obstacle['height'] / 2),
#                                   obstacle['width'], obstacle['height'],
#                                   angle=np.degrees(obstacle['orientation']),
#                                   edgecolor=color, facecolor='none', lw=2)
#         ax.add_patch(rect)

#     # Plot the robot (freeBody as a rectangle)
#     robot = patches.Rectangle((robot_pose[0] - 0.25, robot_pose[1] - 0.15), 0.5, 0.3,
#                               angle=np.degrees(robot_pose[2]), edgecolor='blue', facecolor='none', lw=2)
#     ax.add_patch(robot)

#     ax.set_xlim(0, 20)
#     ax.set_ylim(0, 20)
#     plt.show()

# def main():
#     parser = argparse.ArgumentParser(description="Collision Checking")
#     parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'],
#                         help="Type of robot: 'arm' or 'freeBody'")
#     parser.add_argument('--map', type=str, required=True,
#                         help="Path to the map file")
#     args = parser.parse_args()

#     # Load the environment
#     environment = load_map(args.map)

#     for _ in range(10):  # Spawn robot for 10 seconds (10 poses)
#         robot_pose = generate_random_pose()

#         # Check for collisions
#         colliding_obstacles = []
#         if check_collision(robot_pose, environment["obstacles"]):
#             colliding_obstacles = [obstacle for obstacle in environment["obstacles"] if check_collision(robot_pose, [obstacle])]

#         # Visualize environment and robot
#         visualize_environment(environment, robot_pose, colliding_obstacles)
        
#         time.sleep(1)  # Wait 1 second before spawning the robot again

# if __name__ == "__main__":
#     main()

import argparse
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon
from shapely.affinity import rotate

# Function to load environment map
def load_map(filename):
    environment = {"obstacles": []}
    with open(filename, 'r') as file:
        width, height = map(float, file.readline().split())
        environment["width"] = width
        environment["height"] = height
        
        for line in file:
            x, y, w, h, orientation = map(float, line.split())
            obstacle = {
                "center": (x, y),
                "width": w,
                "height": h,
                "orientation": orientation
            }
            environment["obstacles"].append(obstacle)
    
    return environment

# Function to generate random poses for the robot
def generate_random_pose():
    x = np.random.uniform(0, 20)
    y = np.random.uniform(0, 20)
    theta = np.random.uniform(-np.pi, np.pi)
    return (x, y, theta)

# Function to create a rectangle polygon from the center, width, height, and orientation
def create_rectangle(x, y, width, height, orientation):
    # Create an axis-aligned rectangle at (x, y) with the given width and height
    rect = Polygon([(-width / 2, -height / 2), (width / 2, -height / 2),
                    (width / 2, height / 2), (-width / 2, height / 2)])
    
    # Rotate the rectangle around its center and translate it to (x, y)
    rotated_rect = rotate(rect, np.degrees(orientation), origin=(0, 0))
    rotated_rect = Polygon([(point[0] + x, point[1] + y) for point in rotated_rect.exterior.coords])
    return rotated_rect

# Function to check if the robot collides with any obstacles
def check_collision(robot_pose, obstacles):
    robot = create_rectangle(robot_pose[0], robot_pose[1], 0.5, 0.3, robot_pose[2])

    for obstacle in obstacles:
        obstacle_rect = create_rectangle(obstacle["center"][0], obstacle["center"][1],
                                         obstacle["width"], obstacle["height"], obstacle["orientation"])
        # Check if the robot rectangle intersects the obstacle
        if robot.intersects(obstacle_rect):
            return True

    return False  # No collisions detected

# Function to visualize the environment and the robot
def visualize_environment(environment, robot_pose, colliding_obstacles):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in environment["obstacles"]:
        color = 'green' if obstacle not in colliding_obstacles else 'red'
        rect = patches.Rectangle((obstacle['center'][0] - obstacle['width'] / 2, 
                                  obstacle['center'][1] - obstacle['height'] / 2),
                                  obstacle['width'], obstacle['height'],
                                  angle=np.degrees(obstacle['orientation']),
                                  edgecolor=color, facecolor='none', lw=2)
        ax.add_patch(rect)

    # Plot the robot (freeBody as a rectangle)
    robot = patches.Rectangle((robot_pose[0] - 0.25, robot_pose[1] - 0.15), 0.5, 0.3,
                              angle=np.degrees(robot_pose[2]), edgecolor='blue', facecolor='none', lw=2)
    ax.add_patch(robot)

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.show()

# Main function
def main():
    parser = argparse.ArgumentParser(description="Collision Checking")
    parser.add_argument('--robot', type=str, required=True, choices=['arm', 'freeBody'],
                        help="Type of robot: 'arm' or 'freeBody'")
    parser.add_argument('--map', type=str, required=True,
                        help="Path to the map file")
    args = parser.parse_args()

    # Load the environment
    environment = load_map(args.map)

    for _ in range(10):  # Spawn robot for 10 seconds (10 poses)
        robot_pose = generate_random_pose()

        # Check for collisions
        colliding_obstacles = []
        if check_collision(robot_pose, environment["obstacles"]):
            colliding_obstacles = [obstacle for obstacle in environment["obstacles"] if check_collision(robot_pose, [obstacle])]

        # Visualize environment and robot
        visualize_environment(environment, robot_pose, colliding_obstacles)
        
        time.sleep(1)  # Wait 1 second before spawning the robot again

if __name__ == "__main__":
    main()
