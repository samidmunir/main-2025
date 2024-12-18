# # import argparse
# # import random
# # import math
# # import heapq
# # import matplotlib.pyplot as plt
# # import matplotlib.patches as patches

# # # Normalize angle between 0 and 360
# # def normalize_angle(angle):
# #     return angle % 360

# # # Function to generate random samples in the environment
# # def generate_samples(num_samples, width, height, obstacle_check_fn):
# #     samples = []
# #     for _ in range(num_samples):
# #         while True:
# #             x = random.uniform(0, width)
# #             y = random.uniform(0, height)
# #             orientation = random.uniform(0, 360)
# #             if not obstacle_check_fn((x, y, orientation)):
# #                 samples.append((x, y, orientation))
# #                 break
# #     return samples

# # # Check for collision between a robot and an obstacle
# # def is_collision(config, obstacles):
# #     x, y, _ = config
# #     for obstacle in obstacles:
# #         obs_x, obs_y, obs_width, obs_height = obstacle
# #         if obs_x - obs_width / 2 <= x <= obs_x + obs_width / 2 and obs_y - obs_height / 2 <= y <= obs_y + obs_height / 2:
# #             return True
# #     return False

# # # Get k-nearest neighbors for a given configuration
# # def get_k_nearest_neighbors(config, samples, k):
# #     distances = [(other, math.sqrt((config[0] - other[0]) ** 2 + (config[1] - other[1]) ** 2)) for other in samples]
# #     distances.sort(key=lambda x: x[1])
# #     return [neighbor for neighbor, _ in distances[:k]]

# # # Build the PRM roadmap by connecting samples to their k-nearest neighbors
# # def build_roadmap(samples, obstacles, k, obstacle_check_fn):
# #     roadmap = {sample: [] for sample in samples}
# #     for sample in samples:
# #         neighbors = get_k_nearest_neighbors(sample, samples, k)
# #         for neighbor in neighbors:
# #             if not obstacle_check_fn((sample, neighbor)):
# #                 roadmap[sample].append(neighbor)
# #                 roadmap[neighbor].append(sample)  # Bidirectional connection
# #     return roadmap

# # # Heuristic function for A* search (Euclidean distance)
# # def heuristic(a, b):
# #     return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# # # A* search algorithm to find the shortest path in the roadmap
# # def astar(roadmap, start, goal):
# #     open_list = []
# #     heapq.heappush(open_list, (0, start))
# #     came_from = {start: None}
# #     g_score = {sample: float('inf') for sample in roadmap}
# #     g_score[start] = 0
# #     f_score = {sample: float('inf') for sample in roadmap}
# #     f_score[start] = heuristic(start, goal)

# #     while open_list:
# #         _, current = heapq.heappop(open_list)

# #         if current == goal:
# #             path = []
# #             while current:
# #                 path.append(current)
# #                 current = came_from[current]
# #             return path[::-1]  # Return reversed path

# #         for neighbor in roadmap[current]:
# #             tentative_g_score = g_score[current] + heuristic(current, neighbor)

# #             if tentative_g_score < g_score[neighbor]:
# #                 came_from[neighbor] = current
# #                 g_score[neighbor] = tentative_g_score
# #                 f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
# #                 heapq.heappush(open_list, (f_score[neighbor], neighbor))

# #     return None  # No path found

# # def visualize_roadmap(samples, obstacles, roadmap, path=None, robot_type='freeBody'):
# #     fig, ax = plt.subplots()
    
# #     # Plot obstacles
# #     for obstacle in obstacles:
# #         obs_x, obs_y, obs_width, obs_height = obstacle
# #         rect = patches.Rectangle(
# #             (obs_x - obs_width / 2, obs_y - obs_height / 2),
# #             obs_width, obs_height, edgecolor='black', facecolor='gray'
# #         )
# #         ax.add_patch(rect)

# #     # Plot roadmap (nodes and edges)
# #     for sample in roadmap:
# #         if robot_type == 'arm':
# #             x, y = sample  # Only two values for the arm robot
# #         else:
# #             x, y, _ = sample  # Three values for the freeBody robot
# #         ax.scatter(x, y, color='blue', s=10)
        
# #         for neighbor in roadmap[sample]:
# #             if robot_type == 'arm':
# #                 nx, ny = neighbor  # Only two values for the arm robot
# #             else:
# #                 nx, ny, _ = neighbor  # Three values for the freeBody robot
# #             ax.plot([x, nx], [y, ny], color='blue', lw=0.5)

# #     # Plot the path if available
# #     if path:
# #         for i in range(len(path) - 1):
# #             if robot_type == 'arm':
# #                 x1, y1 = path[i]
# #                 x2, y2 = path[i + 1]
# #             else:
# #                 x1, y1, _ = path[i]
# #                 x2, y2, _ = path[i + 1]
# #             ax.plot([x1, x2], [y1, y2], color='red', lw=2)

# #     ax.set_xlim(0, 20)
# #     ax.set_ylim(0, 20)
# #     plt.gca().set_aspect('equal', adjustable='box')
# #     plt.show()


# # # Load obstacles from the map file
# # def load_obstacles(filename):
# #     obstacles = []
# #     with open(filename, 'r') as file:
# #         for line in file:
# #             parts = list(map(float, line.strip().split(',')))
# #             if len(parts) == 5:
# #                 x, y, width, height, orientation = parts
# #                 # For simplicity, we ignore orientation for now in the obstacle definition
# #                 obstacles.append((x, y, width, height))
# #     return obstacles

# # # Main function to parse input arguments and run the PRM algorithm
# # def main():
# #     parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
# #     parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
# #     parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation)')
# #     parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation)')
# #     parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
# #     args = parser.parse_args()

# #     # Load obstacles from the map file
# #     obstacles = load_obstacles(args.map)

# #     # Generate random samples in free space
# #     num_samples = 750  # You can adjust this
# #     k = 6 # Number of nearest neighbors
# #     width, height = 20, 20  # Environment dimensions

# #     start_config = tuple(args.start)
# #     goal_config = tuple(args.goal)
    
# #     # Add start and goal to samples
# #     samples = generate_samples(num_samples, width, height, lambda config: is_collision(config, obstacles))
# #     samples.append(start_config)
# #     samples.append(goal_config)

# #     # Build the roadmap
# #     roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)  # Simplified obstacle check

# #     # Find the shortest path using A* search
# #     path = astar(roadmap, start_config, goal_config)

# #     # Visualize the roadmap and the path
# #     visualize_roadmap(samples, obstacles, roadmap, path)

# # if __name__ == "__main__":
# #     main()

# import argparse
# import random
# import math
# import heapq
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

# # Normalize angle between 0 and 360
# def normalize_angle(angle):
#     return angle % 360

# # Function to generate random samples in the environment
# def generate_samples(num_samples, width, height, obstacle_check_fn, robot_type):
#     samples = []
#     for _ in range(num_samples):
#         while True:
#             if robot_type == 'arm':
#                 joint1 = random.uniform(0, 360)
#                 joint2 = random.uniform(0, 360)
#                 config = (joint1, joint2)
#             else:  # freeBody robot
#                 x = random.uniform(0, width)
#                 y = random.uniform(0, height)
#                 orientation = random.uniform(0, 360)
#                 config = (x, y, orientation)
#             if not obstacle_check_fn(config):
#                 samples.append(config)
#                 break
#     return samples

# # Check for collision between a robot and an obstacle
# def is_collision(config, obstacles):
#     if len(config) == 3:  # freeBody robot
#         x, y, _ = config
#         for obstacle in obstacles:
#             obs_x, obs_y, obs_width, obs_height = obstacle
#             if obs_x - obs_width / 2 <= x <= obs_x + obs_width / 2 and obs_y - obs_height / 2 <= y <= obs_y + obs_height / 2:
#                 return True
#     return False

# # Get k-nearest neighbors for a given configuration
# def get_k_nearest_neighbors(config, samples, k):
#     distances = [(other, math.sqrt((config[0] - other[0]) ** 2 + (config[1] - other[1]) ** 2)) for other in samples]
#     distances.sort(key=lambda x: x[1])
#     return [neighbor for neighbor, _ in distances[:k]]

# # Build the PRM roadmap by connecting samples to their k-nearest neighbors
# def build_roadmap(samples, obstacles, k, obstacle_check_fn):
#     roadmap = {sample: [] for sample in samples}
#     for sample in samples:
#         neighbors = get_k_nearest_neighbors(sample, samples, k)
#         for neighbor in neighbors:
#             if not obstacle_check_fn((sample, neighbor)):
#                 roadmap[sample].append(neighbor)
#                 roadmap[neighbor].append(sample)  # Bidirectional connection
#     return roadmap

# # Heuristic function for A* search (Euclidean distance)
# def heuristic(a, b):
#     return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# # A* search algorithm to find the shortest path in the roadmap
# def astar(roadmap, start, goal):
#     open_list = []
#     heapq.heappush(open_list, (0, start))
#     came_from = {start: None}
#     g_score = {sample: float('inf') for sample in roadmap}
#     g_score[start] = 0
#     f_score = {sample: float('inf') for sample in roadmap}
#     f_score[start] = heuristic(start, goal)

#     while open_list:
#         _, current = heapq.heappop(open_list)

#         if current == goal:
#             path = []
#             while current:
#                 path.append(current)
#                 current = came_from[current]
#             return path[::-1]  # Return reversed path

#         for neighbor in roadmap[current]:
#             tentative_g_score = g_score[current] + heuristic(current, neighbor)

#             if tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
#                 heapq.heappush(open_list, (f_score[neighbor], neighbor))

#     return None  # No path found

# # Visualize the roadmap and the found path
# def visualize_roadmap(samples, obstacles, roadmap, path=None, robot_type='freeBody'):
#     fig, ax = plt.subplots()

#     # Plot obstacles
#     for obstacle in obstacles:
#         obs_x, obs_y, obs_width, obs_height = obstacle
#         rect = patches.Rectangle(
#             (obs_x - obs_width / 2, obs_y - obs_height / 2),
#             obs_width, obs_height, edgecolor='black', facecolor='gray'
#         )
#         ax.add_patch(rect)

#     # Plot roadmap (nodes and edges)
#     for sample in roadmap:
#         if robot_type == 'arm':
#             joint1, joint2 = sample  # Use joint angles for the arm robot
#             x, y = joint1, joint2  # For visualization, treat the angles as x, y
#         else:
#             x, y, _ = sample  # Use x, y for the freeBody robot
#         ax.scatter(x, y, color='blue', s=10)

#         for neighbor in roadmap[sample]:
#             if robot_type == 'arm':
#                 nx, ny = neighbor  # Use joint angles for the arm robot
#             else:
#                 nx, ny, _ = neighbor
#             ax.plot([x, nx], [y, ny], color='blue', lw=0.5)

#     # Plot the path if available
#     if path:
#         for i in range(len(path) - 1):
#             if robot_type == 'arm':
#                 joint1_1, joint2_1 = path[i]
#                 joint1_2, joint2_2 = path[i + 1]
#                 x1, y1 = joint1_1, joint2_1
#                 x2, y2 = joint1_2, joint2_2
#             else:
#                 x1, y1, _ = path[i]
#                 x2, y2, _ = path[i + 1]
#             ax.plot([x1, x2], [y1, y2], color='red', lw=2)

#     ax.set_xlim(0, 360)  # Arm angles are in degrees from 0 to 360
#     ax.set_ylim(0, 360)
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.show()

# # Load obstacles from the map file
# def load_obstacles(filename):
#     obstacles = []
#     with open(filename, 'r') as file:
#         for line in file:
#             parts = list(map(float, line.strip().split(',')))
#             if len(parts) == 5:
#                 x, y, width, height, orientation = parts
#                 # For simplicity, we ignore orientation for now in the obstacle definition
#                 obstacles.append((x, y, width, height))
#     return obstacles

# # Main function to parse input arguments and run the PRM algorithm
# def main():
#     parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
#     parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
#     parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation or joint1, joint2)')
#     parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation or joint1, joint2)')
#     parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
#     args = parser.parse_args()

#     # Load obstacles from the map file
#     obstacles = load_obstacles(args.map)

#     # Generate random samples in free space
#     num_samples = 100  # You can adjust this
#     k = 5  # Number of nearest neighbors
#     width, height = 20, 20  # Environment dimensions

#     start_config = tuple(args.start)
#     goal_config = tuple(args.goal)
    
#     # Add start and goal to samples
#     samples = generate_samples(num_samples, width, height, lambda config: is_collision(config, obstacles), args.robot)
#     samples.append(start_config)
#     samples.append(goal_config)

#     # Build the roadmap
#     roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)

#     # Find the shortest path using A* search
#     path = astar(roadmap, start_config, goal_config)

#     # Visualize the roadmap and the path
#     visualize_roadmap(samples, obstacles, roadmap, path, args.robot)

# if __name__ == "__main__":
#     main()

import argparse
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq

# Normalize angle between 0 and 360
def normalize_angle(angle):
    return angle % 360

# Function to calculate the end-effector position for a 2-joint arm robot
def calculate_end_effector(joint1_angle, joint2_angle, joint1_length=10.0, joint2_length=10.0):
    # Joint 1 position (fixed at base (0, 0))
    joint1_x = joint1_length * math.cos(math.radians(joint1_angle))
    joint1_y = joint1_length * math.sin(math.radians(joint1_angle))
    
    # Joint 2 position relative to Joint 1
    end_effector_x = joint1_x + joint2_length * math.cos(math.radians(joint1_angle + joint2_angle))
    end_effector_y = joint1_y + joint2_length * math.sin(math.radians(joint1_angle + joint2_angle))
    
    # Return the end-effector position in the workspace
    return end_effector_x, end_effector_y


# Function to generate random samples in the environment
def generate_samples(num_samples, width, height, obstacle_check_fn, robot_type):
    samples = []
    for _ in range(num_samples):
        while True:
            if robot_type == 'arm':
                joint1 = random.uniform(0, 360)
                joint2 = random.uniform(0, 360)
                config = (joint1, joint2)
            else:  # freeBody robot
                x = random.uniform(0, width)
                y = random.uniform(0, height)
                orientation = random.uniform(0, 360)
                config = (x, y, orientation)
            if not obstacle_check_fn(config):
                samples.append(config)
                break
    return samples

# Check for collision between a robot and an obstacle
def is_collision(config, obstacles):
    if len(config) == 3:  # freeBody robot
        x, y, _ = config
        for obstacle in obstacles:
            obs_x, obs_y, obs_width, obs_height = obstacle
            if obs_x - obs_width / 2 <= x <= obs_x + obs_width / 2 and obs_y - obs_height / 2 <= y <= obs_y + obs_height / 2:
                return True
    return False

# Get k-nearest neighbors for a given configuration
def get_k_nearest_neighbors(config, samples, k):
    distances = [(other, math.sqrt((config[0] - other[0]) ** 2 + (config[1] - other[1]) ** 2)) for other in samples]
    distances.sort(key=lambda x: x[1])
    return [neighbor for neighbor, _ in distances[:k]]

# Build the PRM roadmap by connecting samples to their k-nearest neighbors
def build_roadmap(samples, obstacles, k, obstacle_check_fn):
    roadmap = {sample: [] for sample in samples}
    for sample in samples:
        neighbors = get_k_nearest_neighbors(sample, samples, k)
        for neighbor in neighbors:
            if not obstacle_check_fn((sample, neighbor)):
                roadmap[sample].append(neighbor)
                roadmap[neighbor].append(sample)  # Bidirectional connection
    return roadmap

# Heuristic function for A* search (Euclidean distance)
def heuristic(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

# A* search algorithm to find the shortest path in the roadmap
def astar(roadmap, start, goal):
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: None}
    g_score = {sample: float('inf') for sample in roadmap}
    g_score[start] = 0
    f_score = {sample: float('inf') for sample in roadmap}
    f_score[start] = heuristic(start, goal)

    while open_list:
        _, current = heapq.heappop(open_list)

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]  # Return reversed path

        for neighbor in roadmap[current]:
            tentative_g_score = g_score[current] + heuristic(current, neighbor)

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal)
                heapq.heappush(open_list, (f_score[neighbor], neighbor))

    return None  # No path found

# Visualize the roadmap and the found path
def visualize_roadmap(samples, obstacles, roadmap, path=None, robot_type='freeBody'):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        obs_x, obs_y, obs_width, obs_height = obstacle
        rect = patches.Rectangle(
            (obs_x - obs_width / 2, obs_y - obs_height / 2),
            obs_width, obs_height, edgecolor='black', facecolor='gray'
        )
        ax.add_patch(rect)

    # Plot roadmap (nodes and edges)
    for sample in roadmap:
        if robot_type == 'arm':
            # Calculate the end-effector position for the arm robot
            end_effector_x, end_effector_y = calculate_end_effector(sample[0], sample[1])
            x, y = end_effector_x, end_effector_y
        else:
            x, y, _ = sample  # Use x, y for the freeBody robot
        ax.scatter(x, y, color='blue', s=10)

        for neighbor in roadmap[sample]:
            if robot_type == 'arm':
                # Calculate the end-effector position for the arm robot's neighbor
                nx, ny = calculate_end_effector(neighbor[0], neighbor[1])
            else:
                nx, ny, _ = neighbor
            ax.plot([x, nx], [y, ny], color='blue', lw=0.5)

    # Plot the path if available
    if path:
        for i in range(len(path) - 1):
            if robot_type == 'arm':
                # Calculate the end-effector positions for the arm robot
                x1, y1 = calculate_end_effector(path[i][0], path[i][1])
                x2, y2 = calculate_end_effector(path[i + 1][0], path[i + 1][1])
            else:
                x1, y1, _ = path[i]
                x2, y2, _ = path[i + 1]
            ax.plot([x1, x2], [y1, y2], color='red', lw=2)

    ax.set_xlim(0, 20)  # Arm workspace plot limit
    ax.set_ylim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()

# Load obstacles from the map file
def load_obstacles(filename):
    obstacles = []
    with open(filename, 'r') as file:
        for line in file:
            parts = list(map(float, line.strip().split(',')))
            if len(parts) == 5:
                x, y, width, height, orientation = parts
                # For simplicity, we ignore orientation for now in the obstacle definition
                obstacles.append((x, y, width, height))
    return obstacles

# Main function to parse input arguments and run the PRM algorithm
def main():
    parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
    parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
    parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation or joint1, joint2)')
    parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation or joint1, joint2)')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
    args = parser.parse_args()

    # Load obstacles from the map file
    obstacles = load_obstacles(args.map)

    # Generate random samples in free space
    num_samples = 750  # You can adjust this
    k = 6  # Number of nearest neighbors
    width, height = 20, 20  # Environment dimensions

    start_config = tuple(args.start)
    goal_config = tuple(args.goal)
    
    # Add start and goal to samples
    samples = generate_samples(num_samples, width, height, lambda config: is_collision(config, obstacles), args.robot)
    samples.append(start_config)
    samples.append(goal_config)

    # Build the roadmap
    roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)

    # Find the shortest path using A* search
    path = astar(roadmap, start_config, goal_config)

    # Visualize the roadmap and the path
    visualize_roadmap(samples, obstacles, roadmap, path, args.robot)

if __name__ == "__main__":
    main()