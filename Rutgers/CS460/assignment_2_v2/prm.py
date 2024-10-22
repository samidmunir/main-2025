# import argparse
# import random
# import math
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# import heapq

# from collision_checking import get_polygon_corners, is_colliding

# # Normalize angle between 0 and 360
# def normalize_angle(angle):
#     return angle % 360

# # Function to calculate the positions of each joint and end-effector for a 2-joint arm robot
# def calculate_arm_positions(joint1_angle, joint2_angle, joint1_length=10.0, joint2_length=10.0):
#     base = (0, 0)
#     joint1 = (joint1_length * math.cos(math.radians(joint1_angle)),
#               joint1_length * math.sin(math.radians(joint1_angle)))
#     end_effector = (joint1[0] + joint2_length * math.cos(math.radians(joint1_angle + joint2_angle)),
#                     joint1[1] + joint2_length * math.sin(math.radians(joint1_angle + joint2_angle)))
#     return base, joint1, end_effector

# # Interpolate points between two configurations
# def interpolate_path(start, end, num_points=10):
#     """Generate points between start and end for collision checking."""
#     return np.linspace(start, end, num_points)

# # Check if an edge between two nodes is collision-free
# def is_edge_collision_free(start, end, obstacles):
#     """Check if all intermediate points along an edge are free from obstacles."""
#     for point in interpolate_path(start, end):
#         if is_colliding(tuple(point), obstacles):
#             return False  # Collision detected
#     return True  # Edge is valid

# # Generate random samples in the environment
# def generate_samples(num_samples, width, height, obstacle_check_fn, robot_type):
#     samples = []
#     for _ in range(num_samples):
#         while True:
#             if robot_type == 'arm':
#                 joint1 = random.uniform(0, 360)
#                 joint2 = random.uniform(0, 360)
#                 config = (joint1, joint2)
#             else:  # freeBody robot
#                 x = random.uniform(-20, 20)
#                 y = random.uniform(-20, 20)
#                 orientation = random.uniform(0, 360)
#                 config = (x, y, orientation)
#             if not obstacle_check_fn(config):
#                 samples.append(config)
#                 break
#     return samples

# # Build the PRM roadmap with valid edges only
# def build_roadmap(samples, obstacles, k, obstacle_check_fn):
#     """Construct the PRM roadmap ensuring no edges intersect with obstacles."""
#     roadmap = {sample: [] for sample in samples}

#     for i, start in enumerate(samples):
#         distances = [(math.dist(start, end), end) for j, end in enumerate(samples) if i != j]
#         distances.sort()  # Sort by distance to find k-nearest neighbors

#         for _, neighbor in distances[:k]:
#             if is_edge_collision_free(start, neighbor, obstacles):
#                 roadmap[start].append(neighbor)
#                 roadmap[neighbor].append(start)  # Undirected graph

#     return roadmap

# # A* search algorithm to find the shortest path
# def astar(roadmap, start, goal):
#     """A* algorithm to find the shortest path in the PRM roadmap."""
#     open_set = [(0, start)]  # Priority queue with (cost, node)
#     came_from = {}
#     g_score = {node: float('inf') for node in roadmap}
#     g_score[start] = 0

#     while open_set:
#         _, current = heapq.heappop(open_set)

#         if current == goal:
#             path = []
#             while current in came_from:
#                 path.append(current)
#                 current = came_from[current]
#             path.append(start)
#             return path[::-1]  # Return reversed path

#         for neighbor in roadmap[current]:
#             tentative_g_score = g_score[current] + math.dist(current, neighbor)
#             if tentative_g_score < g_score[neighbor]:
#                 came_from[neighbor] = current
#                 g_score[neighbor] = tentative_g_score
#                 heapq.heappush(open_set, (tentative_g_score, neighbor))

#     return []  # Return empty path if no solution found

# # Visualize the PRM roadmap and the path
# def visualize_roadmap(samples, obstacles, roadmap, path, robot_type):
#     fig, ax = plt.subplots()
#     ax.set_xlim(-20, 20)
#     ax.set_ylim(-20, 20)

#     # Draw obstacles
#     for obstacle in obstacles:
#         corners = get_polygon_corners(obstacle)
#         patch = patches.Polygon(corners, closed=True, color='red', alpha=0.5)
#         ax.add_patch(patch)

#     # Draw PRM edges
#     for node, neighbors in roadmap.items():
#         for neighbor in neighbors:
#             ax.plot([node[0], neighbor[0]], [node[1], neighbor[1]], 'gray', linestyle='--', alpha=0.7)

#     # Draw path
#     if path:
#         for i in range(len(path) - 1):
#             ax.plot([path[i][0], path[i + 1][0]], [path[i][1], path[i + 1][1]], 'blue', linewidth=2)

#     # Draw start and goal
#     ax.plot(path[0][0], path[0][1], 'go', markersize=10, label='Start')
#     ax.plot(path[-1][0], path[-1][1], 'ro', markersize=10, label='Goal')

#     plt.legend()
#     plt.show()

# # Load obstacles from the map file
# def load_obstacles(map_file):
#     obstacles = []
#     with open(map_file, 'r') as file:
#         for line in file:
#             parts = list(map(float, line.strip().split(',')))
#             if len(parts) == 5:
#                 x, y, width, height, orientation = parts
#                 obstacles.append((x, y, width, height, orientation))
#     return obstacles

# # Main function to parse input arguments and run the PRM algorithm
# def main():
#     parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
#     parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
#     parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration')
#     parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration')
#     parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
#     args = parser.parse_args()

#     obstacles = load_obstacles(args.map)

#     num_samples = 500
#     k = 6
#     width, height = 20, 20

#     start_config = tuple(args.start)
#     goal_config = tuple(args.goal)

#     samples = generate_samples(num_samples, width, height, lambda config: is_colliding(config, obstacles), args.robot)
#     samples.append(start_config)
#     samples.append(goal_config)

#     roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)
#     path = astar(roadmap, start_config, goal_config)
#     visualize_roadmap(samples, obstacles, roadmap, path, args.robot)

# if __name__ == "__main__":
#     main()

import argparse
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as NP
import heapq

from collision_checking import get_polygon_corners, project, is_colliding_link, is_line_intersecting, get_axes

# Normalize angle between 0 and 360
def normalize_angle(angle):
    return angle % 360

# Function to calculate the positions of each joint and end-effector for a 2-joint arm robot
def calculate_arm_positions(joint1_angle, joint2_angle, joint1_length=10.0, joint2_length=10.0):
    # Base position
    base_x, base_y = 0, 0

    # Joint 1 position
    joint1_x = joint1_length * math.cos(math.radians(joint1_angle))
    joint1_y = joint1_length * math.sin(math.radians(joint1_angle))

    # End-effector position (Joint 2 relative to Joint 1)
    end_effector_x = joint1_x + joint2_length * math.cos(math.radians(joint1_angle + joint2_angle))
    end_effector_y = joint1_y + joint2_length * math.sin(math.radians(joint1_angle + joint2_angle))

    return (base_x, base_y), (joint1_x, joint1_y), (end_effector_x, end_effector_y)

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
                x = random.uniform(-20, 20)
                y = random.uniform(-20, 20)
                orientation = random.uniform(0, 360)
                config = (x, y, orientation)
            if not obstacle_check_fn(config):
                samples.append(config)
                break
    return samples

# Check for collision between a robot and an obstacle
def is_collision(config, obstacles):
    if len(config) == 3:  # freeBody robot
        x, y, orientation = config
        CONFIG_CORNERS = get_polygon_corners((x, y), 0.3, 0.5, orientation)
        for obstacle in obstacles:
            OBSTACLE_CORNERS = get_polygon_corners((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], obstacle[4])
            # obs_x, obs_y, obs_width, obs_height, orientation = obstacle
            # if obs_x - obs_width / 2 <= x <= obs_x + obs_width / 2 and obs_y - obs_height / 2 <= y <= obs_y + obs_height / 2:
            #     return True
            if is_colliding(CONFIG_CORNERS, OBSTACLE_CORNERS):
                return True
    return False

# Get k-nearest neighbors for a given configuration
def get_k_nearest_neighbors(config, samples, k):
    distances = [(other, math.sqrt((config[0] - other[0]) ** 2 + (config[1] - other[1]) ** 2)) for other in samples]
    distances.sort(key=lambda x: x[1])
    return [neighbor for neighbor, _ in distances[:k]]

"""
    function is_colliding():
"""
def is_colliding(robot_corners, obstacle_corners):
    robot_corners = NP.array(robot_corners)
    robot_corners = 2 * robot_corners
    obstacle_corners = NP.array(obstacle_corners)
    obstacle_corners = 2 * obstacle_corners
    AXES = NP.vstack([get_axes(robot_corners), get_axes(obstacle_corners)])
    
    for AXIS in AXES:
        min_1, max_1 = project(robot_corners, AXIS)
        min_2, max_2 = project(obstacle_corners, AXIS)
        
        if max_1 < min_2 or max_2 < min_1:
            return False
    
    return True

"""
    def get_points_on_line():
"""
def get_points_on_line(x1, y1, x2, y2):
    points = []
    n = 5
    for i in range(1, n + 1):
        t = i / (n + 1)
        x = x1 + t *  (x2 - x1)
        y = y1 + t *  (y2 - y1)
        points.append((x, y))
    return points
    

"""
    def bresenham_line():
"""
def bresenham_line(x1, y1, x2, y2):
    points = []
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    while True:
        points.append((x1, y1))
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x1 += sx
        if e2 < dx:
            err += dx
            y1 += sy
    
    return points

def is_point_in_aabb(point, rect):
    """Check if a point (x, y) lies inside an axis-aligned rectangle."""
    px, py = point
    rx, ry, width, height, _ = rect  # Ignore orientation for AABB

    return rx <= px <= rx + width and ry <= py <= ry + height

def rotate_point(point, origin, angle):
    """Rotate a point around an origin by a given angle (in radians)."""
    px, py = point
    ox, oy = origin

    cos_angle = math.cos(-angle)
    sin_angle = math.sin(-angle)

    qx = ox + cos_angle * (px - ox) - sin_angle * (py - oy)
    qy = oy + sin_angle * (px - ox) + cos_angle * (py - oy)

    return qx, qy

def is_point_in_obb(point, rect):
    """Check if a point lies inside an oriented rectangle."""
    x, y, width, height, orientation = rect

    # Rotate the point back to align with the rectangle
    rotated_point = rotate_point(point, (x, y), math.radians(orientation))

    # Use AABB logic on the rotated point
    return is_point_in_aabb(rotated_point, (x, y, width, height, 0))

"""
    function is_line_crossing_obstacle():
    - this function checks if a line (which represents a path from one node to another) intersects any obstacle.
    - if returns True, it means the line crosses the obstacle.
    - else, it returns False.
"""
def is_line_crossing_obstacle(line, obstacle):
    x1, y1 = line[0], line[1]
    x2, y2 = line[2], line[3]
    OBSTACLE_CORNERS = get_polygon_corners((obstacle[0], obstacle[1]), obstacle[2], obstacle[3], obstacle[4])
    CROSSING = (is_colliding_link(link_start=(x1, y1), link_end=(x2, y2), obstacle_corners = OBSTACLE_CORNERS))
    
    return CROSSING
    
# Build the PRM roadmap by connecting samples to their k-nearest neighbors
def build_roadmap(samples, obstacles, k, obstacle_check_fn):
    roadmap = {sample: [] for sample in samples}
    for sample in samples:
        neighbors = get_k_nearest_neighbors(sample, samples, k)
        for neighbor in neighbors:
            for obstacle in obstacles:
                if not obstacle_check_fn((sample, neighbor)) and not obstacle_check_fn((neighbor, obstacle)) and not obstacle_check_fn((sample, obstacle)) and not is_line_crossing_obstacle((sample[0], sample[1], neighbor[0], neighbor[1]), obstacle = obstacle) and not is_line_crossing_obstacle((neighbor[0], neighbor[1], sample[0], sample[1]), obstacle = obstacle) and not is_point_in_obb((neighbor[0], neighbor[1]), obstacle) and not is_point_in_obb((sample[0], sample[1]), (obstacle)):
                    POINTS = get_points_on_line(sample[0], sample[1], neighbor[0], neighbor[1])
                    for POINT in POINTS:
                        if not is_point_in_obb(POINT, obstacle):
                            roadmap[sample].append(neighbor)
                            roadmap[neighbor].append(sample)  # Bidirectional connection
                    # roadmap[sample].append(neighbor)
                    # roadmap[neighbor].append(sample)  # Bidirectional connection
                # BESENHAM_LINE_POINTS = bresenham_line(sample[0], sample[1], neighbor[0], neighbor[1])
                # for POINT in BESENHAM_LINE_POINTS:
                #     if not is_point_in_obb(POINT, obstacle):
                #         roadmap[sample].append(neighbor)
                #         roadmap[neighbor].append(sample)  # Bidirectional connection
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

# Visualize the roadmap and the found path for both arm robot and freeBody robot
def visualize_roadmap(samples, obstacles, roadmap, path=None, robot_type='freeBody'):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        obs_x, obs_y, obs_width, obs_height, orientation = obstacle
        rect = patches.Rectangle(
            (obs_x, obs_y),
            obs_width, obs_height, edgecolor='black', facecolor='gray', angle = orientation
        )
        ax.add_patch(rect)

    # Plot roadmap (nodes and edges)
    for sample in roadmap:
        if robot_type == 'arm':
            # Calculate the end-effector position for the arm robot
            base, joint1, end_effector = calculate_arm_positions(sample[0], sample[1])
            # Plot the arm as connected line segments
            ax.plot([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], 'b-', lw=2, label="Arm")
            # Plot the joints
            ax.scatter([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], color='red', s=50, zorder=5, label="Joints")
        else:
            x, y, _ = sample  # Use x, y for the freeBody robot
            ax.scatter(x, y, color='blue', s=7.5, alpha = 0.75)

        for neighbor in roadmap[sample]:
            if robot_type == 'arm':
                # Calculate the end-effector position for the arm robot's neighbor
                base_n, joint1_n, end_effector_n = calculate_arm_positions(neighbor[0], neighbor[1])
                ax.plot([end_effector[0], end_effector_n[0]], [end_effector[1], end_effector_n[1]], color='blue', lw=0.5)
            else:
                nx, ny, _ = neighbor
                ax.plot([x, nx], [y, ny], color='blue', lw=0.5, alpha = 0.5)

    # Plot the path if available
    # if path:
    #     for i in range(len(path) - 1):
    #         if robot_type == 'arm':
    #             base_1, joint1_1, end_effector_1 = calculate_arm_positions(path[i][0], path[i][1])
    #             base_2, joint1_2, end_effector_2 = calculate_arm_positions(path[i + 1][0], path[i + 1][1])
    #             ax.plot([end_effector_1[0], end_effector_2[0]], [end_effector_1[1], end_effector_2[1]], color='red', lw=2)
    #         else:
    #             x1, y1, _ = path[i]
    #             x2, y2, _ = path[i + 1]
    #             ax.plot([x1, x2], [y1, y2], color='red', lw=2)

    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
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
                obstacles.append((x, y, width, height, orientation))
    return obstacles

# Main function to parse input arguments and run the PRM algorithm
def main():
    parser = argparse.ArgumentParser(description="PRM Algorithm for Path Planning.")
    parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
    parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation for freeBody or joint1, joint2 for arm)')
    parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation for freeBody or joint1, joint2 for arm)')
    parser.add_argument('--map', type=str, required=True, help='Map file containing obstacles')
    args = parser.parse_args()

    # Load obstacles from the map file
    obstacles = load_obstacles(args.map)

    # Generate random samples in free space
    num_samples = 250  # You can adjust this
    k = 3 # Number of nearest neighbors
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