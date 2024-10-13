import argparse
import random
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import heapq

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

# Visualize the roadmap and the found path for both arm robot and freeBody robot
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
            base, joint1, end_effector = calculate_arm_positions(sample[0], sample[1])
            # Plot the arm as connected line segments
            ax.plot([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], 'b-', lw=2, label="Arm")
            # Plot the joints
            ax.scatter([base[0], joint1[0], end_effector[0]], [base[1], joint1[1], end_effector[1]], color='red', s=50, zorder=5, label="Joints")
        else:
            x, y, _ = sample  # Use x, y for the freeBody robot
            ax.scatter(x, y, color='blue', s=10)

        for neighbor in roadmap[sample]:
            if robot_type == 'arm':
                # Calculate the end-effector position for the arm robot's neighbor
                base_n, joint1_n, end_effector_n = calculate_arm_positions(neighbor[0], neighbor[1])
                ax.plot([end_effector[0], end_effector_n[0]], [end_effector[1], end_effector_n[1]], color='blue', lw=0.5)
            else:
                nx, ny, _ = neighbor
                ax.plot([x, nx], [y, ny], color='blue', lw=0.5)

    # Plot the path if available
    if path:
        for i in range(len(path) - 1):
            if robot_type == 'arm':
                base_1, joint1_1, end_effector_1 = calculate_arm_positions(path[i][0], path[i][1])
                base_2, joint1_2, end_effector_2 = calculate_arm_positions(path[i + 1][0], path[i + 1][1])
                ax.plot([end_effector_1[0], end_effector_2[0]], [end_effector_1[1], end_effector_2[1]], color='red', lw=2)
            else:
                x1, y1, _ = path[i]
                x2, y2, _ = path[i + 1]
                ax.plot([x1, x2], [y1, y2], color='red', lw=2)

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
                obstacles.append((x, y, width, height))
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
    num_samples = 500  # You can adjust this
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