import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# Function to animate the robot (freeBody or arm) along the optimal path
def animate_robot(path, obstacles, robot_type):
    fig, ax = plt.subplots()

    # Plot obstacles
    for obstacle in obstacles:
        obs_x, obs_y, obs_width, obs_height = obstacle
        rect = patches.Rectangle(
            (obs_x - obs_width / 2, obs_y - obs_height / 2),
            obs_width, obs_height, edgecolor='black', facecolor='gray'
        )
        ax.add_patch(rect)

    # Initialize robot visualization
    robot_patch = None
    if robot_type == 'freeBody':
        robot_patch = patches.Rectangle((0, 0), 1.0, 0.5, edgecolor='green', facecolor='green')
        ax.add_patch(robot_patch)
    elif robot_type == 'arm':
        robot_patch, = ax.plot([], [], color='green', lw=2)  # Arm as a line

    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)

    # Animate the robot along the path
    for config in path:
        if robot_type == 'freeBody':
            # Update freeBody robot position and orientation
            robot_patch.set_xy((config[0] - 0.5, config[1] - 0.25))  # Update position
            robot_patch.angle = config[2]  # Update orientation
        elif robot_type == 'arm':
            # Update arm robot joint positions
            joint1_length, joint2_length = 1.0, 1.0
            joint1_x = joint1_length * math.cos(math.radians(config[0]))
            joint1_y = joint1_length * math.sin(math.radians(config[0]))
            joint2_x = joint1_x + joint2_length * math.cos(math.radians(config[0] + config[1]))
            joint2_y = joint1_y + joint2_length * math.sin(math.radians(config[0] + config[1]))
            robot_patch.set_data([0, joint1_x, joint2_x], [0, joint1_y, joint2_y])  # Update arm

        plt.draw()
        plt.pause(0.3)  # Delay to create animation effect

    plt.show()

# Function to load obstacles from the map file
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

if __name__ == "__main__":
    import argparse
    from prm_new import astar, build_roadmap, generate_samples, is_collision  # Import from prm.py

    parser = argparse.ArgumentParser(description="Animate PRM Path for Robots.")
    parser.add_argument('--robot', type=str, required=True, help='Type of robot: arm or freeBody')
    parser.add_argument('--start', nargs='+', type=float, required=True, help='Start configuration (x, y, orientation)')
    parser.add_argument('--goal', nargs='+', type=float, required=True, help='Goal configuration (x, y, orientation)')
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
    samples = generate_samples(num_samples, width, height, lambda config: is_collision(config, obstacles), robot_type=args.robot)
    samples.append(start_config)
    samples.append(goal_config)

    # Build the roadmap
    roadmap = build_roadmap(samples, obstacles, k, lambda edge: False)

    # Find the shortest path using A* search
    path = astar(roadmap, start_config, goal_config)

    # Animate the robot following the optimal path
    if path:
        animate_robot(path, obstacles, args.robot)
    else:
        print("No path found!")
