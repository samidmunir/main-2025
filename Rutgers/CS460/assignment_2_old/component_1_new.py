import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Function to generate a random environment with obstacles
def generate_environment(number_of_obstacles):
    environment = []
    for _ in range(number_of_obstacles):
        # Generate random center, width, height, and orientation for each obstacle
        center_x = random.uniform(1, 19)
        center_y = random.uniform(1, 19)
        width = random.uniform(0.5, 2)
        height = random.uniform(0.5, 2)
        orientation = random.uniform(0, 360)  # Random orientation in degrees
        obstacle = (center_x, center_y, width, height, orientation)
        environment.append(obstacle)
    return environment

# Function to save an environment to a file
def scene_to_file(environment, filename):
    with open(filename, 'w') as file:
        for obstacle in environment:
            file.write(f"{obstacle[0]},{obstacle[1]},{obstacle[2]},{obstacle[3]},{obstacle[4]}\n")

# Function to load an environment from a file
def scene_from_file(filename):
    environment = []
    with open(filename, 'r') as file:
        for line in file:
            x, y, w, h, o = map(float, line.strip().split(','))
            environment.append((x, y, w, h, o))
    return environment

# Function to visualize an environment
def visualize_scene(environment):
    fig, ax = plt.subplots()
    for obstacle in environment:
        center_x, center_y, width, height, orientation = obstacle
        rect = patches.Rectangle(
            (center_x - width / 2, center_y - height / 2),
            width,
            height,
            angle=orientation,
            edgecolor='black',
            facecolor='blue',
            alpha=0.5
        )
        ax.add_patch(rect)
    
    # Set plot limits and labels
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f"Number of Obstacles: {len(environment)}")
    plt.show()

# Test example (you can comment these lines if testing separately)
if __name__ == "__main__":
    # Generate and save an environment
    env = generate_environment(15)
    scene_to_file(env, 'environment_1.txt')
    
    # Load and visualize the saved environment
    loaded_env = scene_from_file('environment_1.txt')
    visualize_scene(loaded_env)