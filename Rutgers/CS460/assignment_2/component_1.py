import random
import json
import matplotlib.pyplot as PLT

def generate_environment(number_of_obstacles: int) -> list:
    ENVIRONMENT = []
    for _ in range(number_of_obstacles):
        # Random dimensions for the obstacle.
        width = random.uniform(0.5, 2.0)
        height = random.uniform(0.5, 2.0)
        
        # Random rotation (pose) for the obstacle.
        rotation = random.uniform(0, 350)
        
        # Random center position within the 20x20 environment.
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        
        # Store obstacle as a dictionary or tuple ((center_x, center_y), width, height, rotation).
        obstacle = {
            'center': (x, y),
            'width': width,
            'height': height,
            'rotation': rotation
        }
        print(f'\tGenerated obstacle:')
        print(f'\t\tcenter[{obstacle['center'][0]}, {obstacle['center'][1]}]')
        print(f'\t\twidth: {obstacle["width"]}, height: {obstacle["height"]}]')
        print(f'\t\trotation: {obstacle["rotation"]} degrees\n')
        ENVIRONMENT.append(obstacle)
    
    return ENVIRONMENT

def scene_to_file(environment: list, filename: str):
    with open(filename, 'w') as file:
        json.dump(environment, file)

def scene_from_file(filename: str) -> list:
    with open(filename, 'r') as file:
        environment = json.load(file)
    
    return environment

def visualize_scene(environment: list):
    fig, ax = PLT.subplots()
    
    # Plot each obstacle.
    for obstacle in environment:
        x, y = obstacle['center']
        width, height = obstacle['width'], obstacle['height']
        rotation = obstacle['rotation']
        
        # Create a rectangle to represent the obstacle.
        obstacle_rectangle = PLT.Rectangle((x - width / 2, y - height / 2), width, height, angle = rotation, color = '#0000ff', alpha = 0.5)
        ax.add_patch(obstacle_rectangle)
    
    # Set limits for the 20x20 environment.
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 20)
    
    # Display the number of obstacles as the title of the plot.
    number_of_obstacles = len(environment)
    PLT.title(f'Random Environment with {number_of_obstacles} obstacles')
    
    PLT.gca().set_aspect('equal', adjustable = 'box')
    PLT.show()