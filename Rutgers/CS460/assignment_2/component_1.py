import random

def generate_environment(number_of_obstacles: int) -> list:
    ENVIRONMENT = []
    for _ in range(number_of_obstacles):
        # Random dimensions for the obstacle.
        width = random.uniform(0.5, 2.0)
        height = random.uniform(0.5, 2.0)
        
        # Random center position within the 20x20 environment.
        x = random.uniform(0, 20)
        y = random.uniform(0, 20)
        
        # Store obstacle as a dictionary or tuple (center_x, center_y, width, height).
        obstacle = {'center': (x, y), 'width': width, 'height': height}
        ENVIRONMENT.append(obstacle)
    
    return ENVIRONMENT