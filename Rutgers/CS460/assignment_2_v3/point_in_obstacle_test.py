import math as MATH
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM
# 0.787
def point_in_rotated_rectangle(px, py, x, y, width, height, theta, epsilon = 0.788):
    # Step 1: Translate point to the rectangle's local coordinate system.
    translated_x = px - x
    translated_y = py - y
    
    # Step 2: Rotate the point by -theta to align with the rectangle's axes.
    cos_theta = MATH.cos(-theta)
    sin_theta = MATH.sin(-theta)
    local_x = translated_x * cos_theta - translated_y * sin_theta
    local_y = translated_x * sin_theta + translated_y * cos_theta
    
    # Step 3: Check if the point lies within the rectangle's bounds.
    half_width = width / 2.0
    half_height = height / 2.0
    
    if -half_width - epsilon <= local_x <= half_width + epsilon and -half_height - epsilon <= local_y <= half_height + epsilon:
        return True
    else:
        return False

def main():
    print('\nTesting if point in obstacle.')
    
    point = (1.85, 3.65)
    
    # Define an obstacle
    # x = RANDOM.uniform(-10, 10)
    # y = RANDOM.uniform(-10, 10)
    # width = RANDOM.uniform(1.2, 2.0)
    # height = RANDOM.uniform(1.2, 2.0)
    # theta = RANDOM.uniform(0.0, 2 * NP.pi)
    # obstacle = (x, y, width, height, theta)
    x = 1.1
    y = 2.0
    width = 2.0
    height = 1.5
    theta = 0.25
    obstacle = (x, y, width, height, theta)
    
    # Check if the point is inside the obstacle
    
    # Draw environment.
    FIGURE, AXES = PLT.subplots()
    
    OBSTACLE_RECTANGLE = PTCHS.Rectangle((x, y), width, height, angle = NP.rad2deg(theta), color = '#ff0000')
    AXES.add_patch(OBSTACLE_RECTANGLE)
    
    AXES.plot(point[0], point[1], 'o', ms = 0.75, color = '#000000')
    
    if point_in_rotated_rectangle(point[0], point[1], x, y, width, height, theta):
        print('Point is inside the obstacle.')
    
    PLT.title('Point in Obstacle Test')
    PLT.axis('equal')
    PLT.xlim([-10, 10])
    PLT.ylim([-10, 10])
    
    PLT.show()

if __name__ == '__main__':
    main()