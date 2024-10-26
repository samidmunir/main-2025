import numpy as NP
import random as RANDOM

from utils import (
    ENVIRONMENT_WIDTH_MIN,
    ENVIRONMENT_WIDTH_MAX,
    ENVIRONMENT_HEIGHT_MIN,
    ENVIRONMENT_HEIGHT_MAX,
    OBSTACLE_MIN_SIZE,
    OBSTACLE_MAX_SIZE,
    ARM_ROBOT_LINK_1_LENGTH,
    ARM_ROBOT_LINK_2_LENGTH,
    JOINT_RADIUS,
    FREE_BODY_ROBOT_WIDTH,
    FREE_BODY_ROBOT_HEIGHT
)

def generate_environment(number_of_obstacles: int) -> list:
    print(f'\ngenerate_environment({number_of_obstacles}) called...')
    
    ENVIRONMENT = []
    
    for _ in range(number_of_obstacles):
        x = RANDOM.uniform(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
        y = RANDOM.uniform(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
        width = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        height = RANDOM.uniform(OBSTACLE_MIN_SIZE, OBSTACLE_MAX_SIZE)
        theta = RANDOM.uniform(0.01746, 2 * NP.pi)
    
        OBSTACLE = (x, y, width, height, theta)
    
        ENVIRONMENT.append(OBSTACLE)
        
    print(f'\tsuccessfully generated random environment with {number_of_obstacles} obstacles.')
    
    return ENVIRONMENT