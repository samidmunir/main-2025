import argparse as ARGPRS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
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

def scene_to_file(environment: list, filename: str) -> None:
    print(f'\nscene_to_file({filename}) called...')
    
    with open(filename, 'w') as FILE:
        for OBSTACLE in environment:
            FILE.write(f'{OBSTACLE[0]}, {OBSTACLE[1]}, {OBSTACLE[2]}, {OBSTACLE[3]}, {OBSTACLE[4]}\n')
    
    print(f'\tenvironment saved to FILE <{filename}>.')
    
def scene_from_file(filename: str) -> list:
    print(f'\nscene_from_file({filename}) called...')
    
    ENVIRONMENT = []
    
    with open(filename, 'r') as FILE:
        for LINE in FILE.readlines():
            VALUES = LINE.strip().split(',')
            x = float(VALUES[0])
            y = float(VALUES[1])
            width = float(VALUES[2])
            height = float(VALUES[3])
            theta = float(VALUES[4])
            
            OBSTACLE = (x, y, width, height, theta)
            
            ENVIRONMENT.append(OBSTACLE)
    
    print(f'\tenvironment loaded from FILE <{filename}>.')
    
    return ENVIRONMENT

def visualize_scene(environment: list) -> None:
    print(f'\nvisualize_scene() called...')
    
    FIGURE, AXES = PLT.subplots()
    AXES.set_aspect('equal')
    AXES.set_xlim(ENVIRONMENT_WIDTH_MIN, ENVIRONMENT_WIDTH_MAX)
    AXES.set_ylim(ENVIRONMENT_HEIGHT_MIN, ENVIRONMENT_HEIGHT_MAX)
    
    for OBSTACLE in environment:
        x, y, width, height, theta = OBSTACLE
        
        OBSTACLE_RECTANGLE = PTCHS.Rectangle(xy = (x, y), width = width, height = height, angle = NP.rad2deg(theta), color = '#ff0000', alpha = 0.75)
        
        AXES.add_patch(OBSTACLE_RECTANGLE)
    
    PLT.title(f'Randomly generated environment: {len(environment)} obstacles')
    PLT.show(block = False)
    PLT.pause(2)
    PLT.close(fig = FIGURE)
    
def parse_arguments():
    ARG_PRSR = ARGPRS.ArgumentParser(description = 'Generate random environments')
    
    ARG_PRSR.add_argument('--n', type = int, default = 5, help = 'Number of random environments to generate.')
    ARG_PRSR.add_argument('--ni', type = int, default = 10, help = 'Initial number of obstacles in 1st environment to generate.')
    ARG_PRSR.add_argument('--dn', type = int, default = 5, help = 'Number of obstacles to increase in each environment to generate.')
    
    return ARG_PRSR.parse_args()

def main():
    ARGS = parse_arguments()
    
    ni = ARGS.ni
    dn = ARGS.dn
    
    for i in range(ARGS.n):
        ENVIRONMENT = generate_environment(number_of_obstacles = ni)
        scene_to_file(environment = ENVIRONMENT, filename = f'environment_{(i + 1)}_{ni}.txt')
        visualize_scene(environment = ENVIRONMENT)
        
        ni += dn

if __name__ == "__main__":
    main()