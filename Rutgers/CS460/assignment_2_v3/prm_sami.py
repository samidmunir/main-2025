import argparse as ARGPRS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP
import random as RANDOM

# CONSTANTS
ENVIRONMENT_WIDTH_MIN = -10
ENVIRONMENT_WIDTH_MAX = 10
ENVIRONMENT_HEIGHT_MIN = -10
ENVIRONMENT_HEIGHT_MAX = 10

ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5

def get_arm_config_forward_kinematics(theta_1: float, theta_2: float):

def get_only_valid_sample_configs(sample_configs: list) -> list:
    pass

def get_sample_configs(num_of_samples: int) -> list:
    SAMPLE_CONFIGS = []
    for i in range(num_of_samples):
        theta_1 = RANDOM.uniform(0.01746, 2 * NP.pi)
        theta_2 = RANDOM.uniform(0.01746, 2 * NP.pi)
        
        SAMPLE_CONFIG = (theta_1, theta_2)
    
        SAMPLE_CONFIGS.append(SAMPLE_CONFIG)

    return SAMPLE_CONFIGS

def parse_arguments():
    ARG_PARSER = ARGPRS.ArgumentParser(description = 'Arm robot PRM')
    
    ARG_PARSER.add_argument('--robot', type = str, choices = ['arm', 'freeBody'], required = True, help = 'Type of robot (arm OR freeBody).')
    
    ARG_PARSER.add_argument('--start', type = float, nargs = '+', required = True, help = 'Start pose of robot. (N = 2 for arm, N = 3 for freeBody).')
    
    ARG_PARSER.add_argument('--goal', type = float, nargs = '+', required = True, help = 'Goal pose of robot. (N = 2 for arm, N = 3 for freeBody).')
    
    ARG_PARSER.add_argument('--map', type = str, required = True, help = 'File name of file containg environment description.')
    
    return ARG_PARSER.parse_args()

def main():
    ARGS = parse_arguments()
    
    if ARGS.robot == 'arm':
        print('\nHandle arm robot PRM.\n')
        
        SAMPLE_CONFIGS = get_sample_configs(num_of_samples = 5000)
        VALID_SAMPLE_CONFIGS = get_only_valid_sample_configs(sample_configs = SAMPLE_CONFIGS)
    else:
        print('\n*** NOT YET SUPPORTED ***\n')

if __name__ == '__main__':
    main()