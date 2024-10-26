import argparse as ARGPRS
import matplotlib.pyplot as PLT
import matplotlib.patches as PTCHS
import numpy as NP

# CONSTANTS
ENVIRONMENT_WIDTH_MIN = -10
ENVIRONMENT_WIDTH_MAX = 10
ENVIRONMENT_HEIGHT_MIN = -10
ENVIRONMENT_HEIGHT_MAX = 10

ARM_ROBOT_LINK_1_LENGTH = 2.0
ARM_ROBOT_LINK_2_LENGTH = 1.5

def parse_arguments():
    ARG_PARSER = ARGPRS.ArgumentParser(description = 'Arm robot PRM')
    
    ARG_PARSER.add_argument('--robot', type = str, required = True, choices = ['arm'], default = 'arm', description = 'Type of robot (arm or freeBody)')
    ARG_PARSER.add_argument('--start', type = float, nargs = '2', required = True, description = 'Start configuration (theta_1, theta_2) of arm robot.')
    ARG_PARSER.add_argument('--goal', type = float, nargs = '2', required = True, description = 'Goal configuration (theta_1, theta_2) of arm robot.')
    
    return ARG_PARSER

def main():
    ARGS = parse_arguments()

if __name__ == '__main__':
    main():